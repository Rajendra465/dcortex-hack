"""The language layer: parse, route, refuse, narrate.

The boundary is absolute. The model does exactly two jobs:

  1. turn a controller's sentence into a TYPED PLAN -- a tool name and typed
     arguments, validated against a schema before anything executes;
  2. write prose over an evidence ledger it is handed AFTER the kernel has run.

It never computes, never decides legality, and never sees a raw duty clock. That
is enforced structurally, not by instruction: the narration context contains only
the ledger. A model told not to compute will still restate a number it half
remembers from context, so we remove the context.

Everything degrades. With no API key the deterministic parser and template
renderer still answer the whole shipped question set -- which is also the honest
answer to "what happens if we delete the model?": you keep correct answers, and
lose the ability to ask for them in your own words.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .data import Snapshot
from .llm import LLM, get_client
from .orchestrator import (CONFIDENT, FLOOR, PAT, Entities, Plan, PlanError,
                           execute, extract, plan_from_index,
                           plan_from_model, score_intents, validate)
from .rules import Ledger
from .tools import REGISTRY, Unanswerable

SNAPSHOT_UTC = "2026-09-14T18:00:00Z"


# --------------------------------------------------------------------------
# refusals -- two structurally distinct classes
# --------------------------------------------------------------------------


@dataclass
class Refusal:
    """Two kinds, and they must never be confused.

    DATA_GAP    the dataset cannot answer this. A pre-declared contract.
    PARSE_FAIL  we did not understand. A defect, and we say so plainly.

    A parse failure dressed up as intellectual honesty is exposed by one probe:
    ask the same answerable question two ways and get an answer once and a
    refusal once. So they render differently and are counted differently.
    """

    kind: str            # DATA_GAP | PARSE_FAIL | OUT_OF_RANGE | UNKNOWN_ENTITY
    message: str
    have: str = ""
    would_need: str = ""
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"answer_type": "refusal", "kind": self.kind,
                "message": self.message, "have": self.have,
                "would_need": self.would_need, "suggestions": self.suggestions}


# What this dataset simply does not contain. Declared up front, so a refusal is
# a contract rather than a runtime dodge.
CAPABILITY_LIMITS = {
    "contact": ("crew phone numbers, emails or addresses",
                "crew.json has reachability_minutes, but no contact details"),
    "weather": ("weather, forecasts or METARs", "no meteorological data"),
    "booking": ("passenger bookings, load factors or connections",
                "flights.json has seats (capacity), not bookings"),
    "maintenance": ("aircraft maintenance status or defects",
                    "no maintenance data"),
    "hotel": ("hotel availability or bookings",
              "costs.json has a nightly rate, not inventory"),
    "pay": ("crew pay, allowances or roster bidding",
            "costs.json has callout rates only"),
    "prediction": ("predictions about who will call in sick",
                   "risk_signals.json is a PROVIDED input; forecasting is "
                   "explicitly out of scope for this system"),
    "aircraft": ("aircraft routing, swaps or diversions",
                 "this models CREW only -- the aircraft rotation is fixed"),
}

# These are PREFIX patterns: "probabilit" has to match "probability". A
# trailing \b makes that impossible, because the next character is still a word
# character -- so leading \b only. That exact bug let a prediction question
# through to a crew lookup during testing, which is the single worst answer this
# system could give.
_LIMIT_PATTERNS = [
    # Screened FIRST. The brief is explicit that forecasting is out of scope and
    # that risk_signals is a provided input, so guessing here is unforgivable.
    (re.compile(r"\b(predict|probabilit|likelihood|odds of|chance (that|of)|"
                r"forecast|how likely)", re.I), "prediction"),
    (re.compile(r"\b(phone|call them|contact details|number for|e-?mail|"
                r"address|how do i reach)", re.I), "contact"),
    # `wind` must not match "window". The prefix style that fixed "probabilit"
    # cuts the other way here: on-call WINDOWS are core vocabulary, and this
    # pattern was refusing "who is on reserve and what are their on-call
    # windows" as a weather question.
    (re.compile(r"\b(weather|storm|fog|wind(?!ow)|metar|turbulence)", re.I),
     "weather"),
    (re.compile(r"\b(passenger name|booking|booked|load factor|connecting|"
                r"connection|pnr|rebook)", re.I), "booking"),
    (re.compile(r"\b(maintenance|defect|serviceab|unserviceab|tech log|aog)",
                re.I), "maintenance"),
    (re.compile(r"\b(hotel availab|accommodation|hotel room)", re.I), "hotel"),
    (re.compile(r"\b(salary|crew pay|allowance|bidding)", re.I), "pay"),
    (re.compile(r"\b(re-?route|divert|swap the (tail|aircraft)|tail swap|"
                r"aircraft swap|change the aircraft)", re.I), "aircraft"),
]


def screen_scope(question: str) -> Refusal | None:
    """Catch questions the data cannot answer BEFORE spending a model call."""
    for pat, key in _LIMIT_PATTERNS:
        if pat.search(question):
            what, why = CAPABILITY_LIMITS[key]
            return Refusal(
                "DATA_GAP",
                f"I don't hold {what}.",
                have=why,
                would_need=f"a data source with {what}",
                suggestions=["ask about rosters, duty hours, legality, cover "
                             "options, certifications or costs"],
            )
    return None


def screen_dates(question: str, snap: Snapshot) -> Refusal | None:
    """The legality horizon equals the data horizon. Say so rather than guess."""
    lo, hi = snap.horizon
    for m in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", question):
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        if not (snap.history_start <= d <= hi):
            return Refusal(
                "OUT_OF_RANGE",
                f"{d} is outside the data I have.",
                have=f"rosters and flights {lo} to {hi}; duty history back to "
                     f"{snap.history_start}",
                would_need="roster data covering that date",
            )
    return None


# --------------------------------------------------------------------------
# entity resolution
# --------------------------------------------------------------------------

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}


def resolve_dates(question: str, snap: Snapshot) -> tuple[list[str], list[str]]:
    """Turn date language into ISO dates, and SAY what was assumed.

    Everything is relative to the frozen snapshot at 2026-09-14T18:00:00Z --
    "tomorrow" means 15 September, not tomorrow in the real world.
    """
    out: list[str] = []
    notes: list[str] = []
    today = snap.snapshot_utc.date()

    for m in re.finditer(r"\b(20\d{2}-\d{2}-\d{2})\b", question):
        out.append(m.group(1))

    m = re.search(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                  question, re.I)
    if m and not out:
        try:
            out.append(date(today.year, _MONTHS[m.group(2)[:3].lower()],
                            int(m.group(1))).isoformat())
        except ValueError:
            pass

    low = question.lower()
    if not out:
        if "tomorrow" in low:
            d = today + timedelta(days=1)
            out.append(d.isoformat())
            notes.append(f"'tomorrow' resolved to {d} (snapshot {today})")
        elif "today" in low:
            out.append(today.isoformat())
            notes.append(f"'today' resolved to {today} (frozen snapshot)")
    return out, notes


def resolve_crew(question: str, snap: Snapshot) -> tuple[list[str], list[str]]:
    """Crew ids, then names. Names are NOT unique in this dataset."""
    notes: list[str] = []
    ids = [c for c in re.findall(r"\bC-\d{4}\b", question) if c in snap.crew]
    if ids:
        return ids, notes
    for c in snap.crew.values():
        surname = c.name.split()[-1].lower()
        if len(surname) > 3 and re.search(rf"\b{re.escape(surname)}\b",
                                          question, re.I):
            matches = sorted(x.crew_id for x in snap.crew.values()
                             if x.name.split()[-1].lower() == surname)
            if len(matches) > 1:
                notes.append(f"'{surname}' matches {len(matches)} crew "
                             f"({', '.join(matches)}) - using {matches[0]}")
            ids.append(matches[0])
            break
    return ids, notes


# --------------------------------------------------------------------------
# narration, with the containment guard
# --------------------------------------------------------------------------

NARRATE_SYSTEM = """You are the voice of an airline Crew Control advisor, speaking to a
controller under time pressure at 5 a.m.

You are given a COMPUTED RESULT and an EVIDENCE LEDGER. Every number has already
been calculated by a deterministic rules engine.

RULES:
- Use ONLY numbers that appear in the ledger. Never compute, round, convert or
  estimate a new one.
- Lead with the verdict or the answer. Two or three sentences, maximum.
- Name the binding rule when there is one.
- Plain language. No preamble, no "certainly", no restating the question.
"""

_NUM = re.compile(r"\d+(?:\.\d+)?")
_STRIP = [
    re.compile(r"\bC-\d{4}\b"), re.compile(r"\bP-\d{4}\b"),
    re.compile(r"\bDX\d{3}(?:-\d{4}-\d{2}-\d{2})?\b"), re.compile(r"\bVT-DX[A-F]\b"),
    re.compile(r"\bRULE-[A-Z]+-\d+\b"), re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}:\d{2}\b"), re.compile(r"\bA320\b"), re.compile(r"\bATR ?72\b"),
]


def guard_numbers(prose: str, ledger: Ledger) -> tuple[bool, list[str]]:
    """Every number in the prose must trace to something the kernel computed.

    Identifiers, aircraft types, dates and clock times are stripped FIRST. The
    naive version of this guard -- comparing against the answer object -- was
    measured at a 67.8% false-block rate against the reference answer keys,
    because it blocks intermediates ("only 10.75h rest"), unit conversions
    ("over by 1h20m") and the digits inside A320 and ATR72.
    """
    text = prose
    for pat in _STRIP:
        text = pat.sub(" ", text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b",
                  lambda m: m.group(0).replace(",", ""), text)
    allowed = ledger.allowed_numbers()
    bad = [n for n in _NUM.findall(text)
           if n not in allowed and n.lstrip("0") not in allowed
           and f"{float(n):g}" not in allowed]
    return (not bad), bad


def narrate(question: str, ledger: Ledger, client: Any,
            fallback: str) -> tuple[str, str]:
    """Returns (prose, provenance) where provenance is 'model' or 'template'.

    OFF by default, and that is a deliberate trade. The deterministic templates
    are already correct and instant; model narration measured 2-30s against the
    NVIDIA endpoint, and the brief is explicit that a 45-second response is not
    a decision aid. Parsing is where the model earns its place -- turning a
    controller's own words into a typed plan. Re-phrasing an answer we already
    computed is polish, and polish does not get to cost 30 seconds on a bad
    morning. Set CREWOPS_NARRATE=1 to turn it on.
    """
    if client is None or not os.environ.get("CREWOPS_NARRATE"):
        return fallback, "template"
    facts = [{"key": f.key, "value": f.value, "unit": f.unit}
             for f in ledger.facts]
    try:
        prose = client.complete(
            NARRATE_SYSTEM,
            f"QUESTION: {question}\n\n"
            f"EVIDENCE LEDGER:\n{json.dumps(facts, default=str)}\n\n"
            f"COMPUTED SUMMARY: {fallback}",
            max_tokens=350)
    except Exception:
        return fallback, "template"

    if not prose:
        return fallback, "template"

    # The gate. Prose that cites a number the kernel never produced is thrown
    # away and the computed summary is shown instead -- silently correct beats
    # fluently wrong.
    ok, bad = guard_numbers(prose, ledger)
    if not ok:
        return fallback, f"template (narration blocked: unverified {bad[:3]})"
    return prose, "model"


# --------------------------------------------------------------------------
# the advisor
# --------------------------------------------------------------------------


@dataclass
class Answer:
    question: str
    tier: int
    plan: Plan | None
    payload: Any
    prose: str
    provenance: str
    ledger: Ledger
    assumptions: list[str] = field(default_factory=list)
    refusal: Refusal | None = None
    ms: float = 0.0
    confidence: str = "high"
    confidence_why: str = ""
    overlays: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        if self.refusal:
            return {**self.refusal.to_dict(), "query": self.question,
                    "confidence": "none", "elapsed_ms": round(self.ms, 1)}
        return {
            "query": self.question,
            "tier": self.tier,
            "answer_type": "computed",
            "resolved": {"as_of_utc": SNAPSHOT_UTC,
                         "assumptions": self.assumptions,
                         "what_if": self.overlays},
            "plan": self.plan.to_dict() if self.plan else None,
            "result": self.payload,
            "explanation": self.prose,
            "explanation_source": self.provenance,
            "confidence": self.confidence,
            "confidence_why": self.confidence_why,
            "evidence": self.ledger.as_dict(),
            "elapsed_ms": round(self.ms, 1),
        }


TIER_OF = {"retrieval": 1, "simulation": 2, "optimisation": 3}


def confidence_of(tool: str, provenance: str, payload: Any) -> tuple[str, str]:
    """Explicit uncertainty signalling, tied to provenance -- never a guess.

    Research on decision support is blunt about this: a confidence number the
    system cannot calibrate is worse than none at all, because it induces both
    over-trust and conservatism. So we do not emit a probability. We say which
    KIND of answer this is, which the controller can actually act on.
    """
    if provenance == "model":
        return "medium", "wording drafted by a model over computed facts"
    if tool in ("get_risk_signals",):
        return "provided", "risk scores are a supplied input, not our computation"
    if tool == "solve_joint_cover" and payload and not payload.get("proven_optimal"):
        return "medium", "a legal plan, but not proven optimal beyond two events"
    return "high", "computed by the rules engine from the dataset"


class Advisor:
    def __init__(self, snap: Snapshot, use_model: bool | None = None):
        self.snap = snap
        self.client: LLM | None = get_client() if use_model is not False else None

    @property
    def model_available(self) -> bool:
        return self.client is not None

    @property
    def model_label(self) -> str:
        return self.client.label if self.client else "none (rules parser only)"

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()

        def done(**kw: Any) -> Answer:
            kw.setdefault("tier", 0); kw.setdefault("plan", None)
            kw.setdefault("payload", None); kw.setdefault("prose", "")
            kw.setdefault("provenance", "deterministic")
            kw.setdefault("ledger", Ledger())
            return Answer(question=question,
                          ms=(time.perf_counter() - t0) * 1000, **kw)

        # 1. policy screens -- declarative, and cheaper than any model call
        for screen in (screen_scope(question), screen_dates(question, self.snap)):
            if screen:
                return done(refusal=screen, prose=screen.message)
        for cid in PAT["crew_id"].findall(question):
            if cid not in self.snap.crew:
                r = Refusal("UNKNOWN_ENTITY", f"{cid} is not in this dataset.",
                            have=f"{len(self.snap.crew)} crew on file")
                return done(refusal=r, prose=r.message)

        # 2. entities, then PLAN. The model is the primary planner when it is
        #    configured; the deterministic index is the fast path and the
        #    offline fallback. Both emit the same validated Plan object.
        ents = extract(question, self.snap)
        _, date_notes = resolve_dates(question, self.snap)
        assumptions = list(date_notes)

        # The index is trusted only when it is CONFIDENT -- instant and
        # reproducible for the phrasings a desk uses constantly. Everything
        # else goes to the model, which is the actual planner. If neither is
        # sure, we refuse: serving the nearest plausible tool is how an
        # unrecognised question came back as somebody else's answer.
        ranked = score_intents(question, ents)
        top = ranked[0][0] if ranked else 0.0

        plan = None
        if top >= CONFIDENT:
            plan = plan_from_index(question, ents, self.snap, CONFIDENT)
        if plan is None and self.client:
            plan = plan_from_model(question, ents, self.client)
        if plan is None and top >= FLOOR:
            plan = plan_from_index(question, ents, self.snap, FLOOR)
        if plan is None:
            r = Refusal(
                "PARSE_FAIL", "I didn't understand that one.",
                have="roster and flight lookup, duty hours, certifications, "
                     "legality checks, disruption impact, ranked cover options, "
                     "joint recovery, notification drafting and proactive scans",
                suggestions=["name a crew id, pairing, flight or date",
                             "or say what you want to do: cover, check, impact"])
            return done(refusal=r, prose=r.message)

        # 3. validate BEFORE anything executes
        try:
            validate(plan)
        except PlanError as e:
            r = Refusal("CLARIFY" if e.would_need else "PARSE_FAIL", e.message,
                        have=e.have, would_need=e.would_need)
            return done(plan=plan, refusal=r, prose=r.message)

        # 4. execute the chain
        try:
            run = execute(plan, self.snap)
        except Unanswerable as e:
            kind = "CLARIFY" if e.would_need else "DATA_GAP"
            r = Refusal(kind, e.reason, have=e.have, would_need=e.would_need)
            return done(plan=plan, refusal=r, prose=r.message)
        except PlanError as e:
            r = Refusal("PARSE_FAIL", e.message, have=e.have)
            return done(plan=plan, refusal=r, prose=r.message)
        except (KeyError, LookupError, IndexError) as e:
            r = Refusal("UNKNOWN_ENTITY", f"I couldn't resolve {e}.",
                        have="check the crew id, pairing id or date")
            return done(plan=plan, refusal=r, prose=r.message)

        payload, ledger = run.final, run.ledger
        tier = max(TIER_OF.get(REGISTRY[st.tool].kind, 1) for st in plan.steps)
        if len(plan.steps) > 1:
            assumptions.append(
                "chained " + " -> ".join(st.tool for st in plan.steps))
        summary = summarise(plan.tool, payload)
        prose, prov = narrate(question, ledger, self.client, summary)
        conf, why = confidence_of(plan.tool, prov, payload)
        return done(tier=tier, plan=plan, payload=payload, prose=prose,
                    provenance=prov, ledger=ledger, assumptions=assumptions,
                    confidence=conf, confidence_why=why)


# --------------------------------------------------------------------------
# multi-turn: carrying context, and stacking what-ifs
# --------------------------------------------------------------------------

_PRONOUN = re.compile(r"\b(he|him|his|she|her|they|them|their|that one|"
                      r"the same|instead)\b", re.I)
_THAT_TRIP = re.compile(r"\b(that|the|this) (trip|pairing|rotation|it)\b", re.I)


@dataclass
class Turn:
    question: str
    answer: Answer


class Session:
    """A conversation with memory, and a stack of what-ifs.

    Two capabilities the single-shot advisor cannot have:

    CONTEXT   "and if I move him instead?" only means something after a turn
              that named a person. We resolve pronouns against the ids the
              previous turn actually resolved -- never against a guess -- and
              we SAY what we carried, because a silent carry is how a
              controller ends up reading an answer about the wrong crew member.

    WHAT-IF   Events stack as overlays on the immutable snapshot, so a second
              disruption is evaluated against a world where the first already
              happened. That is the "chained disruptions" case, and it is free
              because the base snapshot is never mutated -- `reset()` drops the
              stack and the world is exactly as it was.
    """

    def __init__(self, snap: Snapshot, use_model: bool | None = None):
        self.base = snap
        self.advisor = Advisor(snap, use_model=use_model)
        self.turns: list[Turn] = []
        self.events: list[Any] = []

    # ---- what-if stack ------------------------------------------------
    @property
    def what_if(self) -> list[str]:
        return [e.describe() for e in self.events]

    def push_event(self, event: Any) -> None:
        from .events import apply
        self.events.append(event)
        self.advisor.snap = apply(self.base, self.events)

    def reset(self) -> None:
        self.events.clear()
        self.turns.clear()
        self.advisor.snap = self.base

    # ---- context ------------------------------------------------------
    def _last_ids(self) -> tuple[str | None, str | None]:
        """The crew and pairing the most recent successful turn resolved."""
        for turn in reversed(self.turns):
            a = turn.answer
            if a.refusal or not a.plan:
                continue
            crew = a.plan.args.get("crew_id")
            pair = a.plan.args.get("pairing_id")
            if not crew and isinstance(a.payload, dict):
                rec = a.payload.get("recommended") or {}
                crew = rec.get("crew_id")
            if crew or pair:
                return crew, pair
        return None, None

    def _carry(self, question: str) -> tuple[str, list[str]]:
        notes: list[str] = []
        has_crew = bool(re.search(r"\bC-\d{4}\b", question))
        has_pair = bool(re.search(r"\bP-\d{4}\b", question))
        if has_crew and has_pair:
            return question, notes
        if not (_PRONOUN.search(question) or _THAT_TRIP.search(question)
                or question.lower().startswith(("and ", "what about", "instead"))):
            return question, notes

        crew, pair = self._last_ids()
        extra = []
        if crew and not has_crew:
            extra.append(crew)
            notes.append(f"carried {crew} from the previous question")
        if pair and not has_pair:
            extra.append(pair)
            notes.append(f"carried {pair} from the previous question")
        if extra:
            question = f"{question} ({' '.join(extra)})"
        return question, notes

    # ---- ask ----------------------------------------------------------
    def ask(self, question: str) -> Answer:
        resolved, notes = self._carry(question)
        a = self.advisor.ask(resolved)
        a.question = question          # show what the controller typed
        a.assumptions = notes + a.assumptions
        a.overlays = self.what_if
        self.turns.append(Turn(question, a))
        return a


# --------------------------------------------------------------------------
# deterministic summaries -- the answer with no model in the loop at all
# --------------------------------------------------------------------------


def summarise(tool: str, p: Any) -> str:
    if tool == "check_legality":
        if p["legal"]:
            s = f"{p['crew_id']} can legally cover {p['pairing_id']}."
            if p.get("consequence"):
                s += " " + p["consequence"]
            return s
        return (f"{p['crew_id']} cannot cover {p['pairing_id']}. "
                + " ".join(p["issues"]))
    if tool == "rank_cover_options":
        legal = [o for o in p["options"] if o["crew_id"]]
        if not legal:
            return (f"No legal cover exists for {p['pairing_id']}. All "
                    f"{p['candidate_pool_size']} {p['role']}s were excluded; "
                    f"cancellation is the only legal action.")
        r = p["recommended"]
        tied = [o for o in legal if o["cost_inr"] == r["cost_inr"]]
        s = (f"{len(legal)} legal option(s). Cheapest: {r['action']} at "
             f"INR {r['cost_inr']:,}.")
        if len(tied) > 1:
            s += f" {len(tied)} options tie at that price."
        s += f" {len(p['excluded_candidates'])} candidates were ruled out."
        return s
    if tool in ("simulate_crew_unavailable", "simulate_station_closure",
                "simulate_delay", "simulate_cert_expiry"):
        return p["summary"]
    if tool == "duty_hours":
        if p["count"] == 1:
            r = p["rows"][0]
            return (f"{r['crew_id']} has {r['hours']}h over {p['window_days']} "
                    f"days ending {p['end_date']}, leaving {r['headroom_hours']}h "
                    f"of headroom against the {p['limit']}h limit.")
        return f"{p['count']} crew match."
    if tool == "get_reserves":
        return (f"{p['count']} reserve(s) on call: "
                + ", ".join(f"{r['crew_id']} ({r['rank']}, "
                            f"{r['window']['start']}-{r['window']['end']}Z)"
                            for r in p["reserves"][:6]))
    if tool == "find_flights":
        if "answer" in p:
            return f"{p['answer']}"
        return f"{p['count']} flight(s): " + ", ".join(p["flights"][:12])
    if tool == "find_crew":
        return (f"{p['count']} crew: "
                + ", ".join(f"{r['crew_id']} ({r['rank']}, {r['base']})"
                            for r in p["rows"][:8]))
    if tool == "get_roster":
        if p["count"] == 1:
            pp = p["pairings"][0]
            return (f"{pp['pairing_id']} ({pp['aircraft']}, {pp['aircraft_type']}): "
                    + ", ".join(f"{c['crew_id']} {c['role']}" for c in pp["crew"]))
        return f"{p['count']} pairing(s)."
    if tool == "get_certifications":
        return (f"{p['count']} certification(s): "
                + ", ".join(f"{r['crew_id']} {r['cert_type']} to {r['valid_to']}"
                            for r in p["certifications"][:8]))
    if tool == "minimal_repair":
        if p.get("already_legal"):
            return f"{p['crew_id']} is already legal for {p['pairing_id']}."
        r = p["repairs"][0] if p["repairs"] else None
        return (f"{p['crew_id']} is not legal for {p['pairing_id']}. "
                + (f"Smallest fix: {r['lever']}." if r else ""))
    if tool == "cover_fragility":
        crit = p["critical"]
        return (f"{len(crit)} trip(s) have one or zero legal replacements. "
                + "; ".join(f"{c['pairing_id']} on {c['date']} ({c['role']}: "
                            f"{', '.join(c['candidates']) or 'none'})"
                            for c in crit[:3]))
    if tool == "latent_breaches":
        if not p["count"]:
            return "No already-illegal assignments found in the roster."
        return ("Already illegal: "
                + "; ".join(f"{b['crew_id']} on {b['date']} ({b['rule']})"
                            for b in p["breaches"]))
    if tool == "reserve_gaps":
        return "; ".join(
            f"{g['rank']}/{g['aircraft_type']}: no cover for "
            f"{len(g['uncovered_hours_utc'])}h of the day" for g in p["gaps"])
    if tool == "solve_joint_cover":
        return (f"Joint plan at INR {p['total_cost_inr']:,}: "
                + "; ".join(f"{k} -> {v['crew_id']}"
                            for k, v in p["assignments"].items())
                + ("" if p["proven_optimal"] else " (legal, not proven optimal)"))
    if tool == "draft_notification":
        lock = "fact-locked" if p["fact_locked"] else (
            "UNVERIFIED: " + ", ".join(p["unverified_tokens"]))
        return "Draft for %s (%s). %s" % (
            p["audience"], lock, p["draft"].split("\n")[0])
    if tool == "notification_packet":
        return (f"Callout packet for {p['crew_id']} on {p['pairing_id']}: "
                f"report {p['days'][0]['report_utc']} at {p['report_place']}.")
    if tool == "compute_duty_period":
        if "answer" in p:
            return f"Earliest next report: {p['answer']}."
        return (f"{p['pairing_id']} on {p['date']}: FDP {p['fdp_hours']}h against "
                f"a {p['fdp_limit_hours']}h limit ({p['sectors']} sectors).")
    if tool == "get_risk_signals":
        if p["count"] == 1:
            s = p["signals"][0]
            return (f"{s['crew_id']} risk {s['score']} ({'; '.join(s['drivers'])}). "
                    f"Provided input, not our prediction.")
        return f"{p['count']} risk signal(s)."
    return json.dumps(p, default=str)[:400]
