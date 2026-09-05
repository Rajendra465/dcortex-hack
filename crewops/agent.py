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
from .trace import CHOOSE, Option, Trace
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


# Different data domains end on different days, so "out of range" is a
# per-capability fact, not one global cliff. Certificates run to 2032; the
# roster stops after a week. One shared horizon meant every licence-expiry
# question past the 20th was refused against data that was sitting right there.
CERT_TOOLS = {"get_certifications", "simulate_cert_expiry"}


def _dates_in(question: str) -> list[date]:
    out = []
    for m in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", question):
        try:
            out.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    return out


def screen_dates(question: str, snap: Snapshot,
                 tool: str | None = None) -> Refusal | None:
    """The answerable horizon equals the data horizon -- of the right domain.

    Called twice. Before planning, `tool` is None and the check is the UNION of
    every domain, so a 2027 certificate question survives to reach the planner.
    After planning it is called again with the chosen capability, and the tight
    horizon for that capability applies -- so asking for a roster in 2027 is
    still refused, and refused with the horizon that actually bounds it.
    """
    lo, hi = snap.horizon
    clo, chi = snap.cert_horizon
    if tool is None:
        upper, what = max(hi, chi), "roster, duty history or certificate"
        need = "data covering that date"
    elif tool in CERT_TOOLS:
        upper, what = chi, "certificate"
        need = "certificate records covering that date"
    else:
        upper, what = hi, "roster"
        need = "roster data covering that date"

    for d in _dates_in(question):
        if snap.history_start <= d <= upper:
            continue
        return Refusal(
            "OUT_OF_RANGE",
            f"{d} is outside the {what} data I have.",
            have=f"rosters and flights {lo} to {hi}; duty history back to "
                 f"{snap.history_start}; certificate validity to {chi}",
            would_need=need,
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

# Tokens whose digits are part of a NAME, a DATE or a CLOCK TIME, and so make
# no claim about quantity. Order matters: the ISO timestamp must be stripped
# before the bare date, or the date pattern eats its first ten characters and
# leaves "T03:30:00Z" behind.
#
# Every entry here earned its place by producing a false block on a correct
# answer. The trailing-Z cases are the instructive ones: `\b\d{1,2}:\d{2}\b`
# matches "04:00" but NOT the "16:00" in "04:00-16:00Z", because Z is a word
# character and kills the closing boundary -- so exactly one half of every
# on-call window was being reported as an uncited figure.
_STRIP = [
    re.compile(r"\b20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z?"),
    re.compile(r"\bC-\d{4}\b"), re.compile(r"\bP-\d{4}\b"),
    re.compile(r"\bDX\d{3}(?:-\d{4}-\d{2}-\d{2})?\b"), re.compile(r"\bVT-DX[A-F]\b"),
    re.compile(r"\bRULE-[A-Z]+-\d+\b"), re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?Z?"),
    re.compile(r"\bA320\b"), re.compile(r"\bATR ?72\b"),
    # A digit glued to the end of a word is part of the word --
    # medical_class1, class2 -- and never a quantity.
    re.compile(r"[A-Za-z_]\d+\b"),
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
    trace: Trace | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.refusal:
            d = {**self.refusal.to_dict(), "query": self.question,
                 "confidence": "none", "elapsed_ms": round(self.ms, 1)}
            if self.trace:
                d["trace"] = self.trace.to_dict()
            return d
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
            "trace": self.trace.to_dict() if self.trace else None,
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
        tr = Trace()

        def done(**kw: Any) -> Answer:
            kw.setdefault("tier", 0); kw.setdefault("plan", None)
            kw.setdefault("payload", None); kw.setdefault("prose", "")
            kw.setdefault("provenance", "deterministic")
            kw.setdefault("ledger", Ledger())
            return Answer(question=question, trace=tr,
                          ms=(time.perf_counter() - t0) * 1000, **kw)

        # 1. policy screens -- declarative, and cheaper than any model call
        for label, screen in (("scope", screen_scope(question)),
                              ("horizon", screen_dates(question, self.snap))):
            if screen:
                tr.gate(label, False, screen.message, refusal=screen.kind)
                return done(refusal=screen, prose=screen.message)
            tr.gate(label, True)
        for cid in PAT["crew_id"].findall(question):
            if cid not in self.snap.crew:
                r = Refusal("UNKNOWN_ENTITY", f"{cid} is not in this dataset.",
                            have=f"{len(self.snap.crew)} crew on file")
                tr.gate("entity exists", False, r.message, refusal=r.kind)
                return done(refusal=r, prose=r.message)
        tr.gate("entity exists", True)

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
        tr.derive("entities", ", ".join(
            f"{k}={v}" for k, v in ents.as_dict().items() if v) or "none found",
            **{k: v for k, v in ents.as_dict().items() if v})

        ranked = score_intents(question, ents)
        top = ranked[0][0] if ranked else 0.0
        tr.choose("route", [
            Option(i.tool, s, chosen=(n == 0 and s >= FLOOR),
                   why=("above the confident bar" if n == 0 and s >= CONFIDENT
                        else "leads, but below the confident bar" if n == 0
                        else "outscored"))
            for n, (s, i) in enumerate(ranked[:5])],
            detail=(f"{len(ranked)} capabilities scored; "
                    f"lead {top:.2f} vs confident bar {CONFIDENT}"),
            confident_bar=CONFIDENT, floor=FLOOR)

        plan, clarify = None, None
        try:
            if top >= CONFIDENT:
                plan = plan_from_index(question, ents, self.snap, CONFIDENT)
        except PlanError as e:
            # The index is sure WHICH capability this is and cannot bind it.
            # Hold that clarifying question: the model may still resolve the
            # missing argument, and if it cannot, asking beats guessing.
            clarify = e

        # The planner race, recorded. Two independent planners can answer this
        # question -- a deterministic intent index and the model -- and either
        # can win. Where both produce a plan we keep BOTH and compare them,
        # because two planners that agree is evidence about the question, and
        # two that disagree is a fact a controller should see rather than a tie
        # somebody silently broke. This is the only "council" in the system,
        # and it works because the arbiter is deterministic: the index is not
        # judging the model's prose, it is proposing a rival typed plan that
        # the same validator has to accept.
        index_plan, model_plan = plan, None
        if self.client:
            model_plan = plan_from_model(question, ents, self.client)
        if plan is None:
            plan = model_plan
        if plan is None and clarify is None and top >= FLOOR:
            try:
                plan = index_plan = plan_from_index(question, ents,
                                                    self.snap, FLOOR)
            except PlanError as e:
                clarify = e

        if index_plan or model_plan:
            agree = (index_plan is not None and model_plan is not None
                     and index_plan.tools == model_plan.tools)
            chosen = "index" if plan is index_plan else "model"
            opts = []
            if index_plan is not None:
                opts.append(Option("index: " + " -> ".join(index_plan.tools),
                                   chosen=plan is index_plan,
                                   why="deterministic, reproducible, instant"))
            elif self.client:
                opts.append(Option("index: no plan", chosen=False,
                                   why=f"lead {top:.2f} below the bar"))
            if model_plan is not None:
                opts.append(Option("model: " + " -> ".join(model_plan.tools),
                                   chosen=plan is model_plan,
                                   why="reads phrasing the index has never seen"))
            elif self.client:
                opts.append(Option("model: no plan", chosen=False,
                                   why="model returned nothing usable"))
            tr.choose("planners", opts, agreement=(
                "both planners produced the same chain" if agree
                else "planners disagreed - the validator arbitrates" if
                (index_plan and model_plan) else
                f"only the {chosen} planner produced a plan"),
                detail=f"{chosen} plan adopted")
        if plan is None and clarify is not None:
            # "add the which crew member or trip" -- the naive f-string read
            # as broken English the moment `would_need` started carrying a
            # phrase rather than a bare parameter name.
            need = clarify.would_need
            tip = (f"say {need}" if need and need[0].islower()
                   and need.split()[0] in {"which", "what", "how", "who"}
                   else f"add the {need}")
            r = Refusal("CLARIFY", clarify.message, have=clarify.have,
                        would_need=need,
                        suggestions=[tip] if need else [])
            return done(refusal=r, prose=r.message)
        if plan is None:
            r = Refusal(
                "PARSE_FAIL", "I didn't understand that one.",
                have="roster and flight lookup, duty hours, certifications, "
                     "legality checks, disruption impact, ranked cover options, "
                     "joint recovery, notification drafting and proactive scans",
                suggestions=["name a crew id, pairing, flight or date",
                             "or say what you want to do: cover, check, impact"])
            return done(refusal=r, prose=r.message)

        # 3. validate BEFORE anything executes. The date screen runs again now
        # that we know WHICH capability was chosen: the pre-plan pass used the
        # union of every domain's horizon so a 2027 certificate question could
        # get this far, and this pass applies the horizon that actually bounds
        # the tool we landed on.
        for st in plan.steps:
            late = screen_dates(question, self.snap, st.tool)
            if late:
                tr.gate("tool horizon", False, late.message, tool=st.tool)
                return done(plan=plan, refusal=late, prose=late.message)
        tr.gate("tool horizon", True,
                f"{plan.tool}'s own horizon, not the union of every domain's")
        try:
            validate(plan)
        except PlanError as e:
            r = Refusal("CLARIFY" if e.would_need else "PARSE_FAIL", e.message,
                        have=e.have, would_need=e.would_need)
            tr.gate("plan valid", False, e.message)
            return done(plan=plan, refusal=r, prose=r.message)
        tr.gate("plan valid", True,
                f"{len(plan.steps)} step(s), every argument bound and typed")

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
        tr.derive("kernel", " -> ".join(st.tool for st in plan.steps),
                  facts=len(ledger.as_dict()), tier=tier)

        # The layer under the plan: the kernel checked every crew member on the
        # payroll against all seven rules and threw most of them away. Those
        # rejections are the answer to "why not someone else", and until now
        # they were computed and discarded.
        if isinstance(payload, dict) and "excluded_candidates" in payload:
            opts = [Option(o.get("crew_id") or "?", chosen=(o.get("rank") == 1),
                           why=(o.get("reasoning") or "")[:120])
                    for o in (payload.get("options") or [])[:8]]
            for ex in (payload.get("excluded_candidates") or [])[:8]:
                if isinstance(ex, dict):
                    opts.append(Option(ex.get("crew_id") or "?", chosen=False,
                                       why=(ex.get("reason") or
                                            ex.get("why") or "")[:120]))
            pool = payload.get("candidate_pool_size")
            legal = sum(1 for o in (payload.get("options") or [])
                        if o.get("legal"))
            tr.choose("candidates", opts,
                      detail=(f"{pool} crew checked against 7 rules; "
                              f"{legal} legal, {len(payload.get('options') or [])} ranked"),
                      pool=pool, legal=legal)
        summary = summarise(plan.tool, payload)
        prose, prov = narrate(question, ledger, self.client, summary)
        tr.speak("narrate", f"prose written by the {prov}", source=prov)

        # The guard reads the finished sentence back against the ledger. It is
        # the last thing between a fluent number and a controller's screen.
        clean, bad = guard_numbers(prose, ledger)
        tr.gate("number guard", clean,
                (f"every figure in the sentence traces to the kernel"
                 if clean else f"unbacked: {', '.join(bad)}"),
                checked=len(ledger.allowed_numbers()),
                unbacked=bad or None)
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
# "which flights are NOW uncrewed?" names nobody and contains no pronoun, but
# it is unmistakably about the disruption just stacked. Without this the
# question fell through to a plain schedule listing and answered with all 147
# flights -- technically true, and completely useless.
_CONSEQUENCE = re.compile(
    r"\b(now|as a result|because of (?:that|this)|knock[- ]on|downstream|"
    r"what breaks|what is affected|what's affected|impact)\b", re.I)


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
        """The crew and pairing the conversation is currently about.

        A stacked what-if outranks a previous answer: once "C-1042 is sick" is
        the world we are in, that is who the next unqualified question means,
        even if the turn after it asked about someone else in passing.
        """
        for ev in reversed(self.events):
            cid = getattr(ev, "crew_id", None)
            if cid:
                return cid, getattr(ev, "pairing_id", None)
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
        refers_back = (_PRONOUN.search(question) or _THAT_TRIP.search(question)
                       or question.lower().startswith(("and ", "what about",
                                                       "instead")))
        # A consequence question only refers back when there IS a disruption
        # to be a consequence of. With an empty stack, "what is the impact"
        # is a fresh question and must not silently inherit a subject.
        if not refers_back and not (self.events and _CONSEQUENCE.search(question)):
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
        a = self._ask_in_the_right_world(resolved, notes)
        a.question = question          # show what the controller typed
        a.assumptions = notes + a.assumptions
        a.overlays = self.what_if
        self.turns.append(Turn(question, a))
        return a

    def _ask_in_the_right_world(self, resolved: str,
                                notes: list[str]) -> Answer:
        """Pick the snapshot the question is actually about.

        A stacked what-if is applied to the snapshot, which is right for every
        LOOKUP -- "who is on P-2291 now" must see the disrupted roster. It is
        wrong for the SIMULATION of the event that was just stacked: the crew
        is already stripped, so the simulator removes nobody and reports that
        nothing broke. Asking "what breaks?" straight after ":whatif C-1042
        sick" answered "0 flights uncrewed", which is the most dangerous kind
        of wrong answer this system can give -- a disruption that reads clean.

        So a consequence question is planned one layer down, in the world
        BEFORE the top event, and the simulator re-applies it itself. If the
        plan turns out to be a lookup after all, we discard that answer and
        redo it against the full overlay. Two passes of a few milliseconds is
        a cheap price for not confusing the two worlds.
        """
        if not self.events:
            return self.advisor.ask(resolved)

        from .events import apply
        overlaid = self.advisor.snap
        try:
            self.advisor.snap = apply(self.base, self.events[:-1])
            a = self.advisor.ask(resolved)
        finally:
            self.advisor.snap = overlaid

        if a.plan and REGISTRY[a.plan.tool].kind == "simulation":
            notes.append(f"evaluated against the world before "
                         f"{self.events[-1].describe()}, which the simulation "
                         f"re-applies")
            return a
        return self.advisor.ask(resolved)


# --------------------------------------------------------------------------
# deterministic summaries -- the answer with no model in the loop at all
# --------------------------------------------------------------------------


def summarise(tool: str, p: Any) -> str:
    """One deterministic sentence per capability. No model involved."""
    # A conjunctive plan carries the other half of the question beside the
    # leader's payload. Summarise both, leader first, so neither half of
    # "what is their rank, and their 28-day hours" is silently dropped.
    if isinstance(p, dict) and p.get("also"):
        rest = p["also"]
        head = summarise(tool, {k: v for k, v in p.items() if k != "also"})
        return " ".join([head] + [summarise(o["tool"], o["result"])
                                  for o in rest])
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
        ans = p.get("answer")
        if isinstance(ans, dict) and "passengers" in ans:
            return (f"{ans['passengers']} seats at risk across {ans['legs']} "
                    f"leg(s). Cancelling costs INR {ans['cost_inr']:,}.")
        if isinstance(ans, dict) and "seats" in ans and "flights" in ans:
            # Every figure in this sentence has to exist in the ledger. The
            # first version wrote "(+12 more)", and 12 was arithmetic done in
            # the sentence rather than by the kernel -- so the guard blocked
            # the answer. It was right to.
            names = ", ".join(ans["flights"][:6])
            tail = f" -- {ans['why']}." if ans.get("why") else ""
            return (f"{ans['seats']} seats, the most of any leg. "
                    f"{ans['count']} legs share it: {names}{tail}")
        if ans is not None:
            return f"{ans}"
        if not p["count"]:
            # "0 flight(s):" is a true answer that reads like a broken one. A
            # controller cannot tell whether nothing matched or the filter was
            # wrong, so an empty result says what it looked for. Asking for DEL
            # departures on an afternoon that has none should read as an
            # answer, not a shrug.
            said = [f"{k.replace('_', ' ')} {v}" for k, v in (
                ("departing", p.get("dep_station")),
                ("arriving", p.get("arr_station")),
                ("on", p.get("date")),
                ("after", p.get("dep_after_utc")),
                ("before", p.get("dep_before_utc")),
            ) if v]
            return ("No flights " + ", ".join(said) + "." if said
                    else "No flights match that.")
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
