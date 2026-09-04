"""Orchestration: question -> validated multi-step plan -> execution.

This replaces a hand-written regex cascade that had become the real router. That
design failed in two ways worth naming, because they drove everything here:

  * It did not generalise. Every unseen phrasing needed a new pattern, so the
    grammar grew by patch -- and each patch was a chance to reintroduce the same
    word-boundary bug that once let "probability" through to a crew lookup.
  * It could only ever emit ONE tool call. "C-1042 is sick, what should I do?"
    is genuinely three steps -- assess the impact, rank the cover, draft the
    message -- and a single-tool router cannot express that.

So planning is now data-driven and multi-step:

  PLANNER    the model reads the typed tool catalog and emits a Plan: an ordered
             list of steps whose arguments may reference earlier results. With
             no model configured, a deterministic index over the same
             declarative examples produces a single-step plan.
  VALIDATOR  every step is schema-checked and every reference resolved against
             steps that actually exist, BEFORE anything executes.
  EXECUTOR   runs the steps, threads results forward, merges evidence ledgers.

What is deliberately still pattern-based: entity extraction (a crew id really is
C-dddd) and the capability screen (a declarative list of what this dataset does
not contain). Those are lexical facts and policy data, not control flow.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date as _date, timedelta
from typing import Any

from .data import Snapshot
from .rules import Ledger
from .tools import REGISTRY, describe_tools


# ==========================================================================
# the plan
# ==========================================================================


@dataclass
class Step:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "why": self.why}


@dataclass
class Plan:
    steps: list[Step]
    source: str = "index"        # index | model:<provider>
    why: str = ""

    @property
    def tool(self) -> str:
        """The step that produced the answer -- the last one."""
        return self.steps[-1].tool if self.steps else ""

    @property
    def args(self) -> dict[str, Any]:
        return self.steps[-1].args if self.steps else {}

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "why": self.why,
                "parsed_by": self.source,
                "steps": [s.to_dict() for s in self.steps]}


class PlanError(Exception):
    """An invalid plan. Carries what was wrong and what would fix it."""

    def __init__(self, message: str, have: str = "", would_need: str = ""):
        super().__init__(message)
        self.message = message
        self.have = have
        self.would_need = would_need


# ==========================================================================
# entity extraction -- lexical, so genuinely regex
# ==========================================================================

PAT = {
    "crew_id": re.compile(r"\bC-\d{4}\b"),
    "pairing_id": re.compile(r"\bP-\d{4}\b"),
    "flight_no": re.compile(r"\bDX\d{3}\b", re.I),
    "aircraft": re.compile(r"\bVT-DX[A-F]\b", re.I),
    "station": re.compile(r"\b(?:BLR|DEL|BOM|MAA|HYD|CCU|COK|GOI)\b"),
    "date": re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),
    "time": re.compile(r"\b(\d{1,2}):(\d{2})\s*Z?\b"),
    "duration": re.compile(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?)\b",
                           re.I),
    "rank": re.compile(r"\b(senior cabin crew|first officer|captain|cabin crew)\b",
                       re.I),
}

RANKS = {"captain": "Captain", "first officer": "First Officer",
         "senior cabin crew": "Senior Cabin Crew", "cabin crew": "Cabin Crew"}


@dataclass
class Entities:
    crew: list[str] = field(default_factory=list)
    pairings: list[str] = field(default_factory=list)
    flights: list[str] = field(default_factory=list)
    aircraft: list[str] = field(default_factory=list)
    stations: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
    hours: float | None = None
    rank: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}


def extract(question: str, snap: Snapshot) -> Entities:
    up = question.upper()
    e = Entities()
    e.crew = [c for c in PAT["crew_id"].findall(question) if c in snap.crew]
    e.pairings = [p for p in PAT["pairing_id"].findall(up) if p in snap.pairings]
    e.flights = sorted(set(PAT["flight_no"].findall(up)))
    e.aircraft = sorted(set(PAT["aircraft"].findall(up)))
    e.stations = sorted(set(PAT["station"].findall(up)))
    e.dates = PAT["date"].findall(question)
    e.times = [f"{int(h):02d}:{m}" for h, m in PAT["time"].findall(question)]
    m = PAT["duration"].search(question)
    if m:
        v = float(m.group(1))
        e.hours = round(v / 60.0, 2) if m.group(2).lower().startswith("min") else v
    r = PAT["rank"].search(question)
    if r:
        e.rank = RANKS[r.group(1).lower()]

    # names, only when no id was given -- they are not unique in this dataset
    if not e.crew:
        for c in snap.crew.values():
            surname = c.name.split()[-1]
            if len(surname) > 3 and re.search(rf"\b{re.escape(surname)}\b",
                                              question, re.I):
                e.crew = [min(x.crew_id for x in snap.crew.values()
                              if x.name.split()[-1] == surname)]
                break
    return e


# ==========================================================================
# the intent index -- DATA, not control flow
#
# Each capability declares what it is for and how people ask for it. Adding a
# capability means adding rows here; it never means editing a branch.
# ==========================================================================


@dataclass
class Intent:
    tool: str
    examples: list[str]
    needs: tuple[str, ...] = ()      # entity kinds required to be usable
    boost: float = 1.0


INTENTS: list[Intent] = [
    Intent("rank_cover_options", [
        "what should i do", "give me options", "ranked options", "resolve this",
        "who can cover", "who can take", "who is free to take", "cheapest cover",
        "find a replacement", "who else can operate", "recommend a replacement",
        "called in sick what now", "options to cover the trip",
    ], needs=("pairing_or_crew",), boost=1.25),

    Intent("check_legality", [
        "is it legal", "does anyone breach", "can they cover", "can he take",
        "if i move onto", "put them on", "swap them onto", "assign instead",
        "would that breach a limit", "does that violate a rule",
        "try them instead", "is that allowed",
    ], needs=("crew",), boost=1.2),

    Intent("simulate_crew_unavailable", [
        "called in sick which flights", "what breaks if", "which flights uncrewed",
        "what is the impact", "what is affected", "which legs are uncovered",
        "now uncrewed",
    ], needs=("crew",), boost=1.1),

    Intent("simulate_station_closure", [
        "station is closed", "airport shut", "closed between", "closure impact",
        "what is affected by the closure", "closed from until",
    ], needs=("station",), boost=1.3),

    Intent("simulate_delay", [
        "is delayed", "delayed by minutes", "runs late", "tech delay",
        "does the crew breach after the delay",
    ], needs=("duration",), boost=1.2),

    Intent("simulate_cert_expiry", [
        "certificate lapsed", "licence expired", "recurrent training expired",
        "operate with an expired certificate", "cert has lapsed",
    ], needs=("crew",)),

    Intent("minimal_repair", [
        "smallest change", "minimum change", "what would it take",
        "how do i make it legal", "what would make them legal", "how to fix",
    ], needs=("crew",), boost=1.2),

    Intent("draft_notification", [
        "draft the notification", "write the message", "notify the crew",
        "send the callout", "draft a callout", "message for the crew",
    ], needs=("crew",), boost=1.3),

    Intent("solve_joint_cover", [
        "both captains sick", "two crew are out", "simultaneous sick calls",
        "joint plan", "allocate across both",
    ], boost=1.2),

    Intent("cover_fragility", [
        "where are we thin", "single point of failure", "cover depth",
        "most exposed trips", "which trips are fragile", "thinnest cover",
    ], boost=1.2),

    Intent("latent_breaches", [
        "already illegal", "anything wrong with the roster", "latent breaches",
        "any illegal assignments", "what have we missed",
    ], boost=1.2),

    Intent("reserve_gaps", [
        "reserve coverage gaps", "standby holes", "hours with no standby",
        "when do we have no reserve",
    ], boost=1.2),

    Intent("duty_hours", [
        "how many duty hours", "duty clock", "headroom", "hours left this week",
        "flight hours", "block hours", "how close to the limit",
    ]),

    Intent("get_reserves", [
        "who is on reserve", "standby crew", "on call", "reserve list",
        "who is on standby",
    ]),

    Intent("get_certifications", [
        "certificates expiring", "licence expiry", "medical validity",
        "whose certificates expire", "expiring within days",
    ]),

    Intent("get_roster", [
        "who is assigned", "roster for", "crew on the pairing",
        "show the pairings", "who is on this trip", "crew complement",
    ]),

    Intent("find_flights", [
        "which flights", "flights departing", "flights arriving", "schedule",
        "how many flights", "longest block time", "which destinations",
        "what legs are there", "how many seats",
    ]),

    Intent("find_crew", [
        "who are the captains", "list the captains", "list the crew",
        "crew based at", "which crew are rated", "how many first officers",
        "show me the crew at", "which captains are based",
    ]),

    Intent("get_risk_signals", [
        "risk score", "disruption risk", "who is highest risk",
    ]),

    Intent("compute_duty_period", [
        "earliest next report", "minimum rest", "when can they report next",
        "how long is the duty",
    ]),

    Intent("get_rulebook", [
        "what does the rule say", "what is the limit", "show me the rules",
        "what are the callout rates", "how much does a callout cost",
    ]),
]

_STOP = {"the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "at",
         "is", "are", "was", "be", "with", "from", "it", "my", "me", "we",
         "us", "you", "your", "please", "there", "their", "did", "does"}

# Kept deliberately: what / should / who / can / how / do / if / any. They look
# like noise but they are the ONLY content in "what should i do", which is the
# single most important thing a controller types. Stripping them made that
# phrase stem to the empty set and match nothing at all.


def _tokens(s: str) -> set[str]:
    """Fold inflection to a stem, so "captains" == "captain".

    Stripping runs to a FIXED POINT. One pass is not enough: "pairings" loses
    the s to give "pairing", while "pairing" loses "ing" to give "pair" -- so a
    single pass makes the singular and plural of the same word disagree, which
    is the very bug this replaced.

    The length guard is what stops over-folding: "leg" is too short to strip, so
    it never collides with "legal".
    """
    out = set()
    for w in re.findall(r"[a-z]+", s.lower()):
        if w in _STOP or len(w) < 2:
            continue
        changed = True
        while changed and len(w) > 4:
            changed = False
            if w.endswith("ies"):
                w, changed = w[:-3] + "y", True
            elif w.endswith("es") and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
                w, changed = w[:-2], True          # "matches" -> "match"
            elif w.endswith("s") and not w.endswith("ss"):
                w, changed = w[:-1], True          # "reserves" -> "reserve"
            elif w.endswith("ing") and len(w) > 6:
                w, changed = w[:-3], True
            elif w.endswith("ed") and len(w) > 5:
                w, changed = w[:-2], True
        out.add(w)
    return out


_INDEX = [(i, [_tokens(x) for x in i.examples]) for i in INTENTS]


def score_intents(question: str, ents: Entities) -> list[tuple[float, Intent]]:
    """Score every capability against the question. Transparent and orderable."""
    q = _tokens(question)
    have = {
        "crew": bool(ents.crew), "pairing": bool(ents.pairings),
        "station": bool(ents.stations), "duration": ents.hours is not None,
        "pairing_or_crew": bool(ents.pairings or ents.crew or ents.flights),
    }
    scored: list[tuple[float, Intent]] = []
    for intent, example_tokens in _INDEX:
        best = 0.0
        for toks in example_tokens:
            if toks:
                overlap = len(q & toks)
                if overlap:
                    best = max(best, overlap / len(toks))
        if not best:
            continue
        s = best * intent.boost
        # Parameter affinity: if the question named something this tool
        # can actually accept, prefer it. "list the captains at DEL" ties
        # on words alone, and only find_crew takes a rank.
        params = REGISTRY[intent.tool].params
        if ents.rank and "rank" in params:
            s += 0.20
        if ents.stations and ("base" in params or "station" in params):
            s += 0.05
        if intent.needs and not all(have.get(n, False) for n in intent.needs):
            s *= 0.25          # plausible, but we lack what it needs
        scored.append((s, intent))
    scored.sort(key=lambda x: -x[0])
    return scored


# ==========================================================================
# argument binding -- entities into a specific tool's parameters
# ==========================================================================


def bind_args(tool: str, ents: Entities, snap: Snapshot,
              question: str) -> dict[str, Any]:
    """Fill a tool's parameters from the entities we extracted.

    Only ever writes parameters the tool actually declares, so a mis-scored
    intent yields an invalid plan (caught by the validator) rather than a
    confidently wrong answer.
    """
    params = REGISTRY[tool].params
    a: dict[str, Any] = {}

    def put(k: str, v: Any) -> None:
        if k in params and v is not None and k not in a:
            a[k] = v

    pairing = ents.pairings[0] if ents.pairings else None
    crew = ents.crew[0] if ents.crew else None

    if not pairing and crew:
        ps = sorted(snap.pairings_for_crew(crew), key=lambda p: p.days[0].date)
        if ents.dates:
            ps = [p for p in ps
                  if any(d.date >= ents.dates[0] for d in p.days)] or ps
        pairing = ps[0].pairing_id if ps else None
    if not pairing and ents.flights:
        cands = [f for f in snap.flights.values()
                 if f.flight_no == ents.flights[0]]
        if ents.dates:
            cands = [f for f in cands if f.date == ents.dates[0]] or cands
        for f in sorted(cands, key=lambda x: x.date):
            hit = snap.pairing_of_flight(f.flight_id)
            if hit:
                pairing = hit[0].pairing_id
                break

    put("crew_id", crew)
    put("pairing_id", pairing)
    put("aircraft", ents.aircraft[0] if ents.aircraft else None)
    put("flight_no", ents.flights[0] if ents.flights else None)
    put("rank", ents.rank)
    put("role", ents.rank)
    put("delay_hours", ents.hours)

    if ents.dates:
        put("date", ents.dates[0])
        put("end_date", ents.dates[0])
    if "end_date" in params and "end_date" not in a:
        a["end_date"] = snap.snapshot_utc.date().isoformat()
    if tool == "get_reserves" and "date" not in a:
        a["date"] = (snap.snapshot_utc.date() + timedelta(days=1)).isoformat()

    if ents.stations:
        if tool == "simulate_station_closure":
            put("station", ents.stations[0])
        elif len(ents.stations) >= 2:
            put("dep_station", ents.stations[0])
            put("arr_station", ents.stations[1])
        elif re.search(r"\barriv|\binto\b", question, re.I):
            put("arr_station", ents.stations[0])
        else:
            put("dep_station", ents.stations[0])
            put("base", ents.stations[0])

    if tool == "simulate_station_closure" and ents.dates and len(ents.times) >= 2:
        a["start_utc"] = f"{ents.dates[0]}T{ents.times[0]}:00Z"
        a["end_utc"] = f"{ents.dates[0]}T{ents.times[1]}:00Z"
    if tool == "compute_duty_period" and ents.dates and ents.times:
        a["release_utc"] = f"{ents.dates[0]}T{ents.times[0]}:00Z"
    if tool == "find_flights" and re.search(r"\bhow many\b", question, re.I):
        a["aggregate"] = "count"
    if tool == "duty_hours" and re.search(r"\b(flight|block)\s*hour", question,
                                          re.I):
        a["metric"] = "flight"
        a["window_days"] = 28
    if tool == "get_certifications" and ents.dates:
        a["expiring_after"] = ents.dates[0]
        a["expiring_before"] = (_date.fromisoformat(ents.dates[0])
                                + timedelta(days=30)).isoformat()
    if tool == "draft_notification":
        low = question.lower()
        a["audience"] = ("occ" if "occ" in low else
                         "duty_manager" if "manager" in low else "crew")
    return a


# ==========================================================================
# planners
# ==========================================================================


def plan_from_index(question: str, ents: Entities, snap: Snapshot) -> Plan | None:
    """Deterministic single-step plan. Fast, reproducible, no network."""
    for s, intent in score_intents(question, ents)[:4]:
        if s < 0.34:
            break
        args = bind_args(intent.tool, ents, snap, question)
        if [r for r in REGISTRY[intent.tool].required if r not in args]:
            continue
        return Plan([Step(intent.tool, args, intent.examples[0])], "index",
                    f"matched '{intent.examples[0]}' (score {s:.2f})")
    return None


PLANNER_SYSTEM = """You are the planner for an airline Crew Control advisor.

You NEVER answer the question and you NEVER do arithmetic. You choose tools.

Return ONLY JSON:
{"steps":[{"tool":"<name>","args":{...},"why":"<short>"}],"why":"<short>"}

Rules:
- 1 to 3 steps. Use several only when the question genuinely needs them, e.g.
  "X is sick, what should I do and tell them" -> impact, then rank, then draft.
- A later step may reference an earlier result with {"$ref":"1.recommended.crew_id"}
  where 1 is the 1-based step number and the rest is a dotted path.
- Only use tools and parameter names from the catalog. Omit anything you are
  unsure of.
- If no tool fits, return {"steps":[],"why":"<what is missing>"}.

Context: fixed synthetic dataset, dCortex Air. Snapshot 2026-09-14T18:00:00Z.
Flights and rosters cover 2026-09-14..2026-09-20 UTC; "tomorrow" is 2026-09-15.
Crew C-1042, pairings (multi-day trips) P-2291, flights DX412, aircraft VT-DXA,
stations BLR DEL BOM MAA HYD CCU COK GOI."""


def plan_from_model(question: str, ents: Entities, client: Any) -> Plan | None:
    try:
        raw = client.complete(
            PLANNER_SYSTEM,
            f"TOOLS:\n{json.dumps(describe_tools())}\n\n"
            f"ENTITIES ALREADY RESOLVED: {json.dumps(ents.as_dict())}\n\n"
            f"QUESTION: {question}",
            max_tokens=700)
    except Exception:
        return None
    obj = _first_json(raw)
    if not obj:
        return None
    steps = [Step(s.get("tool", ""), s.get("args") or {}, s.get("why", ""))
             for s in (obj.get("steps") or []) if s.get("tool")]
    if not steps:
        return None
    return Plan(steps, f"model:{client.provider}", obj.get("why", ""))


def _first_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


# ==========================================================================
# validation and execution
# ==========================================================================

REF = re.compile(r"^\$?(\d+)\.(.+)$")


def _ref_of(v: Any) -> tuple[str, str] | None:
    if isinstance(v, dict) and "$ref" in v:
        m = REF.match(str(v["$ref"]))
        return (m.group(1), m.group(2)) if m else None
    if isinstance(v, str):
        m = REF.match(v)
        return (m.group(1), m.group(2)) if m else None
    return None


def validate(plan: Plan) -> None:
    """Schema-check every step before any of them runs."""
    if not plan.steps:
        raise PlanError("I couldn't work out which capability that needs.")
    if len(plan.steps) > 4:
        raise PlanError("That needs more steps than I will chain.",
                        would_need="a narrower question")
    for i, st in enumerate(plan.steps, 1):
        t = REGISTRY.get(st.tool)
        if not t:
            raise PlanError(f"There is no capability called '{st.tool}'.",
                            have=", ".join(sorted(REGISTRY)))
        unknown = set(st.args) - set(t.params)
        if unknown:
            raise PlanError(f"'{st.tool}' has no parameter {sorted(unknown)}.",
                            have=", ".join(sorted(t.params)))
        for v in st.args.values():
            ref = _ref_of(v)
            if ref and int(ref[0]) >= i:
                raise PlanError(
                    f"Step {i} references step {ref[0]}, which has not run yet.")
        missing = [r for r in t.required if r not in st.args]
        if missing:
            raise PlanError(f"I need {', '.join(missing)} to answer that.",
                            would_need=", ".join(missing))


def _dig(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(obj, list):
            try:
                obj = obj[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


@dataclass
class Execution:
    results: list[Any]
    ledger: Ledger

    @property
    def final(self) -> Any:
        return self.results[-1] if self.results else None


def execute(plan: Plan, snap: Snapshot) -> Execution:
    """Run the steps, threading earlier results into later arguments."""
    ledger = Ledger()
    results: list[Any] = []
    for st in plan.steps:
        args: dict[str, Any] = {}
        for k, v in st.args.items():
            ref = _ref_of(v)
            if not ref:
                args[k] = v
                continue
            idx, path = int(ref[0]) - 1, ref[1]
            resolved = _dig(results[idx], path) if 0 <= idx < len(results) else None
            if resolved is None:
                raise PlanError(f"'{st.tool}' needed {k} from step {idx + 1}, "
                                f"which did not produce it.")
            args[k] = resolved
        payload, led = REGISTRY[st.tool](snap, **args)
        st.args = args                       # record what actually ran
        results.append(payload)
        ledger.extend(led)
    return Execution(results, ledger)
