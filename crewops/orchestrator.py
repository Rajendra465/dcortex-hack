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
    # True when the steps answer two INDEPENDENT halves of one sentence
    # ("what is C-2087's rank, and their 28-day flight hours") rather than
    # feeding each other. Both halves must reach the answer; picking a winner
    # between them silently drops half of what was asked.
    conjunctive: bool = False

    @property
    def tool(self) -> str:
        """The step that produced the answer -- the last one."""
        return self.steps[-1].tool if self.steps else ""

    @property
    def tools(self) -> list[str]:
        return [s.tool for s in self.steps]

    @property
    def args(self) -> dict[str, Any]:
        return self.steps[-1].args if self.steps else {}

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "why": self.why,
                "parsed_by": self.source, "conjunctive": self.conjunctive,
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
    # `[\s-]*` and not `\s*`: controllers write "a 90-minute delay" at least as
    # often as "90 minutes", and the hyphenated form parsed to no duration at
    # all -- which made simulate_delay fail its `needs` check and drop out of
    # scoring entirely, so the whole question fell through to a roster lookup.
    "duration": re.compile(
        r"(\d+(?:\.\d+)?)[\s-]*(hours?|hrs?|h|minutes?|mins?)\b", re.I),
    # Ranks: plurals, abbreviations and CLASS words. A controller says
    # "captains", "FOs", "pilots" -- never "Senior Cabin Crew" in full. The
    # earlier singular-only pattern silently bound no rank at all, so "give me
    # 5 captains" returned all 150 crew, cabin crew included, presented as
    # captains. Longest alternatives first so "senior cabin crew" wins over
    # "cabin crew".
    "rank": re.compile(
        r"\b(senior cabin crew|scc|first officers?|f/?os?\b|cabin crew|"
        r"captains?|skippers?|pilots?|flight ?deck|crew members?)\b", re.I),
    "rating": re.compile(r"\b(a-?320|atr-? ?72|atr)\b", re.I),
    # A number is a COUNT only when it is counting people. The first
    # version matched any digits followed by a word, so "15 Sep",
    # "28 days" and "45 or more hours" all became a result limit and
    # silently truncated or mis-scored almost every dated question.
    "count": re.compile(
        r"(?:top|first|best|any|need|want|give me|show me|list)\s+(\d{1,3})"
        r"|(\d{1,3})\s+(?:more\s+)?(?:pilots?|captains?|skippers?|"
        r"first officers?|f/?os?|cabin crew|senior cabin crew|scc|"
        r"crew members?|reserves?|standbys?|people|bodies)", re.I),
}

# A rank word can name a CLASS, not one rank. "pilots" is two ranks; asking for
# pilots and getting only First Officers is the kind of quietly wrong answer
# this system exists to avoid.
RANKS: dict[str, list[str]] = {
    "captain": ["Captain"], "captains": ["Captain"],
    "skipper": ["Captain"], "skippers": ["Captain"],
    "first officer": ["First Officer"], "first officers": ["First Officer"],
    "fo": ["First Officer"], "fos": ["First Officer"],
    "f/o": ["First Officer"], "f/os": ["First Officer"],
    "senior cabin crew": ["Senior Cabin Crew"], "scc": ["Senior Cabin Crew"],
    "cabin crew": ["Cabin Crew"],
    "pilot": ["Captain", "First Officer"], "pilots": ["Captain", "First Officer"],
    "flight deck": ["Captain", "First Officer"],
    "crew member": [], "crew members": [],      # no filter, but not "unknown"
}

RATINGS = {"a320": "A320", "a-320": "A320",
           "atr72": "ATR72", "atr 72": "ATR72", "atr-72": "ATR72", "atr": "ATR72"}


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
    # "can C-3305 cover P-2291?" wants a VERDICT; "who can cover P-2291?" wants
    # a LIST. Same vocabulary, opposite tools -- and word overlap alone cannot
    # tell them apart, which is why both used to land on the ranker.
    yes_no: bool = False
    wh: bool = False
    min_hours: float | None = None
    ranks: list[str] = field(default_factory=list)   # a class may be 2 ranks
    rating: str | None = None
    limit: int | None = None

    @property
    def rank(self) -> str | None:
        """The single rank, when the phrase named exactly one."""
        return self.ranks[0] if len(self.ranks) == 1 else None

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
    stem = question.strip().lower()
    e.yes_no = bool(re.match(
        r"(can|could|is|are|does|do|will|would|may|should|any issue|ok to)\b",
        stem))
    e.wh = bool(re.match(r"(who|which|what|where|when|how many|list|show)\b",
                         stem))
    mh = re.search(r"(\d+(?:\.\d+)?)\s*(?:h(?:ours?)?\s*)?"
                   r"(?:or more|\+|plus|and above|and over|or above)",
                   question, re.I)
    if mh:
        e.min_hours = float(mh.group(1))

    r = PAT["rank"].search(question)
    if r:
        e.ranks = RANKS.get(r.group(1).lower().replace("/", "").replace("  ", " "), [])
    g = PAT["rating"].search(question)
    if g:
        e.rating = RATINGS.get(g.group(1).lower().replace(" ", "").replace("-", ""),
                               RATINGS.get(g.group(1).lower()))
    c = PAT["count"].search(question)
    if c:
        n = int(c.group(1) or c.group(2))
        if 1 <= n <= 200:
            e.limit = n

    # Relative dates are resolved HERE, not in the caller. They used to be
    # computed for the assumption note and then thrown away, so "tomorrow"
    # never reached the tools -- and get_reserves' own default quietly filled
    # the hole, meaning the system stated one date and queried another.
    if not e.dates:
        today = snap.snapshot_utc.date()
        low = question.lower()
        if "tomorrow" in low:
            e.dates = [(today + timedelta(days=1)).isoformat()]
        elif "today" in low or "tonight" in low:
            e.dates = [today.isoformat()]
        elif "yesterday" in low:
            e.dates = [(today - timedelta(days=1)).isoformat()]
        else:
            m2 = re.search(r"\b(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|"
                           r"sep|oct|nov|dec)", question, re.I)
            if m2:
                mo = ["jan","feb","mar","apr","may","jun",
                      "jul","aug","sep","oct","nov","dec"].index(
                          m2.group(2)[:3].lower()) + 1
                try:
                    e.dates = [_date(today.year, mo, int(m2.group(1))).isoformat()]
                except ValueError:
                    pass

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
            "resolve their assignment",
        "which reserves can cover",
        "who is qualified and available",
        "cheapest legal way to cover",
        "the captain is sick which reserves cover it",
        "cover the first officer",
        "fix this assignment",
], needs=("pairing_or_crew",), boost=1.25),

    Intent("check_legality", [
        "is it legal", "does anyone breach", "can they cover", "can he take",
        "if i move onto", "put them on", "swap them onto", "assign instead",
        "would that breach a limit", "does that violate a rule",
        "try them instead", "is that allowed",
            "does any rule breach",
        "if assigned to cover does anything breach",
        "would assigning them breach a rule",
        "is the assignment legal",
        "legally operate their rostered duty",
], needs=("crew",), boost=1.2),

    Intent("simulate_crew_unavailable", [
        "called in sick which flights", "what breaks if",
        # These were two and three tokens long, so an unrelated question
        # scored 0.67 on them by containing "which" and "flight". That is how
        # "which single leg has the most seats at risk if cancelled" -- a
        # question naming no crew at all -- came within a boost of being
        # answered as a sick call. Longer cues cannot be part-matched into a
        # false positive.
        "which flights are now uncrewed", "which flights are uncrewed now",
        "what is the crew impact", "what is affected by this",
        "which legs are now uncovered", "which sectors are now uncrewed",
        # The brief's own Tier 2 example -- "Captain C-1042 just called in sick
        # for tomorrow, which flights are now uncrewed?" -- routed to
        # get_reserves, because "called in sick" contains "call" and
        # get_reserves cues on "on call". It answered with a list of standby
        # crew: the wrong question, answered confidently.
        "just called in sick", "called in sick for tomorrow",
        "has called in sick", "has reported sick", "calls in sick today",
        # "sick which flights" and "is sick which flights" reduced to the three
        # tokens {sick, which, flight}, which an unrelated question part-matched
        # at 0.67 on "which" and "flight" alone. Every cue here now carries a
        # word that only appears in a sick call.
        # 1.15, arrived at by measurement rather than taste. At 1.10 the
        # brief's own Tier 2 question went to rank_cover_options (1.36 against
        # 1.32) -- a defensible answer to a question nobody asked, since it
        # asks WHICH FLIGHTS are uncrewed, not who could cover them. At 1.25 it
        # also claimed "which leg has the most seats at risk if cancelled",
        # which names no crew at all. 1.15 clears the first and not the second,
        # and the needs=("crew",) penalty is what keeps the second out.
    ], needs=("crew",), boost=1.15),

    Intent("simulate_station_closure", [
        "station is closed", "airport shut", "closed between", "closure impact",
        "what is affected by the closure", "closed from until",
    ], needs=("station",), boost=1.3),

    Intent("simulate_delay", [
        "is delayed", "delayed by minutes", "runs late", "tech delay",
        "does the crew breach after the delay",
            "what should we do about the delay",
        "after the delay what do we do",
        "the delay causes an fdp breach",
        "how do we handle the delay",
], needs=("duration",), boost=1.2),

    Intent("simulate_cert_expiry", [
        "certificate lapsed", "licence expired", "recurrent training expired",
        "operate with an expired certificate", "cert has lapsed",
            "recurrent training lapsed",
        "their certificate has lapsed",
        "can they legally operate with the lapsed certificate",
        "is their licence still valid for the duty",
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
        "both captains are sick give the optimal joint crewing plan",
        "optimal joint crewing plan",
        "two aircraft need a captain at the same time",
        "plan across both aircraft",
    ], boost=1.3),

    Intent("cover_fragility", [
        "where are we thin", "single point of failure", "cover depth",
        "most exposed trips", "which trips are fragile", "thinnest cover",
    ], boost=1.2),

    Intent("latent_breaches", [
        "already illegal", "anything wrong with the roster", "latent breaches",
        "any illegal assignments", "what have we missed",
    ], boost=1.2),

    Intent("morning_brief", [
        "standing morning briefing", "what should the briefing show",
        "which three data points", "morning brief", "what to surface daily",
        "start of day board", "what should the desk look at",
    ], boost=1.4),

    Intent("reserve_gaps", [
        "reserve coverage gaps", "standby holes", "hours with no standby",
        "when do we have no reserve",
    ], boost=1.2),

    Intent("duty_hours", [
        "how many duty hours", "duty clock", "headroom", "hours left this week",
        "flight hours", "block hours", "how close to the limit",
    ]),

    Intent("get_reserves", [
        "who is on reserve", "standby crew", "reserve list", "who is on standby",
        # "on call" reduced to the single token {call}, which any one-token cue
        # does: a lone word scores a perfect 1.0 overlap ratio and outruns a
        # precise multi-word cue. So "Captain C-1042 just called in sick" --
        # the brief's own flagship Tier 2 question -- matched a RESERVE LISTING
        # on the word "called", and got one, confidently.
        # The cue says what it meant all along.
        "who is on call today", "on call window", "which reserves are on call",
    ]),

    Intent("get_certifications", [
        "certificates expiring", "licence expiry", "medical validity",
        "whose certificates expire", "expiring within days",
        # "certification" does not stem to "certificate", so the plural form
        # controllers actually type matched nothing and the question fell
        # through to a flight listing on the strength of "which" plus a date.
        "which certification expire", "certification expiring between",
        "which licence lapse before", "medical class expiring",
        "whose recurrent training runs out",
        # The brief's phrasing. This scored 0.80 against find_crew's 1.00 and
        # the answer came back as all 150 crew -- which reads like a list of
        # people about to lapse and is the opposite of one.
        "licence expires", "licence expires in the next",
        "crew whose licence", "certificates expire in the next",
        "expiring in the next days", "who is about to lapse",
    ], boost=1.35),

    Intent("get_roster", [
        "who is assigned", "roster for", "crew on the pairing",
        "show the pairings", "who is on this trip", "crew complement",
    ]),

    Intent("find_flights", [
        "which flights depart", "which flights operate", "flights departing", "flights arriving", "schedule",
        "how many flights", "longest block time", "which destinations",
        "what legs are there", "how many seats",
    ]),

    Intent("find_crew", [
        "who are the captains", "list the captains", "list the crew",
        "crew based at", "which crew are rated", "how many first officers",
        "show me the crew at", "which captains are based",
        # NOT "give me captains": two tokens, both common, so it scored a
        # perfect 1.0 on any sentence containing "give" and "captain" --
        # including "both captains are sick, give the joint plan".
        "i need a list of pilots", "find me crew who are",
        "what is their base and rating", "what is their rank",
        "what rating do they hold", "what base are they at",
        "how many captains are there",
    ]),

    Intent("get_risk_signals", [
        "risk score", "disruption risk", "who is highest risk",
        "standing morning briefing", "what should the briefing surface",
        "which data points should the desk watch",
    ]),

    Intent("compute_duty_period", [
        "earliest next report", "when can they report next",
        "minimum rest before they report", "how much rest do they need",
    ]),

    Intent("get_rulebook", [
        "what does the rule say", "what is the limit", "show me the rules",
        "what are the callout rates", "how much does a callout cost",
        "what is the rulebook", "show the rule text", "what is the limit for",
    ]),
]

# Which tool parameters consume which extracted entity. Adding a tool means
# adding a row, never editing the scorer.
ENTITY_AFFINITY = {
    "crew":     ("crew_id",),
    "pairing":  ("pairing_id", "exclude_pairing"),
    "flight":   ("flight_no", "flight_id"),
    "aircraft": ("aircraft",),
    "station":  ("station", "base", "dep_station", "arr_station"),
    "date":     ("date", "end_date", "valid_on", "start_utc"),
    "duration": ("delay_hours",),
    "rank":     ("rank", "role"),
    "rating":   ("rating", "aircraft_type"),
}
# Tools that return a VERDICT about one named person, versus tools that return
# a SET. "can C-3305 cover P-2291" and "who can cover P-2291" share almost
# every word and want opposite answers.
VERDICT_TOOLS = {"check_legality", "simulate_cert_expiry", "minimal_repair"}
LIST_TOOLS = {"find_crew", "find_flights", "get_reserves", "get_roster",
              "get_certifications", "cover_fragility"}

AFFINITY_WEIGHT = {
    "pairing": 0.30, "crew": 0.22, "duration": 0.30, "flight": 0.18,
    "aircraft": 0.16, "rank": 0.20, "rating": 0.14, "station": 0.12,
    "date": 0.06,
}

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
    present = {
        "crew": bool(ents.crew), "pairing": bool(ents.pairings),
        "flight": bool(ents.flights), "aircraft": bool(ents.aircraft),
        "station": bool(ents.stations), "date": bool(ents.dates),
        "time": bool(ents.times), "duration": ents.hours is not None,
        "rank": bool(ents.ranks), "rating": bool(ents.rating),
    }
    have = {
        "crew": bool(ents.crew), "pairing": bool(ents.pairings),
        "station": bool(ents.stations), "duration": ents.hours is not None,
        # `aircraft` belongs here. bind_args already resolves a tail number to
        # the trip it flies, so "the VT-DXF First Officer on 20 Sep" DOES name
        # a subject -- but this set disagreed, so rank_cover_options failed its
        # needs check, took the 0.25 penalty, and lost to a plain reserve list
        # on the two most valuable questions in the set.
        "pairing_or_crew": bool(ents.pairings or ents.crew or ents.flights
                                or ents.aircraft),
    }
    scored: list[tuple[float, Intent]] = []
    blocked: dict[str, tuple[float, list[str]]] = {}
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
        # Parameter affinity, generalised. Word overlap alone produces ties
        # that were being broken by position in this list -- so "who is on
        # P-2291" went to get_reserves purely because it is declared earlier.
        # Preferring the tool that can actually CONSUME what the question named
        # is a real signal, and it is declarative rather than another branch.
        params = REGISTRY[intent.tool].params
        for kind, param_names in ENTITY_AFFINITY.items():
            if not present.get(kind):
                continue
            if any(pn in params for pn in param_names):
                s += AFFINITY_WEIGHT.get(kind, 0.10)

        # Question SHAPE, not just vocabulary. A yes/no about a named person is
        # a verdict question; a wh-question about a trip is a list question.
        if ents.yes_no and ents.crew and intent.tool in VERDICT_TOOLS:
            s += 0.28
        if ents.wh and intent.tool in VERDICT_TOOLS:
            s -= 0.18
        if ents.yes_no and intent.tool in LIST_TOOLS:
            s -= 0.12
        # Two clock times plus a station is a closure window, near enough
        # always. Nothing else in the vocabulary carries that signature.
        if len(ents.times) >= 2 and ents.stations and intent.tool == \
                "simulate_station_closure":
            s += 0.45
        if ents.min_hours is not None and "min_hours" in params:
            s += 0.30
        missing = [n for n in intent.needs if not have.get(n, False)]
        if missing:
            s *= 0.25          # plausible, but we lack what it needs
        scored.append((s, intent))
        blocked[intent.tool] = (s / 0.25 if missing else s, missing)
    scored.sort(key=lambda x: -x[0])
    _BLOCKED.clear()
    _BLOCKED.update(blocked)
    return scored


# The pre-penalty score of every intent from the most recent scoring pass,
# with whatever it was missing. Read by `starved_leader` below.
_BLOCKED: dict[str, tuple[float, list[str]]] = {}

_NEED_LABEL = {
    "crew": "which crew member",
    "pairing": "which trip",
    "station": "which station",
    "duration": "how long the delay is",
    "pairing_or_crew": "which crew member or trip",
}


def starved_leader(ranked: list[tuple[float, Intent]], ents: Entities
                   ) -> tuple[str, list[str]] | None:
    """The capability that WOULD have led if it had its arguments.

    "my BLR captain on the DEL overnight is out -- options?" is unmistakably a
    cover question, but the referring expression resolves to nobody, so
    rank_cover_options takes the needs penalty and a plain crew listing wins
    the round. The desk then gets 28 captains at BLR presented as an answer:
    fluent, instant, and about a different question.

    Falling through like that is worse than refusing, because a refusal tells
    the controller what to type next and a confident wrong list does not. So
    when the starved intent would have beaten the actual winner outright, we
    surface what it needed instead of serving the runner-up.

    The subject test is what keeps this from firing on perfectly good
    questions. "If DX404 on 16 Sep is cancelled, how many passengers are
    affected?" makes simulate_crew_unavailable score high on the word
    "affected" while missing a crew id -- but the sentence DID name a subject,
    DX404, and find_flights consumes it and answers correctly. A question that
    names a crew member, a trip, a flight or a tail has given us something to
    act on, and the ranking is allowed to stand. Only a question that named
    nothing at all gets turned back.
    """
    if not ranked:
        return None
    if ents.crew or ents.pairings or ents.flights or ents.aircraft:
        return None
    top_score = ranked[0][0]
    best: tuple[float, str, list[str]] | None = None
    for tool, (raw, missing) in _BLOCKED.items():
        if not missing or raw < CONFIDENT or raw <= top_score:
            continue
        if best is None or raw > best[0]:
            best = (raw, tool, missing)
    if not best:
        return None
    return best[1], [_NEED_LABEL.get(m, m) for m in best[2]]


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

    # Relative expressions a controller actually types. Unbound, these were
    # not refusals -- they were answers to a wider question than the one asked:
    # "licence expires in the next 30 days" returned all 600 certificates, and
    # "depart DEL this afternoon" returned the whole week. Both read as
    # answers. The brief's own Tier 1 examples are phrased exactly this way.
    ql = question.lower()
    today = snap.horizon[0]

    m = re.search(r"next\s+(\d{1,3})\s*(day|week|month)", ql)
    if m:
        n_units = int(m.group(1))
        days = n_units * {"day": 1, "week": 7, "month": 30}[m.group(2)]
        put("expiring_before", (today + timedelta(days=days)).isoformat())
        put("expiring_after", today.isoformat())

    for phrase, (after, before) in (
            ("this morning", ("00:00", "12:00")),
            ("this afternoon", ("12:00", "18:00")),
            ("this evening", ("18:00", "23:59")),
            ("tonight", ("18:00", "23:59")),
            ("overnight", ("18:00", "23:59"))):
        if phrase in ql:
            put("dep_after_utc", after)
            put("dep_before_utc", before)
            # "this afternoon" means today, not every afternoon this week.
            put("date", today.isoformat())
            break

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
    if not pairing and ents.aircraft:
        # "the VT-DXE captain on 16 Sep" names a TAIL, not a trip. Crew are
        # assigned to pairings, so resolve the aircraft (plus a date when one
        # is given) to the trip it flies -- otherwise every aircraft-shaped
        # question falls through to a generic lookup.
        ps = [p for p in snap.pairings.values() if p.aircraft == ents.aircraft[0]]
        if ents.dates:
            ps = [p for p in ps
                  if any(d.date == ents.dates[0] for d in p.days)] or ps
        if ps:
            pairing = sorted(ps, key=lambda p: p.days[0].date)[0].pairing_id

    # the person being replaced, when the question named a role but no id
    if pairing and not crew and ents.rank:
        crew = snap.pairings[pairing].crew_in_role(ents.rank)

    put("crew_id", crew)
    put("pairing_id", pairing)
    put("aircraft", ents.aircraft[0] if ents.aircraft else None)
    put("flight_no", ents.flights[0] if ents.flights else None)
    put("rank", ents.rank)          # only when the phrase named exactly one
    put("role", ents.rank)
    if len(ents.ranks) > 1:
        put("ranks", list(ents.ranks))   # a class: "pilots" is two ranks
    put("rating", ents.rating)
    put("delay_hours", ents.hours)
    if ents.limit is not None:
        put("top_n", ents.limit)
        put("limit", ents.limit)

    if ents.dates:
        put("date", ents.dates[0])
        put("end_date", ents.dates[0])
    if "end_date" in params and "end_date" not in a:
        a["end_date"] = snap.snapshot_utc.date().isoformat()
    if tool == "get_reserves" and "date" not in a:
        # No silent default. Saying "on call" without a date is genuinely
        # ambiguous; the validator turns the missing required arg into a
        # clarifying question instead of quietly answering about tomorrow.
        pass

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

    if tool == "simulate_station_closure" and len(ents.times) >= 2:
        # The brief's own example is "Station BLR is closed 14:00-20:00 --
        # what's the crew impact?" -- a station, a window, and no date. That
        # refused for want of start_utc, which is a fair thing to ask for and
        # the wrong thing to ask when the window is right there in the
        # sentence. An undated closure means today, the same reading
        # "this afternoon" gets a few lines above.
        day = ents.dates[0] if ents.dates else today.isoformat()
        a["start_utc"] = f"{day}T{ents.times[0]}:00Z"
        a["end_utc"] = f"{day}T{ents.times[1]}:00Z"
    if tool == "compute_duty_period" and ents.dates and ents.times:
        a["release_utc"] = f"{ents.dates[0]}T{ents.times[0]}:00Z"
    if tool == "find_flights":
        ql2 = question.lower()
        if re.search(r"most seats|largest aircraft|seats at risk", ql2):
            a["aggregate"] = "max_seats"
        elif re.search(r"cancel", ql2) and re.search(
                r"how many passenger|passengers.*(affect|cost)|what.*cost", ql2):
            a["aggregate"] = "cancel_cost"
        elif re.search(r"\bhow many\b", ql2):
            a["aggregate"] = "count"
    if tool == "duty_hours" and re.search(r"\b(flight|block)\s*hour", question,
                                          re.I):
        a["metric"] = "flight"
        a["window_days"] = 28
    if tool == "get_certifications" and ents.dates:
        # Two dates in the sentence is an explicit window ("between the 14th
        # and March"); one date plus "before/by" is an open-ended deadline;
        # one bare date defaults to the next 30 days, which is what a
        # controller means by "expiring soon".
        if len(ents.dates) >= 2:
            a["expiring_after"], a["expiring_before"] = sorted(ents.dates)[:2]
        elif re.search(r"\b(before|by|prior to|ahead of|until)\b", question,
                       re.I):
            a["expiring_after"] = snap.horizon[0].isoformat()
            a["expiring_before"] = ents.dates[0]
        else:
            a["expiring_after"] = ents.dates[0]
            a["expiring_before"] = (_date.fromisoformat(ents.dates[0])
                                    + timedelta(days=30)).isoformat()
    if tool == "draft_notification":
        low = question.lower()
        a["audience"] = ("occ" if "occ" in low else
                         "duty_manager" if "manager" in low else "crew")
    if tool == "duty_hours" and ents.min_hours is not None:
        a["min_hours"] = ents.min_hours
        a.pop("crew_id", None)      # a threshold question is about everyone

    if tool == "solve_joint_cover":
        # Two tails named in one sentence is the simultaneous-disruption case.
        # Without this the tool was unreachable from the index: its only
        # required argument is a list that nothing ever built.
        evs = []
        for tail in ents.aircraft:
            ps = [p for p in snap.pairings.values() if p.aircraft == tail]
            if ents.dates:
                ps = [p for p in ps
                      if any(d.date == ents.dates[0] for d in p.days)] or ps
            if not ps:
                continue
            pid = sorted(ps, key=lambda p: p.days[0].date)[0].pairing_id
            who = snap.pairings[pid].crew_in_role(ents.rank or "Captain")
            if who:
                evs.append({"crew_id": who, "pairing_id": pid})
        for cid in ents.crew:
            for p in snap.pairings_for_crew(cid):
                evs.append({"crew_id": cid, "pairing_id": p.pairing_id})
                break
        seen, uniq = set(), []
        for ev in evs:
            if ev["pairing_id"] not in seen:
                seen.add(ev["pairing_id"])
                uniq.append(ev)
        if len(uniq) >= 2:
            a["events"] = uniq
    return a


# ==========================================================================
# planners
# ==========================================================================


# Above CONFIDENT the index is trusted outright: it is instant and
# reproducible, which matters on stage. Between FLOOR and CONFIDENT it is only
# a fallback for when no model is configured. Below FLOOR there is no answer --
# we refuse rather than serve the nearest plausible tool, which is exactly how
# "I need 5 pilots" used to come back as somebody else's question.
CONFIDENT = 0.75
FLOOR = 0.40


DOMINANT = 0.22   # how far ahead the top intent must be to own the question


def plan_from_index(question: str, ents: Entities, snap: Snapshot,
                    min_score: float = FLOOR) -> Plan | None:
    """Deterministic single-step plan. Fast, reproducible, no network.

    When the best-scoring capability cannot bind its required arguments we do
    NOT quietly try the next one down. That fallthrough is precisely how "which
    flights are affected by the HYD closure" became a plain flight listing: the
    closure simulator was the top match by a mile, lacked a time window, and the
    runner-up answered a different question with total confidence.

    So if the leader is DOMINANT-ly ahead and cannot be bound, we raise the
    missing argument as a clarifying question instead. Only genuinely close
    calls fall through to the next candidate.
    """
    all_ranked = score_intents(question, ents)
    starved = starved_leader(all_ranked, ents)
    if starved:
        tool, missing = starved
        raise PlanError(
            f"I need to know {', '.join(missing)} before I can answer that.",
            have=f"this reads as a {tool.replace('_', ' ')} question",
            would_need=", ".join(missing))

    ranked = all_ranked[:4]
    for i, (s, intent) in enumerate(ranked):
        if s < min_score:
            break
        args = bind_args(intent.tool, ents, snap, question)
        missing = _unbound(intent.tool, args)
        if missing:
            runner_up = ranked[i + 1][0] if i + 1 < len(ranked) else 0.0
            if s - runner_up >= DOMINANT:
                raise PlanError(
                    f"I need {', '.join(missing)} to answer that.",
                    have=f"this looks like a {intent.tool.replace('_', ' ')} "
                         f"question",
                    would_need=", ".join(missing))
            continue

        second = _conjunct(question, ranked, i, ents, snap)
        if second:
            s2, tool2, args2 = second
            return Plan(
                [Step(tool2, args2, "second half of the question"),
                 Step(intent.tool, args, intent.examples[0])],
                "index",
                f"two-part question: {tool2} ({s2:.2f}) + {intent.tool} "
                f"({s:.2f})",
                conjunctive=True)

        return Plan([Step(intent.tool, args, intent.examples[0])], "index",
                    f"matched '{intent.examples[0]}' (score {s:.2f})")
    return None


# A sentence that asks two things joined by "and" / a comma, e.g.
# "What is C-2087's rank, AND total flight hours over the 28 days ending ...".
_CONJ = re.compile(r",\s*(?:and|as well as)\b|\band\b\s+(?:also\s+)?(?:their|"
                   r"his|her|the|total|what)\b", re.I)


def _conjunct(question: str, ranked: list[tuple[float, Intent]], i: int,
              ents: Entities, snap: Snapshot
              ) -> tuple[float, str, dict[str, Any]] | None:
    """The runner-up when the sentence genuinely asked for two things.

    Deliberately narrow. Answering half a question and sounding certain is the
    failure mode this system exists to avoid, but so is chaining tools that
    were never asked for -- so all four conditions must hold:

      * the sentence carries an explicit conjunction, not just a list comma;
      * both capabilities are plain retrieval (never chain two simulations,
        which would compose two hypotheticals into one unlabelled answer);
      * the runner-up is CONFIDENT on its own, not merely next in line;
      * the two are within DOMINANT of each other -- a clear leader means the
        question had one subject and the runner-up is noise.
    """
    if not _CONJ.search(question):
        return None
    lead_score, lead = ranked[i]
    if REGISTRY[lead.tool].kind != "retrieval":
        return None
    for s2, cand in ranked[i + 1:]:
        if cand.tool == lead.tool or s2 < CONFIDENT:
            continue
        if lead_score - s2 >= DOMINANT:
            break
        if REGISTRY[cand.tool].kind != "retrieval":
            continue
        args2 = bind_args(cand.tool, ents, snap, question)
        if _unbound(cand.tool, args2):
            continue
        return s2, cand.tool, args2
    return None


def _unbound(tool: str, args: dict[str, Any]) -> list[str]:
    """Required arguments the binder could not supply."""
    t = REGISTRY[tool]
    missing = [r for r in t.required if r not in args]
    for group in getattr(t, "requires_one_of", ()) or ():
        if not any(g in args for g in group):
            missing.append(" or ".join(group))
    return missing


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
    # Models sometimes emit the same call twice ("find_crew -> find_crew").
    # A duplicate step costs time and tells the controller nothing.
    dedup: list[Step] = []
    for st in steps:
        if dedup and dedup[-1].tool == st.tool and dedup[-1].args == st.args:
            continue
        dedup.append(st)
    steps = dedup
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
        missing = _unbound(st.tool, st.args)
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

    # A conjunctive plan has no winning step -- both halves were asked for, so
    # both must survive into the answer. `final` is still the leader's payload
    # (every renderer keys off plan.tool), with the other half carried beside
    # it rather than discarded.
    if plan.conjunctive and len(results) > 1 and isinstance(results[-1], dict):
        merged = dict(results[-1])
        merged["also"] = [{"tool": st.tool, "result": r}
                          for st, r in zip(plan.steps[:-1], results[:-1])]
        results[-1] = merged
    return Execution(results, ledger)
