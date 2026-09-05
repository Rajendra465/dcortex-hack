"""What the advisor decided, and why, at every point where it could have gone
another way.

The system is a network of small deciders: policy screens that can refuse
before anything runs, an entity extractor, an intent router, two competing
planners, a validator, an executor that walks a typed chain, a narrator, and a
guard that reads the narrator's prose back against the ledger. Each one either
passes control on or stops the question dead.

None of that was visible. The answer arrived with its evidence, which is the
important half, but a controller could not see that four crew were considered
and rejected before the recommendation, or that the router was 0.31 confident
and the model broke the tie, or that the number guard read every figure in the
sentence and found them all backed. A decision you cannot inspect is a decision
you have to take on faith, and this system's whole claim is that you never have
to do that.

A Trace is append-only and costs nothing when nobody reads it: recording a
stage is a dataclass and a list append, well under the noise floor of the
kernel calls it sits beside.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Every stage belongs to one of these. The UI colours by kind, so a controller
# learns the shape of a decision without reading the labels.
GATE = "gate"          # can refuse: screens, validator, number guard
DERIVE = "derive"      # extracts or computes: entities, dates, execution
CHOOSE = "choose"      # picks between alternatives: router, planner race
SPEAK = "speak"        # produces prose: narrator


@dataclass
class Option:
    """One thing a CHOOSE stage weighed. Rejected options are the interesting
    ones -- "why not someone else" is a controller's first question."""
    label: str
    score: float | None = None
    chosen: bool = False
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"label": self.label, "chosen": self.chosen}
        if self.score is not None:
            d["score"] = round(self.score, 3)
        if self.why:
            d["why"] = self.why
        return d


@dataclass
class Stage:
    name: str
    kind: str
    verdict: str                 # pass | block | picked | skipped | failed
    detail: str = ""
    ms: float = 0.0
    options: list[Option] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "kind": self.kind,
                             "verdict": self.verdict,
                             "ms": round(self.ms, 2)}
        if self.detail:
            d["detail"] = self.detail
        if self.options:
            d["options"] = [o.to_dict() for o in self.options]
        if self.meta:
            d["meta"] = self.meta
        return d


class Trace:
    """Append-only record of one question's journey through the network."""

    def __init__(self) -> None:
        self.stages: list[Stage] = []
        self._t0 = time.perf_counter()
        self._last = self._t0

    def mark(self, name: str, kind: str, verdict: str, detail: str = "",
             options: list[Option] | None = None, **meta: Any) -> Stage:
        now = time.perf_counter()
        st = Stage(name=name, kind=kind, verdict=verdict, detail=detail,
                   ms=(now - self._last) * 1000,
                   options=options or [], meta={k: v for k, v in meta.items()
                                                if v not in (None, "", [], {})})
        self._last = now
        self.stages.append(st)
        return st

    # convenience wrappers -- they read better at the call site than mark()
    def gate(self, name: str, passed: bool, detail: str = "", **meta: Any) -> Stage:
        return self.mark(name, GATE, "pass" if passed else "block", detail, **meta)

    def derive(self, name: str, detail: str = "", **meta: Any) -> Stage:
        return self.mark(name, DERIVE, "pass", detail, **meta)

    def choose(self, name: str, options: list[Option], detail: str = "",
               **meta: Any) -> Stage:
        picked = any(o.chosen for o in options)
        return self.mark(name, CHOOSE, "picked" if picked else "failed",
                         detail, options, **meta)

    def speak(self, name: str, detail: str = "", **meta: Any) -> Stage:
        return self.mark(name, SPEAK, "pass", detail, **meta)

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def counts(self) -> dict[str, int]:
        c = {"gates_passed": 0, "gates_blocked": 0,
             "options_considered": 0, "options_rejected": 0}
        for s in self.stages:
            if s.kind == GATE:
                c["gates_passed" if s.verdict == "pass" else "gates_blocked"] += 1
            c["options_considered"] += len(s.options)
            c["options_rejected"] += sum(1 for o in s.options if not o.chosen)
        return c

    def to_dict(self) -> dict[str, Any]:
        return {"stages": [s.to_dict() for s in self.stages], **self.counts()}
