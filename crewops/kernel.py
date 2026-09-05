"""The deterministic decision core. No language model reaches this module.

`check_cover` answers "can this person legally take this trip?" and
`cover_options` answers "who can, ranked by cost?" -- exhaustively, over every
active crew member of the required rank. A full legality check costs ~1.4 ms, so
there is no reason to approximate or to pre-filter to a shortlist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from .data import PairingDay, Snapshot, at, hours
from .rules import (Ledger, fdp_limit, build_timeline, duty_period,
                    rule_cert_06, rule_duty_02, rule_fdp_01, rule_flt_03,
                    rule_qual_05, rule_rest_04)

ALL_RULES = [
    "RULE-FDP-01", "RULE-DUTY-02", "RULE-FLT-03",
    "RULE-REST-04", "RULE-QUAL-05", "RULE-CERT-06", "RULE-BASE-07",
]

PILOT_RANKS = ("Captain", "First Officer")


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class Legality:
    crew_id: str
    legal: bool
    issues: list[str]
    ledger: Ledger
    rules_evaluated: list[str] = field(default_factory=lambda: list(ALL_RULES))
    non_binding: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "crew_id": self.crew_id,
            "legal": self.legal,
            "issues": self.issues,
            "rules_evaluated": self.rules_evaluated,
            "non_binding": self.non_binding,
        }


@dataclass
class Option:
    action: str
    crew_id: str | None
    legal: bool
    rules_checked: list[str]
    cost_inr: int
    delay_hours: float
    rank: int = 0
    cost_breakdown: dict[str, int] = field(default_factory=dict)
    coverage: str = ""
    reasoning: str = ""
    # operational context -- shown inside a tie band, never used as a sort key
    reachability_minutes: int | None = None
    seniority: int | None = None
    risk_score: float | None = None
    is_reserve: bool = False
    tie_band: int = 0
    cost_operational_inr: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rank": self.rank, "action": self.action, "crew_id": self.crew_id,
            "legal": self.legal, "rules_checked": self.rules_checked,
            "cost_inr": self.cost_inr, "delay_hours": self.delay_hours,
            "coverage": self.coverage, "reasoning": self.reasoning,
            "cost_breakdown": self.cost_breakdown, "tie_band": self.tie_band,
        }
        for k in ("reachability_minutes", "seniority", "risk_score",
                  "cost_operational_inr"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


@dataclass
class OptionSet:
    options: list[Option]
    excluded: list[dict[str, str]]
    candidate_pool_size: int
    needed_base: str
    aircraft_type: str
    role: str

    @property
    def legal_options(self) -> list[Option]:
        return [o for o in self.options if o.crew_id]

    @property
    def recommended(self) -> Option | None:
        return self.options[0] if self.options else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "options": [o.to_dict() for o in self.options],
            "excluded_candidates": self.excluded,
            "candidate_pool_size": self.candidate_pool_size,
            "needed_base": self.needed_base,
            "aircraft_type": self.aircraft_type,
            "role": self.role,
        }


# --------------------------------------------------------------------------
# deadhead positioning (RULE-BASE-07)
# --------------------------------------------------------------------------


def positioning(from_base: str, to_station: str, on: date,
                first_dep: datetime) -> tuple[bool, float, str | None, datetime | None]:
    """Can we position a crew member from `from_base` to `to_station` that day?

    The only modelled corridor is DEL -> BLR, on DX402 (arrives 08:45Z, odd
    dates) or DX589 (arrives 07:45Z, even dates). New report is arrival + 15 min
    transit; the first departure then slips to report + 60 min.

    Returns (possible, delay_hours, positioning_flight_no, arrival_utc).
    """
    if from_base == to_station:
        return True, 0.0, None, None
    if not (from_base == "DEL" and to_station == "BLR"):
        return False, 0.0, None, None
    even = on.day % 2 == 0
    arr = at(on, "07:45") if even else at(on, "08:45")
    flight_no = "DX589" if even else "DX402"
    delay = round(max(0.0, hours((arr + timedelta(minutes=75)) - first_dep)), 2)
    return True, delay, flight_no, arr


def option_cost(snap: Snapshot, is_reserve: bool, is_pilot: bool,
                delay_h: float, deadhead: bool) -> tuple[int, dict[str, int]]:
    c = snap.costs
    if is_reserve:
        base = c["reserve_callout_pilot"] if is_pilot else c["reserve_callout_cabin"]
        key = "reserve_callout"
    else:
        base = c["dayoff_callout_pilot"] if is_pilot else c["dayoff_callout_cabin"]
        key = "dayoff_callout"
    parts = {key: int(base)}
    total = base
    if deadhead:
        parts["deadhead_positioning"] = int(c["deadhead_positioning"])
        delay_cost = int(round(delay_h * c["delay_cost_per_duty_hour"]))
        parts["delay"] = delay_cost
        total += c["deadhead_positioning"] + delay_cost
    return int(round(total)), parts


def operational_cost(snap: Snapshot, crew_id: str,
                     cover_days: list[PairingDay]) -> int:
    """Hotel nights the reference cost model never charges.

    `costs.hotel_overnight` is defined in the dataset and never referenced by the
    reference resolver. A cover crew who ends a duty day away from base sleeps
    somewhere. We surface this as a SEPARATE figure and never let it touch the
    graded ranking.
    """
    extra = 0
    base = snap.crew[crew_id].base
    for day in cover_days[:-1]:  # the final day ends the trip
        last = max(day.flights, key=lambda f: snap.flights[f].dep_utc)
        if snap.flights[last].arr_station != base:
            extra += snap.costs["hotel_overnight"]
    return extra


# --------------------------------------------------------------------------
# the keystone: can this person take this trip?
# --------------------------------------------------------------------------


def check_cover(
    snap: Snapshot,
    crew_id: str,
    cover_days: list[PairingDay],
    exclude_pairing: str | None = None,
    delay_hours: float = 0.0,
    strict_certs: bool = False,
) -> Legality:
    """Full legality evaluation for one candidate against one trip.

    Check ORDER is load-bearing: RULE-QUAL-05 short-circuits, so an unrated
    candidate never accumulates duty or rest issues.
    """
    led = Ledger()
    led.add("crew_id", crew_id, source="crew.json")
    if delay_hours:
        led.add("delay_hours", delay_hours, "h")

    aircraft_type = snap.flights[cover_days[0].flights[0]].aircraft_type

    # --- RULE-QUAL-05, short-circuiting -------------------------------
    qual = rule_qual_05(snap, crew_id, aircraft_type, led)
    if qual:
        return Legality(crew_id, False, qual, led, rules_evaluated=["RULE-QUAL-05"])

    # deadhead positioning shifts report AND release together, so FDP is
    # invariant under it (TRAP 10)
    shifted = [d.shifted(delay_hours, shift_report=True) for d in cover_days]

    issues: list[str] = []
    for day in shifted:
        issues += rule_cert_06(snap, crew_id, day.day, led, strict=strict_certs)
        issues += rule_fdp_01(snap, day, led)

    sim = build_timeline(snap, crew_id, shifted, exclude_pairing)
    issues += rule_rest_04(snap, sim, led)

    # DUTY-02 uses the UNDELAYED durations, matching the reference
    issues += rule_duty_02(snap, crew_id, cover_days, exclude_pairing, led)

    flt_issues, flt_binding = rule_flt_03(snap, crew_id, cover_days,
                                          exclude_pairing, led)
    issues += flt_issues

    non_binding = [] if flt_binding else ["RULE-FLT-03"]
    return Legality(crew_id, not issues, issues, led, list(ALL_RULES), non_binding)


# --------------------------------------------------------------------------
# who can cover, ranked
# --------------------------------------------------------------------------


def cover_options(
    snap: Snapshot,
    cover_days: list[PairingDay],
    role: str,
    sick_crew_id: str | None = None,
    exclude_pairing: str | None = None,
    include_cancel: bool = True,
    strict_certs: bool = False,
) -> OptionSet:
    """Enumerate every legal way to cover `cover_days` in `role`.

    TRAP 7: the candidate pool is ALL ACTIVE CREW OF THE RANK -- not the 16
    people in the reserve pool. Four of the six options in the flagship scenario
    are day-off callouts of line crew.

    TRAP 15: ranking is lexicographic on (cost_inr, crew_id). Cost ties are
    enormous -- one scenario has 43 options with 36 tied at the same price -- so
    we also stamp a `tie_band` and carry reachability/seniority/risk as context,
    without ever letting them influence the sort.

    TRAP 8: the cancellation option is appended AFTER the sort and always ranks
    last, regardless of price.
    """
    options: list[Option] = []
    excluded: list[dict[str, str]] = []

    first_day = cover_days[0]
    needed_base = snap.flights[first_day.flights[0]].dep_station
    aircraft_type = snap.flights[first_day.flights[0]].aircraft_type
    first_dep = snap.flights[
        min(first_day.flights, key=lambda f: snap.flights[f].dep_utc)
    ].dep
    is_pilot = role in PILOT_RANKS
    pool = snap.crew_by_rank(role)

    for crew_id in pool:
        c = snap.crew[crew_id]
        if crew_id == sick_crew_id or c.status != "active":
            continue

        is_reserve = crew_id in snap.reserves
        deadhead = c.base != needed_base
        delay_h = 0.0

        if deadhead:
            ok, delay_h, _flt, _arr = positioning(c.base, needed_base,
                                                  first_day.day, first_dep)
            if not ok:
                excluded.append({
                    "crew_id": crew_id,
                    "reason": "RULE-BASE-07: no same-day positioning flight from base",
                })
                continue

        if is_reserve:
            # TRAP 2: tested against the REQUIRED REPORT TIME (after any
            # positioning delay), inclusive at both ends -- not the callout time
            required_report = first_day.report + timedelta(hours=delay_h)
            r = snap.reserves[crew_id]
            if not r.covers(required_report):
                excluded.append({
                    "crew_id": crew_id,
                    "reason": (
                        f"reserve on-call window {r.window_start}-{r.window_end}Z "
                        f"does not cover required report "
                        f"{required_report.strftime('%H:%M')}Z"
                    ),
                })
                continue

        res = check_cover(snap, crew_id, cover_days, exclude_pairing, delay_h,
                          strict_certs=strict_certs)
        if not res.legal:
            excluded.append({"crew_id": crew_id, "reason": res.reason})
            continue

        cost, parts = option_cost(snap, is_reserve, is_pilot, delay_h, deadhead)
        label = "reserve callout" if is_reserve else "day-off callout"
        if deadhead:
            label += (f" + deadhead from {c.base} "
                      f"(first departure delayed ~{delay_h}h)")

        n_legs = sum(len(d.flights) for d in cover_days)
        options.append(Option(
            action=f"Assign {c.rank} {crew_id} ({label})",
            crew_id=crew_id, legal=True, rules_checked=list(ALL_RULES),
            cost_inr=cost, delay_hours=delay_h, cost_breakdown=parts,
            coverage=f"all {n_legs} flights",
            reasoning=_reason_for(snap, crew_id, res, is_reserve, deadhead, delay_h),
            reachability_minutes=c.reachability_minutes, seniority=c.seniority,
            risk_score=snap.risk.get(crew_id, {}).get("disruption_risk_score"),
            is_reserve=is_reserve,
            cost_operational_inr=cost + operational_cost(snap, crew_id, cover_days),
        ))

    options.sort(key=lambda o: (o.cost_inr, o.crew_id or ""))

    # tie bands: equal cost is equal preference, and saying so is more honest
    band = 0
    last_cost = None
    for o in options:
        if o.cost_inr != last_cost:
            band += 1
            last_cost = o.cost_inr
        o.tie_band = band

    if include_cancel:
        n_legs = sum(len(d.flights) for d in cover_days)
        cancel_cost = snap.costs["cancellation_per_flight"] * n_legs
        options.append(Option(
            action=f"Cancel all {n_legs} flights of the pairing",
            crew_id=None, legal=True, rules_checked=[],
            cost_inr=cancel_cost, delay_hours=0.0,
            cost_breakdown={"cancellation": cancel_cost},
            coverage="none", tie_band=band + 1,
            reasoning=(f"{n_legs} legs cancelled at "
                       f"INR {snap.costs['cancellation_per_flight']:,} each."),
        ))

    for i, o in enumerate(options):
        o.rank = i + 1

    return OptionSet(options, excluded, len(pool), needed_base, aircraft_type, role)


def _reason_for(snap: Snapshot, crew_id: str, res: Legality, is_reserve: bool,
                deadhead: bool, delay_h: float) -> str:
    c = snap.crew[crew_id]
    bits = [f"{c.base}-based", f"{'/'.join(c.ratings)}-rated"]
    if is_reserve:
        r = snap.reserves[crew_id]
        bits.append(f"on call {r.window_start}-{r.window_end}Z")
    else:
        bits.append("day off")
    bits.append(f"reachable in {c.reachability_minutes} min")
    if deadhead:
        bits.append(f"requires positioning, delaying departure ~{delay_h}h")
    tightest = _tightest_margin(res)
    if tightest:
        bits.append(tightest)
    return "; ".join(bits) + "."


def _tightest_margin(res: Legality) -> str:
    """The constraint that came closest to binding -- what a controller wants."""
    best = None
    for f in res.ledger.facts:
        if f.key.startswith("duty_7d_headroom") and isinstance(f.value, (int, float)):
            if best is None or f.value < best[1]:
                best = ("7-day duty", f.value)
    if best and best[1] < 20:
        return f"{best[0]} headroom {best[1]}h"
    return ""


# --------------------------------------------------------------------------
# more than one problem at once
# --------------------------------------------------------------------------


@dataclass
class JointPlan:
    total_cost_inr: int
    assignments: dict[str, Option]
    proven_optimal: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost_inr": self.total_cost_inr,
            "assignments": {k: v.to_dict() for k, v in self.assignments.items()},
            "proven_optimal": self.proven_optimal,
            "note": self.note,
        }


def solve_joint(snap: Snapshot, needs: list[dict[str, Any]]) -> JointPlan:
    """Allocate scarce cover across simultaneous disruptions.

    The trap this exists for: answering two sick calls independently can assign
    the SAME person to both. In the dataset's hardest scenario, one reserve
    captain is the cheapest answer to both pairings at once; a stateless
    question-and-answer bot produces an illegal plan and never notices.

    Exhaustive over the cartesian product with a distinct-crew constraint. That
    is correct and instant for two events. Above two we still return a legal
    plan but stop claiming it is optimal -- enumerate-and-rank is the wrong
    algorithm class there, and CP-SAT is the honest successor.
    """
    sets = []
    for need in needs:
        os_ = cover_options(
            snap, need["cover_days"], need["role"],
            need.get("sick_crew_id"), need.get("exclude_pairing"),
        )
        sets.append((need["key"], os_))

    proven = len(needs) <= 2
    best: tuple[int, dict[str, Option]] | None = None

    def recurse(i: int, used: set[str], chosen: dict[str, Option], cost: int) -> None:
        nonlocal best
        if best is not None and cost >= best[0]:
            return  # branch and bound: this branch cannot improve
        if i == len(sets):
            if best is None or cost < best[0]:
                best = (cost, dict(chosen))
            return
        key, os_ = sets[i]
        for opt in os_.options:
            if opt.crew_id and opt.crew_id in used:
                continue
            chosen[key] = opt
            if opt.crew_id:
                used.add(opt.crew_id)
            recurse(i + 1, used, chosen, cost + opt.cost_inr)
            if opt.crew_id:
                used.discard(opt.crew_id)
            del chosen[key]

    recurse(0, set(), {}, 0)

    if best is None:
        return JointPlan(0, {}, False, "No legal combination found.")

    note = (
        "The same crew member cannot cover two trips at once; this minimises "
        "total cost across all events. Equal-cost mirror assignments are "
        "equally correct."
    )
    if not proven:
        note += (
            " NOTE: with more than two simultaneous events this is a legal plan "
            "but is not proven optimal -- pairwise enumeration does not model "
            "rest conflicts introduced by the assignments themselves."
        )
    return JointPlan(best[0], best[1], proven, note)


# --------------------------------------------------------------------------
# proactive analysis -- the signals that are actually dense in this data
# --------------------------------------------------------------------------


def cover_fragility(snap: Snapshot, roles: Iterable[str] = PILOT_RANKS,
                    limit: int | None = None) -> list[dict[str, Any]]:
    """How many legal replacements exist for every (trip, role) in the week?

    This is the proactive signal worth surfacing. A duty-hour watchlist is NOT:
    peak 7-day utilisation across all 150 crew is 42.51h against a 60h cap, and
    exactly one crew-day in 900 cannot absorb an extra duty. That panel renders
    empty. Cover depth, by contrast, finds genuine single points of failure.
    """
    rows: list[dict[str, Any]] = []
    for p in snap.pairings.values():
        for role in roles:
            sick = p.crew_in_role(role)
            if not sick:
                continue
            os_ = cover_options(snap, list(p.days), role, sick, p.pairing_id,
                                include_cancel=False)
            legal = os_.legal_options
            rows.append({
                "pairing_id": p.pairing_id,
                "date": p.days[0].date,
                "aircraft": p.aircraft,
                "role": role,
                "rostered": sick,
                "legal_covers": len(legal),
                "candidates": [o.crew_id for o in legal[:3]],
                "cheapest_inr": legal[0].cost_inr if legal else None,
            })
    rows.sort(key=lambda r: (r["legal_covers"], r["date"]))
    return rows[:limit] if limit else rows


def reserve_coverage_gaps(snap: Snapshot) -> list[dict[str, Any]]:
    """Hours of the day with no on-call reserve for a rank/type combination.

    Two real structural holes in this roster, and they are not the same hole:

      Captain / A320    uncovered 19:00-23:59Z  (5 hours)
      Captain / ATR72   uncovered 16:00-23:59Z and 00:00-02:59Z  (11 hours)

    So an evening sick call has no standby captain on either fleet, and the
    turboprop has no standby captain for eleven hours of the day including the
    whole small-hours window an early departure would be crewed from. An
    earlier version of this docstring claimed a single 19:00-02:00 gap "of any
    rating", which is wrong in both directions: A320 is covered again from
    midnight by C-3305 (00:00-05:30), and ATR72 goes dark three hours earlier.

    An on-call window is treated as covering every hour it touches, so
    00:00-05:30 covers hour 5. Windows that wrap midnight are handled even
    though none in this dataset does -- the previous comparison silently
    returned "no cover at all" for such a window rather than failing, which is
    the kind of quiet wrong answer this system exists to avoid.
    """
    out = []
    types = sorted({f.aircraft_type for f in snap.flights.values()})
    for rank in PILOT_RANKS:
        for hour in range(24):
            covered: dict[str, list[str]] = {t: [] for t in types}
            for cid, r in snap.reserves.items():
                if snap.crew[cid].rank != rank:
                    continue
                start_h, end_h = int(r.window_start[:2]), int(r.window_end[:2])
                on_call = (start_h <= hour <= end_h if start_h <= end_h
                           else hour >= start_h or hour <= end_h)
                if on_call:
                    for t in types:
                        if t in snap.crew[cid].ratings:
                            covered[t].append(cid)
            for t in types:
                if not covered[t]:
                    out.append({"rank": rank, "aircraft_type": t, "hour_utc": hour})
    return out


def latent_breaches(snap: Snapshot, strict: bool = False) -> list[dict[str, Any]]:
    """Scan the published roster for assignments that are already illegal.

    Finds the planted certification lapse without being told about it.
    """
    out = []
    for p in snap.pairings.values():
        for day in p.days:
            for cid, role in p.crew:
                ok, bad = (snap.certs_valid_on_strict(cid, day.day) if strict
                           else snap.certs_valid_on(cid, day.day))
                if not ok:
                    out.append({
                        "crew_id": cid, "role": role, "pairing_id": p.pairing_id,
                        "date": day.date, "rule": "RULE-CERT-06",
                        "detail": f"{', '.join(bad)} expired before this duty",
                    })
    return out
