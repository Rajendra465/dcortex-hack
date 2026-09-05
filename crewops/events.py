"""Disruptions as composable overlays on an immutable snapshot.

This is the architectural bet that has to be made early. The alternative -- a
switch statement over the six worked scenarios -- answers those six and nothing
else. Held-out scenarios are the SAME event templates with different arguments,
so a parameterised overlay engine answers them with no new code, and "what if I
move him instead?" falls out for free.

Five event types cover the whole space: SICK_CREW, STATION_CLOSURE, DELAY,
CERT_EXPIRY, MULTI_SICK. Anything outside them is refused by name rather than
guessed at.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any

from .data import Pairing, Snapshot, hours, parse_utc
from .kernel import check_cover, cover_options, solve_joint
from .rules import Ledger, duty_period, fdp_limit


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Event:
    """Base for every disruption overlay.

    `kw_only` is load-bearing, not style. `type` is declared first and every
    subclass field comes after it, so `SickCrew("C-1042")` bound the crew id to
    `type` and left `crew_id` empty -- an overlay that stripped nobody, applied
    silently, and produced a confident answer about an undisrupted world. There
    is no safe positional order here, so there are no positional arguments.
    """

    type: str = "EVENT"

    def describe(self) -> str:  # pragma: no cover - overridden
        return self.type


@dataclass(frozen=True, kw_only=True)
class SickCrew(Event):
    crew_id: str = ""
    pairing_id: str | None = None
    reported_utc: str | None = None
    type: str = "SICK_CREW"

    def describe(self) -> str:
        p = f" for {self.pairing_id}" if self.pairing_id else ""
        return f"{self.crew_id} unavailable{p}"


@dataclass(frozen=True, kw_only=True)
class CertExpiry(Event):
    crew_id: str = ""
    cert_type: str = "recurrent_training"
    type: str = "CERT_EXPIRY"

    def describe(self) -> str:
        return f"{self.crew_id} {self.cert_type} lapsed"


@dataclass(frozen=True, kw_only=True)
class StationClosure(Event):
    station: str = ""
    start_utc: str = ""
    end_utc: str = ""
    type: str = "STATION_CLOSURE"

    def describe(self) -> str:
        return f"{self.station} closed {self.start_utc}..{self.end_utc}"


@dataclass(frozen=True, kw_only=True)
class Delay(Event):
    delay_hours: float = 0.0
    aircraft: str | None = None
    pairing_id: str | None = None
    date: str | None = None
    type: str = "DELAY"

    def describe(self) -> str:
        who = self.aircraft or self.pairing_id
        return f"{who} delayed {self.delay_hours}h on {self.date}"


# --------------------------------------------------------------------------
# applying overlays
# --------------------------------------------------------------------------


def apply(snap: Snapshot, events: list[Event]) -> Snapshot:
    """Return a NEW snapshot with the events applied. The base is never mutated.

    Only SICK_CREW and CERT_EXPIRY change world state (they take a person off a
    trip). Closures and delays are evaluated against the plan rather than
    rewritten into it, which matches how the reference treats them.
    """
    if not events:
        return snap

    out = copy.copy(snap)
    out.pairings = dict(snap.pairings)
    out.roster = {k: list(v) for k, v in snap.roster.items()}
    out.crew = dict(snap.crew)
    changed = False

    for ev in events:
        if isinstance(ev, (SickCrew, CertExpiry)):
            cid = ev.crew_id
            if cid not in out.crew:
                continue

            # Strip them from the trip(s) they were rostered on...
            targets = ([ev.pairing_id]
                       if isinstance(ev, SickCrew) and ev.pairing_id
                       else [p.pairing_id for p in snap.pairings_for_crew(cid)])
            for pid in targets:
                p = out.pairings.get(pid)
                if not p or not p.role_of(cid):
                    continue
                out.pairings[pid] = Pairing(
                    pairing_id=p.pairing_id, aircraft=p.aircraft, days=p.days,
                    crew=tuple((c, r) for c, r in p.crew if c != cid),
                )
                out.roster[cid] = [d for d in out.roster.get(cid, [])
                                   if d.pairing_id != pid]

            # ...and mark them unavailable fleet-wide, which is the part that
            # actually matters. A reserve is rostered on nothing, so stripping
            # pairings is a no-op and they would still turn up as the cheapest
            # cover for the NEXT disruption. Only a status change takes them out
            # of the candidate pool.
            if not (isinstance(ev, SickCrew) and ev.pairing_id):
                out.crew[cid] = replace(out.crew[cid], status="unavailable")
            changed = True

    if changed:
        out.__post_init__()
    return out


# --------------------------------------------------------------------------
# impact analysis
# --------------------------------------------------------------------------


@dataclass
class Impact:
    event: dict[str, Any]
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    ledger: Ledger = field(default_factory=Ledger)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.event, "summary": self.summary, **self.data}


def analyse_sick(snap: Snapshot, crew_id: str, pairing_id: str | None = None,
                 from_date: str | None = None) -> Impact:
    """Which flights are now uncrewed, and what breaks downstream."""
    led = Ledger()
    pairings = ([snap.pairings[pairing_id]] if pairing_id
                else snap.pairings_for_crew(crew_id))
    if from_date:
        pairings = [p for p in pairings if p.days[-1].date >= from_date]
    pairings = sorted(pairings, key=lambda p: p.days[0].date)

    by_day: dict[str, list[str]] = {}
    affected = []
    seats_by_day: dict[str, int] = {}
    for p in pairings:
        affected.append({
            "pairing_id": p.pairing_id, "role": p.role_of(crew_id),
            "aircraft": p.aircraft,
            "aircraft_type": snap.flights[p.days[0].flights[0]].aircraft_type,
            "days": [d.date for d in p.days],
        })
        for d in p.days:
            by_day.setdefault(d.date, []).extend(d.flights)
            seats_by_day[d.date] = seats_by_day.get(d.date, 0) + sum(
                snap.flights[f].seats for f in d.flights)

    days = sorted(by_day)
    immediate = by_day.get(days[0], []) if days else []
    downstream = [f for d in days[1:] for f in by_day[d]]
    pax_first = seats_by_day.get(days[0], 0) if days else 0

    led.add("uncovered_flights", len(immediate) + len(downstream))
    # The summary reports the split, not just the total ("3 uncrewed today,
    # 3 more later"), so both halves have to be ledgered or the answer quotes
    # figures it cannot cite.
    led.add("uncovered_day1", len(immediate),
            derivation=f"legs on {days[0]}" if days else "no affected days")
    led.add("uncovered_later", len(downstream),
            derivation="legs on every subsequent day of the trip")
    led.add("passengers_at_risk_day1", pax_first, "seats",
            source="flights.json seats", derivation="sum of seats on day-1 legs")
    for d in days:
        led.add(f"seats[{d}]", seats_by_day[d], "seats")

    overnight = None
    if len(days) > 1 and affected:
        first = snap.pairings[affected[0]["pairing_id"]]
        last_leg = max(first.days[0].flights,
                       key=lambda f: snap.flights[f].dep_utc)
        overnight = snap.flights[last_leg].arr_station

    return Impact(
        event={"type": "SICK_CREW", "crew_id": crew_id, "pairing_id": pairing_id},
        summary=(
            f"{crew_id} is unavailable. "
            f"{len(immediate)} flight(s) uncrewed on {days[0] if days else 'n/a'}"
            + (f", {len(downstream)} more at risk on later days."
               if downstream else ".")
        ),
        data={
            "pairings_affected": affected,
            "pairing_broken": pairing_id or (affected[0]["pairing_id"]
                                             if affected else None),
            "uncovered_by_day": by_day,
            "uncrewed_flights": immediate + downstream,
            "day1": immediate,
            "day2_also_at_risk": downstream,
            "uncovered_flights_day1": immediate,
            "uncovered_flights_day2": by_day.get(days[1], []) if len(days) > 1 else [],
            "passengers_at_risk_day1": pax_first,
            "passengers_day1": pax_first,
            "passengers_affected": pax_first,
            "seats_by_day": seats_by_day,
            "overnight_station": overnight,
        },
        ledger=led,
    )


def analyse_closure(snap: Snapshot, station: str, start_utc: str,
                    end_utc: str) -> Impact:
    """Which flights a station closure hits, and whether the crew survive the delay.

    TRAP 14: the window is HALF-OPEN, [start, end). A flight is hit if it
    departs from that station inside the window, or arrives there inside it.

    The recovery convention: delay to reopening plus a 30-minute turnaround. The
    report time is NOT shifted -- the crew are already there -- so the duty
    period GROWS and may exceed the FDP limit. That is the opposite of how a
    deadhead delay behaves, and conflating the two breaks one scenario or the
    other.
    """
    led = Ledger()
    ws, we = parse_utc(start_utc), parse_utc(end_utc)
    day = ws.date().isoformat()
    led.add("closure_hours", hours(we - ws), "h")

    affected: list[str] = []
    for f in snap.flights_on(day):
        hit_dep = f.dep_station == station and ws <= f.dep < we
        hit_arr = f.arr_station == station and ws <= f.arr < we
        if hit_dep or hit_arr:
            affected.append(f.flight_id)

    per_flight = []
    for fid in affected:
        f = snap.flights[fid]
        found = snap.pairing_of_flight(fid)
        if not found:
            continue
        p, pday = found
        _fdp, rep, rel = duty_period(pday)
        anchor = f.dep if (f.dep_station == station and ws <= f.dep < we) else f.arr
        shift = hours((we + timedelta(minutes=30)) - anchor)
        new_fdp = hours((rel + timedelta(hours=shift)) - rep)
        lim = fdp_limit(pday.sectors)
        feasible = new_fdp <= lim + 1e-6
        per_flight.append({
            "flight_id": fid, "flight_no": f.flight_no,
            "pairing_id": p.pairing_id,
            "hit_on": "departure" if anchor == f.dep else "arrival",
            "min_delay_hours": round(shift, 2),
            "crew_fdp_after_delay": round(new_fdp, 2),
            "fdp_limit": lim,
            "feasible": feasible,
            "action": ("delay (crew legal)" if feasible else
                       "delay exceeds crew FDP — re-crew tail legs from "
                       "reserves or cancel"),
        })

    seats = sum(snap.flights[f].seats for f in affected)
    led.add("affected_flights", len(affected))
    led.add("seats_at_risk", seats, "seats")
    infeasible = [r for r in per_flight if not r["feasible"]]
    led.add("flights_exceeding_fdp", len(infeasible))

    return Impact(
        event={"type": "STATION_CLOSURE", "station": station,
               "window_utc": {"start": start_utc, "end": end_utc}},
        summary=(
            f"{station} closed {ws.strftime('%H:%M')}-{we.strftime('%H:%M')}Z on "
            f"{day}: {len(affected)} flight(s) affected, {seats} seats at risk; "
            f"{len(infeasible)} would push the rostered crew past their duty limit."
        ),
        data={
            "affected_flights": affected,
            "per_flight_assessment": per_flight,
            "pairings_touched": sorted({r["pairing_id"] for r in per_flight}),
            "seats_at_risk": seats,
            "note": ("Delays are measured to reopening plus a 30-minute "
                     "turnaround. Where the extended duty exceeds RULE-FDP-01, "
                     "tail legs need a reserve re-crew or cancellation."),
        },
        ledger=led,
    )


def analyse_delay(snap: Snapshot, delay_hours: float, aircraft: str | None = None,
                  pairing_id: str | None = None,
                  on_date: str | None = None) -> Impact:
    """A technical delay: report holds, release slips, so the duty period GROWS.

    Also computes the longest legal PREFIX of the day -- how many legs the
    original crew can still legally operate -- which is what turns "you have a
    breach" into "operate the first three, re-crew the last one".
    """
    led = Ledger()
    if pairing_id:
        p = snap.pairings[pairing_id]
        pday = (next(d for d in p.days if d.date == on_date) if on_date
                else p.days[0])
    else:
        cands = [(pp, dd) for pp in snap.pairings.values() for dd in pp.days
                 if pp.aircraft == aircraft and (not on_date or dd.date == on_date)]
        if not cands:
            raise LookupError(f"no pairing for aircraft {aircraft} on {on_date}")
        p, pday = cands[0]

    fdp, rep, _rel = duty_period(pday)
    lim = fdp_limit(pday.sectors)
    new_fdp = round(fdp + delay_hours, 2)
    led.add("fdp_before", fdp, "h")
    led.add("delay_hours", delay_hours, "h")
    led.add("fdp_after_delay", new_fdp, "h", derivation=f"{fdp} + {delay_hours}")
    led.add("sectors", pday.sectors)
    led.add("fdp_limit", lim, "h",
            derivation=f"13.0 - 0.5 x max(0, {pday.sectors} - 2)")
    led.add("fdp_overage", round(new_fdp - lim, 2), "h")

    breach = new_fdp > lim + 1e-6

    # Longest legal prefix: how many legs can the ORIGINAL crew still operate?
    # Report stays fixed here, exactly as it does for the breach above -- the
    # crew are already at the airport waiting out the delay. (The shipped S4
    # answer key quotes 9.5h for the three-leg option, which is only reachable
    # by shifting the report as well; that is a slip in a hand-written key. We
    # reach the same operational conclusion -- shed the last leg -- and record
    # the discrepancy in the conformance report rather than reproduce it.)
    legs = sorted(pday.flights, key=lambda f: snap.flights[f].dep_utc)
    prefix: dict[str, Any] = {}
    for k in range(len(legs), 0, -1):
        sub = legs[:k]
        new_rel = (snap.flights[sub[-1]].arr + timedelta(hours=delay_hours)
                   + timedelta(minutes=30))
        pf = hours(new_rel - rep)
        pl = fdp_limit(k)
        if pf <= pl + 1e-6:
            prefix = {"legs": sub, "fdp": round(pf, 2), "limit": pl,
                      "legs_to_shed": legs[k:]}
            break

    # RULE-REST-04 downstream. The delay pushes RELEASE later, which eats the
    # rest before each crew member's next report. FDP-01 alone cannot see this:
    # a duty can sit comfortably inside its own limit and still leave the crew
    # short of rest for tomorrow. On a two-day pairing the overnight is the
    # tightest gap in the whole roster, so that is where it bites first.
    min_rest = 12.0
    new_release = _rel + timedelta(hours=delay_hours)
    led.add("min_rest_hours", min_rest, "h", source="rules.json RULE-REST-04")
    rest_breaches: list[dict[str, Any]] = []
    for cid, role in p.crew:
        nxt = None
        for pp in snap.pairings.values():
            if cid not in [c[0] for c in pp.crew]:
                continue
            for dd in pp.days:
                if dd.report > _rel and (nxt is None or dd.report < nxt[1].report):
                    nxt = (pp, dd)
        if nxt is None:
            continue          # nothing after this inside the roster horizon
        npp, ndd = nxt
        rest = round(hours(ndd.report - new_release), 2)
        if rest < min_rest - 1e-6:
            led.add(f"rest_hours[{cid}]", rest, "h",
                    derivation=f"{ndd.report:%Y-%m-%d %H:%M} report - "
                               f"{new_release:%Y-%m-%d %H:%M} delayed release")
            rest_breaches.append({
                "crew_id": cid, "role": role,
                "next_pairing": npp.pairing_id, "next_date": ndd.date,
                "rest_hours": rest,
                "shortfall_hours": round(min_rest - rest, 2),
                "rule": "RULE-REST-04",
            })
    rest_breach = bool(rest_breaches)
    if rest_breach:
        led.add("rest_breach_count", len(rest_breaches))

    rest_detail = ""
    if rest_breach:
        who = ", ".join(b["crew_id"] for b in rest_breaches)
        w = rest_breaches[0]
        rest_detail = (
            f"RULE-REST-04: the delayed release leaves {w['rest_hours']}h before "
            f"{w['next_pairing']} reports on {w['next_date']} - {w['shortfall_hours']}h "
            f"short of the {min_rest}h minimum. Affects {len(rest_breaches)} crew: {who}."
        )

    return Impact(
        event={"type": "DELAY", "aircraft": p.aircraft,
               "pairing_id": p.pairing_id, "date": pday.date,
               "delay_hours": delay_hours},
        summary=(
            f"{p.aircraft} delayed {delay_hours}h on {pday.date}: duty runs "
            f"{new_fdp}h against a {lim}h limit ({pday.sectors} sectors) - "
            + ("BREACH." if breach else
               "within its own FDP limit." if rest_breach else "still legal.")
            + (f" But {rest_detail}" if rest_breach else "")
        ),
        data={
            "pairing_id": p.pairing_id, "sectors": pday.sectors,
            "fdp_before": fdp, "fdp_after_delay": new_fdp, "fdp_limit": lim,
            "breach": breach,
            "breach_detail": (
                f"RULE-FDP-01: delayed duty runs {new_fdp}h vs {lim}h limit "
                f"({pday.sectors} sectors) - the rostered crew cannot legally "
                f"complete the full day." if breach else ""),
            "max_legal_prefix": prefix,
            "legs_to_shed": prefix.get("legs_to_shed", []),
            "rest_breach": rest_breach,
            "rest_breaches": rest_breaches,
            "rest_detail": rest_detail,
        },
        ledger=led,
    )


def analyse_cert_expiry(snap: Snapshot, crew_id: str) -> Impact:
    """Which rostered duties a certification lapse makes illegal."""
    led = Ledger()
    illegal = []
    for p in snap.pairings_for_crew(crew_id):
        for d in p.days:
            ok, bad = snap.certs_valid_on(crew_id, d.day)
            if not ok:
                illegal.append({
                    "crew_id": crew_id, "date": d.date, "rule": "RULE-CERT-06",
                    "pairing_id": p.pairing_id, "role": p.role_of(crew_id),
                    "expired": bad, "flights": list(d.flights),
                })
    led.add("illegal_duties", len(illegal))
    return Impact(
        event={"type": "CERT_EXPIRY", "crew_id": crew_id},
        summary=(f"{crew_id} has {len(illegal)} rostered duty/duties that are "
                 f"illegal under RULE-CERT-06." if illegal
                 else f"{crew_id} has no rostered duty affected by a lapse."),
        data={"illegal_assignment": illegal[0] if illegal else None,
              "illegal_assignments": illegal},
        ledger=led,
    )


# --------------------------------------------------------------------------
# resolution: impact -> ranked options
# --------------------------------------------------------------------------


def resolve(snap: Snapshot, crew_id: str, pairing_id: str,
            include_cancel: bool = True):
    """Standard resolution path: who can take over this trip, ranked."""
    p = snap.pairings[pairing_id]
    role = p.role_of(crew_id)
    if role is None:
        raise LookupError(f"{crew_id} is not rostered on {pairing_id}")
    return cover_options(snap, list(p.days), role, crew_id, pairing_id,
                         include_cancel=include_cancel)


def resolve_multi(snap: Snapshot, events: list[SickCrew]):
    """Two or more simultaneous disruptions, solved jointly.

    Answering them one at a time can hand the same person both trips.
    """
    needs = []
    for ev in events:
        p = snap.pairings[ev.pairing_id]
        needs.append({
            "key": ev.pairing_id,
            "cover_days": list(p.days),
            "role": p.role_of(ev.crew_id),
            "sick_crew_id": ev.crew_id,
            "exclude_pairing": ev.pairing_id,
        })
    return solve_joint(snap, needs)


# --------------------------------------------------------------------------
# counterfactual: what is the smallest change that makes this legal?
# --------------------------------------------------------------------------


def minimal_repair(snap: Snapshot, crew_id: str, pairing_id: str) -> dict[str, Any]:
    """"No" is half an answer. This computes what "yes" would cost.

    For each binding constraint, the exact margin that has to be recovered, plus
    the concrete lever that would recover it. Incumbent systems tell a
    controller a candidate is illegal; none of them says the assignment becomes
    legal if 1.33 hours of earlier duty is released.
    """
    p = snap.pairings[pairing_id]
    role = p.role_of(crew_id)
    res = check_cover(snap, crew_id, list(p.days), pairing_id)
    if res.legal:
        return {"crew_id": crew_id, "pairing_id": pairing_id,
                "already_legal": True, "repairs": []}

    repairs: list[dict[str, Any]] = []
    for issue in res.issues:
        if issue.startswith("RULE-DUTY-02"):
            worst = None
            for f in res.ledger.facts:
                if f.key.startswith("duty_7d_excess") and isinstance(f.value, float):
                    if worst is None or f.value > worst[1]:
                        worst = (f.key, f.value)
            if worst:
                d = worst[0].split("[")[1].rstrip("]")
                repairs.append({
                    "rule": "RULE-DUTY-02",
                    "shortfall_hours": round(worst[1], 2),
                    "lever": (f"release {round(worst[1], 2)}h of this crew "
                              f"member's other duty inside the 7 days ending {d}"),
                    "alternatives": [
                        "assign a different crew member (see ranked options)",
                        "split the pairing so they operate only the first day",
                    ],
                })
        elif issue.startswith("RULE-REST-04"):
            rest = next((f.value for f in res.ledger.facts
                         if f.key.startswith("rest_hours")), None)
            if rest is not None:
                need = round(12.0 - rest, 2)
                repairs.append({
                    "rule": "RULE-REST-04",
                    "shortfall_hours": need,
                    "lever": (f"move the conflicting duty {need}h later, or "
                              f"release the crew {need}h earlier"),
                    "alternatives": ["re-crew the downstream pairing instead"],
                })
        elif issue.startswith("RULE-QUAL-05"):
            repairs.append({
                "rule": "RULE-QUAL-05", "shortfall_hours": None,
                "lever": "no repair - a type rating is not a schedule problem",
                "alternatives": ["assign a rated crew member"],
            })
        elif issue.startswith("RULE-CERT-06"):
            repairs.append({
                "rule": "RULE-CERT-06", "shortfall_hours": None,
                "lever": "renew the lapsed certification before the duty date",
                "alternatives": ["assign a current crew member"],
            })
        elif issue.startswith("RULE-FDP-01"):
            fdp = res.ledger.get("fdp_hours")
            lim = res.ledger.get("fdp_limit_hours")
            if fdp and lim:
                repairs.append({
                    "rule": "RULE-FDP-01",
                    "shortfall_hours": round(fdp - lim, 2),
                    "lever": (f"shorten the duty by {round(fdp - lim, 2)}h, or "
                              f"drop a sector to raise the limit"),
                    "alternatives": ["re-crew the tail legs"],
                })

    return {
        "crew_id": crew_id, "pairing_id": pairing_id, "role": role,
        "already_legal": False, "issues": res.issues, "repairs": repairs,
    }
