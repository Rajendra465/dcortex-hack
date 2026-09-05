"""The seven legality rules, as pure predicates that emit their arithmetic.

Every function here returns not just a verdict but the numbers it used. That
collection -- the *evidence ledger* -- is the only thing the narrating model is
ever shown, and it is what the numeric containment guard checks prose against.

A naive containment guard ("every number in the sentence must appear in the
answer object") was measured at a 67.8% false-block rate against the reference
answer keys, because the most explanatory numbers are intermediates that never
appear as fields: "only 10.75h rest", "over by 1h20m", "total 61.33h". So the
ledger records intermediates and unit-converted forms too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from .data import COVER, PairingDay, Snapshot, fmt_hm, hours

EPS = 1e-6  # the oracle compares with `> limit + 1e-6`, so exact equality is LEGAL


# --------------------------------------------------------------------------
# evidence ledger
# --------------------------------------------------------------------------


@dataclass
class Fact:
    key: str
    value: Any
    unit: str = ""
    source: str = ""
    derivation: str = ""

    def numbers(self) -> list[float]:
        if isinstance(self.value, bool):
            return []
        if isinstance(self.value, (int, float)):
            return [float(self.value)]
        return []


@dataclass
class Ledger:
    """Everything the kernel computed, in the order it computed it."""

    facts: list[Fact] = field(default_factory=list)

    def add(self, key: str, value: Any, unit: str = "", source: str = "",
            derivation: str = "") -> Any:
        self.facts.append(Fact(key, value, unit, source, derivation))
        return value

    def extend(self, other: "Ledger") -> None:
        self.facts.extend(other.facts)

    def get(self, key: str) -> Any:
        for f in self.facts:
            if f.key == key:
                return f.value
        return None

    def as_dict(self) -> list[dict[str, Any]]:
        return [
            {"key": f.key, "value": f.value, "unit": f.unit,
             "source": f.source, "derivation": f.derivation}
            for f in self.facts
        ]

    def allowed_numbers(self) -> set[str]:
        """Every numeric token the narrator is permitted to write.

        Includes each value's raw form, its 1- and 2-dp forms, its integer form
        when integral, and -- crucially -- the hour/minute decomposition, because
        the engine itself converts 1.33h into "1h20m" and the guard must not
        block its own output.
        """
        out: set[str] = set()
        for f in self.facts:
            for n in f.numbers():
                out.add(f"{n:g}")
                out.add(f"{n:.1f}")
                out.add(f"{n:.2f}")
                if abs(n - round(n)) < 1e-9:
                    out.add(str(int(round(n))))
                if n > 0:
                    hh, mm = int(n), int(round((n - int(n)) * 60))
                    out.add(str(hh))
                    out.add(str(mm))
                    out.add(f"{mm:02d}")
        return out


# --------------------------------------------------------------------------
# duty period construction
# --------------------------------------------------------------------------


def fdp_limit(sectors: int, params: dict[str, Any] | None = None) -> float:
    """RULE-FDP-01. Base fdp, minus reduction per sector beyond the free sectors.

    Counted PER DUTY DAY, not per pairing: 2 legs -> 13.0, 3 -> 12.5, 4 -> 12.0.

    `params` became required in a refactor that updated one of the five call
    sites. The other four -- both delay paths in events.py, the legal-prefix
    search, and compute_duty_period -- raised TypeError, which took out every
    delay simulation and the duty-period tool with it. It is optional again and
    falls back to the shipped rules.json values, so a caller that has the
    snapshot can pass the real params and a caller that does not still gets the
    rulebook rather than an exception.
    """
    params = params or {}
    base = params.get("base_fdp_hours", 13.0)
    red = params.get("reduction_per_extra_sector_hours", 0.5)
    free = params.get("free_sectors", 2)
    return base - red * max(0, sectors - free)


def duty_period(day: PairingDay) -> tuple[float, datetime, datetime]:
    return hours(day.release - day.report), day.report, day.release


def earliest_next_report(release: datetime) -> datetime:
    """RULE-REST-04, forward form: release + 12h."""
    return release + timedelta(hours=12)


# --------------------------------------------------------------------------
# window sums (RULE-DUTY-02 / RULE-FLT-03)
# --------------------------------------------------------------------------


def window_sum(
    snap: Snapshot,
    crew_id: str,
    end: date,
    window_days: int,
    metric: str = "duty",
    include_roster: bool = True,
    exclude_pairing: str | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Sum duty or flight hours over a CALENDAR-DAY window ending `end`.

    TRAP 9: the window is `window_days` UTC dates inclusive of the duty date --
    not a rolling 168 hours. A rolling clock gives different numbers and fails
    the reference answers.

    TRAP 4: 2026-09-14 is contributed by BOTH `daily_history` and the roster,
    for the 11 crew who have duty on that date in both. That double count is
    canonical -- `duty_clocks.duty_hours_7d` embodies it and the dataset's own
    validator reproduces it. "Fixing" it silently changes legality verdicts.

    TRAP 3: never read `duty_clocks.duty_hours_7d` for this. That field is a
    snapshot for the window ending 2026-09-14 only; reusing it for any other
    date flips C-3305 from legal to illegal.
    """
    start = end - timedelta(days=window_days - 1)
    idx = 0 if metric == "duty" else 1
    total = 0.0
    breakdown: list[dict[str, Any]] = []

    for d, v in sorted(snap.history.get(crew_id, {}).items()):
        if start <= d <= end and v[idx]:
            total += v[idx]
            breakdown.append({"date": d.isoformat(), "hours": v[idx], "source": "history"})

    if include_roster:
        for duty in snap.roster.get(crew_id, []):
            if not (start <= duty.day <= end):
                continue
            if exclude_pairing and duty.pairing_id == exclude_pairing:
                continue
            h = duty.duty_hours if metric == "duty" else duty.flight_hours
            if h:
                total += h
                breakdown.append({
                    "date": duty.day.isoformat(), "hours": h,
                    "source": "roster", "pairing_id": duty.pairing_id,
                })

    return round(total, 2), breakdown


# --------------------------------------------------------------------------
# the rule predicates
# --------------------------------------------------------------------------


def rule_qual_05(snap: Snapshot, crew_id: str, aircraft_type: str,
                 led: Ledger) -> list[str]:
    """RULE-QUAL-05 -- valid rating for the assigned aircraft type.

    TRAP 6: this SHORT-CIRCUITS. It is the only early return in the legality
    chain, so an unrated candidate is excluded with exactly one reason string
    and never accumulates duty or rest issues. Running all checks and
    concatenating produces multi-issue strings that do not match the keys.
    """
    ratings = snap.crew[crew_id].ratings
    led.add("aircraft_type", aircraft_type, source="flights.json")
    led.add("ratings", list(ratings), source="crew.json")
    if aircraft_type not in ratings:
        return [f"RULE-QUAL-05: no {aircraft_type} rating"]
    return []


def rule_cert_06(snap: Snapshot, crew_id: str, on: date, led: Ledger,
                 strict: bool = False) -> list[str]:
    """RULE-CERT-06 -- all certifications valid on the duty date.

    TRAP 1: `valid_to` only. See Snapshot.certs_valid_on.
    """
    ok, bad = (snap.certs_valid_on_strict(crew_id, on) if strict
               else snap.certs_valid_on(crew_id, on))
    if not ok:
        led.add("invalid_certs", bad, source="certifications.json")
        # Name the certificate and the day it lapsed. "certification invalid"
        # is true and leaves a controller to go and look up which one, on a
        # question whose whole point is which one.
        detail = []
        for t in bad:
            exp = snap.certs.get(crew_id, {}).get(t)
            if exp:
                led.add(f"cert_expiry[{t}]", str(exp),
                        source="certifications.json")
            detail.append(f"{t} expired {exp}" if exp else t)
        return [f"RULE-CERT-06: {', '.join(detail)}; duty on {on} is illegal"]
    return []


def rule_fdp_01(snap: Snapshot, day: PairingDay, led: Ledger) -> list[str]:
    fdp, _rep, _rel = duty_period(day)
    params = snap.rule_params.get("RULE-FDP-01", {})
    lim = fdp_limit(day.sectors, params)
    
    base = params.get("base_fdp_hours", 13.0)
    red = params.get("reduction_per_extra_sector_hours", 0.5)
    free = params.get("free_sectors", 2)

    led.add("sectors", day.sectors, source="rosters.json")
    led.add("fdp_hours", fdp, "h", derivation="release - report")
    led.add("fdp_limit_hours", lim, "h",
            derivation=f"{base} - {red} x max(0, {day.sectors} - {free})")
    led.add("fdp_margin_hours", round(lim - fdp, 2), "h")
    if fdp > lim + EPS:
        return [f"RULE-FDP-01: FDP {fdp}h > {lim}h limit ({day.sectors} sectors)"]
    return []


def rule_duty_02(snap: Snapshot, crew_id: str, cover_days: list[PairingDay],
                 exclude_pairing: str | None, led: Ledger) -> list[str]:
    """RULE-DUTY-02 -- max 60 duty hours in any 7 consecutive calendar days.

    Evaluated once per cover day, because a multi-day pairing must be legal on
    EVERY day (TRAP 12). C-3305 is the teaching case: fine on 15 Sep, over by
    8h15m on 16 Sep.

    The disrupted pairing is subtracted from the candidate's own load -- they
    are being moved *onto* the trip, not stacked on top of it.
    """
    issues: list[str] = []
    params = snap.rule_params.get("RULE-DUTY-02", {})
    cap = float(params.get("max_duty_hours", 60.0))
    window_days = int(params.get("window_days", 7))
    
    led.add("duty_cap_hours", cap, "h", source="rules.json RULE-DUTY-02")
    led.add("duty_window_days", window_days, "days", source="rules.json RULE-DUTY-02")

    for day in cover_days:
        d = day.day
        base, breakdown = window_sum(
            snap, crew_id, d, window_days, "duty", exclude_pairing=exclude_pairing
        )
        # cumulative cover duty up to and including this date, undelayed length
        add = round(sum(duty_period(x)[0] for x in cover_days if x.day <= d), 2)
        total = round(base + add, 2)
        led.add(f"duty_7d_existing[{d}]", base, "h",
                source="duty_clocks.daily_history + rosters.json",
                derivation=f"{len(breakdown)} contributing days")
        led.add(f"duty_7d_added[{d}]", add, "h", derivation="cover duty up to this date")
        led.add(f"duty_7d_total[{d}]", total, "h", derivation=f"{base} + {add}")
        led.add(f"duty_7d_headroom[{d}]", round(cap - total, 2), "h")
        if total > cap + EPS:
            excess = total - cap
            led.add(f"duty_{window_days}d_excess[{d}]", round(excess, 2), "h")
            issues.append(
                f"RULE-DUTY-02: would exceed {cap:g}h/{window_days}d by {fmt_hm(excess)} "
                f"on {d} (total {total}h)"
            )
    return issues


def rule_flt_03(snap: Snapshot, crew_id: str, cover_days: list[PairingDay],
                exclude_pairing: str | None, led: Ledger) -> tuple[list[str], bool]:
    """RULE-FLT-03 -- max 100 block hours in any 28 consecutive calendar days.

    TRAP 5: the reference resolver lists this rule in every option's
    `rules_checked` but never computes it. We DO compute it -- for honesty and
    for held-out safety -- and report it as *evaluated, non-binding* rather than
    claiming it was "checked". Peak utilisation across the whole dataset is
    79.28h of the 100h cap, so it cannot bind on this data.

    Returns (issues, binding).
    """
    params = snap.rule_params.get("RULE-FLT-03", {})
    cap = float(params.get("max_flight_hours", 100.0))
    window_days = int(params.get("window_days", 28))
    
    binding = False
    issues: list[str] = []
    for day in cover_days:
        d = day.day
        base, _ = window_sum(snap, crew_id, d, window_days, "flight",
                             exclude_pairing=exclude_pairing)
        led.add(f"flight_28d[{d}]", base, "h",
                source="duty_clocks.daily_history + rosters.json")
        led.add(f"flight_28d_headroom[{d}]", round(cap - base, 2), "h")
        if base > cap + EPS:
            binding = True
            issues.append(f"RULE-FLT-03: would exceed {cap:g}h/{window_days}d on {d} (total {base}h)")
    led.add("flight_cap_hours", cap, "h", source="rules.json RULE-FLT-03")
    return issues, binding


def rule_rest_04(snap: Snapshot, sim: list[tuple], led: Ledger) -> list[str]:
    """RULE-REST-04 -- minimum 12h between release and the next report.

    TRAP 13: rest conflicts are frequently NOT on the cover days. C-5837 is
    legal across the trip itself and fails on only 10.75h of rest before his own
    duty two days later. So we merge the candidate's remaining roster with the
    proposed cover, sort by report time, and check every adjacent pair.

    Two independent passes over the same pairs, as the reference does: one for
    the rest gap, one for outright overlap. An overlapping pair therefore yields
    two strings, joined by "; ".
    """
    issues: list[str] = []
    params = snap.rule_params.get("RULE-REST-04", {})
    min_rest = float(params.get("min_rest_hours", 12.0))
    led.add("min_rest_hours", min_rest, "h", source="rules.json RULE-REST-04")

    for a, b in zip(sim, sim[1:]):
        rest = hours(b[1] - a[2])
        if rest < min_rest - EPS:
            # "downstream" only when a COVER duty is followed by a real pairing
            tag = "downstream" if (b[5] != COVER and a[5] == COVER) else "rest"
            led.add(f"rest_hours[{a[5]}->{b[5]}@{b[0]}]", rest, "h",
                    derivation="next report - previous release")
            issues.append(
                f"RULE-REST-04: only {rest}h rest before {b[5]} on {b[0]} ({tag} conflict)"
            )

    for a, b in zip(sim, sim[1:]):
        if b[1] < a[2]:
            issues.append(f"double-booked: {a[5]} overlaps {b[5]} on {b[0]}")

    return issues


def build_timeline(snap: Snapshot, crew_id: str, cover_days: list[PairingDay],
                   exclude_pairing: str | None) -> list[tuple]:
    """Merge the candidate's own remaining duties with the proposed cover.

    Tuples are (date, report, release, duty_hours, flight_hours, pairing_id),
    sorted by report time. Python's sort is stable, which keeps a rostered duty
    ahead of a cover duty reporting at the same instant -- the reference relies
    on that ordering.
    """
    sim: list[tuple] = [
        (d.day, d.report, d.release, d.duty_hours, d.flight_hours, d.pairing_id)
        for d in snap.roster.get(crew_id, [])
        if d.pairing_id != exclude_pairing
    ]
    for day in cover_days:
        fdp, rep, rel = duty_period(day)
        sim.append((day.day, rep, rel, fdp, 0.0, COVER))
    sim.sort(key=lambda x: x[1])
    return sim
