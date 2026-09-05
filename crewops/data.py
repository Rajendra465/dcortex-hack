"""Dataset loading, typed records and indices.

The snapshot is immutable. Every disruption is expressed as an *overlay* applied
on read (see `crewops.events`), never as a mutation of this data. That is what
makes held-out scenarios cost zero new code: they are the same event templates
with different arguments.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

# --------------------------------------------------------------------------
# time helpers -- the whole dataset is UTC, and every window is calendar-day
# --------------------------------------------------------------------------

ISO = "%Y-%m-%dT%H:%M:%SZ"


def parse_utc(s: str) -> datetime:
    return datetime.strptime(s, ISO)


def fmt_utc(d: datetime) -> str:
    return d.strftime(ISO)


def hours(td: timedelta) -> float:
    """Round to 2dp. The oracle does this at every step; matching it matters."""
    return round(td.total_seconds() / 3600.0, 2)


def at(d: date, hhmm: str) -> datetime:
    h, m = hhmm.split(":")
    return datetime(d.year, d.month, d.day, int(h), int(m))


def fmt_hm(excess_hours: float) -> str:
    """Format an overage the way the oracle does: 1.333h -> '1h20m'."""
    hh = int(excess_hours)
    mm = int(round((excess_hours - hh) * 60))
    return f"{hh}h{mm:02d}m"


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Flight:
    flight_id: str
    flight_no: str
    date: str
    dep_station: str
    arr_station: str
    dep_utc: str
    arr_utc: str
    block_hours: float
    aircraft: str
    aircraft_type: str
    seats: int

    @property
    def dep(self) -> datetime:
        return parse_utc(self.dep_utc)

    @property
    def arr(self) -> datetime:
        return parse_utc(self.arr_utc)


@dataclass(frozen=True)
class Crew:
    crew_id: str
    name: str
    rank: str
    base: str
    ratings: tuple[str, ...]
    seniority: int
    reachability_minutes: int
    status: str

    @property
    def is_pilot(self) -> bool:
        return self.rank in ("Captain", "First Officer")


@dataclass(frozen=True)
class PairingDay:
    """One duty day. `report`/`release` bracket the whole day, not one leg."""

    date: str
    flights: tuple[str, ...]
    report_utc: str
    release_utc: str

    @property
    def report(self) -> datetime:
        return parse_utc(self.report_utc)

    @property
    def release(self) -> datetime:
        return parse_utc(self.release_utc)

    @property
    def day(self) -> date:
        return date.fromisoformat(self.date)

    @property
    def sectors(self) -> int:
        return len(self.flights)

    def shifted(self, delay_h: float, shift_report: bool = True) -> "PairingDay":
        """Return a copy moved by `delay_h`.

        TRAP 10 -- two different delay semantics live in this problem:
          * deadhead positioning shifts report AND release, so FDP is invariant;
          * a technical delay holds report and pushes release, so FDP GROWS.
        `shift_report` selects between them. Getting this wrong breaks either
        the S4 delay scenario or the S2 deadhead option.
        """
        if not delay_h:
            return self
        d = timedelta(hours=delay_h)
        return PairingDay(
            date=self.date,
            flights=self.flights,
            report_utc=fmt_utc(self.report + d) if shift_report else self.report_utc,
            release_utc=fmt_utc(self.release + d),
        )


@dataclass(frozen=True)
class Pairing:
    pairing_id: str
    aircraft: str
    days: tuple[PairingDay, ...]
    crew: tuple[tuple[str, str], ...]  # (crew_id, role)

    def role_of(self, crew_id: str) -> str | None:
        for cid, role in self.crew:
            if cid == crew_id:
                return role
        return None

    def crew_in_role(self, role: str) -> str | None:
        for cid, r in self.crew:
            if r == role:
                return cid
        return None


@dataclass(frozen=True)
class Reserve:
    crew_id: str
    base: str
    dates: tuple[str, ...]
    window_start: str
    window_end: str

    def covers(self, when: datetime) -> bool:
        """TRAP 2: inclusive at BOTH ends, and tested against the required
        REPORT time -- not the callout time the rule prose describes. The
        flagship C-3310 answer sits exactly on the 06:00 boundary."""
        return at(when.date(), self.window_start) <= when <= at(when.date(), self.window_end)


@dataclass(frozen=True)
class DutyDay:
    """A duty the crew member is already committed to, from the roster."""

    day: date
    report: datetime
    release: datetime
    duty_hours: float
    flight_hours: float
    pairing_id: str


COVER = "COVER"  # sentinel pairing_id for a simulated cover duty


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------


@dataclass
class Snapshot:
    flights: dict[str, Flight]
    crew: dict[str, Crew]
    pairings: dict[str, Pairing]
    reserves: dict[str, Reserve]
    certs: dict[str, dict[str, date]]  # crew -> cert_type -> valid_to
    cert_rows: list[dict[str, Any]]
    history: dict[str, dict[date, tuple[float, float]]]
    clocks: dict[str, dict[str, Any]]
    risk: dict[str, dict[str, Any]]
    costs: dict[str, Any]
    rules: dict[str, Any]
    flagged_exceptions: list[dict[str, Any]]

    # derived
    roster: dict[str, list[DutyDay]] = field(default_factory=dict)
    flight_to_pairing: dict[str, tuple[str, str]] = field(default_factory=dict)
    snapshot_utc: datetime = datetime(2026, 9, 14, 18, 0, 0)

    # ---- indices -------------------------------------------------------
    def __post_init__(self) -> None:
        self._by_rank: dict[str, list[str]] = defaultdict(list)
        self._by_date: dict[str, list[str]] = defaultdict(list)
        for cid, c in self.crew.items():
            self._by_rank[c.rank].append(cid)
        for fid, f in self.flights.items():
            self._by_date[f.date].append(fid)
        for k in self._by_date:
            self._by_date[k].sort(
                key=lambda x: (self.flights[x].dep_utc, self.flights[x].flight_no)
            )

    def crew_by_rank(self, rank: str) -> list[str]:
        return self._by_rank.get(rank, [])

    def flights_on(self, d: str) -> list[Flight]:
        return [self.flights[f] for f in self._by_date.get(d, [])]

    @property
    def dates(self) -> list[str]:
        return sorted(self._by_date)

    @property
    def horizon(self) -> tuple[date, date]:
        ds = self.dates
        return date.fromisoformat(ds[0]), date.fromisoformat(ds[-1])

    @property
    def history_start(self) -> date:
        any_crew = next(iter(self.history.values()))
        return min(any_crew)

    @property
    def cert_horizon(self) -> tuple[date, date]:
        """How far ahead certificate questions can be answered.

        Not the same as `horizon`. The roster runs one week; certificate
        validity runs to 2032, so "whose licences lapse before March?" is a
        question this dataset CAN answer and was being refused as out of range
        purely because the roster stops on the 20th. A refusal that is wrong is
        worse than a wrong answer -- it teaches the controller not to ask.
        """
        vs = [v for m in self.certs.values() for v in m.values()]
        return (min(vs), max(vs)) if vs else self.horizon

    # ---- lookups -------------------------------------------------------
    def pairing_of_flight(self, flight_id: str) -> tuple[Pairing, PairingDay] | None:
        hit = self.flight_to_pairing.get(flight_id)
        if not hit:
            return None
        pid, dstr = hit
        p = self.pairings[pid]
        return p, next(d for d in p.days if d.date == dstr)

    def pairings_for_crew(self, crew_id: str) -> list[Pairing]:
        return [p for p in self.pairings.values() if p.role_of(crew_id)]

    def certs_valid_on(self, crew_id: str, on: date) -> tuple[bool, list[str]]:
        """TRAP 1 -- `valid_to` ONLY.

        Every one of the 150 crew carries a licence `valid_from` in 2027-2032.
        Implementing the semantically correct `valid_from <= d <= valid_to`
        returns "no legal crew exists" for the entire fleet, and validate.py
        still reports PASS. See `crewops.conformance` for the strict mode that
        demonstrates this.
        """
        bad = [t for t, v in self.certs.get(crew_id, {}).items() if v < on]
        return (not bad), sorted(bad)

    def certs_valid_on_strict(self, crew_id: str, on: date) -> tuple[bool, list[str]]:
        """The semantically correct check. Used only by --strict, to prove the point."""
        bad = []
        for row in self.cert_rows:
            if row["crew_id"] != crew_id:
                continue
            vf = date.fromisoformat(row["valid_from"])
            vt = date.fromisoformat(row["valid_to"])
            if not (vf <= on <= vt):
                bad.append(row["cert_type"])
        return (not bad), sorted(bad)


# --------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "extracted",
    "DCortex - Synthetic dataset",
    "data",
)


def _read(data_dir: str, name: str) -> Any:
    with open(os.path.join(data_dir, name), encoding="utf-8") as fh:
        return json.load(fh)


def load(data_dir: str | None = None) -> Snapshot:
    d = data_dir or os.environ.get("CREWOPS_DATA_DIR") or DEFAULT_DATA_DIR
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"dataset directory not found: {d}\n"
            "Set CREWOPS_DATA_DIR or pass --data <dir>."
        )

    flights = {
        f["flight_id"]: Flight(
            flight_id=f["flight_id"], flight_no=f["flight_no"], date=f["date"],
            dep_station=f["dep_station"], arr_station=f["arr_station"],
            dep_utc=f["dep_utc"], arr_utc=f["arr_utc"], block_hours=f["block_hours"],
            aircraft=f["aircraft"], aircraft_type=f["aircraft_type"], seats=f["seats"],
        )
        for f in _read(d, "flights.json")
    }

    crew = {
        c["crew_id"]: Crew(
            crew_id=c["crew_id"], name=c["name"], rank=c["rank"], base=c["base"],
            ratings=tuple(c["ratings"]), seniority=c["seniority"],
            reachability_minutes=c["reachability_minutes"], status=c["status"],
        )
        for c in _read(d, "crew.json")
    }

    rosters = _read(d, "rosters.json")
    pairings: dict[str, Pairing] = {}
    for p in rosters["pairings"]:
        pairings[p["pairing_id"]] = Pairing(
            pairing_id=p["pairing_id"],
            aircraft=p["aircraft"],
            days=tuple(
                PairingDay(
                    date=day["date"], flights=tuple(day["flights"]),
                    report_utc=day["report_utc"], release_utc=day["release_utc"],
                )
                for day in p["days"]
            ),
            crew=tuple((m["crew_id"], m["role"]) for m in p["crew"]),
        )

    reserves = {
        r["crew_id"]: Reserve(
            crew_id=r["crew_id"], base=r["base"], dates=tuple(r["dates"]),
            window_start=r["oncall_window_utc"]["start"],
            window_end=r["oncall_window_utc"]["end"],
        )
        for r in _read(d, "reserve_pool.json")
    }

    cert_rows = _read(d, "certifications.json")
    certs: dict[str, dict[str, date]] = defaultdict(dict)
    for c in cert_rows:
        certs[c["crew_id"]][c["cert_type"]] = date.fromisoformat(c["valid_to"])

    clock_rows = _read(d, "duty_clocks.json")
    clocks = {c["crew_id"]: c for c in clock_rows}
    history = {
        c["crew_id"]: {
            date.fromisoformat(x["date"]): (x["duty_hours"], x["flight_hours"])
            for x in c["daily_history"]
        }
        for c in clock_rows
    }

    risk = {r["crew_id"]: r for r in _read(d, "risk_signals.json")}

    snap = Snapshot(
        flights=flights, crew=crew, pairings=pairings, reserves=reserves,
        certs=dict(certs), cert_rows=cert_rows, history=history, clocks=clocks,
        risk=risk, costs=_read(d, "costs.json"), rules=_read(d, "rules.json"),
        flagged_exceptions=rosters.get("flagged_exceptions", []),
    )

    # ---- derived: the roster as per-crew duty days, and the flight->pairing map
    roster: dict[str, list[DutyDay]] = defaultdict(list)
    f2p: dict[str, tuple[str, str]] = {}
    for p in pairings.values():
        for day in p.days:
            fh = round(sum(flights[f].block_hours for f in day.flights), 2)
            dh = hours(day.release - day.report)
            for fid in day.flights:
                f2p[fid] = (p.pairing_id, day.date)
            duty = DutyDay(
                day=day.day, report=day.report, release=day.release,
                duty_hours=dh, flight_hours=fh, pairing_id=p.pairing_id,
            )
            for cid, _role in p.crew:
                roster[cid].append(duty)
    for cid in roster:
        roster[cid].sort(key=lambda x: x.report)
    snap.roster = dict(roster)
    snap.flight_to_pairing = f2p
    return snap


_CACHE: dict[str | None, Snapshot] = {}


def load_cached(data_dir: str | None = None) -> Snapshot:
    if data_dir not in _CACHE:
        _CACHE[data_dir] = load(data_dir)
    return _CACHE[data_dir]


# --------------------------------------------------------------------------
# startup invariants -- fail loudly if the dataset is not what we assume
# --------------------------------------------------------------------------


def assert_shape(snap: Snapshot) -> list[str]:
    """Hard assertions about dataset shape, returned as lines so the CLI can
    display them. Raises nothing, so a modified dataset still loads."""
    checks = [
        ("flights", len(snap.flights), 147),
        ("crew", len(snap.crew), 150),
        ("pairings", len(snap.pairings), 39),
        ("reserves", len(snap.reserves), 16),
        ("cert rows", len(snap.cert_rows), 600),
    ]
    out = []
    for name, got, want in checks:
        out.append(f"{'ok  ' if got == want else 'DIFF'} {name}: {got} (expected {want})")
    covered = set(snap.flight_to_pairing)
    missing = set(snap.flights) - covered
    out.append(
        f"{'ok  ' if not missing else 'DIFF'} every flight crewed: "
        f"{len(covered)}/{len(snap.flights)}"
    )
    return out
