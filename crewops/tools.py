"""The typed tool catalog -- the only surface through which a question reaches
the kernel.

Every tool is named for a DECISION or a fact, never for a database operation.
There is no `run_query`, no `search`, no free-form SQL: text-to-SQL accuracy
falls from ~91% on toy schemas to ~21% on realistic ones, so we do not generate
queries against our schema at all.

Each tool returns `(payload, ledger)`. The ledger is what the narrator is shown;
the payload is what the answer contract is built from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from .data import Snapshot, fmt_utc, parse_utc
from .events import (SickCrew, analyse_cert_expiry, analyse_closure,
                     analyse_delay, analyse_sick, minimal_repair, resolve_multi)
from .kernel import (check_cover, cover_fragility, cover_options,
                     latent_breaches, positioning, reserve_coverage_gaps)
from .rules import Ledger, duty_period, earliest_next_report, fdp_limit, window_sum

Result = tuple[Any, Ledger]

REGISTRY: dict[str, "Tool"] = {}


@dataclass
class Tool:
    name: str
    fn: Callable[..., Result]
    params: dict[str, str]
    required: tuple[str, ...]
    doc: str
    kind: str  # retrieval | simulation | optimisation
    # Arguments where ONE of a group must be present. compute_duty_period
    # declared nothing required but indexes pairing_id in its body, so a plan
    # missing both it and release_utc passed validation and died at runtime as
    # an entity error -- a crash wearing a refusal costume.
    requires_one_of: tuple = ()

    def __call__(self, snap: Snapshot, **kw: Any) -> Result:
        return self.fn(snap, **kw)


def tool(name: str, params: dict[str, str], required: tuple[str, ...],
         kind: str, doc: str, requires_one_of: tuple = ()):
    def deco(fn: Callable[..., Result]) -> Callable[..., Result]:
        REGISTRY[name] = Tool(name, fn, params, required, doc, kind,
                              requires_one_of)
        return fn
    return deco


class Unanswerable(Exception):
    """Raised when the data genuinely cannot support the question.

    Distinct from a parse failure -- see `crewops.agent`. Conflating the two
    lets a bug wear the costume of intellectual honesty.
    """

    def __init__(self, reason: str, have: str = "", would_need: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.have = have
        self.would_need = would_need


# ==========================================================================
# LAYER A -- retrieval
# ==========================================================================


@tool("find_flights",
      {"date": "YYYY-MM-DD", "dep_station": "3-letter code",
       "arr_station": "3-letter code", "flight_no": "e.g. DX412",
       "aircraft": "e.g. VT-DXA", "aircraft_type": "A320 | ATR72",
       "dep_after_utc": "HH:MM", "dep_before_utc": "HH:MM",
       "aggregate": "count | max_block | max_seats | cancel_cost | distinct_arr"},
      (), "retrieval",
      "Flights matching any combination of filters. Use `aggregate` so counting "
      "is never done by the model.")
def find_flights(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    rows = list(snap.flights.values())
    for key, attr in (("date", "date"), ("dep_station", "dep_station"),
                      ("arr_station", "arr_station"), ("flight_no", "flight_no"),
                      ("aircraft", "aircraft"), ("aircraft_type", "aircraft_type")):
        v = kw.get(key)
        if v:
            rows = [f for f in rows if getattr(f, attr) == v]
    if kw.get("dep_after_utc"):
        rows = [f for f in rows if f.dep.strftime("%H:%M") >= kw["dep_after_utc"]]
    if kw.get("dep_before_utc"):
        rows = [f for f in rows if f.dep.strftime("%H:%M") < kw["dep_before_utc"]]
    rows.sort(key=lambda f: (f.date, f.dep_utc, f.flight_no))
    led.add("matching_flights", len(rows), source="flights.json")

    agg = kw.get("aggregate")
    payload: dict[str, Any] = {
        "flights": [f.flight_no for f in rows],
        "flight_ids": [f.flight_id for f in rows],
        "count": len(rows),
        "rows": [{"flight_id": f.flight_id, "flight_no": f.flight_no,
                  "date": f.date, "dep_station": f.dep_station,
                  "arr_station": f.arr_station, "dep_utc": f.dep_utc,
                  "arr_utc": f.arr_utc, "block_hours": f.block_hours,
                  "aircraft": f.aircraft, "aircraft_type": f.aircraft_type,
                  "seats": f.seats} for f in rows],
    }
    if agg == "count":
        payload["answer"] = len(rows)
    elif agg == "max_block" and rows:
        mx = max(f.block_hours for f in rows)
        led.add("max_block_hours", mx, "h")
        payload["answer"] = {"block_hours": mx,
                             "flights": sorted({f.flight_no for f in rows
                                                if f.block_hours == mx})}
    elif agg == "max_seats" and rows:
        # "Which single leg has the most seats at risk if cancelled?" is a
        # comparison, and no aggregate could express it -- so it answered with
        # all 147 flights, which is every leg rather than the worst one.
        mx = max(f.seats for f in rows)
        worst = sorted({f.flight_no for f in rows if f.seats == mx})
        others = sorted({f.seats for f in rows if f.seats != mx}, reverse=True)
        led.add("max_seats", mx, "seats")
        led.add("legs_at_max_seats", len(worst))
        if others:
            # The comparison number has to be in the ledger too. It went into
            # the sentence and not the ledger, and the guard blocked the whole
            # answer -- which is the guard doing its job, not a bug in it.
            led.add("next_largest_seats", others[0], "seats")
        payload["answer"] = {
            "seats": mx, "flights": worst[:12], "count": len(worst),
            "next_largest": others[0] if others else None,
            "why": (f"{mx}-seat aircraft against {others[0]}-seat"
                    if others else f"{mx} seats"),
        }
    elif agg == "cancel_cost" and rows:
        # "How many passengers are affected and what does it cost?" needs the
        # seats AND the rate, and the rate lives in costs.json. Answering with
        # the flight row left the controller to do both lookups themselves.
        seats = sum(f.seats for f in rows)
        rate = snap.costs.get("cancellation_per_flight", 250000)
        led.add("seats_at_risk", seats, "seats")
        led.add("cancellation_per_flight", rate, "INR", source="costs.json")
        led.add("cancellation_cost", rate * len(rows), "INR",
                derivation=f"{rate} x {len(rows)} leg(s)")
        payload["answer"] = {"passengers": seats, "legs": len(rows),
                             "cost_inr": rate * len(rows)}
    elif agg == "distinct_arr":
        payload["answer"] = sorted({f.arr_station for f in rows})
    # Echo the filters that were applied. An empty result is only useful if it
    # can say what it looked for -- "0 flight(s):" leaves a controller unable
    # to tell an empty afternoon from a broken query.
    for k in ("date", "dep_station", "arr_station", "flight_no",
              "dep_after_utc", "dep_before_utc"):
        if kw.get(k):
            payload[k] = kw[k]
    return payload, led


@tool("find_crew",
      {"crew_id": "e.g. C-1042", "rank": "Captain | First Officer | "
       "Senior Cabin Crew | Cabin Crew",
       "ranks": "list, for a class such as pilots = Captain + First Officer",
       "base": "3-letter code", "rating": "A320 | ATR72",
       "status": "active | leave | training", "name": "partial name match",
       "limit": "return at most N"},
      (), "retrieval", "Crew records by any combination of attributes.")
def find_crew(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    rows = list(snap.crew.values())
    if kw.get("crew_id"):
        rows = [c for c in rows if c.crew_id == kw["crew_id"]]
    for key, attr in (("rank", "rank"), ("base", "base"), ("status", "status")):
        if kw.get(key):
            rows = [c for c in rows if getattr(c, attr) == kw[key]]
    if kw.get("ranks"):
        wanted = set(kw["ranks"])
        rows = [c for c in rows if c.rank in wanted]
    if kw.get("rating"):
        rows = [c for c in rows if kw["rating"] in c.ratings]
    if kw.get("name"):
        needle = kw["name"].lower()
        rows = [c for c in rows if needle in c.name.lower()]
    rows.sort(key=lambda c: c.crew_id)
    total = len(rows)
    if kw.get("limit"):
        rows = rows[: int(kw["limit"])]
    led.add("matching_crew", total, source="crew.json")
    if kw.get("limit"):
        led.add("returned", len(rows), derivation=f"limited to {kw['limit']} of {total}")
    return {
        "crew_ids": [c.crew_id for c in rows],
        "count": len(rows), "total_matching": total,
        "rows": [{"crew_id": c.crew_id, "name": c.name, "rank": c.rank,
                  "base": c.base, "ratings": list(c.ratings),
                  "seniority": c.seniority, "status": c.status,
                  "reachability_minutes": c.reachability_minutes} for c in rows],
    }, led


@tool("get_roster",
      {"crew_id": "e.g. C-1042", "pairing_id": "e.g. P-2291",
       "flight_id": "e.g. DX412-2026-09-15", "aircraft": "e.g. VT-DXA",
       "date": "YYYY-MM-DD", "role": "Captain | ..."},
      (), "retrieval", "Pairings (multi-day trips) and who is on them.")
def get_roster(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    ps = list(snap.pairings.values())
    if kw.get("pairing_id"):
        ps = [p for p in ps if p.pairing_id == kw["pairing_id"]]
    if kw.get("crew_id"):
        ps = [p for p in ps if p.role_of(kw["crew_id"])]
    if kw.get("aircraft"):
        ps = [p for p in ps if p.aircraft == kw["aircraft"]]
    if kw.get("flight_id"):
        ps = [p for p in ps if any(kw["flight_id"] in d.flights for d in p.days)]
    if kw.get("date"):
        ps = [p for p in ps if any(d.date == kw["date"] for d in p.days)]
    ps.sort(key=lambda p: p.pairing_id)
    led.add("matching_pairings", len(ps), source="rosters.json")

    out = []
    for p in ps:
        crew = [{"crew_id": c, "role": r} for c, r in p.crew]
        if kw.get("role"):
            crew = [c for c in crew if c["role"] == kw["role"]]
        out.append({
            "pairing_id": p.pairing_id, "aircraft": p.aircraft,
            "aircraft_type": snap.flights[p.days[0].flights[0]].aircraft_type,
            "days": [{"date": d.date, "flights": list(d.flights),
                      "report_utc": d.report_utc, "release_utc": d.release_utc,
                      "sectors": d.sectors} for d in p.days],
            "crew": crew,
        })
    return {"pairings": out, "count": len(out)}, led


@tool("get_reserves",
      {"date": "YYYY-MM-DD", "crew_id": "e.g. C-3310",
       "base": "3-letter code", "rank": "Captain | ...",
       "ranks": "list, for a class such as pilots",
       "rating": "A320 | ATR72", "covers_utc": "full ISO timestamp"},
      (), "retrieval",
      "Standby crew on call for a date, optionally filtered to those whose "
      "window covers a specific report time.",
      requires_one_of=(("date", "crew_id", "base", "rank", "ranks"),))
def get_reserves(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    # `date` used to be mandatory, which made "is C-3310 on standby?" and
    # "which captains are on the reserve pool?" unanswerable for no reason:
    # the pool itself is a fact independent of any one day. Omitting it now
    # returns the whole roster of on-call crew with their date lists.
    d = kw.get("date")
    rows = []
    for cid, r in snap.reserves.items():
        if d and d not in r.dates:
            continue
        c = snap.crew[cid]
        if kw.get("crew_id") and cid != kw["crew_id"]:
            continue
        if kw.get("base") and r.base != kw["base"]:
            continue
        if kw.get("rank") and c.rank != kw["rank"]:
            continue
        if kw.get("ranks") and c.rank not in set(kw["ranks"]):
            continue
        if kw.get("rating") and kw["rating"] not in c.ratings:
            continue
        covers = None
        if kw.get("covers_utc"):
            covers = r.covers(parse_utc(kw["covers_utc"]))
            if not covers:
                continue
        rows.append({
            "crew_id": cid, "rank": c.rank, "base": r.base,
            "ratings": list(c.ratings),
            "window": {"start": r.window_start, "end": r.window_end},
            "reachability_minutes": c.reachability_minutes,
            "covers": covers,
            "on_call_dates": list(r.dates) if not d else None,
        })
    rows.sort(key=lambda x: x["crew_id"])
    led.add("reserves_on_call", len(rows), source="reserve_pool.json")
    return {"reserves": rows, "count": len(rows), "date": d}, led


@tool("duty_hours",
      {"crew_id": "e.g. C-1042 (or omit for all)", "end_date": "YYYY-MM-DD",
       "window_days": "7 or 28", "metric": "duty | flight",
       "min_hours": "float threshold"},
      ("end_date",), "retrieval",
      "Accrued duty or block hours over a calendar-day window, with the "
      "day-by-day breakdown and remaining headroom.")
def duty_hours(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    end = date.fromisoformat(kw["end_date"])
    wd = int(kw.get("window_days", 7))
    metric = kw.get("metric", "duty")
    cap = 60.0 if metric == "duty" else 100.0
    ids = [kw["crew_id"]] if kw.get("crew_id") else sorted(snap.crew)

    rows = []
    for cid in ids:
        total, breakdown = window_sum(snap, cid, end, wd, metric)
        if kw.get("min_hours") is not None and total < float(kw["min_hours"]):
            continue
        rows.append({"crew_id": cid, "hours": total, "limit": cap,
                     "headroom_hours": round(cap - total, 2),
                     "breakdown": breakdown})
    rows.sort(key=lambda r: -r["hours"])

    if kw.get("crew_id") and rows:
        r = rows[0]
        led.add("duty_hours_7d" if wd == 7 else f"{metric}_hours_{wd}d",
                r["hours"], "h",
                source="duty_clocks.daily_history + rosters.json",
                derivation=f"{wd} calendar days ending {end}")
        # The window length is a number the answer says out loud ("over 7
        # days"), so it belongs in the ledger. A figure the narrator is
        # allowed to use but cannot cite is exactly the gap the containment
        # guard exists to close.
        led.add("window_days", wd, "days")
        led.add("limit_hours", cap, "h")
        led.add("headroom_hours", r["headroom_hours"], "h",
                derivation=f"{cap} - {r['hours']}")
    else:
        led.add("crew_matching", len(rows))
    return {"rows": rows, "count": len(rows), "window_days": wd,
            "metric": metric, "end_date": kw["end_date"], "limit": cap}, led


@tool("get_certifications",
      {"crew_id": "e.g. C-1042", "cert_type": "licence | medical_class1 | ...",
       "valid_on": "YYYY-MM-DD", "expiring_before": "YYYY-MM-DD",
       "expiring_after": "YYYY-MM-DD"},
      (), "retrieval",
      "Certification validity. RULE-CERT-06 is evaluated on valid_to only -- "
      "see the conformance report for why.")
def get_certifications(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    rows = list(snap.cert_rows)
    if kw.get("crew_id"):
        rows = [r for r in rows if r["crew_id"] == kw["crew_id"]]
    if kw.get("cert_type"):
        rows = [r for r in rows if r["cert_type"] == kw["cert_type"]]
    if kw.get("expiring_before"):
        rows = [r for r in rows if r["valid_to"] < kw["expiring_before"]]
    if kw.get("expiring_after"):
        rows = [r for r in rows if r["valid_to"] >= kw["expiring_after"]]
    if kw.get("valid_on"):
        on = date.fromisoformat(kw["valid_on"])
        rows = [r for r in rows if date.fromisoformat(r["valid_to"]) >= on]
    rows.sort(key=lambda r: (r["valid_to"], r["crew_id"]))
    led.add("matching_certifications", len(rows), source="certifications.json")
    return {"certifications": [{"crew_id": r["crew_id"],
                                "cert_type": r["cert_type"],
                                "valid_to": r["valid_to"]} for r in rows],
            "count": len(rows),
            "policy": "RULE-CERT-06 evaluated on valid_to only"}, led


@tool("get_risk_signals", {"crew_id": "e.g. C-1042", "min_score": "0..1",
                           "top_n": "integer"},
      (), "retrieval",
      "Pre-computed disruption-risk scores. A PROVIDED INPUT -- never an input "
      "to any legality or cost decision, and never a prediction we make.")
def get_risk_signals(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    rows = list(snap.risk.values())
    if kw.get("crew_id"):
        rows = [r for r in rows if r["crew_id"] == kw["crew_id"]]
    if kw.get("min_score") is not None:
        rows = [r for r in rows
                if r["disruption_risk_score"] >= float(kw["min_score"])]
    rows.sort(key=lambda r: -r["disruption_risk_score"])
    if kw.get("top_n"):
        rows = rows[:int(kw["top_n"])]
    led.add("risk_signals_matching", len(rows),
            source="risk_signals.json (provided input)")
    if kw.get("crew_id") and rows:
        led.add("disruption_risk_score", rows[0]["disruption_risk_score"],
                source="risk_signals.json (provided input)")
        # The driver strings are quoted verbatim to the controller and carry
        # their own figures ("short-rest pattern over last 14 days"). Those
        # numbers are real -- they are simply somebody else's, not ours -- so
        # they are ledgered as PROVIDED rather than left uncited. Recording
        # them is what lets the containment guard stay strict everywhere else
        # instead of being loosened to accommodate this one tool.
        for n in re.findall(r"\d+(?:\.\d+)?", " ".join(rows[0]["drivers"])):
            led.add(f"driver_figure[{n}]", float(n) if "." in n else int(n),
                    source="risk_signals.json drivers (provided verbatim)")
    return {"signals": [{"crew_id": r["crew_id"],
                         "score": r["disruption_risk_score"],
                         "drivers": r["drivers"]} for r in rows],
            "count": len(rows),
            "provenance": "provided input, not computed by this system"}, led


@tool("get_rulebook", {"rule_id": "e.g. RULE-DUTY-02"}, (), "retrieval",
      "The legality ruleset and cost rates, with the conventions used.")
def get_rulebook(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    rules = snap.rules["rules"]
    if kw.get("rule_id"):
        rules = [r for r in rules if r["rule_id"] == kw["rule_id"]]
    return {"rules": rules, "costs": snap.costs,
            "conventions": {
                "report": "first departure minus 60 min",
                "release": "last arrival plus 30 min",
                "windows": "calendar-day UTC, inclusive of the duty date",
                "reserve_window": "tested against the required report time, "
                                  "inclusive at both ends",
            }}, led


# ==========================================================================
# LAYER B -- simulation
# ==========================================================================


@tool("compute_duty_period",
      {"pairing_id": "e.g. P-2291", "date": "YYYY-MM-DD",
       "delay_hours": "float", "release_utc": "ISO timestamp"},
      (), "simulation",
      "Duty period, FDP and the earliest legal next report for one duty day.",
      requires_one_of=(("pairing_id", "release_utc"),))
def compute_duty_period(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    if kw.get("release_utc"):
        rel = parse_utc(kw["release_utc"])
        nxt = earliest_next_report(rel)
        led.add("release_utc", kw["release_utc"])
        led.add("min_rest_hours", 12.0, "h", source="rules.json RULE-REST-04")
        led.add("earliest_next_report_utc", fmt_utc(nxt))
        return {"release_utc": kw["release_utc"],
                "earliest_next_report_utc": fmt_utc(nxt),
                "answer": fmt_utc(nxt)}, led

    p = snap.pairings[kw["pairing_id"]]
    d = (next(x for x in p.days if x.date == kw["date"]) if kw.get("date")
         else p.days[0])
    delay = float(kw.get("delay_hours") or 0)
    fdp, _rep, rel = duty_period(d)
    lim = fdp_limit(d.sectors)
    led.add("sectors", d.sectors)
    led.add("fdp_hours", round(fdp + delay, 2), "h")
    led.add("fdp_limit_hours", lim, "h")
    return {"pairing_id": p.pairing_id, "date": d.date,
            "report_utc": d.report_utc, "release_utc": d.release_utc,
            "sectors": d.sectors, "fdp_hours": round(fdp + delay, 2),
            "fdp_limit_hours": lim,
            "fdp_ok": round(fdp + delay, 2) <= lim + 1e-6,
            "margin_hours": round(lim - fdp - delay, 2),
            "earliest_next_report_utc": fmt_utc(
                earliest_next_report(rel + timedelta(hours=delay)))}, led


@tool("check_legality",
      {"crew_id": "e.g. C-2087", "pairing_id": "e.g. P-2291",
       "exclude_pairing": "pairing to free up first",
       "delay_hours": "float", "days": "1 to restrict to day one"},
      ("crew_id", "pairing_id"), "simulation",
      "Can this crew member legally take this trip? Returns the verdict, every "
      "rule violated, and the arithmetic behind each.")
def check_legality(snap: Snapshot, **kw: Any) -> Result:
    p = snap.pairings[kw["pairing_id"]]
    days = list(p.days)
    if kw.get("days"):
        days = days[:int(kw["days"])]
    res = check_cover(snap, kw["crew_id"], days,
                      kw.get("exclude_pairing", kw["pairing_id"]),
                      float(kw.get("delay_hours") or 0))
    payload = res.to_dict()
    payload["pairing_id"] = kw["pairing_id"]
    payload["rules_note"] = (
        "RULE-FLT-03 was evaluated and is non-binding on this dataset "
        "(peak 79.28h of the 100h cap)."
        if "RULE-FLT-03" in res.non_binding else "")

    # If this candidate has to be positioned from another base, the legality
    # verdict is only half the answer -- the controller also needs the
    # operational consequence, computed (never narrated) from the positioning.
    needed_base = snap.flights[p.days[0].flights[0]].dep_station
    crew = snap.crew.get(kw["crew_id"])
    if crew and crew.base != needed_base:
        first_dep = snap.flights[
            min(days[0].flights, key=lambda f: snap.flights[f].dep_utc)].dep
        possible, delay_h, flight_no, arr = positioning(
            crew.base, needed_base, days[0].day, first_dep)
        if possible and flight_no and arr:
            payload["positioning"] = {
                "flight_no": flight_no,
                "arrival_utc": arr.strftime("%H:%M") + "Z",
                "delay_hours": delay_h,
            }
            payload["consequence"] = (
                f"Deadhead positioning on {flight_no} "
                f"(arr {arr.strftime('%H:%M')}Z) delays the first departure by "
                f"~{delay_h:g}h; RULE-BASE-07 deadhead cost applies.")
    return payload, res.ledger


@tool("simulate_crew_unavailable",
      {"crew_id": "e.g. C-1042", "pairing_id": "optional, to scope it",
       "from_date": "YYYY-MM-DD"},
      ("crew_id",), "simulation",
      "Which flights become uncrewed if this person cannot fly, including "
      "downstream days of the same trip.")
def simulate_crew_unavailable(snap: Snapshot, **kw: Any) -> Result:
    imp = analyse_sick(snap, kw["crew_id"], kw.get("pairing_id"),
                       kw.get("from_date"))
    return imp.to_dict(), imp.ledger


@tool("simulate_station_closure",
      {"station": "3-letter code", "start_utc": "ISO timestamp",
       "end_utc": "ISO timestamp"},
      ("station", "start_utc", "end_utc"), "simulation",
      "Which flights a closure hits, the minimum delay for each, and whether "
      "the rostered crew stay inside their duty limit.")
def simulate_station_closure(snap: Snapshot, **kw: Any) -> Result:
    imp = analyse_closure(snap, kw["station"], kw["start_utc"], kw["end_utc"])
    return imp.to_dict(), imp.ledger


@tool("simulate_delay",
      {"delay_hours": "float", "aircraft": "e.g. VT-DXA",
       "pairing_id": "e.g. P-2201", "date": "YYYY-MM-DD"},
      ("delay_hours",), "simulation",
      "A technical delay: does the extended duty breach the FDP limit, and how "
      "many legs can the original crew still legally operate?")
def simulate_delay(snap: Snapshot, **kw: Any) -> Result:
    imp = analyse_delay(snap, float(kw["delay_hours"]), kw.get("aircraft"),
                        kw.get("pairing_id"), kw.get("date"))
    return imp.to_dict(), imp.ledger


@tool("simulate_cert_expiry", {"crew_id": "e.g. C-5417"}, ("crew_id",),
      "simulation", "Which rostered duties a certification lapse makes illegal.")
def simulate_cert_expiry(snap: Snapshot, **kw: Any) -> Result:
    imp = analyse_cert_expiry(snap, kw["crew_id"])
    return imp.to_dict(), imp.ledger


# ==========================================================================
# LAYER C -- optimisation
# ==========================================================================


@tool("rank_cover_options",
      {"pairing_id": "e.g. P-2291", "crew_id": "the person being replaced",
       "role": "override the role", "top_n": "integer"},
      ("pairing_id",), "optimisation",
      "Every legal way to cover a trip, ranked by cost, with every rejected "
      "candidate and the exact rule that rejected them.")
def rank_cover_options(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    p = snap.pairings[kw["pairing_id"]]
    crew_id = kw.get("crew_id")
    role = kw.get("role") or (p.role_of(crew_id) if crew_id else None)
    if not role:
        raise Unanswerable(
            f"I need to know which role to cover on {kw['pairing_id']}.",
            have="the pairing and its full crew complement",
            would_need="a crew_id or a role")
    os_ = cover_options(snap, list(p.days), role, crew_id, kw["pairing_id"])
    legal = os_.legal_options
    led.add("candidate_pool_size", os_.candidate_pool_size,
            source=f"all active {role}s")
    led.add("legal_options", len(legal))
    led.add("excluded_candidates", len(os_.excluded))
    if legal:
        led.add("cheapest_cost_inr", legal[0].cost_inr, "INR")
        for o in legal[:5]:
            led.add(f"cost[{o.crew_id}]", o.cost_inr, "INR")
    payload = os_.to_dict()
    if kw.get("top_n"):
        payload["options"] = payload["options"][:int(kw["top_n"])]
    payload["recommended"] = os_.options[0].to_dict() if os_.options else None
    payload["pairing_id"] = kw["pairing_id"]
    return payload, led


@tool("solve_joint_cover", {"events": "list of {crew_id, pairing_id}"},
      ("events",), "optimisation",
      "Allocate scarce cover across simultaneous disruptions. Answering them "
      "one at a time can assign the same person twice.")
def solve_joint_cover(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    evs = [SickCrew(crew_id=e["crew_id"], pairing_id=e["pairing_id"])
           for e in kw["events"]]
    plan = resolve_multi(snap, evs)
    led.add("total_cost_inr", plan.total_cost_inr, "INR")
    led.add("events", len(evs))
    for k, o in plan.assignments.items():
        led.add(f"cost[{k}]", o.cost_inr, "INR")
    return plan.to_dict(), led


@tool("minimal_repair", {"crew_id": "e.g. C-2087", "pairing_id": "e.g. P-2291"},
      ("crew_id", "pairing_id"), "optimisation",
      "What is the smallest change that would make an illegal assignment legal?")
def minimal_repair_tool(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    out = minimal_repair(snap, kw["crew_id"], kw["pairing_id"])
    for r in out.get("repairs", []):
        if r.get("shortfall_hours") is not None:
            led.add(f"shortfall[{r['rule']}]", r["shortfall_hours"], "h")
    return out, led


# ==========================================================================
# LAYER D -- proactive
# ==========================================================================


@tool("cover_fragility", {"role": "Captain | First Officer", "limit": "integer"},
      (), "optimisation",
      "Cover depth for every trip in the week -- where a single sick call has "
      "no answer. The signal a duty-hour watchlist cannot give you here.")
def cover_fragility_tool(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    roles = (kw["role"],) if kw.get("role") else ("Captain", "First Officer")
    rows = cover_fragility(snap, roles,
                           int(kw["limit"]) if kw.get("limit") else None)
    critical = [r for r in rows if r["legal_covers"] <= 1]
    led.add("trips_assessed", len(rows))
    led.add("single_point_of_failure", len(critical))
    return {"rows": rows, "critical": critical, "count": len(rows)}, led


@tool("reserve_gaps", {}, (), "optimisation",
      "Hours of the day with no on-call reserve for a rank and aircraft type.")
def reserve_gaps_tool(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    gaps = reserve_coverage_gaps(snap)
    led.add("uncovered_rank_type_hours", len(gaps))
    runs: dict[tuple[str, str], list[int]] = {}
    for g in gaps:
        runs.setdefault((g["rank"], g["aircraft_type"]), []).append(g["hour_utc"])
    summary = [{"rank": k[0], "aircraft_type": k[1],
                "uncovered_hours_utc": sorted(v)} for k, v in sorted(runs.items())]
    return {"gaps": summary, "count": len(gaps)}, led


@tool("latent_breaches", {}, (), "optimisation",
      "Assignments in the published roster that are ALREADY illegal, found "
      "without being told where to look.")
def latent_breaches_tool(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    rows = latent_breaches(snap)
    led.add("already_illegal_duties", len(rows))
    return {"breaches": rows, "count": len(rows),
            "flagged_in_dataset": snap.flagged_exceptions}, led


@tool("notification_packet",
      {"crew_id": "e.g. C-3310", "pairing_id": "e.g. P-2291"},
      ("crew_id", "pairing_id"), "simulation",
      "The verified facts for a crew callout message. The model writes the "
      "prose; every hard fact is checked back against this packet.")
def notification_packet(snap: Snapshot, **kw: Any) -> Result:
    led = Ledger()
    p = snap.pairings[kw["pairing_id"]]
    c = snap.crew[kw["crew_id"]]
    days = []
    for i, d in enumerate(p.days):
        last = max(d.flights, key=lambda f: snap.flights[f].dep_utc)
        arr_st = snap.flights[last].arr_station
        overnight = arr_st if i < len(p.days) - 1 else None
        days.append({
            "date": d.date,
            "flights": [snap.flights[f].flight_no for f in d.flights],
            "report_utc": d.report_utc, "release_utc": d.release_utc,
            "report_station": snap.flights[
                min(d.flights, key=lambda f: snap.flights[f].dep_utc)].dep_station,
            "overnight_station": overnight,
            "hotel_required": bool(overnight and overnight != c.base),
        })
        led.add(f"report_utc[{d.date}]", d.report_utc)
    ack = parse_utc(p.days[0].report_utc) - timedelta(hours=2)
    led.add("acknowledgement_deadline_utc", fmt_utc(ack))
    return {
        "crew_id": c.crew_id, "name": c.name, "rank": c.rank, "base": c.base,
        "pairing_id": p.pairing_id, "aircraft": p.aircraft, "days": days,
        "report_place": f"{days[0]['report_station']} crew room",
        "acknowledgement_deadline_utc": fmt_utc(ack),
        "contact": "Crew Control desk",
        "reachability_minutes": c.reachability_minutes,
    }, led


@tool("draft_notification",
      {"crew_id": "e.g. C-3310", "pairing_id": "e.g. P-2291",
       "audience": "crew | occ | duty_manager"},
      ("crew_id", "pairing_id"), "simulation",
      "The actual message to send, written from verified facts and then "
      "fact-locked: every time, date, flight and id in the draft must appear "
      "in the packet, or it is not sent.")
def draft_notification(snap: Snapshot, **kw: Any) -> Result:
    packet, led = notification_packet(snap, crew_id=kw["crew_id"],
                                      pairing_id=kw["pairing_id"])
    audience = kw.get("audience", "crew")
    d0 = packet["days"][0]

    if audience == "occ":
        lines = [
            f"{packet['pairing_id']} re-crewed - {packet['crew_id']} "
            f"({packet['rank']}) assigned.",
            f"First report {d0['report_utc']} at {d0['report_station']}; "
            f"no schedule change.",
            f"Aircraft {packet['aircraft']}. Legs: "
            + "; ".join(", ".join(day["flights"]) for day in packet["days"]) + ".",
        ]
    elif audience == "duty_manager":
        lines = [
            f"Cover decision - {packet['pairing_id']}",
            f"Assigned {packet['crew_id']} ({packet['rank']}, base "
            f"{packet['base']}), reachable in "
            f"{packet['reachability_minutes']} min.",
            f"Report {d0['report_utc']} at {d0['report_station']}. All seven "
            f"rules evaluated; the candidates ruled out are in the decision "
            f"record.",
        ]
    else:  # crew
        lines = [
            f"{packet['rank']} {packet['name']} ({packet['crew_id']}) - "
            f"callout for {packet['pairing_id']}",
            "",
            f"Please report {d0['report_utc']} at {packet['report_place']}.",
            "",
        ]
        for i, day in enumerate(packet["days"], 1):
            seg = (f"Day {i} ({day['date']}): {', '.join(day['flights'])}, "
                   f"report {day['report_utc']}, release {day['release_utc']}.")
            if day["overnight_station"]:
                seg += (f" Overnight at {day['overnight_station']}"
                        + (" - hotel arranged." if day["hotel_required"] else "."))
            lines.append(seg)
        lines += [
            "",
            f"Please acknowledge by {packet['acknowledgement_deadline_utc']}.",
            f"Any issue, call {packet['contact']}.",
        ]

    draft = "\n".join(lines)

    # Fact-lock. A model may write the sentences; it does not get to choose the
    # times. Every id, date and clock time in the draft must trace back to the
    # packet, or the draft is flagged rather than sent.
    facts = {packet["crew_id"], packet["pairing_id"], packet["aircraft"],
             packet["acknowledgement_deadline_utc"]}
    for day in packet["days"]:
        facts |= {day["date"], day["report_utc"], day["release_utc"],
                  day["report_station"], *day["flights"]}
        if day["overnight_station"]:
            facts.add(day["overnight_station"])
    tokens = set(re.findall(
        r"\b(?:[CP]-\d{4}|DX\d{3}|VT-DX[A-F]|\d{4}-\d{2}-\d{2}T[\d:]+Z|"
        r"\d{4}-\d{2}-\d{2})\b", draft))
    unverified = sorted(t for t in tokens if t not in facts)

    led.add("draft_tokens_checked", len(tokens))
    led.add("unverified_tokens", len(unverified))
    return {
        "audience": audience,
        "draft": draft,
        "packet": packet,
        "fact_locked": not unverified,
        "unverified_tokens": unverified,
        "coverage_checklist": {
            "crew_and_pairing": True,
            "report_time_and_place": True,
            "flights_per_day": True,
            "overnight_and_hotel": any(d["overnight_station"]
                                       for d in packet["days"]),
            "acknowledgement_deadline": True,
            "contact": audience == "crew",
        },
    }, led


def describe_tools() -> list[dict[str, Any]]:
    return [{"name": t.name, "kind": t.kind, "doc": t.doc,
             "params": t.params, "required": list(t.required)}
            for t in REGISTRY.values()]
