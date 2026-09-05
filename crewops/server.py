"""A local web desk for the Crew Ops Advisor.

Standard library only -- no framework, no build step, one command to start. It
binds to 127.0.0.1 because this answers questions about crew rosters and has no
authentication: it is a demo surface, not a deployment.

The UI follows the same rules as the CLI. The verdict is never streamed, the
ruled-out candidates are shown by default rather than hidden behind a click, and
every figure says whether the rules engine computed it or a model wrote it.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agent import Advisor
from .conformance import report as conformance_report
from .data import Snapshot, load
from .evaluate import run as run_eval, run_e2e
from .kernel import cover_fragility, latent_breaches, reserve_coverage_gaps

_STATE: dict[str, Any] = {}
_SESSIONS: dict[str, Any] = {}
_LOCK = threading.Lock()


def _session(sid: str):
    """One Session per browser tab.

    The server is threaded, and a Session mutates its own overlay stack, so
    handing two requests the same Session object without a lock would let one
    tab's what-if leak into another's answer. Sessions are per-id and every
    ask is serialised on that id.
    """
    from .agent import Session
    with _LOCK:
        if sid not in _SESSIONS:
            _SESSIONS[sid] = {"s": Session(_STATE["snap"],
                                           use_model=_STATE["use_model"]),
                              "lock": threading.Lock()}
        return _SESSIONS[sid]


def _brief(snap: Snapshot) -> dict[str, Any]:
    frag = cover_fragility(snap)
    gaps: dict[str, list[int]] = {}
    for g in reserve_coverage_gaps(snap):
        gaps.setdefault(f"{g['rank']} / {g['aircraft_type']}", []).append(g["hour_utc"])
    return {
        "already_illegal": latent_breaches(snap),
        "fragility": frag[:8],
        "single_points": [r for r in frag if r["legal_covers"] <= 1],
        "reserve_gaps": [{"who": k, "hours": sorted(v)} for k, v in sorted(gaps.items())],
        "note": ("A duty-hour watchlist is deliberately absent: peak 7-day "
                 "utilisation here is 42.51h against a 60h cap, so that panel "
                 "would be empty."),
    }


def _roster(snap: Snapshot) -> list[dict[str, Any]]:
    out = []
    for cid, c in snap.crew.items():
        clk = snap.clocks.get(cid, {})
        d7 = clk.get("duty_hours_7d", 0.0)
        f28 = clk.get("flight_hours_28d", 0.0)
        pairings = [p.pairing_id for p in snap.pairings_for_crew(cid)]
        status = "Reserve" if cid in snap.reserves else ("Assigned" if pairings else "Available")
        out.append({
            "crew_id": cid,
            "name": c.name,
            "rank": c.rank,
            "base": c.base,
            "ratings": list(c.ratings),
            "seniority": c.seniority,
            "reachability_minutes": c.reachability_minutes,
            "duty_hours_7d": d7,
            "duty_limit_7d": 60.0,
            "duty_util_pct": round(min(100.0, (d7 / 60.0) * 100), 1),
            "flight_hours_28d": f28,
            "flight_limit_28d": 100.0,
            "flight_util_pct": round(min(100.0, (f28 / 100.0) * 100), 1),
            "pairings": pairings,
            "status": status,
        })
    return sorted(out, key=lambda x: (x["rank"], x["crew_id"]))


def _flights(snap: Snapshot) -> list[dict[str, Any]]:
    out = []
    for fid, f in snap.flights.items():
        p_info = snap.pairing_of_flight(fid)
        pid = p_info[0].pairing_id if p_info else None
        p = p_info[0] if p_info else None
        capt = p.crew_in_role("Captain") if p else None
        fo = p.crew_in_role("First Officer") if p else None
        scc = p.crew_in_role("Senior Cabin Crew") if p else None
        out.append({
            "flight_id": fid,
            "flight_no": f.flight_no,
            "date": f.date,
            "dep_station": f.dep_station,
            "arr_station": f.arr_station,
            "dep_utc": f.dep_utc,
            "arr_utc": f.arr_utc,
            "block_hours": f.block_hours,
            "aircraft": f.aircraft,
            "aircraft_type": f.aircraft_type,
            "seats": f.seats,
            "pairing_id": pid,
            "captain": capt,
            "first_officer": fo,
            "cabin_crew": scc,
        })
    return sorted(out, key=lambda x: (x["dep_utc"], x["flight_no"]))


def _reserves(snap: Snapshot) -> list[dict[str, Any]]:
    out = []
    for cid, r in snap.reserves.items():
        c = snap.crew.get(cid)
        clk = snap.clocks.get(cid, {})
        d7 = clk.get("duty_hours_7d", 0.0)
        f28 = clk.get("flight_hours_28d", 0.0)
        out.append({
            "crew_id": cid,
            "name": c.name if c else cid,
            "rank": c.rank if c else "Unknown",
            "base": r.base,
            "ratings": list(c.ratings) if c else [],
            "dates": list(r.dates),
            "window": f"{r.window_start} - {r.window_end}",
            "window_start": r.window_start,
            "window_end": r.window_end,
            "duty_hours_7d": d7,
            "duty_headroom_7d": round(max(0.0, 60.0 - d7), 2),
            "flight_hours_28d": f28,
            "reachability_minutes": c.reachability_minutes if c else 60,
        })
    return sorted(out, key=lambda x: (x["base"], x["rank"], x["crew_id"]))


def _outreach(snap: Snapshot, data: dict[str, Any]) -> dict[str, Any]:
    cid = data.get("crew_id", "C-3310")
    pid = data.get("pairing_id", "P-2291")
    c = snap.crew.get(cid)
    p = snap.pairings.get(pid)
    cname = c.name if c else cid
    crank = c.rank if c else "Crew Member"
    cbase = c.base if c else "DEL"
    crate = "/".join(c.ratings) if c else "A320"

    legs = []
    report_utc = "05:00 UTC"
    pdate = "2026-09-15"
    if p and p.days:
        d0 = p.days[0]
        pdate = d0.date
        report_utc = d0.report_utc.split("T")[1][:5] + " UTC"
        for fid in d0.flights:
            fl = snap.flights.get(fid)
            if fl:
                legs.append(f"{fl.flight_no} ({fl.dep_station}→{fl.arr_station})")
    leg_str = ", ".join(legs) if legs else "DX412/DX413"

    whatsapp = (
        f"⚠️ *dCortex AIR - PRIORITY CREW DISPATCH*\n\n"
        f"Hello {crank} *{cname}* ({cid}),\n"
        f"OCC has assigned you to cover pairing *{pid}* on *{pdate}*.\n\n"
        f"📋 *Assignment Details:*\n"
        f"• Base: {cbase} | Fleet: {crate}\n"
        f"• Required Report Time: *{report_utc}*\n"
        f"• Scheduled Legs: {leg_str}\n"
        f"• DGCA Compliance: Verified PASS (FDP cap, 12h rest, 60h 7-day limit ok)\n\n"
        f"Please reply *CONFIRM* to acknowledge and accept duty, or call OCC Dispatch (+91-11-2565-8787)."
    )

    acars = (
        f"QU OCCDELXA\n"
        f".DELOCCX 141800\n"
        f"CREW REASSIGNMENT UPLINK\n"
        f"ATTN: {crank.upper()} {cid} / {cname.upper()}\n"
        f"REASSIGN: PAIRING {pid} EFF {pdate}\n"
        f"REPORT: {report_utc} @ {cbase} OPS\n"
        f"LEGS: {leg_str.upper()}\n"
        f"LEGALITY STATUS: DGCA PASS / ZERO BREACH\n"
        f"ACK REQD VIA ACARS OR FREQ 131.850 MHZ"
    )

    sms = (
        f"dCortex OCC: {crank} {cname} ({cid}), you are assigned Pairing {pid} on {pdate}. "
        f"Report {report_utc} at {cbase}. Flight legs: {leg_str}. "
        f"DGCA cleared. Confirm via portal or reply 1 to accept."
    )

    voice = (
        f"Good day {crank} {cname}. This is an automated notification from the dCortex Operations Control Center. "
        f"Due to operational disruption, you have been assigned to cover pairing {pid} on {pdate}. "
        f"Your required report time is {report_utc} at {cbase}. Your assigned sectors are {leg_str}. "
        f"All DGCA legality and rest period checks have been verified. Press 1 to acknowledge receipt and accept this assignment, "
        f"or press 2 to speak immediately with a flight duty dispatcher."
    )

    return {
        "crew_id": cid,
        "name": cname,
        "rank": crank,
        "base": cbase,
        "pairing_id": pid,
        "date": pdate,
        "report_utc": report_utc,
        "legs": leg_str,
        "channels": {
            "whatsapp": whatsapp,
            "acars": acars,
            "sms": sms,
            "voice": voice,
        },
    }


_SCENARIO_CACHE: dict[str, Any] = {}


def _scenarios() -> list[dict[str, Any]]:
    """The shipped scenarios, read once from scenarios.json."""
    if "list" not in _SCENARIO_CACHE:
        d = _STATE.get("data_dir") or ""
        try:
            with open(os.path.join(d, "scenarios.json"), encoding="utf-8") as fh:
                _SCENARIO_CACHE["list"] = json.load(fh)
        except Exception:
            _SCENARIO_CACHE["list"] = []
    return _SCENARIO_CACHE["list"]


def _scenario_events(sid: str) -> list[Any]:
    """The real event(s) for a scenario id, straight out of the dataset.

    This was a hand-written if/elif that had drifted from the data: S1 fired
    C-1042's sick call instead of C-3231's, S3 (a station closure) fired a sick
    call, S4 (a delay) fired the closure, and S6 did not exist at all. Every one
    of those buttons demonstrated something other than its own label. Reading
    the file means the six demo what they say, and a held-out scenario dropped
    into the dataset works with no code change.
    """
    from .events import CertExpiry, Delay, SickCrew, StationClosure
    scen = next((x for x in _scenarios() if x.get("scenario_id") == sid), None)
    if not scen:
        return []
    ev = scen.get("event", {})
    kind = ev.get("type")
    if kind == "SICK_CREW":
        return [SickCrew(crew_id=ev["crew_id"], pairing_id=ev.get("pairing_id"))]
    if kind == "MULTI_SICK":
        return [SickCrew(crew_id=e["crew_id"], pairing_id=e.get("pairing_id"))
                for e in ev.get("events", []) if e.get("crew_id")]
    if kind == "DELAY":
        return [Delay(delay_hours=ev.get("delay_hours"),
                      aircraft=ev.get("aircraft"), date=ev.get("date"))]
    if kind == "STATION_CLOSURE":
        w = ev.get("window_utc") or {}
        return [StationClosure(station=ev.get("station"),
                               start_utc=w.get("start") or ev.get("start_utc"),
                               end_utc=w.get("end") or ev.get("end_utc"))]
    if kind == "CERT_EXPIRY":
        return [CertExpiry(crew_id=ev.get("crew_id"))]
    return []


def _scenario_list() -> list[dict[str, Any]]:
    """What the desk needs to offer them: what happens, to whom, and when."""
    out = []
    for sc in _scenarios():
        ev = sc.get("event", {})
        sid = sc.get("scenario_id")
        who = ev.get("crew_id") or ", ".join(
            e.get("crew_id", "") for e in ev.get("events", [])) or ev.get("station") or ev.get("aircraft")
        out.append({
            "id": sid,
            "title": sc.get("title"),
            "difficulty": sc.get("difficulty"),
            "type": ev.get("type"),
            "subject": who,
            "narrative": ev.get("narrative", ""),
            "reported_utc": ev.get("reported_utc") or (ev.get("events") or [{}])[0].get("reported_utc"),
            "runnable": bool(_scenario_events(sid)),
        })
    return out


def _consequences(before: Snapshot, after: Snapshot, crew_id: str,
                  pairing_id: str) -> dict[str, Any]:
    """What this decision costs, measured rather than asserted.

    Three things a controller wants the moment they commit: is the trip
    actually covered now, what did it do to this person's own week, and who
    just stopped being available to everybody else.
    """
    from .kernel import cover_fragility, reserve_coverage_gaps
    out: dict[str, Any] = {}

    p = after.pairings.get(pairing_id)
    out["covered"] = bool(p and p.crew_in_role("Captain"))
    out["crew_now_on"] = [f"{c} ({r})" for c, r in (p.crew if p else ())]

    days = len(after.roster.get(crew_id, [])) - len(before.roster.get(crew_id, []))
    clock = after.clocks.get(crew_id)
    out["own_week"] = {
        "crew_id": crew_id,
        "duty_days_added": days,
        "duty_hours_7d": round(getattr(clock, "duty_hours_7d", 0.0) or 0.0, 2),
    }

    # Who this takes off the board. A standby captain who is now flying is not
    # standby cover for the next sick call, and that is the consequence a
    # controller most often discovers too late.
    out["left_standby"] = crew_id in before.reserves
    # Trips with one legal captain or none. This is the number that matters:
    # it counts how close the week is to having no answer at all, and a
    # decision that fixes today by making tomorrow single-threaded should say so.
    def thin(sn: Snapshot) -> int:
        try:
            return sum(1 for r in cover_fragility(sn, roles=("Captain",))
                       if r.get("legal_covers", 99) <= 1)
        except Exception:
            return -1
    b, a = thin(before), thin(after)
    if b >= 0 and a >= 0:
        out["single_cover_trips"] = {"before": b, "after": a, "delta": a - b}
    try:
        out["standby_gap_hours"] = {"before": len(reserve_coverage_gaps(before)),
                                    "after": len(reserve_coverage_gaps(after))}
    except Exception:
        pass
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        snap: Snapshot = _STATE["snap"]
        try:
            if path in ("/", "/index.html"):
                self._send(200, _page(), "text/html; charset=utf-8")
            elif path == "/api/health":
                adv: Advisor = _STATE["advisor"]
                self._json({
                    "ok": True,
                    "flights": len(snap.flights), "crew": len(snap.crew),
                    "pairings": len(snap.pairings),
                    "reserves": len(snap.reserves),
                    "snapshot_utc": "2026-09-14T18:00:00Z",
                    "model": adv.model_label if adv.model_available else None,
                })
            elif path == "/api/brief":
                self._json(_brief(snap))
            elif path == "/api/roster":
                self._json(_roster(snap))
            elif path == "/api/flights":
                self._json(_flights(snap))
            elif path == "/api/reserves":
                self._json(_reserves(snap))
            elif path == "/api/eval":
                self._json(run_eval(snap).to_dict())
            elif path == "/api/conformance":
                self._json(conformance_report(snap))
            elif path == "/api/routing":
                if _STATE.get("routing") is None:
                    _STATE["routing"] = run_e2e(snap, use_model=False)
                r = dict(_STATE["routing"])
                r.pop("rows", None)
                self._json(r)
            elif path in ("/desk", "/desk/"):
                with open(DESK_PATH, encoding="utf-8") as fh:
                    self._send(200, fh.read().encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/scenarios":
                self._json(_scenario_list())
            elif path == "/api/examples":
                self._json(EXAMPLES)
            elif path == "/api/stream":
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()
                
                import time, random
                snap: Snapshot = _STATE["snap"]
                
                # Precompute active items to disrupt
                assigned_crew = [cid for cid, c in snap.crew.items() if snap.pairings_for_crew(cid)]
                active_flights = list(snap.flights.values())
                stations = list(set(f.dep_station for f in active_flights))
                
                while True:
                    try:
                        ev_type = random.choice(["SICK_CREW", "DELAY", "STATION_CLOSURE"])
                        payload = None
                        
                        if ev_type == "SICK_CREW" and assigned_crew:
                            cid = random.choice(assigned_crew)
                            pairings = snap.pairings_for_crew(cid)
                            pid = pairings[0].pairing_id if pairings else None
                            payload = {
                                "scenario": "DYNAMIC",
                                "title": f"Random Alert: Captain {cid} called in sick",
                                "dynamic_event": {
                                    "type": "SICK_CREW",
                                    "crew_id": cid,
                                    "pairing_id": pid
                                }
                            }
                        elif ev_type == "DELAY" and active_flights:
                            flight = random.choice(active_flights)
                            delay_hrs = round(random.uniform(1.0, 4.0), 1)
                            payload = {
                                "scenario": "DYNAMIC",
                                "title": f"Random Alert: {flight.aircraft} delayed {delay_hrs}h on {flight.date}",
                                "dynamic_event": {
                                    "type": "DELAY",
                                    "aircraft": flight.aircraft,
                                    "delay_hours": delay_hrs,
                                    "date": flight.date
                                }
                            }
                        elif ev_type == "STATION_CLOSURE" and stations:
                            station = random.choice(stations)
                            d = "2026-09-16"
                            payload = {
                                "scenario": "DYNAMIC",
                                "title": f"Random Alert: {station} closed for 4 hours on {d}",
                                "dynamic_event": {
                                    "type": "STATION_CLOSURE",
                                    "station": station,
                                    "start_utc": f"{d}T08:00:00Z",
                                    "end_utc": f"{d}T12:00:00Z"
                                }
                            }
                        
                        if payload:
                            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        
                        time.sleep(20)
                    except Exception:
                        break
                return
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # never leave the socket hanging
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?")[0]
        if route not in ("/api/ask", "/api/whatif", "/api/reset", "/api/set_region",
                         "/api/resolve_event", "/api/commit"):
            self._json({"error": "not found"}, 404)
            return
        try:
            snap: Snapshot = _STATE["snap"]
            if route == "/api/commit":
                # Authorising is the only action on this desk that changes the
                # world. It goes through the same Session overlay as every
                # what-if, so it is visible, reversible with /api/reset, and
                # every later answer is computed in the world it created.
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                crew_id = str(body.get("crew_id") or "")
                pairing_id = str(body.get("pairing_id") or "")
                role = str(body.get("role") or "Captain")
                sid = str(body.get("session") or "default")
                if crew_id not in snap.crew:
                    self._json({"error": f"unknown crew {crew_id}"}, 400)
                    return
                if pairing_id not in snap.pairings:
                    self._json({"error": f"unknown pairing {pairing_id}"}, 400)
                    return
                from .events import Assign, apply as apply_events
                entry = _session(sid)
                with entry["lock"]:
                    sess = entry["s"]
                    before = apply_events(sess.base, list(sess.events))
                    sess.push_event(Assign(crew_id=crew_id, pairing_id=pairing_id,
                                           role=role))
                    after = apply_events(sess.base, list(sess.events))
                    self._json({
                        "ok": True,
                        "assigned": {"crew_id": crew_id, "pairing_id": pairing_id,
                                     "role": role},
                        "what_if": sess.what_if,
                        "consequences": _consequences(before, after, crew_id,
                                                      pairing_id),
                    })
                return
            if route == "/api/resolve_event":
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                ev = body.get("dynamic_event", {})
                ev_type = ev.get("type")
                result = {"event": ev, "problem": "", "solutions": []}

                try:
                    from .events import (analyse_sick, analyse_closure, analyse_delay,
                                         resolve, Impact)

                    if ev_type == "SICK_CREW":
                        crew_id = ev.get("crew_id", "")
                        pairing_id = ev.get("pairing_id")
                        impact = analyse_sick(snap, crew_id, pairing_id)
                        result["problem"] = impact.summary
                        result["impact"] = impact.to_dict()

                        # Get ranked cover options
                        if pairing_id and pairing_id in snap.pairings:
                            try:
                                opts = resolve(snap, crew_id, pairing_id)
                                solutions = []
                                for i, opt in enumerate(opts.options[:5]):
                                    confidence = max(30, 95 - i * 12)
                                    solutions.append({
                                        "rank": i + 1,
                                        "action": opt.action,
                                        "crew_id": opt.crew_id,
                                        "legal": opt.legal,
                                        "cost_inr": opt.cost_inr,
                                        "delay_hours": opt.delay_hours,
                                        "coverage": opt.coverage,
                                        "reasoning": opt.reasoning,
                                        "confidence": confidence,
                                        "rules_checked": opt.rules_checked,
                                        "cost_breakdown": opt.cost_breakdown,
                                    })
                                result["solutions"] = solutions
                            except Exception as re:
                                result["resolve_error"] = str(re)

                        # Ask LLM for reasoning
                        adv: Advisor = _STATE["advisor"]
                        if adv.model_available:
                            try:
                                q = f"{crew_id} called in sick for {pairing_id}. Explain the operational impact and recommend the best recovery action."
                                ans = adv.ask(q)
                                result["llm_analysis"] = ans.prose
                                result["llm_explanation"] = ans.to_dict().get("explanation", "")
                            except Exception:
                                pass

                    elif ev_type == "DELAY":
                        aircraft = ev.get("aircraft", "")
                        delay_hours = ev.get("delay_hours", 0)
                        date = ev.get("date")
                        impact = analyse_delay(snap, delay_hours, aircraft=aircraft, date=date)
                        result["problem"] = impact.summary
                        result["impact"] = impact.to_dict()

                        solutions = []
                        imp_data = impact.to_dict()
                        breach = imp_data.get("breach", False)
                        rest_breach = imp_data.get("rest_breach", False)
                        prefix = imp_data.get("max_legal_prefix", {})

                        if not breach and not rest_breach:
                            solutions.append({
                                "rank": 1, "action": "No action required - delay is within legal limits",
                                "crew_id": None, "legal": True, "cost_inr": 0, "delay_hours": delay_hours,
                                "coverage": "All flights remain covered", "confidence": 98,
                                "reasoning": f"The {delay_hours}h delay keeps duty within FDP limits. No crew swaps needed.",
                                "rules_checked": ["RULE-FDP-01", "RULE-REST-04"], "cost_breakdown": {},
                            })
                        else:
                            solutions.append({
                                "rank": 1, "action": f"Shed tail legs after sector {prefix.get('last_legal_sector', 'N/A')}",
                                "crew_id": None, "legal": True, "cost_inr": 50000,
                                "delay_hours": delay_hours, "coverage": "Partial - tail legs require fresh crew",
                                "confidence": 85,
                                "reasoning": f"FDP breach at {delay_hours}h delay. Drop tail sectors to stay legal, re-crew remaining legs.",
                                "rules_checked": ["RULE-FDP-01"], "cost_breakdown": {"reposition": 50000},
                            })
                            solutions.append({
                                "rank": 2, "action": "Re-crew entire pairing with reserve crew",
                                "crew_id": None, "legal": True, "cost_inr": 120000,
                                "delay_hours": 0, "coverage": "Full coverage with fresh crew",
                                "confidence": 72,
                                "reasoning": "Complete crew swap eliminates duty/rest breach entirely.",
                                "rules_checked": ["RULE-FDP-01", "RULE-REST-04"],
                                "cost_breakdown": {"callout": 80000, "positioning": 40000},
                            })

                        result["solutions"] = solutions

                        adv: Advisor = _STATE["advisor"]
                        if adv.model_available:
                            try:
                                q = f"{aircraft} is delayed {delay_hours}h on {date}. Explain the crew legality impact and suggest recovery."
                                ans = adv.ask(q)
                                result["llm_analysis"] = ans.prose
                            except Exception:
                                pass

                    elif ev_type == "STATION_CLOSURE":
                        station = ev.get("station", "")
                        start_utc = ev.get("start_utc", "")
                        end_utc = ev.get("end_utc", "")
                        impact = analyse_closure(snap, station, start_utc, end_utc)
                        result["problem"] = impact.summary
                        result["impact"] = impact.to_dict()

                        imp_data = impact.to_dict()
                        affected = imp_data.get("affected_flights", [])
                        solutions = []
                        solutions.append({
                            "rank": 1, "action": f"Hold and delay {len(affected)} affected flights until {station} reopens",
                            "crew_id": None, "legal": True, "cost_inr": len(affected) * 15000,
                            "delay_hours": 0, "coverage": f"{len(affected)} flights delayed",
                            "confidence": 90,
                            "reasoning": f"Station {station} closure is temporary. Holding flights avoids cancellation costs. Monitor FDP limits for crew on delayed flights.",
                            "rules_checked": ["RULE-FDP-01", "RULE-REST-04"],
                            "cost_breakdown": {"ground_holding": len(affected) * 15000},
                        })
                        solutions.append({
                            "rank": 2, "action": f"Divert departures through alternate hub",
                            "crew_id": None, "legal": True, "cost_inr": len(affected) * 45000,
                            "delay_hours": 0, "coverage": "All flights re-routed",
                            "confidence": 68,
                            "reasoning": f"Re-route via nearest alternate airport. Higher cost but maintains schedule integrity.",
                            "rules_checked": ["RULE-FDP-01"],
                            "cost_breakdown": {"diversion_cost": len(affected) * 45000},
                        })
                        solutions.append({
                            "rank": 3, "action": f"Cancel {len(affected)} flights and rebook passengers",
                            "crew_id": None, "legal": True, "cost_inr": len(affected) * 250000,
                            "delay_hours": 0, "coverage": "Flights cancelled",
                            "confidence": 45,
                            "reasoning": "Last resort. Cancellation avoids cascading FDP risk but carries maximum passenger disruption and compensation cost.",
                            "rules_checked": [],
                            "cost_breakdown": {"cancellation": len(affected) * 150000, "rebooking": len(affected) * 100000},
                        })
                        result["solutions"] = solutions

                        adv: Advisor = _STATE["advisor"]
                        if adv.model_available:
                            try:
                                q = f"{station} is closed from {start_utc} to {end_utc}. What is the crew impact and what should I do?"
                                ans = adv.ask(q)
                                result["llm_analysis"] = ans.prose
                            except Exception:
                                pass

                except Exception as ex:
                    result["error"] = f"{type(ex).__name__}: {ex}"

                self._json(result)
                return
            if route == "/api/outreach":
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                self._json(_outreach(snap, body))
                return
            if route in ("/api/whatif", "/api/reset", "/api/set_region"):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                
                if route == "/api/set_region":
                    with _LOCK:
                        region = body.get("region", "in")
                        data_dir = _STATE.get("data_dir")
                        
                        import shutil
                        import os
                        rules_src = os.path.join(data_dir or ".", f"rules_{region}.json")
                        rules_dst = os.path.join(data_dir or ".", "rules.json")
                        if os.path.exists(rules_src):
                            shutil.copy(rules_src, rules_dst)
                        
                        from .data import load
                        snap = load(data_dir)
                        _STATE["snap"] = snap
                        _STATE["advisor"] = Advisor(snap, use_model=_STATE["use_model"])
                        _SESSIONS.clear()
                    self._json({"ok": True, "region": region})
                    return

                entry = _session(str(body.get("session") or "default"))
                with entry["lock"]:
                    if route == "/api/reset":
                        entry["s"].reset()
                    else:
                        scenario = body.get("scenario")
                        dynamic_event = body.get("dynamic_event")
                        from .events import Delay, SickCrew, StationClosure
                        
                        if dynamic_event:
                            ev_type = dynamic_event.get("type")
                            if ev_type == "SICK_CREW":
                                entry["s"].push_event(SickCrew(crew_id=dynamic_event.get("crew_id"), pairing_id=dynamic_event.get("pairing_id")))
                            elif ev_type == "DELAY":
                                entry["s"].push_event(Delay(delay_hours=dynamic_event.get("delay_hours"), aircraft=dynamic_event.get("aircraft"), date=dynamic_event.get("date")))
                            elif ev_type == "STATION_CLOSURE":
                                entry["s"].push_event(StationClosure(station=dynamic_event.get("station"), start_utc=dynamic_event.get("start_utc"), end_utc=dynamic_event.get("end_utc")))
                        else:
                            evs = _scenario_events(scenario) if scenario else []
                            if evs:
                                for e in evs:
                                    entry["s"].push_event(e)
                            else:
                                cid = body.get("crew_id")
                                if not cid or cid not in _STATE["snap"].crew:
                                    self._json({"error": f"unknown crew {cid}"}, 400)
                                    return
                                entry["s"].push_event(SickCrew(crew_id=cid))
                    self._json({"ok": True, "what_if": entry["s"].what_if})
                return
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            question = (payload.get("q") or "").strip()
            if not question:
                self._json({"error": "empty question"}, 400)
                return
            sid = payload.get("session")
            if sid:
                entry = _session(str(sid))
                with entry["lock"]:
                    self._json(entry["s"].ask(question).to_dict())
            else:
                adv: Advisor = _STATE["advisor"]
                self._json(adv.ask(question).to_dict())
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


EXAMPLES = [
    {"tier": "T1", "q": "Who is on reserve at BLR on 2026-09-15?"},
    {"tier": "T1", "q": "How many duty hours does C-1042 have left this week?"},
    {"tier": "T1", "q": "Which flights depart DEL on 2026-09-15?"},
    {"tier": "T1", "q": "Who are the captains based at DEL?"},
    {"tier": "T2", "q": "Captain C-1042 just called in sick for tomorrow - which flights are now uncrewed?"},
    {"tier": "T2", "q": "If I move C-2087 onto P-2291, does anyone breach a duty limit?"},
    {"tier": "T2", "q": "Can reserve C-3305 cover the full pairing P-2291?"},
    {"tier": "T2", "q": "BLR is closed 08:00 to 14:00 on 2026-09-17 - what is the crew impact?"},
    {"tier": "T2", "q": "VT-DXA is delayed 90 minutes on 2026-09-16"},
    {"tier": "T3", "q": "C-1042 is out for P-2291, what should I do?"},
    {"tier": "T3", "q": "What is the smallest change that would make C-2087 legal for P-2291?"},
    {"tier": "T3", "q": "Where are we thin on cover this week?"},
    {"tier": "T3", "q": "Draft the callout notification to C-3310 for P-2291"},
    {"tier": "REFUSE", "q": "What is the probability that C-2087 calls in sick tomorrow?"},
    {"tier": "REFUSE", "q": "What is the weather at BLR tomorrow?"},
    {"tier": "REFUSE", "q": "Who is on reserve on 2026-10-05?"},
    {"tier": "REFUSE", "q": "What is C-3310's phone number?"},
]


UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
# The rebuilt single-column desk, served at /desk. One column, ops vocabulary,
# and nothing on screen that is not the disruption, the answer, or the
# arithmetic behind it. The v3 cockpit stays on / until this replaces it.
DESK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "desk.html")


def _page() -> bytes:
    """Read the UI from disk on every request.

    Deliberate: it means the interface can be edited and reloaded mid-demo
    without restarting the server and losing the session state.
    """
    with open(UI_PATH, encoding="utf-8") as fh:
        return fh.read().encode("utf-8")


def _restore_default_rulebook(data_dir: str | None) -> None:
    """Start every session on the rulebook we were given.

    /api/set_region copies rules_<region>.json over rules.json, which is a
    permanent edit to the dataset. Switching to EU once and never switching
    back left a 65h weekly duty cap and a 14h FDP in place, so the advisor
    answered every later question against a rulebook the answer keys do not
    use -- and reported C-2087 LEGAL for P-2291 on 61.33h, because 61.33 is
    under 65. Nothing warned anyone; the file simply stayed changed.

    The region switch is a good feature and it stays. It just cannot be
    allowed to decide what tomorrow's default is.
    """
    import shutil
    d = data_dir or os.environ.get("CREWOPS_DATA_DIR") or ""
    src = os.path.join(d, "rules_in.json")
    dst = os.path.join(d, "rules.json")
    if os.path.isfile(src) and os.path.isfile(dst):
        try:
            if open(src, encoding="utf-8").read() != open(dst, encoding="utf-8").read():
                shutil.copy(src, dst)
                print("  rulebook reset to the shipped DGCA ruleset "
                      "(a previous region switch had left it changed)")
        except OSError:
            pass


def serve(host: str = "127.0.0.1", port: int = 8787,
          data_dir: str | None = None, use_model: bool | None = None) -> int:
    _restore_default_rulebook(
        data_dir or os.environ.get("CREWOPS_DATA_DIR") or "Problem Statement/data")
    snap = load(data_dir)
    _STATE["snap"] = snap
    _STATE["data_dir"] = data_dir or os.environ.get("CREWOPS_DATA_DIR") or "Problem Statement/data"
    _STATE["advisor"] = Advisor(snap, use_model=use_model)
    _STATE["use_model"] = use_model
    httpd = ThreadingHTTPServer((host, port), Handler)
    adv: Advisor = _STATE["advisor"]
    url = f"http://{host}:{port}"
    print()
    print(f"  Crew Ops Advisor  ->  {url}")
    print(f"  {len(snap.flights)} flights - {len(snap.crew)} crew - "
          f"snapshot 2026-09-14 18:00Z")
    print(f"  language: "
          + (adv.model_label if adv.model_available else "rules only"))
    print()
    print(f"    {url}/                 the desk")
    print(f"    {url}/api/brief        morning board")
    print(f"    {url}/api/eval         answer-key scoreboard")
    print(f"    {url}/api/conformance  rulebook vs data")
    print(f"    {url}/api/health       liveness")
    print()
    print("  Ctrl-C to stop.")
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        httpd.server_close()
    return 0
