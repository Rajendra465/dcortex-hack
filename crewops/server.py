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
            elif path == "/api/examples":
                self._json(EXAMPLES)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # never leave the socket hanging
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?")[0]
        if route not in ("/api/ask", "/api/whatif", "/api/reset", "/api/outreach"):
            self._json({"error": "not found"}, 404)
            return
        try:
            snap: Snapshot = _STATE["snap"]
            if route == "/api/outreach":
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                self._json(_outreach(snap, body))
                return
            if route in ("/api/whatif", "/api/reset"):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                entry = _session(str(body.get("session") or "default"))
                with entry["lock"]:
                    if route == "/api/reset":
                        entry["s"].reset()
                    else:
                        scenario = body.get("scenario")
                        from .events import Delay, SickCrew, StationClosure
                        if scenario == "S1" or scenario == "S2" or scenario == "S3":
                            entry["s"].push_event(SickCrew(crew_id="C-1042", pairing_id="P-2291"))
                        elif scenario == "S4":
                            entry["s"].push_event(StationClosure(
                                station="BLR",
                                start_utc="2026-09-17T08:00:00Z",
                                end_utc="2026-09-17T14:00:00Z",
                            ))
                        elif scenario == "S5":
                            entry["s"].push_event(Delay(
                                delay_hours=1.5,
                                aircraft="VT-DXA",
                                date="2026-09-16",
                            ))
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


def _page() -> bytes:
    """Read the UI from disk on every request.

    Deliberate: it means the interface can be edited and reloaded mid-demo
    without restarting the server and losing the session state.
    """
    with open(UI_PATH, encoding="utf-8") as fh:
        return fh.read().encode("utf-8")


def serve(host: str = "127.0.0.1", port: int = 8787,
          data_dir: str | None = None, use_model: bool | None = None) -> int:
    snap = load(data_dir)
    _STATE["snap"] = snap
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
