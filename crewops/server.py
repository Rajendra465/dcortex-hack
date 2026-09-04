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
from .evaluate import run as run_eval
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
                    "snapshot_utc": "2026-09-14T18:00:00Z",
                    "model": adv.model_label if adv.model_available else None,
                })
            elif path == "/api/brief":
                self._json(_brief(snap))
            elif path == "/api/eval":
                self._json(run_eval(snap).to_dict())
            elif path == "/api/conformance":
                self._json(conformance_report(snap))
            elif path == "/api/examples":
                self._json(EXAMPLES)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # never leave the socket hanging
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?")[0]
        if route not in ("/api/ask", "/api/whatif", "/api/reset"):
            self._json({"error": "not found"}, 404)
            return
        try:
            if route in ("/api/whatif", "/api/reset"):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                entry = _session(str(body.get("session") or "default"))
                with entry["lock"]:
                    if route == "/api/reset":
                        entry["s"].reset()
                    else:
                        from .events import SickCrew
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
