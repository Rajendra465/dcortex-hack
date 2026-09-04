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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agent import Advisor
from .conformance import report as conformance_report
from .data import Snapshot, load
from .evaluate import run as run_eval
from .kernel import cover_fragility, latent_breaches, reserve_coverage_gaps

_STATE: dict[str, Any] = {}


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
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
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
        if self.path.split("?")[0] != "/api/ask":
            self._json({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            question = (payload.get("q") or "").strip()
            if not question:
                self._json({"error": "empty question"}, 400)
                return
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


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crew Ops Advisor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#eef1f4; --panel:#fff; --panel2:#e7edf1; --ink:#101720; --ink2:#4a5865;
  --ink3:#7b8894; --rule:#d2dae1; --accent:#0d5f7b; --accent-soft:#dfeef4;
  --ok:#286a45; --ok-soft:#e2f0e8; --warn:#9c281c; --warn-soft:#f9e5e2;
  --caution:#8b5a08; --caution-soft:#f8eeda;
  --d:'Archivo',system-ui,sans-serif; --b:'IBM Plex Sans',system-ui,sans-serif;
  --m:'IBM Plex Mono',ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#080c11; --panel:#131a22; --panel2:#1b242d; --ink:#e9eff4; --ink2:#98a7b5;
  --ink3:#75838f; --rule:#27333e; --accent:#52bcdf; --accent-soft:#102e3a;
  --ok:#63bd8b; --ok-soft:#112a1e; --warn:#ea7e6f; --warn-soft:#2f1713;
  --caution:#dea748; --caution-soft:#2c2411;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--b);font-size:15px;line-height:1.55}
a{color:var(--accent)}
header{position:sticky;top:0;z-index:9;background:var(--panel);border-bottom:1px solid var(--rule);padding:9px 20px;
  display:flex;gap:20px;flex-wrap:wrap;align-items:center;font-family:var(--m);font-size:11px;letter-spacing:.05em;color:var(--ink2)}
header b{color:var(--ink);font-weight:600}
.dot{width:7px;height:7px;border-radius:50%;background:var(--ok);display:inline-block;margin-right:6px}
.wrap{max-width:1180px;margin:0 auto;padding:20px;display:grid;grid-template-columns:250px 1fr;gap:20px}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
h1{font-family:var(--d);font-size:19px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-family:var(--d);font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--ink3);margin:0 0 10px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:16px;margin-bottom:16px}
.ex{display:block;width:100%;text-align:left;background:none;border:0;border-bottom:1px solid var(--rule);
  padding:8px 0;color:var(--ink2);font:inherit;font-size:13px;cursor:pointer}
.ex:last-child{border-bottom:0}
.ex:hover{color:var(--accent)}
.ex .t{font-family:var(--m);font-size:9.5px;color:var(--accent);margin-right:7px}
.ex .t.r{color:var(--caution)}
form{display:flex;gap:8px;margin-bottom:16px}
input{flex:1;padding:12px 14px;border:1px solid var(--rule);border-radius:4px;background:var(--panel);
  color:var(--ink);font:inherit}
input:focus{outline:2px solid var(--accent);outline-offset:-1px}
button.go{padding:12px 20px;border:0;border-radius:4px;background:var(--accent);color:#fff;font:inherit;
  font-weight:600;cursor:pointer}
button.go:disabled{opacity:.5}
.meta{font-family:var(--m);font-size:10.5px;color:var(--ink3);letter-spacing:.05em;margin-bottom:10px}
.prose{font-size:17px;font-weight:600;font-family:var(--d);line-height:1.4;margin-bottom:10px}
.verdict{font-family:var(--d);font-weight:700;font-size:24px;margin:14px 0 8px}
.verdict.no{color:var(--warn)} .verdict.yes{color:var(--ok)}
.issue{background:var(--warn-soft);border-left:3px solid var(--warn);padding:8px 12px;margin:6px 0;
  font-family:var(--m);font-size:12.5px;border-radius:0 3px 3px 0}
.refuse{border-left:3px solid var(--caution);background:var(--caution-soft);padding:14px 16px;border-radius:0 3px 3px 0}
.refuse .k{font-family:var(--m);font-size:10px;letter-spacing:.1em;color:var(--caution);margin-bottom:6px}
table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}
th{text-align:left;font-family:var(--m);font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink3);padding:7px 9px;border-bottom:1px solid var(--rule);background:var(--panel2)}
td{padding:7px 9px;border-bottom:1px solid var(--rule);vertical-align:top}
td.n{font-family:var(--m);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.m{font-family:var(--m);white-space:nowrap}
tr.pick td{background:var(--ok-soft)}
.scroll{overflow-x:auto}
details{margin-top:12px;border-top:1px solid var(--rule);padding-top:10px}
summary{cursor:pointer;font-family:var(--m);font-size:11px;color:var(--ink3);letter-spacing:.06em}
pre{background:var(--panel2);padding:12px;border-radius:3px;overflow-x:auto;font-family:var(--m);font-size:11.5px;margin:8px 0 0}
.pill{display:inline-block;font-family:var(--m);font-size:9.5px;padding:2px 6px;border-radius:2px;
  border:1px solid currentColor;margin-left:6px}
.pill.c{color:var(--accent)} .pill.mdl{color:var(--caution)}
.bar{height:8px;background:var(--panel2);border:1px solid var(--rule);border-radius:2px;position:relative;margin:8px 0 12px;overflow:hidden}
.bar i{position:absolute;top:0;bottom:0;left:0;background:var(--accent);opacity:.5}
.bar u{position:absolute;top:0;bottom:0;background:var(--warn);opacity:.7}
.bar s{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink)}
.err{color:var(--warn);font-family:var(--m);font-size:12px}
.small{font-size:12.5px;color:var(--ink2)}
</style></head><body>
<header>
  <span><b>dCORTEX AIR</b> &middot; CREW OPS ADVISOR</span>
  <span>SNAPSHOT <b>2026-09-14 18:00Z</b> &middot; UTC</span>
  <span id="health"><span class="dot"></span>loading&hellip;</span>
</header>
<div class="wrap">
  <aside>
    <div class="card">
      <h2>Try one</h2>
      <div id="examples" class="small">loading&hellip;</div>
    </div>
    <div class="card">
      <h2>Desk</h2>
      <div class="small">
        <a href="/api/brief" target="_blank">Morning brief</a><br>
        <a href="/api/eval" target="_blank">Scoreboard</a><br>
        <a href="/api/conformance" target="_blank">Conformance report</a><br>
        <a href="/api/health" target="_blank">Health</a>
      </div>
    </div>
  </aside>
  <main>
    <div class="card">
      <h1>Ask the desk</h1>
      <div class="small" style="margin-bottom:12px">The model translates. The kernel decides. Every number below was computed, not written.</div>
      <form id="f"><input id="q" autocomplete="off" placeholder="Captain C-1042 just called in sick for tomorrow &mdash; which flights are now uncrewed?" autofocus><button class="go" id="go">Ask</button></form>
      <div id="out"></div>
    </div>
  </main>
</div>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s === null || s === undefined ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const inr = n => '₹' + Number(n).toLocaleString('en-IN');

fetch('/api/health').then(r=>r.json()).then(d=>{
  $('#health').innerHTML = '<span class="dot"></span>' + d.flights + ' FLIGHTS · ' + d.crew +
    ' CREW · ' + (d.model ? esc(d.model) : 'RULES ONLY');
});
fetch('/api/examples').then(r=>r.json()).then(list=>{
  $('#examples').innerHTML = list.map(e =>
    '<button class="ex" data-q="'+esc(e.q)+'"><span class="t'+(e.tier==='REFUSE'?' r':'')+'">'+
    e.tier+'</span>'+esc(e.q)+'</button>').join('');
  document.querySelectorAll('.ex').forEach(b => b.onclick = () => { $('#q').value = b.dataset.q; ask(); });
});

$('#f').onsubmit = e => { e.preventDefault(); ask(); };

async function ask(){
  const q = $('#q').value.trim(); if(!q) return;
  $('#go').disabled = true; $('#out').innerHTML = '<div class="meta">thinking…</div>';
  try{
    const r = await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q:q})});
    render(await r.json());
  }catch(err){ $('#out').innerHTML = '<div class="err">'+esc(err)+'</div>'; }
  $('#go').disabled = false;
}

function num(ev, prefix){
  const f = (ev||[]).find(x => x.key.indexOf(prefix) === 0);
  return f ? Number(f.value) : null;
}

function render(d){
  if(d.error){ $('#out').innerHTML = '<div class="err">'+esc(d.error)+'</div>'; return; }
  let h = '';
  if(d.answer_type === 'refusal'){
    h += '<div class="refuse"><div class="k">'+esc(d.kind)+'</div>';
    h += '<div class="prose">'+esc(d.message)+'</div>';
    if(d.have) h += '<div class="small"><b>What I do have:</b> '+esc(d.have)+'</div>';
    if(d.would_need) h += '<div class="small"><b>What it would need:</b> '+esc(d.would_need)+'</div>';
    (d.suggestions||[]).forEach(s => h += '<div class="small">Try: '+esc(s)+'</div>');
    h += '</div>';
    $('#out').innerHTML = h; return;
  }

  const p = d.result || {}, tool = d.plan ? d.plan.tool : '?';
  const computed = d.explanation_source !== 'model';
  h += '<div class="prose">'+esc(d.explanation)+'</div>';
  h += '<div class="meta">T'+d.tier+' · '+esc(tool)+' · '+Math.round(d.elapsed_ms)+' ms'+
       '<span class="pill '+(computed?'c':'mdl')+'">'+(computed?'COMPUTED':'MODEL-DRAFTED')+'</span>'+
       '<span class="pill c">parsed by '+esc(d.plan?d.plan.parsed_by:'-')+'</span></div>';
  ((d.resolved&&d.resolved.assumptions)||[]).forEach(a => h += '<div class="meta">assumed '+esc(a)+'</div>');

  if(tool === 'check_legality'){
    h += '<div class="verdict '+(p.legal?'yes':'no')+'">'+(p.legal?'LEGAL':'NOT LEGAL')+'</div>';
    (p.issues||[]).forEach(i => h += '<div class="issue">'+esc(i)+'</div>');
    if(p.consequence) h += '<div class="small">'+esc(p.consequence)+'</div>';
    const tot = num(d.evidence,'duty_7d_total'), cap = num(d.evidence,'duty_cap_hours');
    if(tot && cap && tot > 0){
      const w = Math.min(100, cap/tot*100);
      h += '<div class="bar"><i style="width:'+w+'%"></i><u style="left:'+w+'%;width:'+(100-w)+'%"></u><s style="left:'+w+'%"></s></div>';
    }
    if((p.non_binding||[]).length) h += '<div class="meta">'+esc(p.non_binding.join(', '))+': EVALUATED, NON-BINDING ON THIS DATA</div>';
  }

  if(tool === 'rank_cover_options'){
    h += '<div class="scroll"><table><thead><tr><th>#</th><th>crew</th><th>cost</th><th>tie</th><th>reach</th><th>action</th></tr></thead><tbody>';
    (p.options||[]).forEach(o => {
      h += '<tr'+(o.rank===1?' class="pick"':'')+'><td class="m">'+o.rank+'</td><td class="m">'+esc(o.crew_id||'—')+
        '</td><td class="n">'+inr(o.cost_inr)+'</td><td class="m">'+o.tie_band+'</td><td class="m">'+
        (o.reachability_minutes?o.reachability_minutes+'m':'—')+'</td><td>'+esc(o.action)+'</td></tr>';
    });
    h += '</tbody></table></div>';
    const ex = p.excluded_candidates||[];
    if(ex.length){
      h += '<h2 style="margin-top:16px">Ruled out ('+ex.length+') — the first question a controller asks</h2>';
      h += '<div class="scroll"><table><tbody>';
      ex.forEach(e => h += '<tr><td class="m">'+esc(e.crew_id)+'</td><td>'+esc(e.reason)+'</td></tr>');
      h += '</tbody></table></div>';
    }
  }

  if(tool === 'simulate_crew_unavailable'){
    h += '<div class="scroll"><table><thead><tr><th>date</th><th>uncrewed flights</th><th>legs</th></tr></thead><tbody>';
    Object.keys(p.uncovered_by_day||{}).sort().forEach(dt => {
      const fl = p.uncovered_by_day[dt];
      h += '<tr><td class="m">'+esc(dt)+'</td><td class="m">'+fl.map(f=>esc(f.split('-')[0])).join(', ')+'</td><td class="n">'+fl.length+'</td></tr>';
    });
    h += '</tbody></table></div><div class="meta">'+p.passengers_at_risk_day1+' SEATS AT RISK ON DAY ONE'+
         (p.overnight_station?' · OVERNIGHTS AT '+esc(p.overnight_station):'')+'</div>';
  }

  if(tool === 'simulate_station_closure'){
    h += '<div class="scroll"><table><thead><tr><th>flight</th><th>pairing</th><th>min delay</th><th>FDP / limit</th><th>crew</th></tr></thead><tbody>';
    (p.per_flight_assessment||[]).forEach(r =>
      h += '<tr'+(r.feasible?'':' class="pick"')+'><td class="m">'+esc(r.flight_no)+'</td><td class="m">'+esc(r.pairing_id)+'</td><td class="n">+'+r.min_delay_hours+'h</td><td class="n">'+
        r.crew_fdp_after_delay+' / '+r.fdp_limit+'</td><td class="m">'+(r.feasible?'ok':'EXCEEDS FDP')+'</td></tr>');
    h += '</tbody></table></div>';
  }

  if(tool === 'simulate_delay' && p.breach){
    h += '<div class="verdict no">FDP BREACH</div><div class="issue">'+esc(p.breach_detail)+'</div>';
    const pre = p.max_legal_prefix||{};
    if(pre.legs) h += '<div class="small">Original crew can still operate '+pre.legs.length+' of '+p.sectors+
      ' legs (FDP '+pre.fdp+'h vs '+pre.limit+'h). Re-crew: '+(pre.legs_to_shed||[]).map(x=>esc(x.split('-')[0])).join(', ')+'.</div>';
  }

  if(tool === 'minimal_repair' && !p.already_legal){
    (p.repairs||[]).forEach(r => h += '<div class="issue"><b>'+esc(r.rule)+'</b> — short by '+
      (r.shortfall_hours === null ? '—' : r.shortfall_hours)+'h<br>'+esc(r.lever)+'</div>');
  }

  if(tool === 'cover_fragility'){
    h += '<div class="scroll"><table><thead><tr><th>pairing</th><th>date</th><th>role</th><th>legal covers</th><th>who</th></tr></thead><tbody>';
    (p.rows||[]).slice(0,10).forEach(r => h += '<tr'+(r.legal_covers<=1?' class="pick"':'')+'><td class="m">'+esc(r.pairing_id)+
      '</td><td class="m">'+esc(r.date)+'</td><td>'+esc(r.role)+'</td><td class="n">'+r.legal_covers+
      '</td><td class="m">'+esc((r.candidates||[]).join(', ')||'none')+'</td></tr>');
    h += '</tbody></table></div>';
  }

  if(tool === 'notification_packet'){
    (p.days||[]).forEach(dy => h += '<div class="small"><b>'+esc(dy.date)+'</b> report '+esc(dy.report_utc)+
      ' at '+esc(dy.report_station)+' · '+dy.flights.map(esc).join(', ')+
      (dy.overnight_station?' · overnight '+esc(dy.overnight_station):'')+'</div>');
    h += '<div class="meta">ACKNOWLEDGE BY '+esc(p.acknowledgement_deadline_utc)+'</div>';
  }

  if((d.evidence||[]).length){
    h += '<details><summary>Evidence ledger — every number the engine touched ('+d.evidence.length+')</summary>';
    h += '<div class="scroll"><table><thead><tr><th>fact</th><th>value</th><th>source / derivation</th></tr></thead><tbody>';
    d.evidence.forEach(f => h += '<tr><td class="m">'+esc(f.key)+'</td><td class="n">'+esc(f.value)+' '+esc(f.unit||'')+
      '</td><td class="small">'+esc(f.derivation||f.source||'')+'</td></tr>');
    h += '</tbody></table></div></details>';
  }
  h += '<details><summary>Raw JSON</summary><pre>'+esc(JSON.stringify(d,null,2))+'</pre></details>';
  $('#out').innerHTML = h;
}
</script></body></html>
"""


def serve(host: str = "127.0.0.1", port: int = 8787,
          data_dir: str | None = None, use_model: bool | None = None) -> int:
    snap = load(data_dir)
    _STATE["snap"] = snap
    _STATE["advisor"] = Advisor(snap, use_model=use_model)
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
