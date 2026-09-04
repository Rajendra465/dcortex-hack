"""The desk. A keyboard-first interface for someone under time pressure.

Design decisions worth knowing:
  * The verdict is never streamed. Prose can stream; a legal/illegal call
    appears all at once or not at all.
  * Ruled-out candidates are shown by DEFAULT, not hidden behind a flag. A
    controller's first question is always "why not someone else?", and an
    explanation only reduces over-reliance when checking it is cheaper than
    redoing the work.
  * Every figure carries its provenance: `computed` means the rules engine
    produced it, `model-drafted` means the language model wrote the sentence.
  * Legal / illegal / already-illegal / unverifiable are four states, and none
    of them is encoded by colour alone.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from .agent import Advisor, Answer
from .data import assert_shape, load
from .kernel import cover_fragility, latent_breaches, reserve_coverage_gaps

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False

BANNER = "dCortex Air · Crew Ops Advisor"


def _plain(s: str) -> str:
    return re.sub(r"\[/?[a-z0-9 ._#]+\]", "", s)


class Out:
    """Thin wrapper so the CLI works with or without rich installed."""

    def __init__(self) -> None:
        self.c = Console(highlight=False) if _RICH else None

    def p(self, s: str = "") -> None:
        if self.c:
            self.c.print(s)
        else:
            print(_plain(s))

    def rule(self, s: str = "") -> None:
        if self.c:
            self.c.rule(s, style="dim")
        else:
            print("-" * 70 + (f" {s}" if s else ""))

    def table(self, cols: list[str], rows: list[list], title: str = "") -> None:
        if not rows:
            return
        if self.c:
            t = Table(box=box.SIMPLE_HEAD, title=title or None,
                      title_style="dim", title_justify="left",
                      header_style="dim")
            for col in cols:
                t.add_column(col, overflow="fold")
            for r in rows:
                t.add_row(*[str(x) for x in r])
            self.c.print(t)
        else:
            if title:
                print("  " + title)
            print("  " + " | ".join(cols))
            for r in rows:
                print("  " + " | ".join(str(x) for x in r))


# --------------------------------------------------------------------------
# rendering an answer
# --------------------------------------------------------------------------


def render_answer(out: Out, a: Answer, show_evidence: bool = False) -> None:
    if a.refusal:
        r = a.refusal
        head = {
            "PARSE_FAIL": "[yellow]I didn't understand that[/yellow]",
            "CLARIFY": "[cyan]I need one more detail[/cyan]",
        }.get(r.kind, "[yellow]I can't answer that reliably[/yellow]")
        out.p()
        out.p(f"  {head}   [dim]({r.kind})[/dim]")
        out.p(f"  {r.message}")
        if r.have:
            out.p(f"  [dim]What I do have:[/dim] {r.have}")
        if r.would_need:
            out.p(f"  [dim]What it would need:[/dim] {r.would_need}")
        for s in r.suggestions:
            out.p(f"  [dim]Try:[/dim] {s}")
        out.p()
        return

    p = a.payload or {}
    tool = a.plan.tool if a.plan else "?"
    src = "computed" if a.provenance != "model" else "model-drafted"

    out.p()
    out.p(f"  [bold]{a.prose}[/bold]")
    out.p(f"  [dim]T{a.tier} · {tool} · {a.ms:.0f} ms · {src}"
          + (f" · parsed by {a.plan.source}" if a.plan else "") + "[/dim]")
    for note in a.assumptions:
        out.p(f"  [yellow]assumed[/yellow] {note}")

    if tool == "check_legality":
        verdict = "LEGAL" if p["legal"] else "NOT LEGAL"
        mark = "OK" if p["legal"] else "XX"
        out.p()
        out.p(f"  {mark}  [bold]{verdict}[/bold]")
        for issue in p.get("issues", []):
            out.p(f"      [red]-[/red] {issue}")
        if p.get("consequence"):
            out.p(f"      {p['consequence']}")
        if p.get("non_binding"):
            out.p(f"  [dim]{', '.join(p['non_binding'])}: evaluated, "
                  f"non-binding on this data[/dim]")

    if tool == "rank_cover_options":
        rows = []
        for o in p["options"][:8]:
            extra = []
            if o.get("reachability_minutes"):
                extra.append(f"{o['reachability_minutes']}m")
            if o.get("delay_hours"):
                extra.append(f"+{o['delay_hours']}h delay")
            rows.append([o["rank"], o["crew_id"] or "--", f"{o['cost_inr']:,}",
                         o["tie_band"], " ".join(extra), o["action"][:44]])
        out.p()
        out.table(["#", "crew", "INR", "tie", "notes", "action"], rows,
                  title="Options — ranked by cost; equal cost means equal rank")

        excl = p.get("excluded_candidates", [])
        if excl:
            out.p()
            out.table(["ruled out", "why"],
                      [[e["crew_id"], e["reason"][:86]] for e in excl[:6]],
                      title=f"Ruled out ({len(excl)}) — the first question a "
                            f"controller asks")
            if len(excl) > 6:
                out.p(f"  [dim]... and {len(excl) - 6} more (--json for all)[/dim]")

    if tool == "simulate_crew_unavailable":
        rows = [[d, ", ".join(f.split("-")[0] for f in fl), len(fl)]
                for d, fl in sorted(p["uncovered_by_day"].items())]
        out.p()
        out.table(["date", "uncrewed flights", "legs"], rows)
        out.p(f"  [dim]{p['passengers_at_risk_day1']} seats at risk on day one"
              + (f" · overnights at {p['overnight_station']}"
                 if p.get("overnight_station") else "") + "[/dim]")

    if tool == "simulate_station_closure":
        rows = [[r["flight_no"], r["pairing_id"], f"+{r['min_delay_hours']}h",
                 f"{r['crew_fdp_after_delay']}/{r['fdp_limit']}",
                 "ok" if r["feasible"] else "EXCEEDS FDP"]
                for r in p["per_flight_assessment"][:10]]
        out.p()
        out.table(["flight", "pairing", "min delay", "FDP/limit", "crew"], rows)

    if tool == "simulate_delay" and p.get("breach"):
        out.p()
        out.p(f"  [red]XX {p['breach_detail']}[/red]")
        pre = p.get("max_legal_prefix") or {}
        if pre.get("legs"):
            shed = ", ".join(x.split("-")[0] for x in pre["legs_to_shed"]) or "none"
            out.p(f"  Original crew can still operate {len(pre['legs'])} of "
                  f"{p['sectors']} legs (FDP {pre['fdp']}h vs {pre['limit']}h). "
                  f"Re-crew: {shed}.")

    if tool == "minimal_repair" and not p.get("already_legal"):
        out.p()
        for r in p.get("repairs", []):
            gap = (f"{r['shortfall_hours']}h" if r["shortfall_hours"] is not None
                   else "--")
            out.p(f"  [cyan]{r['rule']}[/cyan]  short by {gap}")
            out.p(f"      {r['lever']}")

    if tool == "cover_fragility":
        out.p()
        out.table(["pairing", "date", "role", "legal covers", "who"],
                  [[r["pairing_id"], r["date"], r["role"], r["legal_covers"],
                    ", ".join(r["candidates"]) or "none"] for r in p["rows"][:8]],
                  title="Thinnest cover in the week")

    if tool == "notification_packet":
        out.p()
        for d in p["days"]:
            out.p(f"  {d['date']}  report {d['report_utc']} at "
                  f"{d['report_station']} · {', '.join(d['flights'])}"
                  + (f" · overnight {d['overnight_station']}"
                     if d["overnight_station"] else ""))
        out.p(f"  [dim]acknowledge by {p['acknowledgement_deadline_utc']}[/dim]")

    if show_evidence and a.ledger.facts:
        out.p()
        out.table(["fact", "value", "source / derivation"],
                  [[f.key, f"{f.value}{(' ' + f.unit) if f.unit else ''}",
                    (f.derivation or f.source)[:52]]
                   for f in a.ledger.facts[:24]],
                  title="Evidence ledger — every number the engine touched")
    out.p()


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> int:
    out = Out()
    snap = load(args.data)
    adv = Advisor(snap, use_model=not args.no_model)
    a = adv.ask(" ".join(args.question))
    if args.json:
        print(json.dumps(a.to_dict(), indent=2, default=str))
        return 0
    render_answer(out, a, show_evidence=args.evidence)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    out = Out()
    snap = load(args.data)
    adv = Advisor(snap, use_model=not args.no_model)

    out.p()
    out.p(f"  [bold]{BANNER}[/bold]")
    out.p(f"  [dim]snapshot 2026-09-14 18:00Z · all times UTC · "
          f"{len(snap.flights)} flights · {len(snap.crew)} crew[/dim]")
    out.p("  [dim]language parsing: "
          + (f"rules + {adv.model_label}" if adv.model_available
             else "rules only (no model key — answers are still exact)")
          + "[/dim]")
    out.p("  [dim]:brief  :evidence  :quit[/dim]")

    for b in latent_breaches(snap):
        out.p()
        out.p(f"  [red]ALREADY ILLEGAL[/red] {b['crew_id']} on {b['date']} "
              f"({b['pairing_id']}) — {b['rule']}: {b['detail']}")
    out.p()

    evidence = False
    while True:
        try:
            q = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            out.p()
            return 0
        if not q:
            continue
        if q in (":q", ":quit", "exit", "quit"):
            return 0
        if q == ":evidence":
            evidence = not evidence
            out.p(f"  [dim]evidence ledger {'on' if evidence else 'off'}[/dim]")
            continue
        if q == ":brief":
            cmd_brief(argparse.Namespace(data=args.data))
            continue
        render_answer(out, adv.ask(q), show_evidence=evidence)


def _runs(hours: list[int]) -> str:
    if not hours:
        return "--"
    out, start, prev = [], hours[0], hours[0]
    for h in hours[1:] + [None]:
        if h is not None and h == prev + 1:
            prev = h
            continue
        out.append(f"{start:02d}:00-{prev:02d}:59")
        if h is not None:
            start = prev = h
    return ", ".join(out)


def cmd_brief(args: argparse.Namespace) -> int:
    """The morning board: what is already broken, and where we are thin."""
    out = Out()
    snap = load(args.data)
    out.p()
    out.rule("MORNING BRIEF")

    breaches = latent_breaches(snap)
    out.p()
    if breaches:
        out.p("  [red]Already illegal as rostered[/red]")
        for b in breaches:
            out.p(f"      {b['crew_id']} ({b['role']}) on {b['date']} · "
                  f"{b['pairing_id']} · {b['rule']} · {b['detail']}")
    else:
        out.p("  [green]No already-illegal assignments.[/green]")

    frag = cover_fragility(snap)
    crit = [r for r in frag if r["legal_covers"] <= 1]
    out.p()
    out.table(["pairing", "date", "role", "covers", "who"],
              [[r["pairing_id"], r["date"], r["role"], r["legal_covers"],
                ", ".join(r["candidates"]) or "none"] for r in frag[:6]],
              title=f"Thinnest cover ({len(crit)} single point(s) of failure)")

    gaps = reserve_coverage_gaps(snap)
    runs: dict[tuple[str, str], list[int]] = {}
    for g in gaps:
        runs.setdefault((g["rank"], g["aircraft_type"]), []).append(g["hour_utc"])
    out.p()
    out.table(["rank", "type", "hours with no standby (UTC)"],
              [[k[0], k[1], _runs(sorted(v))] for k, v in sorted(runs.items())],
              title="Standby coverage gaps")
    out.p()
    out.p("  [dim]A duty-hour watchlist is deliberately absent: peak 7-day "
          "utilisation here is 42.51h against a 60h cap, so that panel would "
          "be empty.[/dim]")
    out.p()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from . import evaluate
    rep = evaluate.run(data_dir=args.data)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep.to_dict(), fh, indent=2, default=str)
        print(f"wrote {args.json}")
    print(evaluate.render(rep, verbose=args.verbose))
    return 1 if rep.count(evaluate.WRONG) else 0


def cmd_status(args: argparse.Namespace) -> int:
    from .tools import REGISTRY
    out = Out()
    snap = load(args.data)
    out.p()
    out.rule("STATUS")
    for line in assert_shape(snap):
        out.p(f"  {line}")
    adv = Advisor(snap)
    out.p(f"  {'ok  ' if adv.model_available else '--  '}language model: "
          + (adv.model_label if adv.model_available
             else "none configured (set NVIDIA_API_KEY or ANTHROPIC_API_KEY); "
                  "rules parser active"))
    out.p(f"  ok   {len(REGISTRY)} typed capabilities registered")
    out.p()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve
    return serve(host=args.host, port=args.port, data_dir=args.data,
                 use_model=not args.no_model)


def cmd_conformance(args: argparse.Namespace) -> int:
    """Where the written rulebook and the shipped data disagree."""
    from .conformance import render, report
    snap = load(args.data)
    if args.json:
        print(json.dumps(report(snap), indent=2, default=str))
        return 0
    print(render(snap))
    return 0


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="crewops", description=BANNER)
    ap.add_argument("--data", help="dataset directory (default: bundled)")
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question", nargs="+")
    a.add_argument("--json", action="store_true", help="machine-readable answer")
    a.add_argument("--evidence", action="store_true", help="show the ledger")
    a.add_argument("--no-model", action="store_true", help="rules parser only")
    a.set_defaults(fn=cmd_ask)

    c = sub.add_parser("chat", help="interactive desk")
    c.add_argument("--no-model", action="store_true")
    c.set_defaults(fn=cmd_chat)

    b = sub.add_parser("brief", help="the morning board")
    b.set_defaults(fn=cmd_brief)

    e = sub.add_parser("eval", help="run the answer-key regression")
    e.add_argument("--json", help="write a JSON summary here")
    e.add_argument("--verbose", action="store_true")
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("status", help="dataset and engine health")
    s.set_defaults(fn=cmd_status)

    sv = sub.add_parser("serve", help="the web desk on localhost")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8787)
    sv.add_argument("--no-model", action="store_true")
    sv.set_defaults(fn=cmd_serve)

    cf = sub.add_parser("conformance", help="rulebook vs data disagreements")
    cf.add_argument("--json", action="store_true")
    cf.set_defaults(fn=cmd_conformance)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 0
    try:
        return args.fn(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
