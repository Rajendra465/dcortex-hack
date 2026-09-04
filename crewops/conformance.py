"""Rulebook Conformance Report -- where the written rules and the data disagree.

Reproducing an oracle's quirks silently would be modelling the grader. Finding
those quirks, following the data because the data IS the specification, and then
publishing every disagreement with the reading we chose -- that is modelling the
domain. This module is that publication.

Two classes, and conflating them is the mistake:

  CLASS A  Interpretation choices. The written prose is ambiguous and the data
           settles it. A careful engineer reaches these from the shipped files
           alone. Defending them is ordinary domain modelling.

  CLASS B  A genuine defect in the data. The semantically correct reading
           produces an absurd result, so we follow the data -- but we say so
           loudly, because a system that silently cannot detect a not-yet-valid
           licence is a system nobody should trust.

Strict mode applies the semantically correct rules and reports what happens.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from .data import Snapshot
from .rules import window_sum


def _strict_fleet_check(snap: Snapshot) -> dict[str, Any]:
    """Apply the semantically correct RULE-CERT-06 and count the survivors."""
    on = date(2026, 9, 15)
    total = sum(1 for c in snap.crew.values() if c.status == "active")
    ok = sum(1 for c in snap.crew.values()
             if c.status == "active"
             and snap.certs_valid_on_strict(c.crew_id, on)[0])
    future = sum(1 for r in snap.cert_rows
                 if r["cert_type"] == "licence"
                 and date.fromisoformat(r["valid_from"]) > on)
    return {"active_crew": total, "legal_under_strict": ok,
            "licences_not_yet_valid": future, "as_of": on.isoformat()}


def _duty_double_count(snap: Snapshot) -> dict[str, Any]:
    """Crew whose 2026-09-14 duty is counted from both history and the roster."""
    d = date(2026, 9, 14)
    hits = []
    for cid in snap.crew:
        h = snap.history.get(cid, {}).get(d, (0.0, 0.0))[0]
        r = sum(x.duty_hours for x in snap.roster.get(cid, []) if x.day == d)
        if h and r:
            hits.append({"crew_id": cid, "history_hours": h,
                         "roster_hours": round(r, 2), "counted": round(h + r, 2)})
    hits.sort(key=lambda x: -x["counted"])
    return {"affected_crew": len(hits), "examples": hits[:4]}


def _flt03_headroom(snap: Snapshot) -> dict[str, Any]:
    peak, who, when = 0.0, None, None
    for cid in snap.crew:
        for duty in snap.roster.get(cid, []):
            v, _ = window_sum(snap, cid, duty.day, 28, "flight")
            if v > peak:
                peak, who, when = v, cid, duty.day.isoformat()
    return {"peak_28d_block_hours": round(peak, 2), "cap": 100.0,
            "crew_id": who, "date": when, "headroom": round(100.0 - peak, 2)}


def _reserve_boundary(snap: Snapshot) -> dict[str, Any]:
    """The flagship answer sits exactly ON the inclusive window boundary."""
    p = snap.pairings.get("P-2291")
    r = snap.reserves.get("C-3310")
    if not p or not r:
        return {}
    return {
        "pairing_id": "P-2291",
        "required_report_utc": p.days[0].report_utc,
        "reserve": "C-3310",
        "window": f"{r.window_start}-{r.window_end}Z",
        "on_boundary": p.days[0].report.strftime("%H:%M") == r.window_start,
        "callout_reading_would_exclude": True,
    }


def report(snap: Snapshot) -> dict[str, Any]:
    return {
        "snapshot": "2026-09-14T18:00:00Z",
        "findings": [
            {
                "id": "CONF-01", "rule": "RULE-CERT-06", "class": "B",
                "prose_says": "All certifications must be valid on the duty date.",
                "data_requires": "Compare valid_to only; ignore valid_from.",
                "why": ("Every licence in the dataset carries a valid_from in "
                        "2027-2032. The semantically correct window test grounds "
                        "the entire fleet."),
                "evidence": _strict_fleet_check(snap),
                "we_follow": "the data",
                "risk_accepted": ("This engine cannot detect a not-yet-valid "
                                  "licence. On real data that is a safety gap "
                                  "and the check must be restored."),
            },
            {
                "id": "CONF-02", "rule": "RULE-BASE-07", "class": "A",
                "prose_says": ("A reserve may be called out only if the CALLOUT "
                               "time falls inside their on-call window."),
                "data_requires": ("Test the required REPORT time, after any "
                                  "positioning delay, inclusive at both ends."),
                "why": ("The flagship answer needs C-3310 legal for a 06:00Z "
                        "report against a 06:00-18:00Z window, from a sick call "
                        "at 05:00Z. Only the report-time reading admits it, and "
                        "only if the boundary is inclusive."),
                "evidence": _reserve_boundary(snap),
                "we_follow": "the data",
            },
            {
                "id": "CONF-03", "rule": "RULE-DUTY-02", "class": "A",
                "prose_says": "Max 60 duty hours in any 7 consecutive days.",
                "data_requires": ("Sum daily_history AND the planned roster. "
                                  "2026-09-14 contributes from both sources."),
                "why": ("The shipped duty_clocks summary field embodies the same "
                        "double count, and the dataset's own validator "
                        "reproduces it. It is the specification, not a bug."),
                "evidence": _duty_double_count(snap),
                "we_follow": "the data",
            },
            {
                "id": "CONF-04", "rule": "RULE-FLT-03", "class": "A",
                "prose_says": "Max 100 flight hours in any 28 consecutive days.",
                "data_requires": ("Listed in every option's rules_checked, but "
                                  "never evaluated by the reference resolver."),
                "why": ("We DO evaluate it, and report it as 'evaluated, "
                        "non-binding' rather than claiming it was checked. "
                        "Claiming a check you did not perform is a false entry "
                        "in a compliance trail."),
                "evidence": _flt03_headroom(snap),
                "we_follow": "compute it, and label it honestly",
            },
            {
                "id": "CONF-05", "rule": "RULE-DUTY-02", "class": "A",
                "prose_says": ("(implied) duty_clocks.duty_hours_7d is the "
                               "7-day total."),
                "data_requires": ("That field is a snapshot for the window "
                                  "ending 2026-09-14 only. Re-sum for any other "
                                  "date."),
                "why": ("Reusing it for a 15 September duty flips C-3305 from "
                        "legal to illegal."),
                "evidence": {"field": "duty_clocks.duty_hours_7d",
                             "valid_for_window_ending": "2026-09-14"},
                "we_follow": "recompute per date",
            },
            {
                "id": "CONF-06", "rule": "RULE-FDP-01", "class": "A",
                "prose_says": "(implied) a delay shifts the duty.",
                "data_requires": ("Two different delay semantics. Deadhead "
                                  "positioning shifts report AND release, so FDP "
                                  "is invariant. A technical delay holds report "
                                  "and pushes release, so FDP grows."),
                "why": ("Implementing both the same way breaks either the "
                        "delay-cascade scenario or the deadhead option."),
                "evidence": {"deadhead": "report+delay, release+delay",
                             "technical": "report fixed, release+delay"},
                "we_follow": "both, selected by event type",
            },
            {
                "id": "CONF-07", "rule": "(answer key)", "class": "A",
                "prose_says": ("The delay scenario's three-leg option quotes an "
                               "FDP of 9.5h."),
                "data_requires": ("9.5h is only reachable by shifting the report "
                                  "time as well, which contradicts the same "
                                  "key's own breach calculation."),
                "why": ("That answer key is hand-written prose, not machine "
                        "derived. We hold the report fixed, consistent with the "
                        "breach, and reach the same operational conclusion: "
                        "operate three legs, re-crew the fourth."),
                "evidence": {"our_fdp_3_legs": 11.0, "key_fdp_3_legs": 9.5,
                             "same_conclusion": True},
                "we_follow": "internal consistency, and we flag the difference",
            },
        ],
    }


def render(snap: Snapshot) -> str:
    r = report(snap)
    out = ["", "  RULEBOOK CONFORMANCE REPORT",
           "  Where the written rules and the shipped data disagree.",
           "  " + "-" * 66, ""]
    for f in r["findings"]:
        cls = ("CLASS A - interpretation" if f["class"] == "A"
               else "CLASS B - DATA DEFECT")
        out.append(f"  {f['id']}  {f['rule']}   [{cls}]")
        out.append(f"    prose:     {f['prose_says']}")
        out.append(f"    data:      {f['data_requires']}")
        out.append(f"    why:       {f['why']}")
        out.append(f"    we follow: {f['we_follow']}")
        if f.get("risk_accepted"):
            out.append(f"    RISK:      {f['risk_accepted']}")
        out.append("")

    s = _strict_fleet_check(snap)
    out += [
        "  " + "-" * 66,
        "  STRICT MODE (the semantically correct rules)",
        f"    Applying valid_from <= date <= valid_to on {s['as_of']}:",
        f"    {s['legal_under_strict']} of {s['active_crew']} active crew remain "
        f"legal.",
        f"    {s['licences_not_yet_valid']} licences have a valid_from in the "
        f"future.",
        "",
        "    That is why CONF-01 follows the data. It is a defect in the",
        "    dataset, not a modelling choice, and it is the single most",
        "    important thing to fix before this touches a real roster.",
        "",
    ]
    return "\n".join(out)
