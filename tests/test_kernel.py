"""Regression tests that lock in the traps.

Each of these encodes a specific way to be silently, confidently wrong. They
exist so a future refactor that "cleans up" one of the odd-looking rules fails
loudly instead of quietly changing a legality verdict.
"""
from __future__ import annotations

import json
import os
from datetime import date

import pytest

from crewops.agent import Advisor
from crewops.data import load
from crewops.events import (SickCrew, analyse_closure, analyse_delay,
                            minimal_repair, resolve_multi)
from crewops.kernel import (check_cover, cover_fragility, cover_options,
                            latent_breaches, positioning)
from crewops.rules import fdp_limit, window_sum
from crewops.tools import REGISTRY

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "extracted", "DCortex - Synthetic dataset", "data")


@pytest.fixture(scope="session")
def snap():
    return load(DATA)


@pytest.fixture(scope="session")
def scen():
    with open(os.path.join(DATA, "scenarios.json"), encoding="utf-8") as fh:
        return {s["scenario_id"]: s for s in json.load(fh)}


def _opt_tuples(options):
    return [(o.crew_id, o.cost_inr, o.delay_hours) for o in options]


def _key_tuples(key):
    return [(o["crew_id"], o["cost_inr"], o["delay_hours"]) for o in key]


# ==========================================================================
# answer-key parity
# ==========================================================================


@pytest.mark.parametrize("sid", ["S1", "S2", "S5"])
def test_ranked_options_match_answer_key(snap, scen, sid):
    s = scen[sid]
    pid, sick = s["event"]["pairing_id"], s["event"]["crew_id"]
    role = snap.pairings[pid].role_of(sick)
    got = cover_options(snap, list(snap.pairings[pid].days), role, sick, pid)
    assert _opt_tuples(got.options) == _key_tuples(s["answer_key"]["options"])


@pytest.mark.parametrize("sid", ["S1", "S2"])
def test_excluded_candidates_match_verbatim(snap, scen, sid):
    """Reason strings are the explainability payload -- they must match exactly."""
    s = scen[sid]
    pid, sick = s["event"]["pairing_id"], s["event"]["crew_id"]
    role = snap.pairings[pid].role_of(sick)
    got = cover_options(snap, list(snap.pairings[pid].days), role, sick, pid)
    mine = {e["crew_id"]: e["reason"] for e in got.excluded}
    theirs = {e["crew_id"]: e["reason"]
              for e in s["answer_key"]["excluded_candidates"]}
    assert mine == theirs


def test_joint_plan_never_double_books(snap, scen):
    """Answering two sick calls independently assigns the same person twice."""
    evs = []
    for ac in ("VT-DXA", "VT-DXB"):
        pid = next(p.pairing_id for p in snap.pairings.values()
                   if p.aircraft == ac
                   and any(d.date == "2026-09-18" for d in p.days))
        evs.append(SickCrew(crew_id=snap.pairings[pid].crew_in_role("Captain"),
                            pairing_id=pid))
    plan = resolve_multi(snap, evs)
    assert plan.total_cost_inr == (
        scen["S6"]["answer_key"]["optimal_joint_plan"]["total_cost_inr"])
    assigned = [o.crew_id for o in plan.assignments.values() if o.crew_id]
    assert len(assigned) == len(set(assigned)), "same crew assigned twice"


# ==========================================================================
# held-out generalisation -- same code paths, unseen arguments
# ==========================================================================


def test_heldout_atr_first_officer(snap):
    fo = snap.pairings["P-2224"].crew_in_role("First Officer")
    got = cover_options(snap, list(snap.pairings["P-2224"].days),
                        "First Officer", fo, "P-2224")
    assert got.options[0].crew_id == "C-3316"
    assert got.options[0].cost_inr == 18500


def test_heldout_hyd_closure(snap):
    imp = analyse_closure(snap, "HYD", "2026-09-19T05:00:00Z",
                          "2026-09-19T09:00:00Z")
    assert sorted(imp.data["affected_flights"]) == ["DX461-2026-09-19",
                                                    "DX462-2026-09-19"]


# ==========================================================================
# the traps
# ==========================================================================


def test_trap01_certs_use_valid_to_only(snap):
    """Every licence has a valid_from in the future; the strict read grounds the
    entire fleet. Reproducing the lenient read is required -- and is a
    documented, accepted risk."""
    on = date(2026, 9, 15)
    lenient = sum(1 for c in snap.crew.values()
                  if c.status == "active"
                  and snap.certs_valid_on(c.crew_id, on)[0])
    strict = sum(1 for c in snap.crew.values()
                 if c.status == "active"
                 and snap.certs_valid_on_strict(c.crew_id, on)[0])
    assert lenient > 100
    assert strict == 0, "if this passes, the dataset was fixed -- revisit CONF-01"


def test_trap02_reserve_window_uses_report_time_inclusively(snap):
    """The flagship answer sits exactly ON the 06:00 boundary."""
    r = snap.reserves["C-3310"]
    report = snap.pairings["P-2291"].days[0].report
    assert report.strftime("%H:%M") == r.window_start
    assert r.covers(report), "boundary must be inclusive"


def test_trap03_duty_clock_field_is_snapshot_only(snap):
    """duty_hours_7d is valid only for the window ending 2026-09-14.

    Two properties. First, our own sum reproduces the shipped field on the one
    date it is defined for. Second, the window genuinely moves for most crew, so
    reusing that single field for any other date is wrong -- it flips C-3305
    from legal to illegal. (C-2087 is a poor probe on its own: its two windows
    coincide by chance.)
    """
    for cid in ("C-2087", "C-3305", "C-1042"):
        mine, _ = window_sum(snap, cid, date(2026, 9, 14), 7, "duty")
        assert mine == pytest.approx(snap.clocks[cid]["duty_hours_7d"], abs=0.05)

    moved = sum(
        1 for cid in snap.crew
        if window_sum(snap, cid, date(2026, 9, 14), 7, "duty")[0]
        != window_sum(snap, cid, date(2026, 9, 17), 7, "duty")[0]
    )
    assert moved > 50, "the 7-day window must move with the date"


def test_trap04_sep14_double_count_is_canonical(snap):
    """History and roster BOTH contribute on 14 Sep. Do not deduplicate."""
    d = date(2026, 9, 14)
    both = [cid for cid in snap.crew
            if snap.history.get(cid, {}).get(d, (0.0, 0.0))[0]
            and any(x.day == d for x in snap.roster.get(cid, []))]
    assert len(both) == 11
    cid = both[0]
    total, _ = window_sum(snap, cid, d, 7, "duty")
    assert total == pytest.approx(snap.clocks[cid]["duty_hours_7d"], abs=0.05)


def test_trap05_flt03_is_evaluated_but_non_binding(snap):
    res = check_cover(snap, "C-3310", list(snap.pairings["P-2291"].days), "P-2291")
    assert res.legal
    assert "RULE-FLT-03" in res.non_binding
    peak = max(window_sum(snap, cid, duty.day, 28, "flight")[0]
               for cid in snap.crew for duty in snap.roster.get(cid, []))
    assert peak < 100.0


def test_trap06_qual05_short_circuits(snap):
    """An unrated candidate gets exactly ONE reason, with nothing appended."""
    res = check_cover(snap, "C-1042", list(snap.pairings["P-2224"].days), None)
    assert res.issues == ["RULE-QUAL-05: no ATR72 rating"]
    assert res.rules_evaluated == ["RULE-QUAL-05"]


def test_trap07_candidate_pool_is_all_active_crew_not_reserves(snap):
    got = cover_options(snap, list(snap.pairings["P-2291"].days), "Captain",
                        "C-1042", "P-2291")
    non_reserve = [o for o in got.legal_options if not o.is_reserve]
    assert non_reserve, "day-off callouts of line crew must be offered"
    assert got.candidate_pool_size > len(snap.reserves)


def test_trap08_cancel_is_appended_after_the_sort(snap):
    got = cover_options(snap, list(snap.pairings["P-2291"].days), "Captain",
                        "C-1042", "P-2291")
    last = got.options[-1]
    assert last.crew_id is None
    assert last.cost_inr > got.options[0].cost_inr


def test_trap10_two_delay_semantics(snap):
    """Deadhead shifts report AND release (FDP invariant); a technical delay
    holds report and pushes release (FDP grows)."""
    days = list(snap.pairings["P-2291"].days)
    plain = check_cover(snap, "C-2210", days, "P-2291", delay_hours=0.0)
    shifted = check_cover(snap, "C-2210", days, "P-2291", delay_hours=3.0)
    assert plain.ledger.get("fdp_hours") == shifted.ledger.get("fdp_hours")

    imp = analyse_delay(snap, 1.5, aircraft="VT-DXA", on_date="2026-09-16")
    assert imp.data["fdp_after_delay"] > imp.data["fdp_before"]
    assert imp.data["breach"] is True


def test_trap11_fdp_limit_reduces_per_sector():
    assert fdp_limit(2) == 13.0
    assert fdp_limit(3) == 12.5
    assert fdp_limit(4) == 12.0


def test_trap12_multiday_pairing_legal_on_every_day(snap):
    """C-3305 is fine on day one and breaches on day two."""
    days = list(snap.pairings["P-2291"].days)
    day1 = check_cover(snap, "C-3305", days[:1], "P-2291")
    both = check_cover(snap, "C-3305", days, "P-2291")
    assert day1.legal
    assert not both.legal
    assert "8h15m" in both.reason


def test_trap13_downstream_rest_conflict(snap):
    """C-5837 is legal across the trip and fails on rest two days later."""
    res = check_cover(snap, "C-5837", list(snap.pairings["P-2291"].days), "P-2291")
    assert not res.legal
    assert "10.75h rest" in res.reason
    assert "downstream conflict" in res.reason


def test_trap14_closure_window_is_half_open(snap):
    """A departure exactly at the reopening time is NOT inside the window."""
    imp = analyse_closure(snap, "BLR", "2026-09-17T08:00:00Z",
                          "2026-09-17T14:00:00Z")
    for fid in imp.data["affected_flights"]:
        f = snap.flights[fid]
        dep_hit = f.dep_station == "BLR" and "08:00" <= f.dep.strftime("%H:%M") < "14:00"
        arr_hit = f.arr_station == "BLR" and "08:00" <= f.arr.strftime("%H:%M") < "14:00"
        assert dep_hit or arr_hit


def test_trap15_tie_break_is_lexicographic_on_crew_id(snap):
    got = cover_options(snap, list(snap.pairings["P-2291"].days), "Captain",
                        "C-1042", "P-2291")
    keys = [(o.cost_inr, o.crew_id or "") for o in got.options[:-1]]
    assert keys == sorted(keys)


def test_positioning_only_models_del_to_blr(snap):
    ok, _d, _f, _a = positioning("BOM", "BLR", date(2026, 9, 15),
                                 snap.flights["DX412-2026-09-15"].dep)
    assert not ok
    ok, delay, flt, arr = positioning("DEL", "BLR", date(2026, 9, 15),
                                      snap.flights["DX412-2026-09-15"].dep)
    assert ok and flt == "DX402" and delay == 3.0
    assert arr.strftime("%H:%M") == "08:45"


# ==========================================================================
# proactive signals
# ==========================================================================


def test_finds_the_single_point_of_failure(snap):
    rows = cover_fragility(snap, ("Captain",))
    critical = [r for r in rows if r["legal_covers"] <= 1]
    assert any(r["pairing_id"] == "P-2289" for r in critical)


def test_finds_the_already_illegal_roster_entry(snap):
    rows = latent_breaches(snap)
    assert any(b["crew_id"] == "C-5417" and b["date"] == "2026-09-19"
               for b in rows)


def test_duty_limit_watchlist_would_be_empty(snap):
    """Documents WHY we did not build the panel the brief suggests."""
    peak = max(window_sum(snap, cid, duty.day, 7, "duty")[0]
               for cid in snap.crew for duty in snap.roster.get(cid, []))
    assert peak < 45.0, "peak utilisation is far below the 60h cap"


def test_minimal_repair_quantifies_the_gap(snap):
    out = minimal_repair(snap, "C-2087", "P-2291")
    assert not out["already_legal"]
    assert out["repairs"][0]["shortfall_hours"] == pytest.approx(1.33, abs=0.01)


# ==========================================================================
# the language layer
# ==========================================================================


@pytest.fixture(scope="session")
def advisor(snap):
    return Advisor(snap, use_model=False)


@pytest.mark.parametrize("question,kind", [
    ("What is the probability that C-2087 calls in sick tomorrow?", "DATA_GAP"),
    ("What is the weather at BLR tomorrow?", "DATA_GAP"),
    ("What is C-3310's phone number?", "DATA_GAP"),
    ("How many passengers are booked on DX412?", "DATA_GAP"),
    ("Reroute the aircraft", "DATA_GAP"),
    ("Who is on reserve on 2026-10-05?", "OUT_OF_RANGE"),
    ("Tell me about C-9999", "UNKNOWN_ENTITY"),
    ("asdfgh qwerty", "PARSE_FAIL"),
])
def test_hostile_questions_are_refused_specifically(advisor, question, kind):
    a = advisor.ask(question)
    assert a.refusal is not None, f"should have refused: {question}"
    assert a.refusal.kind == kind


@pytest.mark.parametrize("question", [
    "Who is on reserve at BLR on 2026-09-15?",
    "How many duty hours does C-1042 have left this week?",
    "Which flights depart DEL on 2026-09-15?",
    "If I move C-2087 onto P-2291, does anyone breach a duty limit?",
    "C-1042 is out for P-2291, what should I do?",
    "BLR is closed 08:00 to 14:00 on 2026-09-17 - what is the crew impact?",
])
def test_answerable_questions_are_answered_without_a_model(advisor, question):
    a = advisor.ask(question)
    assert a.refusal is None, f"should have answered: {question}"
    assert a.prose


@pytest.mark.parametrize("question,tool", [
    ("who are the captains based at DEL?", "find_crew"),
    ("list the captains at DEL", "find_crew"),
    ("how many flights operate on 2026-09-16?", "find_flights"),
    ("which reserves are on call at BLR on 2026-09-15?", "get_reserves"),
    ("show me the pairings on 2026-09-16", "get_roster"),
    ("which certificates expire after 2026-09-15?", "get_certifications"),
    ("what are the legs out of BLR on 2026-09-15?", "find_flights"),
])
def test_plural_nouns_route_correctly(advisor, question, tool):
    r"""Regression: \b(captain)\b cannot match "captains".

    The trailing word boundary fails on the plural s, which silently broke every
    plural noun in the grammar -- "captains", "flights", "reserves", "pairings",
    "certificates" all fell through to a refusal. Same root cause as the
    prediction screen that once let "probability" through. Inflected forms must
    reach the same tool as the singular.
    """
    a = advisor.ask(question)
    assert a.refusal is None, f"refused a plural phrasing: {question}"
    assert a.plan.tool == tool


def test_stemmer_folds_inflection_without_over_matching():
    """The stemmer is what makes the plural bug structurally impossible.

    It replaced a regex cascade where `\\b(captain)\\b` could not match
    "captains". Stems must fold inflection but must not collapse unrelated
    words: "legal" is not a "leg".
    """
    from crewops.orchestrator import _tokens
    assert _tokens("captains") == _tokens("captain")
    assert _tokens("flights") == _tokens("flight")
    assert _tokens("pairings") == _tokens("pairing")
    assert _tokens("legal") != _tokens("leg")
    # the phrase a controller actually types must not stem to nothing
    assert _tokens("what should i do")


def test_planner_is_data_driven_not_control_flow():
    """Every capability is reachable from its declared examples.

    This is the property the regex cascade could not offer: adding a tool means
    adding example rows, and this test proves none of them is unreachable.
    """
    from crewops.orchestrator import INTENTS, _tokens, score_intents, Entities
    from crewops.tools import REGISTRY
    for intent in INTENTS:
        assert intent.tool in REGISTRY, f"{intent.tool} is not a real tool"
        assert intent.examples, f"{intent.tool} declares no examples"
        for ex in intent.examples:
            assert _tokens(ex), f"example stems to nothing: {ex!r}"
        # Give each intent only the entities IT declares it needs. Handing
        # every intent every entity at once rewards whichever tool happens to
        # declare the most parameters, which measures nothing.
        need = set(intent.needs)
        ents = Entities(
            crew=["C-1042"] if {"crew", "pairing_or_crew"} & need else [],
            pairings=["P-2291"] if "pairing_or_crew" in need else [],
            stations=["BLR"] if "station" in need else [],
            hours=1.5 if "duration" in need else None)
        ranked = score_intents(intent.examples[0], ents)
        assert ranked and ranked[0][1].tool == intent.tool, (
            f"{intent.tool} not top for its own example "
            f"{intent.examples[0]!r}; got {ranked[:2]}")


def test_the_same_question_two_ways_gives_the_same_answer(advisor):
    a = advisor.ask("C-1042 is out for P-2291, what should I do?")
    b = advisor.ask("who can cover P-2291 instead of C-1042?")
    assert a.payload["recommended"]["crew_id"] == b.payload["recommended"]["crew_id"]
    assert a.payload["recommended"]["cost_inr"] == b.payload["recommended"]["cost_inr"]


def test_containment_guard_blocks_an_invented_number():
    from crewops.agent import guard_numbers
    from crewops.rules import Ledger
    led = Ledger()
    led.add("duty_7d_total", 61.33, "h")
    led.add("duty_cap_hours", 60.0, "h")
    ok, _ = guard_numbers("Total is 61.33h against a 60h cap.", led)
    assert ok
    ok, bad = guard_numbers("Total is 58.10h against a 60h cap.", led)
    assert not ok and "58.10" in bad


def test_containment_guard_allows_unit_conversions_and_identifiers():
    """The naive guard blocks the engine's own most explanatory output."""
    from crewops.agent import guard_numbers
    from crewops.rules import Ledger
    led = Ledger()
    led.add("duty_7d_excess", 1.33, "h")
    led.add("rest_hours", 10.75, "h")
    ok, bad = guard_numbers(
        "C-2087 is over by 1h20m on P-2291; C-5837 has only 10.75h rest "
        "before DX412 on 2026-09-17, and neither is A320 rated.", led)
    assert ok, f"false block on: {bad}"


# ==========================================================================
# the whole harness
# ==========================================================================


def test_full_evaluation_has_zero_wrong():
    from crewops import evaluate
    rep = evaluate.run(data_dir=DATA)
    assert rep.count(evaluate.WRONG) == 0, [
        (c.ident, c.got, c.want) for c in rep.wrong]
    assert rep.count(evaluate.CORRECT) >= 40


def test_every_tool_is_reachable_and_documented():
    for name, t in REGISTRY.items():
        assert t.doc, f"{name} has no description"
        assert t.kind in ("retrieval", "simulation", "optimisation")
        for r in t.required:
            assert r in t.params, f"{name}: required param {r} not declared"


# ---------------------------------------------------------------- REST-04 cascade
# A delay grows the duty AND pushes release later. FDP-01 only sees the first
# half of that. On P-2291 the overnight rest is 12.50h against a 12h floor, so
# 0.50h of delay is the entire margin -- and the duty is still 2.5h inside its
# own FDP limit when the rest breaks. Before this was fixed the advisor said
# "still legal" about a morning six crew could not legally report for.

def test_delay_reports_rest_cascade_not_just_fdp(snap):
    """0.51h on P-2291 day 1 is legal on FDP and illegal on rest."""
    imp = analyse_delay(snap, delay_hours=0.51, pairing_id="P-2291",
                        on_date="2026-09-15")
    d = imp.data
    assert d["breach"] is False, "FDP-01 is genuinely satisfied here"
    assert d["fdp_after_delay"] < d["fdp_limit"], "duty sits inside its own limit"
    assert d["rest_breach"] is True, "but RULE-REST-04 is breached downstream"
    assert {b["crew_id"] for b in d["rest_breaches"]} == {
        "C-1042", "C-1694", "C-3005", "C-4395", "C-4273", "C-1873"}
    assert d["rest_breaches"][0]["rest_hours"] == 11.99
    assert "still legal" not in imp.summary, "the old wording was the bug"
    assert "RULE-REST-04" in imp.summary


def test_delay_inside_the_rest_margin_stays_clean(snap):
    """0.50h is exactly the margin -- 12.00h rest is legal, so no false alarm."""
    imp = analyse_delay(snap, delay_hours=0.50, pairing_id="P-2291",
                        on_date="2026-09-15")
    assert imp.data["rest_breach"] is False
    assert imp.data["rest_breaches"] == []
    assert "still legal" in imp.summary


def test_s4_delay_unchanged_by_the_rest_check(snap):
    """The shipped S4 key is an FDP breach with 60.75h of rest slack."""
    imp = analyse_delay(snap, delay_hours=1.5, aircraft="VT-DXA",
                        on_date="2026-09-16")
    d = imp.data
    assert (d["fdp_after_delay"], d["fdp_limit"], d["breach"]) == (12.75, 12.0, True)
    assert d["legs_to_shed"] == ["DX404-2026-09-16"]
    assert d["rest_breach"] is False
