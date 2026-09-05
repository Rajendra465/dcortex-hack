"""Routing regression lock.

The kernel suite in test_kernel.py proves the arithmetic. This one proves the
LANGUAGE LAYER: that a typed sentence reaches the capability that owns it.

Those are different claims and they fail independently -- for most of this
build the kernel was at 42/42 while barely two thirds of typed questions
arrived anywhere useful. Anything that regresses routing regresses the product
even with every kernel test green, so the shipped prompt set is asserted here
question by question rather than as an aggregate percentage.
"""
from __future__ import annotations

import json
import os

import pytest

from crewops.agent import Advisor, Session
from crewops.events import SickCrew
from crewops.data import load
from crewops.evaluate import EXPECTED_TOOL
from crewops.orchestrator import extract, score_intents
from crewops.server import EXAMPLES

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "extracted", "DCortex - Synthetic dataset", "data")


@pytest.fixture(scope="module")
def snap():
    return load()


@pytest.fixture(scope="module")
def advisor(snap):
    # use_model=False on purpose: this suite must be deterministic and offline.
    # The model lane is exercised by eval --e2e when a key is configured.
    return Advisor(snap, use_model=False)


def _questions():
    with open(os.path.join(DATA, "questions.json"), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# every shipped prompt reaches its capability
# --------------------------------------------------------------------------


@pytest.mark.parametrize("q", _questions(), ids=lambda q: q["question_id"])
def test_shipped_prompt_routes(advisor, q):
    want = EXPECTED_TOOL[q["question_id"]]
    a = advisor.ask(q["prompt"])
    assert a.refusal is None, (
        f"{q['question_id']} refused ({a.refusal.kind}) a question the data "
        f"can answer: {a.refusal.message}")
    assert want in a.plan.tools, (
        f"{q['question_id']} routed to {a.plan.tools}, wanted {want}\n"
        f"  {q['prompt']}")


# --------------------------------------------------------------------------
# every example the UI offers actually works when clicked
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ex", EXAMPLES, ids=lambda e: e["q"][:44])
def test_ui_example_behaves_as_labelled(advisor, ex):
    """A demo that offers a chip which errors is worse than not offering it."""
    a = advisor.ask(ex["q"])
    if ex["tier"] == "REFUSE":
        assert a.refusal is not None, (
            f"expected a refusal, got {a.plan.tool}: {a.prose[:120]}")
        assert a.refusal.kind in {"DATA_GAP", "PARSE_FAIL", "CLARIFY",
                                  "OUT_OF_RANGE", "UNKNOWN_ENTITY"}
        assert a.refusal.would_need, "a refusal must say what would fix it"
    else:
        assert a.refusal is None, f"unexpected refusal: {a.refusal.message}"
        assert a.payload is not None
        assert a.prose.strip(), "an answer with no prose is not an answer"


# --------------------------------------------------------------------------
# the specific parse bugs that cost real routing accuracy, held down
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,hours", [
    ("VT-DXA is delayed 90 minutes", 1.5),
    ("the 90-minute delay to VT-DXA", 1.5),      # hyphenated form
    ("a 2 hour delay", 2.0),
    ("delayed by 45 mins", 0.75),
    ("2.5 hrs late", 2.5),
])
def test_duration_parses_both_written_forms(snap, text, hours):
    assert extract(text, snap).hours == pytest.approx(hours)


@pytest.mark.parametrize("text", [
    "Who is on reserve at BLR on 2026-09-15?",
    "flights on 2026-09-14",
    "the 28 days ending 2026-09-14",
])
def test_dates_are_never_read_as_a_result_limit(snap, text):
    """"15 Sep" is not a request for fifteen of anything."""
    assert extract(text, snap).limit is None


@pytest.mark.parametrize("text,limit", [
    ("I need 5 pilots", 5), ("give me 3 captains", 3),
    ("top 4 options", 4), ("show me 2 first officers", 2),
])
def test_people_counts_do_become_a_limit(snap, text, limit):
    assert extract(text, snap).limit == limit


@pytest.mark.parametrize("text,ranks", [
    ("who are the captains", ["Captain"]),
    ("list the first officers", ["First Officer"]),
    ("I need pilots", ["Captain", "First Officer"]),
    ("the FOs at DEL", ["First Officer"]),
])
def test_rank_words_including_plurals_and_classes(snap, text, ranks):
    assert sorted(extract(text, snap).ranks) == sorted(ranks)


def test_window_is_not_wind(snap):
    """`wind` inside `window` refused every on-call question as weather."""
    a = Advisor(snap, use_model=False).ask(
        "Which reserve on-call windows cover 03:00Z on 2026-09-16?")
    assert a.refusal is None, a.refusal.message if a.refusal else ""


def test_aircraft_tail_counts_as_naming_a_subject(snap):
    """bind_args resolves a tail to a trip, so scoring must agree it has one."""
    q = ("what is the cheapest legal way to cover the VT-DXF First Officer "
         "on 2026-09-20")
    top = score_intents(q, extract(q, snap))[0]
    assert top[1].tool == "rank_cover_options", top[1].tool


# --------------------------------------------------------------------------
# two-part questions
# --------------------------------------------------------------------------


def test_conjunctive_question_answers_both_halves(advisor):
    a = advisor.ask("What is C-2087's rank, and total flight hours over the "
                    "28 days ending 2026-09-14?")
    assert a.plan.conjunctive, a.plan.tools
    assert set(a.plan.tools) == {"find_crew", "duty_hours"}
    assert "Captain" in a.prose            # the rank half
    assert "23.5" in a.prose               # the hours half


def test_single_subject_question_stays_single_step(advisor):
    """The conjunction rule must not chain tools nobody asked for."""
    a = advisor.ask("Which flights depart DEL on 2026-09-15?")
    assert len(a.plan.steps) == 1, a.plan.tools


# --------------------------------------------------------------------------
# reserves without a date
# --------------------------------------------------------------------------


def test_reserve_pool_is_answerable_without_a_date(snap):
    from crewops.tools import REGISTRY
    payload, _ = REGISTRY["get_reserves"](snap, rank="Captain")
    assert payload["count"] > 0
    assert payload["date"] is None
    assert all(r["on_call_dates"] for r in payload["reserves"])


# --------------------------------------------------------------------------
# horizons are per-domain, not one global cliff
# --------------------------------------------------------------------------


@pytest.mark.parametrize("q,after,before", [
    ("Which certifications expire between 2026-09-14 and 2027-03-01?",
     "2026-09-14", "2027-03-01"),
    ("Whose certificates expire before 2026-12-31?",
     "2026-09-14", "2026-12-31"),
    ("Which licences lapse before 2026-10-01?",
     "2026-09-14", "2026-10-01"),
])
def test_certificate_questions_reach_past_the_roster_horizon(advisor, q,
                                                             after, before):
    """Certificates run to 2032; the roster stops on the 20th."""
    a = advisor.ask(q)
    assert a.refusal is None, a.refusal.message
    assert a.plan.tool == "get_certifications"
    assert a.plan.args["expiring_after"] == after
    assert a.plan.args["expiring_before"] == before


@pytest.mark.parametrize("q", [
    "Which flights depart DEL on 2027-01-01?",
    "Who is on reserve on 2026-10-05?",
    "Who is rostered on 2026-09-25?",
])
def test_roster_questions_still_stop_at_the_roster_horizon(advisor, q):
    """A wider certificate horizon must not widen the roster's."""
    a = advisor.ask(q)
    assert a.refusal is not None and a.refusal.kind == "OUT_OF_RANGE", (
        f"answered a rosterless date: {a.prose[:120]}")
    assert "roster" in a.refusal.message


def test_horizons_are_actually_different(snap):
    assert snap.cert_horizon[1] > snap.horizon[1]


def test_reserve_query_still_needs_some_handle(snap):
    """Not requiring a date is not the same as requiring nothing."""
    from crewops.orchestrator import _unbound
    assert _unbound("get_reserves", {}) == [
        "date or crew_id or base or rank or ranks"]
    assert _unbound("get_reserves", {"date": "2026-09-15"}) == []


# --------------------------------------------------------------------------
# multi-turn: carrying context, and stacking what-ifs
# --------------------------------------------------------------------------


def test_pronoun_carries_the_previous_subject(snap):
    s = Session(snap, use_model=False)
    s.ask("Can reserve C-3305 cover the full pairing P-2291?")
    a = s.ask("And would that breach a duty limit for him?")
    assert a.plan.args.get("crew_id") == "C-3305"
    assert any("C-3305" in n for n in a.assumptions), a.assumptions


def test_a_fresh_question_does_not_inherit_a_subject(snap):
    s = Session(snap, use_model=False)
    s.ask("Can reserve C-3305 cover the full pairing P-2291?")
    a = s.ask("Which flights depart DEL on 2026-09-15?")
    assert a.plan.args.get("crew_id") is None
    assert not a.assumptions


def test_whatif_then_consequence_reports_the_real_impact(snap):
    """The regression that made a stacked disruption read as harmless."""
    s = Session(snap, use_model=False)
    s.push_event(SickCrew(crew_id="C-1042"))
    a = s.ask("which flights are now uncrewed?")
    assert a.plan.tool == "simulate_crew_unavailable"
    assert a.payload["passengers_at_risk_day1"] > 0
    assert sum(len(v) for v in a.payload["uncovered_by_day"].values()) > 0


def test_lookups_after_a_whatif_see_the_disrupted_world(snap):
    s = Session(snap, use_model=False)
    before = s.ask("who is on P-2291?")
    assert any(c["crew_id"] == "C-1042"
               for c in before.payload["pairings"][0]["crew"])
    s.push_event(SickCrew(crew_id="C-1042"))
    after = s.ask("who is on P-2291?")
    assert not any(c["crew_id"] == "C-1042"
                   for c in after.payload["pairings"][0]["crew"])


def test_consequence_wording_needs_a_stack_to_refer_back(snap):
    """"what is the impact" with an empty stack is a fresh question."""
    s = Session(snap, use_model=False)
    s.ask("Can reserve C-3305 cover the full pairing P-2291?")
    a = s.ask("what is the impact")
    assert not any("carried" in n for n in a.assumptions), a.assumptions


def test_reset_restores_the_live_snapshot(snap):
    s = Session(snap, use_model=False)
    s.push_event(SickCrew(crew_id="C-1042"))
    assert s.what_if == ["C-1042 unavailable"]
    s.reset()
    assert s.what_if == []
    assert s.advisor.snap is snap


def test_events_reject_positional_construction():
    """`SickCrew("C-1042")` used to bind the id to `type` and strip nobody."""
    with pytest.raises(TypeError):
        SickCrew("C-1042")
    assert SickCrew(crew_id="C-1042").describe() == "C-1042 unavailable"


# --------------------------------------------------------------------------
# numeric containment: no figure reaches the reader without a fact behind it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("q", [x["prompt"] for x in _questions()],
                         ids=[x["question_id"] for x in _questions()])
def test_every_answer_survives_the_containment_guard(advisor, q):
    """The claim the UI underlines in cyan, asserted on the server.

    Two bugs this catches, and both were live: a summary that quotes a figure
    the kernel never recorded (the ledger is incomplete, so the guard would
    block a model narration of a correct answer), and a summary that quotes a
    figure that is simply wrong. Either way the prose has drifted from the
    evidence, and the whole design rests on it not doing that.
    """
    from crewops.agent import guard_numbers
    a = advisor.ask(q)
    if a.refusal:
        return
    ok, bad = guard_numbers(a.prose, a.ledger)
    assert ok, (f"prose quotes {bad} with no matching fact\n"
                f"  Q: {q}\n  A: {a.prose}")


# --------------------------------------------------------------------------
# refusing beats answering a neighbouring question
# --------------------------------------------------------------------------


def test_unresolvable_referring_expression_refuses(advisor):
    """The documented failure case, held to a refusal rather than a listing.

    Before the starved-leader rule this answered with 28 captains based at
    BLR: the cover intent scored highest, lost its needs check because the
    referring expression resolves to nobody, and a plain crew listing won the
    fall-through. Fluent, instant, and about a different question.
    """
    a = advisor.ask("my BLR captain on the DEL overnight is out — options?")
    assert a.refusal is not None, f"answered instead: {a.prose[:120]}"
    assert a.refusal.kind == "CLARIFY"
    assert "crew member" in a.refusal.would_need


@pytest.mark.parametrize("q", [
    "If DX404 on 16 Sep is cancelled, how many passengers are affected and "
    "what is the direct cancellation cost?",
    "Who are the captains based at DEL?",
    "Which flights depart DEL on 2026-09-15?",
    "Who is on reserve at BLR on 2026-09-15?",
])
def test_starved_leader_does_not_swallow_good_questions(advisor, q):
    """A high-scoring blocked intent must not veto a question that works.

    "how many passengers are affected" makes simulate_crew_unavailable score
    1.10 on the word "affected" while missing a crew id — but the sentence
    named DX404, and find_flights answers it exactly. The subject test is what
    separates the two cases.
    """
    a = advisor.ask(q)
    assert a.refusal is None, (
        f"refused a question it can answer: {a.refusal.message}")
