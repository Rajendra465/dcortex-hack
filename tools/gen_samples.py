"""Regenerate SAMPLES.md by actually running the advisor.

SAMPLES.md drifted badly once already: it documented three routing failures
that had since been fixed, which is worse than having no samples file at all --
a reader who checks one claim and finds it stale stops believing the others.
So the transcripts are generated, never typed, and this script is the only way
the file is allowed to change.

    python tools/gen_samples.py

Runs with the model disabled on purpose. Everything here must be reproducible
offline; a transcript that needed a network call would not be evidence.
"""
from __future__ import annotations

import builtins
import io
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewops.agent import Advisor, Session          # noqa: E402
from crewops.cli import Out, render_answer          # noqa: E402
from crewops.data import load                       # noqa: E402
from crewops.evaluate import render_e2e, run_e2e    # noqa: E402
from crewops.events import SickCrew                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "SAMPLES.md")

SECTIONS: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("Tier 1 — lookup", "", [
        ("Who is on reserve at BLR on 2026-09-15?", ""),
        ("How many duty hours does C-1042 have left this week?", ""),
        ("Which flights depart DEL on 2026-09-15?", ""),
        ("Who are the captains based at DEL?", ""),
        ("What is C-2087's rank, and total flight hours over the 28 days "
         "ending 2026-09-14?",
         "**Two questions in one sentence.** The planner emits a two-step "
         "plan and answers both halves. Picking a winner between them would "
         "silently drop half of what was asked."),
    ]),
    ("Tier 2 — consequence", "", [
        ("If I move C-2087 onto P-2291, does anyone breach a duty limit?", ""),
        ("Can reserve C-3305 cover the full pairing P-2291?", ""),
        ("BLR is closed 08:00 to 14:00 on 2026-09-17 - what is the crew impact?",
         ""),
        ("VT-DXA is delayed 90 minutes on 2026-09-16", ""),
        ("After the 90-minute delay to VT-DXA on 16 Sep, what should Crew "
         "Control do about the FDP breach?",
         "**The hyphenated duration used to parse as nothing at all**, which "
         "made the delay simulator fail its argument check, drop out of "
         "scoring entirely, and hand the question to a roster lookup."),
    ]),
    ("Tier 3 — recommendation", "", [
        ("C-1042 is out for P-2291, what should I do?", ""),
        ("What is the smallest change that would make C-2087 legal for P-2291?",
         ""),
        ("Where are we thin on cover this week?", ""),
        ("What is the cheapest legal way to cover the VT-DXF First Officer "
         "on 20 Sep if they call sick at 03:30Z?",
         "**A tail number names a subject.** The binder resolves `VT-DXF` "
         "plus the date to the trip it flies; until the scorer agreed, this "
         "question fell through to a plain reserve listing."),
        ("Draft the callout notification to C-3310 for P-2291", ""),
    ]),
    ("Refusals — the system declining to guess",
     "Each refusal says what would fix it. A refusal that does not tell you "
     "what to type next is just a failure with better manners.", [
         ("What is the probability that C-2087 calls in sick tomorrow?", ""),
         ("What is C-3310's phone number?", ""),
         ("Who is on reserve on 2026-10-05?", ""),
         ("Tell me about C-9999", ""),
         ("my BLR captain on the DEL overnight is out - options?",
          "**Referring expressions are not resolved offline.** The index "
          "cannot turn this into a crew id. What matters is what it does "
          "next: an earlier version let the runner-up answer and returned 28 "
          "captains based at BLR — fluent, instant, and about a different "
          "question. It now says what is missing instead."),
     ]),
    ("Horizons are per-domain",
     "The roster runs one week; certificate validity runs to 2032. One "
     "shared horizon meant every licence question past the 20th was refused "
     "against data that was sitting right there.", [
         ("Whose certificates expire before 2026-12-31?", ""),
         ("Which flights depart DEL on 2027-01-01?", ""),
     ]),
]


def _capture(fn) -> str:
    """Run `fn`, collecting everything the CLI renderer prints."""
    buf = io.StringIO()
    real = builtins.print
    builtins.print = lambda *a, **k: real(*a, **dict(k, file=buf))
    try:
        fn()
    finally:
        builtins.print = real
    # render_answer opens with a blank line to breathe in a live terminal;
    # inside a fenced block it just looks like a mistake.
    return buf.getvalue().strip("\n")


def _plain_out() -> Out:
    out = Out()
    out.c = None          # force the no-rich path: plain text, no ANSI
    return out


def transcript(adv: Advisor, q: str) -> str:
    out = _plain_out()
    return _capture(lambda: render_answer(out, adv.ask(q)))


def whatif_transcript(snap) -> str:
    out = _plain_out()
    sess = Session(snap, use_model=False)

    def run() -> None:
        print("  > who is on P-2291?")
        render_answer(out, sess.ask("who is on P-2291?"))
        print("  > :whatif C-1042 sick")
        sess.push_event(SickCrew(crew_id="C-1042"))
        print("  stacked: %s" % sess.what_if[-1])
        print("  > which flights are now uncrewed?")
        render_answer(out, sess.ask("which flights are now uncrewed?"))
        print("  > who is on P-2291?   (the same lookup, in the changed world)")
        render_answer(out, sess.ask("who is on P-2291?"))

    return _capture(run)


def main() -> int:
    snap = load()
    adv = Advisor(snap, use_model=False)
    w: list[str] = []
    add = w.append

    add("# Sample inputs and outputs\n")
    add(textwrap.fill(
        "Every transcript below is real terminal output, captured by "
        "`tools/gen_samples.py` running the advisor with no model configured, "
        "so nothing here depends on a network call. Do not edit this file by "
        "hand -- regenerate it.", 78) + "\n")
    add("Snapshot is frozen at `2026-09-14T18:00:00Z`; all times are UTC.\n")

    for title, blurb, items in SECTIONS:
        add("\n---\n")
        add(f"## {title}\n")
        if blurb:
            add(textwrap.fill(blurb, 78) + "\n")
        for q, note in items:
            add(f"\n### `{q}`\n")
            add("```")
            add(transcript(adv, q))
            add("```")
            if note:
                add("")
                add(textwrap.fill(note, 78))

    add("\n---\n")
    add("## Multi-turn: a what-if world\n")
    add(textwrap.fill(
        "Events stack as overlays on the immutable snapshot. Lookups after a "
        "what-if see the disrupted world; the simulation of the event just "
        "stacked is evaluated one layer down, in the world before it -- "
        "otherwise the simulator finds nobody left to remove and reports that "
        "nothing broke, which is the most dangerous wrong answer this system "
        "can give.", 78) + "\n")
    add("```")
    add(whatif_transcript(snap))
    add("```")

    add("\n---\n")
    add("## The two scores, and why both are published\n")
    add(textwrap.fill(
        "The kernel score measures exact arithmetic against the reference "
        "answer keys. The routing score measures whether a typed sentence "
        "reaches the capability that owns it. They are different claims, they "
        "fail independently, and for most of this build the first was perfect "
        "while the second was not.", 78) + "\n")
    add("```")
    add("$ python -m crewops eval --e2e")
    add(render_e2e(run_e2e(snap, use_model=False)).rstrip("\n"))
    add("```")

    text = "\n".join(w) + "\n"
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    n = sum(len(items) for _, _, items in SECTIONS)
    print(f"wrote {OUT}  ({len(text)} chars, {n} transcripts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
