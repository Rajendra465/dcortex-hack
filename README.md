# Crew Ops Advisor

A conversational decision aid for an airline Crew Control desk.

**The model translates. The kernel decides.** Nothing that carries a number
passes through a language model.

```
$ python -m crewops ask "If I move C-2087 onto P-2291, does anyone breach a duty limit?"

  C-2087 cannot cover P-2291. RULE-DUTY-02: would exceed 60h/7d by 1h20m
  on 2026-09-15 (total 61.33h).
  T2 · check_legality · 1 ms · computed · parsed by rules

  XX  NOT LEGAL
      - RULE-DUTY-02: would exceed 60h/7d by 1h20m on 2026-09-15 (total 61.33h)
      - RULE-DUTY-02: would exceed 60h/7d by 1h05m on 2026-09-16 (total 61.08h)
  RULE-FLT-03: evaluated, non-binding on this data
```

---

## Setup

Python 3.11+. No required third-party dependencies for the deterministic core.

```bash
python -m crewops status        # dataset + engine health
python -m crewops eval          # the answer-key regression
python -m crewops brief         # the morning board
python -m crewops chat          # interactive desk
python -m crewops ask "who is on reserve at BLR on 2026-09-15?"
python -m crewops conformance   # rulebook vs data disagreements
```

Optional — a language model widens what phrasings are understood. Either
provider works; copy `.env.example` to `.env` and fill one in:

```bash
pip install rich                      # nicer tables (degrades cleanly without it)

pip install openai                    # NVIDIA NIM (OpenAI-compatible)
NVIDIA_API_KEY=nvapi-...
CREWOPS_MODEL=nvidia/nemotron-3-ultra-550b-a55b

pip install anthropic                 # ...or Anthropic
ANTHROPIC_API_KEY=sk-ant-...
CREWOPS_MODEL=claude-sonnet-5
```

Other switches: `CREWOPS_NO_MODEL=1` forces the deterministic lane,
`CREWOPS_TIMEOUT` (default 12s) bounds any model call, and `CREWOPS_NARRATE=1`
turns on model-written prose (see below for why it is off).

The dataset is found automatically at `extracted/DCortex - Synthetic dataset/data`.
Override with `--data <dir>` or `CREWOPS_DATA_DIR`.

---

## Where the boundary is drawn

This is the whole design.

```
                                  ┌────────────────────────────┐
 controller's question ─▶ ROUTER ─┤ LANE A · typed plan IR     │──┐
                                  │   model emits a PLAN,      │  │
                                  │   never an answer          │  │
                                  ├────────────────────────────┤  │
                                  │ LANE B · kernel-as-tools   │──┤
                                  │   agent loop, for phrasings│  │
                                  │   Lane A cannot parse      │  │
                                  ├────────────────────────────┤  │      ┌────────────┐
                                  │ LANE C · judgement         │  ├─────▶│   KERNEL   │
                                  │   labelled advisory;       │  │      │  7 rules   │
                                  │   asserts no new number    │  │      │  exact     │
                                  └────────────────────────────┘  │      │  ~1.4 ms   │
                                                                  │      │  NO MODEL  │
                                  ┌────────────────────────────┐  │      └─────┬──────┘
  answer ◀── NARRATOR ◀── gate ◀──┤  EVIDENCE LEDGER           │◀─┘            │
             writes the          │  every number the kernel     │◀─────────────┘
             sentence only       │  touched, incl. intermediates│
                                 └────────────────────────────┘
```

**Rule 1 — the model never computes.** Enforced structurally, not by
instruction: raw duty clocks never enter the narration context at all. Only the
evidence ledger does. A model told not to compute will still restate a number it
half-remembers from context, so we remove the context.

**Rule 2 — every number in the prose must trace to the ledger.** Before any
answer is shown, its numeric tokens are checked against what the kernel actually
computed. A mismatch blocks the narration and falls back to the deterministic
template.

**Rule 3 — the kernel is exhaustive.** A full legality check costs ~1.4 ms, so
every candidate is evaluated against every rule, every time. There is no
shortlist and no heuristic pre-filter.

### Why the model writes plans, not prose

Model narration is **off by default**, and that is a measured decision rather
than a missing feature. Against the NVIDIA endpoint, re-phrasing an answer the
kernel had already computed cost **2–30 seconds**; the brief is explicit that a
45-second response is not a decision aid. The deterministic templates are
already correct and render in under a millisecond.

So the model is spent where it earns its place — turning a controller's own
words into a typed plan — and not on restating a sentence we can already write.
`CREWOPS_NARRATE=1` enables it; every model call is bounded by `CREWOPS_TIMEOUT`
and falls back to the template rather than hanging the desk.

### What happens if you delete the language model

You keep every correct answer and lose the ability to phrase questions freely.
Run any command with `--no-model` to see it: the rules grammar and deterministic
templates answer the entire shipped question set on their own. The model buys
the *frame*, not the *facts*.

---

## Correctness

```
$ python -m crewops eval

  Tier 1     16 correct · 0 wrong
  Tier 2     13 correct · 1 rubric · 0 wrong
  Tier 3     3 correct · 5 rubric · 0 wrong
  Scenarios  8 correct · 0 wrong
  Held-out   2 correct · 0 wrong
  ------------------------------------------------------
  CORRECT    42
  ABSTAINED  0
  RUBRIC     6   open-ended, graded by hand
  WRONG      0   (none)
```

Three columns, not two. **Refused is not a failure** — it is the system
declining to guess. The only red number is WRONG.

Six questions are marked RUBRIC because the dataset itself says they are
open-ended (one answer key literally reads *"judged on operational reasoning,
not exact match"*). Silently scoring those as correct would be marking our own
homework.

The **held-out** row matters most: those two checks use event shapes the shipped
questions never exercise, through the same generic code paths. Nothing in the
engine is special-cased to a question id.

---

## Non-obvious things we found

Three, all computed, none of which anyone asked for:

**One trip in the week has exactly one legal captain.** `P-2289` on 14 Sep. Of
28 captains, one could legally take it — and one senior cabin crew member. A
single point of failure sitting inside a published roster.

**The standby roster has a hole at night.** No reserve captain of any rating is
on call 19:00Z–02:00Z, and the only ATR-rated reserve captain covers just
03:00–15:00Z. A sick call at 20:00Z for an early turboprop departure has no
standby answer at all.

**One crew member is already rostered illegally.** Found on startup, before
anyone asks a question.

We deliberately did **not** build a "crew approaching duty limits" watchlist,
even though the brief suggests it. Peak 7-day utilisation across all 150 crew is
**42.51 h against a 60 h cap**, and exactly one crew-day in 900 cannot absorb an
extra duty. That panel would render empty. `brief` says so on screen.

---

## Known limits

| Limit | Why | How we detect it | What we do instead |
|---|---|---|---|
| Legality horizon = data horizon | Rosters stop 2026-09-20 | Candidate's last duty within 12 h of the boundary | Say "legal within the visible horizon; unverifiable beyond 20 Sep" |
| Optimality proven for ≤2 concurrent events | Enumerate-and-rank is the wrong algorithm class above that | Count events | Return a legal plan, drop the word "optimal" |
| Crew only | No aircraft, passenger or hotel model | Scope screen | Refuse by name |
| "Passengers" means seats | Dataset has capacity, not bookings | — | Say "seats", never imply bookings |
| Cannot detect a not-yet-valid licence | See CONF-01 below | — | Documented as an accepted risk |

Four refusal classes, kept structurally distinct, because a parse failure
wearing the costume of intellectual honesty is exposed by one probe — ask the
same answerable question two ways and get an answer once and a refusal once:

- **DATA_GAP** — "I don't hold crew phone numbers." A pre-declared contract.
- **PARSE_FAIL** — "I didn't understand that one." A defect, stated as one.
- **CLARIFY** — "I need to know which role to cover on P-2291." Answerable once
  one more detail arrives; asked rather than guessed.
- **OUT_OF_RANGE / UNKNOWN_ENTITY** — the date or the id is not in the data.

### The intent/plan agreement check

A schema-valid plan can still be the *wrong* plan, and this is the failure that
matters. Asked *"my BLR captain on the DEL overnight is out — options?"* the
model chose `find_crew(rank=Captain, base=BLR)`: schema-valid, and it answered a
completely different question while looking authoritative — *"you have 27
BLR-based captains available"*.

So when the controller clearly asks for a **resolution** and the chosen plan is
a bare lookup, the answer is thrown away and we say which part we could not pin
down. It catches bad plans from the rules lane and the model lane alike.

---

## The failure case we ship

Ask the same answerable question three ways:

| Phrasing | Result |
|---|---|
| `C-1042 is out for P-2291, what should I do?` | ✓ C-3310 @ ₹18,500 |
| `who can cover P-2291 instead of C-1042?` | ✓ C-3310 @ ₹18,500 |
| `my BLR captain on the DEL overnight is out — options?` | ✗ **no answer without a model** |

The third needs Lane B. With no `ANTHROPIC_API_KEY` the rules grammar has no
slot for "my BLR captain on the DEL overnight" — it never resolves the referring
expression to `C-1042`, so it refuses.

**This is the honest shape of the trade-off.** The deterministic lane is exact,
fast and reproducible, and it is *brittle to phrasing*. The model lane is robust
to phrasing and is the only component that can fail on stage. Shipping both, and
showing which one answered (`parsed by rules` / `parsed by model` on every
answer), is the design — not a workaround.

What we would build next: an entity-resolution pass over referring expressions
("my X at Y", "the Z captain") running in the deterministic lane, which would
move most of Lane B's traffic back to Lane A.

---

## Rulebook conformance

`python -m crewops conformance` publishes every point where the shipped rule
prose and the shipped data disagree, which reading we follow, and why. Seven
findings, in two classes:

- **Class A — interpretation.** The prose is ambiguous; the data settles it.
  Calendar-day windows, the inclusive reserve boundary tested against *report*
  time, the 14 Sep double count, two different delay semantics.
- **Class B — a genuine data defect.** `RULE-CERT-06` must compare `valid_to`
  only, because every licence in the dataset carries a `valid_from` in
  2027–2032. The semantically correct check leaves **0 of 142 active crew
  legal**. We follow the data and say loudly that this is a safety gap that must
  be fixed before the engine touches a real roster.

That report is the difference between modelling the domain and modelling the
grader.

---

## Design notes

**Overlays, not scenario switches.** The snapshot is immutable; every disruption
is a composable overlay applied on read. Five event types — `SICK_CREW`,
`STATION_CLOSURE`, `DELAY`, `CERT_EXPIRY`, `MULTI_SICK` — cover the whole space.
This is why the held-out scenarios cost zero new code.

**Cost ties are banded, not falsely ranked.** One scenario has 43 options with
36 tied at the same price. The graded sort is `(cost_inr, crew_id)` and we match
it exactly, but we also stamp a tie band and carry reachability, seniority and
risk as *context inside the band* — never as a sort key, because that would
break parity with the answer keys.

**Cases have state.** Answering two simultaneous sick calls independently
assigns the same reserve captain to both; joint recovery catches it and returns
one legal plan.

### Scalability, measured

| Component | 150 crew | ~10,000 crew | Verdict |
|---|---|---|---|
| One legality check | 1.4 ms | ~92 ms | Survives; `O(crew of rank)` |
| Whole-week precompute | 0.36 s | ~2.2 h | Dies — which is why we don't do it |
| Pairwise joint recovery | 21k pairs | 4.3×10⁸ pairs | Needs CP-SAT above ~2 events |

At 3,000 flights a day, disruptions are concurrent and interacting, so the
two-at-once case stops being the hard scenario and becomes the normal one. That
is where enumerate-and-rank must be replaced.

### Crew PII in production

This dataset is synthetic. In production, `crew.json` is personal data under the
DPDP Act and GDPR. The design already helps: the kernel needs identifiers,
qualifications and clocks — not names, contact details or medical specifics. We
would hold the roster pseudonymously, keep the identity mapping in a separate
store behind its own access control, and never place either in a model's
context. The evidence ledger is the natural enforcement point: it is a declared
allow-list of fields, so PII minimisation becomes a property of the architecture
rather than a policy document.

---

## Layout

```
crewops/
  data.py          loading, typed records, indices, immutable snapshot
  rules.py         the 7 rules as predicates + the evidence ledger
  kernel.py        check_cover, cover_options, joint recovery, fragility
  events.py        overlay engine + impact analysers + minimal repair
  tools.py         21 typed capabilities -- the only route to the kernel
  agent.py         parse, route, refuse, narrate + the containment guard
  evaluate.py      the CORRECT/ABSTAINED/WRONG harness
  conformance.py   rulebook vs data disagreements
  cli.py           the desk
tests/
  test_kernel.py   answer-key parity, traps, refusals, held-out shapes
```

Read `kernel.py` first. Every non-obvious line carries the reason it exists.

## Provenance

The starter pack included `generate.py`, the script that produced the answer
keys; we read it. The engine we ship derives from `data/*.json` only and
reproduces all six scenario answer keys — and both held-out shapes — without it.
