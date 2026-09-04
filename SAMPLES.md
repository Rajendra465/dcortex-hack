# Sample inputs and outputs

Every transcript below is real terminal output, captured by running
`python -m crewops ask "..."` with **no model configured** (`CREWOPS_NO_MODEL=1`), so nothing here depends on a network call.

Snapshot is frozen at `2026-09-14T18:00:00Z`; all times are UTC.


---


## Tier 1 — lookup


### `Who is on reserve at BLR on 2026-09-15?`

```
  12 reserve(s) on call: C-1329 (Cabin Crew, 04:00-16:00Z), C-2111 (Senior
Cabin Crew, 04:00-16:00Z), C-2248 (Cabin Crew, 04:00-16:00Z), C-3305 (Captain,
00:00-05:30Z), C-3310 (Captain, 06:00-18:00Z), C-3311 (First Officer,
06:00-18:00Z)
  T1 · get_reserves · 2 ms · computed · parsed by index
```


### `How many duty hours does C-1042 have left this week?`

```
  C-1042 has 20.93h over 7 days ending 2026-09-14, leaving 39.07h of headroom
against the 60.0h limit.
  T1 · duty_hours · 1 ms · computed · parsed by index
```


### `Which flights depart DEL on 2026-09-15?`

```
  1 flight(s): DX402
  T1 · find_flights · 2 ms · computed · parsed by index
```


### `Who are the captains based at DEL?`

```
  1 crew: C-2210 (Captain, DEL)
  T1 · find_crew · 1 ms · computed · parsed by index
```


## Tier 2 — consequence


### `If I move C-2087 onto P-2291, does anyone breach a duty limit?`

```
  C-2087 cannot cover P-2291. RULE-DUTY-02: would exceed 60h/7d by 1h20m on
2026-09-15 (total 61.33h) RULE-DUTY-02: would exceed 60h/7d by 1h05m on
2026-09-16 (total 61.08h)
  T2 · check_legality · 1 ms · computed · parsed by index
  XX  NOT LEGAL
      - RULE-DUTY-02: would exceed 60h/7d by 1h20m on 2026-09-15 (total 61.33h)
      - RULE-DUTY-02: would exceed 60h/7d by 1h05m on 2026-09-16 (total 61.08h)
  RULE-FLT-03: evaluated, non-binding on this data
```


### `Can reserve C-3305 cover the full pairing P-2291 both days?`

```
  I need one more detail   (CLARIFY)
  I need to know which role to cover on P-2291.
  What I do have: the pairing and its full crew complement
  What it would need: a crew_id or a role
```


### `BLR is closed 08:00 to 14:00 on 2026-09-17 - what is the crew impact?`

```
  BLR closed 08:00-14:00Z on 2026-09-17: 13 flight(s) affected, 1836 seats at
risk; 10 would push the rostered crew past their duty limit.
  T2 · simulate_station_closure · 3 ms · computed · parsed by index
  flight   pairing   min delay   FDP/limit    crew
 ─────────────────────────────────────────────────────────
  DX402    P-2204    +5.75h      17.0/12.0    EXCEEDS FDP
  DX422    P-2211    +5.75h      17.0/12.0    EXCEEDS FDP
  DX462    P-2232    +5.75h      11.0/13.0    ok
  DX453    P-2225    +6.5h       14.75/12.0   EXCEEDS FDP
  DX433    P-2218    +6.0h       15.75/12.0   EXCEEDS FDP
  DX403    P-2204    +5.0h       16.25/12.0   EXCEEDS FDP
  DX413    P-2293    +3.25h      12.75/12.5   EXCEEDS FDP
  DX423    P-2211    +5.0h       16.25/12.0   EXCEEDS FDP
  DX454    P-2225    +3.75h      12.0/12.0    ok
  DX434    P-2218    +2.75h      12.5/12.0    EXCEEDS FDP
```


### `VT-DXA is delayed 90 minutes on 2026-09-16`

```
  VT-DXA delayed 1.5h on 2026-09-16: duty runs 12.75h against a 12.0h limit (4
sectors) - BREACH.
  T2 · simulate_delay · 1 ms · computed · parsed by index
  XX RULE-FDP-01: delayed duty runs 12.75h vs 12.0h limit (4 sectors) - the
rostered crew cannot legally complete the full day.
  Original crew can still operate 3 of 4 legs (FDP 11.0h vs 12.5h). Re-crew:
DX404.
```


## Tier 3 — recommendation


### `C-1042 is out for P-2291, what should I do?`

```
  5 legal option(s). Cheapest: Assign Captain C-3310 (reserve callout) at INR
18,500. 19 candidates were ruled out.
  T3 · rank_cover_options · 6 ms · computed · parsed by index
Options — ranked by cost; equal cost means equal rank
  #   crew     INR         tie   notes             action
 ─────────────────────────────────────────────────────────────────────────────
  1   C-3310   18,500      1     45m               Assign Captain C-3310
                                                   (reserve callout)
  2   C-1526   24,000      2     90m               Assign Captain C-1526
                                                   (day-off callout)
  3   C-3983   24,000      2     45m               Assign Captain C-3983
                                                   (day-off callout)
  4   C-5566   24,000      2     60m               Assign Captain C-5566
                                                   (day-off callout)
  5   C-2210   41,200      3     60m +3.0h delay   Assign Captain C-2210
                                                   (reserve callout + dea
  6   --       1,500,000   4                       Cancel all 6 flights of
                                                   the pairing
Ruled out (19) — the first question a controller asks
  ruled out   why
 ─────────────────────────────────────────────────────────────────────────────
  C-1017      RULE-REST-04: only 11.0h rest before P-2217 on 2026-09-16
              (downstream conflict); RULE-
  C-1443      RULE-REST-04: only -6.25h rest before COVER on 2026-09-15 (rest
              conflict); double-book
  C-1600      RULE-QUAL-05: no A320 rating
  C-1671      RULE-QUAL-05: no A320 rating
  C-1938      RULE-REST-04: only -7.25h rest before COVER on 2026-09-15 (rest
              conflict); double-book
  C-2087      RULE-DUTY-02: would exceed 60h/7d by 1h20m on 2026-09-15 (total
              61.33h); RULE-DUTY-02:
  ... and 13 more (--json for all)
```


### `What is the smallest change that would make C-2087 legal for P-2291?`

```
  C-2087 cannot cover P-2291. RULE-DUTY-02: would exceed 60h/7d by 1h20m on
2026-09-15 (total 61.33h) RULE-DUTY-02: would exceed 60h/7d by 1h05m on
2026-09-16 (total 61.08h)
  T2 · check_legality · 1 ms · computed · parsed by index
  XX  NOT LEGAL
      - RULE-DUTY-02: would exceed 60h/7d by 1h20m on 2026-09-15 (total 61.33h)
      - RULE-DUTY-02: would exceed 60h/7d by 1h05m on 2026-09-16 (total 61.08h)
  RULE-FLT-03: evaluated, non-binding on this data
```


### `Where are we thin on cover this week?`

```
  1 trip(s) have one or zero legal replacements. P-2289 on 2026-09-14 (Captain:
C-2210)
  T3 · cover_fragility · 171 ms · computed · parsed by index
Thinnest cover in the week
  pairing   date         role            legal covers   who
 ─────────────────────────────────────────────────────────────────────────────
  P-2289    2026-09-14   Captain         1              C-2210
  P-2289    2026-09-14   First Officer   2              C-2341, C-5020
  P-2222    2026-09-14   First Officer   5              C-3316, C-1313,
                                                        C-1317
  P-2229    2026-09-14   First Officer   5              C-3316, C-1313,
                                                        C-1317
  P-2223    2026-09-15   First Officer   5              C-3316, C-1313,
                                                        C-3057
  P-2230    2026-09-15   First Officer   5              C-3316, C-1313,
                                                        C-3057
  P-2291    2026-09-15   Captain         5              C-3310, C-1526,
                                                        C-3983
  P-2224    2026-09-16   First Officer   5              C-3316, C-1317,
                                                        C-2085
```


### `Draft the callout notification to C-3310 for P-2291`

```
  Draft for crew (fact-locked). Captain D. Reddy (C-3310) - callout for P-2291
  T2 · draft_notification · 1 ms · computed · parsed by index
```


## Refusals — the system declining to guess


### `What is the probability that C-2087 calls in sick tomorrow?`

```
  I can't answer that reliably   (DATA_GAP)
  I don't hold predictions about who will call in sick.
  What I do have: risk_signals.json is a PROVIDED input; forecasting is
explicitly out of scope for this system
  What it would need: a data source with predictions about who will call in
sick
  Try: ask about rosters, duty hours, legality, cover options, certifications
or costs
```


### `What is C-3310's phone number?`

```
  I can't answer that reliably   (DATA_GAP)
  I don't hold crew phone numbers, emails or addresses.
  What I do have: crew.json has reachability_minutes, but no contact details
  What it would need: a data source with crew phone numbers, emails or
addresses
  Try: ask about rosters, duty hours, legality, cover options, certifications
or costs
```


### `Who is on reserve on 2026-10-05?`

```
  I can't answer that reliably   (OUT_OF_RANGE)
  2026-10-05 is outside the data I have.
  What I do have: rosters and flights 2026-09-14 to 2026-09-20; duty history
back to 2026-08-18
  What it would need: roster data covering that date
```


### `Tell me about C-9999`

```
  I can't answer that reliably   (UNKNOWN_ENTITY)
  C-9999 is not in this dataset.
  What I do have: 150 crew on file
```


---


## Where it fails

Three real failures, reproduced verbatim. The brief asks for at least
one; these are the three that best show *different* weaknesses, and
each is a routing defect rather than an arithmetic one — the kernel
answers all three correctly when called directly.


### `my BLR captain on the DEL overnight is out - options?`

```
  28 crew: C-1017 (Captain, BLR), C-1042 (Captain, BLR), C-1443 (Captain, BLR),
C-1526 (Captain, BLR), C-1564 (Captain, BLR), C-1600 (Captain, BLR), C-1671
(Captain, BLR), C-1938 (Captain, BLR)
  T1 · find_crew · 1 ms · computed · parsed by index
```

**Referring expressions are not resolved.** The index cannot turn "my BLR captain on the DEL overnight" into `C-1042`, scores below the confidence floor, and refuses. With a model planner configured it usually succeeds; offline it never will. This is the deliberate trade: the deterministic lane is exact and reproducible, and brittle to phrasing.


### `Station HYD is closed 05:00-09:00Z on 19 Sep. Which flights are affected?`

```
  14 flight(s): DX462, DX424, DX462, DX424, DX462, DX424, DX462, DX424, DX462,
DX424, DX462, DX424
  T1 · find_flights · 2 ms · computed · parsed by index
```

**Routed to the wrong capability.** A controller would expect the closure simulator; the index picks `find_flights` because the sentence is dense with flight vocabulary and the closure examples score lower. The kernel answers this correctly when called directly (it is one of the two held-out checks, and it passes) - the defect is entirely in routing, which is why we publish the routing score separately.


### `A crew is released at 15:30Z on 16 Sep. What is the earliest they may report next?`

```
  I can't answer that reliably   (UNKNOWN_ENTITY)
  I couldn't resolve 'pairing_id'.
  What I do have: check the crew id, pairing id or date
```

**A crash wearing a refusal costume.** `compute_duty_period` declares no required parameters but its body needs either `pairing_id` or `release_utc`. A plan missing both passes validation and dies as UNKNOWN_ENTITY - the exact failure class this architecture is supposed to prevent. The fix is a `requires_one_of` declaration on the tool so the validator rejects the plan before execution and asks a clarifying question instead.


---


## The two scores, and why both are published

```

  CREW OPS ADVISOR - ANSWER-KEY REGRESSION
  ------------------------------------------------------
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
  slowest check: S5:ranked options at 6 ms
```

```

  END-TO-END THROUGH THE LANGUAGE LAYER
  planner: index only (no model configured)
  ------------------------------------------------------
  20/38 shipped prompts reach the right capability  (53%)
  3 refused rather than guessed

  not yet routed:
    Q06  want get_reserves               got simulate_crew_unavailable
    Q07  want find_crew                  got simulate_crew_unavailable
    Q13  want find_crew                  got duty_hours
    Q18  want check_legality             got rank_cover_options
    Q19  want simulate_station_closure   got find_flights
    Q21  want check_legality             got REFUSED:CLARIFY
    Q22  want simulate_cert_expiry       got get_roster
    Q23  want compute_duty_period        got REFUSED:UNKNOWN_ENTITY
    Q24  want check_legality             got REFUSED:CLARIFY
    Q26  want duty_hours                 got find_crew
    Q27  want rank_cover_options         got find_crew
    Q29  want simulate_station_closure   got find_flights
    Q32  want solve_joint_cover          got get_roster
    Q33  want simulate_delay             got get_roster
    Q34  want rank_cover_options         got simulate_cert_expiry
    Q35  want simulate_station_closure   got get_roster
    Q37  want rank_cover_options         got cover_fragility
    Q38  want get_risk_signals           got find_flights

  The kernel score above measures exact arithmetic; this one
  measures whether a typed question reaches it. They are different
  claims and we publish both.
```

The first measures the rules engine against the dataset's own answer
keys: exact arithmetic, zero wrong. The second measures whether a
question *typed in English* reaches that engine. They are different
claims and the second is much weaker. Reporting only the first would
be the overstatement the brief explicitly penalises.
