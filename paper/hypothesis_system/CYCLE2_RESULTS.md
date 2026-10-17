# Cycle 2 results: sessions w2-K-b, w2-K-blinded, w2-K-shuffled

Date: 2026-10-16. Setup: `CYCLE2_PREP.md`. Code: `training/scilearn/`. Ledgers: `/home/max/scilearn-private/ledger/`. All gains below are dev (Dpost) log loss per game; positive means the term helps. z values are the gate z; "thr" is the session's threshold for that group at test time. Bits are dev bits for predictive and explanation tests and mediator bits for mechanism tests.

## 0. At a glance

| Arm | Cards | Predictive accepted | Explanations accepted | Mechanisms shown | Dev bits (of 120) | Mediator bits (of 60) |
|---|---|---|---|---|---|---|
| w2-K-b | 25 | 1 of 8 | 0 of 7 | 3 of 10 | 25.9 | 15.9 |
| w2-K-blinded | 17 | 2 of 3 (a third, p03, voided by the parameter cap) | 0 of 6 | 1 of 8 | 18.4 | 13.5 |
| w2-K-shuffled | 29 (28 tested) | 0 of 28 | none tested (no table to explain) | 0 of 4 (run on rejected terms) | 42.2 | 10.7 |

- **Explanations: 0 of 13.** Kit and role theories explained -1% to 4% of the confirm lookup tables. None reached the explain bar (best z 1.32 against 3.0). None compressed.
- **Mechanisms: 4 of 18 in the real arms.** The hero x map table works through structures lost by 15 minutes in both arms: w2-K-blinded m02 (towers, z 4.42) and w2-K-b m06 (forts, z 4.45). Hero x tier works through the first fort lost (z 4.12) and counters through the level lead at 10 minutes (z 11.64).
- **Shuffled arm:** 0 of 28 accepted, z from -2.15 to 2.25. Both known-true canaries were rejected, as they should be on shuffled labels.
- **Incumbents on Dpost:** w2-K-b ends at 0.67223 log loss and 58.24% accuracy, 0.0003 better than A0 (SE 0.00006). w2-K-blinded ends at 0.67303 and 58.11%, 0.0005 worse than A0. Both beat K0 by about 0.008 and N by about 0.006.
- **Harness:** the w2-K-blinded "stall" was two incumbent refits holding both 3090 slots for 1 h 50 min. No job was lost. The seven pending results landed 6 to 31 minutes after the session exited. Fixed, along with a missing parameter-cap check that let p03 be accepted (section 6).

## 1. w2-K-b

Incumbent at start: K0 plus wave-1 c01 role multiset, c02 hero x tier, c04 hero x map, c05 ally pairs, c06 counters.

### Predictive (8 tested, 1 accepted)

| Card | Term | Decision | z | thr | Gain (SE) | Released gain | Bits |
|---|---|---|---|---|---|---|---|
| p01_recent_practice_5d | decay, half-life 5 days | **accept** | 3.77 | 3.72 | +0.000106 (0.000028) | 0.0001 | 1.5 |
| p02_range_by_map | product range by map | reject | 0.94 | 3.00 | +0.000001 (0.000001) | 0.0000 | 1.5 |
| p03_siege_by_tier | product siege by tier | reject | 0.09 | 3.00 | 0.000000 | 0.0000 | 1.5 |
| p04_sitting_ducks | indicator, low mobility and low peel | reject | 0.18 | 3.00 | 0.000000 (0.000001) | 0.0000 | 1.5 |
| p05_no_force | indicator, low hard CC and short range | reject | -0.13 | 3.00 | -0.000001 (0.000004) | 0.0000 | 1.5 |
| p06_no_soak_no_answer | indicator, low waveclear and no global | reject | -4.14 | 3.00 | -0.000009 (0.000002) | 0.0000 | 1.5 |
| p07_waveclear_by_map | product waveclear by map | reject | -1.42 | 3.00 | 0.000000 | 0.0000 | 1.5 |
| p08_double_tank_high | rule, 2+ tanks at high tier | reject | 0.36 | 4.66 | +0.000001 (0.000004) | 0.0000 | 1.5 |

### Explanation (7 tested, 0 accepted)

Share is the explained share of the confirm table (Dpre+G fit, 360 to 3,997 cells). Bits saved against the theory's description cost decides compression.

| Card | Table | Theory (parameters) | Share | z | thr | Bits saved / cost | Transfer share (z) | Onset z | Kit perm. p | Bits |
|---|---|---|---|---|---|---|---|---|---|---|
| e01 | c04 hero x map | range by map + waveclear by map (30) | 0.044 | 1.25 | 3.00 | 59.4 / 122.3 | 0.039 (0.73) | -0.53 | 0.005 | 1.5 |
| e02 | c02 hero x tier | siege by tier + sustained damage by tier (8) | 0.036 | 0.61 | 3.00 | 12.0 / 25.5 | 0.009 (0.19) | 0.61 | 0.095 | 1.5 |
| e03 | c05 ally pairs | 3 kit pairs (3) | 0.008 | 0.94 | 3.00 | 7.3 / 17.9 | 0.021 (1.86) | -0.68 | 0.005 | 1.5 |
| e04 | c06 counters | mobility x waveclear (1) | 0.001 | 0.29 | 3.00 | 2.4 / 6.0 | 0.002 (0.46) | 0.95 | 0.050 | 1.5 |
| e05 | c02 hero x tier | Tank role by tier (4) | -0.011 | -0.41 | 3.72 | -3.7 / 12.7 | -0.037 (-1.43) | -0.17 | role only | 1.5 |
| e06 | c06 counters | global presence x range (1) | 0.001 | 0.27 | 3.72 | 1.1 / 6.0 | 0.003 (1.46) | 0.19 | 0.020 | 1.5 |
| e07 | c04 hero x map | Tank role by lanes2 (1) | 0.000 | 0.00 | 3.72 | 0.0 / 5.1 | 0.000 (0.00) | 1.13 | role only | 1.5 |

### Mechanism (10 tested, 3 shown)

z is the mediator z in the declared direction. Std effect is the change in the mediator, in SDs, per SD of the term's fitted contribution. The negative control must stay under |z| 2.

| Card | Term | Mediator (dir.) | z | thr | Std effect | Control (z) | Result | Bits |
|---|---|---|---|---|---|---|---|---|
| m06 | c04 hero x map | forts_lost_15 (-) | 4.45 | 3.00 | -0.053 | first_blood (0.33) | **shown** | 1.5 |
| m08 | c02 hero x tier | first_fort_lost (-) | 4.12 | 3.00 | -0.041 | first_blood (-1.47) | **shown** | 1.5 |
| m10 | c06 counters | level_lead_10 (+) | 11.64 | 3.00 | +0.088 | first_blood (1.55) | **shown** | 1.5 |
| m01 | c06 counters | fights_won_10 (+) | 7.78 | 3.00 | +0.073 | creep_xp_share_10 (4.15) | control moved | 1.5 |
| m02 | c04 hero x map | towers_lost_15 (-) | 4.35 | 3.00 | -0.053 | creep_xp_share_10 (3.05) | control moved | 1.5 |
| m07 | c05 ally pairs | level_lead_15 (+) | 4.25 | 3.00 | +0.036 | first_blood (2.40) | control moved | 1.5 |
| m04 | c02 hero x tier | xp_lead_10 (+) | 2.08 | 3.00 | +0.029 | creep_xp_share_10 (0.86) | below bar | 1.5 |
| m03 | c05 ally pairs | fights_won_post10 (+) | 1.51 | 3.00 | +0.029 | creep_xp_share_10 (-0.44) | below bar | 1.5 |
| m05 | c01 role multiset | isolated_deaths_10 (-) | -0.86 | 3.00 | +0.015 | creep_xp_share_10 (-0.53) | wrong way | 1.5 |
| m09 | c01 role multiset | deaths_10 (-) | -1.25 | 3.00 | +0.018 | first_blood (-0.88) | wrong way | 1.5 |

Budget: 25 cards, 25.9 dev bits (22.5 on tests, 3.4 on Q1), 15.9 mediator bits, 4,580 sandbox numbers, 1 inspection.

## 2. w2-K-blinded

Incumbent at start: K0 plus wave-1 c01 role multiset, c04 hero x tier, c10 ally pairs (10K games), c11 counters (10K games), c12 hero x map.

### Predictive (3 tested, 2 accepted, 1 voided)

| Card | Term | Decision | z | thr | Gain (SE) | Released gain | Bits |
|---|---|---|---|---|---|---|---|
| p01_cross_counters_5k | counters, 5K-game support | **accept** | 12.15 | 3.72 | +0.000618 (0.000051) | 0.0006 | 1.5 |
| p02_warm_hands | decay, half-life 4 days | **accept** | 3.79 | 3.72 | +0.000109 (0.000029) | 0.0001 | 1.5 |
| p03_ally_pairs_5k | ally pairs, 5K-game support | accept, then **voided** (harness error) | 5.90 | 4.02 | +0.000132 (0.000022) | 0.0002 | 1.5, refunded |

p03 passed its gate, but the incumbent plus p03 has 20,221 parameters, over the 20,000 cap. Section 6.3 has the details.

### Explanation (6 tested, 0 accepted)

| Card | Table | Theory (parameters) | Share | z | thr | Bits saved / cost | Transfer share (z) | Onset z | Kit perm. p | Bits |
|---|---|---|---|---|---|---|---|---|---|---|
| e01 | c12 hero x map | range by map (15) | 0.037 | 1.32 | 3.00 | 49.9 / 61.2 | 0.006 (0.13) | -0.74 | 0.010 | 1.5 |
| e02 | c11 counters | global presence x range (1) | 0.001 | 0.36 | 3.00 | 2.1 / 5.5 | 0.008 (2.23) | -0.41 | 0.045 | 1.5 |
| e03 | c04 hero x tier | Tank role by tier + early_vs_late by tier (8) | -0.004 | -0.05 | 3.00 | -1.2 / 25.5 | 0.031 (0.52) | 0.67 | 0.124 | 1.5 |
| e04 | c10 ally pairs | range x waveclear (1) | 0.007 | 0.64 | 3.00 | 3.2 / 5.3 | 0.033 (2.36) | -1.91 | 0.005 | 1.5 |
| e05 | c11 counters | mobility x range (1) | -0.000 | -1.53 | 3.72 | -0.6 / 5.5 | 0.000 (0.00) | 0.65 | 0.542 | 1.5 |
| e06 | c10 ally pairs | self-sustain x waveclear (1) | 0.003 | 0.38 | 3.72 | 1.4 / 5.3 | 0.023 (2.10) | -0.74 | 0.005 | 1.5 |

### Mechanism (8 tested, 1 shown)

| Card | Term | Mediator (dir.) | z | thr | Std effect | Control (z) | Result | Bits |
|---|---|---|---|---|---|---|---|---|
| m02 | c12 hero x map | towers_lost_15 (-) | 4.42 | 3.00 | -0.053 | first_blood (0.32) | **shown** | 1.5 |
| m03 | c12 hero x map | forts_lost_15 (-) | 4.50 | 3.00 | -0.053 | deaths_10 (-4.03) | control moved | 1.5 |
| m01 | c11 counters | xp_lead_10 (+) | 9.18 | 3.00 | +0.075 | creep_xp_share_10 (2.59) | control moved | 1.5 |
| m06 | c11 counters | isolated_deaths_10 (-) | 5.57 | 3.00 | -0.055 | creep_xp_share_10 (4.08) | control moved | 1.5 |
| m07 | c12 hero x map | level_lead_5 (+) | 2.81 | 3.00 | +0.026 | creep_xp_share_10 (3.11) | below bar, control moved | 1.5 |
| m04 | c10 ally pairs | fights_won_10 (+) | 1.63 | 3.00 | +0.014 | creep_xp_share_10 (-0.62) | below bar | 1.5 |
| m05 | c04 hero x tier | deaths_10 (-) | 0.84 | 3.00 | -0.013 | creep_xp_share_10 (1.06) | below bar | 1.5 |
| m08 | c01 role multiset | level_lead_10 (+) | -1.96 | 3.00 | -0.020 | creep_xp_share_10 (-1.73) | wrong way | 1.5 |

Budget: 17 cards, 18.4 dev bits (12.0 on tests after the p03 refund, 6.4 on Q1), 13.5 mediator bits, 3,724 sandbox numbers, 1 inspection.

The session's own `summary.md` was written at 22:02 UTC, before its last seven results landed, so it lists e03, p03, m01 and m05 to m08 as pending. Their results are in the tables above and in the ledger (seq 331 to 339; p03 voided at seq 340).

## 3. w2-K-shuffled

Incumbent: K0 fit on the session's shuffled labels. No accepted terms, so there was no table to explain and no accepted term to attach a mechanism card to.

### Predictive (28 tested, 0 accepted)

| Card | Term | z | thr | Gain | Card | Term | z | thr | Gain |
|---|---|---|---|---|---|---|---|---|---|
| d01 | decay 10 d | -2.03 | 3.00 | -0.000001 | c19 | lookup hero x map class | -0.52 | 3.00 | -0.000001 |
| d02 | decay 120 d | 0.00 | 3.72 | 0.000000 | c20 | product hard CC by tier | -1.74 | 3.00 | -0.000005 |
| c05 | indicator no healer | -1.28 | 3.76 | -0.000001 | c21 | product siege by map | 0.94 | 3.00 | +0.000001 |
| c06 | product global by map | -2.15 | 3.00 | -0.000001 | c22 | product range by map | -0.81 | 3.00 | -0.000002 |
| c07 | product mobility by tier | 0.91 | 3.00 | 0.000000 | c23 | product healing by map | 2.25 | 3.00 | +0.000005 (near miss) |
| c08 | bilinear peel vs mobility | -0.71 | 3.00 | 0.000000 | c23b | product healing by map class | -0.38 | 3.72 | -0.000001 |
| c09 | bilinear waveclear vs siege | 0.00 | 3.00 | 0.000000 | c24 | bilinear mobility vs burst | 0.00 | 3.00 | 0.000000 |
| c10 | lookup hero x map | 1.44 | 3.00 | +0.000015 | c25 | bilinear siege vs early_vs_late | -0.58 | 3.00 | 0.000000 |
| c11 | product early_vs_late by tier | 0.70 | 3.00 | 0.000000 | c26 | bilinear range vs hard CC | -0.27 | 3.00 | 0.000000 |
| c12 | product waveclear by map | 0.10 | 3.00 | 0.000000 | c27 | product peel by tier | 0.11 | 3.00 | 0.000000 |
| c13 | bilinear healing vs range | 0.00 | 3.00 | 0.000000 | c30 | product sustained by map | -0.57 | 3.00 | -0.000001 |
| c14 | bilinear shields vs burst | -1.03 | 3.00 | 0.000000 | c31 | product self-sustain by map class | 0.51 | 3.00 | 0.000000 |
| c15 | bilinear global vs siege | 0.97 | 3.00 | 0.000000 | c34 | product burst by map | 0.31 | 3.00 | +0.000001 |
| c16 | indicator no tank | -1.53 | 4.05 | -0.000001 | c18 | lookup hero x tier | -0.80 | 3.00 | -0.000003 |

Each test cost 1.5 dev bits; every released gain was 0.0000.

### Mechanism tests on rejected terms (4, none shown)

| Card | Mediator (dir.) | z | thr | Std effect | Control (z) | Bits |
|---|---|---|---|---|---|---|
| c12 | minion_xp_share_10 (+) | -1.62 | 3.00 | -0.015 | first_blood (-0.60) | 1.5 |
| c05 | deaths_10 (-) | 0.42 | 3.00 | -0.037 | creep_xp_share_10 (0.25) | 1.5 |
| c13 | deaths_10 (-) | 0.00 | 3.72 | 0.000 | minion_xp_share_10 (0.00) | 1.5 |
| c14 | fights_won_post10 (+) | -0.15 | 3.00 | -0.002 | minion_xp_share_10 (-0.72) | 1.5 |

Budget: 29 cards (28 tested, c28 withdrawn), 42.2 dev bits, 10.7 mediator bits, 564 sandbox numbers, 1 inspection.

## 4. Canaries

Q2 canaries, one set per session, from each session's `logs/canaries.out` (harness ledger seq 66 to 83). Null canaries are scored against the session's incumbent. Known-true canaries are scored against the starting K0 backbone, so in w2-K-b and w2-K-blinded they are the same test on the same data and give the same numbers.

| Canary | w2-K-b z | w2-K-blinded z | w2-K-shuffled z | Should |
|---|---|---|---|---|
| null hash 1 | 0.10 | -0.70 | 1.44 | reject |
| null hash 2 | -1.77 | 0.28 | -0.55 | reject |
| null perm. 1 | 1.14 | -0.90 | -0.65 | reject |
| null perm. 2 | -0.06 | 2.93 (invariance failed) | 1.96 | reject |
| true role multiset | 4.73, gain +0.00022 | 4.73 | -0.19 | accept (real labels) |
| true ally pairs | 8.95, gain +0.00147 | 8.95 | 1.72 | accept (real labels) |
| **Result** | 4 nulls rejected, 2 true accepted | 4 nulls rejected, 2 true accepted | all 6 rejected | |

- **Nulls:** 0 of 12 accepted this cycle. With cycle 1, 0 of 24. The closest was w2-K-blinded null perm. 2 at z 2.93, below 3.0, and it also failed invariance.
- **Shuffled known-true:** both rejected, as they should be when labels are shuffled.
- **Explanation null canaries** (before launch, harness ledger seq 58 to 65, `CYCLE2_PREP.md` 3.4): 0 of 8 accepted, z from -2.60 to 0.48.

## 5. Incumbents on Dpost

Scored on Dpost, 173,394 games. A difference is the paired log-loss difference (negative means the incumbent is better), with the week x player-cluster bootstrap SE used in `results/baselines.md`. The reference rows are from that file.

| Model | Log loss | Acc. | vs K0 | vs A0 | vs N | Acc. vs K0 / A0 / N |
|---|---|---|---|---|---|---|
| K0 (fit S+Dpre+G) | 0.68086 | 56.37% | ref | | | |
| A0 (fit S+Dpre+G) | 0.67253 | 58.14% | | ref | | |
| N (drift-aware MLP) | 0.67896 | 56.82% | | | ref | |
| w2-K-b start (K0 + wave 1) | 0.67248 | 58.22% | -0.00838 (0.00033) | -0.00006 (0.00004) | -0.00648 (0.00035) | +1.85 / +0.09 / +1.40 pp |
| **w2-K-b end** (+ p01), inc_be8151b00daa906a | **0.67223** | **58.24%** | -0.00863 (0.00034) | **-0.00030 (0.00006)** | -0.00673 (0.00036) | +1.87 / +0.11 / +1.42 pp |
| w2-K-blinded start (K0 + wave 1) | 0.67427 | 57.91% | -0.00658 (0.00030) | +0.00174 (0.00015) | -0.00468 (0.00037) | +1.53 / -0.23 / +1.08 pp |
| **w2-K-blinded end** (+ p01, p02), inc_1a768a53b59ba3dc | **0.67303** | **58.11%** | -0.00782 (0.00032) | **+0.00050 (0.00012)** | -0.00592 (0.00037) | +1.74 / -0.02 / +1.29 pp |
| w2-K-shuffled (K0, shuffled labels), inc_a3d8db7a258a8df1 | 0.69334 | 50.17% | scored on its own shuffled labels; a coin scores 0.69315 | | | |

- Cycle 2 moved w2-K-b by 0.00025 and w2-K-blinded by 0.00124. Most of the blinded gain is p01, the 5K-game counter family (+0.0006 on its own Q2 test).
- w2-K-b's incumbent now beats A0 on Dpost by 0.0003 (about 5 SE); at the start of cycle 2 the two were level (-0.00006, SE 0.00004). w2-K-blinded is still 0.0005 behind A0.
- **Dpost is these arms' dev tranche.** Every accepted term was selected on it, so these numbers favour the incumbents. The clean comparison is the C batch.

## 6. Harness problems and fixes

### 6.1 The w2-K-blinded "stall"

Times in UTC on Oct 15. Job times are from the job directories, ledger times from the ledgers.

| Time | Event |
|---|---|
| 20:18 | w2-K-blinded accepts p01. Tests started after this need incumbent K+p01. |
| 20:23 | w2-K-b accepts p01. Its later tests need its own refit. |
| 20:25 | w2-K-blinded accepts p02. Tests started after this need K+p01+p02. |
| 20:29 | Refit inc_c1aafe84 (K+p01) launches and takes 3090 slot 1. It was already out of date. |
| 20:35 | Refit inc_1a768a53 (K+p01+p02) launches and takes slot 2. From here no other scilearn job of any session can run. |
| 22:04 | The w2-K-blinded container exits, with 7 jobs pending. |
| 22:10, 22:18 | The two refits finish (101 and 103 minutes). w2-K-b's refit inc_be8151 starts at 22:12. |
| 22:10 to 22:35 | All 7 pending w2-K-blinded jobs run (seconds to 4 minutes each), are gated, and are charged by the broker. |
| 23:30 to 23:50 | w2-K-b's refit finishes (78 minutes), then its confirm refit (14 minutes on the 3090), then its 15 queued tests. |

**Root cause.** Each accept invalidates the incumbent. Each test, explanation or mechanism test started after an accept first waits for a full refit on S+Dpre+G (2.28M games, 17,700 to 18,900 parameters, 78 to 103 minutes in cycle 2). Refits took slots from the same pool of 2 that serves every session's tests. Two refits filled the pool. The first refit was already superseded when it launched, because tests copy the session state when they start and two threads had started between the accepts.

It was not a deadlock, a lost job or a stale lock:
- `/proc/locks` showed no flock held by any broker;
- every registered job completed on the 3090;
- every test was gated and charged by the broker thread, without any poll from the session.

The session stopped polling 6 minutes before the first result came back. It had no way to tell "waiting behind a refit" from "stuck".

**Why w2-K-b's tests took hours** (its summary says up to about 6 h; the ledger shows at most 3 h 10 min from card registration to result). Its p01 accept at 20:23 needed a refit. That refit could not start until a slot freed at 22:10, and then ran for 78 minutes. Its explanation cards also waited for a 14-minute confirm refit on the 3090. The tests themselves took 1 to 4 minutes, and the mechanism tests about 5 seconds. The wait was scheduling.

**The refit compute itself.** Every EB round of every incumbent fit stops at L-BFGS maxiter 400 without meeting its tolerance (gtol 1e-6 on a sum over 2.28M games). A refit is therefore a fixed 2,000 iterations. A warm start from the previous incumbent with a relative stopping rule would cut that, but it would change how incumbents are fit in the middle of the study. I left it for a decision.

**Fixes** (commits 276c9bf, 3a6a7c4):
- **Refit token.** A refit on the 3090 first takes a single refit token (`locks/refit_slot.lock`). At most one refit holds a slot, so one slot always serves tests.
- **Supersession.** Once a caller holds the token, it rereads the session's live state. If more terms were accepted since its copy, it does not launch the stale refit. It retries on the newest incumbent instead. In the cycle-2 sequence this drops the K+p01 refit.
- **Confirm fits on the 5090.** Confirm-table fits (Dpre+G, about 3 minutes, under 11 GB) run on the 5090 CPU one at a time, instead of holding a 3090 slot for 14 minutes.
- **Visibility.** `hyp test-status` now shows a `stage` for each pending card: waiting for a refit (with the job and its expected length), queued for a slot, or running since a given time. `PROPOSER_GUIDE.md` says that a refit takes about 100 minutes and that a pending test is never lost.
- **Regression test.** `PYTHONPATH=. python -m training.scilearn.tests.test_refit_scheduling` drives the real `finish_q2`, `ensure_fit`, `remote_slot` and `remote.submit` against a fake 3090 on the cycle-2 sequence: two accepts with a test copied between them, a second session that needs a refit, and a short test. New code: 11/11. The stale refit is never launched, at most one refit runs at a time, and the short test runs while refits are pending. The old scheduling on the same events launches 3 refits, 2 at once, and blocks the short test. `test_incumbent_race` still passes 7/7 and `test_ledger` 5/5.

### 6.2 The drained jobs

All seven were run, gated and charged by the w2-K-blinded broker before it was stopped, so their results and charges are in the ledger exactly as if the session had polled.

| Card | Ledger seq | Finished | Result | Charge |
|---|---|---|---|---|
| m05_tier_deaths | 331 | 22:10 | not shown (z 0.84) | 1.5 mediator |
| m07_heromap_early_levels | 332 | 22:18 | not shown (z 2.81, control 3.11) | 1.5 mediator |
| m08_multiset_deaths | 333 | 22:29 | not shown (z -1.96) | 1.5 mediator |
| m06_counters_pickoffs | 334 | 22:29 | not shown (z 5.57, control 4.08) | 1.5 mediator |
| p03_ally_pairs_5k | 335, 336 | 22:33 | accept (z 5.90 against 4.02); voided at seq 340 | 1.5 dev, refunded |
| e03_tier_tank_tempo | 337, 338 | 22:34 | reject (share -0.004, z -0.05) | 1.5 dev |
| m01_counters_xp | 339 | 22:35 | not shown (z 9.18, control 2.59) | 1.5 mediator |

None of the seven hit a harness error during its run. The gate used each test's own incumbent:
- e03, which started before p01's accept, used the starting incumbent and confirm table;
- m05, which started between the two accepts, used K+p01;
- the rest used K+p01+p02.

### 6.3 Parameter cap: p03 accepted, then voided

The closeout refit with p03 (inc_f6c9056e493af396) failed at compile time: "20221 parameters exceed the cap of 20000". FRAMING 6.1 caps a model at 20K parameters. The Q2 job fits a term against the incumbent's frozen logit and never compiles the two together, so the cap was never checked before the accept. The numbers:
- the incumbent K+p01+p02 has 17,668 parameters;
- p03, a 5K-game pair family, has 2,553;
- the total is 20,221.

Following the existing rule (a harness error is charged nothing):
- an `error` record (seq 340) voids the accept and refunds the 1.5 bits on `family:pair_family:`, net 0;
- p03 is out of `accepted_terms`, its status is `error`, and its released result is kept as `voided_release`.

**Fix** (3a6a7c4): the gate now loads the incumbent's parameter count. If the incumbent plus the term's `term_k` exceeds the cap, the card is refused free, before any result is computed or released, and stays testable in a smaller form. Covered by 3 cases in the regression test.

p03's signal was real: z 5.90 and gain +0.00013. It overlaps the inherited c10 (pairs with 10K games), whose cells it contains. Replacing c10 with p03 would fit under the cap. That is a modelling decision for cycle 3.

### 6.4 Close-out

- The three brokers were stopped with `session stop`.
- `ledger verify`: all 13 chains OK:
  - w2-K-b: 342 records;
  - w2-K-blinded: 341;
  - w2-K-shuffled: 495;
  - harness: 87.
- `session closeout <sid>` (new) wrote one `closeout` record per session. Each lists every card's final status, the accepted terms and the final incumbent. It refuses while a broker runs, while a test is pending, or before that incumbent is fitted.
- **Incumbents hold every accepted term**, checked against each model's specs:

| Session | Final incumbent | Parameters | Terms |
|---|---|---|---|
| w2-K-b | inc_be8151b00daa906a | 18,875 | 6 (5 from wave 1, plus p01) |
| w2-K-blinded | inc_1a768a53b59ba3dc | 17,668 | 7 (5 from wave 1, plus p01 and p02) |
| w2-K-shuffled | inc_a3d8db7a258a8df1 | 9,157 | K0 only |

### 6.5 Two rule gaps, not fixed

- **Near-miss retries under a new card id are charged once.** The double charge is keyed to retesting the same card. w2-K-b p01 (a retry of wave-1 c07), w2-K-blinded p02 (of wave-1 c06) and w2-K-shuffled c23b (of c23) were each new cards, so each paid 1.5 bits, not 3. The decisions do not change, because the threshold depends only on bits already on the group. The ledgers under-count those three groups by 1.5 bits each.
- **Released gains of 0.0000 come from rounding.** The shuffled session asked about this. Its private gains ran from -0.000005 to +0.000015, all below the 0.0001 release step.

## 7. Conclusions

**Explanation cards.** 13 kit or role theories were tested across the two real arms, and none was accepted.
- **Explained share:** from -1.1% to 4.4% of the confirm tables, with explain z at most 1.32 against a bar of 3.0. The two largest shares (4.4% and 3.7%) came from by-map theories with 15 and 30 parameters, so they cost more bits than they saved.
- **Transfer:** to 20 held-out heroes, at most 3.9% of the table. Three single-parameter pair and counter theories in w2-K-blinded had transfer z above 2, but they explained under 1% of the table.
- **Onset:** no theory changed onset log loss (all |z| under 2).
- **Kit permutation:** 8 of 11 kit theories beat within-role kit permutations (p at most 0.05). What they explain survives a within-role shuffle of the kit, but it is too little to matter.
- **Sandbox versus confirm:** both sessions saw shares of 4% to 13% on their sandbox (S) tables fall 3 to 10 times on the confirm tables.

Plainly: kit axes explain about 0 to 5% of what the hero x map, hero x tier, pair and counter tables hold, and that small part does not transfer to new heroes. Both sessions concluded that the pair and counter tables hold hero-specific effects that no kit axis describes.

**Mechanisms shown.** 4 of 18 mechanism tests passed in the two real arms:
- **Hero x map works through structures lost by 15 minutes.** In w2-K-blinded m02, the table's fitted advantage predicts fewer towers lost by 15 minutes (z 4.42, -0.053 SD per SD), with first blood flat. In w2-K-b m06, the same holds for forts (z 4.45, -0.053 SD). These are separate sessions with different labels but nearly the same table, so it counts as one finding seen twice. It is not exclusive to structures: w2-K-blinded m03 found forts lost moved too (z 4.50), but early deaths also moved (z -4.03). The map advantage shows in early deaths as well.
- **Hero x tier works through the first fort lost** (w2-K-b m08, z 4.12).
- **Counters work through the level lead at 10 minutes** (w2-K-b m10, z 11.64, +0.088 SD), with first blood flat. Fights won by 10 minutes and the XP lead at 10 minutes moved too (z 7.78 and 9.18), but those two tests used a control that also moved.
- **Not shown:** ally pairs and the role multiset, in either arm. The multiset tests moved their mediators the wrong way or not at all.
- **Controls:** the XP-share control (creep_xp_share_10) moved in 5 of the 11 tests that used it. First blood moved in 1 of 6. Draft-dependent XP shares make poor negative controls, which w2-K-b also noted.

**Shuffled arm.** The arm behaved as a null arm should:
- 0 of 28 predictive tests were accepted;
- 1 of 28 cleared z 2 (c23, z 2.25), about what chance gives;
- its known-true canaries were rejected (z -0.19 and 1.72);
- the 4 mechanism tests on its rejected terms showed nothing (|z| at most 1.62).

Across both cycles it has tested 32 cards with no accept. The gate and the mechanism tests found nothing in label noise after 32 cards and 4 mechanism tests, so the real arms' 4 predictive accepts and 4 mechanisms are unlikely to come from how much the sessions searched.

**Predictive.** Both real arms accepted a short-memory practice term (a decayed count of the player's recent games on the hero, with a half-life of 4 to 5 days). They passed at z 3.77 and 3.79 against 3.72, each gaining about 0.0001 per game. Lower support floors for counters paid most: w2-K-blinded p01, 5K games, gained 0.0006 at z 12.15. Pooled kit-by-context products and kit-conjunction indicators gained nothing in any arm.

## 8. Addendum (2026-10-17): official numbers from cold refits

The incumbents in section 5 stopped at 400 L-BFGS iterations in each of 5 EB rounds and never met their stopping rule. Each cycle-2 final incumbent was refit cold, to convergence: every coefficient within 0.001 posterior SD of its stationary point, and EB precisions at a fixed point (no block moving more than 0.01 in log10). Code: commit d145e9a, `training/scilearn/cycle_refits.py`. These are the official cycle-2 numbers. Sections 0 to 7 are unchanged.

| Session | Cold refit | Iterations (EB rounds) | Fit time | Log loss | Acc. | Slope | vs legacy fit | vs K0 | vs A0 |
|---|---|---|---|---|---|---|---|---|---|
| w2-K-b | inc_ff820fcca10b247f | 3,505 (16) | 141 min | 0.67219 | 58.25% | 0.884 | -0.00004 (0.00001) | -0.00866 (0.00033) | **-0.00034 (0.00006)** |
| w2-K-blinded | inc_377ca3fb446b1275 | 3,743 (16) | 122 min | 0.67300 | 58.12% | 0.885 | -0.00003 (0.00001) | -0.00786 (0.00032) | **+0.00047 (0.00012)** |
| w2-K-shuffled | inc_7f4b9cd10f5a23e2 | 1,371 (30) | 13 min | 0.69331 | 50.14% | 0.132 | on its own shuffled labels | | |

- Differences are paired Dpost log loss (negative means the cold refit is better), with the week x player-cluster block SE. Dpost is these arms' dev tranche, so the caveat in section 5 still applies.
- The conclusions of section 5 do not change. w2-K-b beats A0 by 0.00034 (about 6 SE), and w2-K-blinded trails it by 0.00047.
- **What convergence needs.** On the full 2.28M fit games, plain L-BFGS at the old settings did not converge in 2,000 iterations. With diagonal preconditioning it needs 1,000 to 1,100 iterations for the first EB round, 3,400 to 3,700 in total over 16 EB rounds, and 2 to 2.4 hours on 3090 cores 24 to 35 (about 2.6 s per iteration while the 3090 was shared). The EB loop, not the inner solve, sets the total: 4 EB updates are not enough, and some precisions moved 8x in the 4th update. The legacy fits sat up to 9 posterior SD from the optimum on individual coefficients, but only 0.00003 to 0.00004 log loss away.
- **Warm against cold, measured on w2-K-blinded.** A warm refit of the final terms, started from a cold fit of the cycle-start terms plus the Q2 offset fits of p01 and p02, took 2,998 iterations and 100 minutes, against 3,743 iterations and 122 minutes cold. It reached the same optimum: log loss 0.67299 against 0.67300 (paired difference -0.00001, SE under 0.00001). Warm starting saves about 20%, not more, because the EB rounds still have to settle.
