# P3 full history: early read

**Early read on the current panel, written 2026-10-16.**
- It uses the 32,877 players whose full Heroes Profile (HP) history is pulled so far, out of about 110K after tranche 2.
- It also runs before A1: the backfill waiter has not triggered yet (37,514 replays left at 07:38 EDT on Oct 16).
- So every merged arm here is built on the **window cache (A0)**, not on cache_full.
- The definitive run comes after tranche 2, about Oct 28. The A1/A2 run on cache_full starts by itself once A1 finishes (see "What runs next").

Numbers: `results/p3_c_tests_window.json` / `.txt`. Code: `p3_c_merged.py`, `p3_c_tests.py`, `tests/test_merged.py`.

## Headline

Full histories add **+2.8 log-loss points (x1000) on V2** and **+2.0 on OOT** over today's model. On lobbies where all 10 players are pulled, they add +6.3 on V2 and +5.9 on OOT. Accuracy rises by 0.35pp on V2 and 0.26pp on OOT; on all-10-pulled V2 lobbies it rises by 1.2pp.

The lift is the log-loss gain over M0 (population WP only), x1000. Paired SE is by game; the week CI is a week-block bootstrap.

| set | games | A0 lift | A2 lift (A2w-S) | A2-seen lift | A2 minus A0 (SE) [week CI] | A2-seen minus A0 (SE) | accuracy A0 / A2 |
|---|---|---|---|---|---|---|---|
| V2, all lobbies | 72,474 | +14.31 | **+17.12** | +16.81 | **+2.80** (0.35) [2.21, 3.52] | +2.50 (0.34) | 60.00% / 60.34% |
| V2, all 10 pulled | 5,143 | +15.24 | **+21.49** | +20.95 | **+6.25** (1.73) [4.38, 8.69] | +5.71 (1.70) | 59.48% / 60.66% |
| OOT, all lobbies | 286,016 | +14.84 | **+16.87** | +16.72 | **+2.03** (0.16) [1.70, 2.33] | +1.88 (0.15) | 59.69% / 59.95% |
| OOT, all 10 pulled | 4,725 | +15.72 | **+21.61** | +20.97 | **+5.89** (1.73) [2.43, 10.06] | +5.25 (1.71) | 59.60% / 60.08% |

- **A0** is today's model on the window cache. It reproduces the headline: +14.31 here against +14.26 in `hs_eval` on V2, and +14.84 against +14.9 on OOT.
- **A2 (A2w-S)** is A0 plus every HP row of pulled players that is not already a slot row: 26.2M rows for 32,676 players. The HP rows enter the GP skill state. The experience offset keeps counting slot rows only (see "Choices made on V1").
- **A2-seen** is the same, but uses only matches already uploaded when the scored game started. This is the live-system number. It keeps 89% of the V2 gain and 93% of the OOT gain.
- **A1** (all corpus rows, after the backfill) is not done yet. Here A2 is compared with A0. For all-10-pulled lobbies the difference should be small, because HP's histories contain the corpus rows A1 adds. A2 against A1 comes from the cache_full run.

Over 54.7% of V2 and OOT slots belong to a pulled player. For those slots, the median number of prior games goes from 123 to 235. The share of slots with 50+ prior games on the hero goes from 14% to 27%.

## Choices made on V1

All variants use the same combiner (fit on V1). The choice between them is made on V1.

| arm | V1 | V2 | OOT |
|---|---|---|---|
| A0 | +12.65 | +14.31 | +14.84 |
| A2w: HP rows also count toward the experience offset | +16.35 | +16.23 | +16.21 |
| **A2w-S: offset from slot rows only (chosen)** | **+16.64** | **+17.12** | **+16.87** |
| A2w-S, kernel variances refit on the merged E state | +16.70 | +17.15 | +16.81 |
| A2w-S, proxy 0.5 instead of hero-era win rate | +16.45 | +16.86 | +16.77 |
| A2w-S, HP rows from 2024 on only | +15.29 | +16.11 | +16.22 |
| A2w-S, recency half-life 365 / 730 / 1460 days | +15.38 / +16.24 / +16.61 | +16.11 / +16.91 / +17.21 | +16.27 / +16.84 / +17.01 |
| A0, half-life 365 | +12.67 | +14.19 | +14.80 |

What these show:
- **The experience offset should not count old HP games.** Take cells with 50 to 99 games on the hero in the full history but under 20 in our corpus. They realize -0.89pp, but A2w's offset gave them +1.28pp; A0's offset (-0.61pp) was about right. An old game on a hero does not carry the bonus a recent corpus game does. So HP rows feed the skill state only.
- **Old history helps.** Cutting HP rows to 2024 on loses about 1pp on V2. No recency half-life beats no decay on V1; 1460 days ties.
- **The refit and the proxy matter little.** The refit raises the hero-specific prior sd from 2.3pp to 3.0pp, but its gain is within noise on V1 and V2 and slightly negative on OOT. The hero-era proxy beats 0.5 by about 0.2.

## Test 1: calibration by observed record (R1)

Cells are binned by A2's full-history count of games on the hero and by the MMR-adjusted record to date (mean of y - WP; y - proxy on HP rows). Each cell compares the realized residual on the scored game with the predicted one, both in pp. CIs are week-block bootstraps. Pooled over V2 and OOT slots.

| games on hero | record | slots | realized | A0 predicted | A0 gap [CI] | A2 predicted | A2 gap [CI] |
|---|---|---|---|---|---|---|---|
| 100+ | over +10pp (mean +12.9) | 32,133 | +6.38 | +5.55 | **+0.83** [0.28, 1.37] | +8.17 | **-1.80** [-2.33, -1.24] |
| 100+ | +5 to +10 | 135,310 | +3.66 | +3.23 | +0.43 [0.20, 0.71] | +5.31 | -1.65 [-1.87, -1.36] |
| 100+ | 0 to +5 | 271,864 | +1.94 | +1.81 | +0.13 [-0.08, 0.38] | +2.95 | -1.01 [-1.23, -0.76] |
| 100+ | -5 to 0 | 154,673 | -0.32 | +0.41 | -0.73 [-1.01, -0.45] | +0.50 | -0.82 [-1.09, -0.57] |
| 100+ | under -5pp | 46,676 | -2.84 | -1.11 | **-1.73** [-2.20, -1.24] | -2.18 | -0.66 [-1.12, -0.19] |
| 50-99 | over +10pp | 41,438 | +4.28 | +4.32 | -0.05 [-0.42, 0.41] | +5.70 | -1.43 [-1.81, -0.99] |
| 50-99 | under -5pp | 64,276 | -2.16 | -1.08 | -1.08 [-1.44, -0.68] | -1.80 | -0.36 [-0.71, 0.04] |

- **Max's intuition holds for today's model.** A0 under-predicts proven records: by 0.83pp in the top cell and by 0.43pp in the +5 to +10 cell. It also under-predicts proven losing records, by 1.73pp. Across the 100+ row A0's predictions span 6.7pp against 9.2pp realized, so it is too timid by a factor of about 1.4.
- **With full history the model overshoots.** Its predictions span 10.4pp in the 100+ row against 9.2pp realized, and the top cell is now over-predicted by 1.8pp. The gap runs about -0.5 to -1.0pp in most cells, so part of it is a level shift that the team-difference combiner cancels. The remaining tilt is about 0.9.
- So the truth sits between A0 and A2. Shrinkage was too strong with thin history and is now slightly too weak in heavy cells. A recency half-life does not fix this (see above). The next things to try are a proxy that tracks each player's MMR era, or prior-weighting old HP rows (R4).

## Shrinkage

Prior-weight share = posterior variance / prior variance at the query hero (1 = all prior, 0 = all data). The same V2 and OOT slots are used for every arm, binned by A2's games on the hero.

| cells | slots | median games seen, A0 / A2 | median prior share, A0 / A2 | slope of GP mean on record, A0 / A2 | median abs GP mean, A0 / A2 |
|---|---|---|---|---|---|
| 20-49 games | 507,454 | 18 / 31 | 0.74 / 0.60 | 0.13 / 0.17 | 1.25 / 1.46 pp |
| 50-99 games | 342,299 | 29 / 69 | 0.66 / 0.48 | 0.19 / 0.28 | 1.30 / 1.57 pp |
| 100+ games | 640,656 | 77 / 231 | **0.50 / 0.27** | **0.29 / 0.49** | 1.44 / 1.80 pp |
| 100+ games, record over +5pp | 167,443 | 82 / 224 | 0.50 / 0.28 | 0.41 / 0.50 | **1.78 / 3.64 pp** |

On heavy cells the prior's weight roughly halves, from 0.50 to 0.27, and the model keeps about half of an observed record instead of 29%. For proven winners the posterior edge doubles: the median goes from 1.8pp to 3.6pp.

## Subset lifts (R3)

V2 / OOT lift, A0 → A2, with paired gain (SE). "Proven" means 50+ prior games on the hero and A2 posterior over +5pp. That definition uses A2's own estimate, so it favours A2 somewhat.

| subset | V2 games | V2: A0 → A2, gain (SE) | OOT games | OOT: A0 → A2, gain (SE) |
|---|---|---|---|---|
| 1+ player on a proven hero | 11,305 | +22.5 → +27.4, **+4.9** (1.2) | 36,898 | +23.1 → +26.0, **+2.9** (0.6) |
| 3+ players on a proven hero | 437 | +36.4 → +55.6, +19.2 (7.6) | 1,055 | +28.2 → +22.8, -5.5 (4.5) |
| Deep (all 10 with 500+ prior games) | 466 | +16.6 → +16.9, +0.3 (6.1) | 924 | +6.8 → +9.2, +2.4 (4.0) |
| Diamond | 9,188 | +21.3 → +24.6, **+3.3** (1.2) | 33,192 | +20.8 → +22.1, +1.3 (0.6) |
| Master | 6,477 | +15.8 → +19.1, **+3.3** (1.4) | 29,843 | +17.3 → +20.2, **+2.9** (0.6) |
| Diamond+Master, all 10 pulled | 3,273 | +12.7 → +18.1, +5.4 (2.3) | 3,110 | +22.1 → +27.0, +4.9 (2.1) |

- Proven-hero lobbies are where personalization pays most: +23 to +27 against +15 overall.
- Full history adds +3 to +5 there.
- The 3+ proven and deep sets hold only a few hundred to about a thousand games, so their numbers are noise. The 3+ proven set flips sign between V2 and OOT.

## The drafter (model-internal)

This is the personalized drafter harness (`p3_dr_drafter.py`, 1,000 lobbies, team assignment) rerun with A2's skill state. Pools, off-role flags and counts are unchanged. Every number is the model's own valuation. Nothing here is checked against outcomes.

| | A0 state | A2 state |
|---|---|---|
| Whole draft vs GD opponent: personalized minus population drafter, V | +3.3pp (2.7, 4.0) | **+4.6pp** (4.0, 5.2) |
| Whole draft vs imitation opponent | +4.3pp (3.6, 4.9) | **+5.1pp** (4.3, 5.8) |
| Picks that differ from the population pick (self-play) | 51% | 57% |
| Gain in V when they differ (self-play) | +3.2pp | +3.7pp |
| WP_pop given up when they differ (self-play) | -2.9pp | -3.3pp |

With fuller histories the drafter trusts personal edges more: it deviates more often and claims about 1.3pp more per draft. Since the R1 table shows A2 over-predicting heavy proven cells by about 1.8pp, part of that extra claim is over-confidence.

## Leak checks

`tests/test_merged.py`: 9 of 9 pass, through `online_predict` on a 50-player fixture.
- **M1:** planted future HP rows, on the game's own day and a day later, change no prediction. A lag-0 reader is caught on 25 of 25 probes.
- **M2:** the game's own HP copy, timed 10 minutes before the game, never enters (dropped by replay_id), and flipping its outcome changes nothing. A reader without the exclusion is caught on 25 of 25.
- **M3:** rule "seen" equals deleting a late upload.
- **M4:** counts match brute force.
- **M5:** with no extras, the code reproduces the A1 code path exactly. The slot-count offset and the half-life match `online_state` too.
- **M6:** the clock guard only delays rows.
- **M7:** the proxy reads only in-sample rows.

The store's own tests (`tests/test_full_history.py`, 7 of 7) and `tests/test_assign.py` also pass.

One data guard is worth knowing. Slot days of games from 2026-07-09 on come from caches built before the timezone fix and can run a day late. So HP rows from that date on are placed on the day of end time plus the Eastern offset. That can only delay a row, never move one earlier.

## What runs next

1. The backfill waiter (PID 2419065) exports and launches A1 (`p3_full`) on the 3090. At about 1,900 replays an hour it should trigger at its 01:40 EDT check on Oct 17.
2. The A2 waiter (`remote/p3_merged_waiter.py`, PID 3536119 on this box) watches `p3_full`. When it completes, the waiter:
   - builds the store on A1's export;
   - ships `history.npz`;
   - runs `p3_pipe.py merged` on cache_full as `p3_merged`: arms A1, A2, A2-seen, A2-S, A2-S-seen, then the report `results/oct26_p3c/p3_c_tests_full.*` against A0.
   That gives A2 against A1, the comparison this read could not make. Each arm takes 2 to 5 minutes, not the 6 to 10 hours the plan estimated.
3. After tranche 2 (about Oct 28): rebuild the store and rerun the merged chain. All-10-pulled lobbies go from 5,143 to about 40,600 on V2 and from 4,725 to about 136K on OOT, so the SEs on those sets shrink about 3x on V2 and 5x on OOT. The sealed build stays unscored until every choice is made.

## Conclusions so far

- Full histories help, and by more than the shrinkage argument alone suggested. The gain is +2.8 on V2 over all lobbies, which takes the headline from +14.3 to +17.1 (+20%). On fully pulled lobbies it is +6.3 (+41%). Most of it survives the upload-time rule.
- The extra history should feed the skill estimate, not the experience offset.
- Max's over-shrinkage hunch is confirmed for the thin-history model: proven records under-predicted by 0.4 to 0.8pp. With full history the model swings to over-predicting the same cells by 1.6 to 1.8pp. A heavy-tailed prior (R2) is not the obvious fix. The open question is weighting old games against new ones (R4).
- The lift is still about 0.017 of log loss overall, not a step change. A proven record is a real edge (+6.4pp realized in the top cell), but such cells are a small share of slots.
