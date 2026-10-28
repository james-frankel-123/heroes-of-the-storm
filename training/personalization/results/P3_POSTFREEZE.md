# P3 post-freeze confirmation (2026-10-28)

The definitive build def1027 was scored once, with no refit, on Storm League games of patch 2.57 played from 2026-09-28 to 2026-10-26. These games touched no step of model selection: every export, store and window before them was capped at 2026-09-28.

## Headline

- **The chosen model keeps its lead on games no step of the work had seen: +7.84 log-loss points (x1000) over the previous main model (week-block 95% CI 7.23 to 8.19) and +17.20 over today's kernel (16.29 to 17.89).**
  - On OOT the same contrasts were +9.07 and +18.89. The post-freeze gains are about 15% and 9% smaller.
  - Lift over population WP falls for every model on the new patch (today's kernel 18.49 → 16.27, previous 28.31 → 25.62, chosen 37.38 → 33.47).
  - Accuracy: 60.08% (today's kernel), 61.54% (previous), 62.59% (chosen).
- **Calibration loosens a little on the new patch.**
  - Game-level calibration slope of the chosen model is 0.958, against 0.980 on OOT (today's kernel 1.006 vs 0.971, previous 0.990 vs 0.986).
  - Its slot-level slope is 0.92 overall: about 0.85 to 0.89 on 0 to 3 games on the hero, and 0.93 to 1.01 from 10 to 300 games. On OOT it was 0.995 overall.
  - Our reading: 2.57 changed hero balance, so pre-patch records predict a bit less, and the thinnest records most. The model does not refit, so this is the expected direction.

## Games

| set | games |
|---|---|
| 2.57 Storm League games, 2026-09-28 to 2026-10-26, in the DB | 95,452 |
| complete in the v1 slot table (all ten players with a v1 hero) | 83,955 |

Games with Xal'atath (released in 2.57) cannot be represented in the 90-hero v1 set the models use, so they drop out, as they would for the live v1 model. Haunted Mines games stay, scored as on OOT.

## Scores

| model | gain over population WP x1000 (SE) | accuracy | game calibration slope | slot calibration slope |
|---|---|---|---|---|
| today's kernel | +16.27 (0.60) | 60.08% | 1.006 | 0.944 |
| previous main model | +25.62 (0.76) | 61.54% | 0.990 | 1.037 |
| chosen model | +33.47 (0.91) | 62.59% | 0.958 | 0.923 |

| contrast | gain x1000 (game SE) | week-block 95% CI (5 weeks) | 3-day-block 95% CI | accuracy |
|---|---|---|---|---|
| chosen − previous | +7.84 (0.45) | 7.23 to 8.19 | 6.61 to 9.08 | +1.05pp |
| chosen − today's kernel | +17.20 (0.69) | 16.29 to 17.89 | 15.79 to 18.63 | +2.51pp |
| previous − today's kernel | +9.35 (0.49) | 8.97 to 9.76 | 8.68 to 10.05 | +1.46pp |

Five weeks give a thin week-block bootstrap; the 3-day-block interval is wider and still well clear of zero.

Slot calibration slope by games on the hero (smooth curve, 95% player-bootstrap band):

| games | today's kernel | previous main model | chosen model |
|---|---|---|---|
| 0 | 0.93 (0.75-1.12) | 1.07 (0.96-1.17) | 0.85 (0.78-0.94) |
| 3 | 0.91 (0.81-0.99) | 1.02 (0.94-1.08) | 0.89 (0.84-0.94) |
| 10 | 0.98 (0.90-1.08) | 1.05 (0.96-1.12) | 0.94 (0.85-1.01) |
| 30 | 0.83 (0.76-0.88) | 0.91 (0.84-0.98) | 0.93 (0.87-1.00) |
| 100 | 0.74 (0.64-0.84) | 0.81 (0.72-0.90) | 0.94 (0.85-1.01) |
| 300 | 0.85 (0.74-0.99) | 0.83 (0.70-0.96) | 1.01 (0.86-1.12) |
| 1000 | 0.74 (0.48-0.94) | 0.72 (0.48-0.96) | 0.76 (0.50-0.93) |

The chosen model stays the flattest of the three past 30 games. It over-trusts the record of a player's first games on a hero after the patch.

## How it was built (nothing refit)

- **Frozen-coefficient mode (`P3_RCF_FROZEN`).** Every V1-fitted piece comes from the def1027 arm file: shrinkage function, temporal layer (half-life, shrinkage, scales), direct scoreboard channel with role weights, smooth calibration, and the game combiner. The skill-model parameters (full_ms) and today's kernels are def1027's; the MMR coefficients and the experience-offset table too (offset table identical to def1027's, max difference 0).
- **Regression test.** On def1027's own data, an arm refit on the main box and the same arm scored in frozen mode from its saved coefficients agree bit for bit on all 429,464 games and 4.3M slots (`p3_rcf_frozen_check.py`).
- **Data appended, never re-read.** Post-freeze mode keeps every pre-freeze input of def1027 and appends only the post-freeze games:
  - population WP: the frozen d2c models with each build's statistics through the previous build (all 309,666 pinned games keep their WP exactly);
  - exports: def1027's files byte for byte plus the new rows;
  - history store rebuilt with the cap at 2026-10-27: its 59,668,495 pre-freeze rows equal def1027's in every column;
  - slot table: the 22,448,979 pre-freeze slots equal def1027's in every column; 839,550 post-freeze slots added.
- **The pre-freeze windows reproduce def1027:** today's kernel and the previous model match to three decimals on V1, V2 and OOT; the chosen model differs by at most 0.002 because of the MMR key fix below.

## A bug found on the way: MMR stamp keys

The shared MMR lookup packed (game, player) as replay_id·2^44 + player key. That overflows int64 for replay ids above 2^19, so two games of one player 1,048,576 replay ids apart shared a key (4,049 keys in def1027), and an unstable sort picked either stamp depending on the numpy build. Effect on def1027: the chosen model's OOT gain moves from +37.381 to +37.380 with the fix. The fix (collision-free key, conflicting duplicates unknown) is behind `P3_MMR_KEY_FIX=1`, used here; the default stays the def1027 behaviour until the next model version so the site's weekly publish reproduces def1027. The same packing pattern was fixed in `p3_x_side.py` (unused today) and in the tournament's p2f scripts (their commit).

## Files

- `p3_rcf.py` (`P3_RCF_FROZEN`, `P3_RCF_FROZEN_TABLE`), `p3_rcf_tails.py` (frozen shrinkage function), `p3_c_tests.py` (`P3_PF_WINDOW`), `p3_export.py` (`--pf-cap`, `--base-tag`), `p3_full_history.py` (`P3_CAP_DATE`), `p3_x_fetch_post.py` and `p3_x_wp_post.py` (`P3_PF_CAP`, `P3_PF_OUT`), `p3_mmr_feats.py` (`P3_MMR_KEY_FIX`).
- `remote/pf_build.sh`, `remote/pf_score.sh`, `p3_pf_report.py`, `p3_rcf_frozen_check.py`.
- `results/def_pf1027/P3_POSTFREEZE_pf1027.md`, `results/def_pf1027/pf_report.json`.
