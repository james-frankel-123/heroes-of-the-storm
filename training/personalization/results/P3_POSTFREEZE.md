# P3 post-freeze confirmation (2026-10-28, corrected 2026-11-01)

The definitive build def1027 was scored once, with no refit, on Storm League games of patch 2.57 played from 2026-09-28 to 2026-10-26. These games touched no step of model selection: every export, store and window before them was capped at 2026-09-28.

**Correction (2026-11-01).** The first version of this check (build pf1027) did not exclude the 5,000 sealed 2.57 replays: 4,312 of them entered its slot table and 49,749 history rows. The numbers below come from the weekly build wk261101, which excludes the sealed list in every post-cap path (`p3_sealed.py`), restricted to the same window. That build also has games ingested after 2026-10-27, so the set is 84,658 games: 79,643 shared with the first version, 5,015 late-ingested, 4,312 sealed games removed. The first-version figures were +7.84 over the previous model, +17.20 over today's kernel and 62.59% accuracy.

## Headline

- **The chosen model keeps its lead on games no step of the work had seen: +7.61 log-loss points (x1000) over the previous main model (week-block 95% CI 6.82 to 8.07) and +17.08 over today's kernel (15.96 to 17.94).**
  - On OOT the same contrasts were +9.07 and +18.89. The post-freeze gains are about 16% and 10% smaller.
  - Lift over population WP falls for every model on the new patch (today's kernel 18.49 → 15.87, previous 28.31 → 25.34, chosen 37.38 → 32.95).
  - Accuracy: 57.22% (draft only), 60.20% (today's kernel), 61.59% (previous), 62.69% (chosen).
- **Calibration loosens a little on the new patch.**
  - Game-level calibration slope of the chosen model is 0.956, against 0.980 on OOT (today's kernel 1.005 vs 0.971, previous 0.991 vs 0.986).
  - Its slot-level slope is 0.92 overall: 0.84 to 0.87 on 0 to 3 games on the hero, and 0.91 to 1.01 from 10 to 300 games. On OOT it was 0.995 overall.
  - Our reading: 2.57 changed hero balance, so pre-patch records predict a bit less, and the thinnest records most. The model does not refit, so this is the expected direction.

## Games

| set | games |
|---|---|
| 2.57 Storm League games, 2026-09-28 to 2026-10-26, in the DB (sealed replays excluded) | 95,973 |
| complete in the v1 slot table (all ten players with a v1 hero) | 84,658 |

Games with Xal'atath (released in 2.57) cannot be represented in the 90-hero v1 set the models use, so they drop out, as they would for the live v1 model. Haunted Mines games stay, scored as on OOT.

## Scores

| model | gain over population WP x1000 (SE) | accuracy | game calibration slope | slot calibration slope |
|---|---|---|---|---|
| today's kernel | +15.87 (0.59) | 60.20% | 1.005 | 0.938 |
| previous main model | +25.34 (0.76) | 61.59% | 0.991 | 1.032 |
| chosen model | +32.95 (0.91) | 62.69% | 0.956 | 0.915 |

| contrast | gain x1000 (game SE) | week-block 95% CI (5 weeks) | 3-day-block 95% CI | accuracy |
|---|---|---|---|---|
| chosen − previous | +7.61 (0.44) | 6.82 to 8.07 | 6.36 to 8.87 | +1.09pp |
| chosen − today's kernel | +17.08 (0.69) | 15.96 to 17.94 | 15.45 to 18.66 | +2.49pp |
| previous − today's kernel | +9.47 (0.49) | 8.95 to 9.96 | 8.53 to 10.36 | +1.40pp |

Five weeks give a thin week-block bootstrap; the 3-day-block interval is wider and still well clear of zero.

Slot calibration slope by games on the hero (smooth curve, 95% player-bootstrap band):

| games | today's kernel | previous main model | chosen model |
|---|---|---|---|
| 0 | 0.91 (0.76-1.08) | 1.04 (0.95-1.15) | 0.84 (0.77-0.91) |
| 3 | 0.93 (0.83-1.00) | 1.01 (0.95-1.08) | 0.87 (0.83-0.92) |
| 10 | 0.97 (0.86-1.07) | 1.03 (0.96-1.11) | 0.91 (0.85-0.96) |
| 30 | 0.84 (0.75-0.90) | 0.92 (0.83-0.97) | 0.94 (0.88-1.01) |
| 100 | 0.70 (0.62-0.81) | 0.79 (0.69-0.92) | 0.92 (0.83-1.01) |
| 300 | 0.86 (0.72-1.03) | 0.85 (0.75-1.00) | 1.01 (0.84-1.15) |
| 1000 | 0.69 (0.40-0.96) | 0.69 (0.46-0.90) | 0.73 (0.51-0.98) |

The chosen model stays the flattest of the three past 30 games. It over-trusts the record of a player's first games on a hero after the patch.

## How it was built (nothing refit)

- **Frozen-coefficient mode (`P3_RCF_FROZEN`).** Every V1-fitted piece comes from the def1027 arm file: shrinkage function, temporal layer (half-life, shrinkage, scales), direct scoreboard channel with role weights, smooth calibration, and the game combiner. The skill-model parameters (full_ms) and today's kernels are def1027's; the MMR coefficients and the experience-offset table too (offset table identical to def1027's, max difference 0).
- **Regression test.** On def1027's own data, an arm refit on the main box and the same arm scored in frozen mode from its saved coefficients agree bit for bit on all 429,464 games and 4.3M slots (`p3_rcf_frozen_check.py`).
- **Data appended, never re-read.** Post-freeze mode keeps every pre-freeze input of def1027 and appends only the post-freeze games:
  - population WP: the frozen d2c models with each build's statistics through the previous build (all 309,666 pinned games keep their WP exactly);
  - exports: def1027's files byte for byte plus the new rows;
  - history store rebuilt with the cap at 2026-11-01: its pre-freeze rows equal def1027's;
  - slot table: the 22,448,979 pre-freeze slots equal def1027's in every column.
- **The pre-freeze windows reproduce def1027:** today's kernel and the previous model match to three decimals on V1, V2 and OOT; the chosen model differs by at most 0.002 because of the MMR key fix below.

## A bug found on the way: MMR stamp keys

The shared MMR lookup packed (game, player) as replay_id·2^44 + player key. That overflows int64 for replay ids above 2^19, so two games of one player 1,048,576 replay ids apart shared a key (4,049 keys in def1027), and an unstable sort picked either stamp depending on the numpy build. Effect on def1027: the chosen model's OOT gain moves from +37.381 to +37.380 with the fix. The fix (collision-free key, conflicting duplicates unknown) is behind `P3_MMR_KEY_FIX=1`, used here; the default stays the def1027 behaviour until the next model version so the site's weekly publish reproduces def1027. The same packing pattern was fixed in `p3_x_side.py` (unused today) and in the tournament's p2f scripts (their commit).

## Files

- `p3_rcf.py` (`P3_RCF_FROZEN`, `P3_RCF_FROZEN_TABLE`), `p3_rcf_tails.py` (frozen shrinkage function), `p3_c_tests.py` (`P3_PF_WINDOW`), `p3_export.py` (`--pf-cap`, `--base-tag`), `p3_full_history.py` (`P3_CAP_DATE`), `p3_x_fetch_post.py` and `p3_x_wp_post.py` (`P3_PF_CAP`, `P3_PF_OUT`), `p3_mmr_feats.py` (`P3_MMR_KEY_FIX`).
- `remote/pf_build.sh`, `remote/pf_score.sh`, `p3_pf_report.py`, `p3_rcf_frozen_check.py`.
- `results/def_wk261101/P3_POSTFREEZE_wk261101_to1026.md`, `results/def_wk261101/pf_report_to1026.json` (corrected); `results/def_pf1027/` (first version, includes sealed games; superseded).
