# P3 game-disjoint judge sensitivity (review items C18 / R12, 2026-10-28)

The tournament's judge (the player-skill model fit on the other player half) shared game outcomes with the drafter's model, even though the two halves had no players in common:
- **V1 games:** both arms fit their shrinkage function, direct channel, role weights, calibration, layer and combiner on them.
- **E-window games:** both arms fit their skill parameters on them.

Two judge variants remove that overlap step by step. Neither needs any redrafting. The tournament agent rescored the headline pass with each, CPU only (the tournament side is in `P3_TOURNAMENT.md`, three-judge section).

## The three judges

| judge | skill parameters fit on | V1-type coefficients fit on | shares outcomes with a drafter fit? | OOT gain x1000 |
|---|---|---|---|---|
| original (chosen_flip) | E window, other half | V1 | yes: E and V1 | +36.29 |
| partial (chosen_flip_gd) | E window, other half | V2 | E only, players disjoint | +36.64 |
| full (chosen_v2flip) | V2 window, other half | V2 | no | +31.09 |

- **The full judge's skill parameters** (`full_ms_v2flip`, 5090) come from the predictive fit on V2 games only:
  - conditioning rows 2026-03-29 to 2026-04-24, target slot rows 2026-04-25 to 2026-05-22;
  - 51,519 players, 25,896 in the fit half.
- **V2 is thin.** The median player has 2 conditioning games, so this judge is weaker: overall skill sd 6.7pp against 5.2pp, hero sd 2.1pp against 3.6pp.
- **Both new arms pass** the state-writer selfcheck (1.5e-08) and the filter poison/reference check.

Answers to the four judge query files (judgefit, split judgefit, split states, judge states), against the originals:

| judge | correlation of s on played cells | sd of the difference | base sd |
|---|---|---|---|
| partial | 0.971 to 0.988 | 0.7 to 1.0pp | identical |
| full | 0.69 to 0.77 | 2.6 to 3.7pp | median 2.5 to 3.3pp vs 3.3 to 4.2pp |

## What the rescore shows (tournament agent, P3_TOURNAMENT.md)

**Judge fits:**

| | original | partial | full |
|---|---|---|---|
| personal coefficient c2 | 3.05 | 3.00 | 2.76 |
| lift x1000 | 23.2 | 23.4 | 17.5 |
| attenuation λ | 0.55 | 0.56 | 0.44 |

**Selected design, J against:**

| opponent | original | partial | full |
|---|---|---|---|
| paper 1's MCTS | 59.0 | 59.4 | 55.0 (53.8, 56.1) |
| Real | 63.0 | 63.0 | 62.4 |
| GD | 66.0 | 65.9 | 65.0 |
| DraftRec | 58.0 | 57.9 | 58.7 |
| DraftRec-pv | 55.1 | 55.0 | 54.8 |
| aMCTS-pp | 51.6 | 51.7 | 50.8 (50.3, 51.3) |

- **Headline rows under the full judge:**
  - Every headline row keeps its sign, with its CI above 50.
  - The selected design stays first among the core systems.
  - CONS and the judge-free record do not use the judge and are unchanged.
- **Where the full judge moves things:**
  - Contrasts driven by the personal term shrink, as its lower λ predicts. aMCTS-pp vs paper 1's MCTS goes from 58.2 to 54.6; the BC-prior pair goes from 56.3 to 54.5.
  - Contrasts carried by population win rates barely move.
  - Near-even lower-ranked matchups flip, for example DraftRec vs paper 1's MCTS (52.8 to 48.1).
- **Party results:** they barely move under either judge (solo +2.46, +2.49, +2.33).

**Reading.** The overlap in fitted coefficients made no difference (partial judge). Removing the shared E-window outcomes as well costs judge strength, because V2 is short. That attenuates personal-skill margins by about a quarter but changes no headline conclusion. The paper can state the judge as fully game-disjoint for the headline rows, quoting the full judge's numbers as the conservative version.

## An incident on the way

The tournament's echo guard caught 5 corrupted player keys in one answer file of the full judge. The cause was multi-threaded OpenBLAS (0.3.29, in the numpy 2.2.6 wheel) on the main box overwriting memory nondeterministically (EDAC: 0 memory errors). Single-threaded runs reproduce bit for bit.

- **Rerun:** the file was regenerated single-threaded and verified twice.
- **Checked against single-threaded reruns:** the other answer files, the partial judge's files, the full judge arm and today's site cards are all bit-identical or differ in the last float digit only.
- **Fix:** the state writer now forces single-threaded BLAS and checks every echoed field and every estimate before saving. The weekly site publish runs BLAS single-threaded.

## Files

- `p3_rcf.py`: `P3_PREP_WINDOW`, `P3_RCF_FIT_SUFFIX` (window-restricted fit inputs), `P3_RCF_FIT_WINDOW` (fit window of the arm's coefficients).
- Params: `results/def_def1027/rcf/params/full_ms_v2flip.{npz,json}`.
- Arms: `chosen_flip_gd`, `chosen_v2flip` (`results/def_def1027/rcf/p3c/`, not in git: per-slot keys).
- Answers: `cache_def_def1027/judge_a_gd/`, `cache_def_def1027/judge_a_v2/`.
