# P3 extensions: out-of-time test, cold start, learning curves, smurfs, drafting alternatives, maps, robustness (2026-09-30)

This file extends P3_HERO_STRENGTH, P3_SKILL_DRIFT, P3_VALIDITY and P3_DRAFTER. All new code is in `training/personalization/p3_x_*.py`; no existing file was edited. An independent audit of the shared pipeline is running in parallel. Every number here comes from scripts that read the shared caches (`hs_slots.npz`, `wp_drift.npz`, `hs_kernels.npz`, `dr_*.npz`) and can be rerun end to end if the audit changes any of them (see "Rerunning" at the end).

Conventions, as in the earlier write-ups:

- **Residual** r = y − WP_drift for the player's team. **Skill estimate** s = experience offset + posterior mean of the phase-1 "+CF rank 2" kernel, with state from the player's games on earlier days (lag 1 day).
- **Gain** is the held-out log-loss improvement over the population WP, with the combiner (logit WP + team difference of summed s) fit on V1. CIs are replay bootstraps unless stated.
- **Latent R²** removes game noise: 1 − (MSE − noise) / (Var − noise).
- Resources: 4 CPU cores (`taskset -c 48-63`, 4 threads), no GPU.

## 1. Out-of-time validation (`p3_x_fetch_post.py`, `p3_x_wp_post.py`, `p3_x_common.py`, `p3_x_oot.py`)

**Data.** Every Storm League game of build 2.55.16.97039 after the snapshot and of 2.55.17.97605, 97650, 97771 and 98025, dated up to 2026-09-27. That is 297,342 new games, 297,119 of them with all ten player rows. 286,016 are dated after the snapshot's last day (2026-05-22) and form the out-of-time (OOT) set. 43,524 accounts are new since the snapshot, and 15% of OOT slots belong to them.

**Causal WP for the new games.** Same three d2c_cumprev models, same features, statistics through the previous build:

- 97039 uses the drift2026 cumulative file through 96881, which is what the snapshot scores used;
- 97605 uses the snapshot counts through 96881 plus every DB game of 97039;
- each later 2.55.17 build adds the previous build's games.

No statistic includes a game from its own build. Check: the 12,324 snapshot games of 97039, rescored by the new code, match `wp_drift.npz` to 3e-8.

**What is frozen.** The experience table, the kernels and the combiner weights, all as fit on data through May. Nothing is refit on the new games. Three state variants:

- **online:** the nightly product. State comes from every game on earlier days, including post-May games, with the May hyperparameters.
- **frozen@May:** state and game counts frozen at 2026-05-22 and never updated.
- **frozen@May, counts current:** the residual state is frozen, but the game counts that drive the experience offset stay current. Counts carry no outcomes.

**Reproduction.** The same code on V2 with snapshot-only state gives +0.01418 (phase 1: +0.0142).

### Game level (286,016 OOT games, combiner fit on V1 in the snapshot)

| model | gain vs population WP (95% CI) | accuracy | calibration slope |
|---|---|---|---|
| population WP | | 56.94% | 0.93 |
| experience offset only | +0.0083 (0.0078, 0.0088) | 58.58% | 0.96 |
| player (overall skill) | +0.0116 (0.0110, 0.0120) | 59.23% | 0.99 |
| player+hero | +0.0142 (0.0135, 0.0147) | 59.58% | 0.98 |
| **+CF rank 2, online** | **+0.0149 (0.0142, 0.0154)** | **59.70%** | **0.99** |
| +CF rank 2, frozen@May | +0.0044 (0.0038, 0.0050) | 58.06% | 0.82 |
| +CF rank 2, frozen@May, counts current | +0.0107 (0.0101, 0.0112) | 59.10% | 0.95 |

For comparison, the same model on V2 scored +0.0142 and 59.95%.

**By month and by build** (online +CF rank 2 vs frozen@May):

| period | games | population WP accuracy | population WP calibration slope | online gain | frozen@May gain |
|---|---|---|---|---|---|
| 2026-05-23 to 06-30 | 84,831 | 57.24% | 1.00 | +0.0146 | +0.0071 |
| July | 68,883 | 56.85% | 0.92 | +0.0148 | +0.0043 |
| August | 70,913 | 56.89% | 0.89 | +0.0146 | +0.0023 |
| September (to 27th) | 61,389 | 56.70% | 0.89 | +0.0156 | +0.0034 |
| build 2.55.16.97039 | 127,503 | 57.11% | 0.98 | +0.0149 (0.0141, 0.0157) | +0.0064 |
| builds 2.55.17.* | 158,513 | 56.81% | 0.89 | +0.0148 (0.0139, 0.0155) | +0.0028 |
| 2.55.17.97605 / 97650 / 97771 / 98025 | 9K / 47K / 66K / 36K | 57.0 / 56.8 / 56.8 / 56.8% | 0.88 / 0.88 / 0.89 / 0.90 | +0.0129 / +0.0144 / +0.0149 / +0.0157 | |

Findings:

- **The lift holds four months out and across the 2.55.17 patch.** With nightly state updates and May hyperparameters, the gain is +0.0149 on OOT games against +0.0142 on V2. It is +0.0148 on 2.55.17 games, with every build between +0.0129 and +0.0157 and no downward trend by month.
- **The population WP drifts; the personal layer does not.** The population model loses 0.5pp of accuracy from late May to September. Its calibration slope falls from 1.00 to 0.89 once 2.55.17 ships, because its statistics are one build stale and its weights date from February. The combined model stays calibrated (slope 0.99). Refit on OOT games, the combiner puts 0.94 on the WP logit and 3.83 on the skill sum, against 1.00 and 3.62 fit on V1. The skill weight holds; the WP weight shrinks.
- **The hyperparameters transfer; the state does not.** A model frozen in May keeps only +0.0044, and its value decays from +0.0071 in the first five weeks to +0.0023 to +0.0034 by August and September. Keeping game counts current recovers most of it (+0.0107). The rest (+0.0042) is the value of updating residual state nightly. Two things drive this:
  - The experience offset depends on counts, and these change fastest. New heroes, new accounts and returning players all look like "0 games" to a frozen model.
  - Skill estimates for players who keep playing sharpen with new games.
- **Slot level (per-slot log-loss gain ×1000, estimates used as-is, no fitting).**
  - online: +1.50 (V2: +1.45), slope 0.93;
  - frozen@May: +0.21, slope 0.60;
  - experience offset only: +0.81.
  - By games on the hero, online gains are 2.0 at 0, 1.2 to 1.5 at 1 to 50, and 1.9 to 2.4 at 101+. As in phase 1, slopes fall to 0.73 to 0.79 above 20 games: deep cells overstate old differences.
- **New accounts carry a large share of the value.** Slots of accounts first seen after May gain 2.90 per 1000 (slope 1.30), against 1.25 for known players (slope 0.87). The slope above 1 means the model underpredicts how far new accounts depart from the population expectation (section 4).
- **Interval coverage four months out.** Posteriors were frozen at the snapshot and compared with each cell's OOT mean residual (51,890 cells with 10+ OOT games):
  - Raw 80% coverage is 0.79 (realized/claimed variance 1.52).
  - With the phase-1 display calibration (1.20 s² + (7pp)² for never-played heroes), coverage is 0.81 overall, 0.80 on heroes played before and 0.82 on never-played heroes.
  - The calibration fitted in the spring holds in the autumn.

**Verdict for the product and the paper.** The phase-1 model can ship with its May hyperparameters as long as the state is updated nightly. A monthly refresh of the kernels is not needed for accuracy. What matters is running the state update and keeping counts current. The pre-registration for builds after 2026-09-27 can use +0.0149 (online) as the expected gain, with a per-build range of +0.013 to +0.016.
