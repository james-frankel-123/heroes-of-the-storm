# P3 rich CF: richer skill models than rank-2 CF (interim, 2026-10-16)

This is the interim report. It covers variants 1 to 4 and the variant 7 side comparison. Variants 5 (neural CF) and 6 (temporal dynamics) have not been run.

All numbers use the window cache (A0) plus the oct05 merged store. That is the same setup as `P3_FULL_HISTORY_EARLY.md`: 32,736 pulled players, and A1 has not run yet. The definitive run comes after tranche 2, about Oct 28.

## Headline

- **Per-game performance is the richer signal Max was after.** We model the win residual jointly with 15 scoreboard metrics from past games. The metrics are standardized within hero, map, game length and outcome. This model lifts V2 by **+23.5** log-loss points (x1000) against +17.1 for today's model, a gain of **+6.4 (SE 0.5)**. On OOT the gain is **+6.9 (SE 0.25)**. Accuracy rises from 60.3% to 61.2% on V2 and from 60.0% to 61.1% on OOT.
- **The gain is larger than full history.** With window history only, the metric model scores +22.4 on V2, against +14.3 for today's model on the same history. Full history adds +1.1 on top of the metrics.
- **Win/loss alone supports about 2 dimensions, even with full history.** Ranks 3 to 16, and group-lasso rank selection, never beat rank 2 on held-out players. The win signal limits the rank, not the history depth.
- **Implicit feedback (play preference) gives the best cold start.** Masked cold-start R² goes from 0.38 to 0.49 (+0.11, SE 0.03), with +1.4 on V2. On top of the metrics it adds nothing to game-level lift, and it lowers cold start.
- **Dropping role and melee costs nothing.** With the predictive fit, rank 2 without them is within 0.1 log-loss points of rank 2 with them.
- **Kit axes add nothing.** They add +0.1 to the win-only model, which is within noise, and cost 0.8 on the metric model.
- **Recommendation:** use the metric model (`met2_hs2`) for the Oct 28 rerun, with a refit display scale for its intervals. Use the implicit-feedback model (`imp2`) only for ranking never-played heroes. Details are in the last section.

## What was built

Code: `p3_rcf.py` (model, fit, online walk, Cassia, coverage), `p3_rcf_data.py` (scoreboard metrics), `p3_rcf_report.py` (table), `tests/test_rcf.py`, and `remote/run_rcf_{fits,eval}.sh`. The scoreboard export is `p3_export.py scoreboard`. It reads the local research DB, with the same build and date rule as the other exports, so it contains no 2.57 games.

**One factor form for every variant.**
- Skill θ_ph = U_h·a_p + e_ph, with a_p ~ N(W z_p, I) and e_ph ~ N(0, s2_h).
- U contains an overall-skill column, the learned factors V, and (optionally) role, fine-role, melee or kit columns. Today's kernel is the special case K = UUᵀ + s2·I.
- The posterior is exact K×K algebra (Woodbury), so it is fast.
- Test R2 confirms that today's kernel in this form equals the kernel file. The online walk with that kernel reproduces A0 (+14.31 V2, +14.84 OOT) and A2wS (+17.12, +16.87) to the last digit.

**Implicit feedback (variant 3).**
- z_p is the player's as-of play preference: log1p game counts per hero as shares, standardized per hero, plus log volume. It is computed from all history rows, HP rows included.
- W maps z_p to the prior mean of the latent, in the style of SVD++. Counts never contain outcomes.

**Multi-target (variant 4).**
- For each metric, the cell mean of past standardized performance equals G_m·a_p + cell noise + game noise. Loadings are shared across heroes. The latent is the same one the win channel uses.
- Metrics:
  - hero, siege and structure damage;
  - healing, self-healing and protection;
  - damage taken;
  - deaths, kills, assists and outnumbered deaths;
  - XP contribution;
  - merc captures, escapes and regen globes.
- `time_cc_enemy_heroes` is always 0 in HP data. The scoreboard has no objective-time field.
- HP history rows have no scoreboard, so metrics come from corpus games only.

**Fit (as-of by construction).**
- E is split at 2025-03-01.
- Conditioning set: every history row before the split, plus its metrics and counts.
- Target: the player's E slot rows from the split on (win channel).
- Objective: predictive log likelihood of the target given the conditioning set, on the phase-1 fit half of players.
- Rank, priors and structure were chosen on a 20% inner validation split of the fit half (`val_*` rows in `results/rcf/params/`). The real held-out half was used only for test 1.

**Tests, all on the same games and cells.**
1. Held-out predictive LL: 105,597 held-out players, E targets from 2025-03-01 given everything before. Shown as the paired sum difference vs `today:merged`, with SE over players.
2. V2 / OOT lift: logistic combiner fit on V1, as in `p3_c_tests`, with SE over games. The reference is `today:merged`, which is A2wS.
3. Masked cold start (Cassia): 19,927 cells and 2,432 held-out players. All target-cell rows, counts and metrics are hidden. The difference in R² vs ref has a player-cluster bootstrap SE.
4. First-game adoption: 60,480 V2 events. An event is the first game on a hero in the window (lag-1 count 0) for a player with 20+ earlier window games. The log-loss difference vs ref has a player-cluster SE.
5. Interval coverage: the posterior as of the V1/V2 split vs the cell mean of V2 games, for 8,678 cells with 10+ V2 games.

**As-of checks.**
- `tests/test_rcf.py`, 4 of 4 pass:
  - the factor posterior equals `gp_query`;
  - today's kernel in factor form equals the kernel file;
  - in the online walk, flipping a game's own outcome, changing its own metrics, or planting rows on the same or later days changes no prediction, while a lag-0 walk is caught on 25 of 25 probes;
  - the predictive LL equals a brute-force Gaussian.
- The existing fixtures pass: `test_merged.py` 9 of 9, `test_full_history.py` 7 of 7.
- Metric standardization uses population statistics from E rows only. Later quarters reuse the last E quarter.

## Per-metric reliability per game (E rows, split-half)

| metric | player×hero | player |
|---|---|---|
| **win residual y − WP** | **0.013** | **0.005** |
| hero damage | 0.23 | 0.12 |
| siege damage | 0.28 | 0.14 |
| structure damage | 0.14 | 0.07 |
| healing (35 heroes) | 0.23 | 0.12 |
| self-healing | 0.39 | 0.11 |
| protection (17 heroes) | 0.40 | 0.18 |
| damage taken | 0.28 | 0.17 |
| deaths | 0.17 | 0.12 |
| kills | 0.07 | 0.03 |
| assists | 0.04 | 0.03 |
| XP contribution | 0.21 | 0.11 |
| merc captures | 0.16 | 0.10 |
| outnumbered deaths | 0.09 | 0.07 |
| escapes | 0.06 | 0.04 |
| regen globes | 0.20 | 0.14 |

Each column is the share of one game's variance that is a stable player×hero (or player) effect. A single game's scoreboard is 3 to 30 times as reliable as its win/loss. Source: `results/rcf/reliability.json`.

## Comparison table

Lift is the log-loss gain over population WP (x1000). "vs ref" is paired against `today:merged` (A2wS). All-10 means lobbies with all 10 players pulled: 5,143 V2 games and 4,725 OOT games. Calibration: var ratio is the realized latent error variance over the claimed posterior variance in the coverage test.

| variant | rank | held-out LL vs ref (SE) | V2 lift | V2 vs ref (SE) | V2 all-10 vs ref (SE) | OOT lift | OOT vs ref (SE) | OOT all-10 vs ref (SE) | Cassia R² (vs ref, SE) | adoption LL vs ref (SE) | cover80 | var ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1. Baselines** | | | | | | | | | | | | |
| today, window (A0) | 2 | −581 (60) | +14.31 | −2.80 (0.35) | −6.25 (1.73) | +14.84 | −2.03 (0.16) | −5.89 (1.73) | 0.209 (−0.170, 0.035) | −0.15 (0.12) | 0.790 | 1.72 |
| **today, merged (A2wS), ref** | 2 | 0 | +17.12 | 0 | 0 | +16.87 | 0 | 0 | 0.380 | 0 | 0.787 | 2.05 |
| refit, today's method (marginal LL), with role | 2 | −45 (13) | +17.06 | −0.06 (0.08) | −0.13 (0.36) | +16.77 | −0.11 (0.04) | −0.81 (0.39) | 0.385 (+0.006, 0.015) | +0.03 (0.04) | 0.786 | 2.09 |
| same, role and melee removed | 2 | −42 (18) | +16.99 | −0.13 (0.11) | −0.46 (0.52) | +16.71 | −0.16 (0.05) | −0.81 (0.57) | 0.356 (−0.024, 0.021) | +0.01 (0.05) | 0.787 | 2.06 |
| predictive fit, with role | 2 | +139 (28) | +17.46 | +0.35 (0.15) | −0.09 (0.65) | +17.49 | +0.61 (0.08) | +0.24 (0.62) | 0.434 (+0.054, 0.019) | +0.12 (0.08) | 0.788 | 1.76 |
| **predictive fit, no role/melee** (`cf2_pred`) | 2 | +128 (30) | +17.40 | +0.28 (0.17) | −0.26 (0.73) | +17.42 | +0.55 (0.09) | +0.21 (0.71) | 0.402 (+0.022, 0.023) | +0.11 (0.09) | 0.788 | 1.75 |
| **2. Rank on full history** | | | | | | | | | | | | |
| rank 4, Gaussian prior | 4 | +72 (37) | +16.98 | −0.14 (0.21) | −1.64 (0.94) | +17.18 | +0.31 (0.10) | −0.10 (0.95) | 0.302 (−0.077, 0.034) | −0.01 (0.11) | 0.788 | 1.76 |
| group lasso, kmax 16 (λ = 100 chosen on validation) | 9 | +116 (27) | +17.07 | −0.05 (0.17) | +0.49 (0.82) | +17.14 | +0.27 (0.09) | −0.02 (0.84) | 0.335 (−0.044, 0.029) | +0.10 (0.08) | 0.787 | 1.97 |
| rank 2, per-hero s2 | 2 | +201 (34) | +17.44 | +0.32 (0.19) | −0.19 (0.85) | +17.59 | +0.72 (0.10) | +0.19 (0.85) | 0.416 (+0.037, 0.028) | +0.13 (0.09) | 0.786 | 1.77 |
| **3. Implicit feedback** (`imp2`) | 2 | +304 (39) | +18.51 | **+1.39 (0.25)** | +0.55 (0.90) | +18.56 | **+1.69 (0.13)** | +1.55 (0.92) | **0.488 (+0.109, 0.031)** | +0.55 (0.11) | 0.787 | 1.89 |
| **4. Multi-target** | | | | | | | | | | | | |
| rank 1 | 1 | +1491 (79) | +23.47 | +6.35 (0.49) | +3.75 (1.93) | +23.14 | +6.26 (0.25) | +0.88 (2.04) | 0.233 (−0.146, 0.068) | +0.79 (0.23) | 0.778 | 2.70 |
| rank 2 | 2 | +1608 (78) | +22.82 | +5.71 (0.47) | +2.06 (1.86) | +22.90 | +6.03 (0.24) | +2.17 (1.97) | 0.389 (+0.009, 0.066) | +1.01 (0.23) | 0.780 | 2.66 |
| rank 1, per-hero s2 | 1 | +1629 (80) | +23.67 | +6.55 (0.50) | +4.51 (2.01) | +23.51 | +6.64 (0.25) | +1.92 (2.08) | 0.166 (−0.214, 0.075) | +0.87 (0.24) | 0.778 | 2.59 |
| **rank 2, per-hero s2** (`met2_hs2`) | 2 | **+1725 (78)** | **+23.48** | **+6.37 (0.48)** | **+4.35 (1.87)** | **+23.72** | **+6.85 (0.25)** | +2.76 (1.93) | 0.362 (−0.018, 0.069) | **+1.03 (0.23)** | 0.779 | 2.58 |
| same, window history only | 2 | +1048 (89) | +22.40 | +5.28 (0.55) | −0.64 (2.35) | +23.32 | +6.45 (0.27) | −2.15 (2.42) | 0.448 (+0.068, 0.051) | +0.90 (0.21) | 0.785 | 2.04 |
| 3 + 4 combined | 2 | +1699 (78) | +23.30 | +6.18 (0.47) | +4.17 (1.75) | +23.39 | +6.52 (0.24) | +2.29 (1.80) | 0.298 (−0.082, 0.067) | +1.00 (0.22) | 0.781 | 2.57 |
| **7. Kit axes (side)** | | | | | | | | | | | | |
| `cf2_pred` + 12 kit axes | 2+12 | +145 (29) | +17.51 | +0.39 (0.16) | −0.01 (0.70) | +17.50 | +0.63 (0.08) | +0.15 (0.68) | 0.432 (+0.053, 0.022) | +0.12 (0.09) | 0.788 | 1.75 |
| `met2_hs2` + 12 kit axes | 2+12 | +1705 (79) | +22.68 | +5.56 (0.49) | +2.80 (1.91) | +22.47 | +5.59 (0.25) | −0.03 (2.01) | 0.282 (−0.098, 0.064) | +1.07 (0.24) | 0.779 | 2.62 |

Accuracy, V2 / OOT:

| model | V2 | OOT |
|---|---|---|
| today, window | 60.00% | 59.69% |
| today, merged | 60.34% | 59.95% |
| `imp2` | 60.58% | 60.27% |
| `met2_hs2` | 61.18% | 61.11% |

Full rows, including first-game bias: `results/rcf/report_interim.{txt,json}`.

**Adoption first games.** Realized −3.19pp. Predicted: −2.86pp by today's model, −2.99pp by `imp2`, −4.00pp by `met2_hs2`.
- Today's model is optimistic on first games, as in P3_HERO_STRENGTH 5b.
- The metric model is slightly pessimistic, but it has the best log loss on those games: +1.87 over the offset alone, against +0.84 for today's model.

## Supported dimensionality

Inner validation LL, relative to Gaussian rank 2 (−60,157.7). Higher is better.

| win channel only, full history | val LL |
|---|---|
| rank 2 / 3 / 4 | 0 / −4.6 / −0.2 |
| rank 8 / 16 | −48.8 / −68.7 |
| group lasso λ 30 / 50 / 100 / 150 / 200 (effective rank 16 / 14 / 8 / 4 / 0) | −42 / −16 / **+14** / +2 / −98 |
| rank 2 with implicit feedback (W sd 0.1) | **+27** |
| rank 4 with implicit feedback (W sd 0.1) / rank 8 (W sd 0.3) | +2 / −98 |

| with metrics | val LL |
|---|---|
| conditional fit, rank 1 / 2 / 3 / 4 (shared loadings) | +276 / +272 / +263 / +239 |
| rank 2, per-hero s2 | +296 |
| rank 1, per-hero s2 | **+315** |
| per-hero metric loadings (sd 0.02) / per-role loadings | −185 / +197 |
| generative (joint) fit of metrics and wins, rank 2 / 4 / 8 / 16 / 32 | −32 / +1 / +53 / **+70** / +64 |

- **Win/loss supports 2 dimensions** on top of the overall-skill column. Doubling to tripling the games per cell did not change that.
  - The group lasso keeps 8 to 9 small factors, but on held-out players it ties rank 2 (+116 vs +128 predictive LL) and loses on lift and Cassia.
  - With ~±49pp of noise per game there is no measurable third factor.
- **The performance data hold about 16 dimensions.** The generative fit improves up to rank 16 and is flat at 32.
- **What predicts winning is one or two of them.** Trained on the win target, rank 1 to 2 is best. The generative 16-dim latent, which explains the metrics, predicts the win target worse (+70 vs +315). Most performance variance is style, not winning.
- **What the metric latent loads on** (`met2_hs2`, shared across heroes): the overall-skill direction is mostly hero damage (+0.93), self-healing (+0.73) and low damage taken (−0.65), then structure damage. Kills and assists carry little.
- **Fitted prior sds.** Today's model: overall 1.7pp, hero-specific 2.2pp, CF factors 2.5 / 2.3pp. With metrics: overall 4.1pp, factors 4.1 / 2.3pp, hero-specific 3.3pp. Once performance pins down general skill, the model gives it about 2.4 times the sd it had from win/loss alone.

## Masked cold start (Cassia) R² by role

| model | tank | bruiser | healer | ranged | melee | support | all |
|---|---|---|---|---|---|---|---|
| today, window | −1.02 | 0.17 | 0.03 | 0.25 | 0.29 | 0.17 | 0.21 |
| today, merged | −3.63 | 0.29 | 0.27 | 0.35 | 0.34 | 0.06 | 0.38 |
| `cf2_pred` | −3.47 | 0.37 | 0.33 | 0.33 | 0.36 | 0.12 | 0.40 |
| **`imp2`** | −3.41 | 0.37 | **0.42** | **0.43** | **0.57** | **0.36** | **0.49** |
| `met2_hs2` | −5.18 | 0.14 | 0.16 | 0.40 | 0.29 | −0.04 | 0.36 |
| `met2_hs2`, window | −3.69 | 0.32 | 0.29 | 0.47 | 0.40 | 0.07 | 0.45 |
| `cf2_pred` + kit | −3.96 | 0.38 | 0.38 | 0.35 | 0.43 | 0.09 | 0.43 |

- **Full history.** For a hero the player barely plays, full history alone lifts R² from 0.21 to 0.38. Play preference lifts it to 0.49.
- **Metrics help level, not hero-specific transfer.** They identify general level, but they do not predict hero-specific skill on unplayed heroes better than preference does.
- **Tanks stay broken.** Every model is worse than predicting zero on rarely played tank cells, and more so with full history. Keep the overall-skill fallback for tanks.

## Interval coverage

Every model covers 78% to 79% at nominal 80%, overall. The difference shows in the variance ratio: 1.75 for the win-only predictive fit, 2.0 for today's model, 2.6 for the metric models.

The metric model's posterior is too narrow by a factor of about 1.6 in sd. That comes from the conditional fit: the metric channel is trained only as a predictor, so its noise terms are not variance components. Coverage by games on the hero at the split, `met2_hs2`:

| games on the hero at the split | 0 | 1-5 | 6-20 | 21-100 | 101+ |
|---|---|---|---|---|---|
| coverage | 0.72 | 0.75 | 0.78 | 0.78 | 0.81 |

The display band needs its own scale refit on V1, as P3_HERO_STRENGTH did with 1.25.

## What did not work

- **Profiled ARD (Bayesian PCA with the column scale profiled out) collapsed every factor.** The log penalty has a singularity at zero worth more than any factor's data gain. It was replaced by a group lasso. The group lasso is sharp in λ: 150 keeps 4 factors and 200 keeps 0.
- **Per-hero or per-role metric loadings overfit.** Validation LL fell by 458 and by 75 from the shared loadings.
- **Generative (joint) training of the metric model** is worse for predicting wins than conditional training, by 245 validation LL at its best rank. It is the right model for describing play style, not for win prediction.
- **Implicit feedback on top of metrics** adds no lift and lowers Cassia.

## Recommendation for the full-history rerun (about Oct 28)

1. **Main skill model: `met2_hs2`.**
   - Win residual plus 15 standardized scoreboard metrics.
   - An overall-skill column plus 2 learned factors.
   - Metric loadings shared across heroes, and a per-hero hero-specific variance.
   - No role, fine-role or melee columns.
   - Fit by the predictive objective on the merged store.

   It is +6.4 on V2 and +6.9 on OOT over A2wS: that is 3 to 4 times what full history added. The cost is calibration: refit the display variance scale on V1 (raw intervals are about 1.6 times too narrow in sd) and report coverage on V2.
2. **Cold-start hero suggestions: `imp2`** (rank 2 plus play-preference prior). It has the best Cassia R², 0.49, and still lifts +1.4. Keep the overall-skill fallback for tanks.
3. **For the rerun itself:**
   - Re-export the scoreboard for the A1 corpus. The current export starts on 2024-04-01, and A1 adds older corpus rows with scoreboards.
   - Refit on the tranche-2 store.
   - Rerun `tests/test_rcf.py` with the merged fixtures.

   The all-10-pulled gain for the metric model is +4.4 (SE 1.9) on V2 and +2.8 (SE 1.9) on OOT. Those sets grow about 8 times and 30 times after tranche 2, so this is where the rerun will say most.
4. **Next variants:**
   - Variant 5, a nonlinear two-tower: the metric results show headroom in the inputs, but rank 1 to 2 already saturates the win target, so expect little. Run it on the metric features only if Max wants the paper-2 neural comparison.
   - Variant 6, temporal dynamics: per-player drift and a per-player learning rate on new heroes. That is the open calibration issue on first games: today's model is optimistic and the metric model slightly pessimistic.

## Compute and files

- **Fits:** 5090 GPU through hotsjob, `rcf_*` jobs, under 11 GB host RAM, seconds to minutes each.
- **Online, Cassia and coverage:** this box, 4 cores (48-51), nice 19. About 3 to 7 minutes per variant.
- **Untouched:** the 3090 was not used. Its waiters (PIDs 2419065 and 3536119) and their chains were not touched.
- **Results:** `results/rcf/` holds the report, reliability, Cassia and coverage JSONs, and the fit metadata JSONs. Per-slot arrays with player keys stay in the gitignored cache.
