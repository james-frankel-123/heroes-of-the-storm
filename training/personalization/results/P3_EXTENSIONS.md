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

## 2. Heroes a player has not played: a natural experiment (`p3_x_predall.py`, `p3_x_fetch_side.py`, `p3_x_side.py`, `p3_x_adopt.py`)

**Events.** An adoption event is a (player, hero) cell whose first game in the snapshot window comes after the player already has 50+ games and 90+ days in the window. There are 386,826 such events. The prediction is the causal posterior at the first game (lag 1 day, `cache/x_predall.npz`, which reproduces phase 1's V2 per-slot gain of 1.45 per 1000).

**Many "new" heroes are not new.** The corpus holds only replays uploaded to Heroes Profile. The hero level recorded on the first game tells the cases apart:

- level ≤ 3 (truly new): 55,511 events (14%);
- level 4 to 9: 138,594;
- level ≥ 10 (a hero the player already knew): 192,721 (50%).

**Survivorship.** Only 7.8% of adoptions reach 10 games. The detrended first-game residual of those that do was +6.6pp; for those that stop before 10 it was −0.8pp. Players keep a new hero when the first games go well. Any sample restricted to "10+ games on the new hero" is selected on its own early outcomes. This is why phase 1 needed a +7pp band for never-played heroes that players go on to play.

### A. First game on the new hero, every event (no survivorship)

Realized raw residual −4.54pp; the experience offset predicts −4.32pp. The offset is right on average. Per-slot log-loss gain over the population WP (×1000), with the prediction used as shown (offset at 0 games + posterior mean):

| group | events | offset only | player-only | player+hero | role pooling | co-play | similarity (CF rank 2) |
|---|---|---|---|---|---|---|---|
| all | 386,826 | 4.51 | 4.61 | 4.68 | 4.69 | 4.72 | **4.80** |
| truly new (level ≤ 3) | 55,511 | 6.52 | 7.19 | 7.28 | 7.27 | 7.31 | **7.51** |
| known hero (level ≥ 10) | 192,721 | **3.89** | 3.75 | 3.83 | 3.83 | 3.86 | 3.85 |
| tanks | | 3.51 | 3.63 | 3.69 | 3.72 | 3.76 | **3.91** |
| bruisers | | 3.01 | 3.16 | **3.21** | **3.21** | 3.20 | 3.15 |
| healers | | 2.50 | 2.66 | 2.71 | 2.70 | 2.70 | **2.78** |
| ranged assassins | | 5.74 | 5.84 | 5.92 | 5.92 | 5.96 | **6.08** |

Paired differences (×1000 per slot, player-clustered bootstrap):

- similarity − role pooling: +0.105 (0.033, 0.171);
- similarity − player-only: +0.185 (0.117, 0.246);
- role − player-only: +0.080 (0.058, 0.102);
- similarity − offset only: +0.281 (0.165, 0.388).

### B. The first 10 or 20 games of players who stay

Target: the mean detrended residual over the first N games. N = 10 gives 30,252 events (latent sd 3.7pp); N = 20 gives 10,134 (latent sd 3.9pp).

**Uncorrected, the displayed number is badly biased for this group.** The display (offset at 0 games plus posterior mean) says −4.4pp. The realized mean over the first 10 games is +1.4pp raw. The offset path over those games (which falls as the player gains games) predicts −1.7pp. The as-is posterior means are also over-dispersed for this group: their slopes are 0.4 to 0.5. Both are selection effects.

Latent R² after a cross-fitted recalibration (intercept and slope, 2 folds split by player):

| predictor | N = 10 | N = 20 |
|---|---|---|
| player-only | 0.021 | 0.011 |
| player+hero (no hero pooling) | 0.024 | 0.013 |
| role pooling | 0.025 | 0.014 |
| co-play pooling | 0.029 | 0.016 |
| **similarity pooling (CF rank 2)** | **0.047** | **0.029** |
| similarity − role, paired (95% CI) | +0.022 (0.005, 0.038) | +0.016 (0.002, 0.028) |
| similarity − player-only, paired | +0.026 (0.011, 0.041) | +0.019 (0.005, 0.032) |

**Side information** (cross-fitted ridge, penalty chosen inside the training fold):

| model | N = 10 | N = 20 |
|---|---|---|
| recalibrated GP means (CF, player, role) | 0.046 | 0.025 |
| + hero attributes (role, melee, hand-labeled high mechanics, CF loadings, and their products with the player's level) | 0.094 | 0.057 |
| + scoreboard style, raw | 0.113 | 0.042 |
| + scoreboard style, outcome-neutral | 0.124 | 0.046 |
| + style + talent conformity | 0.113 | 0.042 |
| + hero level on the first game | 0.115 | 0.045 |

Paired, at N = 10:

- attributes add +0.048 (0.024, 0.070);
- outcome-neutral style adds +0.030 (−0.002, 0.059);
- raw style adds +0.019 (−0.004, 0.047);
- talents and hero level add nothing (+0.001).

At N = 20, attributes add +0.032 (0.014, 0.049) and style adds −0.010 to −0.015.

**Tanks** (5,561 events at N = 10):

| model | tank R², N = 10 | tank R², N = 20 |
|---|---|---|
| similarity, recalibrated | 0.010 | 0.005 |
| role pooling, recalibrated | −0.011 | −0.009 |
| + attributes | 0.030 | −0.010 |
| + outcome-neutral style | 0.116 | −0.013 |

Style helps tanks over the first 10 games and not over 20, so I treat it as unconfirmed.

**Interval coverage.** The calibrated display band (1.2 s² + (7pp)² for never-played heroes) covers the mean of the first 10 games for 81% of players who stay (raw posterior 77%), and 81% at N = 20.

### Reading

- **How good is the pre-adoption prediction?**
  - On a single first game it is right on average, with the off-pool penalty of about −4.5pp.
  - For players who stay with the hero, it explains about 5% of the latent spread of their first 10 games once its level is corrected. The best side-information model explains 9 to 12%.
  - Most of what happens on a newly adopted hero cannot be predicted from other heroes.
- **Similarity pooling beats role pooling and player-only.**
  - It wins on the unbiased first-game test (+0.10 and +0.19 per 1000 slots).
  - It wins on the adopters' first 10 and 20 games (+0.016 to +0.026 in R²).
  - Role pooling barely beats player-only (+0.004 in R², +0.08 per 1000).
- **Tanks.** The phase-1 Cassia test found pooling hurt on rarely played tank cells. On real first games of new tanks, the similarity kernel does better than every alternative (3.91 vs 3.51 for the offset alone). The Cassia failure was specific to heroes a veteran dabbles in, and does not carry over to real adoptions.
- **Known heroes behave differently.** On heroes with level ≥ 10, pooling adds nothing over the offset (3.85 vs 3.89). The player has experience with the hero that our window does not see.
  - A lifetime hero level from the player's profile would fix the offset for these heroes.
  - As a side feature for the first 10 games it added nothing, because the offset's selection problem dominates.
- **Side information.** Hero attributes help (+0.03 to +0.05 R²). They capture which kinds of heroes adopters do well on relative to the model, and how that depends on the player's overall level. Outcome-neutral scoreboard style adds a little at 10 games. Talent conformity adds nothing.
- **Product.** Keep showing the off-pool penalty for a hero the player has not played; it is calibrated on first games. Label the band as "first games". Once a player has played 3 to 5 games, the phase-1 estimate takes over, and the +7pp band covers the survivors.

## 3. Learning curves and rust (`p3_x_learn.py`, figure `fig_x_learning_curves.png`)

### A. Learning a new hero

**Setup.** Adoption events come from section 2. "New" means hero level ≤ 5 on the first game; "returning" means level ≥ 10.

- **Gap at game k:** the raw residual r on the new hero minus the player's own level, where the level is his mean r on established heroes (20+ earlier games) over the same calendar span, with 10+ such games required.
- **Two designs bracket the truth:**
  - The **balanced** panel keeps events with at least K games. Its mix of players is fixed, but players keep a hero after good early games, so its early games are selected upward.
  - The **unbalanced** curve uses every event still playing at game k. Game k's own result does not decide whether it is seen. But the final games before a player quits are selected downward, and the mix shifts toward players who stayed.
- **Game 1** of the unbalanced curve includes every event, so it has no selection at all.
- CIs are player-clustered bootstraps.

Gap to own established-hero level (pp), new heroes:

| games on the new hero | 1 | 2 | 3-5 | 6-10 | 11-20 | 21-30 | 31-50 |
|---|---|---|---|---|---|---|---|
| unbalanced (39,185 events) | **−10.5** (−11.0, −9.9) | −8.6 | −7.0 | −5.2 | −4.1 | −2.7 (−4.0, −1.6) | |
| balanced, 30+ games (931 events) | −1.2 (games 1-2) | | −3.0 (−4.9, −1.2) | −0.7 | −0.8 | −0.6 (−1.8, +0.5) | |
| balanced, 50+ games (357 events) | −3.8 (games 1-2) | | −2.1 | −1.2 | +0.7 | +1.0 | +0.6 (−0.8, +1.8) |

Returning heroes (level ≥ 10):

- unbalanced: −8.6 (game 1), −7.0, −6.0, −4.5, −3.6, −3.1;
- balanced 30+ games (2,698 events): +0.3, +0.5, +1.4, +0.9, −0.6.

Findings:

- **The first game on a new hero costs about 10pp** against the player's own established-hero level. This estimate is free of selection. About half of that is the off-pool penalty the experience offset already prices (section 2A: −4.5pp against the population expectation). The rest is the comfort bonus the player has on his usual heroes.
- **Players who keep the hero reach their own level in roughly 10 games.** In the 50+ game panel the gap closes from −3.8pp (games 1-2) to −1.2pp (games 6-10) and is zero or positive from game 11 on. An exponential fit gives a time constant of 4 games and "within 1pp" by game 6. The 30+ game panel sits within 1pp from game 6 on.
  - Early games in these panels are biased upward by survivorship, so the true early deficit is larger than shown and the time to baseline is if anything longer.
  - The unbalanced curve is still −2.7pp at games 21-30, which is an upper bound on the remaining gap for a typical adopter.
  - A fair summary: most of the deficit is gone after 5 to 10 games, and the last 1 to 3pp takes 20 or more.
- **By hero type** (30+ game panel, games 3-5):
  - high-execution heroes (top third of CF axis 1): −7.8pp (−11.4, −4.3), back within about 2pp by games 6-10;
  - straightforward heroes (bottom third): −1.0pp (−4.3, +2.8).
  - The 50+ game panel agrees: −7.8pp at games 1-2 on high-execution heroes.
  - The hand-labeled "high mechanics" flag shows the same direction with wider CIs (−3.5 vs −2.9 at games 3-5).
- **By role** (30+ game panel, games 3-5):
  - bruisers −6.7pp (−10.6, −2.3), ranged assassins −4.4pp (−8.2, −0.5);
  - healers +1.2pp, tanks +0.7pp, both with wide CIs.
  - New tanks and healers cost little; new bruisers and ranged assassins cost the most. Each role group has 160 to 280 events, so these are directional.
- **By player.**
  - Players in the bottom third of overall level still trail by −2.8pp (−4.7, −1.1) at games 11-20. The top third is at −0.2pp.
  - Volume does not separate cleanly.
- **Heterogeneity.** The early-gap spread across events is about 10pp after removing noise (9 to 11pp in every group). That is two to three times the spread of settled hero skill. Early-late correlations are not estimable at these sample sizes (the latent variances are too small and noisy).
- **Returning heroes** (level ≥ 10) show no learning curve for players who stay: +0.3 to +1.4pp over games 1-20. Their unbalanced game-1 gap (−8.6pp) is as large as for new heroes, which is mostly rust (see B).

### B. Rust

**Setup.** Established players only: 100+ earlier games at the time of the game. Outcome e = r − skill estimate (lag 1 day). Gaps are measured in play-time order.

**Overall: days since the player's previous game (any hero)**

| gap | 0-0.5 d | 0.5-1 | 1-3 | 3-7 | 7-14 | 14-30 | 30-90 | 90-180 | 180+ |
|---|---|---|---|---|---|---|---|---|---|
| e (pp) | −0.1 | −0.4 | −0.5 | −0.3 | −0.3 | −0.4 | **+1.4** (0.8, 2.1) | **+1.9** (0.6, 3.4) | +0.8 (−2.0, 4.1) |
| slots | 2.45M | 546K | 533K | 236K | 83K | 38K | 19K | 3.9K | 1.1K |

After a 30+ day break:

- first game +1.5pp (1.0, 2.1);
- games 2-3 +0.8;
- games 4-10 +0.4;
- games 11-30 −0.3;
- games 31-100 −0.3.

**Per hero: days since the player last played this hero** (player active, with another game in the last 3 days; 20+ earlier games on the hero)

| gap | 0-1 d | 1-7 | 7-30 | 30-90 | 90-180 | 180-365 |
|---|---|---|---|---|---|---|
| e (pp) | +0.1 | −0.3 | −0.5 | −0.5 | **−1.5** (−2.4, −0.5) | **−2.2** (−4.0, −0.1) |
| slots | 681K | 528K | 211K | 54K | 10K | 2.3K |

**Game level (V2, on top of the skill model):**

| added term | gain |
|---|---|
| log days since the player's last game | +0.00016 (0.00006, 0.00026) |
| log days since this hero was last played | +0.00039 (0.00018, 0.00055) |
| all rust terms | +0.00052 (0.00029, 0.00071) |

Findings:

- **Hero rust is real and grows with time.** A hero untouched for 3 to 12 months costs 1.5 to 2.2pp beyond the skill model, even for an active player. The static skill model ignores this. The state-space model in P3_SKILL_DRIFT found no drift in hero-specific skill, and the two fit together: this looks like a temporary loss that a few games restore, with the underlying level unchanged.
- **Overall "rust" runs the other way.** Players returning after 1 to 6 months win 1.4 to 1.9pp more than the skill model expects for their first few games. The effect fades within about 10 games. Two likely causes, neither tested here:
  - Matchmaking: returning accounts may be placed against weaker lobbies (the residual adjusts for the draft only, so weaker opponents show up as a positive residual).
  - Players stopping after a bad run of form.
- **Product.**
  - Add "days since this hero was last played" to the combiner (+0.0004).
  - In the per-hero display, flag heroes unplayed for 90+ days as "rusty: about −2pp for the first games back".
  - Do not penalize a returning player overall.

## 4. Smurfs: purification loop, new-account prior, detector (`p3_x_smurf.py`, `p3_x_newacct.py`)

**New accounts.** As in P3_VALIDITY: first seen on or after 2024-07-01, with median hero level ≤ 5 over the first 10 games. There are 51,446 such accounts, and 13,048 have 20+ games.

### A. The purification loop (proposal #7)

Each round:

1. For every candidate, compute x = mean over its first 20 games of the residual net of the other nine players' skill estimates.
2. Fit a two-component Gaussian mixture to x by EM, with known per-account noise.
3. Flag accounts with P(smurf) > 0.5.
4. For each flagged account, estimate its excess per block of games (1-20, 21-50, 51-100), shrunk toward 0 by 30 games.
5. Correct every other player's residual in those games for the flagged teammates and opponents.
6. Rebuild everyone's skill estimates from the corrected residuals, and repeat.

The flag set converged in 4 rounds (changes 247, 60, 7).

| round | flagged | smurf share of new accounts | normal new account: mean / sd | smurf: mean | excess in games 1-20 / 21-50 / 51-100 |
|---|---|---|---|---|---|
| 1 | 1,641 | 19.6% | +3.7 / 5.6pp | +21.7pp | +30.8 / +10.4 / +2.4pp |
| 4 (final) | **1,327** | 16.8% | +4.1 / 5.8pp | +23.0pp | **+32.6 / +11.9 / +2.8pp** |

**Census.** 1,327 accounts are flagged, 10.2% of new accounts with 20+ games.

- **Games touched.** A flagged account in its first 100 games appears in 6.3% of all games. The share rises over time: 3.7% in 2024 H2, 6.2% in 2025 H1, 8.0% in 2025 H2 and 8.1% in 2026 H1.
- **Flag rate by the tier of the account's first game:** high 14.3%, mid 10.3%, low 9.6%.
- **Flag rate by region:** Americas 8.6%, Europe 11.7%, Asia 1.8% (560 candidates).
- **Raw residual of flagged accounts by game index:**

| games | 1-5 | 6-10 | 11-20 | 21-50 | 51-100 | 101-300 | 301+ |
|---|---|---|---|---|---|---|---|
| flagged accounts | +35.2pp | +33.8 | +30.8 | +12.0 | +6.3 | +3.0 | +2.2 |
| other new accounts | +6.4 | +5.0 | +3.4 | +3.3 | +2.0 | +1.1 | +0.7 |

- Flagged accounts stay above expectation for hundreds of games: +2.2pp even after 300 games.
- The unflagged new accounts are also above expectation: +3 to +6pp over their first 50 games.

### B. What purification changes

**Skill estimates of everyone else** (V2 slots, unflagged players):

- correlation with the unpurified estimates: 0.9988;
- mean change: +0.03pp; mean absolute change: 0.07pp; 1.1% of slots move by more than 0.5pp.
- For players whose histories are most exposed (top 10%, where 8 to 15% of their games had a flagged opponent), the mean change is +0.07pp.

**Headline lift** (V2, combiner fit on V1):

| skill state built from | gain (95% CI) | vs raw (95% CI) |
|---|---|---|
| raw residuals | +0.01418 (0.01311, 0.01516) | |
| retroactive purification (uses smurfs' later games) | +0.01392 | −0.00026 (−0.00038, −0.00015) |
| boundary purification (flags and excess from games before the V1/V2 split only) | +0.01416 | −0.00001 (−0.00010, +0.00008) |

Purification does not change the headline. The causal version is neutral; the retroactive one is slightly worse. Two reasons:

- Smurf games are a small share of any regular player's history.
- The skill model shrinks each cell hard, so a −30pp shock in 1 to 2% of games moves an estimate by well under 0.1pp.

This agrees with P3_VALIDITY's exclusion test, and settles proposal #7's worry for the headline numbers: raw and purified labels give the same result.

### C. A new-account prior

Account status at each game (causal):

- 0: first seen before 2024-07-01;
- 1: first seen later, first day only;
- 2: first seen later, every hero level on earlier days ≤ 5 (a genuinely new account);
- 3: first seen later, some earlier hero level > 5 (an old account new to our corpus).

The experience table is split by status and fit on E, and the skill state is rebuilt on the matching residuals.

**Offset for status 2** (pp, by games seen × games on hero):

- 1 to 4 games seen: +3.4 (hero not played) to +8.8;
- 10 to 19 games: +8.2 to +13.2;
- 20 to 49 games: +9.4 (hero not played), up to +13.7;
- 50 to 99 games: +3.1 to +7.0.

For comparison, status 0 at 20 to 49 games seen runs from −2.8 (hero not played) to +2.4.

| gain over the phase-1 table | V2 (combiner fit on V1) | OOT (frozen, 286K games) |
|---|---|---|
| 4 statuses | **+0.0024 (0.0020, 0.0029)** | **+0.0021 (0.0019, 0.0023)** |
| new low-level accounts (status 2) split out alone | +0.0024 (0.0019, 0.0028) | +0.0018 (0.0016, 0.0020) |
| old accounts new to the corpus (status 3) split out alone | +0.0006 | +0.0007 |
| in games with a new low-level account (18% of V2 games) | +0.0128 (0.0100, 0.0151) | +0.0112 (0.0098, 0.0125) |
| in games without one | +0.0002 (−0.0001, +0.0004) | +0.0003 (0.0002, 0.0004) |

- The headline moves from +0.0142 to +0.0166 on V2, and from +0.0149 to +0.0169 out of time.
- The whole gain sits in the 18% of games that contain a genuinely new account. In those games the prior adds +0.013, about as much as the whole phase-1 model adds on an average game (+0.014).
- The experience offset alone could not do this, because it treats a new account and a veteran new to our corpus alike: both have few games in the window.

### D. A causal smurf detector

**Setup.**

- At game k (10 or 20) of a new account, predict from its first k games only whether its next 80 games run hot: mean net residual > +8pp.
- Logistic regression, trained on accounts that started before 2025-07-01 and tested on later accounts.
- Accounts: 3,322 at k = 10 and 2,896 at k = 20 (test sets of 1,268 and 1,104).
- Label rates: 22% train and 36% test at k = 10; 19% and 32% at k = 20. The label rate drifts up over time.

| features | AUC, k = 10 | AUC, k = 20 | future-excess latent R², k = 20 | next-80-game excess, flagged vs not (k = 20) |
|---|---|---|---|---|
| residual z of the first k games | 0.67 | 0.70 | 0.04 | +8.9 vs +3.5pp |
| + net residual + hero level | 0.68 | 0.71 | 0.13 | +8.9 vs +3.0 |
| scoreboard style only | 0.72 | 0.74 | 0.22 | +9.8 vs +3.1 |
| side information without outcomes (hero level, talents, party, variety, pace) | 0.65 | 0.66 | 0.03 | +7.9 vs +3.3 |
| **all** | **0.73** | **0.74** | **0.29** | **+9.6 vs +2.7** |

- Precision at the training flag rate is 0.57 to 0.60 and recall is 0.49 to 0.57.
- Scoreboard style (per-minute numbers relative to other players of the same hero) beats the win record as a smurf signal. It sees how the account plays, while ten games of results are mostly noise.
- The detector is useful and honest, and not strong enough to act on alone.
- The new-account prior in C already captures most of the average effect. The detector's extra value is ranking which new accounts will stay hot: 29% of the latent spread at game 20.

**Product.**

- Ship the status-split experience table: +0.002 overall and +0.011 to +0.013 in games with a new account.
- Show new accounts in a lobby as "new account: expect +8 to +13pp over their first 50 games", with the detector score as a secondary flag.
- Do not purify residuals for the skill model.
