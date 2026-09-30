# P3 extensions: out-of-time test, cold start, learning curves, smurfs, drafting alternatives, maps, robustness (2026-09-30)

This file extends P3_HERO_STRENGTH, P3_SKILL_DRIFT, P3_VALIDITY and P3_DRAFTER. All new code is in `training/personalization/p3_x_*.py`; no existing file was edited. An independent audit of the shared pipeline is running in parallel. Every number here comes from scripts that read the shared caches (`hs_slots.npz`, `wp_drift.npz`, `hs_kernels.npz`, `dr_*.npz`) and can be rerun end to end if the audit changes any of them (see "Rerunning" at the end).

Conventions, as in the earlier write-ups:

- **Residual** r = y − WP_drift for the player's team. **Skill estimate** s = experience offset + posterior mean of the phase-1 "+CF rank 2" kernel, with state from the player's games on earlier days (lag 1 day).
- **Gain** is the held-out log-loss improvement over the population WP, with the combiner (logit WP + team difference of summed s) fit on V1. CIs are replay bootstraps unless stated.
- **Latent R²** removes game noise: 1 − (MSE − noise) / (Var − noise).
- Resources: 4 CPU cores (`taskset -c 48-63`, 4 threads), no GPU.

## Summary

1. **Out of time.** Four months past the snapshot (286,016 games, June to September 2026, including all of 2.55.17) the model with May hyperparameters and nightly state gains **+0.0149** (0.0142, 0.0154), against +0.0142 on V2.
   - Every 2.55.17 build scores +0.013 to +0.016. The model stays calibrated (slope 0.99) while the population WP drifts to slope 0.89.
   - Frozen state is what fails: a model never updated after May keeps +0.0044. Keeping game counts current restores it to +0.0107.
   - The display bands still cover 81%.
2. **Heroes a player has not played.** 386,826 adoption events.
   - On every first game the off-pool offset is calibrated (−4.3pp predicted, −4.5pp realized).
   - Similarity pooling beats role pooling (+0.10 per 1000 slots, CI 0.03 to 0.17) and player-only (+0.19).
   - For players who keep the hero (7.8%, strongly selected: their first game was +6.6pp), a recalibrated prediction explains 5% of the latent spread of their first 10 games (role 2.5%, player-only 2.1%).
   - Hero attributes add +0.05 R²; outcome-neutral scoreboard style adds +0.03 at 10 games and nothing at 20; talents and hero level add nothing.
   - Tanks are no longer a failure case on real adoptions: similarity pooling is best on first tank games.
3. **Learning curves and rust.**
   - The first game on a new hero costs 10.5pp against the player's own established-hero level, free of selection.
   - Players who stay are within about 1pp after roughly 10 games (time constant about 4 games); high-execution heroes start about 8pp down.
   - Hero rust: a hero unplayed for 90 to 365 days costs 1.5 to 2.2pp beyond the skill model (+0.0004 as a combiner term).
   - Players returning from a 1-to-6-month break run +1.4 to +1.9pp for a few games.
4. **Smurfs.**
   - The purification loop converges in 4 rounds and flags 1,327 accounts (10.2% of new accounts with 20+ games; +32.6pp in their first 20 games). They touch 6.3% of games, up to 8.1% in 2026.
   - Purified labels leave the headline unchanged (causal: −0.00001; retroactive: −0.00026) and move other players' estimates by 0.07pp on average.
   - A **new-account prior** is the real win: +0.0024 (0.0020, 0.0029) on V2 and +0.0021 out of time, almost all from the 18% of games with a genuinely new account, where it adds +0.011 to +0.013.
   - A causal detector reaches AUC 0.74 at game 20 on a later cohort. Scoreboard style is its best signal.
5. **Drafting alternatives.**
   - Personalized bans: +0.3pp per draft (not significant).
   - Teammates-only mode keeps 97% (86%, 107%) of the full-information value.
   - Hero assignment within a team: teams leave +2.0pp on the table on average, and realized outcomes follow the model's assignment term at 80 to 90% of its slope.
   - Imitation carries outcome information the value model lacks (+32pp per unit agreement next to +13pp for the personal term). Put recency and share features into the value function; do not blend rankings.
   - Premade chemistry: +0.0004 as team terms, hero-independent, so it changes no picks.
6. **Maps.** Player × map is 3 to 5% of personal variance (sd 0.8pp), and map terms add +0.00000 to +0.00004 to prediction. Personalization can drop the map axis.
7. **Robustness.**
   - The lift holds in every tier and in both large regions in V2 and OOT: high tier +0.019, mid +0.013, low +0.010 to +0.013; Americas and Europe +0.014 to +0.016; Asia +0.006 to +0.009 on few games.
   - The low tier needs its own experience and party terms.

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
- Almost the whole gain sits in the 18% of games that contain a genuinely new account. In those games the prior adds +0.013, about as much as the whole phase-1 model adds on an average game (+0.014).
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

## 5. Alternatives in personalized drafting (`p3_x_draft.py`)

**Harness.** The one-step drafter of P3_DRAFTER (imported, unmodified), with the same value model V = sigmoid(b0 + b1 logit WP_pop + b2 ΔS + b3 ΔO) and the same held-out V2 lobbies:

- 1,000 full lobbies for simulation;
- 4,000 lobbies (8,000 teams) for realized checks at the real states.

`cuda_personal/` was not touched. The simulations use separate random streams for decisions and for the rest of the draft, so the compared drafters face the same behavioral opponent. They ran on 4 CPU processes in 31 minutes.

### (a) Personalized bans

**Setup.**

- The controlled team's three bans are chosen by V (which knows the opponents' personal strengths), by WP_pop, or kept as the real bans.
- Ban candidates: the top 10 by the GD model's ban probability plus the top 10 by the strongest opponent's personal term.
- 8 rollouts per candidate.
- All picks on both sides come from the pool-restricted behavioral policy, so only the bans differ.
- 4 trajectories per lobby and mode.

| comparison (whole draft, controlled team) | change in V (pp) | change in WP_pop (pp) |
|---|---|---|
| personalized bans − population bans | +0.29 (−0.09, +0.65) | +0.14 (−0.18, +0.47) |
| personalized bans − real bans | +0.12 (−0.30, +0.57) | +0.22 (−0.15, +0.59) |
| population bans − real bans | −0.17 (−0.62, +0.26) | +0.08 (−0.29, +0.43) |

- At the decision itself, the personalized ban differs from the population ban 53% of the time. When it differs, it claims +2.5pp of V for −2.1pp of WP_pop.
- The whole-draft difference is only +0.3pp and not significant. The decision-level claim is inflated by choosing the best of about 20 noisy 8-rollout means.
- More fundamentally, one ban removes one hero from a pool of about 34, and the opponent simply picks his next-best hero.
- **Realized check** (8,000 teams, real ban states; each real ban placed in the personalized and population ban rankings; team residual y − WP_pop by agreement quintile): top minus bottom quintile is −0.4pp (−4.1, +3.1) for personalized-ban agreement, +1.4pp (−2.2, +4.6) for population-ban agreement, and −2.3pp (−5.7, +1.2) for the personal-minus-population difference. None is distinguishable from zero.

**Verdict.** Personalized bans are worth at most a few tenths of a point per draft. Not worth a product feature beyond showing "their strongest heroes" as information.

### (b) Teammates-only mode (opponents unknown)

**Setup.**

- The controlled team drafts with three levels of information:
  - full information (all ten identities);
  - teammates only: in its rollouts the opponents pick from unrestricted GD, and its value sets their S and O to 0;
  - population WP.
- The opponent really is the identified team, drafting from its own pools.
- Final drafts are scored with the full-information V.

| controlled team drafts with | gain in V over the population drafter (pp) | change in WP_pop (pp) |
|---|---|---|
| full information | +5.93 (5.28, 6.60) | −2.90 |
| teammates only | +5.75 (4.99, 6.48) | −2.96 |
| full − teammates only | +0.18 (−0.43, +0.86) | +0.06 |

The teammates-only drafter keeps **97% (86%, 107%)** of the full-information value.

**Realized check** (8,000 teams, same states and teams). Top minus bottom quintile of team residual by agreement:

- teammates-only ranking: +6.4pp (3.1, 9.9);
- full-information ranking: +5.7pp (2.4, 9.2).

**Verdict.** The realistic lobby, where only teammates are known, loses essentially nothing. The personal value of a pick is the picking player's own term. The opponents' terms are fixed by their picks and hardly change which hero is best for us. The degraded mode is the product.

### (c) Role assignment within a team, given the picks

**Setup.** For each real team, all 120 mappings of its five players to its five heroes are scored with b2 S + b3 O, and compared with the real mapping. Lane assignments are not in the data, so this is hero-to-player assignment only.

| | all post-cutoff teams (230,132) | V2 teams with all ten players 50+ games (28,486) |
|---|---|---|
| real mapping is the best of 120 | 41.7% | 48.9% |
| mean rank of the real mapping | 6.1 | 4.2 |
| predicted gain, best mapping − real | +2.0pp | +2.0pp |
| share of teams leaving 2pp or more | 35% | 34% |
| real mapping − random mapping | +12.8pp | +16.9pp |
| realized slope of the assignment component (model: 0.93) | 0.75 (0.71, 0.78) | 0.85 (0.76, 0.93) |
| team residual by assignment-component quintile | −5.0, −1.9, −0.4, +2.0, +5.2pp | −7.1, −3.7, −1.3, +0.5, +5.9pp |

The assignment component is the real team's S minus its mean over all 120 mappings. It isolates who-plays-what from which heroes were picked.

- Teams already assign heroes far better than chance (+13 to +17pp over a random mapping), and half pick the model's best mapping.
- The model says the rest leave +2.0pp on average, and a third of teams leave 2pp or more.
- Realized outcomes track the assignment component at 80 to 90% of the model's slope. So the advice "swap heroes with a teammate" is backed by realized outcomes as well as by the model. Applied at the realized slope, it would be worth about +1.7pp per team.
- In practice this is the pick-order / trade question: who should take the hero the team needs.

### (d) Combining imitation and outcome

**Setup.** At the 57,240 real picks of the realized lobbies, each candidate gets the outcome value V and the DraftRec-style imitation log-probability (within the same candidate set). The combined score is V + λ log p_imit.

**Team-level joint regression.** Team residual (y − WP_pop) on the mean agreement of the team's real picks with each ranking, both in the model:

| signals in the regression | first signal (pp per unit agreement) | imitation (pp per unit) | correlation of the two agreements |
|---|---|---|---|
| V and imitation | +7.4 (0.5, 14.5) | +36.6 (28.9, 43.6) | 0.34 |
| personal component (V − WP) and imitation | +12.8 (5.8, 18.9) | +31.7 (23.8, 41.0) | 0.50 |

**Choosing λ.** λ was chosen on half of the lobbies and scored on the other half.

- The agreement-slope criterion picks pure imitation.
- The criterion "residual when the real pick equals the recommender's top-1" picks λ = 0.002.

| ranking (test half) | top minus bottom agreement quintile | real pick = top-1 | residual when it is | residual otherwise |
|---|---|---|---|---|
| outcome V (λ = 0) | +8.6pp (4.6, 12.3) | 12.2% | +3.4 (1.8, 5.0) | −0.5 |
| combined, λ = 0.002 | +9.7pp (5.9, 13.6) | 13.5% | +3.2 (1.6, 5.0) | −0.5 |
| imitation | +14.1pp (10.1, 18.1) | 33.3% | +2.3 (1.3, 3.3) | −1.1 |

Findings:

- **Both signals carry independent outcome information.** The imitation signal is the larger one.
  - Teams whose picks look like what the players usually pick beat the population expectation by far more than the outcome model's personal term implies. This holds even after the model's comfort offset.
  - The imitation features that the value model lacks are recency and share of recent play: EWMA pick shares over 20 and 100 games, and days since the hero was last played. Section 3's hero rust points the same way: a hero played recently is worth more than its lifetime count says.
- **As a recommender,** a small imitation weight (λ = 0.002) keeps the outcome drafter's matched-pick residual (+3.2 vs +3.4pp) and slightly raises agreement. Pure imitation recommends what players already do (33% match), and its matches win less (+2.3pp).
- **Recommendation.** Put the recency and share features into the value function (the outcome side), and keep the outcome drafter as the recommender. Do not blend rankings at a large weight. P3_DRAFTER's finding that "picks matching both do best (+3.3pp)" is confirmed, and the joint regression explains why: the outcome model is missing a recency/comfort term.

### (e) Premade chemistry in the value function

**Setup.** There are 4.24M premade pair-games (801K distinct pairs). The outcome is the team residual after the skill model (y − WP − own team's S + opponents' S).

**Duos, by number of earlier joint games:**

| earlier joint games | 0 | 1-4 | 5-19 | 20-49 | 50-199 | 200+ |
|---|---|---|---|---|---|---|
| residual (pp) | +0.9 (0.6, 1.1) | +0.8 | −0.4 | −0.4 | −0.3 | −0.6 |

New duos over-perform and established duos slightly under-perform. That fits new accounts duoing with friends (section 4), and does not fit chemistry that grows with practice.

**Pair-specific spread.** For 51,903 pairs with 10+ joint games, the spread was measured net of each member's own level away from the partner:

- 4.9pp sd when splitting alternating games;
- 3.7pp sd when splitting first vs second half in time.

The forward test says this is not stable chemistry (below).

**Joint comfort** (a hero pair the premade has played together before; pairs with 10+ joint games): −0.6pp for a first time on that hero pair, and −0.0 to −0.3pp after that.

**Game level** (V2, on top of the skill model):

| added term | gain |
|---|---|
| 3+ stack count | −0.000001 (−0.00013, +0.00013) |
| pair chemistry (causal, shrunk) | +0.00009 (−0.00005, +0.00022) |
| pair familiarity (log joint games) | +0.00006 (0.00002, 0.00010) |
| joint comfort | +0.0000 |
| all chemistry terms together | +0.00042 (0.00012, 0.00068) |

Findings:

- The split-half spread is large, yet a causal, shrunk pair term predicts almost nothing forward. Most of the spread is temporary: premades play in bursts, and the two members share their form within a burst.
- All terms together add +0.0004 to the value function. They are hero-independent (joint comfort adds nothing), so they shift the team's win probability without changing which hero to pick.
- **Verdict.** Add a small party term to the displayed win probability. Chemistry does not belong in the pick search.

## 6. Does personal skill depend on the map? (`p3_x_map.py`, proposal #5)

**Method.** Cells are (player, hero, map), over snapshot games 2024-04 to 2026-05. No game-noise model is needed, because covariances between different cells of the same player share no games:

- inside a cell (odd vs even games): s²_p + s²_ph + s²_pm + s²_phm;
- same hero, different map: s²_p + s²_ph;
- same map, different hero: s²_p + s²_pm;
- different hero and map: s²_p.

Hero × map effects of the population are already in the WP (it has hero-map win rates). CIs are 200 player bootstraps.

| residual | player | player × hero | player × map | player × hero × map |
|---|---|---|---|---|
| after the experience offset, all players (pp²) | 3.34 (3.07, 3.61) | 13.30 (12.59, 14.14) | **0.68 (0.27, 1.19)** | 5.24 (2.94, 7.46) |
| share of personal variance | 15% | 59% | **3%** | 23% |
| sd (pp) | 1.8 | 3.6 | 0.8 | 2.3 |
| after the experience offset, players with 300+ games (pp²) | 1.23 (0.94, 1.54) | 11.50 (10.48, 12.39) | 0.78 (0.22, 1.44) | 0.67 (−2.45, 3.79) |
| share (300+ games) | 9% | 81% | 5% | 5% |
| after the skill model, all players (pp²) | −0.53 | 2.65 (2.14, 3.14) | 0.65 (0.14, 1.18) | 2.83 (0.53, 5.21) |

**Predictive check.** A causal player × map (or player × hero × map) mean of the post-skill-model residual was computed over earlier days and added to the V1-fit combiner. Shrinkage comes from the components above: k = 3,741 and 856 games.

| added to the skill model | V2 gain (95% CI) |
|---|---|
| player × map | +0.000005 (−0.000003, +0.000014) |
| player × hero × map | +0.000017 (−0.000062, +0.000082) |
| both | +0.000043 (−0.000045, +0.000113) |

Findings:

- **Map conditioning of personal skill does not matter.**
  - Player × map is detectable but small: sd 0.8 to 0.9pp, 3 to 5% of the personal variance.
  - Player × hero × map is large only in the all-player sample. For heavy players it is indistinguishable from zero. The all-player value most likely reflects games in the same cell clustering in time (a short session on one hero and map shares form), which inflates the within-cell covariance.
  - Neither adds anything to prediction.
- **Product.** MAWP personalization can drop the map axis. Player × hero is where the personal signal lives (59 to 81% of it).

## 7. Robustness by skill tier and region (`p3_x_robust.py`)

**Setup.**

- Game-level gain of the phase-1 model, with the combiner fit on all V1 games and scored within subsets of V2 and of the OOT games (task 1, everything frozen).
- Tier is the game's Heroes Profile skill tier. Region codes: 1 Americas, 2 Europe, 3 Asia.
- The last two columns are a per-subset refit of the skill coefficient and the spread of the team skill difference.

| subset | V2 games | V2 gain (95% CI) | OOT games | OOT gain (95% CI) | skill coef, refit (V2) | sd of team skill difference (pp) |
|---|---|---|---|---|---|---|
| all | 72,474 | +0.0142 (0.0132, 0.0152) | 286,016 | +0.0149 (0.0142, 0.0155) | 3.79 | 9.3 |
| tier low | 13,140 | +0.0099 (0.0079, 0.0123) | 52,738 | +0.0126 (0.0114, 0.0137) | 3.80 | 7.7 |
| tier mid | 36,737 | +0.0127 (0.0112, 0.0146) | 148,951 | +0.0131 (0.0122, 0.0138) | 3.64 | 9.1 |
| tier high | 22,597 | **+0.0191** (0.0171, 0.0215) | 84,327 | **+0.0194** (0.0183, 0.0207) | 3.98 | 10.4 |
| Americas | 22,332 | +0.0146 (0.0127, 0.0166) | 91,555 | +0.0159 (0.0150, 0.0170) | 3.87 | 9.3 |
| Europe | 46,820 | +0.0143 (0.0131, 0.0156) | 180,669 | +0.0150 (0.0145, 0.0157) | 3.78 | 9.4 |
| Asia | 3,322 | +0.0090 (0.0038, 0.0130) | 13,792 | +0.0060 (0.0035, 0.0086) | 3.44 | 8.1 |

Tier × region cells on V2 range from +0.0095 (low, Europe) to +0.0198 (high, Europe).

**Key effects by subset** (V1 + V2 slots, player-clustered CIs):

| subset | never-played hero, 300+ games: raw r | 50+ games on hero, 300+ games: raw r | never-played, after skill model | 3+ stack party member, after skill model | solo, after skill model |
|---|---|---|---|---|---|
| all | −7.8 (−8.7, −6.7) | +2.2 | −0.8 (−2.0, +0.3) | +1.0 (0.8, 1.2) | −0.6 |
| tier low | −3.6 (−7.2, −0.2) | +5.5 (4.6, 6.3) | **+3.9 (0.2, 7.3)** | **+3.0 (2.5, 3.5)** | −0.0 |
| tier mid | −8.6 | +2.1 | −1.6 | +0.7 | −0.5 |
| tier high | −7.8 | +1.8 | −1.0 | +0.7 | −1.2 |
| Americas | −8.8 | +2.1 | −1.8 (−3.4, −0.2) | +0.8 | −0.6 |
| Europe | −7.4 | +2.3 | −0.4 | +1.1 | −0.6 |

Findings:

- **The lift holds in every tier and in both large regions, in both test periods.**
  - It is largest in the high tier (+0.019) and smallest in the low tier (+0.010 to +0.013).
  - The per-unit skill coefficient is about the same everywhere (3.6 to 4.0). The tier difference comes from how much the model knows: the team skill difference spreads 10.4pp in the high tier against 7.7pp in the low tier, where more players are new or light.
  - Americas and Europe agree within 0.001.
  - Asia is small (3% of games) and lower (+0.006 to +0.009). The corpus holds few Asian players with deep histories.
- **The off-pool penalty and comfort bonus hold across tiers and regions**, with one exception: the low tier.
  - Low-tier veterans lose only 3.6pp on a never-played hero and gain 5.5pp on comfort heroes.
  - The shared experience table over-penalizes them (+3.9pp left after the skill model) and under-credits premade stacks (+3.0pp).
  - A tier-specific experience table and party term would recover part of the low-tier gap. Section 4's new-account prior should help there too, since new accounts start low.
- **Party effects are larger in the low tier** (+3.0pp for 3+ stacks against +0.7pp in mid and high). Solo players in the high tier sit 1.2pp below the model.

## Rerunning (for the audit)

Everything reads the shared caches through `p3_hs_core` and `p3_x_common`. If the audit changes `hs_slots.npz`, `wp_drift.npz`, `hs_kernels.npz` or the drafter tables, rerun in this order from `training/`, with `/usr/bin/python3` (numba, scipy) except the two DB fetches, which use `python3` with psycopg2. All steps use `nice -n 19 taskset -c 48-63` with 4 threads.

1. `p3_x_fetch_post.py`, `p3_x_fetch_side.py`: DB pulls, read-only, no battletags. They only need rerunning if the DB changes.
2. `p3_x_wp_post.py`: causal WP for post-snapshot games, about 1 minute. It checks itself against `wp_drift.npz`.
3. `p3_x_common.py ext`: extended slot table.
4. `p3_x_predall.py`: per-slot causal predictions, about 4 minutes.
5. `p3_x_side.py`: scoreboard style and talent conformity.
6. `p3_x_oot.py` (task 1), `p3_x_adopt.py` (2), `p3_x_learn.py` (3, needs 2), `p3_x_smurf.py` and `p3_x_newacct.py` (4), `p3_x_map.py` (6), `p3_x_robust.py` (7): about 3, 1, 0.5, 7, 4, 1 and 1 minutes.
7. `p3_x_draft.py assign | combine | chem | sim --procs 4 | analyze` (5): `sim` takes about 31 minutes on 4 processes. It needs `cache/dr_runs.pkl.gz`, `dr_lobbies.npz` and `dr_personal_post.npz` from the P3_DRAFTER pipeline.

No GPU was used.

Results files are `results/p3_x_*.json` with matching `.txt` logs, and `fig_x_learning_curves.png`. Large intermediate caches (`cache/x_*.npz`, `cache/x_draft_sim.pkl.gz`) are not committed.
