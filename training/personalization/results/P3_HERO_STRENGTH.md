# P3: per-hero player strength with honest uncertainty (2026-09-30)

## Summary

Revised 2026-10-01 after the consolidated audit; changes are logged in `P3_AUDIT_FIXES.md`. All results here use one count contract: game counts and state from earlier days only, with the experience table fit on those counts (`results/fix/`).

- **What a strength estimate is.** It is the expected draft-adjusted residual when player p plays hero h, in win-probability points. The residual is outcome minus the drift-aware WP. The estimate has three parts:
  - an experience offset that depends only on how many games we have seen from the player, overall and on this hero;
  - a Gaussian-process posterior across the player's 90 heroes;
  - a calibrated interval.

  Every estimate is causal: it uses games through the previous day only.
- **Game-level lift.** Test games are the same 72,474 held-out V2 games as P3_NESTED_LIFT. The population WP scores 57.19% accuracy and log loss 0.67647. The best hero-strength model (online, rank-2 CF kernel) scores 60.02% and 0.66224, a gain of +0.0142 (replay-bootstrap CI 0.0131 to 0.0152; about 0.0127 to 0.0157 once recurring players are allowed for). That is 8x the static nested-lift M4 (+0.0017). MMR adds little on top: with MMR as known when the game started, the three MMRs add +0.0006 (0.0004, 0.0009).
- **Where the lift comes from.** Experience offsets alone give +0.0083. Players with long histories average 5 to 7pp below what the draft predicts on heroes outside their pool, and 2 to 3pp above it on heroes they have played 20+ times. Residual-based skill adds +0.0059 on top. Updating nightly matters: a model frozen at the cutoff, counts included, keeps +0.0059; keeping only the game counts current brings it to +0.0106; nightly updates of the skill state add the last +0.0036 (0.0031, 0.0042).
- **Pooling.** Pooling across a player's heroes helps most at small n, and most of that is the player's overall level. Per 1000 slots, against a cell shrunk on its own:
  - pooling through the player's overall level (player+hero kernel): +0.48 at n = 0, +0.22 to +0.38 at n = 1 to 10, nothing measurable at n ≥ 11;
  - the learned hero similarity (rank-2 CF factors) on top of that: +0.07 at n = 0 (CI −0.02 to +0.16, not significant), +0.16 to +0.21 at n = 1 to 10. At game level it adds +0.0010 (0.0005, 0.0014).

  Role and fine-role groupings add almost nothing once overall skill is in the model (+0.018 per 1000 slots). Co-play similarity adds a little.
- **Sample size.** One game on a hero carries very little information. The prior sd of a player×hero cell is about 4pp and game noise is about 49pp, so a cell needs about 130 games before its own data outweighs the prior. The calibrated 80% band is ±5.5pp at 1 to 2 games, ±4.7pp at 21 to 50 games, and ±3.1pp at 200+ games.
- **Coverage.** For heroes with history, raw 80% posterior intervals cover the future cell mean 79% of the time, and 80% after a two-number calibration fit on V1 and tested on V2. For never-played heroes a coverage test on future cells only counts heroes the player went on to play 10+ times, which depends on how the first games went. Tested on every adoption's first game (55,495 events in V2), the mean prediction is close (−3.0pp predicted, −3.7pp realized) and improves log loss over the offset alone, but extra variance for never-played heroes is not seen: first-game residuals are no more dispersed than on deep cells. The +7pp band fit on future cells describes players who stick with a hero, not the first game.
- **MMR.** Player MMR adds almost nothing. Hero MMR as known when the game started (causal, P3_MMR_AT_GAME) is the best MMR signal: +0.0025 (0.0021, 0.0030) alone and +0.0005 (0.0003, 0.0008) on top of the skill estimate.
- **Product.** Show strength vs expectation with an 80% band and games played. Sort heroes by estimate but label only bands that exclude zero. For a hero the player has never played, say what players with the same history have averaged (about −7pp for veterans with 300+ games). Part of that gap is which heroes players choose to skip, so it is not what playing the hero would cost. Details are in the last section.

## Setup

- **Data.** Every Storm League game from 2024-04-01 through the snapshot (replay_id ≤ 63653039, last game 2026-05-22) with all ten `replay_players` rows. That is 10,822,691 player slots and 320,954 players. Players are keyed by (region, blizz_id); 1,502 blizz_ids appear in two regions. No post-snapshot games are used.
- **Residuals.** r = y_team − wp_team, using the three paper-2 `d2c_cumprev` seeds (the corrected `wp_drift.npz`). Game noise per slot is v = wp(1 − wp), about 0.242.
- **Windows.**
  - E, estimation: 2024-04-01 up to the WP training cutoff. It has 9.39M slots, and all hyperparameters are fit here.
  - V1: 2026-02-10 to 2026-03-28, 70,974 games. Used to fit combiner weights and interval calibration.
  - V2: 2026-03-29 to 2026-05-22, 72,474 games. The test set. The split is the median game day (day 20541).
- **Online state.** For a game on day t, each player's per-hero sufficient statistics use every game (E, V1, V2) from days ≤ t − lag, with lag = 1 unless stated. The statistics are S = Σ r/v and P = Σ 1/v, optionally with exponential decay. "Static" means frozen at the cutoff. "Every game" means every game eventually uploaded with an earlier date, not what had been uploaded by draft time (audit P3-04, measured at about 0.0004 on the headline); production should filter on ingestion time.
- **Experience offset.** This is the mean residual by (games seen from the player, games seen on this hero), in 8×8 bins, fit on E and shrunk toward 0. It uses counts only, never outcomes. It is subtracted from every residual before the GP and added back to the prediction. Counts are of games on earlier days (lag 1), when fitting the table and when using it.
- **Kernels.** For one player, θ over the 90 heroes follows N(0, K). K is a weighted sum of:
  - overall skill (all-ones J);
  - Blizzard role, fine role (`HERO_ROLE_FINE`) and melee/ranged blocks;
  - an independent hero term (I);
  - a similarity term, either co-play (cosine similarity of 16-d SVD hero vectors from play shares) or CF (V Vᵀ with V a 90×r matrix fit by marginal likelihood, i.e. probabilistic matrix factorization of the residual matrix with player factors integrated out).

  Weights and V are fit by exact marginal likelihood on a random half of players. Kernels are compared on the other half.
- **Metrics.**
  - Game level: logistic combiner of logit(WP) and the team difference of summed slot estimates, fit on V1 and scored on V2, with replay bootstrap CIs. Players recur across games, which these CIs ignore. A bootstrap over players (one per game) gives (0.0131, 0.0154) for the headline; the audit's estimate of the full design effect (about 1.7 in variance) gives about (0.0127, 0.0157).
  - Slot level: per-slot log-loss gain of p = wp_team + estimate, with no fitting (the estimate is used as-is); calibration slope of r on the estimate; posterior sd.
  - Coverage: the posterior as of a date vs the mean residual of the same cell afterwards.

## 1. Kernel comparison (held-out players, E window)

Held-out marginal log likelihood, relative to the player-only kernel, on 148,546 held-out players. Fitted component sd is in pp.

| kernel | Δ held-out LL | components (sd pp) |
|---|---|---|
| player | 0 | player 2.89 |
| player+hero | +436 | player 2.69, hero 3.40 |
| player+role+hero | +462 | player 2.55, role 1.29, fine 1.11, hero 3.08 |
| + melee side feature | +461 | melee 0.75 |
| + co-play similarity | +489 | co-play 1.87 |
| + co-strength (moment estimate) | +468 | overfits; fit-half gain 690 |
| + CF rank 1 | +632 | |
| **+ CF rank 2** | **+694** | player 1.73, role 1.05, fine 1.07, melee 0.72, hero 2.20, CF 3.48 |
| + CF rank 3 / 4 / 8 | +620 / +566 / +535 | higher ranks overfit |
| + CF rank 2 + co-play | +694 | no gain over rank 2 |

The total sd of a player×hero cell is about 4pp: 2.9pp is overall skill and the rest is hero-specific. CF factor 1 (sd 2.6pp) looks like mechanical demand, running from Leoric, Qhira, D.Va, Butcher, Li Li, Kharazim and Malthael up to Zeratul, Alarak, Hanzo, Kerrigan, Genji, Cho and Medivh. So some players are specifically better on high-execution heroes. Factor 2 (2.3pp) is harder to name. At one end it has Maiev, Mephisto, Abathur, Medivh, Valeera, Cho and Lost Vikings (niche picks); at the other it has Probius, Thrall, Jaina, Lunara, Imperius, Chen and Johanna. The hand-made side features (role, fine role, melee) explain much less than two learned factors.

## 2. Game-level lift (V2, 72,474 games)

| model | acc % | log loss | gain vs M0 (95% CI) |
|---|---|---|---|
| M0 population WP | 57.19 | 0.67647 | |
| raw player×hero WR (online) | 57.75 | 0.67377 | +0.0027 (0.0022, 0.0032) |
| raw mean residual (online) | 57.78 | 0.67355 | +0.0029 (0.0024, 0.0034) |
| experience offset only | 59.00 | 0.66815 | +0.0083 (0.0074, 0.0092) |
| player (overall skill) | 59.50 | 0.66565 | +0.0108 (0.0098, 0.0118) |
| player+hero | 59.84 | 0.66321 | +0.0133 (0.0122, 0.0142) |
| player+role+melee+hero | 59.83 | 0.66306 | +0.0134 (0.0124, 0.0143) |
| + co-play | 59.79 | 0.66300 | +0.0135 (0.0124, 0.0143) |
| **+ CF rank 2** | **60.02** | **0.66224** | **+0.0142 (0.0131, 0.0152)** |
| + CF rank 2, state and counts frozen at the cutoff | 58.39 | 0.67053 | +0.0059 (0.0051, 0.0067) |
| + CF rank 2, state frozen at the cutoff, counts current | 59.32 | 0.66586 | +0.0106 (0.0097, 0.0115) |
| + CF rank 2, state and counts lagged 7 days | 59.48 | 0.66530 | +0.0112 (0.0102, 0.0121) |
| + CF rank 2, state and counts lagged 30 days | 58.92 | 0.66813 | +0.0083 (0.0075, 0.0091) |

Everything except M0 includes the experience offset, which is fit on E. Source: `results/fix/p3_hs_eval.json` (kernel rows) and `results/fix/p3_fix_arms.json` (frozen and lagged rows, each built with its own counts). Calibration stays good: ECE ≤ 0.005 and calibration slope 1.00 to 1.03 for every row.

Paired differences (replay bootstrap):

- CF rank 2 vs player+hero: +0.0010 (0.0005, 0.0014).
- player+hero vs no pooling: +0.0016 (0.0011, 0.0021).
- CF rank 2 vs no pooling: +0.0026 (0.0019, 0.0031).
- Role blocks vs player+hero: +0.00015 (0.00006, 0.00027).

**Decay.** Exponential down-weighting of old games does not help on V2. The rank-2 CF model gains +0.0142 with no decay, +0.0141 with a 365-day half-life, +0.0136 at 180 days and +0.0128 at 90 days. V1 preferred 365 days (+1.298 vs +1.279 per 1000 slots). The choice between the two is therefore made on V1: 365 days (59.85%, +0.0141) is the pre-registered model. Decay also fixes the calibration slope at high n (0.68 to 0.74 for 101 to 200 games). Staleness matters more: with state and counts both lagged, a 7-day lag costs 0.0031 and a 30-day lag 0.0059.

## 3. Sample-size curve (V2 slots, online, lag 1 day)

Per-slot log-loss gain ×1000 over the population WP, with estimates used as-is. The columns are the number of games the player had on this hero in the window before the game.

| games on hero | 0 | 1-2 | 3-5 | 6-10 | 11-20 | 21-50 | 51-100 | 101-200 | 201+ | all |
|---|---|---|---|---|---|---|---|---|---|---|
| share of slots % | 19.4 | 16.4 | 13.2 | 12.2 | 12.3 | 13.5 | 6.7 | 3.8 | 2.5 | |
| raw WR − 0.5 | 0 | −1278 | −337 | −113 | −42 | −18 | −6.7 | −1.1 | +0.3 | −276 |
| experience offset only | 1.36 | 0.63 | 0.80 | 0.53 | 0.59 | 0.54 | 0.80 | 1.61 | 1.18 | 0.83 |
| no pooling (cell shrunk alone) | 1.36 | 0.77 | 1.05 | 0.89 | 1.26 | 1.09 | 1.31 | 2.41 | 2.06 | 1.17 |
| player+hero | 1.84 | 1.15 | 1.33 | 1.12 | 1.25 | 0.96 | 1.31 | 2.15 | 2.05 | 1.36 |
| + CF rank 2 | **1.91** | **1.32** | **1.51** | **1.32** | 1.20 | 1.02 | 1.45 | 2.17 | 2.00 | **1.46** |
| CF slope of r on estimate | 1.35 | 1.14 | 1.16 | 1.02 | 0.94 | 0.74 | 0.75 | 0.68 | 0.80 | 0.97 |
| CF posterior sd, median pp | 3.94 | 3.83 | 3.72 | 3.61 | 3.48 | 3.27 | 2.98 | 2.65 | 2.15 | |
| **calibrated display sd pp** | **4.4** | **4.3** | **4.2** | **4.0** | **3.9** | **3.7** | **3.3** | **3.0** | **2.4** | |

The display sd is √1.25 × the posterior sd (section 5); at 0 games it is the first-game band of section 5b.

What pooling buys at each n: the paired difference (×1000 per slot) with 95% CIs from a player-clustered bootstrap.

| games on hero | 0 | 1-2 | 3-5 | 6-10 | 11-20 | 21-50 | 51-100 | 101-200 | 201+ |
|---|---|---|---|---|---|---|---|---|---|
| player+hero − no pooling | +0.48 (0.38, 0.59) | +0.38 (0.22, 0.50) | +0.28 (0.12, 0.43) | +0.22 (0.06, 0.40) | −0.01 | −0.13 | −0.00 | −0.26 (−0.46, −0.05) | −0.01 |
| CF rank 2 − player+hero | +0.07 (−0.02, 0.16) | +0.16 (0.05, 0.26) | +0.18 (0.04, 0.31) | +0.21 (0.07, 0.34) | −0.05 | +0.06 | +0.15 (−0.00, 0.29) | +0.02 | −0.06 |
| CF rank 2 − no pooling | +0.55 (0.43, 0.68) | +0.54 (0.35, 0.69) | +0.46 (0.23, 0.64) | +0.43 (0.20, 0.67) | −0.06 | −0.08 | +0.15 | −0.24 | −0.06 |

Reading:

- **Raw win rate is useless at typical sample sizes.** A raw WR from 1 to 20 games is far worse than predicting nothing. Its slope is 0.03 to 0.15, meaning 85 to 97% of the raw number is noise.
- **Pooling pays at 0 to 10 games.** That covers 61% of slots. At 11+ games the cell's own data matches the pooled estimate, and at 100+ games pooling pulls slightly too hard toward the player's other heroes.
- **The uncertainty barely shrinks for a long time.** The display sd goes from 4.3pp at 1 game to 3.7pp at 21 to 50 games and 2.4pp at 200+. The prior-to-noise ratio is k = σ²/τ² ≈ 0.242 / 0.04² ≈ 150 games (130 in the nested-lift MoM fit). Half the display width at 100 games is still prior.
- **The shrinkage level is about right.** CF slopes are 0.94 to 1.16 from 1 to 20 games. Two exceptions:
  - At n = 0 the slope is 1.35: the offset for never-played heroes is if anything too small.
  - At 21+ games the slope is 0.68 to 0.80: deep cells overstate old differences. A 365-day half-life brings these slopes to 0.74 to 0.83.

## 4. Cassia test (masked cold start)

Setup: 2,432 held-out players (never used for kernel or CF fitting) with ≥ 300 E games. Target cells are heroes they rarely play: 10 to 60 games and ≤ 5% of their games, 19,927 cells in total. All games on the target hero are hidden and the kernel predicts the cell's mean residual from the player's other heroes. Latent R² is the share of the true cell-effect variance explained, after removing game noise.

| method | latent R² | latent RMSE pp | claimed sd pp | 80% coverage |
|---|---|---|---|---|
| zero (population) | 0 | 2.96 | | |
| raw player mean over other heroes | −0.17 | 3.21 | | |
| player (overall skill, shrunk) | 0.06 | 2.87 | 1.69 | 0.79 |
| player+hero | 0.16 | 2.72 | 3.82 | 0.81 |
| player+role+melee+hero | 0.17 | 2.70 | 3.81 | 0.81 |
| + co-play | 0.20 | 2.66 | 3.82 | 0.81 |
| **+ CF rank 2** | **0.21** | **2.63** | 3.50 | 0.80 |
| + CF rank 2 + co-play | 0.21 | 2.63 | 3.50 | 0.80 |

Findings:

- Pooled information predicts about a fifth of a player's true effect on a hero they barely play. The CF kernel explains 33% more than player+hero.
- **By role (CF rank 2):**
  - ranged assassin 0.25, melee assassin 0.29, bruiser 0.18, support 0.18, healer 0.04;
  - tank −1.0, which is worse than predicting zero. Rarely-played tank cells have little player-specific spread and the pooled estimate adds error there.
- Coverage is 0.80, but the realized error variance is only 0.57x the claimed variance, so intervals are conservative for this population.
- **Revealing the cell's first k games.** The other variants are scored on the remaining games:
  - CF rank 2: R² 0.23 (k = 1), 0.18 (k = 3), 0.23 (k = 5);
  - player+hero: 0.16, 0.10, 0.10.
  - The raw mean of the revealed games scores R² −316 at k = 1 and −82 at k = 5. It should never be shown.

## 5. Interval coverage

Setup: the posterior as of a date, compared with the mean residual of the same player×hero cell in the following ~7 weeks. The 80% predictive interval is m ± 1.28 √(s² + game noise / n_future). The table uses CF rank 2 with no decay.

Column 0 (never-played heroes) does not test the never-played band. A never-played cell enters only if the player went on to play the hero 10+ times, which depends on how the first games went (V2 adoptions that reached 3 games had a first-game residual of +7.0pp, the rest −5.3pp), and the target uses the offset at each future game's count rather than the prediction made before the first game. Section 5b tests never-played heroes on every adoption's first game.

| as of V1/V2 split, cells in V2 with ≥ 10 future games | all | 0 | 1-2 | 3-5 | 6-10 | 11-20 | 21-50 | 51-100 | 101-200 | 201+ |
|---|---|---|---|---|---|---|---|---|---|---|
| cells | 8,678 | 1,407 | 612 | 631 | 738 | 1,016 | 1,667 | 1,244 | 892 | 471 |
| raw posterior | 0.79 | 0.73 | 0.76 | 0.79 | 0.79 | 0.80 | 0.82 | 0.82 | 0.79 | 0.82 |
| calibrated | 0.80 | (0.79) | 0.76 | 0.79 | 0.79 | 0.81 | 0.83 | 0.82 | 0.79 | 0.82 |

**Cells with ≥ 30 future games.** Raw coverage is 0.77 overall and calibrated coverage 0.79.

**Calibration.** It was fit on the cutoff → V1 cells (28,629 cells with ≥ 5 future games): s²_display = 1.25 s² for heroes with history. The raw posterior is 25% too narrow for cells with history. The same fit on never-played cells asks for an extra (7.3pp)², but those cells are selected on early outcomes (above), and the extra variance does not appear on first games (5b), so it is not part of the display. The fitted drift term is zero once the scale is applied. Refitting with extra terms for 1 to 5 games gave zero weight. 

The cold-start populations point in different directions:

- heroes a player newly takes up and keeps playing show a much wider spread (+7pp), but that population is selected on its early results;
- first games of every adoption show no extra spread (5b);
- heroes a veteran dabbles in rarely (the Cassia cells) are over-covered.

The "should I pick this" question is about the first games, so the product should use the first-game band (5b).

Single-game coverage (nf ≥ 1) is 0.85 to 0.87. That is expected: one game's residual is bounded, so a Gaussian interval over-covers. It is not evidence of good calibration.

### 5b. Never-played heroes without selection (`p3_fix_neverplayed.py`)

**Events.** Every adoption event in V2: a player's first game in the window on a hero, for players with 20+ earlier games. 55,495 events, 20,967 players. The prediction is made before the game: lag-1 state, the lag-1 offset at 0 games on the hero, and the GP posterior mean and variance. Every event is scored on its first game, so nothing is selected on outcomes. CIs are player-cluster bootstraps.

| first game, all events | value |
|---|---|
| realized mean residual | −3.7pp |
| predicted (offset + GP mean) | −3.0pp (offset alone −3.2pp) |
| bias, realized − predicted | −0.74pp (−1.13, −0.33) |
| bias by player history: 20-100 / 100-300 / 300+ games | −0.6 / −0.9 / −1.6pp |
| log loss: WP_pop / + offset / + offset + GP mean | 0.67617 / 0.67272 / 0.67200 |

- The mean is nearly right and the GP mean adds to the offset. Predictions are slightly optimistic, most for veterans.
- **Variance.** A single game's residual has variance about 2,400 pp², so the band can only be tested through the excess over game noise, e² − wp(1 − wp), averaged over events. That excess is negative on every population tested (−4 to −7 pp² on established cells of the same players), because the WP noise model is slightly off (P3-10). Against the deepest cells (50+ games on the hero, claimed display variance 10 pp²) as reference:
  - adoption first games show an excess 14 ± 5 pp² *below* the reference;
  - the display with the never-played term would claim 65 pp² *above* it (the raw posterior claims 10 pp² above).
- So the (7pp)² never-played term has no support on first games. The extra spread it was fit to is real only among players who keep the hero (section 2B of P3_EXTENSIONS; the first-3-games subset here gives a variance ratio of 0.58 against that display and coverage of 0.74, on a selected 13% of events).

**What this means for the display.** For "how will my first games on this hero go", show the offset plus GP mean with the raw-posterior band scaled by 1.25 (about ±6pp at 80% for the typical event: mean raw variance 17.7 pp²). The wider band is the spread among players who stick with a hero; if shown, label it that way.

## 6. MMR as known when the game started

`hero_mmr`, `role_mmr` and `player_mmr` are stamped when Heroes Profile parses a replay, and the stamp includes that game's result. A value taken from an earlier game is usable only if that game was parsed before the current game started. The causal values used here (P3_MMR_AT_GAME) are the ratings of the latest stamp parsed before the game's start. Hero MMR is causal for 81% of V2 slots; missing values fall back hero → role → player → global mean. Same protocol as section 2 (`p3_fix_mmr.py`, `results/fix/p3_fix_mmr.json`):

| added to logit WP | alone (gain vs M0) | on top of the skill estimate |
|---|---|---|
| player MMR | +0.0005 (0.0004, 0.0007) | |
| role MMR | +0.0011 (0.0009, 0.0015) | |
| hero MMR | +0.0025 (0.0021, 0.0030) | +0.0005 (0.0003, 0.0008) |
| all three | +0.0025 (0.0021, 0.0030) | +0.0006 (0.0004, 0.0009) |
| residual strength (CF rank 2) | +0.0142 (0.0131, 0.0152) | |

- Residual-based strength carries almost six times the information of the best MMR, and MMR adds little on top of it.
- Hero MMR is the useful MMR; player MMR adds almost nothing once the draft is known (the lobby is matched on it).
- Lagging MMR by whole days does not make it safe: a source game can be parsed after the current game at any lag. Upload-order-safe lagged MMR (P3_SKILL_DRIFT section 0) gives the same picture as the causal values: +0.0017 alone at a 7-day lag and +0.0008 on top of the skill model.

## 7. The experience offset

The offset is the mean residual in pp by games seen overall on earlier days (rows) × games seen on this hero on earlier days (columns), fit on E (`results/fix/p3_fix_mmr.json`).

| games seen \ on hero | 0 | 1 | 2 | 3-4 | 5-9 | 10-19 | 20-49 | 50+ |
|---|---|---|---|---|---|---|---|---|
| 0 | −1.5 | | | | | | | |
| 1-4 | −1.2 | +0.5 | +1.4 | +2.5 | | | | |
| 5-9 | −1.0 | +0.7 | +1.4 | +2.2 | +3.0 | | | |
| 10-19 | −1.1 | +0.4 | +0.9 | +1.8 | +2.6 | +2.9 | | |
| 20-49 | −2.0 | −0.3 | +0.2 | +1.1 | +1.9 | +2.8 | +3.1 | |
| 50-99 | −3.2 | −1.6 | −1.0 | −0.1 | +0.9 | +1.7 | +2.7 | +2.5 |
| 100-299 | −4.4 | −2.8 | −2.1 | −1.4 | −0.5 | +0.6 | +1.9 | +2.8 |
| 300+ | −6.6 | −4.6 | −4.1 | −3.0 | −2.0 | −0.9 | +0.4 | +2.3 |

- Players with 300+ games average 6.6pp below what the draft predicts when they play a hero they have not played in two years of data, and 2.3pp above it on heroes with 50+ games.
- This is the largest per-hero difference in the data, and it is known before the game.
- It is an average over the choices players made. Players stay on heroes that work for them and skip heroes they expect to do badly on, so the table describes "a player like you on a hero like this". It does not say what playing a new hero would cost, or what practice would do.
- It also bears on proposal experiment #1 (collapse to comfort picks): a personalized drafter will see a large comfort-pick bonus.

## Product recommendation (hotsfever)

1. **Model.** Rank-2 CF kernel plus experience offset, updated nightly with games through the previous day. No decay and a 365-day half-life are within 0.0001 of each other on V2; V1 preferred 365 days, which also has better calibration slopes, so 365 days is the pre-registered choice. Refit kernel weights and CF factors with the monthly production refresh.
2. **Per-hero row.** Show "vs expected: +3.9pp (80% band −0.1 to +7.9), 134 games".
   - Use the calibrated sd: 1.25 × posterior variance, for never-played heroes too (section 5b).
   - Show games played next to every number.
   - Never show raw WR as a strength measure. At 1 to 20 games it is mostly noise.
3. **Labels.** Label a hero "clear strength" or "clear weakness" only when the 80% band excludes zero; everything else is "unclear". In the example profile (a 533-game held-out healer main, `results/fix/p3_hs_display.txt`), only Whitemane (86 games, +4.5pp, band +0.7 to +8.3) is a clear strength. Ana (134 games, +3.9pp) just misses. That is the honest state of per-hero knowledge for a typical active player.
4. **Never-played heroes.** Show them with the off-pool offset and the first-game band (section 5b). The example player's never-played heroes sit at about −7pp, with an 80% band of roughly ±6pp. The text should say what this means, e.g. "players with your volume average −7pp in their first games on heroes new to them". Do not show a hero as a pooled "hidden strength" suggestion unless the CF estimate beats the off-pool penalty by more than its band. For tanks, fall back to the overall-skill estimate.
5. **Sorting and draft use.** Sort by the point estimate. In draft recommendations, add the strength estimate (not the band) to the population WP leaf value. It is already shrunk and calibrated (game-level slope 1.02).
6. **Leave out.** Leave hero_level out. Use MMR only as known when the game started. If MMR is shown, show it separately and date it.

## Caveats and next steps

- **In-sample WP.** E residuals are in-sample for the WP model (residual variance 0.2382 vs 0.2424 expected). Every variance-component fit (kernels, Kalman, jump, calibration) assumes game noise wp(1 − wp), so latent variances are somewhat overstated: the audit shows a 2% change in noise moves the Cassia latent R² from 21% to 15%, and the jump redo profiles the noise at 0.96 × wp(1 − wp) (audit P3-10). Treat latent R² and variance components as approximate. This probably also explains part of the 1.20 variance scale.
- **Experience counts.** These are window counts from 2024-04 onward, so returning veterans look new. Lifetime hero level from Heroes Profile would be better, but only a lagged version is usable.
- **Scope.** Only the snapshot is used; games after 2026-05-22 would need causal WP features built first. No map conditioning (proposal #5) and no party effects yet.
- **Next steps.**
  - A state-space version (random-walk skill) instead of exponential decay, which should fix the high-n slope.
  - The frozen paper-1 WP ablation.
  - Proposal #1 (collapse test) with these estimates, since the comfort-pick offset is large.
- **Compute.** Everything ran on CPU (numba). No GPU was used.

## Files

In `training/personalization/`:

- `p3_fetch_hero_mmr.py`: pulls hero_mmr, role_mmr, hero_level and region (DB read-only).
- `p3_hs_core.py`: slot table, experience offsets, kernels, numba GP posterior and marginal likelihood, causal online state, lagged values.
- `p3_hs_fit.py slots | fit`: kernel fitting on E. Outputs `cache/hs_kernels.npz` and `results/p3_hs_fit.json`.
- `p3_hs_eval.py`: online/static evaluation, sample-size tables, game-level lift, lagged MMR, raw coverage. Outputs `results/p3_hs_eval.json` and `.txt`.
- `p3_hs_nopool.py` and `p3_hs_paired.py`: the no-pooling baseline and paired bootstrap comparisons.
- `p3_hs_cassia.py`: masked cold-start test.
- `p3_hs_calib.py`: interval calibration (fit on V1, test on V2).
- `p3_hs_display.py`: sd-vs-n curve and the example profile.

- `p3_fix_counts.py`: the lag-1 count contract (`prepare_l1`) and `run_fixed`, which runs the scripts above on it; their outputs are in `results/fix/p3_hs_*.json` and `.txt`.
- `p3_fix_arms.py arms`: frozen and lagged arms with their own counts, and the player-cluster CI (`results/fix/p3_fix_arms.json`).
- `p3_fix_mmr.py`: causal MMR rows and the experience table (`results/fix/p3_fix_mmr.json`).
- `p3_fix_neverplayed.py`: section 5b (`results/fix/p3_fix_neverplayed.json`).

To reproduce, run from `training/` with `/usr/bin/python3` (it has numba and scipy): fetch → `p3_hs_fit.py slots` → `p3_hs_fit.py fit` → each of eval, nopool, paired, cassia, calib, display through `p3_fix_counts.run_fixed` (eval also needs the lag-1 table in its kernels file; see `P3_AUDIT_FIXES.md`) → `p3_fix_arms.py arms`, `p3_fix_mmr.py`, `p3_fix_neverplayed.py`. The full run takes about 1 hour on CPU.
