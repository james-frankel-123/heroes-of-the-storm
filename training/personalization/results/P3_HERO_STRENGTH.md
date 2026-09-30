# P3: per-hero player strength with honest uncertainty (2026-09-30)

## Summary

- **What a strength estimate is.** It is the expected draft-adjusted residual when player p plays hero h, in win-probability points. The residual is outcome minus the drift-aware WP. The estimate has three parts:
  - an experience offset that depends only on how many games we have seen from the player, overall and on this hero;
  - a Gaussian-process posterior across the player's 90 heroes;
  - a calibrated interval.

  Every estimate is causal: it uses games through the previous day only.
- **Game-level lift.** Test games are the same 72,474 held-out V2 games as P3_NESTED_LIFT. The population WP scores 57.19% accuracy and log loss 0.67647. The best hero-strength model (online, rank-2 CF kernel) scores 59.95% and 0.66229, a gain of +0.0142 (CI 0.0131 to 0.0152). That is 8x the static nested-lift M4 (+0.0017). Adding lagged MMR brings it to 60.81% and +0.0201.
- **Where the lift comes from.** Experience offsets alone give +0.0082. Players are 5 to 7pp worse than the draft predicts on heroes outside their pool, and 2 to 3pp better on heroes they have played 20+ times. Residual-based skill adds +0.0060 on top. Updating daily instead of freezing at the cutoff adds +0.0035.
- **Pooling.** A player-specific hero similarity helps most at small n. It is a rank-2 factor model of the player×hero residual matrix, fit by marginal likelihood. Its gain over no pooling is:
  - n = 1 to 10 games on the hero: +0.43 to +0.54 per 1000 slots, 43 to 71% more than the no-pooling gain;
  - n = 0 (hero never played): +0.57 per 1000;
  - n ≥ 11: nothing measurable.

  Role and fine-role groupings add almost nothing once overall skill is in the model (+0.018 per 1000 slots). Co-play similarity adds a little.
- **Sample size.** One game on a hero carries very little information. The prior sd of a player×hero cell is about 4pp and game noise is about 49pp, so a cell needs about 130 games before its own data outweighs the prior. The calibrated 80% band is ±5.4pp at 1 to 2 games, ±4.6pp at 21 to 50 games, and ±3.1pp at 200+ games.
- **Coverage.** Raw 80% posterior intervals cover the future cell mean 79% of the time overall. They cover only 73% on heroes the player had never played (62% when the player goes on to play 30+ games). After a two-number calibration fit on V1 and tested on V2, coverage is 80% overall and 79% on never-played heroes.
- **Lagged MMR.** Hero MMR as of the previous day is the best single MMR signal (+0.0088). It fades quickly: +0.0052 at 7 days old and +0.0026 at 30 days old. It stacks with the residual estimate.
- **Product.** Show strength vs expectation with an 80% band and games played. Sort heroes by estimate but label only bands that exclude zero. Say plainly that a never-played hero costs about 7pp for a veteran. Details are in the last section.

## Setup

- **Data.** Every Storm League game from 2024-04-01 through the snapshot (replay_id ≤ 63653039, last game 2026-05-22) with all ten `replay_players` rows. That is 10,822,691 player slots and 320,954 players. Players are keyed by (region, blizz_id); 1,502 blizz_ids appear in two regions. No post-snapshot games are used.
- **Residuals.** r = y_team − wp_team, using the three paper-2 `d2c_cumprev` seeds (the corrected `wp_drift.npz`). Game noise per slot is v = wp(1 − wp), about 0.242.
- **Windows.**
  - E, estimation: 2024-04-01 up to the WP training cutoff. It has 9.39M slots, and all hyperparameters are fit here.
  - V1: 2026-02-10 to 2026-04-01, 70,974 games. Used to fit combiner weights, the half-life choice and interval calibration.
  - V2: 2026-04-01 to 2026-05-22, 72,474 games. The test set.
- **Online state.** For a game on day t, each player's per-hero sufficient statistics use every game (E, V1, V2) from days ≤ t − lag, with lag = 1 unless stated. The statistics are S = Σ r/v and P = Σ 1/v, optionally with exponential decay. "Static" means frozen at the cutoff.
- **Experience offset.** This is the mean residual by (games seen from the player, games seen on this hero), in 8×8 bins, fit on E and shrunk toward 0. It uses counts only, never outcomes. It is subtracted from every residual before the GP and added back to the prediction.
- **Kernels.** For one player, θ over the 90 heroes follows N(0, K). K is a weighted sum of:
  - overall skill (all-ones J);
  - Blizzard role, fine role (`HERO_ROLE_FINE`) and melee/ranged blocks;
  - an independent hero term (I);
  - a similarity term, either co-play (cosine similarity of 16-d SVD hero vectors from play shares) or CF (V Vᵀ with V a 90×r matrix fit by marginal likelihood, i.e. probabilistic matrix factorization of the residual matrix with player factors integrated out).

  Weights and V are fit by exact marginal likelihood on a random half of players. Kernels are compared on the other half.
- **Metrics.**
  - Game level: logistic combiner of logit(WP) and the team difference of summed slot estimates, fit on V1 and scored on V2, with replay bootstrap CIs.
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
| experience offset only | 59.01 | 0.66830 | +0.0082 (0.0073, 0.0090) |
| player (overall skill) | 59.42 | 0.66568 | +0.0108 (0.0098, 0.0117) |
| player+hero | 59.82 | 0.66329 | +0.0132 (0.0121, 0.0141) |
| player+role+melee+hero | 59.83 | 0.66314 | +0.0133 (0.0123, 0.0142) |
| + co-play | 59.84 | 0.66309 | +0.0134 (0.0123, 0.0142) |
| **+ CF rank 2** | **59.95** | **0.66229** | **+0.0142 (0.0131, 0.0152)** |
| + CF rank 2, static at cutoff | 59.36 | 0.66578 | +0.0107 (0.0098, 0.0116) |
| + CF rank 2, residuals lagged 7 days | 59.80 | 0.66382 | +0.0126 (0.0116, 0.0137) |
| + CF rank 2, residuals lagged 30 days | 59.49 | 0.66484 | +0.0116 (0.0107, 0.0125) |
| + CF rank 2 + lagged player/role/hero MMR (G = 1 day) | 60.81 | 0.65632 | +0.0201 (0.0189, 0.0216) |
| + CF rank 2 + lagged MMR (G = 7 days) | 60.42 | 0.65918 | +0.0173 (0.0161, 0.0185) |
| both lagged 7 days | 60.13 | 0.66077 | +0.0157 (0.0145, 0.0170) |

Everything except M0 includes the experience offset, which is fit on E. Calibration stays good: ECE ≤ 0.005 and calibration slope 0.98 to 1.03 for every row.

Paired differences (replay bootstrap):

- CF rank 2 vs player+hero: +0.0010 (0.0006, 0.0014).
- player+hero vs no pooling: +0.0017 (0.0012, 0.0022).
- CF rank 2 vs no pooling: +0.0027 (0.0020, 0.0033).
- Role blocks vs player+hero: +0.00015 (0.00006, 0.00026).

**Decay.** Exponential down-weighting of old games does not help. The rank-2 CF model gains +0.0142 with no decay, +0.0140 with a 365-day half-life, +0.0135 at 180 days and +0.0126 at 90 days. V1 slightly preferred 365 days (+1.291 vs +1.273 per 1000 slots). Decay does fix the calibration slope at high n (0.68 to 0.82 for 101 to 200 games). Recency matters through staleness: a 7-day lag costs 0.0016 and a 30-day lag costs 0.0026. Down-weighting old games does not buy much.

## 3. Sample-size curve (V2 slots, online, lag 1 day)

Per-slot log-loss gain ×1000 over the population WP, with estimates used as-is. The columns are the number of games the player had on this hero in the window before the game.

| games on hero | 0 | 1-2 | 3-5 | 6-10 | 11-20 | 21-50 | 51-100 | 101-200 | 201+ | all |
|---|---|---|---|---|---|---|---|---|---|---|
| share of slots % | 19.4 | 16.4 | 13.2 | 12.2 | 12.3 | 13.5 | 6.7 | 3.8 | 2.5 | |
| raw WR − 0.5 | 0 | −1278 | −337 | −113 | −42 | −18 | −6.7 | −1.1 | +0.3 | −276 |
| experience offset only | 1.30 | 0.63 | 0.81 | 0.53 | 0.59 | 0.53 | 0.79 | 1.61 | 1.18 | 0.82 |
| no pooling (cell shrunk alone) | 1.30 | 0.76 | 1.05 | 0.89 | 1.24 | 1.08 | 1.32 | 2.41 | 2.06 | 1.15 |
| player+hero | 1.80 | 1.14 | 1.33 | 1.11 | 1.25 | 0.95 | 1.31 | 2.15 | 2.05 | 1.35 |
| + CF rank 2 | **1.86** | **1.30** | **1.51** | **1.32** | 1.19 | 1.01 | 1.46 | 2.17 | 1.99 | **1.45** |
| lagged hero MMR, G = 1 (linear map fit on V1) | −0.45 | 1.06 | 1.03 | 1.27 | 1.52 | 1.50 | 1.89 | 2.39 | 2.01 | 1.03 |
| CF slope of r on estimate | 1.32 | 1.06 | 1.12 | 1.01 | 0.93 | 0.74 | 0.75 | 0.68 | 0.80 | 0.93 |
| CF posterior sd, median pp | 3.94 | 3.83 | 3.72 | 3.61 | 3.48 | 3.27 | 2.98 | 2.65 | 2.15 | |
| **calibrated display sd pp** | **8.3** | **4.2** | **4.1** | **4.0** | **3.8** | **3.6** | **3.3** | **2.9** | **2.4** | |

What pooling buys at each n: the paired difference (×1000 per slot) with 95% CIs from a player-clustered bootstrap.

| games on hero | 0 | 1-2 | 3-5 | 6-10 | 11-20 | 21-50 | 51-100 | 101-200 | 201+ |
|---|---|---|---|---|---|---|---|---|---|
| player+hero − no pooling | +0.50 (0.40, 0.60) | +0.38 (0.23, 0.50) | +0.28 (0.12, 0.43) | +0.23 (0.06, 0.41) | +0.01 | −0.13 | −0.01 | −0.27 (−0.47, −0.06) | −0.02 |
| CF rank 2 − player+hero | +0.07 (−0.02, 0.16) | +0.16 (0.05, 0.26) | +0.18 (0.04, 0.30) | +0.20 (0.07, 0.34) | −0.06 | +0.06 | +0.15 (0.00, 0.30) | +0.02 | −0.06 |
| CF rank 2 − no pooling | +0.57 (0.45, 0.70) | +0.54 (0.36, 0.69) | +0.46 (0.24, 0.65) | +0.43 (0.19, 0.67) | −0.05 | −0.07 | +0.14 | −0.25 | −0.07 |

Reading:

- **Raw win rate is useless at typical sample sizes.** A raw WR from 1 to 20 games is far worse than predicting nothing. Its slope is 0.03 to 0.15, meaning 85 to 97% of the raw number is noise.
- **Pooling pays at 0 to 10 games.** That covers 61% of slots. At 11+ games the cell's own data matches the pooled estimate, and at 100+ games pooling pulls slightly too hard toward the player's other heroes.
- **The uncertainty barely shrinks for a long time.** The display sd goes from 4.2pp at 1 game to 3.6pp at 21 to 50 games and 2.4pp at 200+. The prior-to-noise ratio is k = σ²/τ² ≈ 0.242 / 0.04² ≈ 150 games (130 in the nested-lift MoM fit). Half the display width at 100 games is still prior.
- **The shrinkage level is about right.** CF slopes are 0.93 to 1.12 from 1 to 20 games. Two exceptions:
  - At n = 0 the slope is 1.3: the offset for never-played heroes is if anything too small.
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
| + CF rank 2 + co-play | 0.22 | 2.62 | 3.50 | 0.80 |

Findings:

- Pooled information predicts about a fifth of a player's true effect on a hero they barely play. The CF kernel explains 33% more than player+hero.
- **By role (CF rank 2):**
  - ranged assassin 0.25, melee assassin 0.29, bruiser 0.18, support 0.17, healer 0.04;
  - tank −1.0, which is worse than predicting zero. Rarely-played tank cells have little player-specific spread and the pooled estimate adds error there.
- Coverage is 0.80, but the realized error variance is only 0.56x the claimed variance, so intervals are conservative for this population.
- **Revealing the cell's first k games.** The other variants are scored on the remaining games:
  - CF rank 2: R² 0.22 (k = 1), 0.18 (k = 3), 0.23 (k = 5);
  - player+hero: 0.16, 0.09, 0.10.
  - The raw mean of the revealed games scores R² −314 at k = 1 and −81 at k = 5. It should never be shown.

## 5. Interval coverage

Setup: the posterior as of a date, compared with the mean residual of the same player×hero cell in the following ~7 weeks. The 80% predictive interval is m ± 1.28 √(s² + game noise / n_future). The table uses CF rank 2 with no decay.

| as of V1/V2 split, cells in V2 with ≥ 10 future games | all | 0 | 1-2 | 3-5 | 6-10 | 11-20 | 21-50 | 51-100 | 101-200 | 201+ |
|---|---|---|---|---|---|---|---|---|---|---|
| cells | 8,678 | 1,407 | 612 | 631 | 738 | 1,016 | 1,667 | 1,244 | 892 | 471 |
| raw posterior | 0.79 | 0.73 | 0.76 | 0.79 | 0.79 | 0.79 | 0.83 | 0.82 | 0.79 | 0.82 |
| calibrated | 0.80 | 0.79 | 0.77 | 0.79 | 0.79 | 0.80 | 0.83 | 0.82 | 0.79 | 0.82 |

**Cells with ≥ 30 future games.** Raw coverage is 0.77 overall and 0.62 on never-played heroes. Calibrated coverage is 0.79 overall and 0.75 on never-played heroes.

**Calibration.** It was fit on the cutoff → V1 cells (28,629 cells with ≥ 5 future games): s²_display = 1.20 s² + (7.0pp)² × [no games on this hero]. The raw posterior is 20% too narrow for cells with history. It is far too narrow for heroes a player takes up, because adopters vary much more than the kernel implies (learning, and selection on who keeps playing). The fitted drift term is zero once the scale is applied. Refitting with extra terms for 1 to 5 games gave zero weight.

The two cold-start populations point in opposite directions:

- heroes a player newly takes up and keeps playing need wider bands (+7pp);
- heroes a veteran dabbles in rarely (the Cassia cells) are over-covered.

For the product I recommend the wider band for never-played heroes, because that is the "should I pick this" question.

Single-game coverage (nf ≥ 1) is 0.85 to 0.87. That is expected: one game's residual is bounded, so a Gaussian interval over-covers. It is not evidence of good calibration.

## 6. Hero MMR (lagged only)

> **Correction (phase 2, 2026-09-30):** the lagged MMR in this section is the naive definition and leaks at every lag. A source game uploaded after the current game carries an MMR that already includes the current result. Upload-order-safe lagged MMR is worth +0.0017 alone at G = 7 (naive +0.0048) and +0.0009 on top of the skill model. See P3_SKILL_DRIFT.md, section 0.

`hero_mmr`, `role_mmr` and `player_mmr` are recorded when Heroes Profile parses the replay, so the value on a game is not usable for that game. Everything below uses the latest value from a game at least G days earlier. Role MMR is keyed by player × Blizzard role; missing values fall back hero → role → player → global mean.

| alone, game level (gain vs M0) | G = 1 | G = 7 | G = 30 |
|---|---|---|---|
| lagged player MMR | +0.0043 | +0.0019 | +0.0005 |
| lagged role MMR | +0.0054 | +0.0028 | +0.0011 |
| lagged hero MMR | +0.0088 | +0.0052 | +0.0026 |
| all three | +0.0096 | +0.0054 | +0.0026 |
| residual strength (CF rank 2) at the same lag | +0.0142 | +0.0126 | +0.0116 |
| residual strength + all three MMR | +0.0201 | +0.0157 | +0.0127 |

- Residual-based strength beats lagged MMR at every lag.
- It also decays much more slowly with staleness: 0.0142 to 0.0116 from 1 to 30 days, while MMR drops from 0.0096 to 0.0026.
- The two are complementary. MMR carries recent form and volume; the residual carries a draft-adjusted, hero-specific level.
- Per slot, lagged hero MMR is stronger than the residual estimate at 11+ games (1.5 to 2.4 vs 1.0 to 2.2 per 1000). It is useless at 0 games, because the fallback to player MMR has the wrong sign relative to the off-pool penalty.
- The G = 1 MMR numbers carry some leak risk. A replay from yesterday may have been parsed by Heroes Profile after today's game. G = 7 is the safe headline.

## 7. The experience offset

The offset is the mean residual in pp by games seen overall (rows) × games seen on this hero (columns), fit on E:

| games seen \ on hero | 0 | 1 | 2 | 3-4 | 5-9 | 10-19 | 20-49 | 50+ |
|---|---|---|---|---|---|---|---|---|
| 0 | −2.8 | | | | | | | |
| 1-4 | −1.6 | +1.1 | +2.1 | +3.1 | | | | |
| 5-9 | −1.3 | +0.9 | +1.6 | +2.6 | +3.3 | | | |
| 10-19 | −1.5 | +0.6 | +1.2 | +2.1 | +2.9 | +3.4 | | |
| 20-49 | −2.2 | −0.3 | +0.3 | +1.2 | +2.0 | +3.0 | +3.1 | |
| 50-99 | −3.5 | −1.7 | −1.0 | 0.0 | +0.9 | +1.8 | +2.9 | +2.5 |
| 100-299 | −5.0 | −2.9 | −2.1 | −1.4 | −0.4 | +0.6 | +1.9 | +2.8 |
| 300+ | −6.9 | −5.1 | −4.4 | −3.2 | −2.0 | −0.9 | +0.5 | +2.3 |

- A player with 300+ games is 6.9pp worse than the draft predicts on a hero they have not played in two years of data. They are 2.3pp better on a hero with 50+ games.
- This is the largest per-hero effect in the data, and it is known before the game.
- Part of it is selection: players stay on heroes that work for them. So it describes "you on an off-pool hero". It does not describe what practice would do.
- It also bears on proposal experiment #1 (collapse to comfort picks): a personalized drafter will see a large comfort-pick bonus.

## Product recommendation (hotsfever)

1. **Model.** Rank-2 CF kernel plus experience offset, updated nightly with games through the previous day and no decay. A 365-day half-life is an equally good option with slightly better calibration slopes. Refit kernel weights and CF factors with the monthly production refresh.
2. **Per-hero row.** Show "vs expected: +3.8pp (80% band −0.1 to +7.8), 134 games".
   - Use the calibrated sd: 1.20 × posterior variance, plus (7pp)² for never-played heroes.
   - Show games played next to every number.
   - Never show raw WR as a strength measure. At 1 to 20 games it is mostly noise.
3. **Labels.** Label a hero "clear strength" or "clear weakness" only when the 80% band excludes zero; everything else is "unclear". In the example profile (a 533-game held-out healer main, `p3_hs_display.txt`), only Whitemane (86 games, +4.5pp, band +0.8 to +8.3) is a clear strength. Ana (134 games, +3.8pp) just misses. That is the honest state of per-hero knowledge for a typical active player.
4. **Never-played heroes.** Show them with the off-pool penalty and a wide band. The example player's never-played heroes sit at about −7pp (band −17 to +3). The text should say what this means, e.g. "players with your volume average −7pp on heroes new to them". Do not show a hero as a pooled "hidden strength" suggestion unless the CF estimate beats the off-pool penalty by more than its band. For tanks, fall back to the overall-skill estimate.
5. **Sorting and draft use.** Sort by the point estimate. In draft recommendations, add the strength estimate (not the band) to the population WP leaf value. It is already shrunk and calibrated (game-level slope 1.02).
6. **Leave out.** Leave hero_level out. Use MMR only lagged. If MMR is shown, show it separately and date it.

## Caveats and next steps

- **In-sample WP.** E residuals are in-sample for the WP model (residual variance 0.2382 vs 0.2424 expected). This probably explains part of the 1.20 variance scale.
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

To reproduce, run from `training/` with `/usr/bin/python3` (it has numba and scipy): fetch → `p3_hs_fit.py slots` → `p3_hs_fit.py fit` → eval, nopool, paired, cassia, calib, display. The full run takes about 1 hour on CPU.
