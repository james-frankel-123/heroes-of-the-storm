# P3 rich CF: richer skill models than rank-2 CF (2026-10-16, updated 2026-10-17)

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

## Update 2026-10-17: scoreboard cards, display scale, temporal dynamics

Max's decisions after the interim report:
- skip variant 5;
- drop role and melee for good;
- run variant 6;
- extend the scoreboard export to every corpus row;
- refit the display scale on V1;
- use `met2_hs2` as the main model and `imp2` for ranking never-played heroes.

He also asked for the scoreboard normalization cards below.

### Headline of the update

- **Extending the export gives a large gain.** The export now holds all 24.6M corpus player-games, including those before 2024-04-01. As a result 3.64M Heroes Profile history rows that are also corpus games now carry a scoreboard. The same model and normalization (`sb_base`) lift V2 to **+25.3** and OOT to **+25.3**, against +23.5 and +23.7 for `met2_hs2`. That is +1.8 (SE 0.24) on V2 and +1.6 (SE 0.12) on OOT. Over today's model (A2wS) it is +8.2 and +8.5.
- **No normalization card clearly beats the current one.**
  - Shrinking thin hero×map cells (`a_shrink`) ties on lift and raises cold-start R² from 0.24 to 0.35 (+0.11, SE 0.02). It is the recommended normalization.
  - Rank transforms gain +0.3 on lift but cost cold start.
  - Tier cells, smooth length curves, team shares and same-role opponent differences are all worse.
  - The data do not support weighting games by kill margin.
- **Display scale, fit on V1.**
  - `met2_hs2`: display variance 2.37·s² + (2.3pp)². V2 coverage at nominal 80% goes from 0.779 raw to 0.804 scaled.
  - `imp2`: 2.05·s² + (0.9pp)². Coverage goes from 0.787 to 0.804.
- **Temporal dynamics (variant 6): the per-player layer adds +1.4 (SE 0.20) on V2 and +1.6 (SE 0.10) on OOT** on top of `a_shrink`. Its two terms:
  - a per-player new-hero learning term: +1.3 alone;
  - a per-player residual level: +1.1 alone. V1 picks the longest half-life, so this term corrects the player's general level rather than tracking short-term form.

  Plain recency forgetting of the evidence is within noise on V1 and V2. It is +0.3 to +0.4 on OOT.
- **Best model so far:** `a_shrink` plus the layer. V2 lift +26.8, against +17.1 for today's model. OOT lift +26.9, against +16.9. Accuracy 61.9% on V2 and 61.5% on OOT, against 60.3% and 60.0%.

### How `met2_hs2` normalizes (stated exactly)

This is `p3_rcf_data.py`, the export from 2024-04-01 on. For each of the 15 metrics and each player-game:

1. **Rate.** rate = value / game minutes.
2. **Transform.** t = asinh(rate / s), where s is the median positive rate over E rows. This is a log-like transform that keeps zeros.
3. **Standardize.** z = (t − mean) / sd within the cell (hero, map, length bin, outcome).
   - Length bins: under 14, 14-17, 17-20, 20-24 and 24+ minutes.
   - Cell means and sds come from E rows only (2024-04-01 to 2026-02-10).
   - A cell with fewer than 50 E rows falls back to (hero, length bin, outcome). This is a hard switch: there is no shrinkage and no tier.
4. **Clip** at ±5.
5. **Center by era.** Subtract the (hero, calendar quarter) mean of E rows. Later quarters use the last E quarter.
6. **Mask.** A value is missing where the metric is uninformative for the hero: fewer than 5% nonzero E rows, or a mean cell sd of 0.02 or less.

Outcome is part of the cell, so z measures performance relative to other games on the same hero, map and length with the same result. It carries no information about its own game's result. Only games on earlier days enter a prediction.

### Scoreboard cards

Code: `p3_rcf_cards.py`, which writes `cache/rcf/sb_<card>.npz` for every export row.

Every card is fit with `met2_hs2`'s settings: rank 2, metric loadings shared across heroes, per-hero s², the predictive objective, merged history. Every card gets the same tests. "vs base" is paired against `sb_base` (x1000 log loss, SE by game). Cards are chosen on V1 lift and inner-validation LL; V2 and OOT are the test.

| card | what changes | V1 lift | val LL | V2 lift | V2 vs base (SE) | OOT vs base (SE) | OOT all-10 vs base (SE) | Cassia R² (vs base, SE) | first-game LL vs base (SE) | cover80 scaled |
|---|---|---|---|---|---|---|---|---|---|---|
| `met2_hs2` (first export) | slot rows only | 22.39 | | 23.48 | −1.80 (0.24) | −1.61 (0.12) | −2.49 (0.99) | 0.362 (+0.12, 0.03) | −0.31 (0.09) | 0.804 |
| **base** | the normalization above, on the full export | 23.90 | −59,811.9 | 25.28 | 0 | 0 | 0 | 0.238 | 0 | 0.804 |
| **a_shrink** | hero×map mean and variance shrunk toward hero (100 pseudo-games) | 23.85 | −59,821.6 | 25.34 | +0.07 (0.14) | +0.05 (0.07) | +0.50 (0.56) | **0.351 (+0.113, 0.017)** | +0.00 (0.05) | 0.804 |
| a_tier | + hero×map×tier (5 groups), shrunk | 23.52 | −59,815.6 | 24.83 | −0.45 (0.20) | −0.48 (0.10) | +0.97 (0.76) | 0.310 (+0.072, 0.026) | −0.10 (0.07) | 0.804 |
| b_len | totals; smooth expected mean and sd by length within hero×outcome; shrunk map shift | 23.69 | −59,814.6 | 25.04 | −0.24 (0.20) | −0.37 (0.10) | +0.95 (0.79) | 0.322 (+0.084, 0.027) | −0.05 (0.07) | 0.804 |
| c_share | share of the team total | 23.54 | **−59,805.9** | 24.83 | −0.44 (0.24) | −0.76 (0.12) | −1.93 (0.72) | 0.323 (+0.085, 0.022) | −0.09 (0.08) | 0.808 |
| c_opp | minus the same-role opponents' mean | 23.19 | −59,832.1 | 23.86 | −1.42 (0.23) | −1.34 (0.11) | −1.75 (0.63) | 0.273 (+0.035, 0.017) | −0.13 (0.08) | 0.804 |
| d_rank | normal scores of the E rank within the cell | **24.33** | −59,820.3 | **25.59** | **+0.32 (0.12)** | **+0.27 (0.06)** | −0.33 (0.36) | 0.173 (−0.065, 0.011) | +0.04 (0.04) | 0.806 |
| d_wins | clip at ±2.5 | 24.06 | −59,817.7 | 25.46 | +0.18 (0.09) | +0.17 (0.05) | +0.80 (0.35) | 0.288 (+0.050, 0.011) | +0.01 (0.03) | 0.805 |
| e_weight | per-game weight by kill-margin informativeness | 23.99 | −59,812.5 | 25.41 | +0.13 (0.13) | −0.10 (0.06) | −0.40 (0.43) | 0.202 (−0.036, 0.014) | +0.05 (0.05) | 0.805 |

Val LL is the inner-validation predictive LL (20,978 players). Differences of about 15 are within noise there.

Cassia R² by role (tank / bruiser / healer / ranged / melee / support):
- base: −5.32 / 0.07 / 0.00 / 0.23 / 0.37 / −0.08;
- a_shrink: −6.37 / 0.13 / 0.08 / 0.35 / 0.43 / 0.01;
- d_rank: −4.68 / 0.02 / −0.11 / 0.19 / 0.38 / −0.09.

Reading the cards:
- **(a) Hero×map cells.** The hard 50-row fallback was the only real weakness of the current normalization. On the full export it matters for old maps and thin cells. Shrinkage fixes cold start (R² 0.24 to 0.35) at no cost in lift. Adding tier cells costs 0.45 to 0.5 on lift: the outcome-conditioned hero×map cell already absorbs what tier explains, and tier cells are thin.
- **(b) Game length.** A smooth curve of expected value by length, within hero, is no better than per-minute rates in five bins: −0.24 V2 and −0.37 OOT. The linear per-minute rate is not what limits the model.
- **(c) Context.** Team share is slightly worse, and the difference from the same-role opponent is clearly worse (−1.4). Both remove the player's absolute level. The absolute level is the part that predicts the next game, because teammates and opponents change from game to game.
- **(d) Robust transforms.** Ranks gain +0.3 on V2 and OOT (best on V1) and winsorizing at ±2.5 gains +0.2, so the heavy tails cost a little. But ranks lower cold-start R² to 0.17. Not adopted; the trade-off is open if pure lift is the goal.
- **(e) Informativeness weighting.** For each kill-margin quintile I measured how much a game's z moves the player's leave-one-out hero mean. The weights come out 0.999, 1.006, 1.012, 1.008 and 0.976 from closest to biggest stomp. Close games and stomps are equally informative once z is outcome-conditioned, so weighting is not justified. It changes lift by +0.13 and −0.10.

### Display scale refit on V1

Method: posterior as of the end of E, against cells with 5+ V1 games. Fit display variance = c·s² + k0² by maximum likelihood. Test: posterior as of the V1/V2 split, against cells with 10+ V2 games (8,678 cells).

| model | c | k0 | V2 cover80 raw / scaled | games on the hero at the split: 0 / 1-5 / 6-20 / 21-100 / 101+ (scaled) | median display sd |
|---|---|---|---|---|---|
| `met2_hs2` | 2.37 | 2.3pp | 0.779 / **0.804** | 0.77 / 0.78 / 0.80 / 0.80 / 0.82 | 4.7pp (8.5 at 0 games, 3.9 at 101+) |
| base, full export | 2.21 | 2.6pp | 0.779 / 0.804 | 0.77 / 0.78 / 0.80 / 0.80 / 0.82 | 4.7pp |
| a_shrink | 2.15 | 2.5pp | 0.780 / 0.804 | 0.78 / 0.78 / 0.80 / 0.80 / 0.82 | 4.6pp |
| `imp2` | 2.05 | 0.9pp | 0.787 / **0.804** | 0.75 / 0.80 / 0.81 / 0.82 / 0.82 | 4.6pp |

With the scale, realized-to-claimed variance goes from about 2.4 to 0.89 for every metric model. The scaled band is honest overall and slightly wide on heavy cells. Never-played cells in the selected coverage population are still 3 to 5 points under nominal, for the reason in P3_HERO_STRENGTH 5b.

### Variant 6: temporal dynamics

All arms are on `a_shrink`. Choices are made on V1 lift; V2 and OOT are the test. SE is paired by game.

| arm | V1 lift | V2 lift | V2 vs a_shrink (SE) | V2 all-10 | OOT lift | OOT vs a_shrink (SE) | OOT all-10 | first-game bias pp | first-game LL vs a_shrink (SE) |
|---|---|---|---|---|---|---|---|---|---|
| a_shrink (static) | 23.85 | 25.34 | 0 | 0 | 25.38 | 0 | 0 | +0.66 | 0 |
| 6a forgetting, metrics half-life 365 d | 23.87 | 25.41 | +0.07 (0.11) | +0.16 (0.43) | 25.64 | +0.26 (0.06) | −0.65 (0.46) | +0.47 | −0.02 (0.04) |
| 6a forgetting, metrics 120 d | 23.87 | 25.43 | +0.08 (0.19) | +0.45 (0.70) | 25.81 | +0.43 (0.10) | −1.52 (0.75) | +0.38 | −0.06 (0.07) |
| 6a forgetting, metrics 40 d | 23.59 | 25.07 | −0.27 (0.28) | +0.45 (0.99) | 25.63 | +0.25 (0.14) | −2.15 (1.04) | +0.28 | −0.14 (0.11) |
| 6a forgetting, win 730 d | 23.78 | 25.23 | −0.12 (0.18) | −1.01 (0.90) | 25.35 | −0.03 (0.09) | +0.23 (1.01) | +0.73 | −0.05 (0.03) |
| 6a forgetting, win and metrics 365 d | 23.68 | 25.17 | −0.18 (0.26) | −1.63 (1.29) | 25.75 | +0.36 (0.13) | −0.98 (1.45) | +0.55 | −0.09 (0.05) |
| **6b+6c per-player layer** | **25.21** | **26.78** | **+1.43 (0.20)** | +0.15 (0.38) | **26.93** | **+1.55 (0.10)** | +0.55 (0.33) | +0.46 | **+0.33 (0.10)** |
| layer + metrics 120 d | 25.25 | 26.87 | +1.52 (0.27) | +0.70 (0.75) | 27.34 | +1.96 (0.14) | −0.91 (0.80) | +0.20 | +0.23 (0.13) |

**The layer** (`temporal_layer` in `p3_rcf.py`).
- u = r − (the model's own as-of prediction) for every past slot game of the player, from games on earlier days only.
- Two per-player terms:
  - drift: a decayed mean of u with half-life h, shrunk with k_d pseudo-games;
  - new-hero learning: the mean of u over the player's earlier first-5 games on any hero, shrunk with k_e. It applies when the query hero has fewer than 5 earlier games in the full history.
- h, k_d and k_e are chosen on V1 (h 730 days, k_d 10, k_e 5). Their weights are the V1 combiner coefficients.

Ablations (gain over the same model without the layer, SE by game):

| layer | V1 | V2 | OOT |
|---|---|---|---|
| drift only | +0.99 (0.17) | +1.12 (0.16) | +1.22 (0.08) |
| new-hero learning only | +1.29 (0.19) | +1.33 (0.19) | +1.43 (0.09) |
| both | +1.36 (0.19) | +1.43 (0.20) | +1.55 (0.10) |

What this says:
- **Per-player learning on new heroes is real.** Players differ, stably, in how they do on heroes new to them, beyond the population offset by experience. A player's past first games predict their next first games. On first games the layer improves log loss by +0.33 (SE 0.10) and moves the bias from +0.66 to +0.46pp. Max's veteran-optimism point now runs the other way: the metric model is slightly pessimistic on first games, and the layer halves that.
- **There is no evidence for short-term drift.** The drift term prefers the longest half-life tried (730 days over 45 and 180). Short half-lives do worse on V1, and forgetting the evidence itself does not help on V1 or V2. The +1.1 from "drift only" is a per-player correction of the model's general level: the factor model still shrinks a player's overall level slightly too hard.
- **The gain is mostly on thin-history players.** On all-10-pulled lobbies the layer adds only +0.15 (V2) and +0.55 (OOT), both within noise. Full histories already pin down the level.
- **Coverage of the layer is not refit yet.** The layer moves only the mean.

Leak checks for the update: `tests/test_rcf.py` passes 6 of 6. New checks:
- R5: evidence weights and win and metric half-lives in the walk. Planted same-day and later rows, with changed outcomes, metrics and weights, change no prediction.
- R6: the layer statistics of a game do not move when its own or any later residual changes, and a same-day reader is caught on 25 of 25 probes.

`test_merged` and `test_full_history` still pass.

### What full replay timelines could add (scope only, nothing built)

**What the parser stores per game** (`training/replays/parse_replay.py`; I read only the code and the index, never a timeline):
- Per player:
  - deaths, with time, position and killers;
  - talent picks, with times;
  - APM;
  - the player's toon id, which equals blizz_id.
- Per team:
  - periodic XP (minion, creep, structure, hero and trickle) and team level;
  - merc captures;
  - structure losses;
  - map-objective events.
- Not stored: continuous hero positions. Positioning and rotations would need a parser extension that keeps `SUnitPositionsEvent` samples, which come about every 15 s for units that dealt or took damage.

**Candidate per-player signals.**

| signal | from | level | note |
|---|---|---|---|
| deaths before 10:00, and before the first objective | death list | player | the cleanest early-game skill signal; likely as reliable per game as scoreboard deaths, and earlier in the game |
| death positions (in the enemy half, alone) | positions, killers | player | an overextension proxy |
| early takedown participation | killer lists | player | |
| APM | per-player APM | player | likely a high-reliability mechanical trait |
| early XP and level lead at 5 and 10 min | XP periodic | team | shared by 5 players, so weak per player; best used as a lobby feature, not a skill channel |
| objective and merc participation | team events | team | per-player attribution needs the position extension |
| rotations | not available | | needs the parser extension |

**Coverage and tranche rules.**
- 15,508 timelines are parsed so far, builds 2.57.0.98285 to 98348.
- Files live 28 days after upload. The download quota is 25K a week.
- Every 2.57 game is assigned to exactly one of R (sealed), M (cohort B and the interesting cohorts), the interesting pool, C or F (FRAMING.md §4.2). So every existing timeline is paper 3's M tranche:
  - M-explore is for inspection;
  - M-test is for charged aggregate queries.

  FRAMING gives no use of M as P3 features. Using M timelines for P3 needs Max's recorded decision; I did not use them. R is never touched, and the parser refuses sealed paths.
- Scoring P3 on 2.57 outcomes also reads C or F games:
  - C has no interactive access until its day-14 batch;
  - each F block is scored once by paper 3, then released as fit and dev data 4 days after the block ends.

**Is a prospective test from about Oct 20 feasible?** Yes, but only in this form:
1. **A P3 download cohort.** Max assigns part of the quota left after cohort B (about 16K a week from week 2, shared with the interesting cohorts). Games are selected by panel membership only (lobbies with many pulled players), never by outcome. About 5K to 8K games a week gives a panel player 2 to 4 timeline games a week, so about 10 per player after 4 weeks.
2. **Features as-of.** Per-player timeline aggregates enter as one more metric channel in the same factor model, from games on earlier days only, never from the scored game's own timeline.
3. **Score only after release.** Score on games from the F1 block (Oct 27 to Nov 23), after its release around Nov 27, with every choice frozen beforehand. Games from Oct 20 to 26 are C and wait for C's release.

**Expected value.** The scoreboard already holds end-of-game totals for every corpus game, and it is +8 over today's model. Timelines add early-game timing (deaths before 10 min) and APM. My guess is a fraction of a log-loss point, with most of it on players with thin history. The test is worth running only if Max wants mechanism evidence on early skill, not for lift.

### Updated recommendation for the Oct 28 rerun

1. **Main skill model:** `met2_hs2`'s structure on the full scoreboard export with the `a_shrink` normalization.
   - Overall column plus 2 factors, shared metric loadings, per-hero s².
   - No role, fine-role or melee features.
   - Plus the per-player layer: drift half-life 730 d with k_d 10, new-hero learning with k_e 5.
   - V2 +26.8 and OOT +26.9, against +17.1 and +16.9 for today's model.
2. **Display:** variance 2.15·s² + (2.5pp)², which gives 0.80 coverage on V2. Refit c and k0 on V1 after the rerun, and refit them with the layer included.
3. **Never-played ranking:** `imp2` (Cassia 0.49), display 2.05·s² + (0.9pp)². Tanks fall back to overall skill.
4. **A1 rerun:**
   - The export (tag `rcf1017`) already covers the pre-2024 corpus rows A1 adds. Rebuild the cards there with `p3_rcf_cards.py --tag rcf1017`, then `p3_rcf.py prep --sb a_shrink` on the A1 cache.
   - Refit on the tranche-2 store.
   - Rerun `tests/test_rcf.py`, `test_merged.py` and `test_full_history.py`.
5. **Not adopted:**
   - tier cells, smooth length curves, team shares and opponent differences: all worse;
   - kill-margin weighting: weights are flat;
   - plain forgetting: noise on V1 and V2;
   - rank transforms: +0.3 lift but worse cold start. Max's call if lift is all that matters.

Files for the update, in `training/personalization/`:
- `p3_rcf_cards.py`;
- `p3_export.py` (the scoreboard export now covers all dates and adds team, outcome, tier and end time);
- `p3_rcf.py` (aug-row metrics with weights, half-lives, `temporal_layer`, V1 display scale in `cover`);
- `tests/test_rcf.py` (R5, R6);
- `remote/run_rcf_cards.sh`;
- `results/rcf/report_{cards,temporal}.{txt,json}`.

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

Superseded by the 2026-10-17 recommendation at the top of this file; kept as the record of the interim reasoning.

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
