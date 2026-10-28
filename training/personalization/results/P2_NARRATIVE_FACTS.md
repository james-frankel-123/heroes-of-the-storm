# Paper 2 narrative facts (2026-10-19)

Answers to Max's questions on `NARRATIVE_v2.md`. Each section starts with a short plain answer, then gives the numbers and how they were made.

**Units.** "Edge" means accuracy minus 50, in points. On the V2 test window (72,474 games, 2026-03-29 to 2026-05-22):
- draft only: 57.19%, an edge of 7.19;
- draft plus players (main model on the full_1017 panel, `P3_TAILS.md`): 62.00%, an edge of 12.00.

So knowing the players adds 4.81 points, the "personal gain". These are the current panel numbers. The definitive rerun (about Oct 28) will move them a little.

**Sections 12 and 13** (added 2026-10-21): party structure, and Korea's sample. Main box only.

**Section 11** (added 2026-10-21): checks of the carry-forward numbers (simulation, matchmaking, recency, definitions, specialists, variance decomposition).

**Sections 7 to 10** (added 2026-10-19 for REMAINING_WORK.md items 6, 7, 9 and 11): drift on one horizon, the player-model ladder, normalization in edge points, and Korea.

**New code** (all in `training/personalization/`, run on this box at nice 19 on cores 48-51; no GPU, no Neon, no 2.57 or sealed games):
- `p2f_game_clock.py`: the game's clock;
- `p2f_player_clock.py`: the players' clock;
- `p2f_clocks.py`: both clocks with one measurement-error estimator, the primary result of section 2;
- `p2f_arms.py`: shrinkage, regions, top tiers and specialists, from the scored arms of the full-history panel. Those arms and the full slot table were copied from the 3090. No job ran there.

Outputs are in `results/p2f/`: `clocks.json`, `game_clock.json`, `player_clock_history.json`, `player_clock_slots.json` and `arms.json`.

Two corrections to the narrative come out of this work. Both are in section 2:
1. The "410 days" label is wrong. The drift paper's 16-build lag counts sizable builds only, and those builds are spread out: 16 of them span a median of 954 days, about 2.6 years.
2. When both clocks are measured the same way, the game is not faster than the players. On heavily played heroes, players' relative skill moves somewhat faster over one to two years.

---

## 1. Drift in edge-over-chance units

**Plain answer.** Yes, drift matters, but the two halves differ in why.
- **Draft model.** Never updating it for 2.8 years costs about 1 point of a 6.9-point edge, roughly a seventh. Any retraining schedule up to every six builds (about every six months) wins almost all of that back.
- **Player model.** Freezing it costs more, and sooner. Four months frozen loses about 1.6 points, which is 59% of what the players add in accuracy (70% in log loss). That is more than the draft model loses in 2.8 years.

The player loss is mostly not skill changing. It comes from new games, new heroes and new accounts that a frozen model cannot see. Keeping game counts current recovers most of it.

### Draft side (maintenance replay, drift paper Table 3; `training/drift_v2/results/R9_REPLAY.md`)

The reference is "retrain every build", which averages 56.87% over the 2.8-year deployment, weighting each build by its games. That is an edge of 6.87.

| policy | accuracy cost vs reference (points) | share of the 6.87 edge | share of the 4.81 personal gain |
|---|---|---|---|
| never retrain, never refresh | 1.02 | 14.9% | 21% |
| refresh statistics every build, never retrain | 0.66 (recovers 0.36, 35% of the loss) | 9.6% (recovers 5.2%) | 14% |
| retrain every 3 builds, refresh between | 0.00 (recovers 1.02, all of it) | 0% | 0% |
| retrain every 3 builds, no refresh between | 0.03 (recovers 0.99, 97%) | 0.4% | 0.6% |
| retrain every 6 builds, refresh between | 0.08 | 1.2% | 1.7% |
| retrain every 9 / 12 builds, refresh between | 0.17 / 0.20 | 2.5% / 2.9% | |

- On the deployed V2 unit the same 1.02 points is 14% of the 7.19 draft edge and 8.5% of the 12.00 total edge.
- On the main split, freezing statistics for one quarter costs 0.035 points, 0.5% of the edge.
- The 1.02 is an average over the 2.8 years, not the loss at the end. The loss grows with time.

### Player side

These numbers come from the phase-1 player model (win-only CF), where the frozen and lagged arms were run. Sources: `P3_HERO_STRENGTH.md` section 2 and `P3_EXTENSIONS.md` section 1.

V2 window: draft only 57.19%; phase-1 player model 60.02%, an edge of 10.02 and a personal gain of 2.83 points. Log-loss gain 0.0142.

| arm | accuracy | points lost | share of personal gain lost (accuracy / log loss) | share of the 10.02 edge |
|---|---|---|---|---|
| updated nightly (reference) | 60.02% | | | |
| state and counts 7 days old | 59.48% | 0.54 | 19% / 21% | 5% |
| state and counts 30 days old | 58.92% | 1.10 | 39% / 42% | 11% |
| frozen at the Feb 10 cutoff (V2 is 7 to 15 weeks later) | 58.39% | 1.63 | 58% / 58% | 16% |
| skill frozen at cutoff, game counts current | 59.32% | 0.70 | 25% / 25% | 7% |

Out-of-time window, June to September 2026 (286,016 games, four months after the May snapshot): draft only 56.94%, nightly model 59.70% (edge 9.70, personal gain 2.76, log-loss gain 0.0149).

| arm | accuracy | points lost | share of personal gain lost (accuracy / log loss) | share of the 9.70 edge |
|---|---|---|---|---|
| frozen in May, state and counts | 58.06% | 1.64 | 59% / 70% | 17% |
| frozen in May, counts current | 59.10% | 0.60 | 22% / 28% | 6% |

By month, the frozen model keeps this share of the log-loss gain:

| period | share kept |
|---|---|
| first five weeks | 49% |
| July | 29% |
| August | 16% |
| September | 23% |

**The main model.** It has a larger personal gain (4.81 points), and its lagged and frozen arms have not been run. If it lost the same shares, the costs would be about:
- 1 point at a 7-day lag;
- 2 points at a 30-day lag;
- 2.8 to 3.4 points frozen for four months.

This is an extrapolation, labeled as such. A direct run needs a `--lag` and a freeze option in `p3_rcf.py online`, which is about an hour on the 3090.

### Does drift matter?

| what goes stale | cost | relative to the edge | relative to the personal gain |
|---|---|---|---|
| draft model, never maintained, 2.8 years | 1.02 points | 15% of 6.87 | 21% of 4.81 |
| draft statistics frozen one quarter (model retrained) | 0.035 points | 0.5% | 1% |
| player model frozen four months | 1.64 points | 17% of 9.70 | 59% (70% in log loss) |
| player model one week behind | 0.54 points | 5% | 19% |

The cheap fixes are different for each half:
- **Draft side:** retrain a few times a year. Nightly refresh adds little.
- **Player side:** keep the game counts and skill state current every day. This is bookkeeping, not refitting. The May hyperparameters still worked in September.

---

## 2. The two clocks, at 3 months, 6 months, 1 year and 2 years

**Plain answer.** Measured the same way, in the same units, on the same games, the game does not drift faster than the players.
- **A hero's win rate in a tier** keeps a true correlation with itself of 0.93 after 3 months, 0.90 after 6 months, 0.85 after 1 year and 0.77 after 2 years. This does not depend on any cutoff. Heroes the patch notes left untouched keep 0.91 to 0.95; heroes changed twice or more keep 0.75 to 0.80.
- **A player's skill on a hero, relative to their own average,** keeps 0.91, 0.85, 0.90 and 0.86 when every player-hero cell with play in both windows is used. The confidence intervals are wide (about ±0.15), and the value depends on the cutoff. Restricted to heroes played 20+ times in each 90-day window, it is 0.83, 0.79, 0.74 and 0.57.

So within the uncertainty the two clocks overlap. On heroes a player plays a lot, the players' clock looks *faster* than the game's over one to two years. The narrative's "game much faster than players" is not supported and should be dropped or reversed.

Two things made the old contrast (0.62 against 0.90 to 0.93) look large:
- **The game number was a 2.6-year lag, not 410 days.** The drift paper's lag of 16 *sizable* builds spans 765 to 1,097 days (median 954).
- **The player number came from the drift model's fitted parameters, not from measured records.**

### Method in plain words (`p2f_clocks.py`)

- **One table, one estimator for both sides.** All 22.4M player slots of every ranked game from December 2021 to September 2026, each with the draft-only win probability (WP) of its game.
- **Independent windows with new games only.**
  - Window A is 90 days. Window B is 90 days and starts h days after A *ends* (h = 3 months, 6 months, 1 year, 2 years).
  - Every cell is estimated twice, once from the games inside A and once from the games inside B. Nothing carries over.
  - A player who does not play a hero in B has no B value for it. They cannot "trivially not change"; they enter only the selection check below.
  - A new A starts every 30 days, and all window pairs are pooled (50 pairs at 3 months, 29 at 2 years).
- **Same unit: points of win probability.**
  - **Game cell:** hero × tier group (Bronze-Silver, Gold-Platinum, Diamond-Master). The value is the hero's win rate minus 50, in points.
  - **Player cell:** player × hero. The value is the player's average result on the hero minus the draft-only WP, in points. That is how much better or worse than the draft predicts they do. Primary is the *hero-specific* version: the same minus the player's own average over all heroes in that window. The raw cell also contains the player's overall level, which matchmaking keeps pulling back toward 50% and which MMR already tracks; it is reported but cannot be read as skill drift (see the flags).
- **Measurement-error model instead of cutoffs.**
  - Each cell's window value = its true value + luck. The luck has a known variance:
    - game cells: binomial, p(1 - p)/n;
    - player cells: the spread of that cell's own game residuals divided by n ("residual-based"). This needs 2+ games in the window.
  - Pooling all cells, weighted by how much each tells us (inverse variance), we estimate the true variance in A, the true variance in B, and the true covariance between them. Luck in A and luck in B are independent, so only the variances need correcting.
  - Stability = true covariance / sqrt(true variance A × true variance B): the correlation of the true values, also called the disattenuated correlation.
  - 95% CIs come from 200 bootstrap draws over heroes (game side) or over players (player side, 200 player groups).
- **Sensitivity curves** repeat the estimate with a minimum of 5, 10, 20, 50 and 100 games per cell in each window, and on the game side with a minimum pick rate from 0.25% to 5%. If the noise model is right, the value should stay roughly flat.

### Primary table

True-effect correlation (95% CI). Every cell with play in both windows: game cells with any games, player cells with 2+ games.

| horizon (gap between windows) | game: hero × tier | player: hero-specific | player: hero-specific, 20+ games in each window |
|---|---|---|---|
| 3 months | **0.93** (0.90, 0.95) | **0.91** (0.80, 1.04) | 0.83 (0.76, 0.93) |
| 6 months | **0.90** (0.87, 0.93) | **0.85** (0.75, 0.98) | 0.79 (0.68, 0.92) |
| 1 year | **0.85** (0.79, 0.89) | **0.90** (0.78, 1.08) | 0.74 (0.61, 0.90) |
| 2 years | **0.77** (0.69, 0.83) | **0.86** (0.67, 1.11) | 0.57 (0.38, 0.84) |
| cell pairs, 3 months / 2 years | 13,500 / 7,830 | 5.3M / 1.7M | |

Spread of the true values, in points of win probability:

| horizon | game: true sd in A / B | game: part carried to B | player: true sd in A / B | player: part carried to B |
|---|---|---|---|---|
| 3 months | 2.4 / 2.4 | 2.3 | 3.3 / 3.5 | 3.3 |
| 2 years | 2.5 / 2.3 | 2.1 | 3.0 / 3.3 | 2.9 |

The "part carried to B" is sqrt(true covariance). A player's edge on a hero, relative to their own average, is a somewhat larger effect than a hero's tier win-rate edge, and in absolute points about as much of it carries forward.

### Sensitivity curves

True-effect correlation by the minimum games per cell in each window:

| side, horizon | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| game, 3 months | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 |
| game, 1 year | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| game, 2 years | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 |
| player hero-specific, 3 months | | 0.91 | 0.90 | 0.88 | 0.83 | 0.89 | 1.10 (CI 0.79 to 2.7) |
| player hero-specific, 6 months | | 0.85 | 0.86 | 0.80 | 0.79 | 0.81 | 0.92 (very wide) |
| player hero-specific, 1 year | | 0.90 | 0.85 | 0.80 | 0.74 | 0.66 (CI 0.50 to 1.07) | too few |
| player hero-specific, 2 years | | 0.86 | 0.63 | 0.62 | 0.57 | 0.65 (very wide) | too few |

Game side by minimum pick rate (share of the tier group's games, in both windows):

| horizon | 0.25% | 0.5% | 1% | 2% | 5% |
|---|---|---|---|---|---|
| 3 months | 0.93 | 0.93 | 0.93 | 0.92 | 0.92 |
| 1 year | 0.85 | 0.85 | 0.85 | 0.85 | 0.82 |
| 2 years | 0.77 | 0.77 | 0.77 | 0.76 | 0.71 |

**Flags.**
1. **The game side is flat.** Every hero × tier cell has hundreds of games a quarter, so the binomial noise model holds and no cutoff matters. Rarely picked heroes do not drag it down.
2. **The player side is not flat at 1 and 2 years.** It falls from 0.90 to 0.74 at 1 year and from 0.86 to 0.57 at 2 years as the minimum rises from 2 to 20 games. The noise model is an unlikely cause. A single-window check, by games per cell, shows the residual-based noise at 2 games is if anything slightly *too small*, which would pull the all-cells value down, not up. The more likely reading is selection: the cells with 20+ games are heavy players' main heroes, and their relative skill on those heroes moves more over one to two years than a light player's skill on a hero. The defensible player number for meaningfully played heroes is therefore **0.74 at 1 year and 0.57 at 2 years**; the all-cells value describes the average cell, most of which are light. Our corpus sees only part of each player's games, so cells with 50+ and 100+ games are too few at long horizons.
3. **The WP-based noise model fails on the player side.** Using the WP's own p(1 - p) as luck gives true correlations of 2 to 10 at low cutoffs. A single-game check shows why: one game's residual varies 2.5% less than the WP says. That error is 60 points² per game, about four times a player-hero cell's whole true variance. We therefore use the cells' own residual spread as their noise, and need 2+ games per window.
4. **The raw player cell (overall level included) comes out at or above 1** at most cutoffs with either noise model, so it is not a usable stability measure. Matchmaking moves a player's rating after every game, which ties consecutive results together within a window. Overall level is MMR's job and is out of scope here.

### Selection checks

**Player side: who keeps playing?** Only a minority of player-hero cells with 2+ games in A also have 2+ games in B:

| horizon | share of A cells continuing | A value of continuers / stoppers (hero-specific, points) | true sd in A, continuers / all | correlation, both windows active (equal weights) | intent-to-treat bound |
|---|---|---|---|---|---|
| 3 months | 21% | +1.16 / +0.31 | 3.4 / 3.1 | 1.01 | 0.24 |
| 6 months | 19% | +1.10 / +0.48 | 3.4 / 3.1 | 0.93 | 0.19 |
| 1 year | 15% | +1.15 / +0.66 | 3.4 / 3.0 | 1.06 | 0.18 |
| 2 years | 12% | +1.07 / +0.79 | 2.7 / 2.9 | 1.23 | 0.14 |

- Players keep the heroes they do well on: continuers were about +1.1 points over their own average in A, against +0.3 to +0.8 for those who stopped. That is selection on performance, mild in size.
- The intent-to-treat bound sets every stopper's B value to the population mean, as if nothing persisted for them. It is only a floor (0.14 to 0.24), because most cells simply stop, which says nothing about skill.
- The equal-weight values above 1 are the low-game cells again (flag 2); the weighted primary is the better estimate.

**Game side: reworked heroes.** Cells are split by how many Live patches published between the end of A and the end of B list the hero with a balance or rework section (`drift_v2/patch_notes_ground_truth.json`; bug-fix-only and ARAM-only mentions excluded):

| horizon | no listed change | changed once | changed 2+ times |
|---|---|---|---|
| 3 months | 0.95 (0.93, 0.97), 9,336 cell pairs | 0.92 (0.88, 0.96), 3,711 | 0.75 (0.61, 0.87), 453 |
| 6 months | 0.94 (0.91, 0.96), 7,230 | 0.90 (0.85, 0.94), 4,323 | 0.80 (0.68, 0.90), 1,137 |
| 1 year | 0.91 (0.86, 0.95), 3,792 | 0.86 (0.82, 0.90), 4,563 | 0.79 (0.66, 0.86), 2,715 |
| 2 years | 0.71 (0.53, 0.89), 405 | 0.82 (0.74, 0.90), 2,310 | 0.78 (0.71, 0.84), 5,115 |

Patches drive the game's clock: untouched heroes barely move within a year. By 2 years almost every hero has been touched, and the few untouched cells are noisy.

### Earlier, simpler measurements (kept for the record)

These used start-to-start horizons and fixed cutoffs; `p2f_game_clock.py` and `p2f_player_clock.py`, in `game_clock.json` and `player_clock_*.json`. They agree in direction:
- **Hero win rate, quarter windows, 2022 to 2026:** 0.97, 0.93, 0.87, 0.77. A 1% or 2% pick-rate floor changes nothing.
- **The drift paper's build pairs, grouped by real day gap:** 0.89, 0.83, 0.78, 0.72.
- **Hero-specific player skill from 42K full Heroes Profile histories** (split-half correction; residual against the hero's quarterly win rate, not the WP):
  - 20+ games in each quarter: 0.99, 0.96, 0.93, 0.72;
  - 50+ games: 0.98, 0.97, 0.87, 0.73;
  - any play: 1.02, 1.01, 0.91, 0.80.
- **Values implied by the drift model's fitted parameters** (hero-specific part): 0.98, 0.97, 0.94, 0.88. These are too high at 2 years.

**For the paper.**
- Say "both halves keep most of their information for about a year and lose a quarter to a half of it over two years; the game's changes come in patches".
- Do not claim the players' clock is slower.
- The upkeep conclusions still stand, for these reasons:
  - the draft side needs retraining a few times a year, because patches move hero strength;
  - the player side needs daily bookkeeping (new games, heroes and accounts);
  - player records are best read over the last year or two. Old history still helps (`P3_FULL_HISTORY_EARLY.md`), but it should not be treated as permanent.

---

## 3. Empirical justification for shrinkage

**Plain answer.** Take a player's record on a hero (how far above or below the draft's prediction they have done) and see how much of it shows up in their next games:
- about a quarter at 30 to 59 games;
- about 30% at 60 to 119;
- about 43% at 120 to 249;
- about 57% at 250+.

That is the shrinkage rule's shape. One curve, keep = n / (n + 180), fits all four bins, where n is games on the hero.

**Worked example, measured directly:**
- Records of +10 to +14 points (mean +11.7) over 70 to 90 games realize +3.8 points in the next game (SE 0.5). That keeps 32%, about a third.
- Over 200 to 260 games the same records (mean +11.6) realize +6.6 (SE 0.8). That keeps 57%, about 60%.
- Losing records behave the same way: −11.7 over 70 to 90 games realizes −3.5 (keeps 30%), and −11.5 over 200 to 260 realizes −6.9 (keeps 60%).

So "12 points over 80 games keeps about a third; over 230 games about 60%" holds as measured.

### Data and method (`p2f_arms.py`, arm A2S)

- **Slots:** every slot of V2 and the out-of-time window (4.3M slots). Each player's record comes from their full history in the full_1017 store, the merged corpus plus Heroes Profile rows, through the day before.
- **Record x:** the mean residual on the hero to date. That is win minus WP on our games, where WP sees the draft and the lobby's average MMR, and win minus the hero's quarterly win rate on Heroes Profile rows.
- **n:** games on the hero in that history.
- **Outcome:** the scored game's residual, win minus WP, which is adjusted for the lobby's MMR.
- **Keep:** the slope of outcome on record within each n bin (ordinary least squares). SEs come from a jackknife over 20 player groups.
- **Formula:** keep = τ²/(τ² + σ²/n), with σ² = 0.245, the variance of one game. τ² is measured in each bin as the variance of records minus their average luck variance. That uses no future games, so the formula's prediction is a fair test.

| games on hero | slots | median n | measured keep (SE) | formula, τ measured | formula, k = 150 | spread of records (sd) | real spread τ | persistent spread |
|---|---|---|---|---|---|---|---|---|
| 30-59 | 402,186 | 42 | 0.24 (0.01) | 0.31 | 0.22 | 9.3 pts | 5.1 pts | 4.6 pts |
| 60-119 | 345,652 | 83 | 0.30 (0.01) | 0.38 | 0.36 | 6.9 | 4.3 | 3.8 |
| 120-249 | 273,574 | 167 | 0.43 (0.02) | 0.51 | 0.53 | 5.5 | 3.9 | 3.6 |
| 250+ | 303,380 | 466 | 0.57 (0.02) | 0.68 | 0.77 | 3.9 | 3.2 | 3.0 |

- **Persistent spread** is the square root of the covariance between record and next-game outcome: the part of a player-hero difference that carries forward.
- The measured keep rises with n the way the formula says. It sits at about 0.8 times the formula with the measured τ, because part of a record's real spread does not carry into the future: drift, and old Heroes Profile rows scored against a coarser baseline.
- The best single k is 180, which gives 31% at 80 games and 56% at 230.
- The production prior (k about 150) is close. It trusts very deep records slightly too much, which matches the tail finding in `P3_TAILS.md`.

---

## 4. "How they played": scoreboard reliability

**Plain answer.** One game's win or loss says very little about a player's level on a hero. Only 1.3% of its variance is a stable player-on-hero effect, so it takes about 74 games before half of a record is real. One game's damage, healing, deaths or XP share carries 3 to 30 times as much stable signal, and reaches the same point in 2 to 6 games for most metrics.

The model learns which of these stable traits go with winning. Using them is the single biggest step in the player model: +8 log-loss points (×1000) over win/loss alone.

**Tennis analogy.** A match is one win or loss, but the share of points won and break points saved in that match tell you far more about how well each player played. Scoreboards are our points-won count.

| per-game measure | stable share of one game, player × hero | times the win/loss | games until half is real |
|---|---|---|---|
| win or loss, beyond the draft | 0.013 | 1 | 74 |
| protection given to allies (17 heroes) | 0.40 | 30 | 1.5 |
| self-healing | 0.39 | 29 | 1.6 |
| damage taken | 0.28 | 21 | 2.6 |
| siege damage | 0.28 | 21 | 2.6 |
| hero damage | 0.23 | 17 | 3.4 |
| healing (35 heroes) | 0.23 | 17 | 3.4 |
| XP contribution | 0.21 | 16 | 3.8 |
| regen globes | 0.20 | 15 | 4.1 |
| deaths | 0.17 | 13 | 4.8 |
| merc captures | 0.16 | 12 | 5.2 |
| structure damage | 0.14 | 10 | 6.4 |
| outnumbered deaths | 0.09 | 7 | 9.9 |
| kills | 0.07 | 5 | 14 |
| escapes | 0.06 | 4 | 17 |
| assists | 0.04 | 3 | 23 |

How the table was made:
- **Stable share** is split-half reliability on estimation-window games (`results/rcf/reliability.json`, 9.4M rows).
- **Games until half is real** is (1 - r)/r, the Spearman-Brown rule.
- **Caveat for the text.** Reliability is how consistent a metric is, not how much it matters for winning. Most of the stable variation in scoreboards is play style. The model's win-relevant direction loads mostly on hero damage, self-healing and low damage taken (`P3_RICH_CF.md`, "Supported dimensionality"). So the honest sentence is "a single game's scoreboard pins down a player's stable traits far faster than wins do, and the model learns which traits win". "Tells us 3 to 30 times more about skill" overstates it.

### How the per-game scoreboard stats are normalized

This is `p3_rcf_data.py`, the `a_shrink` card used by the main model. Each step has a reason.

1. **Per minute.** rate = value / game minutes. Long games inflate every total; a 30-minute game has more damage than a 15-minute one for the same player.
2. **A log-like transform.** t = asinh(rate / s), where s is the median positive rate on estimation rows. Scoreboard numbers are heavily skewed (a few huge damage games). The transform compresses the tail like a log but keeps zeros, for example a game with no healing.
3. **Standardize within hero × map × game-length band × outcome.** z = (t - cell mean) / cell sd, with five length bands (under 14, 14-17, 17-20, 20-24, 24+ minutes). Each part removes something that is not the player:
   - **hero:** healers heal and tanks take damage;
   - **map:** maps differ in fights, objectives and lanes;
   - **length band:** per-minute rates still shift with game length, because late games are teamfight-heavy;
   - **outcome:** winners' stats look better because they won. Comparing a game only with games of the same result means z carries no information about its own game's result. That is what makes it safe to combine with win/loss.

   Cell means and sds come from estimation-window rows only. Thin hero × map cells are shrunk toward the hero's all-map cell with 100 pseudo-games, so a rarely played hero-map pair does not get a noisy baseline. This replaced a hard fallback at 50 rows.
4. **Clip at ±5**, so one broken game (a disconnect or an AFK teammate) cannot dominate.
5. **Quarter-by-hero adjustment.** Subtract the (hero, calendar quarter) mean of estimation rows; later quarters use the last estimation quarter. Balance patches change what a hero's numbers look like (a damage buff raises everyone's damage), and this keeps a patch from looking like a change in the player.
6. **Mask uninformative metrics per hero:** fewer than 5% nonzero rows, or a cell sd of 0.02 or less, for example healing on a hero with no heals.

Only games on earlier days enter a prediction. Leak tests R1 to R7 in `tests/test_rcf.py` pass.

**The choices were tested** (`P3_RICH_CF.md`, "Scoreboard cards"). Each alternative was refit and scored against the base normalization on V1 (choice), V2 and the out-of-time window (test), cold start and coverage. Lift is log-loss ×1000.

| card | change | V2 / OOT lift vs base | cold-start R² (base 0.24) | verdict |
|---|---|---|---|---|
| hero × map shrinkage (`a_shrink`) | shrink thin cells toward the hero | +0.07 / +0.05 | **0.35** | adopted: same lift, much better cold start |
| tier cells | hero × map × 5 tier groups, shrunk | −0.45 / −0.48 | 0.31 | worse: the outcome-conditioned cell already absorbs tier, and tier cells are thin |
| smooth length curves | expected value by length within hero | −0.24 / −0.37 | 0.32 | worse: per-minute rates in five bands are enough |
| team share | stat as a share of the team total | −0.44 / −0.76 | 0.32 | worse: removes the player's absolute level, the part that carries to the next game |
| opponent difference | minus same-role opponents | −1.42 / −1.34 | 0.27 | clearly worse, for the same reason |
| ranks | normal scores of the rank within the cell | +0.32 / +0.27 | 0.17 | small lift, worse cold start; not adopted |
| winsorize at ±2.5 | tighter clip | +0.18 / +0.17 | 0.29 | small; not adopted |
| informativeness weighting | weight games by kill margin | +0.13 / −0.10 | 0.20 | weights came out flat (0.98 to 1.01); not justified |

### The draft model's hero statistics

The draft model's win-rate features (hero, hero on map, hero pairs with and against, role compositions) are computed per skill tier from our own corpus (`drift2026/build_patch_stats.py`).
- **Thresholds:** a hero needs 20 games in a tier, otherwise it gets 50%. A hero-on-map rate is used only with 50+ games, otherwise the hero's tier rate. Pairs need 30+ games.
- **Leak-free.** The WP behind every residual in P3 (`d2c_cumprev`) uses statistics accumulated through each game's previous build, so no game's own build, and hence never the game itself, enters its features. When statistics frozen at a cutoff are given to training rows, they are computed out of fold: five hash folds, with each row's own fold removed. Without that, the old leaky recipe inflated validation accuracy by 1.7 points and wrecked calibration (drift paper, Section 2).
- **Tiers.** The P3 WP uses the snapshot's original tier grouping, which was later found mislabeled. Rebuilding the features with corrected tiers changed future accuracy by under 0.05 points (`drift_rebuild/results/r14_tierfix.json`: 57.11% corrected against 57.09%, three seeds).

So both sides are normalized the same way in spirit: by tier and map, against a baseline built only from earlier games.

---

## 5. Regions

**Plain answer.** The draft alone is equally predictive in the Americas and Europe (57.4% and 57.1% on V2), and the players add the same amount in both (+4.9 and +4.8 points). Korea is a small slice (3 to 5% of games). It gains less from players (+4.2 on V2, +3.3 out of time), most likely because the panel holds few deep Korean histories. Hero win rates drift at the same pace in all three regions. China is not in the data.

Main model (`full_ms_layer`) and win-only full-history model (A2S), by the game's region:

| window | region | games | draft only | + players, main | personal gain | log-loss gain ×1000, main | + players, win-only |
|---|---|---|---|---|---|---|---|
| V2 | Americas | 22,332 | 57.42% | 62.31% | +4.89 | 31.0 | 60.36% |
| V2 | Europe | 46,820 | 57.12% | 61.94% | +4.82 | 28.0 | 60.41% |
| V2 | Korea | 3,322 | 56.56% (SE 0.9) | 60.75% | +4.18 | 22.1 | 59.39% |
| OOT | Americas | 91,555 | 56.82% | 61.80% | +4.98 | 29.7 | 60.10% |
| OOT | Europe | 180,669 | 56.97% | 61.65% | +4.69 | 27.7 | 60.25% |
| OOT | Korea | 13,792 | 57.48% (SE 0.4) | 60.79% | +3.31 | 16.1 | 59.01% |

**Drift by region.** These are quarter windows, noise-corrected, 2024-04 to 2026-05 only, because region is known only from 2024 on.

| region | games | 3 months | 6 months | 1 year |
|---|---|---|---|---|
| Americas | 357,900 | 0.93 | 0.85 | 0.73 |
| Europe | 692,099 | 0.94 | 0.87 | 0.76 |
| Korea (heroes with 30+ games) | 32,433 | 0.88 | 0.84 | 0.77 |

Draft-only accuracy from V2 to the out-of-time window falls 0.6 points in the Americas and 0.15 in Europe. In Korea it rises 0.9, which is within its noise. Per-region player-model staleness was not run.

---

## 6. The professional drafting angle

**Plain answer, kept to what the data show:**
- **Personal hero edges are a little larger at the top.** The part of a player-hero record that carries into the next game has a spread of 4.2 points in Diamond and Master, against 3.7 to 3.9 in Bronze to Gold. The model's per-slot estimates spread 3.6 points in Master against 3.1 in Bronze, and 3.8 in the top 1% of lobbies by MMR. Players add the most in Gold to Diamond (+5.2 to +6.0 points of accuracy) and still +4.9 in Master. In the top 1% of lobbies by MMR they add +7.0 points on V2 (696 games) and +6.6 out of time (2,689 games).
- **The lift does not come from one-trick specialists.** Slots where the player is on a hero with 100+ games that makes up 20%+ of their play are 6 to 7% of slots. Those players realize about +3 points on that hero, a real but modest edge, and they account for only 5 to 6% of the main model's gain (8 to 11% of the win-only model's). The top 1% and 5% of slots by games on the hero carry about 1% and 4 to 5% of the gain.
- **What carries the lift is a minority of large personal signals, mostly negative.**
  - The 1% of slots with the largest personal estimates carry 14 to 15% of the gain. The top 5% carry 38%, and the top 20% carry 73 to 75%.
  - About 60% of those largest signals are negative: a player on a hero they have barely played (median 1 to 2 games). In the top 1% such slots realize −11.4 points, exactly as predicted.
  - The large positive signals realize even more than predicted (+17.7 against +12.2), likely strong players on new or thin accounts.
- **For scouting:** what pros say matters, knowing an opponent's pool, does show up. It shows up mainly as knowing when someone is off their pool, and in bans. Banning a player's main costs that player about 1.8 points in that game (`P3_TAILS.md` section 3). One player's best hero is worth about +3 points to their team.

### By real league tier

Real tier = stored league tier minus 1; NULL = Master. Grandmaster is inside Master. The "top 1% lobby MMR" row stands in for it.

| tier | V2 games | draft only | + players (main) | gain, points | log-loss gain ×1000 (V2 / OOT) | persistent spread of player-hero records (30+ games) | keep at median n | sd of slot estimates | sd of team difference |
|---|---|---|---|---|---|---|---|---|---|
| Bronze | 13,140 | 58.00% | 61.17% | +3.2 | 19.1 / 21.1 | 3.85 pts | 0.31 (n 83) | 3.07 pts | 9.8 pts |
| Silver | 15,489 | 57.33% | 61.55% | +4.2 | 24.1 / 22.8 | 3.66 | 0.29 (n 88) | 3.14 | 10.1 |
| Gold | 14,771 | 56.97% | 62.85% | +5.9 | 33.5 / 30.2 | 3.75 | 0.30 (n 94) | 3.28 | 10.8 |
| Platinum | 13,409 | 56.90% | 62.11% | +5.2 | 34.9 / 34.2 | 3.99 | 0.34 (n 105) | 3.48 | 11.5 |
| Diamond | 9,188 | 57.15% | 62.98% | +5.8 | 34.9 / 31.7 | 4.19 | 0.36 (n 116) | 3.59 | 12.0 |
| Master | 6,477 | 56.32% | 61.19% | +4.9 | 26.0 / 29.3 | 4.20 | 0.34 (n 110) | 3.64 | 11.4 |
| top 1% lobby MMR (≥ 3,104) | 696 | 56.03% | 63.07% | +7.0 | 37.1 / 26.5 | | | 3.82 | 12.2 |

Persistent spread = sqrt(cov(record, next-game outcome)), from the A2S full-history slots of V2 and OOT.

### Where the gain comes from

V2 shown; OOT within a point. The counterfactual keeps the model's personal estimate only on the chosen slots, sets the rest to the population expectation, and rescores with the same combiner.

| slots kept | share of slots | share of games touched | share of the main model's gain | realized residual on those slots |
|---|---|---|---|---|
| specialists: 100+ games on the hero and 20%+ of the player's games | 6.9% | 50% | 5.9% | +3.0 pts |
| top 1% by games on the hero (≥ 1,267) | 1% | 9% | 1.1% | +2.6 |
| top 5% by games on the hero (≥ 419) | 5% | 38% | 4.8% | +2.8 |
| top 1% by share of the player's games (≥ 66%) | 1% | 10% | 0.7% | +2.4 |
| top 5% by share of the player's games (≥ 31%) | 5% | 39% | 4.7% | +2.9 |
| top 1% by size of the estimate (≥ 9.9 pts) | 1% | 9% | 13.8% | 59% negative: −11.4 realized |
| top 5% by size of the estimate (≥ 6.9 pts) | 5% | 40% | 37.7% | 65% negative: −8.4 realized |
| top 20% by size of the estimate (≥ 4.2 pts) | 20% | 89% | 74.6% | |

What we cannot say from this data:
- **Professional play.** Pro play is a different setting: scheduled opponents, coordinated teams and prepared strategies. These are ranked solo-queue games.
- **Pre-scouted drafts.** We do not observe drafts made with scouting, so we cannot say what scouting is worth beyond what our model already infers from history.

---

## 7. Drift on one horizon: a year without updates

**Plain answer.** Leave each half alone for a year and score it on the same games:
- **Draft model a year stale:** the draft alone drops from 56.8% to 55.6%. That is 1.2 of its 6.8 points of edge (18%). With the players included, the full model loses only 0.45 points, because the player terms make up for part of it.
- **Player state frozen a year:** the full model drops from 61.5% to 58.4%. That is 3.2 points, two thirds of the 4.7 points the players add. In log loss it keeps only 13% of the player gain.
- **Both a year stale:** 57.5%, a loss of 4.0 of the 11.5-point edge (35%).

So on one clock, the player side goes stale much faster than the draft side. The decomposition below shows why: about 80% of the player-side loss is players and heroes the year-old state had never seen. Real changes in known players' skill cost almost nothing at one year.

**Method.** This reuses the tournament agent's fixed vintages and models (`P3_TOURNAMENT.md`, "Vintages for drift"; `p3t_value.py predeval`). It adds the one combination that run did not score: current draft model with year-old player state. That cell was computed on the 3090 CPU from the tournament's saved state files; nothing was retrained.
- **Games:** 60,500 real ranked games from 2026-05-23 to 06-30. The players are tournament-eligible: all ten have 50+ earlier games. That makes the personal gain a little larger than on V2.
- **Draft side:** "current" is the population model and statistics through the May 22 snapshot. "A year stale" (`s1y`) is the same recipe through build 2.55.10.94189 (2025-06-02), about 12 to 13 months before these games.
- **Player side:** player state is the main skill model's state through the day before ("current"), or frozen at 2025-06-02.
- **Combiner:** each draft-model vintage has its own combiner, fit on the 3 months before its cutoff. This mimics a model frozen at that date.

Accuracy, with the edge in points in brackets:

| draft model \ player state | none | current | 4 months old | 1 year old |
|---|---|---|---|---|
| current | 56.84% (6.84) | **61.51% (11.51)** | 58.87% (8.87) | 58.36% (8.36) |
| 4 months old | 56.34% (6.34) | 61.54% (11.54) | 58.54% (8.54) | 57.98% (7.98) |
| 1 year old | 55.62% (5.62) | 61.06% (11.06) | 58.14% (8.14) | 57.51% (7.51) |

Log-loss gain over a coin flip (×1000):

| draft model \ player state | none | current | 1 year old |
|---|---|---|---|
| current | 14.4 | 43.0 | 18.2 |
| 1 year old | 10.3 | 38.8 | 12.9 |

What is kept after a year without updates:

| part | accuracy edge kept | log-loss gain kept |
|---|---|---|
| draft model alone | 5.62 of 6.84 points (82%) | 72% |
| player gain, with the current draft model | 1.52 of 4.67 points (33%) | 13% |
| whole model, both a year stale | 7.51 of 11.51 points (65%) | 30% |

Notes:
- The "about 72%" and "about 30%" in the coordinator's note are the log-loss figures in the first and last rows.
- The fair player-side figure, holding the draft model current, is 33% in accuracy and 13% in log loss.
- These are one vintage each, scored on one window, and the eligible players are heavier than average.
- The drift paper's replay (section 1) averages over 2.8 years of builds; this table is one snapshot one year out. They agree on size: about a point of draft edge per year without retraining.

### Why the player side loses more: a decomposition

**Plain answer.** The player model's year-old state loses mostly because it does not know who is playing, not because known skills changed.
- A quarter of the players in these games (27% of slots) had no games at all in our data a year earlier. Bringing only them up to date recovers 2.2 of the 3.2 points (about 70%).
- Heroes players newly picked up (13% of slots) are worth another 0.4 points.
- Updated game counts on already-known cells are worth 0.15 to 0.25.
- Updated estimates on cells we already knew are worth 0.3 to 0.5, and nearly all of that is on cells with under 10 games a year earlier. On cells with 50 or more games a year earlier, a year of new results changes accuracy by −0.07 to +0.01 points: nothing measurable.

The draft model has no such coverage gap:
- all 90 heroes were known a year earlier;
- every hero × tier cell already had at least 3,779 games, with a median of 37,919;
- so its 1.2-point loss is all real change in quantities it already knew.

**For the paper:** "The correlations measure how stable an effect is once it is known. On that score the two halves are similar. The accuracy lost by a stale model mostly measures coverage. The draft model's units are 90 heroes that never change and are measured on tens of thousands of games, so everything it loses is genuine drift. The player model's units are player-hero pairs, and those turn over: a quarter of the players in a typical game were not in the data a year earlier, and others have picked up new heroes. A year-old player model loses about two thirds of its value, and about 80% of that loss is people and heroes it has never seen. Real changes in known players' skill cost almost nothing at one year."

**Method** (`p2f_stale_decomp.py`, 3090 CPU, hotsjob `p2f_stale_decomp`):
- **Data:** the same 60,500 games as above, the current draft model and its combiner, and the tournament's saved player states (year-old and current).
- **Swaps:** each slot's played-hero value is swapped from year-old to current in steps.
- **Categories,** taken from the year-old state's history (corpus plus Heroes Profile rows):
  - (a) the player had no games at all;
  - (b) the player had history but under 10 games on this hero, and 10 or more by the game;
  - otherwise "established".
- **For established cells,** the value is split into:
  - (d) the experience offset, which depends only on game counts and is recomputed at the current counts, with the off-role flag updated;
  - (c) the rest: the skill estimate and layer, i.e. new evidence on known cells, which includes any real drift.
- **Order:** both orders are reported, since the order shifts credit a little.

| step (cumulative) | share of slots | accuracy | gain, points: forward order | gain, points: reverse order |
|---|---|---|---|---|
| all year-old (current draft model) | | 58.36% | | |
| (a) players new to us | 26.9% | | **+2.20** | **+2.16** |
| (b) heroes newly picked up | 12.7% | | +0.43 | +0.38 |
| (d) established cells: counts only | 60.4% | | +0.25 | +0.15 |
| (c) established cells: estimates | 60.4% | | +0.27 | +0.47 |
| all current | | 61.51% | total +3.15 | total +3.15 |

Step (c), forward order, split by the cell's games a year earlier:

| games on the hero a year earlier | share of slots | gain, points | correlation of year-old vs current estimate | median new games on the hero |
|---|---|---|---|---|
| under 10, still under 10 | 24.8% | +0.29 | | |
| 10-49 | 16.4% | +0.03 | 0.86 | 11 |
| 50-199 | 12.1% | +0.01 | 0.92 | 28 |
| 200+ | 7.1% | −0.07 | 0.96 | 101 |

In log loss (gain over the draft model, ×1000; current state 28.6, year-old 3.8):
- (a) is worth +14.5 (forward) or +15.0 (reverse);
- (b) +4.6 or +4.2;
- (d) +2.1 or +1.3;
- (c) +3.5 or +4.2.

The picture is the same.

**The same accounting for the draft side.** Year-old draft model: 55.62% against 56.84% current, 1.22 points.

| possible source | draft side | player side |
|---|---|---|
| units unknown to the stale model | 0 of 90 heroes; the pool has been fixed since December 2021 | 27% of slots are players with no history; 13% are newly picked-up heroes |
| sample size a year earlier | every hero × tier cell 3,779+ games, median 37,919 (median 11,390 in the last year alone; sampling sd about 0.5 points) | under 10 games on the hero for 64% of slots (including the 27% with no history) |
| real change in known units | all of the 1.22-point loss: hero win rates keep 0.85 correlation over a year (section 2) | about 0 on cells with 50+ games; at most 0.3 to 0.5 points over all known cells, mixed with sharper estimates |

How the draft side's own 1.22 points splits between stale statistics and stale weights is in the drift paper's replay: refreshing statistics alone recovers about a third of the loss, and retraining recovers the rest. All of it is change in known quantities.

Caveat: "new to us" means no games in our data before 2025-06-02. That covers new accounts, players who only appeared in our corpus later, and players whose full Heroes Profile history was not pulled. For a deployed system with full histories, part of (a) would be known. So (a) is partly a coverage problem of our data, not only of the world.

---

## 8. What each player-model step adds, in edge

**Plain answer.** The 7.2 to 12.0 gain builds up in four steps of real size:
- how many games each player has on each hero, with no results at all: +1.9 points;
- their actual win records, trusted in proportion to sample size: +0.8;
- borrowing from the player's other heroes: +0.5;
- per-game scoreboard performance: +1.4.

The per-player learning speed and drift layer adds the last +0.2. Learned hero traits are worth almost nothing in accuracy once the player's overall level is borrowed, though they help in log loss and on cold start.

**Method.** Every rung is a model scored on the same games, full histories (full_1017 panel), with its own combiner fit on V1:
- **draft only:** the draft-only WP;
- **experience:** the offset by games played (overall and on this hero), no outcomes;
- **win records, no borrowing:** each player-hero cell shrunk toward the offset on its own, with sample-size weighting (prior sd 3.9 points, the phase-1 no-pooling fit);
- **borrowing, overall level:** the player-plus-hero kernel;
- **learned hero traits:** the rank-2 CF kernel, arm A2S;
- **scoreboard:** the main model without its layer;
- **layer:** the main model.

The "experience" and "no borrowing" arms are new runs of `p3_c_tests.predict` with the A2S settings and only the kernel changed (`p2f_ladder.py`, 3090 CPU, about 3 minutes each). Step SEs are paired by game.

| step | V2 accuracy (edge) | step, points (SE) | OOT accuracy (edge) | step, points (SE) | V2 / OOT log-loss gain ×1000 |
|---|---|---|---|---|---|
| draft only | 57.19% (7.19) | | 56.94% (6.94) | | 0 / 0 |
| + experience (games played, no results) | 59.07% (9.07) | +1.88 (0.17) | 58.67% (8.67) | +1.73 (0.08) | 8.6 / 8.2 |
| + win records, sample-size weighted, no borrowing | 59.91% (9.91) | +0.84 (0.13) | 59.48% (9.48) | +0.81 (0.06) | 13.8 / 13.1 |
| + borrowing: the player's overall level | 60.36% (10.36) | +0.45 (0.11) | 59.99% (9.99) | +0.50 (0.06) | 17.0 / 16.7 |
| + borrowing: learned hero traits (win-only model) | 60.35% (10.35) | −0.01 (0.10) | 60.14% (10.14) | +0.16 (0.05) | 17.7 / 17.4 |
| + scoreboard performance | 61.76% (11.76) | +1.41 (0.14) | 61.50% (11.50) | +1.36 (0.07) | 27.5 / 26.5 |
| + per-player learning speed and drift layer | **62.00% (12.00)** | +0.24 (0.08) | **61.66% (11.66)** | +0.16 (0.04) | 28.6 / 27.8 |

Notes:
- **The experience step is large because the offset is large.** Veterans do about 7 points worse than the draft predicts on heroes they have never played, and 2 to 3 better on heroes they play a lot (`P3_HERO_STRENGTH.md` section 7). It uses no results, only what players chose to play.
- **The order matters for how credit is split.** Each step is measured with all earlier steps in place. Scoreboard first would take some credit from win records.
- **On the first panel (oct05) the scoreboard step was +1.4 to +1.6 points over the win-only model** (`P3_RICH_CF.md`), the same size.

**Cold start: share of real variation explained.** For heroes a player barely plays, with that hero's games hidden, the share of the true player-on-hero effect predicted from their other heroes (latent R², 19,264 cells, 2,320 held-out players, full-history panel unless noted):

| model | share explained |
|---|---|
| no borrowing (nothing to go on once the hero's own games are hidden) | 0 by construction |
| borrowing, win-only model (oct05 panel; `P3_RICH_CF.md`) | 0.38 |
| borrowing, main model with scoreboard | 0.30 |
| borrowing plus play preference (`imp2`) | 0.46 |

Borrowing is what makes any prediction on an unplayed hero possible. Play preference (which heroes a player chooses) is the best signal for it. The scoreboard sharpens a player's general level more than their hero-specific skill.

---

## 9. Normalization alternatives, in edge points

**Plain answer.** Every tested normalization choice moves accuracy by at most about 0.2 points, usually under 0.1. That is within noise on the spring test set (SE about 0.08) and at most 0.2 on the later set. The choice of normalization is not where the scoreboard's value comes from.

**Method.** The arms are the scored card arms of `P3_RICH_CF.md` (oct05 panel), each paired by game against the base card. The log-loss differences are those reported there.

| card | change | V2 accuracy diff, points (SE) | OOT accuracy diff, points (SE) | V2 / OOT log-loss diff ×1000 |
|---|---|---|---|---|
| base | the normalization of section 4 without shrinkage (61.60% / 61.35%) | 0 | 0 | 0 / 0 |
| **hero × map shrinkage (adopted)** | thin cells shrunk toward the hero | +0.06 (0.08) | −0.02 (0.04) | +0.07 / +0.05 |
| tier cells | hero × map × tier, shrunk | −0.04 (0.09) | −0.06 (0.05) | −0.45 / −0.48 |
| smooth length curves | | +0.08 (0.09) | −0.10 (0.05) | −0.24 / −0.37 |
| team share | | −0.05 (0.10) | −0.15 (0.05) | −0.44 / −0.76 |
| opponent difference | | −0.09 (0.10) | −0.21 (0.05) | −1.42 / −1.34 |
| ranks | | +0.06 (0.07) | 0.00 (0.03) | +0.32 / +0.27 |
| winsorize at ±2.5 | | +0.08 (0.06) | +0.01 (0.03) | +0.18 / +0.17 |
| informativeness weighting | | +0.10 (0.07) | −0.00 (0.04) | +0.13 / −0.10 |
| for scale: the first scoreboard export (`met2_hs2`, slot rows only) | | −0.43 (0.10) | −0.24 (0.05) | −1.80 / −1.61 |
| for scale: no scoreboard at all (win-only, `today`) | | −1.26 (0.15) | −1.40 (0.07) | −8.16 / −8.46 |

Accuracy moves less than log loss: on this panel about 0.15 points of accuracy per log-loss point (×1000).

---

## 10. Korea's smaller gain

**Plain answer.** Partly thin histories, but not only.
- Korean games have much shallower histories in our data: the median player in a Korean game has 58 earlier games, against 300 to 400 in the Americas and Europe. Only about 4 of the 10 players have pulled full histories, against 6 to 7.
- The main model's gain is 3.5 points in Korea against 5.0 (Americas) and 4.7 (Europe). Re-weighted to Korea's depth mix, the Americas and Europe gain 5.3 and 4.3: matching depth narrows the gap to Europe a little and does not narrow the gap to the Americas.
- Within the thinnest-history games (players' median under 100 games), Korea still gains less: 3.5 points against 5.3 and 4.2.
- So depth explains some of the gap for Europe and none for the Americas.
- The rest is unexplained with this data. Korea is small (17K test games), its player pool is small, and the draft model and statistics are dominated by Western games.

**Method.**
- **Games:** V2 and OOT pooled, main model (`full_ms_layer`).
- **History depth of a game:** the median, over its ten players, of games seen in the full history as of the day before.
- **Gain:** full model accuracy minus draft-only accuracy, on the same games.
- **Matched depth:** the Americas and Europe re-weighted to Korea's mix of depth bins.

| region | games | median depth | players with full history (of 10) | slots with 50+ games on the hero | gain, points | log-loss gain ×1000 | gain at Korea's depth mix | log-loss at Korea's mix |
|---|---|---|---|---|---|---|---|---|
| Americas | 113,887 | 396 | 6.6 | 3.2 | 4.97 | 30.0 | 5.25 | 30.9 |
| Europe | 227,489 | 315 | 6.3 | 2.8 | 4.71 | 27.8 | 4.34 | 26.0 |
| Korea | 17,114 | 58 | 3.9 | 1.1 | 3.48 | 17.2 | (3.48) | (17.2) |

By depth bin, gain in points (SE):

| median depth | Korea share | Korea | Americas | Europe |
|---|---|---|---|---|
| under 100 | 75% | 3.50 (0.45) | 5.31 (0.39) | 4.20 (0.22) |
| 100-299 | 22% | 3.28 (0.83) | 5.09 (0.31) | 4.71 (0.22) |
| 300-999 | 3% | 4.23 (2.22) | 4.97 (0.25) | 5.02 (0.19) |

**The win-only model (A2S) shows depth mattering more.**
- Korea gains 1.8 points.
- The Americas and Europe gain 3.2 to 3.3 overall, falling to 2.8 and 2.6 at Korea's depth mix.
- So about half of the win-only gap is depth. With the scoreboard added, depth matters less everywhere, and the remaining Korea gap is mostly not depth.

**For the paper:** "Korea gains less (3.5 against 4.7 to 5.0 points). Its players have far shorter histories in our data, which explains part of the gap. At equal history depth Korea still gains about a point less, which we leave open."

---

## 11. Are the carry-forward numbers right?

**Plain answer.** The numbers are right, but the headline should be reframed.

- **The low numbers are mostly noise in the record, not skill decay.** Section 3 found that 24%, 30%, 43% and 57% of a past record shows up later (by games-on-hero bin). We simulated players whose true hero skill never changes and ran the exact same estimator on them. It reports exactly how noisy their record is: between about 15% and 90% kept, depending on sample size and spread, with no real decay. Our estimator is therefore measuring noise in the record, as intended.
- **The noise alone predicts 30%, 38%, 50%, 62%, 70% and 79%** by games bin, rising to 800+ games on the hero. The measured values are 24%, 30%, 43%, 50%, 58% and 81%.
- **About 80 to 85% of the real skill difference persists.** That is measured over noise-only for 30 to 799 games, and about 100% at 800+.
- **Better framing for the paper:** "about four fifths of a player's true edge on a hero persists into their next games. A record is only partly skill: at 80 games, about 40% of what you see is real, so about a third shows up next time."
- **Matchmaking feedback does not explain it.** Adding both teams' at-game ratings changes the carry-forward by 0.01 or less in every bin.
- **Recency does not fix it.** Records of the last 25 to 100 games, or with a half-life under two years, predict worse than the full record. Recent records do carry slightly more real signal, but they are much noisier.
- **One-trick specialists are the one group that persists clearly less.** Heroes that are over half of a player's games keep only 56 to 66% of their true edge at 120 to 499 games. Heroes under 15% of a player's games keep 91 to 97%. The cause is not the visible ratings, though: adding them does not change it.
- **Max's Ana:**
  - **Record:** 237 games at 15.6 points above what the drafts predicted.
  - **Model expectation:** 9.4 points better than the draft predicts in his next Ana game; about 60% of the record kept, the noise-only expectation at that sample size. His other heroes average about −0.5 to +1.6 points.
  - **Plain terms:** in a draft the model calls 50-50, he wins about 59% on Ana and about 50% on anything else.
  - **What happened next:** in the test windows he went +17.8 points on 53 Ana games (SE about 7) and −5.9 on 170 other games.
- **Why one player's share of a game is small.** One game's outcome splits roughly as:
  - draft: 3.4%;
  - each player's stable hero-specific edge: 0.45 to 0.54%, so about 5% for all ten;
  - stable overall skill beyond the matchmaker's rating: 2.7% for all ten, but about zero for players with 300+ games;
  - game-to-game performance and luck: the rest, about 89%.

  It is not 10% per player, and it is not zero. The matchmaker rates players, not player-hero pairs, so it cannot remove a hero-specific edge. Out of sample, the players add 5.3 points of explained variance on all games and 4.2 on games where all ten players have 300+ earlier games.

### What exactly is measured (definitions)

- **Past edge:** the player's mean result on the hero beyond the draft-only win probability. It is not relative to the player's own average unless stated.
  - Our games: win minus WP, where WP sees the draft, map, tier and lobby average MMR.
  - Heroes Profile history rows: win minus the hero's win rate that quarter.
  - The record covers the player's full history on that hero through the day before.
- **Later:** each scored game in the spring test window (V2) and the summer window (OOT). Each uses the record as of the day before, so a player contributes one point per later game.
- **Population:** every slot in those windows whose player had 30+ games on the hero; all players, all tiers, all regions. That is 1.33M slots on the full-history panel.
- **Carry-forward:** the least-squares slope of the next game's result on the past edge, within each games bin.
- **Noise-only prediction:** the slope we would see if true edges never changed: 1 − mean(0.245/n) / variance of the records in the bin. Here 0.245 is one game's variance.

The record rebuilt on this machine reproduces section 3 to three decimals in every bin (0.243, 0.300, 0.428, 0.568), so the numbers do not depend on the 3090 (see the hardware note at the end).

### Check 1: simulation with 100% persistent skill

**Method** (`p2f_carry.py`, `simulate`):
- **Inputs kept real:** each player's games, dates and baselines on the full-history panel (48M rows).
- **Outcomes replaced:** draws around each row's baseline plus a fixed true edge per player-hero (sd 3, 4 or 5 points), plus one version where the edge drifts (correlation 0.74 at one year, sd 4 points).
- **Estimator unchanged.** Two seeds each.

| games on hero | 30-59 | 60-119 | 120-249 | 250-499 | 500-799 | 800+ |
|---|---|---|---|---|---|---|
| simulated, sd 3 points, persistent: estimator / noise-only | 0.13 / 0.12 | 0.23 / 0.23 | 0.38 / 0.37 | 0.53 / 0.55 | 0.71 / 0.69 | 0.84 / 0.83 |
| simulated, sd 4 points, persistent | 0.21 / 0.20 | 0.35 / 0.34 | 0.52 / 0.51 | 0.66 / 0.69 | 0.82 / 0.80 | 0.90 / 0.89 |
| simulated, sd 5 points, persistent | 0.30 / 0.28 | 0.45 / 0.45 | 0.63 / 0.62 | 0.76 / 0.78 | 0.89 / 0.86 | 0.93 / 0.93 |
| simulated, sd 4 points, drifting (0.74 a year) | 0.16 / 0.16 | 0.25 / 0.28 | 0.35 / 0.44 | 0.50 / 0.61 | 0.62 / 0.74 | 0.54 / 0.83 |
| **real data: estimator / noise-only** | **0.24 / 0.30** | **0.30 / 0.38** | **0.43 / 0.50** | **0.50 / 0.62** | **0.58 / 0.70** | **0.81 / 0.79** |
| real: implied true persistence (ratio) | 0.80 | 0.79 | 0.85 | 0.81 | 0.83 | 1.02 |
| real: SE of the estimate | 0.009 | 0.016 | 0.019 | 0.033 | 0.055 | 0.082 |

Reading the table:
- **With no decay, the estimator returns the noise-only prediction** (within 0.03 in every bin). So low numbers such as 24% or 30% are mostly regression to the mean, which is what a 40- or 80-game record should show.
- **The real data sit at about 0.8 times the noise-only prediction.** That ratio is the implied true persistence.
- **The pattern matches neither extreme.** It is less than full persistence but flatter than the drifting simulation: the drift version falls to about 0.65 of noise-only in the heaviest bins, while the real ratio stays at 0.8 or above.
- **Max's 800-game example:** a 62% record over 800 games, with an SE of about 1.8 points, sits in the 800+ bin. There the measured carry-forward is 0.81 against 0.79 from noise alone, so nothing is missing at that depth.
- **The true spread of player-hero edges shrinks with depth:** 5.1, 4.3, 3.9 and 3.2 points from the lightest to the 250+ bin. In the simulations it stays flat. Heavy cells span years of play, so part of their old edge has faded, and heavy one-trick players get rated up (check 2).

### Check 2: matchmaking feedback, and specialists vs mixed players

**Method** (`p2f_carry_mmr.py`). At-game player rating is the latest Heroes Profile rating parsed before the game started (`P3_MMR_AT_GAME`). (Lookup by game and player without key collisions, 2026-10-28: the earlier packed key matched some games of one player 2^20 replay ids apart; the fix moves four cells of the tables below by 0.01.) We use slots where all ten players have one (0.63M of the 1.33M), and add two controls:
- own team's mean rating minus the opponents';
- the player's rating minus their team's mean.

| games on hero | 30-59 | 60-119 | 120-249 | 250-499 | 500-799 | 800+ |
|---|---|---|---|---|---|---|
| carry-forward, no ratings | 0.25 | 0.34 | 0.44 | 0.55 | 0.66 | 0.81 |
| with both teams' ratings | 0.25 | 0.34 | 0.43 | 0.54 | 0.64 | 0.81 |
| noise-only prediction | 0.30 | 0.39 | 0.49 | 0.62 | 0.69 | 0.80 |
| implied true persistence | 0.83 | 0.88 | 0.88 | 0.88 | 0.95 | 1.01 |

- Matchmaking feedback is not hiding carry-forward. A strong record does not buy measurably tougher team matchups (−1 to +2 rating points of team gap per 10 points of record), and a team rating gap moves results by only 2 to 3 points of win probability per 100 rating points beyond what the draft and lobby already price.
- Within fifths of the player's own rating, or of the team rating gap, carry-forward moves around the pooled value with no pattern.
- Adding the player's hero-specific rating lowers carry-forward by 0.02 to 0.09. That is expected, since a hero rating already contains part of the same hero edge.

**Specialists vs mixed players** (share of the player's games on that hero), as implied true persistence:

| games on hero | hero under 15% of games | 15-50% | over 50% |
|---|---|---|---|
| 30-59 | 0.96 | 0.68 | 0.73 |
| 60-119 | 0.91 | 0.79 | 1.02 |
| 120-249 | 0.93 | 0.83 | **0.56** |
| 250-499 | 0.96 | 0.83 | **0.66** |
| 500-799 | 0.96 | 0.91 | 0.96 |
| 800+ | (1.46, 5K slots, noisy) | 0.91 | 0.89 |

Max's mechanism shows up where he predicted. A hero that is a small share of a player's games keeps nearly all of its true edge. Heroes that are most of a player's games keep less, most clearly at 120 to 499 games.

Adding the at-game ratings changes the specialist numbers by under 0.1, so the reversion is not visible as a rating effect in our data. The likely reason: a one-trick's hero edge and their overall level are the same thing, and the pre-game rating already prices most of it. The draft-plus-lobby baseline then over-states their record.

### Check 3: recency

Recency-weighted records have very different noise levels, so they are compared by how much of the next game's variance they predict: cov(record, next)² / var(record), in points².

| record | 30-59 | 60-119 | 120-249 | 250-499 | 500-799 | 800+ |
|---|---|---|---|---|---|---|
| **full record** | **5.1** | **4.3** | **5.5** | **4.8** | **4.5** | **6.1** |
| last 25 games | 4.1 | 2.2 | 1.9 | 1.3 | 0.5 | 0.4 |
| last 50 games | 5.1 | 3.5 | 3.2 | 2.4 | 1.0 | 0.9 |
| last 100 games | 5.1 | 4.3 | 4.4 | 3.2 | 2.5 | 1.5 |
| half-life 180 days | 2.6 | 2.0 | 2.9 | 3.0 | 2.5 | 3.9 |
| half-life 365 days | 3.9 | 3.4 | 4.5 | 4.5 | 4.0 | 5.4 |
| half-life 730 days | 4.9 | 4.2 | 5.5 | 5.1 | 4.8 | 6.1 |

- Recent windows carry slightly more real signal: the covariance with the next game is 11.9 against 9.5 points² at 250 to 499 games for the last 25 to 50 games.
- But they are far noisier, so the full record, or a two-year half-life, predicts best. Old games are not dragging the numbers down; they add more signal than drift.

### Check 4: robustness to the definitions

| definition | 30-59 | 60-119 | 120-249 | 250-499 | 500-799 | 800+ |
|---|---|---|---|---|---|---|
| section 3: next game, edge vs the draft, all players | 0.24 | 0.30 | 0.43 | 0.50 | 0.58 | 0.81 |
| edge relative to the player's own average (controlling for it) | 0.21 | 0.28 | 0.42 | 0.54 | 0.61 | 0.87 |
| "later" = every later game: record frozen at the spring test start, mean of all its later games, cell-weighted | 0.20 | 0.27 | 0.40 | 0.47 | 0.59 | 0.79 |
| slots with all ten ratings, plus rating controls | 0.25 | 0.34 | 0.43 | 0.54 | 0.64 | 0.81 |

The answer does not depend on which of these definitions is used.

### Check 5: how one game's outcome variance splits

**Method** (`p2f_vardecomp.py`):
- **One game's variance:** 0.25.
- **Draft:** the variance of the draft-only win probability across games (spring and summer test windows).
- **Stable player effects:** split halves on our corpus games. In 18 half-year windows, each player's (or player-hero's) games are cut into an earlier and a later half, and the covariance between the halves measures the part that holds across about three months. The residual is against the draft-only WP, which already includes the lobby's average rating.
  - Overall effect: player mean, 20+ games in each half.
  - Hero-specific effect: player-hero mean minus the player's own mean in the same half, 10+ games in each half.
  - Shares are per game; "ten players" multiplies by 10.

| part of one game's variance (0.25) | all players | players with 300+ earlier games | players with 1,000+ |
|---|---|---|---|
| draft (variance of the draft-only WP) | 3.4% | | |
| one player's stable overall effect, beyond the lobby's rating (sd) | 0.27% (2.6 points) | 0% | 0% |
| ... all ten players | 2.7% | 0% | 0% |
| one player's stable hero-specific effect (sd) | 0.45% (3.4 points) | 0.51% (3.6 points) | 0.54% (3.7 points) |
| ... all ten players | **4.5%** | **5.1%** | **5.4%** |
| the rest: game-to-game performance and luck | about 89% | about 91% | |

**What the matchmaker removes.**
- It groups players by rating: the spread of player ratings drops from an sd of 256 in the population to 165 inside a lobby, removing about 58% of the variance.
- It does not balance the two teams beyond that. The real team rating gap (sd 112) is the same as shuffling the same ten players at random (sd 110).
- Rating gaps barely move results: 1.7 points of win probability per 100 points of team rating gap, about 0.15% of a game's variance.
- Taken together, for established players any stable overall skill beyond the lobby's rating is absorbed. The 2.7% "overall" share comes entirely from players with under 300 games, mostly accounts still climbing. That matches `P3_SKILL_DRIFT.md`.

**Why a hero-specific share survives.** Matchmaking is zero-sum in the player's overall rating, not in the hero they pick. A player's rating is an average over all their heroes. When they pick the hero they are best at, they are better than their rating says, and on a hero they rarely play they are worse. The matchmaker cannot see which hero will be picked. The hero-specific share is 0.45 to 0.54% per player, about 5% for ten, and it does not shrink for veterans, so it is not new accounts climbing.

**Why section 4's "1.3% of one game per player-hero" differs.**
- That number is the whole cell (overall plus hero), on estimation-window games, including new accounts.
- Here the overall part is about zero for veterans, and the stable hero part is about 0.5%.
- Either way, ten players' stable effects total about 5 to 13% of one game's variance, not 100% and not 0%.

**Out-of-sample confirmation** (share of outcome variance explained, i.e. Brier skill):

| games | draft only | full model | players add | accuracy, draft → full |
|---|---|---|---|---|
| all 358,490 test games | 3.0% | 8.3% | **+5.3 points** | 57.0% → 61.7% |
| all ten players with 300+ earlier games (5,120) | 2.4% | 6.6% | **+4.2 points** | 55.6% → 60.6% |
| all ten with 1,000+ (242 games, noisy) | 6.3% | 13.5% | +7.2 points | 61.6% → 64.9% |

The model explains out of sample about what the stable shares say is there: 5.3 points of variance from players against roughly 5 to 7 points of stable player variance. It keeps explaining it in lobbies of established players only.

### Hardware note: the 3090's memory

Four runs on the 3090 (the two ladder arms on Oct 19, a rerun, and this check's first attempt) failed with array indices too large to be possible. Each wrong index equals a valid index plus exactly 2^28: one flipped bit at position 28. That is a memory fault on that machine (its RAM is not error-correcting), not a code bug.
- **Integer arrays:** a flip usually crashes, as here.
- **Floating-point arrays:** a flip at that bit is harmless in float64 (a change near the 7th to 8th significant digit). In float32 it hits the exponent and silently produces a wildly wrong value.

Results already produced on the 3090 that were checked elsewhere agree:
- this section's records, rebuilt on the main box, reproduce section 3's arm (computed on the 3090) to three decimals;
- the section 7 decomposition's "all current" row reproduces the tournament's own score (61.51%) exactly.

But silent errors in float32 arrays cannot be ruled out. The 3090's RAM should be tested (memtest86 or Windows Memory Diagnostic) before more definitive runs there. Everything in this section was computed on the main box.

---

## 12. Party (premade stack) structure

**Plain answer.**
- **How common stacks are:** 46% of teams are five solo players. 25% have one duo, 11% a trio, 5% two duos, 6% a four-stack, 4% a full five-stack, and 2% a trio plus a duo. In 58% of games the two teams' structures differ; in 69% at least one team has a stack.
- **Raw effect:** stacks win slightly more. Against the draft-only prediction, a team's largest stack being one size bigger than the other team's is worth about 1.2 points of win probability, and three sizes bigger about 2.4.
- **Most of that is the stacked players' own skill.** Against the main player model, the stack effect falls by a third to a half. After controlling for each member's skill, one more player in a 3+ stack is worth 0.12 points (SE 0.05), against 0.44 before the control. The team rating gap adds almost nothing on top.
- **Five-stacks win no more than the player model already expects** (+0.0 points).
- **Value for prediction:** a party term adds +0.1 to +0.2 log-loss points (×1000) on top of the main model, and −0.01 to +0.02 points of accuracy (SE 0.03 to 0.06). That is no measurable edge. On the draft-only model it adds +0.4 log-loss points and up to +0.1 points of accuracy. This confirms `P3_EXTENSIONS.md` (+0.0002 to +0.0004 log loss).
- **By tier:** Bronze stands out, as before. Bronze 3+ stacks are +2.2 points beyond the draft and +2.1 beyond the player model. Elsewhere the remainder is +0.2 to +1.3. Bronze's player model has the thinnest histories, so it captures the least of the stack members' skill.

**Method** (`p2f_party.py`, main box).
- **Party groupings:** `replay_players.party` (0 = solo; members of one team share an id), from the full-panel slot table.
- **Games:** structure counts use all 2.23M corpus games with ten labelled players. Win rates use the spring and summer test games, one row per team.
- **"Beyond the draft"** = result minus the draft-only WP. **"Beyond the player model"** = result minus the main model's prediction.
- **Value:** the main model's combiner plus three team differences (players in a 2+ party, players in a 3+ party, a five-stack flag), fit on V1 and scored on V2 and OOT, paired by game.
- **Check:** a regression of the result beyond the draft on the 3+-stack player difference, then adding the player-model skill difference and then the at-game team rating gap.

By own team structure (test games; points of win probability; SE of each cell about 0.1 to 0.35):

| own structure | teams | raw win % | beyond the draft | beyond the player model |
|---|---|---|---|---|
| all solo | 240,775 | 48.8 | −1.03 | −0.66 |
| duo | 232,031 | 49.7 | −0.23 | −0.08 |
| two duos | 49,057 | 50.8 | +0.76 | +0.55 |
| trio | 91,775 | 51.5 | +1.38 | +1.05 |
| trio + duo | 20,678 | 51.3 | +1.12 | +0.59 |
| four-stack | 47,630 | 52.1 | +1.75 | +0.86 |
| five-stack | 35,034 | 51.5 | +0.96 | +0.01 |

By the difference in the two teams' largest stack:

| own largest minus opponents' | 0 | +1 | +2 | +3 | +4 (five-stack vs five solos) |
|---|---|---|---|---|---|
| teams | 237,652 | 143,776 | 59,952 | 34,840 | 1,096 |
| raw win % | 50.0 | 51.3 | 52.5 | 53.1 | 48.7 |
| beyond the draft | 0 | +1.17 | +2.13 | +2.41 | −1.97 (SE 1.5) |
| beyond the player model | 0 | +0.78 | +1.20 | +1.05 | −3.61 (SE 1.5) |

- Five-stacks almost never meet five solo players (1,096 test games); they are matched against other stacks.
- Some matchups: four-stack vs solos +3.5 beyond the draft and +2.1 beyond the player model; trio vs solos +2.9 and +1.9; five-stack vs duo +1.6 and +0.3; five vs five 0.

3+ stacks vs all-solo teams by tier, beyond the draft / beyond the player model, in points:

| tier | 3+ stack | all solo |
|---|---|---|
| Bronze | +2.22 / +2.05 | −0.49 / −0.54 |
| Silver | +1.00 / +0.84 | −0.84 / −0.76 |
| Gold | +1.09 / +0.64 | −0.80 / −0.40 |
| Platinum | +1.25 / +0.28 | −1.89 / −1.04 |
| Diamond | +1.43 / +0.16 | −1.72 / −0.67 |
| Master | +2.18 / +1.29 | −1.18 / −0.65 |

**Is the stack effect just better players?** Mostly, yes (159K test games with every player's rating known):
- each extra player in a 3+ stack: +0.44 points beyond the draft (SE 0.05);
- with each member's player-model skill: +0.12;
- with skill and the team rating gap: +0.11.

Stacks carry 0.29 points more summed player-model skill per stacked player, and a slightly higher team rating (+0.9 rating points per stacked player). About three quarters of the raw stack effect is the members' measured skill. The small remainder (about 0.1 points per stacked player) is consistent with coordination, and only Bronze shows a sizable one.

---

## 13. Korea: a sample problem, and of which kind?

**Plain answer.** Both kinds of "small" exist, but not in the place we feared.
- **Our corpus does not under-sample Korean games relative to Heroes Profile.** For the same panel accounts, our corpus holds 67% of a Korean account's 2025 Heroes Profile games, against 59 to 60% in the Americas and Europe.
- **Korean accounts have far fewer recorded games.**
  - An active Korean panel account (20+ games in 2025) has a median of 101 games in 2025, against 160 to 172.
  - Across all accounts in our corpus, a Korean account with a 2025 game has 13 corpus games that year, against 28 to 32.
  - We cannot tell from Heroes Profile data alone whether Korean players play less or upload less to Heroes Profile. Its histories only contain uploaded games. Either way, the thin histories come from upstream, not from our sampling.
- **Is the gap statistically clear?** On the two test windows together, yes.
  - Korea's gain is 3.5 points (95% CI 2.7 to 4.2, 17,114 games), against 5.0 (4.7 to 5.3) in the Americas and 4.7 (4.5 to 4.9) in Europe.
  - Differences: −1.5 (−2.3 to −0.7) against the Americas and −1.2 (−2.0 to −0.4) against Europe.
  - On the spring window alone (3,322 Korean games) the accuracy differences are not clear (−0.7 and −0.6, CIs spanning zero). The log-loss gap against the Americas is (−8.9, CI −16.3 to −1.5).
  - These CIs resample games. Korea's player pool is small and players recur often, so the true CIs are somewhat wider. The pooled differences would likely remain clear.

**Method** (`p2f_korea.py`, main box).
- **Panel accounts:** the Heroes Profile match histories of the pulled panel (store full_1017).
  - Rows ending in 2025, per account.
  - "Full history" = all rows (Heroes Profile and our corpus); "in our corpus" = rows that are corpus games.
  - Region is from the account key. The panel was chosen by corpus activity (100+ window games for panel 1, 20+ games since 2025 for panel 2), the same rule for every region.
- **All accounts:** corpus games in 2025 per account, from the slot table.
- **Test:** gain = main model accuracy minus draft-only accuracy on the same games; CIs from 2,000 resamples of games within region.

| region | panel accounts | active in 2025 | median 2025 games, full history | median 2025 games in our corpus | corpus share of full history |
|---|---|---|---|---|---|
| Americas | 13,743 | 10,869 (79%) | 172 | 103 | 60% |
| Europe | 27,067 | 21,421 (79%) | 160 | 97 | 59% |
| Korea | 1,307 | 889 (68%) | 101 | 69 | 67% |

| region | accounts with a 2025 corpus game | with 20+ | median corpus games, 20+ accounts | corpus games per account with a game |
|---|---|---|---|---|
| Americas | 61,490 | 19,876 | 52 | 32.5 |
| Europe | 138,273 | 39,886 | 48 | 27.8 |
| Korea | 12,815 | 2,027 | 37 | 12.8 |

Gain from the players by region, main model, accuracy points (95% CI):

| window | Americas | Europe | Korea | Korea minus Americas | Korea minus Europe |
|---|---|---|---|---|---|
| spring (V2) | 4.89 (4.20, 5.60), 22,332 games | 4.82 (4.34, 5.29), 46,820 | 4.18 (2.47, 5.93), 3,322 | −0.67 (−2.55, 1.18) | −0.59 (−2.39, 1.22) |
| summer (OOT) | 4.98 (4.65, 5.33), 91,555 | 4.69 (4.44, 4.92), 180,669 | 3.31 (2.39, 4.17), 13,792 | −1.68 (−2.63, −0.75) | −1.38 (−2.27, −0.50) |
| both | 4.97 (4.67, 5.28), 113,887 | 4.71 (4.50, 4.92), 227,489 | 3.48 (2.74, 4.25), 17,114 | **−1.49 (−2.28, −0.68)** | **−1.23 (−2.00, −0.44)** |

In log loss (×1000) the pooled differences are −12.7 (−16.2, −9.2) and −10.5 (−13.9, −7.2).

**For the paper:** "Korea gains less (3.5 against 4.7 to 5.0 points; the difference is statistically clear on the pooled test games). Korean accounts have about 40% fewer recorded games per year in Heroes Profile itself, and our corpus covers them at least as well as other regions. So the shortfall reflects thinner public histories, and at equal history depth about a point remains unexplained (section 10)."

---

## Files

- Code: `training/personalization/p2f_clocks.py`, `p2f_game_clock.py`, `p2f_player_clock.py`, `p2f_arms.py`, `p2f_more.py` (sections 8 to 10), `p2f_ladder.py` (the two new ladder arms, run on the 3090 CPU), `p2f_stale_decomp.py` (section 7's decomposition, 3090 CPU), `p2f_carry.py`, `p2f_carry_mmr.py` and `p2f_vardecomp.py` (section 11, main box), `p2f_party.py` (section 12) and `p2f_korea.py` (section 13), and `p2f_stale_cell.py` (the year-stale table of section 7, run on the 3090 CPU against the tournament's saved states).
- Results: `training/personalization/results/p2f/*.json`.
- Large inputs, all gitignored:
  - `cache/p2f_snapshot_games.npz`, a compact copy of the snapshot built by the game-clock script;
  - the A2S and `full_ms_layer` arms (`results/rcf_full/p3c/`) and the full slot table (`cache_full/x_slots_ext.npz`) from the 3090, copied to a scratch directory and not committed. They hold player keys only.
- Compute: this box, cores 48-51, nice 19, about an hour in total. The 3090 ran only CPU work (cores 24-31, nice 19, hotsjob jobs `p2f_ladder_*`, about 10 minutes). No GPU was used, and the tournament's jobs and files were only read.
- One reliability note: the 3090 has a memory fault (single bit flips at bit 28; section 11, hardware note). Four runs failed on it; everything in section 11 was computed on the main box.
