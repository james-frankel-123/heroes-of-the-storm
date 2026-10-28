# P3: heroes you'll become good at, how much stats matter, and replay timelines (2026-10-21)

Two requests from Max and Ernest:
- **A.** A backtest of "heroes you'll become good at": rank heroes a player has barely played, then check how the ones they took up actually went.
- **B.** Push the scoreboard further (per-metric ablations, a nonlinear model, its role in cold start), and test the parsed replay timelines.

Everything runs on the full_1017 panel (the P3_TAILS rows: 22.4M corpus slots plus 25.5M Heroes Profile rows, 42,009 pulled players with full histories), with the fitted `full_ms` (main model) and `full_imp2` params. Local research DB only, no Neon, no sealed replays. "Edge" or "pp" below is a game's result minus what the draft predicted, in points of win probability: y − draft WP on corpus games, y − the hero's quarterly win rate on Heroes Profile rows.

## Headline

- **The main model is the best hero recommender, and its scores are close to calibrated.** Candidates are heroes with under 10 games. Among those a player went on to play 20+ times, the main model's top third realized **+1.9 points** more edge in games 11-50 than its bottom third, within the same player (SE 0.3). The two thirds realized +1.85 and −0.13 points. Within a player, each point the model predicts is worth 0.84 points of realized later edge.
- **Every simple baseline fails.** Hero win rate, hero popularity and "same role as your best hero" each predict −0.4 to +0.4 points (SE 0.3) for steady-state edge, the same as a random ranking. Popularity and same-role do predict which heroes a player will *take up* (2.6% of their top 3 against 1.0% for random), and a better first game, but not who becomes good.
- **The scoreboard is what makes the model's recommendation work.** The same model with its scoreboard evidence removed falls from +1.9 to +0.9 (SE 0.3) on later edge, and from +1.8 to +0.4 on the first 10 games. Play preference alone (no results at all) is strong on first games, +4.0 for the very first game, but fades to +1.1 at steady state.
- **For heroes a player is bad at, the model predicts the future level well, but nothing predicts who improves.** Of 217K bad cells (20-299 games, record −5 pp or worse), 80% are dropped within six months and 3.8% are kept for 20+ more games. On the kept ones the model's estimate predicts games 11-50 with slope 0.83 (SE 0.06); the raw record predicts with slope 0.25. Player learning speed and recent trend add nothing (0.02 ± 0.03 and 0.00 ± 0.01).
- **B1 scoreboard: the win-relevant signal is in the fight metrics, spread across them.** In a frozen-state prospective test on held-out players, hero damage, damage taken, deaths, kills, assists, outnumbered deaths and escapes together keep 96% of the scoreboard's value, and dropping them loses 51%. No single metric is indispensable: the biggest single loss is kills (13%), then damage taken (11%). Alone, hero damage carries the most (40%). Healing, protection and escapes add nothing at the margin.
- **No nonlinear gain, but linear signal is left on the table.** A small MLP on the metric profile beats a linear model only when the main model is absent. With the main estimate included it is slightly worse (−0.07, SE 0.03). A direct linear use of the per-hero metric means adds +0.16 (SE 0.02) over the main model, half as much again as the scoreboard's whole contribution through the factor model (+0.31). A higher-rank metric channel is the next model change to try.
- **The scoreboard is what lets the model recommend never-played heroes.** On never-played heroes the main model separates later edge by +1.6 points top third vs bottom third, against +0.2 with its scoreboard evidence removed and +1.1 for imp2.
- **B2 replays:** APM is the most stable per-game trait measured in this project (0.91 of one game's variance is a stable player×hero effect) and the scoreboard does not contain it. Every early-death, positioning and participation feature is 2% to 13% stable, about half explained by the scoreboard, and adds nothing measurable. Over the scoreboard the timeline features add +0.24 (SE 0.10) log-loss points under cross-validation and −0.07 (SE 0.19) on a forward split. That is early and underpowered. Build APM now; wait about 4 weeks of the p2_panel_v1 cohort for the rest.

**The selection caveat, stated plainly.** Players choose which heroes to keep, and they keep the ones that start well: heroes taken up to 10+ games average +3.0 points in their first 10 games, while the first game of every hero tried averages −4.5. So realized edges among taken-up heroes are not what a random player would get by following the recommendation. The comparisons below are within player and by the recommender's rank, so they ask "of the heroes you took up, did the ones the model liked go better?" That is answerable. "Would you have done well on the model's top pick if you had played it?" is only partly answerable, through the first-game outcome of every hero tried, which is not selected on results.

---

## A. Heroes you'll become good at: the backtest

### A1. Set-up (`p3_hr_backtest.py --pooled`)

- **Cutoffs.** Four cutoffs, each followed for 183 days, so the outcome windows do not overlap:
  - 2024-08-10, 2025-02-10 and 2025-08-10: the held-out half of pulled players only, because the params were fit on the other half with targets that run into these windows;
  - 2026-02-10 (end of E, after every fit): all pulled players.
- **Players.** Active players only: 50+ games before the cutoff and 30+ in the follow-up. That is 8,716 to 19,273 per cutoff, 49,640 player-cutoffs in all.
- **Candidates.** Heroes with fewer than 10 games (primary) or fewer than 5 games in the player's full history before the cutoff: 72 per player at <10, 65 at <5.
- **Outcomes** in the 183 days after the cutoff, every game in the player's full history (corpus plus Heroes Profile):
  - **first game** of every candidate the player tried: 541,451 heroes. Not selected on results;
  - **early edge**, games 1-10, for heroes taken up to 10+ games: 38,550 heroes. This is the learning cost;
  - **later edge**, games 11-50, for heroes taken up to 20+ games (11,666) and 30+ games (5,267). This is the steady state.
- **Recommenders.** Each scores all 90 heroes from history before the cutoff only.

| name | what it ranks by |
|---|---|
| main | `full_ms` posterior mean: win records plus 15 scoreboard metrics, rank 2 |
| main, no scoreboard | the same params with the scoreboard evidence removed |
| impmet | new fit for this study: play preference plus scoreboard plus wins (the oct05 "3+4 combined", refit on full_1017) |
| imp2 | `full_imp2`: win records plus play preference |
| preference only | imp2's prior from play counts alone, with no results at all |
| main + preference | average of the main and preference ranks |
| scoreboard similarity | per hero, a ridge regression of a player's record on that hero on their hero-relative scoreboard profile from their other heroes (fit on cells with 20+ games before the cutoff), applied to the player's profile |
| today's win-only CF | the A1 kernel (win records only, with role columns) |
| hero win rate | the hero's win rate in the 90 days before the cutoff |
| popularity | the hero's pick count in the 90 days before the cutoff |
| same role as best hero | Blizzard role of the player's best 20+ game hero (main model), ties by popularity |
| random | uniform |

- **Statistics.** The recommender ranks the player's candidates, and each taken-up hero gets its rank percentile (0 = top). "Top third vs bottom third" compares taken-up heroes in the top and bottom thirds of the candidate list.
  - **Within player:** each edge minus the mean of the same player-cutoff's taken-up heroes, using player-cutoffs with 2+ taken-up heroes. This removes the player's level and any between-player selection.
  - SEs come from a Poisson bootstrap over players (200 draws), pooled across cutoffs.

### A2. Main table: <10 games before the cutoff, four cutoffs pooled

Realized edge in pp. Top / bottom = mean edge of taken-up heroes the recommender ranked in its top / bottom third of candidates. "Within" is the same difference demeaned within player (SE).

| recommender | first game, all tried: within | games 1-10 (10+ taken): top / bottom | within | games 11-50 (20+ taken): top / bottom | **within** | games 11-50 (30+ taken): within |
|---|---|---|---|---|---|---|
| **main** | +3.85 (0.16) | +3.92 / +1.51 | **+1.83 (0.18)** | +1.85 / −0.13 | **+1.87 (0.30)** | +0.99 (0.44) |
| impmet (preference + scoreboard) | **+4.09 (0.16)** | +3.98 / +1.57 | **+1.89 (0.19)** | +1.87 / +0.04 | **+1.92 (0.30)** | +1.03 (0.41) |
| imp2 | +3.35 (0.16) | +3.74 / +1.71 | +1.21 (0.18) | +1.72 / −0.00 | +1.36 (0.30) | +0.54 (0.39) |
| preference only (no results) | +4.03 (0.18) | +3.67 / +1.97 | +1.64 (0.20) | +1.11 / +0.54 | +1.12 (0.29) | +0.81 (0.39) |
| main + preference ranks | +4.11 (0.17) | +3.79 / +1.52 | +1.91 (0.18) | +1.42 / +0.03 | +1.49 (0.29) | +1.00 (0.39) |
| scoreboard similarity | +0.81 (0.17) | +4.04 / +2.16 | +1.69 (0.19) | +1.84 / +0.46 | +1.08 (0.29) | +1.16 (0.39) |
| main, no scoreboard | +1.35 (0.17) | +3.41 / +2.27 | +0.41 (0.19) | +1.73 / +0.15 | +0.90 (0.27) | +0.31 (0.38) |
| today's win-only CF | +1.67 (0.16) | +3.60 / +2.04 | +0.64 (0.18) | +1.68 / +0.07 | +1.01 (0.29) | +0.27 (0.41) |
| hero win rate | −0.01 (0.15) | +2.67 / +3.18 | −0.37 (0.19) | +0.76 / +0.90 | +0.40 (0.27) | +0.69 (0.40) |
| popularity | +2.89 (0.16) | +2.93 / +3.13 | −0.01 (0.20) | +0.68 / +1.56 | −0.15 (0.32) | −0.51 (0.45) |
| same role as best hero | +2.29 (0.17) | +2.88 / +2.82 | +0.05 (0.21) | +0.89 / +1.71 | −0.35 (0.30) | −0.31 (0.43) |
| random | −0.11 (0.16) | +3.10 / +2.96 | +0.14 (0.20) | +0.99 / +0.94 | −0.09 (0.30) | −0.11 (0.42) |
| heroes / player-cutoffs with 2+ | 541,451 / 43,790 | 38,550 | 8,035 | 11,666 | 2,292 | 5,267 / 857 |

Mean edges: first game of every tried hero −4.5; games 1-10 of taken-up heroes +3.0; games 11-50 of 20+ heroes +1.0; games 11-50 of 30+ heroes +2.3. The <5-game candidate set (`pooled_n5.json`) gives the same ordering with about 1.5 times the SEs: within-player games 11-50, main +1.93 (0.43), impmet +2.14 (0.37), imp2 +1.34 (0.39), preference +1.03 (0.39), no scoreboard +0.75 (0.41), popularity −0.54 (0.45), random +0.25 (0.40).

**By cutoff** (within player, <10 games; games 1-10 / games 11-50):

| cutoff | main | imp2 | preference only | win-only CF | random |
|---|---|---|---|---|---|
| 2024-08-10 (held-out half) | +1.97 / +1.77 | +1.29 / +1.54 | +1.94 / +1.24 | +0.52 / +0.52 | −0.05 / −0.07 |
| 2025-02-10 (held-out half) | +2.02 / +2.31 | +1.53 / +1.13 | +1.58 / +1.34 | +1.09 / +1.51 | +0.03 / −0.32 |
| 2025-08-10 (held-out half) | +2.16 / +1.55 | +1.64 / +1.34 | +1.70 / +0.78 | +0.92 / +0.78 | +0.27 / +0.32 |
| 2026-02-10 (all players) | +1.36 / +1.76 | +0.60 / +1.46 | +1.44 / +1.05 | +0.16 / +1.11 | +0.25 / −0.19 |

The main model leads at every cutoff on games 1-10 and at three of four on games 11-50.

### A3. Calibration, hit rates, take-up

- **The main model's numbers mean what they say.** Among taken-up heroes (20+ games, <10 before), regressing realized later edge on the model's score within player gives a slope of **0.84**, and 0.67 pooled. imp2 gives 0.66 / 0.42 and today's win-only CF 0.49 / 0.52. Within a player the main model's scores spread with sd 2.6 points across these heroes.
- **Hit rate.** The share of taken-up heroes (20+ games) that end with a positive later edge:

  | | top third | bottom third |
  |---|---|---|
  | main | 57.4% | 51.4% |
  | imp2 | 56.7% | 51.5% |
  | win-only CF | 56.9% | 51.5% |
  | random | 54.2% | 54.0% |
  | popularity | 53.6% | 55.4% |

- **Take-up ("precision at 3").** Players take up very few new heroes: 1.1% of candidates reach 10 games in six months, about 0.8 per player.

  | recommender | top 3 tried | top 3 taken up (10+) | bottom 3 taken up |
  |---|---|---|---|
  | random | 14.9% | 1.0% | 1.0% |
  | main | 18.4% | 1.2% | 0.5% |
  | impmet | 18.9% | 1.2% | 0.4% |
  | popularity | 28.0% | 2.6% | 0.2% |
  | same role as best | 27.0% | 2.7% | 0.2% |

  So popularity and role predict *what* people pick up. The skill models predict *how it goes*. A product list should say which one it is showing.

- **Literal top-3 vs bottom-3 per player.** Only a few dozen player-cutoffs took up 6+ candidates, too few to report.

### A4. Selection, and what is and is not identified

- **The first game is the cleanest test.** It is the only outcome observed for every hero a player tried, whatever happened next. Main and preference both separate it by about 4 points. Part of that is learning cost, not skill: popularity (+2.9) and same-role (+2.3) also predict a better first game, because familiar kinds of heroes cost less to learn. The skill models beat them by 1 to 1.5 points there, and only the skill models keep predicting at steady state.
- **Games 11-50 are conditioned on reaching 20 games.** That depends on how games 1-10 went: players drop heroes that start badly. If the model's low-ranked heroes are kept only when they start unusually well, the within-player difference understates the model. If its high-ranked heroes are kept more readily, the difference could overstate it. Two checks:
  - the model's top third is not taken up at a higher rate than random (1.2% vs 1.0%), so take-up is not strongly sorted by the model's rank;
  - the ranking separates the unselected first game by a similar amount.
  
  So the steady-state result is unlikely to be a selection artefact. It is still an observational comparison among heroes players chose.
- **What a recommendation is worth.** Read literally, "pick the model's top-third hero instead of its bottom-third hero" is worth about 2 points of edge at steady state, for the kind of player who took both up. That is roughly the gap between a hero you have 20 games on and one you have 2 games on (experience offset, P3_HERO_STRENGTH section 7).

### A5. Heroes a player is bad at: who improves? (`p3_hr_traj.py`)

- **Cells.** The same four cutoffs and players. Player×hero cells with 20-299 games and a raw record of −5 pp or worse.
- **Outcome.** The mean edge over games 11-50 after the cutoff, for cells the player kept playing 20+ times. Games 1-10 are selected on (players keep heroes that start well) and are shown separately.

| | value |
|---|---|
| bad cells | 217,325 |
| dropped (under 5 games in the next 6 months) | 172,908 (80%) |
| kept for 20+ games | 8,216 cells, 4,888 players (3.8%) |
| mean record at the cutoff | −9.9 |
| main model's estimate at the cutoff (offset + posterior) | −3.2 |
| realized, games 1-10 after | +1.7 (selected) |
| realized, games 11-50 after | **−0.9** |
| share with games 11-50 at or above 0 | 48% |

| main model's estimate, tercile | estimate | record | realized games 1-10 | **realized games 11-50** | share ≥ 0 |
|---|---|---|---|---|---|
| "stays bad" | −5.9 | −11.9 | −0.6 | **−3.0** | 42% |
| middle | −3.0 | −9.2 | +2.0 | −0.8 | 48% |
| "recovers" | −0.7 | −8.6 | +3.6 | **+1.0** | 54% |

| raw record, tercile | record | estimate | realized games 11-50 |
|---|---|---|---|
| worst | −15.1 | −4.3 | −2.0 |
| middle | −8.7 | −3.0 | −0.8 |
| least bad | −6.0 | −2.3 | +0.0 |

| games on the hero at the cutoff | estimate | record | realized games 11-50 |
|---|---|---|---|
| 20-39 | −2.9 | −11.9 | −0.2 |
| 40-99 | −3.2 | −9.4 | −0.9 |
| 100-299 | −3.6 | −7.8 | **−2.0** |

Regressions of realized games 11-50 on as-of predictors (player-bootstrap SE). The outcome's variance is 149 pp², of which about 132 is game noise, so about 17 pp² is predictable at most.

| predictors | coefficients | R² (share of the ~17 pp² true variance) |
|---|---|---|
| raw record | record 0.25 (0.04) | 0.009 (8%) |
| main estimate | estimate **0.83 (0.06)** | 0.027 (24%) |
| main + record | estimate 0.76 (0.06), record 0.09 (0.04) | 0.029 |
| main + learning speed + new-hero level + recent trend + log games | estimate 0.76 (0.06); speed 0.02 (0.03); new-hero level 0.08 (0.03); trend 0.00 (0.01); log games −0.73 (0.18) | 0.030 |

Predictor definitions:
- **learning speed:** over the player's other heroes with 30+ games, the mean of games 6-30 minus games 1-5, shrunk and centred;
- **new-hero level:** the player's mean on first-5 games of other heroes, which is the temporal layer's learning term;
- **trend:** the cell's last 10 games minus its earlier games.

Reading:
- **The model sorts bad heroes into "will stay bad" and "is not really that bad", and its numbers are close to right.** Its pessimistic third realizes −3.0 and its hopeful third +1.0. The raw record overstates badness by a factor of four (slope 0.25): most of a −10 record over 40 games is bad luck.
- **Nobody is measurably a fast improver.** A player's past improvement on other heroes does not predict improvement on this one (0.02 ± 0.03 per point). The temporal layer's new-hero term carries a little (0.08 ± 0.03), consistent with P3_RICH_CF, but it is a level, not a speed. Recent trend is pure noise. The one systematic pattern is practice already spent: heroes that stay bad after 100+ games stay bad (−2.0), while bad records over 20-39 games mostly wash out (−0.2).
- **The model is about 2 points too pessimistic on kept cells** (intercept +1.7). That is the selection again: players keep the bad heroes that are going better.

### A6. Worked example (an anonymized heavy player, cutoff 2026-02-10, `results/hr/example_player.txt`)

A player with about 5,500 games before the cutoff and 28 heroes with 20+ games, mostly Hogger, Orphea, Abathur, Gazlowe and Samuro. They have 52 candidates (under 5 games).
- **Main model's top 5:** Li Li −0.9, Leoric −1.8, Probius −2.3, D.Va −2.4, Sgt. Hammer −2.5. These are expected edge in pp before learning cost, relative to the draft.
- **Preference-only top 5:** Lunara, E.T.C., Arthas, Thrall, Whitemane.
- **What they took up in the next 7.5 months:** ten candidates reached 10+ games, none from either top 5. That is typical: top-3 take-up is about 1%.

| taken up | games after | main rank percentile | main score | games 1-10 | games 11-50 |
|---|---|---|---|---|---|
| Deathwing | 169 | 0.16 | −2.6 | −8.9 | −11.5 |
| Stitches | 15 | 0.22 | −2.8 | −5.0 | |
| Qhira | 85 | 0.25 | −2.8 | −10.1 | +19.2 |
| Zagara | 25 | 0.27 | −3.0 | +10.3 | −30.2 |
| Thrall | 62 | 0.49 | −3.6 | −45.1 | +1.4 |
| Chromie | 14 | 0.76 | −5.1 | −33.3 | |
| Sylvanas | 25 | 0.80 | −5.8 | −35.6 | −13.3 |
| Kerrigan | 13 | 0.82 | −5.9 | −19.0 | |
| Chen | 25 | 0.90 | −6.2 | +40.8 | −4.4 |
| Genji | 34 | 0.98 | −7.3 | −38.5 | −29.3 |

- The model's top-third pickups averaged −3.4 in games 1-10, against −17.1 for its bottom-third pickups. In games 11-50 the averages were −7.5 and −15.7.
- Single heroes swing by ±30 points over 10-40 games (one game's sd is 49 points), so no individual row is evidence. The model's ordering is only visible in aggregate.
- For this player the honest product sentence is: "you are likely to be about 3 points below your usual level on any new hero for a while; Li Li, Leoric and Probius are the least costly to learn". It is not "you will be good at Li Li".

### A7. For the product

1. Rank never- or rarely-played heroes by the main model's estimate, or by impmet, which ranks the same at steady state (+1.9) and slightly better for first games (+4.1 against +3.9). The main + preference rank average buys the same first-game gain but loses steady-state accuracy (+1.5).
2. Show the estimate as a number with the first-game band (P3_HERO_STRENGTH 5b), not as "you'll be good". The typical top-third pick is still about 3 points below the player's comfort heroes for its first 10 games.
3. Do not use hero win rate, popularity or same-role as a "you'll be good at" signal. Popularity is fine as a "players like you also play" list, labelled as such.
4. For heroes the player is bad at, the main estimate is the right "stays bad / may recover" signal. Do not show a "fast learner" badge: nothing in the data supports it.

---

## B1. How much the scoreboard matters

Code: `p3_hr_ablate.py` (refits, 5090), `p3_hr_sbeval.py` (prospective evaluation), `p3_hr_backtest.py` (part A arms). Results: `results/hr/sbeval.json`, `results/hr/ablate/abl_*.json`.

### B1.1 Design: a frozen-state prospective test

- **State.** Frozen at 2026-02-10 (end of E) from every history row before that day. The script asserts that rewriting every row on or after the cutoff (results, residuals, metrics) leaves the state unchanged.
- **Targets.** Every corpus slot game after the cutoff (V1, V2 and OOT windows, to 2026-09-27): 4.30M slots from 172,433 players.
- **Baseline per slot.** Draft WP plus the experience offset at the slot's lag-1 counts. Counts only, no results.
- **Skill score.** Enters as p = baseline + β·s.
- **Player split.** β, and every learned model, is fit on the fit-half players' slots. Everything is scored on the held-out half: 2.15M slots, 86,408 players. These are the same RandomState(7) halves the params were fit on.
- **Gain.** Slot log-loss improvement over the baseline, ×1000 per slot, with player-bootstrap SE.
- **Scale.** Slot-level numbers are per player, not per game, and the state does not update after the cutoff. On the held-out slots the offset alone gives +0.81 over WP. The main model adds **+0.74 (SE 0.04)** on top. The no-scoreboard version adds +0.43, so the scoreboard's contribution here is **+0.31 (SE 0.03)**, about 40% of the skill gain. That is close to its share in the nightly game-level model: +9.8 of the +20 log-loss points beyond the experience offset (P2_NARRATIVE_FACTS section 8).

| arm | held-out gain ×1000 (SE) | V2 | OOT |
|---|---|---|---|
| main (`full_ms`) | +0.742 (0.038) | +1.00 | +0.61 |
| main, scoreboard evidence removed | +0.434 (0.027) | +0.51 | +0.35 |
| impmet (preference + scoreboard, new fit) | +0.708 (0.037) | +0.91 | +0.60 |
| imp2 (preference, no scoreboard) | +0.504 (0.028) | +0.62 | +0.42 |
| today's win-only CF | +0.441 (0.027) | +0.53 | +0.37 |

Paired: main − no scoreboard +0.308 (0.028); main − imp2 +0.238 (0.026); impmet − main −0.034 (0.011); impmet − imp2 +0.204 (0.023). Adding play preference to the scoreboard model does not help prediction. Its held-out fit LL is also 56 worse than main's, as on the oct05 panel.

### B1.2 Per-metric ablations

- **The refits.** Each arm refits the main model's exact spec on the 5090 (rank 2, shared loadings, per-hero s², predictive objective, `a_shrink`, 3 L-BFGS rounds) with a subset of metrics. A metric is removed by zeroing its conditioning-set sums and counts, which also masks its loadings.
- **Reproduction.** The all-metrics refit reproduces `full_ms`: held-out predictive LL −302,117 against −302,114.
- **Columns.** Two tests per arm:
  - "prospective" is the paired difference against the all-metrics refit in the frozen-state test above, with its share of the scoreboard's +0.308;
  - "held-out LL" is the fit's own test, the predictive LL of 105,453 held-out players' E targets, relative to all metrics.
- **Groups.** fight = hero damage, damage taken, deaths, kills, assists, outnumbered deaths, escapes; macro = siege and structure damage, XP, merc captures, regen globes; sustain = healing, self-healing, protection.

| metric or group | drop it: prospective (SE) | share of the scoreboard's value lost | held-out LL | keep only it: prospective (SE) | share kept | held-out LL |
|---|---|---|---|---|---|---|
| hero damage | −0.026 (0.010) | 8% | −89 | −0.185 (0.021) | **40%** | −382 |
| kills | **−0.041 (0.010)** | **13%** | +19 | −0.192 (0.019) | 38% | −408 |
| damage taken | −0.033 (0.009) | 11% | −12 | −0.284 (0.024) | 8% | −1,425 |
| regen globes | −0.028 (0.009) | 9% | −22 | −0.267 (0.023) | 13% | −1,438 |
| merc captures | −0.022 (0.005) | 7% | −26 | −0.233 (0.024) | 24% | −612 |
| assists | −0.017 (0.005) | 5% | +9 | −0.254 (0.024) | 18% | −718 |
| structure damage | −0.017 (0.007) | 5% | −20 | −0.217 (0.022) | 30% | −595 |
| self-healing | −0.014 (0.004) | 5% | −20 | −0.247 (0.021) | 20% | −654 |
| siege damage | −0.009 (0.004) | 3% | −24 | −0.249 (0.022) | 19% | −648 |
| XP contribution | −0.009 (0.004) | 3% | −33 | −0.247 (0.021) | 20% | −561 |
| outnumbered deaths | −0.005 (0.005) | 2% | −13 | −0.220 (0.023) | 28% | −604 |
| protection | −0.002 (0.002) | 1% | −4 | −0.290 (0.024) | 6% | −904 |
| deaths | −0.001 (0.005) | 0% | −15 | −0.207 (0.025) | 33% | −651 |
| healing | −0.000 (0.004) | 0% | −3 | −0.271 (0.023) | 12% | −841 |
| escapes | −0.000 (0.003) | 0% | −26 | −0.245 (0.024) | 20% | −637 |
| **fight (7)** | **−0.156 (0.014)** | **51%** | −239 | **−0.012 (0.012)** | **96%** | −57 |
| macro (5) | −0.015 (0.008) | 5% | −38 | −0.199 (0.016) | 35% | −326 |
| sustain (3) | −0.017 (0.006) | 6% | +8 | −0.219 (0.019) | 29% | −541 |
| hero damage + damage taken + deaths | | | | −0.107 (0.017) | 65% | −213 |

Reading:
- **The win-relevant signal is in the fight metrics, and it is spread across them.** The seven fight metrics alone keep 96% of the scoreboard's value; dropping them loses half. No single metric is indispensable: the biggest single loss is kills (13%), then damage taken (11%), regen globes (9%) and hero damage (8%). Each metric carries some of the same underlying skill, so the others cover for any one.
- **Alone, hero damage is the most informative single metric** (40% of the value), then kills (38%), deaths (33%) and structure damage (30%). Protection, damage taken and healing alone carry little (6% to 12%). Damage taken matters in combination (11% lost when dropped) but not by itself: it is informative once hero damage says what role the player played.
- **The two tests mostly agree.** On the fit's own held-out LL, hero damage is the costliest single drop (−89), and the fight group is again the big one (−239 dropped, −57 kept alone). The prospective test is the one to trust for "win-relevant": it scores later games of held-out players.
- **Healing, protection, escapes and deaths add nothing at the margin** (0% to 1% each), though deaths alone carries 33%. The model could drop protection, healing and escapes with no measurable loss. Keeping them costs nothing either.

### B1.3 Nonlinear and interaction use of the metrics

- **Models.** Learned models of the scored slot's residual (y − WP − offset), fit on the fit-half players' slots after the cutoff and scored on the held-out half's. These are the same slots, splits and β fitting as B1.1.
- **Features**, all as of the cutoff:
  - the cell's 15 metric means (shrunk) and its metric count;
  - the player's 15-metric profile over all heroes;
  - games on the hero, play share, total games;
  - the cell and player win records;
  - an 8-dim hero embedding;
  - optionally the main model's estimate.
- **Fitting.** "ridge" is linear in these features (plus a hero bias). "MLP" is two 64-unit layers, so metrics can interact with each other and with the hero. Both use early stopping on a 10% player split.
- **Recalibration control.** These models learn from post-cutoff outcomes (of other players), so they can also correct any level shift since the cutoff. To keep the metrics comparison fair, each "with metrics" arm has a twin without metrics.

| arm | held-out gain ×1000 (SE) |
|---|---|
| main (linear factor model, β only) | +0.742 (0.038) |
| MLP, no metrics (records, counts, preference, hero) | +0.708 (0.034) |
| ridge, metric profile, no main | +0.829 (0.035) |
| MLP, metric profile, no main | +0.936 (0.036) |
| ridge, no metrics, + main | +0.981 (0.038) |
| MLP, no metrics, + main | +0.973 (0.039) |
| **ridge, metric profile + main** | **+1.141 (0.041)** |
| MLP, metric profile + main | +1.072 (0.041) |

| paired difference | ×1000 (SE) | reading |
|---|---|---|
| MLP profile − MLP no metrics | +0.228 (0.027) | the scoreboard's value inside a learned model, close to its +0.31 in the factor model |
| MLP profile − ridge profile (no main) | +0.107 (0.021) | without the factor model, nonlinearity helps |
| MLP profile + main − ridge profile + main | **−0.070 (0.025)** | with the factor model, nonlinearity does not help (it overfits slightly) |
| ridge profile + main − ridge no metrics + main | **+0.160 (0.018)** | the metric profile still holds signal the main model does not use |
| ridge no metrics + main − main | +0.239 (0.026) | recalibration on recent outcomes alone |

Reading:
- **No evidence for nonlinear or interaction effects beyond the factor model.** The MLP's edge over ridge disappears once the main model's estimate is a feature. The factor model's hero loadings already supply the hero × metric interactions the MLP would otherwise learn.
- **But the main model leaves linear scoreboard signal on the table.** A direct linear use of the per-hero metric means and the player profile adds +0.16 (SE 0.02) on top of the main estimate. That is half as much again as the scoreboard's whole contribution through the factor model (+0.31). The factor model passes 15 metrics through an overall-skill column plus 2 factors. P3_RICH_CF found per-hero loadings overfit, but a higher-rank shared metric channel, or a direct per-metric term in the combiner, was never tried on the full panel. That is the most promising next model change. It needs the same nightly-walk test before adoption, since this ridge was fit on post-cutoff outcomes.
- **Leak checks:** the state assertion above; features from rows before the cutoff only; targets after it; player-disjoint fit and test.

### B1.4 Does scoreboard-derived skill help cold start and the recommendations?

**In the part A backtest, yes, and most for never-played heroes.** <10-game candidates, four cutoffs, top third minus bottom third (pooled, player-bootstrap SE):

| candidates | outcome | main | main, no scoreboard | imp2 | impmet |
|---|---|---|---|---|---|
| never played (0 games) | first game (145,315 tried) | +4.06 (0.34) | +0.43 (0.33) | +3.34 (0.34) | +4.48 (0.34) |
| never played | games 1-10 (10,196 taken up) | +2.72 (0.45) | +0.51 (0.36) | +1.97 (0.37) | +2.63 (0.44) |
| never played | games 11-50 (3,404) | +1.61 (0.57) | +0.19 (0.51) | +1.05 (0.48) | +1.72 (0.49) |
| 1-9 games | first game after the cutoff (396,136) | +3.68 (0.20) | +2.13 (0.21) | +3.45 (0.19) | +3.85 (0.21) |
| 1-9 games | games 1-10 (28,354) | +2.23 (0.24) | +1.29 (0.23) | +2.03 (0.24) | +2.28 (0.24) |
| 1-9 games | games 11-50 (8,262) | +2.03 (0.33) | +2.14 (0.31) | +1.89 (0.35) | +1.79 (0.32) |

- **For never-played heroes, all of the main model's ranking power comes from the scoreboard.** Without it the same model ranks them no better than chance. The player's stat profile, mapped through hero-specific loadings, says which unplayed heroes suit them.
- **For heroes with 1-9 games, those few games' win records do as well as the scoreboard at steady state.**
- **The scoreboard-similarity recommender** (a per-hero regression on the stat profile, part A) is weaker than the main model on every outcome. It ties on games 1-10 (+1.7 within player) but barely predicts the first game (+0.8). The factor model's shared loadings are the better use of the same information.
- **This disagrees with the masked cold-start test (Cassia)**, where the main model scores R² 0.30 against imp2's 0.46 (P3_TAILS). Cassia hides every row of a veteran's rarely played 10-60 game hero and predicts its whole-cell level. The backtest instead ranks heroes the player actually tried next and scores the games that followed. For the product question ("which new hero should I try?"), the backtest is the direct test, and it favours the scoreboard model. Cassia stays useful as a calibration check on cell levels.
- **impmet**, the model with both preference and scoreboard, is the best ranker of never-played heroes for first games (+4.5), but no better at steady state, and it is slightly worse for prediction (−0.03 prospective, −56 held-out LL). Keep `full_ms` as the main model. impmet, or the main + preference rank average, is a reasonable choice for the "try next" list.


---

## B2. Replay timelines

Code: `p3_hr_tl_features.py`, `p3_hr_tl_analyze.py` (stages: reliability, redundancy, targets, prospective), `p3_hr_tl_prospective.py`. Results: `results/hr/tl_*.json`. Run on this box, cores 60-61, nice 19.

### B2.1 Data and checks

- **20,188 parsed timelines**, games from 2026-09-28 to 10-15, builds 98285 to 98348. Cohorts: stratified_b 14,613; p2_panel_v1 4,680; surprise_v1 508, plus 387 in both B and surprise. They hold 201,880 player-games from 47,339 players; 27,488 players have 2+ games, in 31,132 player×hero cells with 2+ games.
- **Sealed check.** Every id read was checked against `sealed_ids.txt`: overlap 0, asserted in code. Nothing under `hots-replays/sealed` was opened. The 4,998 sealed games in the DB were dropped from the prospective targets by id alone, without reading their rows.
- **Agreement with the DB.** On 193,830 shared player-games, every one of the 15 scoreboard stats in the replay equals Heroes Profile's value exactly. The one exception is CC time: HP's `time_cc_enemy_heroes` is always 0, and the replay's `TimeCCdEnemyHeroes` is populated.
- Stun, root and silence times are already in the DB (`stunning_enemies`, `rooting_enemies`, `silencing_enemies`), just not in the model's 15-metric export.
- Paper 3's tranche rules no longer apply (paper 3 was cancelled 2026-10-17; `training/replays/README.md`). Only the sealed rule stands.

### B2.2 Features

Clock zero for "before 10:00" is gates open. Positions are sampled every 15 s.

| feature | definition | coverage |
|---|---|---|
| deaths before 10:00 | count, games of 10+ min | 0.99 |
| deaths before the first map objective | first map event after gates | 0.83 |
| deaths in the enemy half / isolated / overextended, per min | death position on the core-to-core axis; isolated = no ally sample within 10 units within ±8 s; overextended = both | 1.00 |
| early takedown participation | share of the team's takedowns before 10:00 with the player among the killers | 0.99 |
| objective takedown participation | the same, within 30 s of a map objective event | 0.76 |
| enemy-half share, movement per min | from the position samples | 1.00 |
| merc participation | within 12 units of the own team's capture in the 20 s before it | 0.99 |
| APM: commands / key presses / camera moves | parser's `actions_per_min` | 1.00 |
| time dead share, minutes to first death | | 1.00 |
| CC, stun, root, silence time; teamfight damage, taken, healing; minion and creep damage | replay score screen, per min | masked per hero |
| XP lead and level lead at 10:00, first-merc lead | team-level | 0.99 |

Normalization follows the scoreboard's `a_shrink` card: per minute for totals, asinh, z within hero × outcome with hero × map cells shrunk (100 pseudo-games), clip at ±5, mask uninformative hero-feature pairs. Deviation: the population means and sds use all timeline games, including target-period games. They are population statistics only, and z is outcome-conditioned, so no game's own result enters; with a longer cohort they should be fit on earlier games.

### B2.3 Stable share of one game

Split-half reliability (odd vs even games per player×hero), the same estimator as `p3_rcf_data.reliability` and the scoreboard table in P2_NARRATIVE_FACTS section 4. 95% CIs from 200 player bootstraps. The scoreboard rows on these same 20K games reproduce the 9.4M-row E table within about 0.03, so the timeline numbers are comparable.

| per-game measure | player×hero (95% CI) | player | games until half is real | E table, player×hero |
|---|---|---|---|---|
| win minus the hero's 2.57 win rate | 0.004 (−0.004, 0.010) | 0.009 | ~250 | 0.013 |
| *scoreboard on the same games:* protection / self-healing | 0.39 / 0.37 | 0.17 / 0.12 | 1.6 / 1.7 | 0.40 / 0.39 |
| siege damage / damage taken / healing | 0.29 / 0.27 / 0.24 | | 2.5 / 2.8 / 3.1 | 0.28 / 0.28 / 0.23 |
| hero damage | 0.21 (0.20, 0.22) | 0.12 | 3.7 | 0.23 |
| deaths | 0.15 | 0.12 | 5.5 | 0.17 |
| kills / assists | 0.08 / 0.04 | 0.04 / 0.03 | 12 / 23 | 0.07 / 0.04 |
| *replay score screen, not in the export:* CC time | 0.40 (0.39, 0.41) | 0.10 | 1.5 | (0 in HP data) |
| root / silence / stun time | 0.35 / 0.33 / 0.19 | 0.18 / 0.15 / 0.05 | 1.8 / 2.0 / 4.2 | |
| minion damage / creep damage | 0.32 / 0.20 | 0.16 / 0.10 | 2.1 / 4.1 | |
| teamfight hero damage / taken / healing | 0.19 / 0.23 / 0.19 | | 4.3 / 3.3 / 4.4 | |
| ***timeline:* APM, camera moves** | **0.95 (0.90, 0.99)** | 0.93 | 0.06 | |
| **APM, commands** | **0.91 (0.88, 0.95)** | 0.83 | 0.09 | |
| **APM, key presses** | **0.88 (0.85, 0.92)** | 0.75 | 0.13 | |
| time dead share | 0.14 | 0.11 | 6.3 | |
| movement per min | 0.13 | 0.09 | 6.9 | |
| deaths before 10:00 | 0.11 (0.10, 0.12) | 0.08 | 8.1 | |
| merc participation | 0.11 | 0.06 | 8.2 | |
| deaths in the enemy half / isolated deaths | 0.09 / 0.09 | 0.07 / 0.07 | 9.8 / 9.7 | |
| minutes to first death | 0.08 | 0.06 | 11 | |
| enemy-half share / overextended deaths | 0.07 / 0.07 | 0.05 / 0.06 | 13 / 13 | |
| deaths before the first objective | 0.05 | 0.03 | 18 | |
| early takedown participation | 0.04 | 0.03 | 25 | |
| objective takedown participation | 0.02 | 0.02 | 49 | |
| XP lead / level lead at 10:00, first-merc lead (team) | 0.006 to 0.009 | 0.006 to 0.009 | 100+ | |

**How much of each timeline trait the scoreboard already holds.** 10,869 players with 6+ timeline games. Scoreboard means on even games predict the feature's odd-game mean (5-fold ridge); share = out-of-sample R² over the half-mean reliability (`tl_redundancy.json`).

| feature | share of its stable part predicted by the scoreboard |
|---|---|
| time dead / first death / deaths before 10:00 | 0.49 / 0.48 / 0.47 |
| isolated or enemy-half deaths / overextended deaths | 0.45 / 0.38 |
| merc participation / enemy-half share | 0.32 / 0.29 |
| early takedown participation | 0.24 |
| movement per min | 0.04 |
| APM (all three) | 0.00 to 0.01 |

Reading: the early-death features are a slice of the end-of-game death count with fewer events, so they are about two thirds as reliable per game and half redundant. APM is a near noise-free personal trait (two games pin it down) and almost entirely new information. Team-level leads are lobby noise.

### B2.4 Incremental value over the scoreboard, prospective (early and underpowered)

- **Targets:** every 2.57 Storm League game in the local research DB with 10 player rows, sealed games excluded. That is 66,177 games (661,770 slots), 2026-09-28 to 10-22. Outcome: team 0 wins.
- **History:** each player's games on strictly earlier UTC days only. 417,576 target slots (37,068 players) have 1+ earlier timeline game, median 4. With p2_panel_v1 history only, 226,556 slots (13,321 players), median 2. 514,998 slots have earlier DB 2.57 games (scoreboard), median 8.
- **Base model:** team differences of the hero's 2.57 win rate on earlier days, the player's 2.57 win record on earlier days, and the main model's pre-2.57 estimate (offset + posterior mean, frozen 2026-09-28; `p3_hr_state.py`, covers 60% of slots). No game's own MMR anywhere.
- **Model:** logistic on team differences, fixed L2 penalty. Player means are shrunk with k = (1 − r)/r pseudo-games. Two schemes: 10-fold CV over games, and a forward split (fit before 2026-10-09, test on the 35,317 games from then on). Paired SEs from a day-block bootstrap.

Log-loss gain ×1000 (SE). Base log loss 0.672.

| step | CV, all games | CV, games with 6+ players with timeline history (42,890) | forward split |
|---|---|---|---|
| + scoreboard from the timeline games only | +2.87 (0.33) | +3.87 (0.40) | +2.78 (0.41) |
| + timeline features, same games | +0.41 (0.13) | +0.65 (0.20) | +0.08 (0.21) |
| + scoreboard from all earlier DB 2.57 games | **+7.61 (0.67)** | +8.98 (0.61) | **+9.38 (0.58)** |
| + 8 extra score-screen stats (CC, stun, root, silence, teamfight, minion, creep) | +0.12 (0.10) | +0.12 (0.14) | −0.05 (0.24) |
| **+ timeline features, over the full scoreboard** | **+0.24 (0.10)** | +0.45 (0.17) | **−0.07 (0.19)** |
| + team-level timeline features | +0.04 (0.06) | −0.00 (0.08) | −0.08 (0.15) |
| p2_panel_v1 history only: + timeline over the full scoreboard | −0.04 (0.07) | −0.11 (0.22), 18,383 games | −0.62 (0.14) |

Single features over the full scoreboard (CV, all history): key presses per minute +0.24 (0.08), movement per minute +0.11 (0.06), camera APM +0.03 (0.04); every death, positioning, participation and merc feature between −0.01 and +0.01 (SE 0.01 to 0.02). Without the main-model baseline the picture is the same (+0.22 CV, +0.01 forward).

Reading:
- The timeline features carry under 3% of what the scoreboard adds on the same games.
- Only APM-type mechanics show a cross-validated increment, which matches the redundancy table.
- The forward split loses even that. Coefficients fit on 11 days do not transfer to the next 13 while timeline coverage is still filling in. The panel-only forward −0.62 is overfitting on thin history.
- **Power.** The minimum detectable effect (80% power, 5% two-sided) is about 2.8 × SE: 0.3 points under CV and 0.5 forward. A plausible true effect, from the reliabilities and the overlap with the scoreboard, is 0.1 to 0.3 points (0.02 to 0.05 points of accuracy). Not detectable yet.

### B2.5 Recommendation

| feature | decision | why |
|---|---|---|
| APM: commands, key presses, camera moves | **build now**, as one more metric channel with as-of means from timeline games and a player-level loading | 0.88 to 0.95 stable per game, about 1% overlap with the scoreboard, the only positive single-feature signal (+0.24, SE 0.08). Expect a small lift; frame it as mechanism evidence. |
| movement per min | build with APM (cheap, same channel) | 0.13 stable, 4% redundant, +0.11 (0.06) |
| stun, root, silence time | **add to the scoreboard export now** | in the DB for every corpus game, 0.19 to 0.35 stable, near-zero cost; increment so far +0.12 (0.10) |
| replay CC time | carry, do not rely on | 0.40 stable but only on timeline games |
| deaths before 10:00 / before first objective, overextended or isolated deaths, early and objective takedown participation, enemy-half share, merc participation | **wait** | 0.02 to 0.11 stable, 24% to 49% redundant with the scoreboard, each 0.00 ± 0.01 |
| XP lead, level lead, first-merc lead | drop as player traits | team-level, under 0.01 stable; lobby features only |

**How long to wait.** Panel-only history is about 4,700 timelines today (median 2 per player). At about 8,000 panel games a week, four weeks gives about 32K to 37K, 7 to 8 times as many player-games. That cuts the forward SE to about 0.05 and the minimum detectable effect to about 0.15, enough to confirm or rule out a +0.2 to +0.3 mechanics effect. Rerun `p3_hr_tl_analyze.py` (features, reliability, prospective) around 2026-11-18, with every choice frozen beforehand and normalization statistics fit on games before the targets.

**Caveats.**
- The all-history arms mix stratified_b and surprise_v1 (selected by their own cohort rules) with the panel. The panel-only arm is the clean design, and it shows no gain.
- All targets fall in 25 days of one patch.
- Position features rest on 15 s samples and a core-to-core axis: overextension is a proxy, not vision or rotation data.


---

## Files

All in `training/personalization/`.

**Code**
- `p3_hr_common.py`: compact history-row cache (`cache/hr/aug.npz`, gitignored), as-of state, all-hero posteriors.
- `p3_hr_backtest.py`: part A. `--pooled --max-n 10|5` writes `results/hr/pooled_n{10,5}.json`; the single-cutoff run writes `backtest_t0_n5.json` (the interim table).
- `p3_hr_traj.py`: A5, `results/hr/trajectories.json`.
- `p3_hr_example.py`: A6, `results/hr/example_player.txt` (no ids).
- `p3_hr_ablate.py`: B1 refits on the 5090, params in `results/hr/ablate/abl_*.{npz,json}` plus `impmet.{npz,json}`.
- `p3_hr_sbeval.py`: B1 prospective test, `results/hr/sbeval.json`.
- `p3_hr_state.py`: the 2026-09-28 main-model state for B2's baseline (`cache/hr/state_0928.npz`, gitignored).
- `p3_hr_tl_features.py`, `p3_hr_tl_analyze.py`, `p3_hr_tl_prospective.py`: B2, `results/hr/tl_*.json`.

**Gitignored, with player keys:** `cache/hr/` (aug cache, per-candidate records, timeline rows and targets).

**Compute**
- This box: nice 19, cores 56-57 (parts A and B1) and 60-61 (B2), about 1.5 hours in total.
- 5090 GPU: 39 refits (38 ablations plus impmet) as hotsjob jobs `hr_ablate*` and `hr_impmet`, under 12 GB host RAM, about 3 hours. Code ran from a separate copy in `~/hots/hr/` so the shared repo copy was not touched.
- Not used: the 3090 and the 3080.
- Not touched: the other agents' processes and files. One exception: `p3_hr_common.build` read the shrinkage study's `cache_full/rcf/aug_merged.npz` through `p3_rcf.load_aug`, read-only.

**Leak and data checks**
- Every as-of state uses history rows with day < cutoff. `p3_hr_sbeval.py` asserts that rewriting all later rows changes nothing.
- The cutoffs before 2026-02-10 use only held-out-half players.
- Sealed replays: zero overlap, asserted (B2).
- Local research DB only, read-only.

