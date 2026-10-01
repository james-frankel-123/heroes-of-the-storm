# P3: the personalized drafter and the DraftRec-style contrast (2026-09-30)

Revised 2026-10-01 after the consolidated audit; changes are logged in `P3_AUDIT_FIXES.md`. Every drafter here runs under one protocol (`p3_fix_drafters.py`): pools are the heroes a player played on earlier days, no real bans are replayed, the personal term and the combiner use counts from earlier days, and the imitation features use earlier days only.

## Design (one unified model, no model of "everyone")

The value of a finished draft, from team 0's side, is:

V = sigmoid(b0 + b1·logit WP_pop + b2·(S_0 − S_1) + b3·(O_0 − O_1))

- **WP_pop:** paper 2's drift-aware d2c_cumprev ensemble (3 seeds, swap-symmetrized), with features from cumulative statistics through the previous build. My batched evaluator reproduces the cached scores of real games to 1e-7.
- **S_t:** the sum over team t's players of s(p, h). This is the experience offset plus the posterior mean of the phase-1 "+CF rank 2" kernel (pooled skill and the similarity embedding), with state and counts through the previous day.
- **O_t:** the number of off-role players on team t (fine role under 10% of 50+ earlier games).
- **b:** the V1-fit combiner, refit on lag-1 counts (b1 = 1.00 on the WP logit, b2 = 3.61 on summed skill, b3 = −0.055 per off-role player).
- **Population drafter:** values drafts by WP_pop alone.

**Lobbies.** 5,724 held-out V2 games (2026-03-29 to 05-22) in which all ten players had 50+ earlier games:

- 1,000 are used for full simulations and all 5,724 for the real-state rankings.
- The player making each pick is the one who played the hero picked at that step. This pick order is fixed by the lobby before the draft, so it is not outcome information.
- A player's available heroes are the ones he played on earlier days; the median pool is 34 heroes. If the pool is exhausted, any free hero is allowed.
- Bans are not replayed. In the one-step drafter every ban is sampled from the GD model given the draft state.

**Search.** Search-only with a behavioral prior:

- At each decision the candidates are the acting player's free pool heroes, capped to the union of the top 15 by GD probability and the top 15 by the player's personal term (21.6 candidates on average).
- Each candidate gets 4 rollouts that complete the draft with the paper-1 generic-draft (GD) models, restricted to each picking player's pool.
- The personalized and population values are computed on the same rollouts, so every pick comparison is paired.

I did not put personal terms into the CUDA MCTS kernel for this part: it hard-codes the paper-1 WP and its lookup tables. This harness runs on 4 CPU cores in about 24 minutes. No GPU was used.

## Headline numbers (model-internal)

**How often and how much the recommendation changes** (paired, same state, same rollouts):

| trajectory | decisions | picks that differ | when they differ: gain in V (pp) | when they differ: change in WP_pop (pp) |
|---|---|---|---|---|
| (a) vs GD opponent | 5,000 | 51% | +3.4 (3.3, 3.5) | −2.9 (−2.9, −2.8) |
| (a) vs imitation opponent | 5,000 | 51% | +3.6 (3.4, 3.7) | −3.0 |
| (b) self-play | 10,000 | 52% | +3.6 (3.5, 3.7) | −3.0 |

- About half of all personalized picks differ from the population pick.
- When they differ, the personalized pick trades about 3pp of population WP for about 3.4 to 3.6pp of personalized value.
- In 23 to 25% of decisions the personalized gain exceeds 3pp.

**Whole drafts** (controlled team, paired by lobby):

| opponent | personalized drafter V | population drafter V | real draft V | difference in V | difference in WP_pop |
|---|---|---|---|---|---|
| GD | 0.738 | 0.689 | 0.503 | +4.9pp (4.3, 5.6) | −2.7pp (−3.2, −2.0) |
| imitation model | 0.665 | 0.599 | 0.503 | +6.6pp (5.9, 7.3) | −2.5pp (−3.1, −1.9) |

The imitation opponent is a tougher opponent than GD: V is lower by about 7 to 9pp for both drafters. These are the model's own valuations of its own drafts; nothing here is checked against outcomes.

## Team-level calibration at the real states (not evidence for the drafter)

In the real drafts of the 5,724 held-out lobbies (11,448 teams), each real pick is placed in both drafters' rankings at the real draft state (percentile among candidates). A team's agreement is the mean percentile over its picks. One definition is used for every drafter in this file: personalized agreement, population agreement, and the personal component = personalized minus population agreement. The realized outcome is the team residual y − WP_pop of the real game. The predicted gap is V − WP_pop of the same real draft under the skill model. CIs are game bootstraps: the two teams of a game are resampled together. Script: `p3_fix_realized.py`, `fix/p3_fix_realized.json`.

Why this is calibration: agreement ranks the real picks by V, which contains the skill term s, and the outcome is the residual that s was fit to predict. Teams that agree with the drafter are teams the skill model already favors, so a high-agreement quintile re-tests the skill model. The table therefore puts the model's own predicted gap next to each realized spread; only the remainder could speak for the drafter's ranking.

Q5 minus Q1 of agreement (pp):

| agreement measure | realized | predicted by the skill model | remainder |
|---|---|---|---|
| personalized drafter ranking (V) | +8.5 (6.0, 11.2) | +7.1 (6.6, 7.6) | +1.5 (−1.1, +4.1) |
| population drafter ranking (WP_pop) | +1.9 (−0.7, 4.6) | +0.0 (−0.5, 0.6) | +1.8 (−0.8, +4.5) |
| personal component | +11.3 (8.3, 14.2) | +9.8 (9.2, 10.3) | +1.5 (−1.3, +4.5) |

- The realized spread is what the skill model predicts for those drafts. Nothing is left over for the drafter's ranking once the model's own forecast is subtracted.
- Regressing the realized team residual on agreement and the predicted gap: the predicted-gap slope is 1.00 (0.88, 1.13), and agreement adds +4.0pp per unit (−2.3, +10.1).
- So the value function's personal terms are calibrated on held-out games, at team level. That is real evidence for the inputs the drafter uses. It is not evidence that the drafter's own choices win more, because the real teams did not play the drafter's picks.

## Collapse: does it degenerate to "everyone plays their 3 best heroes"?

This is the proposal's experiment #1. Setup: 300 players, each placed into 30 random real partial drafts at a team-0 pick step.

| per player (median) | personalized drafter | population drafter | real play |
|---|---|---|---|
| effective hero pool (exp entropy, 30 draws) | 10.4 | 12.7 | 11.3 (30 sampled games) / 18.1 (all games) |
| distinct heroes in 30 decisions | 13 | 16 | |
| share on the player's single most-played hero | 6.7% (mean 11%) | 3.3% (mean 4%) | 18% (mean 21%) |
| share on the player's top-3 most-played heroes | 23% (mean 29%) | 10% (mean 13%) | 41% (mean 43%) |

**No collapse.**

- The personalized drafter uses a slightly narrower pool than the population drafter (10.4 vs 12.7 effective heroes). It is about as wide as the player's own play at the same sample size (11.3).
- It concentrates less on the player's favorite heroes than the player does: 29% of its picks go to the player's top-3 heroes, against 43% of his real games.
- Skill estimates are shrunk (the typical per-hero spread is 2 to 4pp) while draft context moves WP_pop by several pp, so the draft keeps most of the weight.

Figure: `fig_dr_collapse.png`.

## Emergent meta (1,000 lobbies, both teams)

Figure: `fig_dr_meta.png`. Panels: personalized vs population self-play pick shares; personalized self-play vs real; role shares.

| | personalized self-play | population self-play | real drafts (same lobbies) |
|---|---|---|---|
| effective number of heroes (exp entropy) | 69.0 | 65.0 | 73.5 |
| share of the 10 most-picked heroes | 31.0% | 32.5% | 25.7% |
| correlation of hero shares with real | 0.47 | 0.38 | |

- **Correlation between the two self-play metas:** 0.97. Personalization shifts the meta at the margin rather than rewriting it.
- **Both drafters concentrate on WP-favored heroes.** Auriel (4.6% personalized, 5.1% population, 2.2% real), Rehgar (4.2%, 4.6%, 2.3%), Tyrael, Illidan, Leoric, Hogger and Kerrigan are all over-picked relative to real drafts. By role, both drafters pick more bruisers (25% vs 18% real) and melee assassins (12 to 13% vs 6%) and fewer ranged assassins (27 to 28% vs 35%) and tanks (13 to 14% vs 19.5%).
- **Heroes that rise when drafters know who is playing** (personalized vs population self-play share): Hanzo (0.64% vs 0.20%; real 2.4%), Genji (0.29 vs 0.07), Chen (0.86 vs 0.39), Yrel, Sylvanas (1.15 vs 0.66), The Lost Vikings, Anduin (1.25 vs 0.79), Muradin, Nova, Diablo.
- **Heroes that fall:** Malfurion (0.92 vs 1.45), Ragnaros, Lt. Morales, Nazeebo (1.45 vs 1.96), Jaina, Azmodan, Artanis, Zagara, Sgt. Hammer, The Butcher.
- **Reading:** the risers are the high-execution specialist heroes that sit at the top of the skill embedding's first axis (Hanzo, Genji, Yrel, Chen), plus popular heroes that real players are good at. The fallers are "generically strong for anyone" picks the population model likes. Knowing who plays moves the meta toward real play (correlation 0.38 to 0.47) by giving specialists their heroes.

## Off-role handling

Share of picks that land on a player whose fine-role share for that hero is under 10%:

| | personalized self-play | population self-play | real drafts |
|---|---|---|---|
| off-role picks | **17.3%** | 32.7% | 15.8% |
| mean role share of the assigned player | 27.4% | 21.5% | 30.3% |

The population drafter ignores who is playing and puts a third of heroes on players who rarely play that role. The personalized drafter assigns roles almost as comfortably as real players choose for themselves.

## Item 2: DraftRec-style imitation model (`p3_dr_imitation.py`)

**Model.** A conditional logit over untaken heroes. The inputs are the GD log-probability given the draft state, plus the player's history: log(1 + games on the hero), a never-played flag, EWMA pick shares at 20 and 100 games, days since last played, and fine-role share. Fit on 250K V1 picks, tested on 250K V2 picks. Every history feature uses games on earlier days only (`p3_fix_counts.recency_features_l1`), the same contract as the skill model.

| model (V2 picks) | top-1 | top-3 | top-5 | top-10 | log-lik |
|---|---|---|---|---|---|
| **imitation (GD + personal history)** | **31.0%** | 53.4% | 64.1% | 77.4% | −2.67 |
| GD alone (non-personal) | 9.1% | 21.2% | 30.3% | 46.8% | −3.78 |
| frequency alone (EWMA-20 share) | 25.9% | 46.3% | 56.6% | 69.5% | −3.52 |
| personal history alone (fitted) | 26.1% | 46.3% | 56.6% | 69.8% | −2.99 |
| uniform over the player's pool | 7.2% | 19.3% | 30.5% | 49.3% | −7.72 |

- Top-1 accuracy by the team's pick position: 39% on the first pick, falling to 27 to 32% on later picks. GD alone gets 8 to 12%.
- This is held-out prediction of real player actions, so it is evidence for the imitation model as an opponent model, independent of any outcome.
- Personal history carries most of the signal, and the draft state adds 5 points of top-1 on top.

**Imitation vs outcome-based recommendations** (57,240 real picks at the real states):

| | share |
|---|---|
| real pick equals imitation top-1 | 33.0% |
| real pick equals personalized (outcome) drafter top-1 | 12.2% |
| real pick equals population drafter top-1 | 7.6% |
| imitation top-1 equals personalized top-1 (same candidate set) | 14.2% |
| personalized top-1 equals population top-1 | 48.1% |

The two personal systems rarely agree. Imitation recommends what the player usually plays; the outcome-based drafter recommends what is predicted to win with this player in this draft.

**Team residual (won − WP_pop) by which system the real pick matched** (calibration, not a test of either recommender: a pick that matches the outcome drafter is one the skill model already rates highly, so its residual is partly what the skill model predicts):

| the real pick matched | picks | team residual (pp) |
|---|---|---|
| both | 3,738 | +3.2 (1.6, 4.8) |
| outcome-based drafter only | 3,264 | +2.5 (0.7, 4.2) |
| imitation only | 15,206 | +1.5 (0.7, 2.2) |
| neither | 35,032 | −1.2 (−1.7, −0.7) |

Picks matching either recommender sit above the population expectation and picks matching neither below it. The imitation-only class is the cleaner signal, since the imitation model does not use outcomes; P3_EXTENSIONS section 5(d) separates it from the skill model's predicted gap.

The imitation model also serves as the personalized opponent model in version (a). It makes the opponent about 7pp tougher in V than GD, and the personalized drafter's predicted edge over the population drafter grows (+6.6pp vs +4.9pp).

## Caveats

- **Predicted gains are model-internal.** The +3.4 to +3.6pp per differing pick and +5 to +7pp per draft are the personalized model's own claims. No outcome data test them: the real teams did not play the drafter's picks. What the outcome data show is that the value function's personal terms are calibrated on held-out games (team level, slope 1.00).
- **Pool proxy.** "Played before" ignores heroes owned but never played, so it understates true pools. It also plays in favor of the drafter's comfort.
- **Search depth.** One-ply search with 4 behavioral rollouts per candidate. Deeper search (MCTS) might exploit WP_pop further; the population self-play concentration on Rehgar and Auriel is a sign of that. Self-play uses GD rollouts for both sides.
- **Bans.** Bans are sampled from GD, not optimized.
- **Availability.** State and pools use every game eventually uploaded with an earlier date, not only what had been uploaded by draft time.

## Files (training/personalization/)

- `p3_fetch_drafts_post.py`: full draft orders, maps and tiers for post-cutoff games.
- `p3_dr_core.py`: lobbies, personal tables (s, off-role, pool, counts for every candidate), the GD policy, and the WP evaluator.
- `p3_dr_imitation.py`: imitation model (`results/p3_dr_imitation.json`).
- `p3_dr_drafter.py`: drafter runs, versions (a) and (b), real-state rankings and collapse contexts (`cache/dr_runs.pkl.gz`).
- `p3_dr_analyze.py`: all tables above (`results/p3_dr_analyze.json`, `fig_dr_meta.png`, `fig_dr_collapse.png`).
- `p3_fix_drafters.py` (the protocol above, applied to the modules above at run time; `tables`, `onestep`, `analyze_onestep` → `cache/fix/`, `results/fix/p3_dr_analyze.json`, `results/fix/p3_dr_imitation.json`), `p3_fix_realized.py` (calibration table, `results/fix/p3_fix_realized.json`; `--old` gives the same for the pre-audit runs, `p3_fix_realized_old.json`).

# Personalized MCTS (2026-09-30)

## Kernel (`cuda_personal/`)

**Where it lives.** A copy of the guarded overfit2026 kernel (`cuda_ofit`); `training/cuda_mcts` is untouched. `cuda_personal/ref/` is an unmodified copy of `cuda_ofit`, built under another module name as the reference. Built with nvcc 12.9 for sm_120. Source: `personal_kernel.cu` and `personal_bindings.cpp`.

**What changed** (all per episode, so one launch serves up to 256 lobbies):

- **Leaf value.** The base is the population WP (the 3 d2c_cumprev seeds as a mean-logit net per orientation, then swap-symmetrized, from our side). On the logit scale it adds b2·(S_our − S_opp) + b3·(O_our − O_opp), with S and O summed over the ten slots' per-hero skill and off-role tensors (10×90 each). Coefficients b1 = 1.0006, b2 = 3.613 and b3 = −0.055 come from the V1 combiner on counts from earlier days; b0 is set to 0 (fitted value 0.0010).
- **Pick step to player.** The kernel uses Storm League order: bans 0,1,0,1; picks 0,1,1,0,0; bans 1,0; picks 1,1,0,0,1, with kernel team 0 picking first. Real lobbies where team 1 picked first are relabeled. The player at a pick step is the player who made that pick in the real draft (`draft_order`, the order fetched by `p3_fetch_pickorder.py`). Slots 0 to 4 are kernel team 0's players in pick order, and 5 to 9 are team 1's.
- **Pools and bans.** Every pick is restricted to the acting slot's pool (heroes played on earlier days), falling back to any free hero if the pool is exhausted. The kernel can force given bans; the runs below force none, so our bans are searched and the other team's come from the GD.
- **Decide-only mode.** Replays a prefix of actions, then runs one search for the acting team and reports root visits and Q. This is used for the real-state and collapse checks.
- **Self-play.** Every non-forced step is searched from the acting team's side.
- **Imitation opponent** (main draft only): softmax(w0·log p_GD + personal bias of the acting slot).
- **Capacity guard.** The tree-capacity guard (4,096 nodes, with a bound check before every expansion) is kept. MAX_OUR_TURNS is raised to 16 for self-play.

**Verification** (`p3_mcts_verify.py`, `results/p3_mcts_verify.json`):

- **Leaf values.** 300 real held-out drafts with realistic personal tensors (rows of the personal tables), random slot assignments and random perspective. Kernel vs a float64 Python mirror: population WP max |Δ| 5.2e-7 (mean 1.2e-7); personal value max |Δ| 5.8e-7 (mean 1.2e-7).
- **Population identity.** With personal terms zeroed (b = 0, 1, 0, 0, no pools, no forced bans), and separately with personal valuation switched off, the kernel reproduces the reference population kernel exactly: 64 episodes each at 100 and 400 sims, with identical win probabilities (bitwise), final drafts and root visit distributions.
- **Guard.** Nodes used peak at 1,218 (100 sims), 2,662 (400 sims) and 3,528 (1,000 sims), with no capacity hits. At 3,000 sims without pools the tree fills (4,034 nodes) and the guard blocks 798 expansions in 11 of 16 episodes. The runs below use at most 1,500 sims with pools.
- **Ensembling.** The kernel's mean-logit ensemble and the one-step drafter's mean-probability ensemble differ by at most 0.0017 WP (mean 0.0001).

## Protocol

- **Search.** Search-only: the prior is the outcome-free behavioral-cloning policy distilled from GD (`overfit2026/models/bc_prior.pt`). The in-tree opponent and all rollouts use GD, one of the five paper-1 models per launch, cycled; paired personalized and population runs share the model and seed. The root pick is the argmax of visits, with c_puct 2.0 and no Dirichlet noise.
- **Data.** Same lobbies, controlled teams and players as the one-step run: 1,000 full lobbies, the first 2,000 of the 5,724 realized lobbies (4,000 teams) for real-state searches, and 300 collapse players (30 new contexts each).
- **Main level.** 400 sims per decision. The sims curve uses 25, 100, 400 and 1,500 sims on the same lobbies.
- **Compute.** GPU 3, one process, on 2 host threads.

## MCTS vs one-step vs population

| metric | one-step personalized | MCTS personalized (400 sims) | MCTS population (400 sims) |
|---|---|---|---|
| picks that differ from the population drafter at the same state | 51% | 39% | |
| when they differ: personal gain / population WP cost (pp) | +3.4 / −2.9 | +3.9 (3.7, 4.1) / −1.4 (−1.6, −1.2) | |
| whole draft vs GD: gain in V over the population drafter | +4.9 (4.3, 5.6) | +6.2 (5.4, 6.9) | |
| whole draft vs GD: change in WP_pop | −2.7 | −1.1 (−1.8, −0.5) | |
| whole draft vs GD: mean V of the controlled team | 0.738 | 0.687 | 0.625 |
| whole draft vs imitation opponent: gain in V / change in WP_pop | +6.6 / −2.5 | +7.7 (6.9, 8.5) / −0.4 (−1.1, 0.2) | |
| calibration, personalized agreement Q5 − Q1: realized / predicted / remainder | +8.5 / +7.1 / +1.5 (−1.1, +4.1) | +4.1 / +4.0 / +0.1 (−4.0, +4.9) | |
| calibration, personal component Q5 − Q1: realized / predicted / remainder | +11.3 / +9.8 / +1.5 (−1.3, +4.5) | −0.1 / +5.9 / −6.0 (−11.1, −1.0) | |
| collapse: effective pool per player (median) | 10.4 | 8.9 | 10.1 |
| collapse: share of picks on the player's top-3 heroes (mean) | 29% | 25% | 13% |
| self-play: effective number of heroes (real 73.5) | 69.0 | 41.3 | 38.6 |
| self-play: share of the 10 most-picked heroes (real 25.7%) | 31.0% | 50.2% | 52.0% |
| self-play: correlation of hero shares with real drafts | 0.47 | 0.65 | 0.62 |
| off-role picks, self-play (real 15.8%) | 17.3% | 20.5% | 33.6% |
| off-role picks, vs GD (controlled team) | 26.0% | 28.1% | 34.4% |
| real picks equal to the drafter's top-1 | 12.2% | 19.5% | 13.6% |
| imitation top-1 equal to the drafter's top-1 | 14.2% | 31.7% | |

The one-step calibration rows use all 11,448 teams, the MCTS rows the 4,000 teams of the 2,000-lobby subset. The population one-step drafter's off-role rates are 32.7% (self-play) and 34.1% (vs GD).

**Reading:**

- **Cheaper personal gains.** MCTS changes fewer picks than one-step (39% vs 51%). The picks it changes buy more personal gain at about half the population-WP cost (−1.4pp vs −2.9pp per pick; −1.1 vs −2.7 per draft). Search finds personal picks that also hold up in the draft.
- **Stronger opponent-model effect.** Against the imitation opponent, the personalized MCTS drafter's predicted edge is +7.7pp at almost no WP cost.
- **Calibration.** For both drafters the realized spread across agreement quintiles is what the skill model predicts for those drafts (remainders within their CIs), except the MCTS personal component, whose realized spread is 6pp below the model's prediction (CI −11.1 to −1.0). That is one of about twenty such remainders in this file, P3_PERSONAL_GD and P3_EXTENSIONS section 5, and no other excludes zero.
- **Off-role handling.** About 20% of personalized MCTS self-play picks go to off-role players, against 33 to 34% for population drafters and 16% in real drafts.
- **No collapse** onto comfort heroes. The effective pool is 8.9 heroes (population MCTS 10.1, real play 11.3 at the same sample size), and only 25% of picks go to the player's top-3 heroes (43% in real play).

## Emergent meta under MCTS

Figure: `results/fix/fig_mcts_meta.png`.

- **Concentration.** With the behavioral prior and argmax at the root, MCTS self-play concentrates on the prior's favorite heroes far more than one-step search or real drafts. The effective number of heroes is 41.3 (personalized) and 38.6 (population), against 73.5 in real drafts. The top 10 heroes take 50 to 52% of picks. The leaders are Valla (7.7% personalized, 8.2% population, real 2.9%), Anduin (7.4%), Muradin (7.2%), Rehgar (5.5%), Falstad, Anub'arak, Leoric, Nazeebo, Tychus and Li-Ming.
- **Role mix.** It is close to real: tanks 21%, bruisers 18%, healers 21%, ranged 35%, melee assassins 5%. Hero shares correlate 0.65 with real (one-step 0.47), because the prior is behavioral.
- **Personalization widens the pool slightly** (41.3 vs 38.6 heroes) and moves the meta toward real play (correlation 0.65 vs 0.62).
- **Risers under personalized MCTS:** Zagara (0.33% vs 0.09%), Maiev (0.60 vs 0.25), D.Va, Junkrat, Hanzo (0.67 vs 0.32), Alarak (1.39 vs 0.76), Tyrael, Sylvanas, Gul'dan, Imperius.
- **Fallers:** Varian (1.10% vs 1.59%), Mephisto, Rexxar, Ragnaros, Deckard, Sonya, Azmodan (1.82 vs 2.31), Jaina, Arthas, Tracer.
- **The one-step conclusion holds.** Knowing who plays lifts specialist and high-execution heroes (Hanzo, Maiev, Alarak, D.Va, Sylvanas) and trims generic picks.

## Sims curve: predicted value and calibration by search effort

Figure: `results/fix/fig_mcts_sims_curve.png`. Predicted rows use the same 1,000 lobbies; calibration rows the same 2,000-lobby subset of 4,000 teams.

| search | predicted whole-draft gain in V vs GD (pp) | change in WP_pop (pp) | personalized agreement Q5 − Q1: realized / remainder (pp) | personal component Q5 − Q1: realized / remainder |
|---|---|---|---|---|
| MCTS 25 sims | +2.1 (1.4, 2.9) | −0.1 | −2.2 / −2.3 (−6.9, +3.2) | −0.5 / −0.4 (−5.4, +4.3) |
| MCTS 100 sims | +4.2 (3.4, 5.0) | −0.2 | +0.4 / −1.8 (−6.4, +2.8) | −0.3 / −1.6 (−6.5, +3.1) |
| MCTS 400 sims | +6.2 (5.5, 6.9) | −1.1 | +4.1 / +0.1 (−4.0, +4.9) | −0.1 / −6.0 (−11.1, −1.0) |
| MCTS 1,500 sims | +7.7 (7.0, 8.4) | −2.7 | +5.8 / −0.8 (−5.5, +3.8) | +7.9 / −0.3 (−5.5, +4.9) |
| one-step (≈86 rollouts per decision) | +4.9 (4.3, 5.6) | −2.7 | +8.5 / +1.5 (−1.1, +4.1) (all teams) | +11.3 / +1.5 (−1.3, +4.5) |

Reading:

- **Predicted gains rise steadily with search:** +2.1pp at 25 sims, +4.2pp at 100, +6.2pp at 400 and +7.7pp at 1,500. MCTS passes one-step between 100 and 400 sims.
- **Realized spreads follow the skill model's prediction at every level.** At low sims the root visit ranking is mostly the behavioral prior, so agreement with it carries little of the personal value and the predicted and realized spreads are both small. At 1,500 sims both are large (+5.8 realized, +6.6 predicted). The remainders are zero within their CIs except the 400-sim personal component noted above. Whether deeper search buys value beyond what the skill model predicts cannot be read from these rows.
- **Paper 1's warning applies to the predicted column.** The predicted gains are the model's own valuations of its own drafts, and they keep rising with search. More sims beyond 1,500 would need a larger tree (the 4,096-node guard starts to bind around 2,000 to 3,000 sims).

## Imitation vs outcome under MCTS

At the real states (20,000 real picks of the 2,000-lobby subset), the personalized MCTS top-1 matches the real pick 19% of the time (one-step 12%) and matches the imitation model's top-1 32% of the time (one-step 14%). The behavioral prior pulls search toward what players usually pick.

Team residual by which system the real pick matched (calibration, as for one-step):

| the real pick matched | picks | residual (pp) |
|---|---|---|
| both | 2,538 | +1.7 (−0.4, 3.6) |
| imitation only | 4,098 | +1.9 (0.4, 3.4) |
| personalized MCTS only | 1,355 | +2.2 (−0.4, 4.9) |
| neither | 12,009 | −1.2 (−2.1, −0.3) |

Picks matching either system sit 1.7 to 2.2pp above the population expectation and picks matching neither 1.2pp below; the classes do not separate from each other at this sample size.

## Stretch: player-conditioned policy trained by self-play (not started; cost estimate)

- **Kernel work.** The policy backbone takes a fixed 290-d state, and the acting slot's personal vector would have to enter it. Threading that input through the backbone and every expansion and re-verifying is a new kernel variant, about one working day.
- **Self-play data.** Measured throughput is about 100 searched decisions per second at 400 sims on the shared GPU 3. That makes about 5.5 GPU-hours per 200K self-play drafts, and paper-1-style training needs many iterations: several GPU-days under the 2-process cap.
- **Cheaper alternative.** Distill the root visit distributions already produced here into a player-conditioned policy by supervised learning, about one GPU-hour. It would serve as a better prior than the behavioral one (the low-sims curve shows the prior dominates below about 400 sims), but it would not test self-play training. Awaiting a decision before either is started.

## Files (training/personalization/)

- `cuda_personal/`: `personal_kernel.cu`, `personal_bindings.cpp`, `setup.py`, and `ref/` (the unmodified reference kernel).
- `p3_mcts_core.py`: host side, draft-order and slot conventions, the Python leaf mirror.
- `p3_mcts_verify.py`: verification (`results/p3_mcts_verify.json`).
- `p3_mcts_drafter.py`: stages full, curve, real and collapse (`cache/mcts_*_s*.pkl.gz`, logs `results/p3_mcts_*_s*.txt`).
- `p3_mcts_analyze.py`: all MCTS tables.
- `p3_fix_drafters.py mcts | analyze_mcts` runs and analyzes these under the protocol at the top of this file (`cache/fix/mcts_*_s*.pkl.gz`, `results/fix/p3_mcts_analyze.json`, `results/fix/fig_mcts_sims_curve.png`, `results/fix/fig_mcts_meta.png`); `p3_fix_realized.py` gives the calibration rows (`results/fix/p3_fix_realized.json`).

# Distilled player-conditioned prior (2026-09-30)

## What was built

- **Prior.** For the acting player p at draft state x, the prior logit of hero h is α·log p_BC(h | x) + g(f(p, h)):
  - p_BC is the kernel's behavioral-cloning prior, softmaxed over the valid mask (free heroes in p's pool).
  - f(p, h) is 9 per-hero player features: the personal skill term b2·s(p, h), off-role, log(1 + games on h), never played, EWMA pick shares at half-lives of 20 and 100 games, log(1 + days since last played), never played recently, and log fine-role share.
  - g is an MLP (9 → 32 → 32 → 1) shared across heroes.
- **Why this form.** g depends only on the player, so for a lobby it is a 10×90 per-slot term that the kernel can add to its PUCT priors. The draft state enters through p_BC.
- **Targets.** Personalized MCTS root visit distributions (400 sims, BC prior) at all 56,860 real pick states of 5,686 V1 lobbies (2026-02-10 to 03-28), all ten players with 50+ games. V1 is disjoint from every V2 evaluation lobby. Split: 90% of lobbies for training and 10% for validation.
- **Test.** The 20,000 personalized MCTS searches at real states of the 2,000-lobby V2 subset.
- **Compute.** 5 GPU-minutes for targets and 2 CPU-minutes for training (`p3_ds_targets.py`, `p3_ds_train.py`, `cache/fix/ds_prior.pt`).
- **Kernel.** `cuda_prior/prior_kernel.cu` is a new copy of the personal kernel. When cfg[56] = 1 the acting slot's priors become softmax(α·log p + term) over the same valid heroes, and root priors are reported. `personal_kernel.cu` is untouched.
- **Kernel verification** (`p3_ds_verify.json`):
  - With the term off, it is identical to the personal kernel: 64 full drafts and 64 decide-only searches, with identical win probabilities, actions and root visits.
  - Root priors vs Python: max |Δ| is 2.2e-7 for the BC prior (α = 1, zero term) and 3.3e-7 for the distilled prior.

**Fit** (held-out V2 searches):

| prior | α | KL to the MCTS visit distribution | top-1 = MCTS argmax | top-1 = real pick |
|---|---|---|---|---|
| BC prior | 1 | 0.235 | 50.9% | 13.5% |
| α·BC (fitted α) | 1.26 | 0.211 | 50.9% | 13.5% |
| g only (no draft state) | | 0.875 | 17.2% | 23.3% |
| **distilled (α·BC + g)** | 1.23 | **0.171** | **57.7%** | 18.3% |

The distilled prior cuts the KL to personalized MCTS by 27% and agrees with its top pick 7 points more often than the BC prior. Real picks of heroes the player had not played before are outside the pool, so the prior gives them no mass; top-1 rates count them as misses.

## Evaluation

Setup:

- **Same data and seeds.** Same held-out lobbies, controlled teams, opponent seeds and GD cycling as the earlier runs. The realized check uses the same 2,000-lobby subset (4,000 teams) for every drafter.
- **Greedy drafters.** They pick the prior's argmax with no search.
- **Pairing.** Each personalized drafter is paired with its non-personal counterpart: greedy BC for greedy distilled, population MCTS for the MCTS drafters.

| metric | greedy BC (non-personal) | greedy distilled | MCTS personalized, BC prior | MCTS personalized, distilled prior | MCTS population |
|---|---|---|---|---|---|
| vs GD: gain in V over the non-personal counterpart | | +4.6 (4.0, 5.2) | +6.2 (5.4, 6.9) | +6.9 (6.1, 7.6) | |
| vs GD: change in WP_pop | | −0.5 (−0.9, −0.0) | −1.1 (−1.8, −0.4) | −3.3 (−4.0, −2.6) | |
| vs GD: mean V of the controlled team | 0.528 | 0.574 | 0.687 | 0.694 | 0.625 |
| vs imitation opponent: gain in V / change in WP_pop | | +5.3 / +0.1 | +7.7 / −0.4 | +8.3 / −2.7 | |
| calibration, own agreement Q5 − Q1: realized / predicted / remainder | | +9.4 / +6.7 / +2.7 (−0.0, +5.5) (all teams) | +4.1 / +4.0 / +0.1 (−4.0, +4.9) | +7.9 / +6.7 / +1.2 (−3.4, +5.8) | |
| calibration, personal component Q5 − Q1: realized / predicted / remainder | | +11.2 / +9.3 / +1.9 (−0.7, +4.9) (all teams) | −0.1 / +5.9 / −6.0 (−11.1, −1.0) | +11.8 / +7.6 / +4.2 (−1.1, +8.8) | |
| collapse: effective pool (median) | 8.4 | 7.9 | 8.9 | 7.9 | 10.1 |
| collapse: share of picks on the player's top-3 heroes | 13% | 23% | 25% | 32% | 13% |
| self-play: effective number of heroes (real 73.5) | 29.5 | 31.3 | 41.3 | **42.8** | 38.6 |
| self-play: share of the 10 most-picked heroes (real 25.7%) | 59.4% | 57.7% | 50.2% | 48.6% | 52.0% |
| self-play: correlation of hero shares with real | 0.75 | 0.78 | 0.65 | 0.68 | 0.62 |
| self-play: off-role picks (real 15.8%) | 33.0% | 23.1% | 20.5% | **15.7%** | 33.6% |
| vs GD: effective heroes (controlled team) | | 46.3 | 52.5 | 52.5 | |

Calibration rows use the 4,000 teams of the 2,000-lobby subset unless marked "all teams" (11,448). The greedy rows rank by the distilled prior and the personal component is distilled minus BC ranking.

Additional comparisons:

- **At 100 sims**, calibration of own agreement: distilled prior realized +6.9 / predicted +5.4 / remainder +1.5 (−2.8, +6.3); BC prior +0.4 / +2.2 / −1.8 (−6.4, +2.8).
- **Distilled vs BC-prior MCTS, both personalized, paired.** The distilled prior adds +0.7pp V (0.1, 1.3) vs GD and +0.7pp (−0.0, 1.5) vs the imitation opponent. It costs −2.2pp of WP_pop (−2.8, −1.6).

## Reading

- **Greedy.** A greedy drafter with the distilled prior gets most of the predicted personal value of full MCTS with the behavioral prior (+4.6pp vs +6.2pp), at a small population-WP cost (−0.5pp) and with no search. It is the cheapest usable personalized drafter so far.
- **The distilled prior moves MCTS toward personal picks.**
  - Predicted V: +0.7pp over the BC prior, paid for with −2.2pp of population WP.
  - Its agreement with real picks tracks a larger predicted skill gap (+6.7 vs +4.0pp at 400 sims; +5.4 vs +2.2pp at 100 sims), and the realized spreads match those predictions. A personal prior makes the search's ranking follow the personal value even when search is cheap.
  - Off-role picks in self-play: 15.7%, the same as real drafts (15.8%) and below the BC-prior MCTS's 20.5%.
- **It does not widen the meta much.**
  - Self-play breadth is 42.8 effective heroes (BC prior 41.3; population 38.6; real 73.5).
  - The prior is personal, but the population WP at the leaf still pulls every team toward the same high-WP core: Valla, Anduin, Muradin, Rehgar, Falstad, Leoric, Li-Ming, Anub'arak, Nazeebo and Thrall take 49% of picks.
  - What changes is who gets those heroes and the tail. Risers: Ana (0.24% vs 0.08%), Hanzo (1.18% vs 0.67%, real 2.41%), Li Li, Genji, Alexstrasza, Sylvanas (1.22% vs 0.78%), Deckard, Zarya, Qhira, Medivh. Fallers: Tracer (0.18% vs 0.37%), Arthas, Lt. Morales, Tyrael, Illidan, Deathwing, The Butcher, Zul'jin, Xul, Anub'arak (3.52% vs 4.27%).
  - Hero shares move toward real play (correlation 0.68 vs 0.65).
- **Cost of concentration.** The personal prior concentrates each player a little more: 32% of picks on the player's top-3 heroes (BC-prior MCTS 25%, real 43%) and an effective pool of 7.9 (8.9). It also gives up more population WP than the BC-prior MCTS (−3.3 vs −1.1pp vs GD), because it steers search toward comfort picks earlier.
- **What the outcome data say.** For every drafter the realized spread across agreement quintiles matches the skill model's own prediction for the same real drafts. They show that the skill model is calibrated at team level; they do not rank the drafters.

Figure: `results/fix/fig_ds_meta.png`. Panels: distilled-prior vs BC-prior MCTS self-play hero shares; distilled-prior self-play vs real; meta breadth of all drafters.

## Files (training/personalization/)

- `cuda_prior/`: `prior_kernel.cu`, `prior_bindings.cpp`, `setup.py`.
- `p3_ds_common.py`: prior model, features, kernel-format state.
- `p3_ds_targets.py`, `p3_ds_train.py`: targets and fit (`results/p3_ds_train.json`).
- `p3_ds_mcts.py`: verification and MCTS runs with the distilled prior (`results/p3_ds_verify.json`, `cache/dsmcts_*.pkl.gz`).
- `p3_ds_greedy.py`: greedy drafters (`cache/dsgreedy_*.pkl.gz`).
- `p3_ds_analyze.py`: the tables above.
- `p3_fix_drafters.py ds_targets | ds_train | ds_mcts | ds_greedy | analyze_ds` runs these under the protocol at the top of this file (`cache/fix/`, `results/fix/p3_ds_train.json`, `results/fix/p3_ds_analyze.json`, `results/fix/fig_ds_meta.png`).
