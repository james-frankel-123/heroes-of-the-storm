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

**Rerun 2026-10-19 on race-fixed kernels.** A race in every P3 kernel's main draft loop let thread 0 rewrite the shared step flags before slower warps had read them. It mattered when steps were skipped (decide-only prefix replay, forced bans): the real-state, collapse, distillation-target and paired-decision searches. The kernels now copy the flags behind a barrier, seed each episode with a splitmix64 hash and floor the prior before its log. Every MCTS number in this part and in the distilled-prior part was rerun on the fixed kernels, with the v2 search (opponent turns as chance nodes, `--search-mode chance`), slot assignment (`--assign slot`) and the GD pool and BC prior trained on the training window only (`cache/gd_trainonly/`). Paired personalized and population runs also no longer share a batch-cache key (the key now hashes the run's inputs). Largest changes: a differing pick now buys more personal value at a larger population-WP cost; the edge against the imitation opponent is now about the same as against GD; no calibration remainder excludes zero; and the distilled prior adds no predicted value over the BC prior in MCTS. One-step columns are the one-step run at the top of this file, unchanged.

## Kernel (`cuda_personal/`)

**Where it lives.** A copy of the guarded overfit2026 kernel (`cuda_ofit`); `training/cuda_mcts` is untouched. `cuda_personal/ref/` is an unmodified copy of `cuda_ofit`, built under another module name as the reference (with the same hashed episode seed). The reruns used sm_120 (RTX 5090) and sm_86 (RTX 3090 and 3080) builds. Source: `personal_kernel.cu` and `personal_bindings.cpp`.

**What changed** (all per episode, so one launch serves up to 256 lobbies):

- **Leaf value.** The base is the population WP (the 3 d2c_cumprev seeds as a mean-logit net per orientation, then swap-symmetrized, from our side). On the logit scale it adds b2·(S_our − S_opp) + b3·(O_our − O_opp), with S and O summed over the ten slots' per-hero skill and off-role tensors (10×90 each). Coefficients b1 = 1.0005, b2 = 3.594 and b3 = −0.056 come from the V1 combiner on counts from earlier days; b0 is set to 0.
- **Pick step to player.** The kernel uses Storm League order: bans 0,1,0,1; picks 0,1,1,0,0; bans 1,0; picks 1,1,0,0,1, with kernel team 0 picking first. Real lobbies where team 1 picked first are relabeled. The player at a pick step is the player who made that pick in the real draft (`draft_order`, the order fetched by `p3_fetch_pickorder.py`). Slots 0 to 4 are kernel team 0's players in pick order, and 5 to 9 are team 1's.
- **Pools and bans.** Every pick is restricted to the acting slot's pool (heroes played on earlier days), falling back to any free hero if the pool is exhausted. The kernel can force given bans; the runs below force none, so our bans are searched and the other team's come from the GD.
- **Decide-only mode.** Replays a prefix of actions, then runs one search for the acting team and reports root visits and Q. This is used for the real-state and collapse checks.
- **Self-play.** Every non-forced step is searched from the acting team's side.
- **Imitation opponent** (main draft only): softmax(w0·log p_GD + personal bias of the acting slot).
- **Tree arena.** The v2 search sizes its node arena to 3 × sims + 8 per episode and checks the bound before every expansion. MAX_OUR_TURNS is raised to 16 for self-play.

**Verification** (`p3_mcts_verify.py`, `results/racefix/p3_mcts_verify_chance.json`, race-fixed build):

- **Leaf values.** 2,000 real held-out drafts with realistic personal tensors (rows of the personal tables), random slot assignments and random perspective. Kernel vs a float64 Python mirror: population WP max |Δ| 6.7e-7 (mean 1.2e-7); personal value max |Δ| 6.0e-7 (mean 1.2e-7).
- **Population identity.** With personal terms zeroed (b = 0, 1, 0, 0, no pools, no forced bans), and separately with personal valuation switched off, the kernel reproduces the reference population kernel exactly: 64 episodes each at 100 and 400 sims, with identical win probabilities (bitwise), final drafts and root visit distributions.
- **Arena.** Nodes used peak at 186 (100 sims), 722 (400 sims), 1,774 (1,000 sims) and 5,265 (3,000 sims), with no capacity hits.
- **Depth.** At 100 and 400 sims the tree reaches our later pick blocks (steps 7, 8, 10, 13 and 14).
- **Ensembling.** The kernel's mean-logit ensemble and the one-step drafter's mean-probability ensemble differ by at most 0.0019 WP (mean 0.0001).

## Protocol

- **Search.** Search-only: the prior is the outcome-free behavioral-cloning policy distilled from the training-window GD (`cache/gd_trainonly/bc_prior.pt`). Opponent turns in the tree are chance nodes drawn from GD, and all rollouts use GD, one of the five training-window GD models per batch, cycled; paired personalized and population runs share the model and seed. The root pick is the argmax of visits, with c_puct 2.0 and no Dirichlet noise.
- **Data.** Same lobbies, controlled teams and players as the one-step run: 1,000 full lobbies, the first 2,000 of the 5,724 realized lobbies (4,000 teams) for real-state searches, and 300 collapse players (30 new contexts each).
- **Main level.** 400 sims per decision. The sims curve uses 25, 100, 400 and 1,500 sims on the same lobbies.
- **Compute.** One RTX 5090 process at a time, on 2 host threads; analyses on CPU.

## MCTS vs one-step vs population

| metric | one-step personalized | MCTS personalized (400 sims) | MCTS population (400 sims) |
|---|---|---|---|
| picks that differ from the population drafter at the same state | 51% | 40% | |
| when they differ: personal gain / population WP cost (pp) | +3.4 / −2.9 | +4.4 (4.2, 4.6) / −1.9 (−2.0, −1.7) | |
| whole draft vs GD: gain in V over the population drafter | +4.9 (4.3, 5.6) | +6.4 (5.5, 7.2) | |
| whole draft vs GD: change in WP_pop | −2.7 | −0.9 (−1.5, −0.1) | |
| whole draft vs GD: mean V of the controlled team | 0.738 | 0.691 | 0.627 |
| whole draft vs imitation opponent: gain in V / change in WP_pop | +6.6 / −2.5 | +6.6 (5.7, 7.6) / −0.8 (−1.6, −0.0) | |
| calibration, personalized agreement Q5 − Q1: realized / predicted / remainder | +8.5 / +7.1 / +1.5 (−1.1, +4.1) | +2.5 / +1.5 / +0.9 (−3.8, +5.3) | |
| calibration, personal component Q5 − Q1: realized / predicted / remainder | +11.3 / +9.8 / +1.5 (−1.3, +4.5) | +3.5 / +4.4 / −0.9 (−5.3, +4.0) | |
| collapse: effective pool per player (median) | 10.4 | 9.0 | 10.0 |
| collapse: share of picks on the player's top-3 heroes (mean) | 29% | 24% | 12% |
| self-play: effective number of heroes (real 73.5) | 69.0 | 42.0 | 38.8 |
| self-play: share of the 10 most-picked heroes (real 25.7%) | 31.0% | 48.3% | 51.1% |
| self-play: correlation of hero shares with real drafts | 0.47 | 0.65 | 0.60 |
| off-role picks, self-play (real 15.8%) | 17.3% | 20.1% | 33.9% |
| off-role picks, vs GD (controlled team) | 26.0% | 27.7% | 34.3% |
| real picks equal to the drafter's top-1 | 12.2% | 18.7% | 13.0% |
| imitation top-1 equal to the drafter's top-1 | 14.2% | 29.6% | |

The one-step calibration rows use all 11,448 teams, the MCTS rows the 3,996 teams of the 2,000-lobby subset whose real picks were all among the drafter's candidates. The population one-step drafter's off-role rates are 32.7% (self-play) and 34.1% (vs GD).

**Reading:**

- **Cheaper personal gains.** MCTS changes fewer picks than one-step (40% vs 51%). The picks it changes buy more personal gain (+4.4 vs +3.4pp) at a smaller population-WP cost (−1.9 vs −2.9pp per pick; −0.9 vs −2.7 per draft). Search finds personal picks that also hold up in the draft.
- **Opponent model.** Against the imitation opponent the personalized MCTS drafter's predicted edge is +6.6pp at −0.8pp of WP_pop, about the same as against GD (+6.4pp).
- **Calibration.** For both drafters the realized spread across agreement quintiles is what the skill model predicts for those drafts: every remainder in this file is within its CI of zero. At 400 sims the MCTS ranking carries less of the personal value than the one-step ranking (predicted spread +1.5 vs +7.1pp for personalized agreement), so these rows say little either way.
- **Off-role handling.** 20% of personalized MCTS self-play picks go to off-role players, against 34% for population drafters and 16% in real drafts.
- **No collapse** onto comfort heroes. The effective pool is 9.0 heroes (population MCTS 10.0, real play 11.3 at the same sample size), and 24% of picks go to the player's top-3 heroes (43% in real play).

## Emergent meta under MCTS

Figure: `results/racefix/fig_mcts_meta.png`.

- **Concentration.** With the behavioral prior and argmax at the root, MCTS self-play concentrates on the prior's favorite heroes far more than one-step search or real drafts. The effective number of heroes is 42.0 (personalized) and 38.8 (population), against 73.5 in real drafts. The top 10 heroes take 48 to 51% of picks. The leaders are Valla (7.0% personalized, 7.4% population, real 2.9%), Muradin (6.7%), Anduin (6.6%), Rehgar (5.7%), Leoric, Anub'arak, Falstad, Nazeebo, Li-Ming and Tychus.
- **Role mix.** It is close to real: tanks 20%, bruisers 19%, healers 21%, ranged 35%, melee assassins 4.5%. Hero shares correlate 0.65 with real (one-step 0.47), because the prior is behavioral.
- **Personalization widens the pool** (42.0 vs 38.8 heroes) and moves the meta toward real play (correlation 0.65 vs 0.60). The two self-play metas correlate 0.98.
- **Risers under personalized MCTS:** Alarak (1.53% vs 0.49%), Hanzo (0.83 vs 0.28), Tyrael (0.47 vs 0.14), Genji, Junkrat, Sylvanas (0.71 vs 0.37), D.Va, Maiev, Malthael, Hogger (2.36 vs 1.54).
- **Fallers:** Rexxar, Malfurion (1.05% vs 1.54%), Varian (1.55 vs 2.17), Stukov, Deathwing, Jaina (2.19 vs 2.90), Diablo (2.48 vs 3.22), Azmodan (1.53 vs 1.99), Gazlowe, Zeratul.
- **The one-step conclusion holds.** Knowing who plays lifts specialist and high-execution heroes (Hanzo, Genji, Maiev, Alarak, D.Va, Sylvanas) and trims generic picks.

## Sims curve: predicted value and calibration by search effort

Figure: `results/racefix/fig_mcts_sims_curve.png`. Predicted rows use the same 1,000 lobbies; calibration rows the same 2,000-lobby subset (3,996 teams).

| search | predicted whole-draft gain in V vs GD (pp) | change in WP_pop (pp) | personalized agreement Q5 − Q1: realized / remainder (pp) | personal component Q5 − Q1: realized / remainder |
|---|---|---|---|---|
| MCTS 25 sims | +2.1 (1.3, 2.9) | −0.5 | −3.5 / −2.1 (−6.5, +2.5) | −3.2 / −2.8 (−7.4, +1.8) |
| MCTS 100 sims | +3.4 (2.5, 4.3) | −0.6 | +2.0 / +2.1 (−2.4, +6.5) | +1.7 / +1.5 (−3.6, +5.7) |
| MCTS 400 sims | +6.4 (5.5, 7.2) | −0.9 | +2.5 / +0.9 (−3.8, +5.3) | +3.5 / −0.9 (−5.3, +4.0) |
| MCTS 1,500 sims | +7.4 (6.7, 8.1) | −2.5 | +3.4 / −0.6 (−5.1, +4.4) | +7.2 / +0.3 (−4.4, +5.1) |
| one-step (≈86 rollouts per decision) | +4.9 (4.3, 5.6) | −2.7 | +8.5 / +1.5 (−1.1, +4.1) (all teams) | +11.3 / +1.5 (−1.3, +4.5) |

Reading:

- **Predicted gains rise steadily with search:** +2.1pp at 25 sims, +3.4pp at 100, +6.4pp at 400 and +7.4pp at 1,500. MCTS passes one-step between 100 and 400 sims.
- **Realized spreads follow the skill model's prediction at every level.** At low sims the root visit ranking is mostly the behavioral prior, so agreement with it carries little of the personal value and the predicted and realized spreads are both small. At 1,500 sims the personal component is large and matches its prediction (+7.2 realized, +6.9 predicted). All remainders are zero within their CIs. Whether deeper search buys value beyond what the skill model predicts cannot be read from these rows.
- **Paper 1's warning applies to the predicted column.** The predicted gains are the model's own valuations of its own drafts, and they keep rising with search. The v2 arena grows with sims, so deeper search is not capped by tree size (no capacity hits at 3,000 sims).

## Imitation vs outcome under MCTS

At the real states (20,000 real picks of the 2,000-lobby subset), the personalized MCTS top-1 matches the real pick 18.7% of the time (one-step 12%) and matches the imitation model's top-1 29.6% of the time (one-step 14%). The behavioral prior pulls search toward what players usually pick.

Team residual by which system the real pick matched (calibration, as for one-step):

| the real pick matched | picks | residual (pp) |
|---|---|---|
| both | 2,331 | +1.8 (−0.2, 3.5) |
| imitation only | 4,176 | +1.6 (0.1, 3.1) |
| personalized MCTS only | 1,408 | +2.2 (−0.2, 4.7) |
| neither | 12,085 | −1.1 (−2.1, −0.2) |

Picks matching either system sit 1.6 to 2.2pp above the population expectation and picks matching neither 1.1pp below; the classes do not separate from each other at this sample size.

## Stretch: player-conditioned policy trained by self-play (not started; cost estimate)

- **Kernel work.** The policy backbone takes a fixed 290-d state, and the acting slot's personal vector would have to enter it. Threading that input through the backbone and every expansion and re-verifying is a new kernel variant, about one working day.
- **Self-play data.** Measured throughput is about 110 searched decisions per second at 400 sims on one shared RTX 5090 (the real-state stage ran 40,000 searches in 6 minutes). That makes about 5 GPU-hours per 200K self-play drafts, and paper-1-style training needs many iterations: several GPU-days under the 2-process cap.
- **Cheaper alternative.** Distill the root visit distributions already produced here into a player-conditioned policy by supervised learning, about one GPU-hour. It would serve as a better prior than the behavioral one (the low-sims curve shows the prior dominates below about 400 sims), but it would not test self-play training. Awaiting a decision before either is started.

## Files (training/personalization/)

- `cuda_personal/`: `personal_kernel.cu`, `personal_bindings.cpp`, `setup.py`, and `ref/` (the unmodified reference kernel).
- `p3_mcts_core.py`: host side, draft-order and slot conventions, the Python leaf mirror.
- `p3_mcts_verify.py`: verification (`results/racefix/p3_mcts_verify_chance.json`).
- `p3_mcts_drafter.py`: stages full, curve, real and collapse (`cache/mcts_v2/chance/mcts_*_s*.pkl.gz` on the workers).
- `p3_mcts_analyze.py`: all MCTS tables (`results/racefix/p3_mcts_analyze.json`, `results/racefix/fig_mcts_sims_curve.png`, `results/racefix/fig_mcts_meta.png`); `p3_fix_realized.py` gives the calibration rows (`results/racefix/p3_fix_realized.json`).
- `remote/p3_racefix_queue.json` and `remote/p3_racefix_waiter.py`: the exact commands and hosts of the 2026-10-19 reruns.

# Distilled player-conditioned prior (2026-09-30)

## What was built

- **Prior.** For the acting player p at draft state x, the prior logit of hero h is α·log p_BC(h | x) + g(f(p, h)):
  - p_BC is the kernel's behavioral-cloning prior, softmaxed over the valid mask (free heroes in p's pool).
  - f(p, h) is 9 per-hero player features: the personal skill term b2·s(p, h), off-role, log(1 + games on h), never played, EWMA pick shares at half-lives of 20 and 100 games, log(1 + days since last played), never played recently, and log fine-role share.
  - g is an MLP (9 → 32 → 32 → 1) shared across heroes.
- **Why this form.** g depends only on the player, so for a lobby it is a 10×90 per-slot term that the kernel can add to its PUCT priors. The draft state enters through p_BC.
- **Targets.** Personalized MCTS root visit distributions (400 sims, BC prior, race-fixed kernel) at all 56,860 real pick states of 5,686 V1 lobbies (2026-02-10 to 03-28), all ten players with 50+ games. V1 is disjoint from every V2 evaluation lobby. Split: 90% of lobbies for training and 10% for validation.
- **Test.** The 20,000 personalized MCTS searches at real states of the 2,000-lobby V2 subset.
- **Compute.** 5 GPU-minutes for targets and 2 CPU-minutes for training (`p3_ds_targets.py`, `p3_ds_train.py`, `cache/mcts_v2/chance/ds_prior.pt` on the workers).
- **Kernel.** `cuda_prior/prior_kernel.cu` is a new copy of the personal kernel. When cfg[56] = 1 the acting slot's priors become softmax(α·log p + term) over the same valid heroes, and root priors are reported. `personal_kernel.cu` is untouched.
- **Kernel verification** (`results/racefix/p3_ds_verify.json`, race-fixed build):
  - With the term off, it is identical to the personal kernel: 64 full drafts and 64 decide-only searches, with identical win probabilities, actions and root visits.
  - Root priors vs Python: max |Δ| is 1.8e-7 for the BC prior (α = 1, zero term) and 3.3e-7 for the distilled prior.

**Fit** (held-out V2 searches; `results/racefix/p3_ds_train.json`):

| prior | α | KL to the MCTS visit distribution | top-1 = MCTS argmax | top-1 = real pick |
|---|---|---|---|---|
| BC prior | 1 | 0.256 | 46.6% | 12.9% |
| α·BC (fitted α) | 1.25 | 0.235 | 46.6% | 12.9% |
| g only (no draft state) | | 0.886 | 16.4% | 21.5% |
| **distilled (α·BC + g)** | 1.22 | **0.192** | **53.3%** | 17.8% |

The distilled prior cuts the KL to personalized MCTS by 25% and agrees with its top pick 6.7 points more often than the BC prior. Real picks of heroes the player had not played before are outside the pool, so the prior gives them no mass; top-1 rates count them as misses.

## Evaluation

Setup:

- **Same data and seeds.** Same held-out lobbies, controlled teams, opponent seeds and GD cycling as the earlier runs. The realized check uses the same 2,000-lobby subset (3,996 teams) for every MCTS drafter.
- **Greedy drafters.** They pick the prior's argmax with no search.
- **Pairing.** Each personalized drafter is paired with its non-personal counterpart: greedy BC for greedy distilled, population MCTS for the MCTS drafters.

| metric | greedy BC (non-personal) | greedy distilled | MCTS personalized, BC prior | MCTS personalized, distilled prior | MCTS population |
|---|---|---|---|---|---|
| vs GD: gain in V over the non-personal counterpart | | +5.2 (4.6, 5.8) | +6.4 (5.5, 7.2) | +6.5 (5.7, 7.3) | |
| vs GD: change in WP_pop | | −0.1 (−0.6, +0.4) | −0.9 (−1.7, −0.1) | −4.1 (−4.8, −3.2) | |
| vs GD: mean V of the controlled team | 0.519 | 0.571 | 0.691 | 0.692 | 0.627 |
| vs imitation opponent: gain in V / change in WP_pop | | +5.5 / +0.0 | +6.6 / −0.8 | +7.0 / −3.5 | |
| calibration, own agreement Q5 − Q1: realized / predicted / remainder | | +2.5 / +3.7 / −1.2 (−4.6, +2.1) | +2.5 / +1.5 / +0.9 (−3.8, +5.3) | +4.2 / +4.2 / −0.1 (−5.2, +4.8) | |
| calibration, personal component Q5 − Q1: realized / predicted / remainder | | +10.3 / +9.2 / +1.1 (−2.1, +4.5) | +3.5 / +4.4 / −0.9 (−5.3, +4.0) | +9.1 / +6.6 / +2.5 (−1.9, +7.2) | |
| collapse: effective pool (median) | 7.6 | 7.4 | 9.0 | 7.7 | 10.0 |
| collapse: share of picks on the player's top-3 heroes | 13% | 23% | 24% | 32% | 12% |
| self-play: effective number of heroes (real 73.5) | 28.6 | 31.0 | 42.0 | **44.0** | 38.8 |
| self-play: share of the 10 most-picked heroes (real 25.7%) | 60.7% | 58.9% | 48.3% | 47.4% | 51.1% |
| self-play: correlation of hero shares with real | 0.71 | 0.75 | 0.65 | 0.68 | 0.60 |
| self-play: off-role picks (real 15.8%) | 33.8% | 24.2% | 20.1% | **15.4%** | 33.9% |
| vs GD: effective heroes (controlled team) | | 46.2 | 53.3 | 52.6 | |

Calibration rows use the 3,996 teams of the 2,000-lobby subset for the MCTS drafters and 7,988 teams (all lobbies whose real picks were all among the candidates) for the greedy drafters. The greedy rows rank by the distilled prior and the personal component is distilled minus BC ranking.

Additional comparisons:

- **At 100 sims**, calibration of own agreement: distilled prior realized +1.9 / predicted +3.0 / remainder −1.0 (−5.7, +4.0); BC prior +2.0 / −0.1 / +2.1 (−2.4, +6.5). Personal component: distilled +6.7 / +5.6 / +1.0 (−3.6, +5.7); BC prior +1.7 / +0.2 / +1.5 (−3.6, +5.7).
- **Distilled vs BC-prior MCTS, both personalized, paired.** The distilled prior adds +0.1pp V (−0.6, +0.9) vs GD and +0.4pp (−0.5, +1.2) vs the imitation opponent. It costs −3.2pp of WP_pop (−3.9, −2.5).

## Reading

- **Greedy.** A greedy drafter with the distilled prior gets most of the predicted personal value of full MCTS with the behavioral prior (+5.2pp vs +6.4pp), at no measurable population-WP cost (−0.1pp) and with no search. It is the cheapest usable personalized drafter so far.
- **The distilled prior moves MCTS toward personal picks without adding predicted value.**
  - Predicted V: +0.1pp over the BC prior (CI −0.6 to +0.9), paid for with −3.2pp of population WP. Search with the BC prior already finds about as much personal value; the distilled prior reaches it through different, lower-WP drafts.
  - Its agreement with real picks tracks a larger predicted skill gap (personal component +6.6 vs +4.4pp at 400 sims; +5.6 vs +0.2pp at 100 sims), and the realized spreads match those predictions. A personal prior makes the search's ranking follow the personal value even when search is cheap.
  - Off-role picks in self-play: 15.4%, the same as real drafts (15.8%) and below the BC-prior MCTS's 20.1%.
- **It widens the meta a little.**
  - Self-play breadth is 44.0 effective heroes (BC prior 42.0; population 38.8; real 73.5).
  - The prior is personal, but the population WP at the leaf still pulls every team toward the same high-WP core: Muradin, Anduin, Valla, Rehgar, Leoric, Anub'arak, Nazeebo, Falstad, Li-Ming and Tychus take 47% of picks.
  - What changes is who gets those heroes and the tail. Risers: The Lost Vikings (0.17% vs 0.04%), Zagara (0.31 vs 0.13), D.Va, Hanzo (1.37% vs 0.83%, real 2.41%), Sylvanas (1.11% vs 0.71%), Kael'thas, Gall, Deckard, Valeera, Cho. Fallers: Lt. Morales (0.28% vs 0.48%), Xul, Rexxar, Tyrael, Tracer, Zul'jin (2.16% vs 2.87%), Mei, Sonya, Jaina, Ragnaros.
  - Hero shares move toward real play (correlation 0.68 vs 0.65).
- **Cost of concentration.** The personal prior concentrates each player more: 32% of picks on the player's top-3 heroes (BC-prior MCTS 24%, real 43%) and an effective pool of 7.7 (9.0). It also gives up more population WP than the BC-prior MCTS (−4.1 vs −0.9pp vs GD), because it steers search toward comfort picks earlier.
- **What the outcome data say.** For every drafter the realized spread across agreement quintiles matches the skill model's own prediction for the same real drafts. They show that the skill model is calibrated at team level; they do not rank the drafters.

Figure: `results/racefix/fig_ds_meta.png`. Panels: distilled-prior vs BC-prior MCTS self-play hero shares; distilled-prior self-play vs real; meta breadth of all drafters.

## Files (training/personalization/)

- `cuda_prior/`: `prior_kernel.cu`, `prior_bindings.cpp`, `setup.py`.
- `p3_ds_common.py`: prior model, features, kernel-format state.
- `p3_ds_targets.py`, `p3_ds_train.py`: targets and fit (`results/racefix/p3_ds_train.json`).
- `p3_ds_mcts.py`: verification and MCTS runs with the distilled prior (`results/racefix/p3_ds_verify.json`, `cache/mcts_v2/chance/dsmcts_*.pkl.gz` on the workers).
- `p3_ds_greedy.py`: greedy drafters (`cache/dsgreedy_*.pkl.gz` on the workers).
- `p3_ds_analyze.py`: the tables above (`results/racefix/p3_ds_analyze.json`, `results/racefix/fig_ds_meta.png`).
- Commands and hosts of the 2026-10-19 reruns: `remote/p3_racefix_queue.json`.
