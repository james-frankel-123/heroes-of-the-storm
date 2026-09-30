# P3: the personalized drafter and the DraftRec-style contrast (2026-09-30)

## Design (one unified model, no model of "everyone")

The value of a finished draft, from team 0's side, is:

V = sigmoid(b0 + b1·logit WP_pop + b2·(S_0 − S_1) + b3·(O_0 − O_1))

- **WP_pop:** paper 2's drift-aware d2c_cumprev ensemble (3 seeds, swap-symmetrized), with features from cumulative statistics through the previous build. My batched evaluator reproduces the cached scores of real games to 1e-7.
- **S_t:** the sum over team t's players of s(p, h). This is the experience offset plus the posterior mean of the phase-1 "+CF rank 2" kernel (pooled skill and the similarity embedding), with state through the previous day.
- **O_t:** the number of off-role players on team t (fine role under 10% of 50+ earlier games).
- **b:** the V1-fit combiner (b1 = 1.0 on the WP logit, b2 = 3.71 on summed skill, b3 = −0.052 per off-role player).
- **Population drafter:** values drafts by WP_pop alone.

**Lobbies.** 5,724 held-out V2 games (2026-04-01 to 05-22) in which all ten players had 50+ earlier games:

- 1,000 are used for full simulations and all 5,724 for the realized-value check.
- The player making each pick is the one who played the hero picked at that step.
- A player's available heroes are the ones he played before the game day, plus the hero he actually played; the median pool is 34 heroes.
- Real bans are replayed whenever the hero is still free.

**Search.** Search-only with a behavioral prior:

- At each decision the candidates are the acting player's free pool heroes, capped to the union of the top 15 by GD probability and the top 15 by the player's personal term (21.5 candidates on average).
- Each candidate gets 4 rollouts that complete the draft with the paper-1 generic-draft (GD) models, restricted to each picking player's pool.
- The personalized and population values are computed on the same rollouts, so every pick comparison is paired.

I did not put personal terms into the CUDA MCTS kernel: it hard-codes the paper-1 WP and its lookup tables. This harness runs on 4 CPU cores in about 16 minutes. No GPU was used.

## Headline numbers

**How often and how much the recommendation changes** (paired, same state, same rollouts):

| trajectory | decisions | picks that differ | when they differ: gain in V (pp) | when they differ: change in WP_pop (pp) |
|---|---|---|---|---|
| (a) vs GD opponent | 5,000 | 52% | +3.6 (3.5, 3.7) | −3.0 (−3.1, −2.9) |
| (a) vs imitation opponent | 5,000 | 53% | +3.8 (3.7, 3.9) | −3.1 |
| (b) self-play | 10,000 | 53% | +3.9 (3.8, 4.0) | −3.1 |

- About half of all personalized picks differ from the population pick.
- When they differ, the personalized pick trades about 3pp of population WP for about 3.6 to 3.9pp of personalized value.
- In 24 to 27% of decisions the personalized gain exceeds 3pp.

**Whole drafts** (controlled team, paired by lobby):

| opponent | personalized drafter V | population drafter V | real draft V | difference in V | difference in WP_pop |
|---|---|---|---|---|---|
| GD | 0.733 | 0.682 | 0.503 | +5.1pp (4.5, 5.9) | −3.1pp (−3.7, −2.6) |
| imitation model | 0.655 | 0.597 | 0.503 | +5.8pp (5.0, 6.5) | −3.2pp |

The imitation opponent is a tougher opponent than GD: V is lower by about 8pp for both drafters.

## Realized-value check (no model in between)

In the real drafts of the 5,724 held-out lobbies (11,448 teams), each real pick is placed in both drafters' rankings at the real draft state (percentile among candidates). A team's agreement is the mean percentile over its five picks. The outcome is scored only against the population model: residual = team won − WP_pop of the real game.

Residual win rate (pp) by agreement quintile:

| agreement measure | Q1 | Q2 | Q3 | Q4 | Q5 | Q5 minus Q1 (95% CI) |
|---|---|---|---|---|---|---|
| personalized drafter ranking (V) | −3.2 | −2.8 | +0.3 | +1.2 | +4.5 | **+7.6 (4.9, 10.5)** |
| population drafter ranking (WP_pop) | +0.2 | −0.1 | +0.7 | −1.1 | +0.3 | +0.1 (−2.5, 2.9) |
| personal component only (V − WP ranking) | −4.5 | −2.7 | +0.1 | +2.0 | +5.1 | **+9.6 (6.7, 12.4)** |

Mean agreement runs from about 0.41 to 0.44 in Q1 to 0.81 to 0.82 in Q5.

- Teams whose real picks the personalized drafter would also have made beat the population model's prediction by 7.6pp more than teams whose picks it would not have made.
- Agreement with the population drafter carries no residual, which is the sanity check: the population model already prices its own preferences.
- Controlling for population agreement, the personal-component slope is +26pp per unit of agreement (20, 32).
- The personal layer has real, model-free value at the team level. Part of it is the experience offset: teams that put players on unfamiliar heroes lose more than WP_pop expects.

## Collapse: does it degenerate to "everyone plays their 3 best heroes"?

This is the proposal's experiment #1. Setup: 300 players, each placed into 30 random real partial drafts at a team-0 pick step.

| per player (median) | personalized drafter | population drafter | real play |
|---|---|---|---|
| effective hero pool (exp entropy, 30 draws) | 10.7 | 12.9 | 11.3 (30 sampled games) / 18.1 (all games) |
| distinct heroes in 30 decisions | 14 | 16 | |
| share on the player's single most-played hero | 6.7% (mean 11%) | 3.3% (mean 4%) | 18% (mean 21%) |
| share on the player's top-3 most-played heroes | 25% (mean 28%) | 10% (mean 13%) | 41% (mean 43%) |

**No collapse.**

- The personalized drafter uses a slightly narrower pool than the population drafter (10.7 vs 12.9 effective heroes). It is about as wide as the player's own play at the same sample size (11.3).
- It concentrates less on the player's favorite heroes than the player does: 28% of its picks go to the player's top-3 heroes, against 43% of his real games.
- Skill estimates are shrunk (the typical per-hero spread is 2 to 4pp) while draft context moves WP_pop by several pp, so the draft keeps most of the weight.

Figure: `fig_dr_collapse.png`.

## Emergent meta (1,000 lobbies, both teams)

Figure: `fig_dr_meta.png`. Panels: personalized vs population self-play pick shares; personalized self-play vs real; role shares.

| | personalized self-play | population self-play | real drafts (same lobbies) |
|---|---|---|---|
| effective number of heroes (exp entropy) | 69.5 | 63.7 | 73.5 |
| share of the 10 most-picked heroes | 30.7% | 32.9% | 25.7% |
| correlation of hero shares with real | 0.47 | 0.38 | |

- **Correlation between the two self-play metas:** 0.97. Personalization shifts the meta at the margin rather than rewriting it.
- **Both drafters concentrate on WP-favored heroes.** Rehgar (4.9% personalized, 5.4% population, 2.3% real), Auriel, Tyrael, Leoric, Illidan, Kerrigan and Hogger are all over-picked relative to real drafts. By role, both drafters pick more bruisers (24% vs 18% real) and melee assassins (12% vs 6%) and fewer ranged assassins (27 to 28% vs 35%) and tanks (15% vs 19.5%).
- **Heroes that rise when drafters know who is playing** (personalized vs population self-play share): Hanzo (0.65% vs 0.16%; real 2.4%), Genji (0.30 vs 0.08), Yrel, Brightwing (0.63 vs 0.25), Valeera, Chen, Nova, Sylvanas (0.79 vs 0.46), Lúcio, Anduin (1.31 vs 0.82).
- **Heroes that fall:** Nazeebo (1.39 vs 2.10), Zarya, Lt. Morales, Malfurion, Deathwing, Illidan (2.55 vs 3.32), Ragnaros, Xul, Jaina, Orphea.
- **Reading:** the risers are the high-execution specialist heroes that sit at the top of the skill embedding's first axis (Hanzo, Genji, Yrel, Chen), plus popular heroes that real players are good at. The fallers are "generically strong for anyone" picks the population model likes. Knowing who plays moves the meta toward real play (correlation 0.38 to 0.47) by giving specialists their heroes.

## Off-role handling

Share of picks that land on a player whose fine-role share for that hero is under 10%:

| | personalized self-play | population self-play | real drafts |
|---|---|---|---|
| off-role picks | **17.2%** | 34.3% | 15.8% |
| mean role share of the assigned player | 27.5% | 21.1% | 30.3% |

The population drafter ignores who is playing and puts a third of heroes on players who rarely play that role. The personalized drafter assigns roles almost as comfortably as real players choose for themselves.

## Item 2: DraftRec-style imitation model (`p3_dr_imitation.py`)

**Model.** A conditional logit over untaken heroes. The inputs are the GD log-probability given the draft state, plus the player's history: log(1 + games on the hero), a never-played flag, EWMA pick shares at 20 and 100 games, days since last played, and fine-role share. Fit on 250K V1 picks, tested on 250K V2 picks.

| model (V2 picks) | top-1 | top-3 | top-5 | top-10 | log-lik |
|---|---|---|---|---|---|
| **imitation (GD + personal history)** | **31.9%** | 54.7% | 65.6% | 78.7% | −2.61 |
| GD alone (non-personal) | 9.1% | 21.2% | 30.3% | 46.8% | −3.78 |
| frequency alone (EWMA-20 share) | 26.8% | 47.8% | 58.3% | 71.3% | −3.42 |
| personal history alone (fitted) | 27.0% | 47.8% | 58.2% | 71.5% | −2.93 |
| uniform over the player's pool | 7.2% | 19.3% | 30.5% | 49.3% | −7.72 |

- Top-1 accuracy by the team's pick position: 39% on the first pick, falling to 27 to 28% on later picks. GD alone gets 8 to 12%.
- Personal history carries most of the signal, and the draft state adds 5 points of top-1 on top.

**Imitation vs outcome-based recommendations** (57,240 real picks at the real states):

| | share |
|---|---|
| real pick equals imitation top-1 | 33.3% |
| real pick equals personalized (outcome) drafter top-1 | 12.4% |
| real pick equals population drafter top-1 | 7.7% |
| imitation top-1 equals personalized top-1 (same candidate set) | 14.2% |
| personalized top-1 equals population top-1 | 47.2% |

The two personal systems rarely agree. Imitation recommends what the player usually plays; the outcome-based drafter recommends what is predicted to win with this player in this draft.

**Whose picks win more against the population expectation?** Team residual (won − WP_pop) by which system the real pick matched:

| the real pick matched | picks | team residual (pp) |
|---|---|---|
| both | 3,766 | +3.3 (1.8, 4.8) |
| outcome-based drafter only | 3,304 | +1.8 (0.2, 3.4) |
| imitation only | 15,365 | +1.6 (0.9, 2.4) |
| neither | 34,805 | −1.3 (−1.8, −0.7) |

When the two disagree, picks matching either one win about equally often, about 1.7pp above the population expectation. Picks matching both win the most. The imitation model captures the comfort-pick benefit; the outcome drafter captures skill-in-context. They are complementary, and neither dominates at this sample size.

The imitation model also serves as the personalized opponent model in version (a). It makes the opponent about 8pp tougher in V than GD, and the personalized drafter's edge over the population drafter grows slightly (+5.8pp vs +5.1pp).

## Caveats

- **Predicted gains are model-internal.** The +3.6pp per differing pick and +5pp per draft are the personalized model's own claims. The realized check supports the direction and the order of magnitude at the team level; it does not validate the exact size.
- **Pool proxy.** "Played before" ignores heroes owned but never played, so it understates true pools. It also plays in favor of the drafter's comfort.
- **Search depth.** One-ply search with 4 behavioral rollouts per candidate. Deeper search (MCTS) might exploit WP_pop further; the population self-play concentration on Rehgar and Auriel is a sign of that. Self-play uses GD rollouts for both sides.
- **Bans.** Real bans are kept, not optimized.

## Files (training/personalization/)

- `p3_fetch_drafts_post.py`: full draft orders, maps and tiers for post-cutoff games.
- `p3_dr_core.py`: lobbies, personal tables (s, off-role, pool, counts for every candidate), the GD policy, and the WP evaluator.
- `p3_dr_imitation.py`: imitation model (`results/p3_dr_imitation.json`).
- `p3_dr_drafter.py`: drafter runs, versions (a) and (b), real-state rankings and collapse contexts (`cache/dr_runs.pkl.gz`).
- `p3_dr_analyze.py`: all tables above (`results/p3_dr_analyze.json`, `fig_dr_meta.png`, `fig_dr_collapse.png`).

# Personalized MCTS (2026-09-30)

## Kernel (`cuda_personal/`)

**Where it lives.** A copy of the guarded overfit2026 kernel (`cuda_ofit`); `training/cuda_mcts` is untouched. `cuda_personal/ref/` is an unmodified copy of `cuda_ofit`, built under another module name as the reference. Built with nvcc 12.9 for sm_120. Source: `personal_kernel.cu` and `personal_bindings.cpp`.

**What changed** (all per episode, so one launch serves up to 256 lobbies):

- **Leaf value.** The base is the population WP (the 3 d2c_cumprev seeds as a mean-logit net per orientation, then swap-symmetrized, from our side). On the logit scale it adds b2·(S_our − S_opp) + b3·(O_our − O_opp), with S and O summed over the ten slots' per-hero skill and off-role tensors (10×90 each). Coefficients b1 = 1.0015, b2 = 3.709 and b3 = −0.052 come from the V1 combiner; b0 is set to 0 (fitted value 0.0013).
- **Pick step to player.** The kernel uses Storm League order: bans 0,1,0,1; picks 0,1,1,0,0; bans 1,0; picks 1,1,0,0,1, with kernel team 0 picking first. Real lobbies where team 1 picked first are relabeled. The player at a pick step is the player who made that pick in the real draft (`draft_order`, the order fetched by `p3_fetch_pickorder.py`). Slots 0 to 4 are kernel team 0's players in pick order, and 5 to 9 are team 1's.
- **Pools and bans.** Every pick is restricted to the acting slot's pool, falling back to any free hero if the pool is exhausted. Real bans are forced wherever the hero is still free: in the main draft, in the tree and in rollouts.
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
- **Data.** Same lobbies, controlled teams and players as the one-step run: 1,000 full lobbies, 5,724 realized lobbies and 300 collapse players (30 new contexts each).
- **Main level.** 400 sims per decision. The sims curve uses 25, 100, 400 and 1,500 sims. The realized curve is scored on the same 2,000-lobby subset (4,000 teams) at every level, together with the one-step drafter on that subset.
- **Compute.** GPU 3, at most 2 of my processes, about 1.5 GPU-hours in total, on 2 host threads per process.

## MCTS vs one-step vs population

| metric | one-step personalized | MCTS personalized (400 sims) | MCTS population (400 sims) |
|---|---|---|---|
| picks that differ from the population drafter at the same state | 52% | 38% | |
| when they differ: personal gain / population WP cost (pp) | +3.6 / −3.0 | +3.9 (3.7, 4.1) / −1.6 (−1.8, −1.4) | |
| whole draft vs GD: gain in V over the population drafter | +5.1 (4.5, 5.9) | +5.5 (4.7, 6.4) | |
| whole draft vs GD: change in WP_pop | −3.1 | −1.9 (−2.6, −1.2) | |
| whole draft vs GD: mean V of the controlled team | 0.733 | 0.698 | 0.643 |
| whole draft vs imitation opponent: gain in V / change in WP_pop | +5.8 / −3.2 | +7.3 (6.5, 8.1) / −0.5 (−1.2, 0.2) | |
| realized: residual top minus bottom agreement quintile, all 11,448 teams | +7.6 (4.9, 10.5) | +4.7 (1.9, 7.7) | +2.1 (−0.6, 4.9) |
| realized: personal component, top minus bottom | +9.6 (6.7, 12.4) | +5.2 (2.2, 7.9) | |
| collapse: effective pool per player (median) | 10.7 | 8.9 | 9.9 |
| collapse: share of picks on the player's top-3 heroes (mean) | 28% | 26% | 13% |
| self-play: effective number of heroes (real 73.5) | 69.5 | 42.6 | 41.0 |
| self-play: share of the 10 most-picked heroes (real 25.7%) | 30.7% | 47.9% | 49.2% |
| self-play: correlation of hero shares with real drafts | 0.47 | 0.82 | 0.79 |
| off-role picks, self-play (real 15.8%) | 17.2% | 18.3% | 32.0% |
| off-role picks, vs GD (controlled team) | 26.0% | 26.3% | 32.7% |
| real picks equal to the drafter's top-1 | 12.4% | 20.2% | 14.6% |
| imitation top-1 equal to the drafter's top-1 | 14.2% | 30.7% | |

The population one-step drafter's off-role rates are 34.3% (self-play) and 35.0% (vs GD); its realized agreement is +0.1pp.

**Reading:**

- **Cheaper personal gains.** MCTS changes fewer picks than one-step (38% vs 52%). The picks it changes buy the same personal gain at about half the population-WP cost (−1.6pp vs −3.0pp per pick; −1.9 vs −3.1 per draft). Search finds personal picks that also hold up in the draft.
- **Stronger opponent-model effect.** Against the imitation opponent, the personalized MCTS drafter's predicted edge is +7.3pp at almost no WP cost.
- **Off-role handling.** The same as one-step: about 18% of personalized picks go to off-role players, against 32 to 34% for population drafters and 16% in real drafts.
- **No collapse** onto comfort heroes. The effective pool is 8.9 heroes (population MCTS 9.9, real play 11.3 at the same sample size), and only 26% of picks go to the player's top-3 heroes (43% in real play).

## Emergent meta under MCTS

Figure: `fig_mcts_meta.png`.

- **Concentration.** With the behavioral prior and argmax at the root, MCTS self-play concentrates on the prior's favorite heroes far more than one-step search or real drafts. The effective number of heroes is 42.6 (personalized) and 41.0 (population), against 73.5 in real drafts. The top 10 heroes take 48 to 49% of picks. The leaders are Rehgar (6.8% personalized, 7.4% population, real 2.3%), Valla (6.7%), Johanna (6.4%), Anduin (5.2%), Muradin, Leoric, Falstad, Thrall, Auriel and Li-Ming.
- **Role mix.** It is close to real: tanks 21%, bruisers 18%, healers 21%, ranged 33%, melee assassins 7%. Hero shares correlate 0.82 with real (one-step 0.47), because the prior is behavioral.
- **Personalization widens the pool slightly** (42.6 vs 41.0 heroes) and moves the meta toward real play (correlation 0.82 vs 0.79).
- **Risers under personalized MCTS:** Blaze (0.45% vs 0.16%), D.Va, Zagara, Orphea, Hanzo (1.48% vs 0.89%), Lúcio, Gall, Cho, Chen, Junkrat.
- **Fallers:** Uther (0.13% vs 0.25%), Mephisto, Rexxar, Raynor, Tracer, Malfurion, Illidan (1.42% vs 1.75%), Samuro, Deathwing, Ragnaros.
- **The one-step conclusion holds.** Knowing who plays lifts specialist and high-execution heroes (Hanzo, Chen, Cho and Gall, D.Va, Blaze) and trims generic picks.

## Sims curve: does deeper search buy realized value or only predicted value?

Figure: `fig_mcts_sims_curve.png`. All rows use the same 1,000 lobbies (predicted) and the same 2,000-lobby subset of 4,000 teams (realized).

| search | predicted whole-draft gain in V vs GD (pp) | change in WP_pop (pp) | realized: personalized agreement Q5 − Q1 (pp) | realized: personal component Q5 − Q1 | realized: population agreement Q5 − Q1 |
|---|---|---|---|---|---|
| MCTS 25 sims | +2.2 (1.5, 3.0) | −0.6 | −1.3 (−6.0, 3.7) | −2.9 (−7.6, 1.8) | −0.7 |
| MCTS 100 sims | +2.9 (2.2, 3.7) | −1.6 | +0.1 (−4.6, 4.7) | +2.8 (−2.1, 7.8) | −0.1 |
| MCTS 400 sims | +5.5 (4.7, 6.3) | −1.9 | +2.9 (−2.0, 7.6) | +3.7 (−0.7, 8.2) | +0.2 |
| MCTS 1,500 sims | +7.2 (6.5, 7.9) | −3.3 | +5.3 (0.9, 10.2) | +7.3 (2.4, 11.9) | +1.5 |
| one-step (≈86 rollouts per decision) | +5.1 (4.5, 5.8) | −3.1 | +7.1 (2.4, 12.1) | | |

Reading:

- **Predicted gains rise steadily with search:** +2.2pp at 25 sims to +7.2pp at 1,500. MCTS passes one-step between 100 and 400 sims.
- **Realized value rises too, with no sign of overfitting yet** over this range. The personalized-agreement signal climbs from about 0 at 25 to 100 sims to +5.3pp at 1,500, and the personal component from −2.9 to +7.3.
  - At low sims the root visit ranking is mostly the behavioral prior, so agreeing with it says little about outcomes.
  - As search deepens, the ranking follows the personal value and its realized signal approaches the one-step drafter's (+7.1pp on the same teams).
- **MCTS has not yet exceeded one-step on realized value**, even at 1,500 sims. The confidence intervals are wide (±5pp) and overlap fully.
- **Paper 1's warning applies to the predicted column.** Predicted gains outrun realized ones: MCTS at 1,500 sims claims +7.2pp against one-step's +5.1pp, while its realized signal is if anything lower. More sims beyond 1,500 would need a larger tree (the 4,096-node guard starts to bind around 2,000 to 3,000 sims) and a larger realized sample to separate the curves.

## Imitation vs outcome under MCTS

At the real states (57,240 real picks), the personalized MCTS top-1 matches the real pick 20% of the time (one-step 12%) and matches the imitation model's top-1 31% of the time (one-step 14%). The behavioral prior pulls search toward what players usually pick.

Team residual by which system the real pick matched:

| the real pick matched | picks | residual (pp) |
|---|---|---|
| both | 7,228 | +2.4 (1.3, 3.5) |
| imitation only | 11,877 | +1.7 (0.8, 2.5) |
| personalized MCTS only | 4,329 | −0.3 (−1.7, 1.2) |
| neither | 33,806 | −1.1 (−1.6, −0.5) |

Under MCTS, the picks where the outcome-based drafter departs from the imitation model do not win more than expected. For one-step the same class was +1.8 (0.2, 3.4). The extra agreement MCTS gets from the behavioral prior shifts its distinct picks toward cases with little realized value.

## Stretch: player-conditioned policy trained by self-play (not started; cost estimate)

- **Kernel work.** The policy backbone takes a fixed 290-d state, and the acting slot's personal vector would have to enter it. Threading that input through the backbone and every expansion and re-verifying is a new kernel variant, about one working day.
- **Self-play data.** Measured throughput is about 100 searched decisions per second at 400 sims on the shared GPU 3. That makes about 5.5 GPU-hours per 200K self-play drafts, and paper-1-style training needs many iterations: several GPU-days under the 2-process cap.
- **Cheaper alternative.** Distill the root visit distributions already produced here into a player-conditioned policy by supervised learning, about one GPU-hour. It would serve as a better prior than the behavioral one (the low-sims curve shows the prior dominates below about 400 sims), but it would not test self-play training. Awaiting a decision before either is started.

## Files (training/personalization/)

- `cuda_personal/`: `personal_kernel.cu`, `personal_bindings.cpp`, `setup.py`, and `ref/` (the unmodified reference kernel).
- `p3_mcts_core.py`: host side, draft-order and slot conventions, the Python leaf mirror.
- `p3_mcts_verify.py`: verification (`results/p3_mcts_verify.json`).
- `p3_mcts_drafter.py`: stages full, curve, real and collapse (`cache/mcts_*_s*.pkl.gz`, logs `results/p3_mcts_*_s*.txt`).
- `p3_mcts_analyze.py`: all MCTS tables (`results/p3_mcts_analyze.json`, `fig_mcts_sims_curve.png`, `fig_mcts_meta.png`).

# Distilled player-conditioned prior (2026-09-30)

## What was built

- **Prior.** For the acting player p at draft state x, the prior logit of hero h is α·log p_BC(h | x) + g(f(p, h)):
  - p_BC is the kernel's behavioral-cloning prior, softmaxed over the valid mask (free heroes in p's pool).
  - f(p, h) is 9 per-hero player features: the personal skill term b2·s(p, h), off-role, log(1 + games on h), never played, EWMA pick shares at half-lives of 20 and 100 games, log(1 + days since last played), never played recently, and log fine-role share.
  - g is an MLP (9 → 32 → 32 → 1) shared across heroes.
- **Why this form.** g depends only on the player, so for a lobby it is a 10×90 per-slot term that the kernel can add to its PUCT priors. The draft state enters through p_BC.
- **Targets.** Personalized MCTS root visit distributions (400 sims, BC prior) at all 56,860 real pick states of 5,686 V1 lobbies (2026-02-10 to 04-01), all ten players with 50+ games. V1 is disjoint from every V2 evaluation lobby. Split: 90% of lobbies for training and 10% for validation.
- **Test.** The 57,240 personalized MCTS searches at real states of the 5,724 V2 lobbies.
- **Compute.** 9 GPU-minutes for targets and 4 CPU-minutes for training (`p3_ds_targets.py`, `p3_ds_train.py`, `cache/ds_prior.pt`).
- **Kernel.** `cuda_prior/prior_kernel.cu` is a new copy of the personal kernel. When cfg[56] = 1 the acting slot's priors become softmax(α·log p + term) over the same valid heroes, and root priors are reported. `personal_kernel.cu` is untouched.
- **Kernel verification** (`p3_ds_verify.json`):
  - With the term off, it is identical to the personal kernel: 64 full drafts and 64 decide-only searches, with identical win probabilities, actions and root visits.
  - Root priors vs Python: max |Δ| is 2.2e-7 for the BC prior (α = 1, zero term) and 3.3e-7 for the distilled prior.

**Fit** (held-out V2 searches):

| prior | α | KL to the MCTS visit distribution | top-1 = MCTS argmax | top-1 = real pick | log-lik of the real pick |
|---|---|---|---|---|---|
| BC prior | 1 | 0.241 | 50.2% | 14.4% | −2.92 |
| α·BC (fitted α) | 1.26 | 0.218 | 50.2% | 14.4% | −2.99 |
| g only (no draft state) | | 0.876 | 17.1% | 24.5% | −2.90 |
| **distilled (α·BC + g)** | 1.23 | **0.174** | **57.0%** | 19.3% | −2.77 |

The distilled prior cuts the KL to personalized MCTS by 28% and agrees with its top pick 7 points more often than the BC prior.

## Evaluation

Setup:

- **Same data and seeds.** Same held-out lobbies, controlled teams, opponent seeds and GD cycling as the earlier runs. The realized check uses the same 2,000-lobby subset (4,000 teams) for every drafter.
- **Greedy drafters.** They pick the prior's argmax with no search.
- **Pairing.** Each personalized drafter is paired with its non-personal counterpart: greedy BC for greedy distilled, population MCTS for the MCTS drafters.

| metric | greedy BC (non-personal) | greedy distilled | MCTS personalized, BC prior | MCTS personalized, distilled prior | MCTS population |
|---|---|---|---|---|---|
| vs GD: gain in V over the non-personal counterpart | | +6.1 (5.3, 6.9) | +5.5 (4.7, 6.3) | +6.5 (5.7, 7.2) | |
| vs GD: change in WP_pop | | −0.2 (−0.8, 0.4) | −1.9 (−2.6, −1.2) | −4.1 (−4.8, −3.3) | |
| vs GD: mean V of the controlled team | 0.529 | 0.590 | 0.698 | 0.708 | 0.643 |
| vs imitation opponent: gain in V / change in WP_pop | | +5.9 / −0.1 | +7.3 / −0.5 | +8.3 / −2.5 | |
| realized: own agreement, Q5 − Q1 (pp) | +1.0 (−4.1, 5.8) | +3.8 (−1.0, 8.4) | +2.9 (−2.0, 7.9) | **+5.1 (0.0, 9.8)** | +0.2 (−4.6, 5.0) |
| realized: personal component, Q5 − Q1 | | **+14.2 (9.5, 18.9)** | +3.7 (−1.0, 8.3) | +6.3 (1.7, 11.0) | |
| collapse: effective pool (median) | 8.4 | 7.9 | 8.9 | 7.9 | 9.9 |
| collapse: share of picks on the player's top-3 heroes | 13% | 24% | 26% | 33% | 13% |
| self-play: effective number of heroes (real 73.5) | 31.6 | 33.8 | 42.6 | **43.4** | 41.0 |
| self-play: share of the 10 most-picked heroes (real 25.7%) | 58.4% | 56.4% | 47.9% | 48.2% | 49.2% |
| self-play: correlation of hero shares with real | 0.81 | 0.83 | 0.82 | **0.84** | 0.79 |
| self-play: off-role picks (real 15.8%) | 30.2% | 21.0% | 18.3% | **13.6%** | 32.0% |
| vs GD: effective heroes (controlled team) | | 48.9 | 52.4 | 53.7 | |

Additional comparisons:

- **All 5,724 lobbies (11,448 teams), greedy drafters.** Greedy distilled agreement gives +6.4pp (3.5, 9.3) Q5 − Q1; its personal component (distilled minus BC ranking) gives +12.8pp (9.8, 15.6); greedy BC agreement gives +1.2pp (−1.5, 4.0). For reference, one-step search on these teams gave +7.6pp and BC-prior MCTS +4.7pp.
- **At 100 sims** the distilled-prior MCTS realized agreement is +3.1pp (−2.2, 7.8), against +0.1pp with the BC prior at 100 sims.
- **Distilled vs BC-prior MCTS, both personalized, paired.** The distilled prior adds +1.0pp V (0.3, 1.7) vs GD and +1.0pp (0.2, 1.8) vs the imitation opponent. It costs −2.2pp of WP_pop (−2.9, −1.4).

## Reading

- **Greedy.** A greedy drafter with the distilled prior gets as much predicted personal value as full MCTS with the behavioral prior (+6.1pp vs +5.5pp), at almost no population-WP cost (−0.2pp) and with no search.
  - On real games, agreement with it separates winners from losers (+6.4pp over all teams).
  - Its personal component separates them most sharply of any drafter tested: +12.8 to +14.2pp. It is the cheapest usable personalized drafter so far.
- **The distilled prior improves MCTS on every personal metric.**
  - Predicted V: +1.0pp over the BC prior.
  - Realized agreement on the common subset: +5.1 vs +2.9pp.
  - Realized agreement at 100 sims: +3.1 vs +0.1pp. A personal prior is most useful when search is cheap.
  - Off-role picks in self-play: 13.6%, below real drafts' 15.8% and the BC-prior MCTS's 18.3%.
- **It does not widen the meta much.**
  - Self-play breadth is 43.4 effective heroes (BC prior 42.6; population 41.0; real 73.5).
  - The prior is personal, but the population WP at the leaf still pulls every team toward the same high-WP core: Valla, Johanna, Rehgar, Anduin, Muradin, Leoric, Thrall, Falstad, Li-Ming and Brightwing take 48% of picks.
  - What changes is who gets those heroes and the tail. Risers: Yrel (0.10% vs 0.02%), Uther, Ana, Lunara, Hanzo (2.19% vs 1.48%, real 2.41%), Deckard, Lost Vikings, Mei, Butcher, Diablo. Fallers: Sgt. Hammer, Xul, Rexxar, Orphea, Tyrael (1.84% vs 2.49%), Whitemane, Deathwing, Tracer, Lt. Morales, Illidan.
  - Hero shares move toward real play (correlation 0.84 vs 0.82).
- **Cost of concentration.** The personal prior concentrates each player a little more: 33% of picks on the player's top-3 heroes (BC-prior MCTS 26%, real 43%) and an effective pool of 7.9 (8.9). It also gives up more population WP than the BC-prior MCTS (−4.1 vs −1.9pp vs GD), because it steers search toward comfort picks earlier.
- **Uncertainty.** Every realized difference between drafters is within the confidence intervals (about ±5pp per estimate on 4,000 teams). Only the greedy distilled personal-component signal, and the gap between personalized and non-personal agreement, are clearly separated.

Figure: `fig_ds_meta.png`. Panels: distilled-prior vs BC-prior MCTS self-play hero shares; distilled-prior self-play vs real; meta breadth of all drafters.

## Files (training/personalization/)

- `cuda_prior/`: `prior_kernel.cu`, `prior_bindings.cpp`, `setup.py`.
- `p3_ds_common.py`: prior model, features, kernel-format state.
- `p3_ds_targets.py`, `p3_ds_train.py`: targets and fit (`results/p3_ds_train.json`).
- `p3_ds_mcts.py`: verification and MCTS runs with the distilled prior (`results/p3_ds_verify.json`, `cache/dsmcts_*.pkl.gz`).
- `p3_ds_greedy.py`: greedy drafters (`cache/dsgreedy_*.pkl.gz`).
- `p3_ds_analyze.py`: the tables above (`results/p3_ds_analyze.json`, `fig_ds_meta.png`).
