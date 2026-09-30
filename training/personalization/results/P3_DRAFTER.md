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
