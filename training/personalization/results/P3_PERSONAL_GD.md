# P3: personalized generic-draft model (2026-10-01)

**Question.** Given who is on both teams and the current meta, what gets banned and picked at each draft step? And how does the personalized MCTS change when opponents are modeled as themselves?

## Summary

- **Picks: personalization is most of the predictable signal.** On held-out V2 picks, top-1 accuracy rises from 8.9% (causal meta GD) to **32.7%**, and log loss falls from 3.82 to **2.57**. On post-May games, the clean out-of-sample test for every model, it rises from 8.5% to **33.2%** (log loss 3.85 to 2.55). The model beats the imitation model by 0.6 to 0.9 points of top-1 and 0.03 to 0.04 of log loss.
- **One-tricks are modeled correctly.** With his main available, a one-trick picks it 67.6% of the time. The model predicts 68.6%, against 64.2% for imitation and 2.4% for GD (the failure behind the old ban result).
- **Accuracy by history depth.** Top-1 is 7.4% with no history (as for GD), 29.5% at 1 to 20 games, 33.2% at 21 to 100 and 34.2% at 100+. One-tricks reach 68.7%, specialists 43.6% and flexible players 26.0%.
- **Clusters do not buy anything measurable.** At 1 to 20 games, the cluster version has log loss 2.996 against 3.004 without clusters (top-1 29.5% vs 29.7%). At 0 games there is no history to place a player in a cluster, so both fall back to the population. Overall the no-cluster variant is slightly better (2.556 vs 2.573).
- **Bans: identity barely improves prediction.** V2 top-1 is 16.5% vs 16.4% for the meta GD (log loss 3.468 vs 3.482). With a one-trick opponent still to pick: 15.6% vs 15.4%. Real teams target mains, but such bans are a small share of all bans, and most bans follow the meta.
- **In the drafter (MCTS, 400 sims, 2,000 held-out lobbies):**
  - Modeling opponents as themselves changes 24% of personalized pick recommendations and 30% of ban recommendations.
  - The predicted V of the controlled team falls from 0.691 to 0.616, because opponents get their comfort heroes. The predicted gain over the population drafter under the same opponent model rises from +6.6 to +8.2pp per draft.
- **Redone ban value inside MCTS** (personalized vs population ban at the same real state, personalized tree's Q):
  - with GD opponents: +0.53pp per ban decision (+0.62 with a one-trick opponent still to pick);
  - with personal-GD opponents: +0.63pp (+0.83);
  - adding the personal-GD prior: +0.83pp (+1.53).

  These are a fifth to a quarter of the +3.4pp from the closed-form ban model (P3_EXTENSIONS section 8). The search finds part of that value. The share of recommended bans that hit a remaining opponent's main rises from 10.6% to 20.0% (real teams: 13.3%).
- **A personal-GD prior for our own picks makes self-play much broader.** It gives 60.9 effective heroes (BC prior 41.5, population 38.6, real 73.5), correlation with real hero shares 0.77 (vs 0.65), and off-role picks of 7.5% (vs 20.0%).

Drafter values are model-based. Following AUDIT_P3 A1, the realized-agreement check is not used as evidence here. The non-circular evidence is the prediction accuracy of real opponent actions on held-out drafts in part 1.

## 1. The model

**Data** (`p3_pgd_data.py`).

- **Coverage.** 1,092,762 games with standard 16-step drafts whose picks join to the ten players:
  - snapshot, 2024-04 to 2026-05-22: 866,977 games (80% of snapshot games have all six bans recorded);
  - post-snapshot, 2026-05-23 to 2026-09-27 (builds 2.55.16.97039 and 2.55.17.*): 225,785 games. 10,819 games on Haunted Mines, outside the 14 maps GD encodes, are excluded.
- **Checks.** Every kept game was checked against the Storm League step order (first team f: bans f o f o; picks f o o f f; bans o f; picks o o f f o). Picks come from pick order, bans from the draft order.

**Base: causal meta GD** (`p3_pgd_gd.py`).

- **Why a new base.** The paper-1 GD was trained on a random 98% of the whole snapshot, so it is in sample for V2.
- **Model.** The base keeps the GD architecture and draft-state input (picks, bans, map, tier, step). It adds the hero's pick rate and ban rate per game for the tier over the 28 days ending the day before the game.
- **Training.** 736,075 games before 2026-02-10, all 16 steps, early-stopped on a 2% game holdout.
- **Accuracy.** It is close to the paper-1 GD: picks top-1 8.9% vs 9.1% on V2 and 8.5% vs 8.6% on post; bans top-3 36.0% vs 34.8% on V2. It is strictly causal.

**Personal heads** (`p3_pgd_feats.py`, `p3_pgd_model.py`):

- **Pick:** u(h) = α_p · log p_metaGD(h | state) + f_p(acting player's context for h), with α_p = 0.90.
- **Ban:** u(h) = α_b · log p_metaGD(h | state) + f_b(both teams' players still to pick, h), with α_b = 1.02.

**Player context.** Causal, from games that ended before the draft started:

- games on each hero;
- EWMA pick shares at half-lives of 20 and 100 games;
- days since last played;
- Max's MAWP, exactly as in `src/lib/mawp.ts` (via `p3_x_ban_feat._mawp`);
- main and main share.

**Preference structure**, fit on pre-cutoff players with 100+ games (22,666 players):

- a rank-12 SVD embedding of volume-centered log(1 + games), the P3_PREF_SIMILARITY definition;
- 12 k-means clusters. Examples: a tank cluster (Diablo, Muradin, Johanna, Garrosh, Stitches); a healer cluster (Anduin, Brightwing, Rehgar, Auriel, Stukov); a mage cluster (Li-Ming, Valla, Kael'thas, Chromie); a bruiser cluster (Dehaka, Leoric, Hogger, Thrall, Blaze); a pusher cluster (Nazeebo, Azmodan, Zagara); a mechanical cluster (Hanzo, Genji, Sylvanas); an odd-kit cluster (Alarak, Kel'Thuzad, Illidan, Abathur, D.Va).

**Smooth cold start.** A multinomial posterior over clusters given the player's counts gives a cluster share. With no games it is the population share. Features:

- shrunken share (n_h + 10 · cluster share) / (n + 10);
- cluster share;
- embedding affinity (rank-12 reconstruction of the player's centered log counts, damped by n / (n + 20));
- log games on the hero, never played;
- EWMA shares;
- recency;
- shrunken fine-role share;
- main × main share;
- MAWP − 0.5;
- volume.

**Who acts.** At a pick step, the player making that pick. At a ban step, both teams' players still to pick in that phase:

- opponents, for targeting: max and sum of shrunken share, max main share, max share-weighted MAWP, max EWMA share, how many have 10+ games on the hero, max affinity;
- the banning team's own players, for protection: max and sum of shrunken share, max main share.

f_p and f_b are MLPs (64-64). Both personal terms are fixed within a draft phase, so the MCTS kernel takes them as per-slot and per-team-per-phase vectors.

**Training and testing.**

- **Train:** 50,000 games from 2025-04-01 to the cutoff (500,000 picks, 300,000 bans).
- **Validation:** 5,000 other pre-cutoff games.
- **Test:** 25,000 V2 games and 25,000 post-snapshot games.
- **Ablation:** the same model with clusters and embedding removed (shrinkage toward the population share).

## 2. Prediction accuracy (held-out real drafts)

**Picks.** Cells show top-1 / top-5 / log loss. Paper-1 GD is in sample on V2.

| V2 picks | n | paper-1 GD | meta GD | imitation | personal GD | personal GD, no clusters |
|---|---|---|---|---|---|---|
| all | 249,996 | 9.1 / 30.3 / 3.779 | 8.9 / 30.0 / 3.816 | 32.1 / 65.8 / 2.604 | **32.7 / 66.4 / 2.573** | 33.3 / 66.9 / 2.556 |
| history 0 | 4,027 | 7.8 / 25.7 / 3.959 | 7.3 / 25.4 / 3.976 | 7.8 / 25.7 / 3.952 | 7.4 / 25.4 / 3.968 | 7.0 / 25.4 / 3.979 |
| history 1-20 | 42,151 | 8.2 / 28.1 / 3.865 | 8.0 / 28.0 / 3.891 | 29.2 / 55.9 / 3.035 | **29.5 / 56.4 / 2.996** | 29.7 / 56.2 / 3.004 |
| history 21-100 | 69,149 | 8.8 / 29.7 / 3.796 | 8.7 / 29.3 / 3.830 | 32.8 / 66.9 / 2.600 | 33.2 / 67.1 / 2.576 | 33.9 / 67.6 / 2.564 |
| history 100+ | 134,669 | 9.5 / 31.4 / 3.737 | 9.3 / 31.1 / 3.781 | 33.4 / 69.4 / 2.431 | 34.2 / 70.3 / 2.397 | 34.9 / 71.2 / 2.370 |
| one-trick | 14,811 | 6.6 / 25.5 / 3.953 | 6.4 / 24.9 / 3.985 | 68.2 / 92.1 / 1.217 | 68.7 / 92.2 / 1.176 | 69.2 / 92.3 / 1.164 |
| specialist | 50,681 | 9.3 / 31.2 / 3.760 | 8.9 / 30.9 / 3.799 | 42.9 / 80.9 / 2.030 | 43.6 / 81.6 / 1.991 | 44.4 / 81.9 / 1.973 |
| flexible | 126,007 | 9.6 / 31.5 / 3.726 | 9.6 / 31.2 / 3.770 | 25.4 / 61.2 / 2.799 | 26.0 / 62.0 / 2.774 | 26.7 / 63.0 / 2.747 |

Type definitions (players with 30+ games): one-trick = main ≥ 50% of games, specialist = 25 to 50%, flexible = under 25%.

| post-snapshot picks (out of sample for all models) | n | paper-1 GD | meta GD | imitation | personal GD | no clusters |
|---|---|---|---|---|---|---|
| all | 249,999 | 8.6 / 28.9 / 3.824 | 8.5 / 28.9 / 3.853 | 32.3 / 66.2 / 2.587 | **33.2 / 67.0 / 2.546** | 33.8 / 67.7 / 2.525 |
| history 0 | 3,550 | 7.3 / 25.0 / 3.991 | 6.6 / 24.9 / 4.014 | 7.3 / 25.0 / 3.980 | 6.6 / 25.1 / 4.006 | 7.1 / 25.0 / 4.014 |
| history 1-20 | 38,059 | 7.9 / 27.2 / 3.896 | 7.8 / 27.2 / 3.920 | 29.3 / 56.5 / 3.026 | 29.8 / 56.9 / 2.983 | 30.0 / 56.9 / 2.987 |
| history 21-100 | 69,584 | 8.5 / 28.5 / 3.839 | 8.3 / 28.5 / 3.866 | 32.9 / 67.4 / 2.582 | 33.8 / 67.8 / 2.552 | 34.3 / 68.4 / 2.535 |
| history 100+ | 138,806 | 8.9 / 29.7 / 3.792 | 8.8 / 29.8 / 3.825 | 33.5 / 69.3 / 2.433 | 34.5 / 70.4 / 2.386 | 35.2 / 71.3 / 2.355 |
| one-trick | 14,869 | 7.3 / 25.3 / 3.967 | 6.6 / 25.0 / 3.990 | 68.3 / 92.2 / 1.216 | 69.1 / 92.3 / 1.173 | 69.2 / 92.5 / 1.163 |
| specialist | 54,062 | 9.0 / 29.7 / 3.804 | 8.7 / 29.7 / 3.832 | 43.0 / 81.0 / 2.035 | 43.9 / 81.7 / 1.988 | 44.5 / 82.2 / 1.966 |
| flexible | 127,889 | 8.9 / 29.7 / 3.786 | 9.0 / 29.8 / 3.819 | 25.2 / 61.1 / 2.798 | 26.3 / 62.1 / 2.760 | 26.9 / 63.2 / 2.727 |

**One-trick main, when available** (probability of picking it):

| | real | personal GD | no clusters | imitation | GD |
|---|---|---|---|---|---|
| V2 | 0.676 | 0.686 | 0.682 | 0.642 | 0.024 |
| post | 0.681 | 0.693 | 0.689 | 0.653 | 0.025 |

**Bans.** Cells show top-1 / top-3 / log loss.

| V2 bans | n | paper-1 GD | meta GD | personal GD | no clusters |
|---|---|---|---|---|---|
| all | 150,000 | 16.2 / 34.7 / 3.407 | 16.4 / 35.9 / 3.482 | 16.5 / 36.2 / 3.468 | 16.6 / 36.3 / 3.469 |
| opponents' mean history 0-20 | 4,642 | 18.4 / 37.5 / 3.324 | 18.4 / 37.8 / 3.373 | 18.3 / 38.1 / 3.372 | 18.5 / 38.2 / 3.372 |
| opponents' mean history 21-100 | 35,663 | 16.3 / 35.2 / 3.387 | 16.4 / 36.5 / 3.416 | 16.6 / 36.7 / 3.415 | 16.8 / 36.9 / 3.412 |
| opponents' mean history 100+ | 109,695 | 16.1 / 34.5 / 3.417 | 16.3 / 35.6 / 3.508 | 16.4 / 36.0 / 3.490 | 16.4 / 36.0 / 3.492 |
| a one-trick opponent still to pick | 31,482 | 15.2 / 33.1 / 3.473 | 15.4 / 34.9 / 3.529 | 15.6 / 35.2 / 3.511 | 15.7 / 35.3 / 3.510 |
| specialist opponents (no one-trick) | 68,582 | 15.6 / 34.1 / 3.430 | 15.9 / 35.5 / 3.503 | 16.0 / 35.9 / 3.488 | 16.1 / 35.9 / 3.489 |
| flexible opponents only | 46,560 | 17.6 / 36.5 / 3.339 | 17.4 / 37.0 / 3.429 | 17.5 / 37.2 / 3.420 | 17.6 / 37.2 / 3.422 |
| post-snapshot, all | 150,000 | 15.2 / 32.3 / 3.688 | 14.5 / 32.3 / 3.733 | 14.9 / 32.6 / 3.719 | 14.8 / 32.6 / 3.722 |
| post-snapshot, one-trick opponent | 31,538 | 13.7 / 30.9 / 3.775 | 13.2 / 30.9 / 3.798 | 13.6 / 31.3 / 3.782 | 13.6 / 31.3 / 3.784 |

**Reading:**

- **Picks.** Identity is worth 21 to 25 points of top-1 at 1+ games of history and 62 points for one-tricks. Most of the gain is already in the imitation model; the personal GD adds 0.5 to 1 point and calibrates mains (0.69 vs 0.68 real).
- **Clusters.** The ablation shows the preference clusters and embedding do not help where they were meant to. Even at 1 to 20 games the player's own counts and recency carry the signal (cluster gain: −0.2 points of top-1, +0.008 log loss). With no history there is nothing to place a player in a cluster with. Real cold start (0 games) is about 1.5% of slots; for those, both variants equal the population model. Preference structure is very stable across players (split-half 0.98), but for an individual its information is already in his own few games.
- **Bans.** They are mostly meta. The personal ban head improves log loss by only 0.014 to 0.018 over the meta GD and does not reach the paper-1 GD's ban log loss (3.41 on V2, in sample; 3.69 on post), which has much more training data. Targeting mains is real (13.3% of real bans hit a remaining opponent's main, against 10.4% for the population ban choice in MCTS) but too rare to move accuracy.
- **Base model.** The causal meta GD is a little worse than the paper-1 GD even out of sample (post log loss 3.85 vs 3.82 for picks, 3.73 vs 3.69 for bans). The meta-rate inputs do not make up for less training data. A personal head on the paper-1 GD would likely be slightly more accurate; for V2 it would be in sample.

## 3. The personalized GD as the opponent model in the personalized MCTS

**Kernel** (`cuda_pgd/pgd_kernel.cu`). A new copy of `cuda_personal`:

- The engine's GD weights are the meta GD in the kernel layout (shared hero order). Its meta inputs are folded into a per-lobby first-layer offset.
- With cfg[56] = 1, every GD use (the in-tree opponent, rollouts for both teams, the opponent in the main draft) becomes α · log softmax + the personal term. The term is the acting slot's pick vector, or the (team, ban phase) vector at ban steps.
- With cfg[57] = 1, our own PUCT priors also come from the personal GD.

**Verification** (`p3_pgd_verify.json`):

- With both flags off, it is identical to the personal kernel (actions, WP and root visits on 64 episodes).
- The kernel's personal-GD probabilities match Python to 3.3e-7 over 336 real pick and ban states.

**Protocol.** Audit fixes: pools are the heroes played before the game day (the real hero is no longer added), and no real bans are forced (our bans are searched; the other team's come from the GD). 400 sims, argmax root, same lobbies and seeds as before.

| configuration | value | prior | opponent, in-tree and rollout model |
|---|---|---|---|
| R1 | personal | BC | paper-1 GD (previous model, new protocol) |
| R2 | personal | BC | personal GD |
| R3 | personal | personal GD | personal GD |
| P1 | population | BC | paper-1 GD |
| P2 | population | BC | personal GD |

**Recommendation changes at real states** (2,000 lobbies, 20,000 pick and 12,000 ban states):

| | picks | bans |
|---|---|---|
| R2 vs R1: recommendation differs | 23.8% | 30.2% |
| R3 vs R1: recommendation differs | 64.9% | 58.3% |
| P2 vs P1 (population drafter) | 22.3% | 24.1% |
| recommendation equals the real action: R1 / R2 / R3 | 19.6 / 19.6 / 33.7% | 15.1 / 15.3 / 15.3% |

**Personal gain per decision** (Q in the personalized tree, personalized choice minus the population choice under the same opponent model):

| | picks | bans | bans, one-trick opponent with main free (20% of states) | bans, otherwise | recommended ban hits a remaining opponent's main |
|---|---|---|---|---|---|
| R1 vs P1 (GD opponents) | +1.71pp (1.66, 1.77) | +0.53 (0.51, 0.55) | +0.62 (0.56, 0.67) | +0.51 | 10.6% (population 10.0%) |
| R2 vs P2 (personal-GD opponents) | +1.69 (1.64, 1.74) | **+0.63** (0.61, 0.66) | **+0.83** (0.75, 0.91) | +0.58 | 12.8% (population 10.4%) |
| R3 vs P2 (+ personal-GD prior) | +2.00 (1.93, 2.08) | **+0.83** (0.79, 0.88) | **+1.53** (1.39, 1.67) | +0.66 | 20.0% |

Real bans hit a remaining opponent's main 13.3% of the time.

**Whole drafts** (1,000 lobbies, controlled team vs the configuration's own opponent model, paired with the population drafter under the same opponent model):

| | gain in V | change in WP_pop | mean V of the controlled team |
|---|---|---|---|
| R1 vs P1 | +6.6pp (5.9, 7.3) | −0.8 | 0.691 |
| R2 vs P2 | +8.2pp (7.4, 9.1) | −0.4 | 0.616 |
| R3 vs P2 | +12.6pp (11.6, 13.5) | −4.0 | 0.659 |

**Self-play meta:**

| | R1 | R2 | R3 | P1 | P2 | real |
|---|---|---|---|---|---|---|
| effective heroes | 41.5 | 41.9 | **60.9** | 38.6 | 38.7 | 73.5 |
| share of the 10 most-picked heroes | 49.6% | 49.0% | 37.1% | 52.0% | 51.5% | 25.7% |
| correlation with real hero shares | 0.65 | 0.67 | **0.77** | 0.62 | 0.62 | |
| off-role picks | 20.0% | 19.6% | **7.5%** | 33.6% | 33.6% | 15.8% |

**Reading:**

- **Opponents as themselves.** About a quarter of pick and a third of ban recommendations change. Predicted absolute value is more realistic: the controlled team's V drops 7.5pp because opponents now play their comfort heroes. Relative to a population drafter facing the same opponents, the personalized drafter's predicted edge grows (+8.2 vs +6.6pp per draft), because it can plan around who the opponents are.
- **Ban value under MCTS is much smaller than the closed-form +3.4pp.** It is +0.63pp per ban with personal-GD opponents and +0.83 to +1.53pp (one-trick opponents) with the personal prior. Two reasons:
  - The search compares two complete bans inside a model where the replacement pick is also searched and opponents re-optimize. The closed form assumed no adaptation beyond the next pick.
  - With the BC prior, main bans get little prior mass, so search rarely explores them. The personal-GD prior doubles the one-trick case.
  
  So the ban value is real but smaller once opponents adapt. A 400-sim search still under-explores it.
- **A personal prior fixes the narrow meta.** With priors from the personal GD (R3), self-play uses 61 effective heroes, tracks real hero shares (0.77) and puts fewer players off role than real drafts do (7.5% vs 15.8%). R3's recommendations also match real picks 34% of the time (vs 20%), as expected from a behavioral prior. Its larger predicted gain (+12.6pp at −4.0pp WP_pop) leans more on the personal value model, so it inherits that model's calibration limits (AUDIT_P3 A1).

## Caveats

- Drafter values are model-based, and the realized-agreement check is not used.
- The paper-1 GD remains a slightly more accurate base than the causal meta GD. A personal head on it is a cheap improvement for production, where the V2 in-sample issue does not matter.
- The ban head reads only the players still to pick; it ignores the draft-specific comp.
- MCTS runs use one opponent model at a time; the search's model and the simulated opponents are the same.
- The no-cluster variant is slightly better overall. I kept the cluster variant as the main model because it is the one asked for and the differences are small.

## Files (training/personalization/)

- `p3_pgd_data.py`: unified draft table (`cache/pgd_games.npz`).
- `p3_pgd_common.py`: state encoding, meta rates, windows, scoring.
- `p3_pgd_gd.py`: causal meta GD (`cache/pgd_metagd.pt`, `results/p3_pgd_gd.json`).
- `p3_pgd_feats.py`: player-context walker (exact MAWP), preference embedding and clusters.
- `p3_pgd_model.py`: personal pick and ban heads, ablation, breakdowns (`cache/pgd_personal.pt`, `results/p3_pgd_model.json`).
- `cuda_pgd/`: kernel copy with the personalized GD.
- `p3_pgd_mcts.py`: verification and MCTS runs (`results/p3_pgd_verify.json`, `cache/pgdmcts_*.pkl.gz`).
- `p3_pgd_analyze.py`: MCTS tables (`results/p3_pgd_mcts.json`).

To rerun, from `training/` with `/usr/bin/python3`, 4 threads, `nice -n 19 taskset -c 48-63`, in this order: data (1 min), gd (18 min), model (80 min), mcts verify, real and full (1 GPU process, 16 min), analyze.
