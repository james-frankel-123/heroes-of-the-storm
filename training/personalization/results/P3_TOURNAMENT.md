# P3 tournament: player-aware vs player-blind drafters, and drafters under drift (started 2026-10-19)

Status: results on current data (store full_1017, CAP_TS 2026-09-28), all from the race-fixed kernels (2026-10-19, see "Kernel fix" below). The full-data rerun after the player-history pull (about Oct 26 to 27) is one command: `training/personalization/remote/p3t_pipeline.sh`. The pre-stated comparisons below were committed before the tournament ran (commits af7427c, 5194ad6, 7ffda4c).

**Kernel fix (2026-10-19).** An independent verification of the search kernels (`P3_MCTS_VERIFICATION.md`) found a race in the main draft loop of all five P3 kernels (`cuda_personal`, `cuda_prior`, `cuda_pgd`, `cuda_pvnet`, `cuda_explore`): threads read the shared skip and team flags after thread 0 had rewritten them for the next step. It bites only when steps are skipped (prefix replay, forced bans). It is fixed (local copies behind a barrier, commit 2ad31c1), and every number in this file comes from a rerun on the fixed kernels: every tournament pair, the tune set, the probe, the distilled, rich and self-play priors, the agreement backtests, the paper-1 bench and the timings.
- Checks after the fix: identical reruns are byte-identical (decide-only searches at steps 2, 7 and 13 in all four tournament kernels); full drafts with two forced bans repeat no hero; the parity checks against the population reference pass (`p3t_determinism.py`, `p3t_verify.py`, `p3t_pvnet_verify.py`, `p3t_explore_verify.py`). The verifier reproduced these on its own build.
- Smaller fixes in the same commit: hashed per-episode random seeds (neighbouring seeds were correlated; the opponent draw is now uniform, chi² p 0.42), a floor on the prior before its log so a bias can revive an underflowed hero, the rich prior no longer overwrites the ban bias, and score files record the judge they were scored with so stale scores are recomputed.
- Open, very low severity: in full-draft sampling (self-play and forced-ban runs only, never in a tournament decision) a rounding overrun falls back to hero 0, about once per 10^7 draws (0.03 expected in the whole self-play run). TODO (tournament lane, before the Oct 26 to 27 rerun): switch those samplers to `v2_sample`, which falls back to the last hero with positive mass, then rerun the determinism and parity checks.
- The paper-1 kernel `cuda_mcts` had a separate stream-ordering bug in a build no job used (fixed in b995800); nothing here uses that kernel's engine.

## Primary comparison (stated before the tournament ran)

- **Contrast:** the player-aware search drafter `aMCTS-cur` against its player-blind twin `pMCTS-cur`, head to head on the same real lobbies, each seat played both ways.
  - Same kernel, prior policy seeds, leaf WP, GD opponent model, simulations (400) and kernel seeds.
  - They differ only in the personal terms of the leaf value.
- **Metric:** mean probability that `aMCTS-cur` wins, averaged over both seatings. The judge is the player-aware judge J:
  - consensus of paper 1's composition-corrected judges;
  - plus the personal term from the skill model refit on the other player half, with evidence from the judge window only (2026-07-01 to 09-11, after every lobby).
- **Interval:** 95% CI by lobby-cluster bootstrap, widened for seed-to-seed variation by the crossed seed random effects of the drift paper (prior seed of each side).
- **Reported next to it, not instead of it:**
  - the same contrast under the population-only consensus judge;
  - the judge-free scores from later real games (future hero win rates, synergy, counters; each player's realized record on the assigned hero);
  - the drafters' own valuations (model-internal).
  - Every number is labeled as model-judged, judge-free (realized later outcomes) or model-internal.

**Secondary comparison** (added before the tournament ran, after a decision probe; see "Search prior" below):
- Contrast: `aMCTS-bc` against `pMCTS-bc`, the same search over the behavioral (BC) prior.
- Metric, judge and interval: as for the primary comparison.

**Exploration design study** (stated before any design-study system played a tournament lobby):
- **Tuning.** Every knob is tuned on a separate set of 300 tune lobbies, never the tournament lobbies. These are tournament-window lobbies that meet the same eligibility rules.
  - Each configuration plays `pMCTS-cur` and `aMCTS-pp` on both seats.
  - The selection score is its mean J win probability over those four matchups.
- **Primary design comparison.** The selected design ("best design") against `aMCTS-pp` on the 300 tournament lobbies, with J and the same interval rule as above.
- **Ablation table.** Each component added to `aMCTS-pp` alone: c_puct, prior mixing weight per pick position, personal bias scale, optimism and pessimism from posterior sd, assignment-aware root bias, opponent personal model inside the search, ban denial bias, and the rich learned prior by training signal.
  - Each row reports judged win rate against `aMCTS-pp` with CI, compute per draft, and training GPU-hours.
- **Split-history judge.** Versions using a drafter state built from odd days only are also scored by the split-history judge (even days).

## Summary

- **How much better is player-aware drafting?** On 300 real lobbies (May 23 to June 30, 2026), the selected player-aware search drafter (`x-combo1`) beats paper 1's player-blind MCTS 60.8% of the time (60.2, 61.5) under the player-aware judge J. It beats the real human drafts of the same lobbies 61.8% (61.1, 62.5).
  - The same design at 1,600 simulations (`x-combo3`, selected before the kernel fix) scores the same: 60.7% (60.1, 61.3) and 62.6% (62.0, 63.2).
  - These are model-judged and conservative. J sees players only through 2.5 months of later games, so it credits about 0.42 of a true personal difference.
  - A judge built from years of disjoint history (split-history, attenuation 0.71) puts `x-combo3`, run on half of each player's history, at 65.7% (65.0, 66.4) against paper 1's MCTS.
  - The judge-free check agrees: the players `x-combo1` puts on heroes later win more on those heroes than the players paper 1's MCTS assigns, +4.7pp per player (2.7, 6.6).
  - Under paper 1's own judges (CONS), the selected design beats real drafts 58.7% and paper 1's MCTS beats them 63.5%, a figure that reproduces paper 1.
- **Adding the personal term to the leaf of a blind-trained search does almost nothing.** The pre-stated primary, `aMCTS-cur` vs `pMCTS-cur`, comes in at 51.4% (50.8, 52.0).
  - Paper 1's F policy is so sharp that the leaf moves 4% of picks.
  - The players have to shape the exploration (the prior), not only the scoring:
    - BC prior with a personal leaf: 57.6% (57.0, 58.2) over its blind twin (the pre-stated secondary);
    - player-feature prior distilled from aware search: 60.4% over paper-1 MCTS;
    - ten-player exploration on top (opponent comfort inside the tree, denial bans, assignment-aware root): +1.6pp against that prior on the tournament lobbies (51.6%, CI 51.0 to 52.1).
- **Population vs personal.** Paper 1's blind MCTS gains 13pp of population win probability over real drafts (consensus 63.5%). It gives almost all of that back by putting players on heroes they are bad at: J has it at 50.5% against real drafts.
  - The best player-aware drafters keep a population edge over real drafts (consensus 55 to 60%) and add the personal edge.
- **Drift.**
  - Prediction: a 1-year-stale WP keeps 72% of its log-loss lift. A frozen player state is far costlier: the full value keeps 30% of its lift with player state frozen 1 year, and 43% at 4 months.
  - Drafters: a 1-year-stale paper-1 MCTS loses 2.2pp (CONS) and 3.4pp (J) to its retrained twin. Refreshing its statistics tables recovers nothing; a 2 GPU-hour fine-tune leaves 1.7 and 1.3pp. Freezing the player state costs the distilled-prior drafter 1.7pp (J) at 4 months and 3.8pp at a year, with no population cost, and updating the state is CPU-cheap.
  - The drafters lose a smaller share of their edge than the prediction models lose of their lift at the same staleness.
- **DraftRec**, rebuilt on our data as a reviewer baseline, is strong: 36% top-1 pick accuracy against 9% for GD. Its paper's p+v rule ranks just below the distilled-prior drafters under J (BT 0.57 vs 0.69 for `aMCTS-pp` and 0.74 for `x-combo1`).
- **Backtests on real games:**
  - Teams whose real pick matched the player-aware search's recommendation won 0.9pp more (SE 0.4), with draft-only WP and causal MMR controlled. That is what the skill model predicts: with its predicted gap added, the residual is −0.5 (0.4).
  - When a player's best hero is banned, the realized cost is −2.1pp (0.2) against −1.3pp predicted. Forced switches cost more than the model says.

All numbers below are labeled: **J** (player-aware judge, model-judged, primary), **CONS** (population consensus judges, model-judged), **judge-free** (realized outcomes of later games), **own** (the drafter's own value, model-internal).

## Reconciliation with paper 1 (sanity gate, `p3t_sanity.py`)

**Metric.**
- A head-to-head number is the mean judged probability that A's final draft beats B's in the same lobby, averaged over the 300 lobbies and both seatings. The judge gives P(team 0 wins), symmetrized over team order.
- This is paper 1's tournament metric. It is not a share of lobbies won.
- CONS is paper 1's composition-corrected consensus (gN, gN-naive, RN, QM2026). J adds the personal term.

**Paper 1's benchmark through this tournament's code** (1,000 configs, mid tier, random maps, F seeds pooled vs the GD pool sampled, no pools, no mask, 200 sims, root temperature 1):

| | gN | RN | QM26 | degenerate |
|---|---|---|---|---|
| paper 1 Table (F) | .662 | .641 | .669 | 4.5% |
| reproduced | .655 | .636 | .665 | 5.2% |
| this tournament's settings (400 sims, argmax, mask) | consensus .656 | | | 1.6% |

**On the 300 real lobbies, under paper 1's consensus judges:**
- blind paper-1 MCTS vs GD: 63.0% (gN 64.3, RN 62.7, QM26 62.1);
- vs the real drafts: 63.5% (gN 63.4, RN 61.6, QM26 64.8);
- real drafts vs GD: 50.2%.

Paper 1's headline 0.61 is a mean over a field of ten agents, and paper 1 made no comparison with real drafts.

**Why the margin over real drafts shrinks when players count.**
- Paper 1's MCTS puts players on heroes they are bad at. Its teams' judge skill sum is 11.4pp below the real teams' (14.4pp in the drafter's own state).
- In later real games, the players it assigns realize 3.9pp less per player on those heroes than the real teams' players (judge-free, CI −6.0 to −1.8).
- Under J it ties the real drafts (50.5%). The player-aware drafters keep most of the population edge and add the personal one: 58.7% (CONS) and 61.8% (J) for `x-combo1`, 59.5% and 62.6% for `x-combo3`.

**Checks that this is not an artifact.**
- **J's scale.** On all 60,500 real tournament-window drafts, J predicts the REAL outcomes better than CONS:

  | | log loss | accuracy | AUC | calibration slope |
  |---|---|---|---|---|
  | CONS | 0.6710 | 59.1% | 0.629 | 1.73 |
  | J | 0.6564 | 61.1% | 0.655 | 1.30 |

  A slope above 1 means underconfident, so J's personal differences are if anything too small.
- **Pools.** In the tournament lobbies, 2.9% of real picks are heroes new to the player and 0.07% fall outside the team's pool union, whose median is 87 of 90 heroes. The pool rule does not handicap the drafters against real drafts.
- **Same terms.** Real drafts are scored by the same judges with the same best hero-to-player assignment, and their rule-degenerate teams (8.5%) go through the same composition correction.
- **Caveat.** gN and RN were trained on paper 1's post-snapshot no-drift games, which include window games on build 97039. CONS's accuracy on the real window drafts is therefore partly in-sample. This does not touch drafted teams or J's personal term, which uses later games only.

## Results

### Head to head (300 tournament lobbies, both seats, 95% CI)

| matchup | J | CONS | personal part, judge (pp of team skill sum) | judge-free: realized record diff per player (pp) | judge-free: future hero WR diff |
|---|---|---|---|---|---|
| **primary:** aMCTS-cur vs pMCTS-cur | **51.4 (50.8, 52.0)** | 49.9 (49.7, 50.2) | +1.3 (1.0, 1.6) | +0.3 (−1.1, 1.8) | +0.0 |
| **secondary:** aMCTS-bc vs pMCTS-bc | **57.6 (57.0, 58.2)** | 49.5 (49.1, 50.0) | +7.3 (6.8, 7.8) | +2.1 (0.1, 4.2) | +0.1 |
| a1step-cur vs p1step-cur | 55.4 (54.6, 56.1) | 48.8 (48.3, 49.2) | +6.0 (5.4, 6.5) | +1.8 (−0.4, 4.0) | −0.2 |
| aMCTS-pp vs pMCTS-cur | 60.4 (59.7, 61.0) | 44.2 (43.8, 44.6) | +14.3 (13.6, 14.9) | +2.4 (0.3, 4.5) | −1.0 |
| aMCTS-pp vs aMCTS-bc | 56.5 (55.9, 57.1) | 54.1 (53.7, 54.6) | +2.1 (1.7, 2.6) | +1.2 (−0.3, 2.7) | +1.1 |
| aMCTS-sp vs pMCTS-cur | 57.4 (56.0, 58.9) | 49.3 (48.3, 50.2) | +7.3 (6.7, 7.9) | +3.8 (1.9, 5.7) | +0.1 |
| aMCTS-sp vs aMCTS-pp | 44.9 (44.4, 45.5) | 55.1 (54.8, 55.5) | −9.0 | −0.9 | +1.0 |
| aMCTS-vn vs aMCTS-bc | 51.1 (50.6, 51.6) | 49.8 (49.4, 50.1) | +1.2 | +0.6 | −0.0 |
| aPol-pp vs pPol-cur (no search) | 57.7 (57.0, 58.4) | 42.3 (41.7, 42.8) | +13.6 | +4.2 (2.3, 6.0) | −1.1 |
| x-combo1 vs pMCTS-cur (selected design) | 60.8 (60.2, 61.5) | 43.8 (43.1, 44.4) | +15.2 | +4.7 (2.7, 6.6) | −1.0 |
| x-combo1 vs Real | 61.8 (61.1, 62.5) | 58.7 (58.2, 59.2) | +3.0 | +0.9 (−0.6, 2.4) | +1.5 |
| x-combo3 vs pMCTS-cur | 60.7 (60.1, 61.3) | 44.3 (43.8, 44.8) | +14.6 | +4.4 (2.4, 6.5) | −1.1 |
| x-combo3 vs Real | 62.6 (62.0, 63.2) | 59.5 (59.0, 60.0) | +3.2 | +1.3 (−0.3, 2.9) | +1.5 |
| pMCTS-cur vs pPol-cur (paper 1, search vs none) | 49.7 (49.3, 50.2) | 49.6 (49.4, 49.8) | +0.2 | −1.6 | −0.1 |
| pMCTS-cur vs Real | 50.5 (49.6, 51.3) | 63.5 (62.9, 64.0) | −11.4 | −3.9 (−6.0, −1.8) | +2.4 |
| aMCTS-pp vs Real | 59.8 (59.1, 60.5) | 59.1 (58.6, 59.6) | +0.8 | +0.4 | +1.6 |
| a1step-cur vs Real | 56.3 (55.5, 57.0) | 55.6 (55.1, 56.1) | +0.7 | +0.1 | +0.8 |
| aMCTS-bc vs Real | 55.0 (54.2, 55.7) | 55.3 (54.8, 55.8) | −0.1 | −1.3 | +0.5 |
| Real vs GD | 55.2 (54.4, 56.0) | 50.2 (49.7, 50.7) | +4.7 | +1.9 (0.2, 3.6) | +0.4 |
| DraftRec vs Real | 54.9 (54.2, 55.5) | 51.8 (51.3, 52.2) | +2.8 | +1.2 | +0.4 |
| DraftRec vs a1step-cur | 50.4 (49.7, 51.0) | 47.4 (47.0, 47.9) | +2.5 | +0.2 | −0.3 |

- The personal part is c2 × the judge's personal-sum difference. The judge-free record is each assigned player's mean residual (y − WP) on that hero in his later games, averaged over the covered players; hero-to-player assignment is by earlier play counts, which is outcome-free.
- CIs come from a lobby bootstrap, widened for prior-seed effects with the drift paper's crossed random effects, taking the larger SE. The seed SE is 0.0 to 1.0pp; the paired-seed designs are at the low end.

### Bradley–Terry strengths and safety (all systems, all pairs)

| system | BT under J (logit, GD = 0) | BT under CONS | degenerate % | off-role players per team | record coverage % |
|---|---|---|---|---|---|
| x-combo3 | 0.75 (0.73, 0.77) | 0.40 (0.38, 0.41) | 0.0 | 0.62 | 77 |
| x-combo1 (selected) | 0.74 (0.72, 0.76) | 0.37 (0.35, 0.38) | 0.0 | 0.55 | 78 |
| x-pp-ban3 | 0.72 (0.69, 0.75) | 0.36 (0.34, 0.38) | 0.0 | 0.59 | 77 |
| aMCTS-pp | 0.69 (0.67, 0.70) | 0.39 (0.38, 0.41) | 0.0 | 0.62 | 76 |
| aPol-pp | 0.59 (0.57, 0.61) | 0.32 (0.30, 0.34) | 0.0 | 0.60 | 75 |
| DraftRec-pv | 0.57 (0.56, 0.60) | 0.37 (0.36, 0.39) | 0.4 | 0.79 | 72 |
| DraftRec | 0.51 (0.49, 0.53) | 0.16 (0.15, 0.17) | 0.5 | 0.50 | 86 |
| a1step-cur | 0.49 (0.47, 0.51) | 0.27 (0.26, 0.28) | 0.0 | 0.77 | 72 |
| aMCTS-sp | 0.49 (0.47, 0.51) | 0.59 (0.57, 0.60) | 0.0 | 1.16 | 57 |
| aMCTS-vn | 0.47 (0.45, 0.49) | 0.23 (0.22, 0.24) | 0.0 | 0.76 | 72 |
| aMCTS-bc | 0.43 (0.42, 0.45) | 0.23 (0.22, 0.24) | 0.0 | 0.82 | 70 |
| aMCTS-cur | 0.31 (0.29, 0.33) | 0.61 (0.60, 0.62) | 0.0 | 1.47 | 48 |
| pPol-cur | 0.29 (0.27, 0.31) | 0.62 (0.60, 0.63) | 0.0 | 1.44 | 47 |
| pMCTS-cur | 0.27 (0.25, 0.30) | 0.61 (0.59, 0.62) | 0.0 | 1.52 | 46 |
| Real | 0.25 (0.23, 0.28) | 0.04 (0.02, 0.05) | 8.7 | 0.81 | 77 |
| p1step-cur | 0.26 (0.24, 0.27) | 0.34 (0.33, 0.34) | 0.0 | 1.15 | 57 |
| pMCTS-bc | 0.15 (0.14, 0.17) | 0.26 (0.25, 0.27) | 0.0 | 1.13 | 55 |
| GD | 0.00 (0.00, 0.00) | 0.00 (0.00, 0.00) | 2.7 | 0.97 | 61 |

All 59 systems, with standings: `P3_TOURNAMENT_TABLES.md`.

- Seat (first pick) term: about +0.02 logit; the seat does not matter here.
- Every search drafter fields 0% degenerate teams (structural mask). `Real` is degenerate by paper 1's rule in 8.7% of its tournament drafts (8.5% of the lobbies' own drafts; replaying against other drafters forces some GD fallbacks), GD 2.7%, DraftRec 0.5%.
- Off-role players per team: 1.5 for paper 1's blind searches, 0.8 for real teams, 0.55 to 0.62 for the distilled-prior drafters.


### Why the judged gain is below the drafter's own estimate (diagnosis)

The one-step drafter's own value said +7.2pp vs GD (the swag). Judged, the aware-vs-blind gains are 5 to 8pp, with 14pp of personal sum against paper 1's MCTS. Max expected more, so before reporting we checked four explanations.

1. **The judge's attenuation, which is the main reason.**
   - On real teams, the judge's personal sum moves 0.42 per unit of the drafter's (correlation 0.54). The judge sees 2.5 months of later games; the drafter sees the player's whole history.
   - A calibrated judge with less information credits only about that fraction of a true personal difference (paper 1 found the same attenuation for its population judges, 0.59).
   - The aware drafters' judged personal gains over their blind counterparts are 0.54 to 0.66 of their own estimates. That is above 0.42, so there is no sign of overfitting to the personal term.
   - Corrected for attenuation (a model-based sensitivity, labeled "J corrected" in the tables):
     - aMCTS-bc vs pMCTS-bc: 65.8%;
     - aMCTS-pp vs pMCTS-cur: 75.3%.
2. **A less attenuated judge.**
   - The split-history judge uses even-day evidence, years deep, outside the tournament window, and the other player half's fit. The drafters for this check use odd-day evidence only.
   - Its attenuation slope is 0.71.
   - Judged this way, aMCTS-bc@split vs pMCTS-bc is 59.5% (58.8, 60.2) and a1step@split vs p1step-cur 57.7% (56.9, 58.5), against 57.8% and 56.5% under the window judge. The drafter here has only half of each player's history.
3. **Judge-free.**
   - Each player's realized later record on the hero he is assigned is noisy but unshrunk. Against paper 1's blind MCTS it favors the aware drafters whose prior or candidates carry the players by 2.4 to 4.7pp per player, and each CI excludes zero; the value-net arm is +1.6 (−0.7, 3.9).
   - For the one-step aware drafter against paper 1's MCTS it is about 70% of the drafter's own estimate (+3.0 against +4.2pp per player).
4. **Pools are not the limit.** A team's pool union holds a median of 87 of 90 heroes (10th percentile 76), and 99.9% of real picks are in it.

What does cost the aware drafters is population win probability.
- Against paper 1's blind MCTS, CONS says 41 to 44% for every strongly personal drafter.
- The selected design trades 6pp of population WP for 15pp of personal sum.
- The aware drafters that keep CONS near 50% (`aMCTS-sp`, per-position mixing) give up part of the personal edge.

### Putting the players into the search (paper section)

**Options (a) to (d).** "vs" is the blind twin with the same prior family.

| option | system | J vs paper-1 MCTS | J vs its blind twin | CONS vs paper-1 MCTS | GPU ms per draft (5090, 8 decisions, 400 sims) | training for the player part |
|---|---|---|---|---|---|---|
| (a) leaf only, blind-trained prior | aMCTS-cur | 51.4 | 51.4 | 49.9 | 35.8 (blind 34.2) | none |
| (a') leaf only, behavioral prior | aMCTS-bc | 53.8 | 57.6 | 41.0 | 31.1 (blind 30.1) | none |
| (b) player-feature prior (distilled) | aMCTS-pp | **60.4** | | 44.2 | 40.6 | 0.4 GPU-h targets + 1 CPU-min fit |
| (c) value net with player features | aMCTS-vn | 54.5 | 58.8 (vs pMCTS-bc) | 41.0 | 31.2 | CPU minutes |
| (d) self-play with real rosters | aMCTS-sp | 57.4 | | 49.3 | 38.2 | 1.1 GPU-h (4 iterations × 4,096 rosters, 800 sims) |
| (b) + ten-player exploration (selected design) | x-combo1 | **60.8** | | 43.8 | 41.3 | as (b), plus the opponent model fit (CPU) |
| the same at 1,600 sims | x-combo3 | 60.7 | | 44.3 | 148 | as (b) |

- Computing the ten players' states costs about 0.1 ms per player-day (one walk for all players of a lobby).
- The F policy itself took 8.3 GPU-hours per seed (3080).

**Design study.** Tune lobbies first (300, never the tournament lobbies; selection score = mean J against pMCTS-cur and aMCTS-pp). The design is the distilled player-feature prior + denial bans (λ 3) + opponent comfort model inside the tree (×2) + assignment-aware root bias (×1).
- Before the kernel fix the rule picked it at 1,600 sims (`x-combo3`). On the rerun tune lobbies the rule picks the same design at 400 sims (`x-combo1`, tune J 56.07 against 55.84 for `x-combo3`; the gap is inside the noise). `x-combo1` is reported as the selected design and `x-combo3` next to it; on the tournament lobbies they tie.
- A split-history version (`x-combo3-split`, drafter state from odd days) also played the tune lobbies (56.39); it is a robustness arm for the split judge, not a candidate.

**Pre-stated design comparison:** the selected design against `aMCTS-pp` on the tournament lobbies is **51.6% under J (51.0, 52.1)** for `x-combo1`, with CONS 49.1 (48.7, 49.5). `x-combo3`: 51.6% (51.1, 52.1), CONS 49.4 (49.0, 49.8).

Each component added alone to `aMCTS-pp`, against `aMCTS-pp`:

| design | J vs aMCTS-pp (CI) | CONS vs aMCTS-pp (CI) | tune-set J | GPU ms per draft |
|---|---|---|---|---|
| x-combo1 (selected) | 51.6 (51.0, 52.1) | 49.1 (48.7, 49.5) | 56.1 | 41 |
| x-combo3 (same, 1,600 sims) | 51.6 (51.1, 52.1) | 49.4 (49.0, 49.8) | 55.8 | 148 |
| denial bans (λ 3) | 51.0 (50.4, 51.5) | 49.1 (48.7, 49.5) | 55.5 | 41 |
| opponent comfort model (×2) | 50.4 (50.0, 50.8) | 50.2 (49.8, 50.5) | 54.9 | 41 |
| assignment-aware root bias | 49.9 (49.5, 50.3) | 49.4 (49.1, 49.8) | 54.5 | 41 |
| value net v2 at the leaf | 50.4 (50.0, 50.8) | 49.7 (49.4, 50.1) | 54.9 | about 41 |
| 1,600 sims | 50.0 (49.5, 50.5) | 50.1 (49.8, 50.5) | 55.0 | about 145 (not timed) |
| c_puct 4 | 49.6 (49.2, 50.1) | 49.8 (49.5, 50.2) | 54.4 | |
| c_puct 1 | 48.5 (48.0, 49.0) | 49.5 (49.0, 49.8) | 52.6 | |
| per-position mixing (F weight 0.5 early) | 49.0 (48.4, 49.5) | 52.1 (51.7, 52.5) | 54.7 | |
| personal bias ×0.5 | 49.1 (48.7, 49.6) | 52.0 (51.6, 52.3) | 54.1 | |
| personal bias ×2 | 49.3 (48.9, 49.7) | 46.6 (46.3, 47.0) | 53.4 | |
| α 0.3 (more F) | 49.0 (48.5, 49.5) | 53.1 (52.7, 53.5) | 54.4 | |
| α 0 (no F prior) | 48.6 (48.1, 49.1) | 46.0 (45.6, 46.5) | 52.5 | |
| optimism 0.5 sd | 49.4 (49.0, 49.8) | 50.1 (49.7, 50.4) | 54.4 | |
| optimism 1 sd | 49.1 (48.7, 49.6) | 49.8 (49.4, 50.1) | 53.4 | |
| rich prior, search-distilled | 47.3 (46.8, 47.8) | 47.8 (47.4, 48.2) | 52.6 | |
| rich prior, outcome-weighted BC | 48.9 (48.4, 49.5) | 46.8 (46.4, 47.3) | 52.9 | |
| rich prior, one-step-distilled | 48.7 (48.1, 49.2) | 48.0 (47.6, 48.5) | 53.2 | |
| rich prior alone, no search (best: one-step) | 47.2 (46.6, 47.8) | 46.1 (45.7, 46.6) | 51.4 | |
| (reference) `x-pp`, the distilled prior in `cuda_explore` with every mode off | 50.4 (50.0, 50.9) | 50.1 (49.8, 50.5) | 55.1 | 40 |

- `x-pp` is `aMCTS-pp`'s design run through the design-study code path; it differs only in its kernel seeds (seeds are per pair, lobby and step). Its distance from 50% is the run-to-run noise of the search itself, about 0.4pp, which the lobby-bootstrap CIs do not include. Rows within about 0.5pp of `x-pp` are ties.


**Reading.**
- The distilled player-feature prior is close to a local optimum.
- Denial bans are the one single component that helps on both lobby sets: +1.0pp on the tournament lobbies, with the CI excluding 50, and +0.4 over `x-pp` on the tune lobbies.
- The ten-player combination adds +1.6pp, at a cost of about 0.9pp of population WP.
- Knobs that shift weight between the population prior and the personal prior trade along a frontier:
  - per-position mixing, smaller personal bias and α 0.3 gain 2 to 3pp of CONS and lose about 1pp of J;
  - α 0 and bias ×2 lose on both (CONS 3.4 to 4pp, J 0.7 to 1.4pp).
- The rich learned priors (all ten players' skill vectors, assignment gains, comfort, roles and step) predict their targets well. The search-distilled one matches 38% of target argmaxes against 20% for the F prior. Yet each drafts worse than the simple distilled prior, alone or inside the search, under both judges.
- So capacity is not the bottleneck. What matters is a prior that keeps paper 1's population knowledge (α about 0.16 on log p_F) and adds a personal term.
- Training signals for the rich prior, best to worst: one-step distillation ≈ outcome-weighted BC > search distillation.

**Value side.** On 112K pre-tournament games (paper 2's causal WP as offset, each team assigned by the drafter's own rule; last 15% of days held out), a richer value net predicts outcomes better than the linear personal sum:

| value model | gain over linear, x1000 log loss (SE) |
|---|---|
| the kernel's value net (assigned roster, 12 features) | +5.9 (0.7) |
| per-player set net | +7.7 (0.9) |
| plus the full 90-hero skill vectors | +10.7 (1.2) |

- The first value net, fit on 42,640 games the paper-1 WP never trained on, gained +0.8 (0.6) and moved the tournament by +1.1pp (`aMCTS-vn` vs `aMCTS-bc`, 51.1%).
- Refit on the 112K games (`x-pp-vn2`), it moves J by −0.2pp on the tune lobbies and 0.0 on the tournament lobbies (against `x-pp`).
- So a better leaf predicts better but does not draft better. In this search the prior and the exploration decide what gets evaluated, and the linear term is close to sufficient for choosing among those candidates.

### DraftRec baseline

- Rebuilt from the paper and its released code (player transformer over the last 50 games with scoreboard stats; draft transformer; pick and win heads, λ 0.1). Trained on 862,604 drafts through 2026-05-22 (1.2 GPU-hours).
- OOT accuracy: pick top-1 36.0%, top-5 68.5%; post-draft win AUC 0.654.
- Tournament results:
  - BT under J: the pick head 0.51 and the paper's p+v rule 0.57, against 0.69 for `aMCTS-pp`, 0.74 for `x-combo1`, 0.49 for `a1step-cur` and 0.27 for paper-1 MCTS.
  - Head to head: DraftRec vs real drafts 54.9% (J), vs a1step 50.4%.
  - Under CONS the pick head is weak (BT 0.16): it imitates comfortable picks.

### Drift

Each row is the current twin against the stale version, both seats, in pp above 50 (the stale drafter's loss), 95% CI with seed effects. "Edge lost" is the BT gap as a share of the current drafter's BT edge over GD. Stale vintages have 2 F seeds (fine-tune: 1), so the J intervals at 1 year are wide; the crossed seed effect dominates them.

**Population search (`pMCTS`), the upkeep arms.**

| arm | 4 months: CONS | 4 months: J | 1 year: CONS | 1 year: J | upkeep cost |
|---|---|---|---|---|---|
| (a) stale weights and statistics | 2.0 (0.9, 3.1) | 1.2 (0.1, 2.2) | 2.2 (1.6, 2.8) | 3.4 (−0.3, 7.0) | none |
| (b) stale weights, refreshed statistics (`-rs`) | 2.0 (0.8, 3.1) | 1.2 (−0.1, 2.4) | 2.2 (1.5, 2.9) | 3.6 (−0.1, 7.4) | CPU minutes |
| (c) fine-tune (`-ft`, seed 0) | 1.4 (1.0, 1.9) | 1.1 (−0.4, 2.7) | 1.7 (1.0, 2.4) | 1.3 (0.4, 2.2) | 1.6 and 2.1 GPU-h, plus under 1 GPU-minute for the WP |
| (d) full retrain (`pMCTS-cur`) | 0 | 0 | 0 | 0 | 8.3 GPU-h per seed (3080), plus 0.3 h of statistics and WP |

- Edge lost under CONS: 11% at 4 months and 12% at a year for (a); 6% and 11% for (c).
- Refreshing the statistics tables in the kernel changes nothing (`-rs` vs stale: 50.0% and 50.1%). The stale WP and policy, not the tables, carry the loss.
- The fine-tune recovers about a third of the population loss at 4 months and a quarter at a year (CONS), and most of the J loss at a year.

**Player-aware drafters.**

| drafter | what is stale | 4 months: CONS | 4 months: J | 1 year: CONS | 1 year: J | edge lost (J), 4m / 1y |
|---|---|---|---|---|---|---|
| `aMCTS` (leaf only) | everything, player state frozen at the cutoff | 2.1 (0.9, 3.3) | 1.3 (0.0, 2.6) | 2.1 (1.6, 2.7) | 3.7 (−0.2, 7.6) | 9% / 41% |
| `aMCTS-cur@v` | player state only | 0.0 (−0.2, 0.2) | 0.1 (−0.2, 0.5) | 0.1 (−0.2, 0.3) | 0.5 (0.2, 0.9) | 6% / 10% |
| `aMCTS-pp@v` | player state only | −0.4 (−0.8, −0.0) | 1.7 (1.2, 2.2) | −0.5 (−0.8, −0.1) | 3.8 (3.3, 4.2) | 11% / 23% |
| `pPol` (policy, no search) | everything | 2.0 (1.1, 3.0) | 1.5 (0.5, 2.5) | 2.3 (1.3, 3.3) | 3.8 (0.1, 7.6) | 11% / 45% |
| `p1step` | everything | | | 1.3 (0.8, 1.7) | 3.0 (2.2, 3.8) | – / 46% |
| `a1step` | everything | | | 1.2 (0.8, 1.6) | 4.9 (4.2, 5.6) | – / 37% |

- A frozen player state costs a drafter in proportion to how much its choices depend on the players. The leaf-only drafter barely uses them and barely notices; the distilled-prior drafter loses 1.7pp at 4 months and 3.8pp at a year under J, about a quarter of its edge, while its population side is unchanged.
- Keeping the player state current is the cheap part of upkeep: one walk over the players' games, about 0.1 ms per player-day on CPU.

**Drafters vs the prediction model at the same staleness** (prediction: share of the log-loss lift over a coin flip lost on tournament-window games).

| staleness | WP only | WP + player state | paper-1 MCTS, edge lost (CONS) | `aMCTS-pp`, player state only, edge lost (J) |
|---|---|---|---|---|
| 4 months | 9% | 56% | 11% | 11% |
| 1 year | 28% | 70% | 12% | 23% |

- The drafters lose less than the models they are built on. A search on a stale WP still ranks most of the same drafts on top, and a stale player state still knows most players' best heroes.

### Real-game backtests

**(a) Natural agreement** (20,000 OOT games, 200,000 picks). The drafter is the behavioral-prior search, aware and blind (`aMCTS-bc`, `pMCTS-bc`), at every real pick state.

| | value |
|---|---|
| real pick equals the aware recommendation | 9.3% |
| real pick equals the blind recommendation | 6.4% |
| aware and blind recommend the same hero | 40.7% |

Picking team's win, linear probability model, pp, SE clustered by player and game:

| model | matched aware | other terms |
|---|---|---|
| raw | +3.4 (0.4) | |
| + draft-only WP + causal MMR | **+0.9 (0.4)** | |
| matched only the aware pick (aware ≠ blind) + controls | +1.8 (0.5) | matched blind −0.4 (0.4) |
| + the skill model's predicted gap | −0.5 (0.4) | gap coefficient 1.02 (0.03) per unit |

- Agreeing with the player-aware drafter predicts winning, and the skill model's own gap accounts for all of it. The gap is calibrated on real outcomes (slope 1.02).
- This is evidence for the value the drafter uses, not for choices beyond it.
- With paper 1's F-prior drafter, matches are too rare to say anything (2.5% of picks).
- The slot table's MMR is the post-game rating and leaks the result, so the controls use the causal MMR (P3_MMR_AT_GAME).

**(b) Forced switches** (OOT, 1.05M pick slots of players with a 30+ game hero). Within player: realized r = y − WP in games where the hero was taken before his pick, against his other games, compared with the model's predicted r for the hero he played.

| situation | share of slots | realized pp (SE) | predicted pp | realized − predicted (SE) |
|---|---|---|---|---|
| best hero banned | 9.0% | −2.13 (0.18) | −1.26 | −0.86 (0.17) |
| best hero taken by a teammate | 2.5% | −1.44 (0.31) | −0.95 | −0.48 (0.31) |
| best hero taken by an opponent | 3.8% | −0.79 (0.26) | −1.00 | +0.21 (0.26) |
| main banned | 9.8% | −1.85 (0.17) | −1.10 | −0.75 (0.17) |
| main taken by a teammate | 2.8% | −1.74 (0.30) | −0.83 | −0.91 (0.30) |
| main taken by an opponent | 4.3% | −1.34 (0.23) | −0.84 | −0.50 (0.23) |
| best hero taken, any way | 15.4% | −1.90 (0.14) | −1.33 | −0.58 (0.14) |

- The main-ban effect replicates P3_TAILS (−1.76 there, −1.85 here on OOT games).
- The model predicts about 60% of the realized cost of a forced switch. Bans and teammates that take a player's hero cost more than the skill gap between his heroes.
- This is the case for denial bans in the drafter: the ban-denial prior helps on both lobby sets (+0.4pp on the tune lobbies, +1.0pp on the tournament lobbies).
- Caveats:
  - This is observational. Bans may target players in form.
  - When an opponent takes the hero, the cost matches the prediction, which hints that ban-driven selection explains part of the gap.

### Compute

**Training** (GPU-hours on one GPU each; "3090" means its WSL GPU; CPU work at nice 19):

| item | where | cost |
|---|---|---|
| paper-1 F policies (5 seeds, existing) | 3080 / 3090 | 8.3 GPU-h per seed alone (21 h when two shared a 3090) |
| stale vintages: statistics, features, 3 WP seeds | 3090 | 0.31 h (s4m) and 0.24 h (s1y) wall, of which 0.08 GPU-h |
| stale GD (s1y era, 2 variants) | 3090 | 1.34 GPU-h (s4m reuses the drift paper's cutoff GD) |
| stale F policies (2 seeds per vintage) | 3090, 3080 | 7.9 to 8.6 GPU-h per seed (33 GPU-h for the four) |
| fine-tune upkeep: WP | 3090 | under 1 GPU-minute per vintage |
| fine-tune upkeep: F continuation (60K episodes) | 3080 / 3090 | 1.6 GPU-h (s4m, 3080) and 2.1 GPU-h (s1y, 3090) |
| judge skill fit (other player half) | 5090 | 147 s |
| player states, combiners, judge fits | 3090 CPU | minutes each (one walk: 0.1 ms per player-day) |
| (b) distilled player-feature prior | 5090 + CPU | 0.4 GPU-h targets (shared GPU), 1 CPU-minute fit |
| (c) value net | CPU | minutes (12-feature MLP) |
| (d) self-play prior | 5090 | 1.1 GPU-h (4 iterations) |
| rich priors (3 signals) | 3090 CPU | 3 to 15 minutes each |
| opponent comfort model | CPU | 1 minute |
| DraftRec | 5090 | 1.2 GPU-h (plus 0.4 in aborted attempts) |
| kernels (`cuda_pvnet`, `cuda_explore`) | all hosts | CPU builds, about 10 minutes each per host |

**Per draft** (one side, 8 decisions, 5090):
- paper-1 MCTS 34 ms;
- aware leaf 36 ms (+5%);
- distilled prior 41 ms (+19%, mostly the CPU bias);
- selected design 41 ms (`x-combo1`, 400 sims); 148 ms for `x-combo3` (1,600 sims);
- one-step about 40 ms of CPU across 12 processes;
- policy-only drafters about 2 ms.

**Tournament compute.** The 300-lobby round robin, spokes and design ablation are 472 ordered pairs, 4.7 hours of drafting time (5090 searches plus the one-step CPU pool); the tune set is 164 pairs, 2.1 hours. Scoring is CPU minutes.

## Design (fixed before the tournament)

**Windows.**

| what | dates |
|---|---|
| drafters' population data (current vintage) | pinned snapshot, through 2026-05-22 |
| tournament lobbies | real OOT games 2026-05-23 to 06-30 |
| judge window (personal evidence, truth statistics, judge combiner) | 2026-07-01 to 09-11 |
| backtest games | OOT 2026-05-23 to 09-11 |

- The sealed personalization build 2.55.17.98025 (from 2026-09-12) is never read.
- The personalization store keeps CAP_TS 2026-09-28.
- The 5,000 sealed replays are not touched.

**Lobbies.** Real games with a full 16-step draft whose ten picks join to known players:
- all ten players have 50+ earlier games;
- each has 10+ games in the judge window;
- 300 lobbies drawn with a fixed seed. Every ordered pair plays the same lobbies, with seat A as the real first-pick team.

Lobby rules, the same for every system:
- a team's picks come from the union of its players' pools (heroes each played before the lobby day);
- the team picks a hero, and personal values use each team's best assignment of heroes to players. The real pick order (who picked what) is never used.

**Player state** (the drafter's s(p, h) for all 90 heroes). From the main skill model: `met2_hs2` structure, full scoreboard export, `a_shrink`, merged full_1017 history.
- s = experience offset + posterior mean + temporal layer (drift and new-hero learning), with state through the day before.
- The state reproduces the `full_ms_layer` arm's slot predictions to 5e-6.

**Systems.**

| name | what |
|---|---|
| `pMCTS-cur` | paper 1's best MCTS (config F, 5 seeds, leak-free enriched WP, deploy statistics, chance-node search, mid_high_low composition fallback), searching at decision time (400 sims) with the population leaf |
| `aMCTS-cur` | the same search with the personal leaf V = sigmoid(b0 + b1 logit WP + b2 (S_ours − S_theirs) + b3 (O_ours − O_theirs)). The term sits inside the CUDA kernel (`cuda_personal`, assign mode): a per-lobby leaf adjustment, with no retraining |
| `pMCTS-bc`, `aMCTS-bc` | the same search with the outcome-free behavioral prior (train-window BC policy) in place of the F policy |
| `pPol-cur` | the F policy argmax, no search (paper 1's tournament agent) |
| `a1step-cur`, `p1step-cur` | the one-step drafter (top 15 GD plus top 15 personal candidates, 4 GD rollouts each), aware and blind, with the same WP |
| `GD` | behavioral cloning, argmax |
| `Real` | the real lobby's actions, replayed when still legal |

**Vintages for drift.**
- `s4m`: population data through build 2.55.14.95918 (2026-02-10), player state frozen at that date.
- `s1y`: through 2.55.10.94189 (2025-06-02).

Upkeep arms of the search drafter:

| arm | what |
|---|---|
| stale | stale weights and statistics |
| `-rs` | stale weights with the current statistics tables in the kernel |
| `-ft` | fine-tune from the stale weights on recent data |
| `pMCTS-cur` | the full retrain |

Player-state upkeep: `aMCTS-cur@s4m` and `aMCTS-cur@s1y` use the current population side with player state frozen at the vintage cutoff.

**Judges** (none shares data with a drafter):
- **Population:** paper 1's consensus (gN, gN-naive, RN, QM2026, composition-corrected). These are trained on post-snapshot or Quick Match games.
- **Personal term:** a skill model fit on the OTHER phase-1 half of players (`full_ms_flip`), with evidence from the judge window only. It is combined with the consensus as a fixed offset; the coefficients are fit on judge-window games.

**Statistics** (as paper 1 and the drift paper):
- pair means with lobby-cluster bootstrap CIs;
- standings over opponents;
- Bradley–Terry strengths on all pairs with bootstrap CIs;
- crossed random effects for the two sides' seeds;
- degenerate-team, healer and off-role rates.


## The drafters, exactly

**Shared by every drafter.**
- Draft order: Storm League (bans 0,1,0,1; picks 0,1,1,0,0; bans 1,0; picks 1,1,0,0,1). The real first-pick team is kernel team 0.
- Hero set: the 90-hero v1 set.
- Picks are from the union of the team's pools, falling back to any free hero.
- The team picks. Personal terms use each team's best hero-to-player assignment over all 120 permutations (assign mode).
- Every search drafter picks under paper 1's structural constraint. A hero is kept only if the team can still reach ≥ 1 healer, ≥ 1 frontline, ≥ 1 ranged and ≤ 2 of each stacking role from its legal heroes. With search, the constraint applies at the root: the drafter takes the most-visited allowed hero. GD and Real are unconstrained.
- The constraint is required here. Without it, the unconstrained first tournament pair fielded 19% and 43% degenerate teams: the first-pick search took two healers, and the second team's pools then had none left.

**Population side (vintage cur), paper 1's site-tier rebuild.**
- **Snapshot:** 1,956,753 replays through 2026-05-22. The split is the paper's test set (38,869), a 2% validation hash, and the rest for training.
- **Leaf WP:** `enriched` in `paper1_revision/train_wp.py`.
  - Inputs: 283 (197 base + 86 enriched features).
  - MLP 256 → 128, dropout 0.3, AdamW lr 5e-4, wd 5e-3, cosine, batch 4,096, ≤ 200 epochs, patience 25.
  - Training rows use out-of-fold statistics (5 hash folds).
  - Seed 42 of 3 was selected on validation. Symmetrized over team order.
- **Kernel tables:** the deploy statistics (all training games), with the own-corpus role-composition table (cells of ≥ 50 games). The composition fallback order is mid, high, low.
- **GD:** paper 1's five `generic_draft_{0..4}` models.
  - Architecture: 289 → 256 → 128 → 90, ReLU, dropout 0.2 / 0.1.
  - Data: the p1site rebuild, next-action cloning on snapshot drafts.
  - In search, GD is the chance-node opponent and the rollout policy. Draft i uses GD model i mod 5.
- **F policy (prior of `*MCTS-cur`, the `pPol` agents):** AlphaZeroDraftNet, base size, 4.4M parameters.
  - Residual backbone 289 → 768, 3 blocks, → 256; linear policy head; perspective-aware value head.
  - Bootstrapped from GD.
  - Self-play: 300K episodes, 400 sims per move, chance-node search (progressive widening k 1.0, α 0.5), c_puct 2.0.
  - Opponent: GD model 0. Buffer 150K, train batch 512, Adam lr 1e-3 with cosine T_max 150K.
  - Value pretraining on the training split's replays.
  - Seeds 0 to 4; draft i uses seed i mod 5. About 8.3 GPU-hours per seed on a 3080.
- **BC prior (`*MCTS-bc`):** AlphaZeroDraftNet base, distilled by behavioral cloning from GD variant 0. Train-window states only (games before 2026-02-10), outcome-free.

**Player side.**
- **State s(p, h):** the main skill model, as of the day before the lobby, for all 90 heroes.
  - Model: `met2_hs2` structure (overall column + 2 factors, shared metric loadings, per-hero s²). Inputs: the full scoreboard export with `a_shrink`, and the merged full_1017 history.
  - s = experience offset + posterior mean + temporal layer (drift half-life 730 d, k_d 10; new-hero learning k_e 2).
- **Off-role:** fine role under 10% of 50+ earlier games.
- **Combiner:** V = sigmoid(b1 logit WP + b2 (S_ours − S_theirs) + b3 (O_ours − O_theirs)).
  - Fit on 5,397 games the WP never trained on (2026-02-11 to 05-22: paper-1 test and validation splits plus post-snapshot uploads).
  - Fitted values: b1 1.20, b2 4.54 (SE 0.31), b3 −0.015. b0 = 0.

**Search.**
- Decision-time decide-only search in `cuda_personal` (aware leaf) or `cuda_prior` / `cuda_pvnet` (variants below).
- 400 simulations, chance-node opponent, c_puct 2.0, no root noise, root argmax of visits under the constraint.
- Kernel seeds are fixed per pair, lobby and step.
- Aware and blind twins share the prior seed, GD model and kernel seed of every decision.

**Ways to put players into the search (the four options).**

| option | system | prior | leaf | training for the player part |
|---|---|---|---|---|
| (a) leaf only, blind-trained prior | `aMCTS-cur` | F (blind self-play) | WP + personal terms | none (combiner fit, CPU seconds) |
| (a') leaf only, behavioral prior | `aMCTS-bc` | BC | WP + personal terms | none |
| (b) player-conditioned prior | `aMCTS-pp` | α log p_F + log mean_team exp g(f(p, h)) | WP + personal terms | targets: aware BC search at 1,600 sims on 30,000 real pick states of 3,000 non-tournament lobbies (0.4 GPU-h); fit α and g by cross-entropy (CPU, 1 min) |
| (c) value net with player features | `aMCTS-vn` | BC | b1 logit WP + h(F_ours) − h(F_theirs), h an MLP on the assigned roster (12 features) | 42,640 held-out games, CPU |
| (d) self-play with sampled real rosters | `aMCTS-sp` | as (b), refit each iteration | WP + personal terms | 4 iterations × 4,096 rosters (2026-03-29 to 05-22), 800 sims, root Dirichlet ε 0.25, temperature 1 |

- g is an MLP, 6 → 32 → 32 → 1. Its features are b2 s(p, h), off-role, log(1 + games on h), never played, the share of p's games on h, and log of p's fine-role share.
- The prior-only versions of (b) and (d) are `aPol-pp` and `aPol-sp`.

**One-step (`a1step`, `p1step`).**
- At each pick, candidates are the top 15 by GD plus the top 15 by the team's best personal term.
- Each candidate gets 4 rollouts completing the draft with the vintage's GD; bans in the rollouts are sampled from GD.
- Finished drafts are valued by the vintage WP (`p1step`) or V (`a1step`) on the same rollouts. Own bans: GD argmax.

**Imitation and the human reference.** `GD` is the 5-model GD average, argmax. `Real` replays the lobby's real actions; the first illegal action falls back to GD argmax (11.4% of its steps).

**Vintages for drift.** These are built by the same recipe on training games at or before the cutoff build, with that era's GD.

| vintage | cutoff | train games | WP test acc / slope | GD |
|---|---|---|---|---|
| s4m | 2.55.14.95918 (2026-02-10) | 1,731,780 | 0.5753 / 1.12 | drift `gd_cutoff` 0, 1 |
| s1y | 2.55.10.94189 (2025-06-02) | 1,262,321 | 0.5725 / 1.10 | era GD 2.55.10.94189 0, 1 |

- Each vintage has F seeds 0 and 1, and its player state is frozen at the cutoff.
- Upkeep arms:
  - `-rs`: the stale F policies, WP and GD with the current statistics tables.
  - `-ft`: the WP fine-tuned on post-cutoff training games (early stop on post-cutoff validation), plus the F policy continued 60K episodes from seed 0 against it, with current statistics and GD.
  - `aMCTS-cur@v`: the current population side with the vintage's frozen player state.

**Time rule for the learned player-side pieces.**
- The distilled prior's targets, the opponent comfort model and the search-distilled rich prior come from tournament-window lobbies (2026-05-23 to 06-30) that are not tournament, probe or tune lobbies.
  - They use no outcome, only real picks or search visits, and the players' states as of each lobby.
  - Their parameters are a small MLP over hero-agnostic features, so no player-specific information carries over.
- Everything else on the population side ends on 2026-05-22. The self-play rosters, the outcome-weighted BC and one-step targets, and DraftRec's training games come from before 2026-05-23.

**Exploration design study (systems `x-*`, `p3t_explore.json`).**
- All are `aMCTS-pp`-like decide-only searches in `cuda_explore`, a copy of `cuda_pvnet` with two gated modes. Parity:
  - byte-identical to `cuda_pvnet` when the new modes are off;
  - population path identical to the reference.
- The knobs:
  - **c_puct**: 1, 2 (default) or 4.
  - **α**: the weight on log p_F in the prior. Fixed at 0, 0.3 or 0.6; the fitted value (0.160); or per pick position (0.5 for the first three picks, 0.3 for picks 4 and 5, 0.156 after).
  - **Bias scale**: ×0.5 or ×2 on the distilled personal bias.
  - **Optimism**: the leaf uses s + κ sd for our players and s − κ sd for theirs (κ 0.5 or 1).
  - **Assignment bias**: at the decision state, each hero's best-assignment gain for our team (b2 units) is added to our pick prior (×1 or ×3).
  - **Opponent model** (`cfg[6]=2`): opponent picks inside the tree and in rollouts follow softmax(0.79 log p_GD + scale × log mean_j exp(0.64 log1p n_jh − 1.26 [n_jh = 0])), fit on 30,000 real picks (scale 1, 2 or 3).
  - **Denial bans** (`cfg[56]=2`): our ban prior gets λ × b2 × the opponents' best-player skill on each hero (λ 1, 3 or 5).
  - **Value net v2**: the leaf's value net refit on 112K pre-tournament games.
  - **Rich prior**: at the decision state the prior becomes softmax(rich logits). It is applied as a static bias for our later in-tree picks.
  - **Sims**: 400 or 1,600.
- **Selected design `x-combo3`** (pre-stated rule): the distilled prior + denial bans λ 3 + opponent model scale 2 + assignment bias ×1, at 1,600 sims. `x-combo1` is the same at 400 sims.

## Search prior: a player-blind trained prior hides the personal leaf (probe, before the tournament)

Probe setup (`p3t_probe.py`): 500 tournament-window lobbies outside the tournament set, all 10 real pick states each (5,000 decisions). At each state two decide-only searches run with the same seeds, personal leaf on and off. The table is model-internal.

| search prior | sims | picks where aware and blind differ | aware search's own Q gain when they differ (pp) | root visit entropy, blind (nats) |
|---|---|---|---|---|
| paper-1 F policy | 400 | 4.4% | +5.2 | 0.44 |
| paper-1 F policy | 1,600 | 8.3% | +5.0 | 0.43 |
| behavioral (BC) | 400 | 59.0% | +6.4 | 2.69 |
| behavioral (BC) | 1,600 | 65.1% | +6.7 | 2.78 |

- Search cost per decision (5090, uncontended, 400 sims): 4.3 ms with the F prior, 3.8 ms with the BC prior; about 4 times that at 1,600 sims.
- The F policy was trained by self-play against a player-blind value. It is sharp: the blind search puts almost all visits on one or two heroes. A personal term added only at the leaf moves 4 to 8% of picks.
- Over a flat behavioral prior the same leaf moves about 60% of picks.
- So "train player-blind, add the personal term at the leaf" works only when the prior leaves room for search.
- The tournament therefore carries both priors. A third arm, a prior trained with player features (`aMCTS-pp`), follows when its training finishes.

## Files and rerun

In `training/personalization/`:

| file | role |
|---|---|
| `p3t_data.py` | lobbies, player states (cur, stale, judge, split), judge fit data, rosters, pre-tournament lobbies |
| `p3t_value.py` | combiners and prediction drift |
| `p3t_systems.py` | every drafting system and the structural mask |
| `p3t_tourney.py` | lobby selection, schedules, the runner |
| `p3t_judge.py` | judges, truth statistics, scoring |
| `p3t_analyze.py` | head-to-heads, Bradley–Terry, seed effects, standings |
| `p3t_diag.py` | attenuation diagnostics |
| `p3t_report.py` | `P3_TOURNAMENT_TABLES.md` |
| `p3t_backtest.py` | backtests (a) and (b) |
| `p3t_sanity.py` | the paper-1 reconciliation |
| `p3t_probe.py` | decision probe and timing |
| `p3t_pprior.py`, `p3t_selfplay.py`, `p3t_vnet.py`, `p3t_vstudy.py`, `p3t_rprior.py`, `p3t_oppmodel.py` | design options and the design study |
| `p3t_draftrec.py` | the DraftRec baseline |
| `p3t_verify.py`, `p3t_pvnet_verify.py`, `p3t_explore_verify.py`, `p3t_determinism.py` | kernel parity and determinism checks |
| `p3t_vintages.json`, `p3t_explore.json` | the drafter configurations |

- Kernels: `cuda_pvnet/` and `cuda_explore/` (new), plus `cuda_personal/` and `cuda_prior/`; all race-fixed on 2026-10-19 (`build_p3_kernels.sh`, `remote/p3t_build_all.sh`).
- Stale vintages: `paper1_revision/vintage.py`, `vintage_ft.py` and `vintage_launch.py` (`P1_VINTAGE=s4m|s1y|s4m_ft|s1y_ft`).
- Results: `results/p3t/*.json` (small JSON, no player keys) and `results/P3_TOURNAMENT_TABLES.md` (generated). Caches stay in `cache_full/p3t/` on the workers.
- **Rerun:** `nohup nice -n 19 taskset -c 48-51 training/personalization/remote/p3t_pipeline.sh > training/personalization/logs/p3t_pipeline.log 2>&1 &`, after the personalization rerun has refit `full_ms` on the new store. It refits the judge on the other player half and runs every stage above in order.
  - The pipeline first rebuilds every P3 kernel and stops unless the determinism and parity checks pass.
  - The exploration design is the one the rule picked on the rerun tune lobbies (`x-combo1`, with `x-combo3` reported next to it). The full-data rerun re-reports the tune table and keeps that choice.
