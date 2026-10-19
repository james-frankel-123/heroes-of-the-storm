# P3 tournament: player-aware vs player-blind drafters, and drafters under drift (started 2026-10-19)

Status: in progress. This first version records the design and the primary comparison before any tournament draft exists. Results are added below as they land; the full-data rerun after the player-history pull (about Oct 26 to 27) uses the same pipeline (`p3t_pipeline.sh`).

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

## Search prior: a player-blind trained prior hides the personal leaf (probe, before the tournament)

Probe setup (`p3t_probe.py`): 500 tournament-window lobbies outside the tournament set, all 10 real pick states each (5,000 decisions). At each state two decide-only searches run with the same seeds, personal leaf on and off. The table is model-internal.

| search prior | sims | picks where aware and blind differ | aware search's own Q gain when they differ (pp) | root visit entropy, blind (nats) | GPU seconds per 1,000 decisions (5090) |
|---|---|---|---|---|---|
| paper-1 F policy | 400 | 4.4% | +5.1 | 0.44 | 4.3 |
| paper-1 F policy | 1,600 | 8.6% | +5.0 | 0.43 | 15.4 |
| behavioral (BC) | 400 | 59.9% | +6.6 | 2.69 | 3.7 |
| behavioral (BC) | 1,600 | 64.8% | +6.7 | 2.79 | 14.0 |

- The F policy was trained by self-play against a player-blind value. It is sharp: the blind search puts almost all visits on one or two heroes. A personal term added only at the leaf moves 4 to 9% of picks.
- Over a flat behavioral prior the same leaf moves about 60% of picks.
- So "train player-blind, add the personal term at the leaf" works only when the prior leaves room for search.
- The tournament therefore carries both priors. A third arm, a prior trained with player features (`aMCTS-pp`), follows when its training finishes.
