# Player-axis support: how familiar are placed players with their heroes?

Script: `p3t_familiarity.py`. Numbers: `familiarity.json`. 300 headline tournament lobbies, 2000 lobby-bootstrap resamples (percentile 95% CIs in parentheses).

**Prior games** = the placed player's earlier Storm League games on the hero: the merged play count of the drafters' cur state (slot rows plus Heroes Profile Storm League rows, days up to the day before the lobby; `state_cur.npz` n, built by `p3t_data.py states`). It is the count behind the personal model's experience offset, its new-hero layer (fewer than 5 earlier games) and the drafters' pools (at least 1).

Only the arm's own team counts. Each arm is scored against the same opponent panel (GD, Real, DraftRec, both seats; 6 drafts × 5 players per lobby). Humans are the real final teams of the same lobbies (both teams).

Drafters pick heroes, not players. **Record assignment** (main table) puts the five players on the five heroes to maximize the sum of log(1 + prior games), the judge-free record's outcome-free rule. **Model assignment** uses the drafter's own leaf (b2 S + b3 O on the cur state). Humans also get their **actual** placement.

Coverage = share of placed players with at least one judge-window game on the hero, the condition for entering the judge-free paired record.

**Not computed:** `s-pcal-floor25`: no pair files on this machine (drafted on the 3090; not copied back). `s-floor25` (same graded floor 25, personal term not rescaled) is reported as the closest stand-in.

## Record assignment (main), panel opponents

| arm | placed | 0 | 1-4 | 5-19 | 20+ | median | below 5 | below 10 | coverage |
|---|---|---|---|---|---|---|---|---|---|
| humans, actual placement | 3,000 | 3.2 (2.6, 3.8) | 9.9 (8.8, 11.0) | 21.7 (20.1, 23.3) | 65.3 (63.5, 67.0) | 41 (38, 44) | 13.1 (11.9, 14.3) | 22.7 (21.1, 24.3) | 82% |
| humans, record assignment | 3,000 | 2.3 (1.7, 2.8) | 7.1 (6.2, 8.0) | 21.1 (19.6, 22.7) | 69.5 (67.9, 71.2) | 47 (44, 51) | 9.4 (8.4, 10.4) | 18.0 (16.6, 19.4) | 80% |
| s-floor25 | 9,000 | 3.6 (3.1, 4.1) | 6.1 (5.5, 6.8) | 15.6 (14.6, 16.6) | 74.6 (73.2, 76.0) | 62 (57, 68) | 9.8 (9.0, 10.6) | 15.6 (14.5, 16.6) | 75% |
| x-combo1-dr-chance | 9,000 | 3.7 (3.2, 4.2) | 5.5 (4.9, 6.2) | 15.6 (14.6, 16.7) | 75.2 (73.8, 76.5) | 63 (58, 68) | 9.2 (8.5, 10.1) | 15.1 (14.2, 16.2) | 75% |
| aMCTS-pp | 9,000 | 4.5 (4.0, 5.0) | 6.2 (5.5, 6.8) | 16.0 (15.1, 16.9) | 73.3 (72.0, 74.6) | 59 (54, 64) | 10.7 (9.9, 11.5) | 16.9 (15.8, 17.9) | 73% |
| aMCTS-cur | 9,000 | 17.9 (16.9, 18.9) | 14.3 (13.3, 15.4) | 21.9 (20.8, 23.1) | 45.9 (44.4, 47.4) | 16 (15, 17) | 32.2 (30.9, 33.4) | 41.5 (40.0, 42.8) | 46% |
| a1step-cur | 9,000 | 5.1 (4.5, 5.7) | 7.5 (6.8, 8.2) | 19.8 (18.7, 20.8) | 67.6 (66.1, 69.2) | 44 (42, 48) | 12.6 (11.7, 13.6) | 20.5 (19.3, 21.7) | 69% |
| pMCTS-cur | 9,000 | 18.5 (17.5, 19.5) | 14.8 (13.7, 15.9) | 21.8 (20.6, 23.0) | 45.0 (43.4, 46.5) | 15 (14, 16) | 33.2 (31.9, 34.5) | 42.6 (41.1, 44.0) | 45% |
| p1step-cur | 9,000 | 11.1 (10.4, 11.8) | 12.1 (11.3, 13.0) | 22.5 (21.5, 23.5) | 54.3 (52.8, 55.7) | 24 (22, 26) | 23.2 (22.2, 24.3) | 32.4 (31.1, 33.7) | 55% |

## Model assignment, panel opponents

| arm | placed | 0 | 1-4 | 5-19 | 20+ | median | below 5 | below 10 | coverage |
|---|---|---|---|---|---|---|---|---|---|
| humans, actual placement | 3,000 | 3.2 (2.6, 3.8) | 9.9 (8.8, 11.0) | 21.7 (20.1, 23.3) | 65.3 (63.5, 67.0) | 41 (38, 44) | 13.1 (11.9, 14.3) | 22.7 (21.1, 24.3) | 82% |
| humans, model assignment | 3,000 | 5.1 (4.4, 6.0) | 8.5 (7.6, 9.5) | 19.8 (18.3, 21.2) | 66.6 (64.8, 68.3) | 43 (40, 46) | 13.7 (12.5, 14.9) | 22.1 (20.6, 23.6) | 78% |
| s-floor25 | 9,000 | 5.3 (4.7, 5.8) | 6.5 (5.9, 7.3) | 15.7 (14.7, 16.8) | 72.5 (71.0, 74.0) | 57 (53, 62) | 11.8 (10.9, 12.7) | 17.8 (16.7, 18.9) | 74% |
| x-combo1-dr-chance | 9,000 | 5.4 (4.9, 6.0) | 5.9 (5.3, 6.6) | 15.2 (14.2, 16.2) | 73.5 (72.2, 74.9) | 59 (54, 64) | 11.3 (10.5, 12.2) | 17.3 (16.2, 18.4) | 74% |
| aMCTS-pp | 9,000 | 6.4 (5.8, 7.1) | 6.5 (5.8, 7.2) | 15.7 (14.7, 16.6) | 71.4 (70.1, 72.8) | 55 (50, 59) | 12.9 (12.0, 13.8) | 19.1 (18.0, 20.2) | 72% |
| aMCTS-cur | 9,000 | 24.4 (23.2, 25.5) | 12.5 (11.6, 13.5) | 20.6 (19.5, 21.7) | 42.5 (41.1, 44.1) | 13 (12, 14) | 36.9 (35.5, 38.2) | 45.8 (44.4, 47.3) | 45% |
| a1step-cur | 9,000 | 7.8 (7.1, 8.5) | 7.6 (6.9, 8.3) | 19.3 (18.2, 20.4) | 65.3 (63.8, 66.9) | 42 (39, 45) | 15.4 (14.3, 16.4) | 22.9 (21.6, 24.1) | 68% |
| pMCTS-cur | 9,000 | 25.2 (24.0, 26.4) | 12.8 (11.9, 13.8) | 20.4 (19.2, 21.4) | 41.6 (40.1, 43.1) | 12 (11, 13) | 38.1 (36.7, 39.5) | 46.9 (45.4, 48.5) | 44% |
| p1step-cur | 9,000 | 15.9 (15.0, 16.8) | 12.0 (11.1, 13.0) | 21.0 (19.9, 22.0) | 51.1 (49.5, 52.6) | 21 (19, 22) | 27.9 (26.7, 29.2) | 36.7 (35.3, 38.1) | 53% |

## Restricted to placements the paired record covers (record assignment, panel)

| arm | placed | 0 | 1-4 | 5-19 | 20+ | median | below 5 | below 10 | coverage |
|---|---|---|---|---|---|---|---|---|---|
| humans, actual placement | 2,475 | 1.5 (1.0, 2.0) | 7.1 (6.1, 8.1) | 19.1 (17.5, 20.7) | 72.3 (70.5, 74.0) | 53 (49, 59) | 8.6 (7.5, 9.6) | 16.5 (15.0, 18.0) | 82% |
| humans, record assignment | 2,403 | 0.5 (0.2, 0.8) | 4.6 (3.7, 5.5) | 18.0 (16.5, 19.5) | 76.9 (75.2, 78.6) | 63 (58, 69) | 5.1 (4.2, 6.0) | 12.1 (10.8, 13.4) | 80% |
| s-floor25 | 6,729 | 0.2 (0.1, 0.3) | 3.3 (2.7, 3.8) | 12.0 (11.0, 13.2) | 84.5 (83.2, 85.7) | 95 (86, 102) | 3.4 (2.9, 4.0) | 7.5 (6.6, 8.4) | 75% |
| x-combo1-dr-chance | 6,759 | 0.2 (0.1, 0.3) | 2.7 (2.2, 3.2) | 12.1 (11.0, 13.2) | 85.0 (83.8, 86.3) | 94 (84, 99) | 2.9 (2.3, 3.4) | 7.0 (6.2, 7.8) | 75% |
| aMCTS-pp | 6,613 | 0.3 (0.2, 0.4) | 2.8 (2.3, 3.3) | 12.5 (11.4, 13.6) | 84.5 (83.2, 85.7) | 94 (85, 101) | 3.1 (2.6, 3.6) | 7.5 (6.6, 8.3) | 73% |
| aMCTS-cur | 4,176 | 1.2 (0.8, 1.6) | 6.6 (5.5, 7.7) | 20.9 (19.1, 22.7) | 71.3 (69.3, 73.5) | 46 (42, 51) | 7.7 (6.6, 8.9) | 15.5 (13.9, 17.1) | 46% |
| a1step-cur | 6,187 | 0.5 (0.3, 0.7) | 3.9 (3.3, 4.5) | 15.5 (14.3, 16.7) | 80.1 (78.7, 81.5) | 71 (66, 78) | 4.4 (3.8, 5.1) | 10.2 (9.2, 11.2) | 69% |
| pMCTS-cur | 4,063 | 1.3 (0.9, 1.7) | 6.8 (5.6, 8.0) | 21.1 (19.1, 23.1) | 70.8 (68.6, 73.1) | 46 (42, 50) | 8.1 (6.9, 9.3) | 15.9 (14.3, 17.6) | 45% |
| p1step-cur | 4,924 | 1.0 (0.7, 1.3) | 6.4 (5.6, 7.2) | 19.9 (18.6, 21.3) | 72.7 (70.9, 74.3) | 50 (46, 54) | 7.4 (6.6, 8.3) | 15.0 (13.7, 16.3) | 55% |

## Every headline opponent the arm met (record assignment)

| arm | placed | 0 | 1-4 | 5-19 | 20+ | median | below 5 | below 10 | coverage |
|---|---|---|---|---|---|---|---|---|---|
| s-floor25 (6 opponents) | 18,000 | 3.6 (3.2, 4.0) | 6.0 (5.4, 6.6) | 16.1 (15.2, 17.1) | 74.3 (73.0, 75.6) | 61 (56, 66) | 9.6 (8.9, 10.3) | 15.7 (14.7, 16.7) | 74% |
| x-combo1-dr-chance (25 opponents) | 75,000 | 4.4 (4.0, 4.8) | 6.1 (5.6, 6.6) | 17.2 (16.3, 18.1) | 72.3 (71.0, 73.6) | 54 (50, 58) | 10.5 (9.8, 11.2) | 17.1 (16.1, 18.2) | 73% |
| aMCTS-pp (54 opponents) | 162,000 | 4.6 (4.2, 4.9) | 6.1 (5.6, 6.6) | 16.8 (15.9, 17.6) | 72.6 (71.3, 73.9) | 56 (52, 60) | 10.7 (10.0, 11.4) | 17.1 (16.1, 18.0) | 73% |
| aMCTS-cur (38 opponents) | 114,000 | 17.5 (16.7, 18.3) | 14.5 (13.6, 15.4) | 22.6 (21.6, 23.6) | 45.4 (44.1, 46.8) | 15 (14, 17) | 32.0 (30.9, 33.2) | 41.9 (40.6, 43.1) | 46% |
| a1step-cur (18 opponents) | 54,000 | 5.5 (5.1, 6.0) | 7.8 (7.2, 8.3) | 19.6 (18.7, 20.5) | 67.1 (65.7, 68.5) | 43 (40, 46) | 13.3 (12.4, 14.1) | 21.0 (19.8, 22.2) | 68% |
| pMCTS-cur (52 opponents) | 156,000 | 18.3 (17.5, 19.2) | 14.9 (14.1, 15.9) | 22.7 (21.7, 23.7) | 44.1 (42.7, 45.5) | 14 (13, 16) | 33.3 (32.1, 34.5) | 43.3 (42.0, 44.6) | 45% |
| p1step-cur (18 opponents) | 54,000 | 11.2 (10.6, 11.7) | 12.1 (11.4, 12.7) | 23.2 (22.3, 24.1) | 53.6 (52.2, 55.0) | 23 (22, 25) | 23.2 (22.2, 24.3) | 32.9 (31.7, 34.2) | 54% |

Arms: `s-pcal-floor25` final drafter (promoted 2026-11-14); `s-floor25` support floor 25, unscaled personal term (stand-in for s-pcal-floor25); `x-combo1-dr-chance` selected unconstrained design; `aMCTS-pp` player-aware MCTS, player prior; `aMCTS-cur` player-aware MCTS; `a1step-cur` player-aware one-step; `pMCTS-cur` population MCTS (paper 1); `p1step-cur` population one-step.
