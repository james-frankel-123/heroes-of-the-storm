# Paper 2 paired record: player-clustered and two-way CIs, raw differences, coverage (2026-11-08)

Script: `training/personalization/p3t_paired_robust.py` (extract on the 3080 from the def1027 headline, rep50 and loose namespaces; analyze on the main box). Read-only on every existing artifact.

Points in pp, selected design (`x-combo1-dr-chance`) minus comparator; 95% CI = point +- 1.96 SE. Bootstrap draws: 2000 (lobby/player/two-way); the paper's lobby CI is reproduced with its own 500 draws, seed 0. Two-way CGM = Cameron-Gelbach-Miller multiway variance V_lobby + V_player - V_iid (V_iid from an unclustered slot bootstrap; floored at the larger one-way), the headline two-way interval. Two-way pigeonhole = product of independent lobby and player multinomial weights (Owen 2007), which double-counts the iid term and is conservative. Raw = realized win rate on the assigned hero, no WP subtraction, same covered players.

## Table V rows (300 headline lobbies)

| vs | point | lobby CI (paper, 500) | lobby CI (2k) | player CI | two-way CGM | two-way pigeonhole | SE x vs paper (player / CGM) | raw diff (CGM CI) | coverage sel / opp / both |
|---|---|---|---|---|---|---|---|---|---|
| The earlier search drafter | +4.77 | (2.2, 7.3) | (2.2, 7.3) | (2.2, 7.3) | **(2.1, 7.4)** | (0.5, 9.0) | 0.98 / 1.03 | +4.5 (1.8, 7.1) | 77% / 44% / 37% |
| Real drafts | +3.07 | (1.1, 5.1) | (1.1, 5.1) | (1.1, 5.1) | **(1.0, 5.2)** | (-0.5, 6.6) | 1.00 / 1.04 | +4.0 (1.9, 6.2) | 74% / 73% / 56% |
| Imitation drafter | +3.62 | (1.4, 5.8) | (1.4, 5.9) | (1.3, 5.9) | **(1.3, 6.0)** | (-0.3, 7.5) | 1.06 / 1.08 | +5.5 (3.1, 7.8) | 76% / 60% / 48% |
| DraftRec, pick head | +2.10 | (0.4, 3.8) | (0.5, 3.7) | (0.4, 3.8) | **(0.4, 3.8)** | (-0.8, 5.0) | 1.02 / 1.02 | +3.4 (1.6, 5.2) | 75% / 83% / 64% |
| DraftRec, pick-and-win rule | +3.52 | (1.6, 5.5) | (1.6, 5.4) | (1.6, 5.4) | **(1.6, 5.4)** | (0.2, 6.9) | 0.98 / 0.98 | +3.0 (1.1, 5.0) | 75% / 68% / 53% |
| Player-prior search | +0.67 | (-1.0, 2.3) | (-0.9, 2.2) | (-0.9, 2.2) | **(-0.9, 2.2)** | (-2.0, 3.3) | 0.92 / 0.95 | +0.6 (-1.0, 2.2) | 77% / 69% / 60% |

**Reading.** Player-cluster SEs are within 8% of the paper's lobby SEs, and both are close to the unclustered slot SE: a recurring player rarely gets the same hero pair again (repeated (player, hero A, hero B) triples are 4-9% of covered slots), so reuse of later games adds little correlation. The two-way CGM SE is 0.95-1.08x the paper's; intervals newly reaching 0 under CGM: none. The pigeonhole bound counts the slot-level variance two extra times (here about sqrt(3)x the one-way SE) and would put ['Real drafts', 'Imitation drafter', 'DraftRec, pick head'] at or below 0; it is reported as an upper bound only.


## Pooled figure (paper + A + loose's 171 = 771 lobbies)

Fixed-effect IV pool with the paper's weights; the SE from a joint bootstrap (lobbies within draw, players shared across draws). `iv-indep` re-pools with each scheme's per-draw SEs assuming independent draws.

| vs | point | paper IV CI | lobby (joint) | player (joint) | two-way CGM (joint) | two-way pigeonhole (joint) | raw diff (CGM CI) | per-draw (paper / A / loose171) |
|---|---|---|---|---|---|---|---|---|
| pMCTS-cur | +3.45 | (1.9, 5.0) | (1.9, 5.0) | (1.9, 5.0) | **(1.8, 5.1)** | (0.8, 6.1) | +2.7 (1.0, 4.4) | +4.8 / +3.9 / -0.3 |
| Real | +2.21 | (1.0, 3.4) | (1.0, 3.4) | (0.9, 3.5) | **(0.8, 3.6)** | (-0.0, 4.4) | +3.4 (2.0, 4.8) | +3.1 / +2.4 / +0.1 |
| GD | +3.37 | (2.0, 4.8) | (1.9, 4.8) | (1.9, 4.9) | **(1.8, 4.9)** | (0.9, 5.8) | +5.1 (3.5, 6.7) | +3.6 / +5.1 / -0.1 |
| DraftRec | +2.12 | (1.1, 3.2) | (1.1, 3.2) | (1.0, 3.2) | **(1.0, 3.2)** | (0.3, 3.9) | +3.4 (2.2, 4.5) | +2.1 / +3.0 / +0.9 |

## Player recurrence

- paper: {'slots': 3000, 'players': 2181, 'players_2plus': 536, 'slots_of_2plus': 1355, 'max': 9}
- A_replicate: {'slots': 3000, 'players': 2184, 'players_2plus': 548, 'slots_of_2plus': 1364, 'max': 7}
- loose_171: {'slots': 1710, 'players': 1474, 'players_2plus': 193, 'slots_of_2plus': 429, 'max': 6}
- pooled_771: {'slots': 7710, 'players': 4226, 'players_2plus': 1688, 'slots_of_2plus': 5172}
- players_shared_between_draws: {'paper&A': 831, 'paper&loose171': 542, 'A&loose171': 552}

Reuse of the same later games among covered slots (headline):

- The earlier search drafter: 1102 covered slots, 863 distinct players, 65 slots repeat a (player, hero A, hero B) triple
- Real drafts: 1692 covered slots, 1282 distinct players, 70 slots repeat a (player, hero A, hero B) triple
- Imitation drafter: 1432 covered slots, 1090 distinct players, 112 slots repeat a (player, hero A, hero B) triple
- DraftRec, pick head: 1928 covered slots, 1444 distinct players, 171 slots repeat a (player, hero A, hero B) triple
- DraftRec, pick-and-win rule: 1595 covered slots, 1234 distinct players, 120 slots repeat a (player, hero A, hero B) triple
- Player-prior search: 1807 covered slots, 1352 distinct players, 159 slots repeat a (player, hero A, hero B) triple

## Covered (both arms) vs not, headline lobbies

Medians (covered / uncovered) and the standardized mean difference.

| vs | earlier games (all heroes) | judge-window games (all heroes) | earlier games on the selected design's hero | earlier games on the comparator's hero | as-of rank (MMR) | slot MMR | party size | solo share |
|---|---|---|---|---|---|---|---|---|
| The earlier search drafter | 1323 / 830 (d +0.32) | 105 / 56 (d +0.54) | 90 / 53 (d +0.14) | 41 / 5 (d +0.62) | 2833 / 2799 (d +0.22) | 2837 / 2803 (d +0.22) | 1 / 1 (d -0.03) | 53% / 52% |
| Real drafts | 1216 / 754 (d +0.31) | 94 / 48 (d +0.58) | 98 / 34 (d +0.33) | 61 / 17 (d +0.44) | 2826 / 2788 (d +0.22) | 2831 / 2793 (d +0.21) | 1 / 1 (d -0.06) | 54% / 51% |
| Imitation drafter | 1378 / 744 (d +0.37) | 101 / 51 (d +0.60) | 90 / 42 (d +0.21) | 57 / 15 (d +0.53) | 2827 / 2797 (d +0.19) | 2831 / 2801 (d +0.18) | 1 / 1 (d +0.04) | 52% / 53% |
| DraftRec, pick head | 1186 / 710 (d +0.30) | 89 / 45 (d +0.58) | 99 / 23 (d +0.44) | 72 / 27 (d +0.41) | 2824 / 2785 (d +0.23) | 2828 / 2787 (d +0.23) | 1 / 1 (d -0.01) | 53% / 51% |
| DraftRec, pick-and-win rule | 1264 / 750 (d +0.35) | 94 / 50 (d +0.55) | 96 / 36 (d +0.27) | 54 / 15 (d +0.49) | 2824 / 2799 (d +0.19) | 2828 / 2802 (d +0.19) | 1 / 1 (d +0.00) | 53% / 51% |
| Player-prior search | 1219 / 713 (d +0.34) | 90 / 48 (d +0.58) | 105 / 29 (d +0.33) | 77 / 19 (d +0.52) | 2824 / 2792 (d +0.27) | 2829 / 2795 (d +0.26) | 1 / 1 (d -0.01) | 53% / 52% |

Tier shares covered/uncovered by row are in the JSON.

## Extraction checks

Recomputed per-slot records match the scored `rec_slot` in every pair file and draw: True; slot-table residual vs state max abs diff: 2.98e-08.

