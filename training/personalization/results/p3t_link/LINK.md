# Does the missing player factor explain the value function's overconfidence?

Script: `personalization/p3t_link.py`. Numbers: `link.json`. Real games only, judge window 2026-07-01 to 09-11. 153,705 of 162,657 games join to map and tier (94.5%). GLMs run on half B (odd games, 76,859 games, two team rows each), with SEs clustered by game.

- b is paper 1's main-effects baseline (current split).
- I = logit WP_pop − logit b, the hero-generic model's interaction claim.
- P is the drafter's pcal personal term, 0.864·(b2·dS + b3·dO) in logit units, with the state as of 06-30.
- A slot is flagged when the player had fewer than 5 earlier games on the hero at game time, or the hero is off his role.

## Answer: no

The player factor is essentially uncorrelated with I: corr(I, P) is 0.005 to 0.010 in every group. Adding P moves beta by +0.01 at most. The overconfidence in I is a hero-generic effect.

| group | team rows | beta | beta with P | gamma on P |
|---|---|---|---|---|
| all | 153,718 | 0.843 (0.026) | 0.852 (0.026) | 0.865 (0.019) |
| min3 < 25 | 9,465 | 0.936 (0.070) | 0.950 (0.070) | 0.819 (0.052) |
| min3 25–99 | 35,140 | 0.916 (0.041) | 0.931 (0.042) | 0.835 (0.030) |
| min3 ≥ 100 | 109,113 | 0.805 (0.029) | 0.811 (0.029) | 0.881 (0.021) |
| flagged 0 | 12,035 | 0.896 (0.071) | 0.916 (0.072) | 1.000 (0.051) |
| flagged 1 | 34,886 | 0.853 (0.043) | 0.861 (0.044) | 0.910 (0.032) |
| flagged 2+ | 106,797 | 0.836 (0.029) | 0.843 (0.029) | 0.815 (0.021) |
| never-played slots 0 | 82,008 | 0.844 (0.032) | 0.854 (0.032) | 0.899 (0.023) |
| never-played slots 1 | 49,040 | 0.822 (0.037) | 0.831 (0.038) | 0.853 (0.027) |
| never-played slots 2+ | 22,670 | 0.895 (0.053) | 0.899 (0.054) | 0.704 (0.039) |

Fixing gamma at its half-A value (0.878) instead of fitting it gives the same betas to within 0.005.

- **Beta does not move more where players are off-pool.** The shift is +0.020 for flagged-0 teams and +0.007 for flagged-2+ teams. Both are far inside the SEs.
- **On these human teams, beta is lower on common compositions than on rare ones.** It is 0.81 at min3 ≥ 100 and 0.94 at min3 < 25.
- **What the flag does change is the personal term's own calibration.** Gamma on P falls from 1.00 (no flagged slots) to 0.82 (2+ flagged), and to 0.70 with 2+ never-played slots. The personal term is over-trusted mainly where players are on unfamiliar heroes. That is a personal-term issue; it does not explain beta.

## Composition (all judge-window human teams)

| own min3 | share of teams | flagged slot share | never-played slot share | teams with 2+ flagged |
|---|---|---|---|---|
| < 25 | 6.2% | 46.4% | 15.2% | 72.8% |
| 25–99 | 22.9% | 45.6% | 13.9% | 72.1% |
| 100–499 | 51.5% | 43.5% | 12.8% | 69.1% |
| ≥ 500 | 19.4% | 42.0% | 12.1% | 66.7% |

Rare compositions coincide only weakly with off-pool players: about 4 points more flagged slots and 3 points more never-played slots from the most common bin to the rarest.

Caveats:
- The familiarity counts start with the slot table (2021-12), so "fewer than 5 earlier games" flags 44% of slots.
- This is selection on human choices, and the drafter's own teams are not tested here.
