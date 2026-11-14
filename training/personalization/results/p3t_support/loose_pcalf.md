| arm | <25 | <10 | median min3 | J vs sel | J vs pMCTS-cur | record vs pMCTS-cur | J vs Real | record vs Real | CONS vs pMCTS-cur |
|---|---|---|---|---|---|---|---|---|---|
| humans' own teams | 4.7 | 1.2 | 208 | | | | | | |
| x-combo1-dr-chance | 34.6 | 11.6 | 42 | - | 59.1 (58.5, 59.7) | 0.7 (-2.1, 3.4) | 62.9 (62.3, 63.6) | 1.0 (-1.0, 3.0) | 44.6 (43.9, 45.3) |
| s-pcal-floor25 | 1.0 | 0.0 | 54 | - | 58.9 (58.2, 59.5) | 1.0 (-1.6, 3.7) | - | - | 44.9 (44.4, 45.3) |

| arm | vs | J | judge-free record (pp, fair metric) | CONS | coverage both |
|---|---|---|---|---|---|
| x-combo1-dr-chance | pMCTS-cur | 59.1 (58.5, 59.7) | 0.7 (-2.1, 3.4) | 44.6 (43.9, 45.3) | 36% |
| x-combo1-dr-chance | aMCTS-pp | 52.3 (51.8, 52.9) | 1.0 (-0.5, 2.6) | 49.1 (48.7, 49.5) | 57% |
| x-combo1-dr-chance | GD | 66.0 (65.4, 66.6) | 0.5 (-1.9, 3.0) | 59.1 (58.4, 59.7) | 46% |
| x-combo1-dr-chance | DraftRec | 57.9 (57.0, 58.8) | 0.9 (-0.8, 2.5) | 57.8 (57.2, 58.3) | 62% |
| x-combo1-dr-chance | Real | 62.9 (62.3, 63.6) | 1.0 (-1.0, 3.0) | 59.9 (59.4, 60.3) | 54% |
| s-pcal-floor25 | pMCTS-cur | 58.9 (58.2, 59.5) | 1.0 (-1.6, 3.7) | 44.9 (44.4, 45.3) | 35% |
| s-pcal-floor25 | aMCTS-pp | 51.9 (51.3, 52.5) | 1.2 (-0.4, 2.8) | 49.0 (48.6, 49.5) | 56% |

Caveat (s-pcal arms): J and CONS use the uncalibrated personal model (the judge's c2 and the drafters' unscaled b2), so they are biased toward the selected design; the judge-free paired record is the fair metric for the pscale arms. pscale = 0.864, by the rule fixed in advance: mean of all_known p1 min3-bin slope_P (results/p3_team_support.json residual stage).
