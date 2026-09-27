# Signal decay (D2a)

Games-weighted Pearson r between build-N stats and build-N+k stats,
averaged over all sizable-build pairs at each lag; r_adj corrects for
binomial sampling attenuation. half-life from exp fit over day-lags.

| signal | r@1 | r_adj@1 | r@3 | r_adj@3 | r@8 | r_adj@8 | r@16 | r_adj@16 | half-life (d) | floor |
|---|---|---|---|---|---|---|---|---|---|---|
| hero_wr | 0.846 | 0.941 | 0.788 | 0.878 | 0.700 | 0.783 | 0.565 | 0.622 | 777.5 | 0.48 |
| hero_pickrate | 0.982 | 0.984 | 0.963 | 0.965 | 0.936 | 0.939 | 0.884 | 0.887 | 3465.7 | 0.50 |
| pair_with_raw | 0.558 | 0.938 | 0.516 | 0.871 | 0.448 | 0.773 | 0.380 | 0.617 | 718.6 | 0.49 |
| pair_with_net | 0.152 | - | 0.151 | - | 0.130 | - | 0.144 | - | 3465.7 | 0.09 |
| pair_against_raw | 0.570 | 0.938 | 0.527 | 0.873 | 0.457 | 0.773 | 0.386 | 0.611 | 841.1 | 0.45 |
| pair_against_net | 0.178 | 1.000 | 0.184 | 1.000 | 0.175 | - | 0.182 | 1.000 | - | - |
| comp_wr | 0.871 | 0.983 | 0.836 | 0.944 | 0.824 | 0.952 | 0.844 | 0.947 | 45.8 | 0.95 |
