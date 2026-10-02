# Signal decay (D2a)

Games-weighted Pearson r between build-N stats and build-N+k stats,
averaged over all sizable-build pairs at each lag; r_adj corrects for
binomial sampling attenuation. half-life from exp fit over day-lags.

| signal | r@1 | r_adj@1 | r@3 | r_adj@3 | r@8 | r_adj@8 | r@16 | r_adj@16 | half-life (d) | floor |
|---|---|---|---|---|---|---|---|---|---|---|
| hero_wr | 0.845 | 0.941 | 0.788 | 0.878 | 0.700 | 0.783 | 0.565 | 0.623 | 777.5 | 0.48 |
| hero_pickrate | 0.982 | 0.984 | 0.963 | 0.965 | 0.936 | 0.939 | 0.884 | 0.886 | 3465.7 | 0.50 |
| pair_with_raw | 0.558 | 0.939 | 0.515 | 0.871 | 0.448 | 0.774 | 0.379 | 0.618 | 718.6 | 0.49 |
| pair_with_net | 0.151 | - | 0.151 | - | 0.130 | - | 0.142 | - | 3465.7 | 0.08 |
| pair_against_raw | 0.569 | 0.938 | 0.526 | 0.873 | 0.456 | 0.773 | 0.385 | 0.612 | 841.1 | 0.45 |
| pair_against_net | 0.177 | 1.000 | 0.184 | 1.000 | 0.175 | - | 0.180 | 1.000 | - | - |
| comp_wr | 0.871 | 0.983 | 0.834 | 0.942 | 0.824 | 0.950 | 0.847 | 0.948 | 42.3 | 0.95 |
