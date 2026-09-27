# W4 — MCTS policies under best vs worst drift-regime VFs

5 seeds/arm, 200 sims, 300000 episodes; opponents/benchmark = future-GD (post-cutoff meta); metrics vs future-truth stats; terminal WP by neutral wp_enriched_256.

| arm | n | judge WP | counter | synergy | healer % | degen % | R_early | R_late |
|---|---|---|---|---|---|---|---|---|
| d2c_cumprev | 5 | 0.7134 ± 0.010 | +0.076 | +0.427 | 91.0 | 9.3 | -0.008 | +0.039 |
| d2b_allhist | 5 | 0.7031 ± 0.010 | +0.064 | +0.668 | 84.7 | 17.6 | -0.061 | -0.047 |
