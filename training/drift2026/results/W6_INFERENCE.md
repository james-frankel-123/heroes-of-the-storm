# W6 inference under crossed seed effects (method-of-moments ANOVA)

Per-draft seed labels reconstructed from the deterministic run order and
verified against stored pairing means. Two-way random-effects decomposition
of the 5x5 pairing matrix (seed_maintained x seed_frozen), grand-mean SE
includes seed-A, seed-B, interaction-beyond-sampling, and within-pairing
sampling components.

| estimand | estimate | SE | z vs 0.5 |
|---|---|---|---|
| consensus score, crossed RE | 0.5084 | 0.0112 | 0.75 |
| binary win share, crossed RE | 0.5340 | 0.0514 | 0.66 |
| conservative 5 disjoint pairings | 0.5053 | 0.0150 | 0.35 |

Units: the earlier "SE 0.002" was the iid SE of the mean CONTINUOUS
consensus score (0.0019); the binomial SE
of 2,000 binary outcomes is 0.0112. The crossed-RE SEs
above are the reportable ones. Variance components (continuous):
seed_maintained 3.90e-04, seed_frozen 1.82e-04,
interaction 1.94e-04, within-pairing sampling 8.29e-05.

## Judge-free paired deltas under the same crossed-RE decomposition

| metric | estimate | SE | z |
|---|---|---|---|
| future_hero_wr_delta_pp | +0.8017 | 0.1973 | 4.06 |
| synergy_delta | +0.7611 | 0.3245 | 2.35 |
| degen_delta | -0.2080 | 0.1141 | -1.82 |
