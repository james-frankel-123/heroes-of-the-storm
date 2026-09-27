# W2c addendum — warm-start finetune row (Q8)

Same deployment simulation and evaluation as W2C_SUMMARY.md (weights,
accuracy matrix and oracle read from results/w2c_policies.json). New
policy: at each retrain point of the K=3 cadence, instead of a cold
full retrain, initialize from the previous checkpoint in the chain
(chain starts at the cold C0 model w2c_cut08_s42) and finetune for
5 epochs at lr 5e-05 (10% of the protocol 5e-4) on the trailing 6-build
window, cumulative_prev causal features, seed 42, best-val-loss
checkpoint selection within the finetune epochs.

| policy | retrains | weighted acc % | regret vs oracle (pp) |
|---|---|---|---|
| oracle (retrain every build) | 35 | 56.811 | +0.000 |
| retrain every K=3 builds + refresh (cold) | 11 | 56.782 | +0.029 |
| **warm-start finetune every K=3 builds + refresh (5 ep @ lr 5e-05, trailing 6-build window)** | 11 | 56.722 | +0.089 |
| never retrain + stats refresh | 0 | 56.080 | +0.730 |

Cold K=3 cross-check recomputed from the matrix: 56.782 (published 56.782).

## Per-retrain-point comparison (games-weighted over the builds each checkpoint serves)

| retrain point | cutoff build | builds served | games | cold acc | finetune acc | delta (pp) | ft epochs | ft minutes |
|---|---|---|---|---|---|---|---|---|
| 11 | 2.55.3.91081 | 3 | 123,532 | 57.061 | 57.129 | +0.068 | 5 | 0.2 |
| 14 | 2.55.4.91418 | 3 | 230,761 | 56.785 | 56.624 | -0.162 | 5 | 0.2 |
| 17 | 2.55.6.92665 | 3 | 55,250 | 56.568 | 56.525 | -0.043 | 5 | 0.2 |
| 20 | 2.55.7.93151 | 3 | 51,422 | 56.879 | 56.763 | -0.116 | 5 | 0.2 |
| 23 | 2.55.9.93565 | 3 | 137,869 | 56.531 | 56.385 | -0.146 | 5 | 0.2 |
| 26 | 2.55.10.93810 | 3 | 117,470 | 56.439 | 56.493 | +0.054 | 5 | 0.2 |
| 29 | 2.55.10.94470 | 3 | 117,047 | 56.497 | 56.288 | -0.208 | 5 | 0.2 |
| 32 | 2.55.13.95213 | 3 | 166,270 | 56.847 | 56.783 | -0.064 | 5 | 0.2 |
| 35 | 2.55.14.95817 | 3 | 131,146 | 56.937 | 57.024 | +0.087 | 5 | 0.2 |
| 38 | 2.55.15.96370 | 3 | 82,783 | 57.285 | 57.173 | -0.112 | 5 | 0.2 |
| 41 | 2.55.16.96846 | 3 | 45,483 | 57.248 | 57.368 | +0.120 | 5 | 0.2 |

Total finetune compute: 2.2 min across 11 sequential jobs (vs 11 full cold retrains for the K=3 row).
