# W2c — auto-retrain policy simulation over the build timeline

Deployment starts after build 8 (2.55.3.89754, ~638K games of history, 2023-07); simulated forward over the remaining 36 builds (~2.8 years, 1,309,961 games). All retrain policies train regime=all on cumulative_prev (causal) features, seed 42 (wave-1 seed std <= 0.16pp). Stats refresh = per-build-
boundary refresh of the feature aggregates (cumprev). Accuracy is
games-weighted over the simulated builds; regret is vs the
always-retrain oracle.

| policy | retrains | weighted acc % | regret vs oracle (pp) |
|---|---|---|---|
| oracle (retrain every build) | 35 | 56.811 | +0.000 |
| never retrain + frozen stats | 0 | 55.128 | +1.682 |
| never retrain + stats refresh | 0 | 56.080 | +0.730 |
| retrain every K=1 builds + refresh | 35 | 56.811 | +0.000 |
| retrain every K=3 builds + refresh | 11 | 56.782 | +0.029 |
| retrain every K=6 builds + refresh | 5 | 56.743 | +0.067 |
| trigger: acc drop > 0.25pp + refresh | 11 | 56.755 | +0.056 |
| trigger: acc drop > 0.5pp + refresh | 7 | 56.733 | +0.078 |
| trigger: acc drop > 1.0pp + refresh | 3 | 56.503 | +0.308 |

## Decomposition

- Stats refresh alone (no retrain): +0.952 pp over frozen stats (56.080 vs 55.128).
- Retraining on top of refresh (oracle vs never-retrain+refresh): +0.730 pp.
