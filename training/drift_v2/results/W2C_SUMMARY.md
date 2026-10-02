# W2c — auto-retrain policy simulation over the build timeline

Deployment starts after build 8 (2.55.3.89754, ~638K games of history, 2023-07); simulated forward over the remaining 36 builds (~2.8 years, 1,304,406 games). All retrain policies train regime=all on cumulative_prev (causal) features, seed 42 (wave-1 seed std <= 0.16pp). Stats refresh = per-build-
boundary refresh of the feature aggregates (cumprev). Accuracy is
games-weighted over the simulated builds; regret is vs the
always-retrain oracle.

| policy | retrains | weighted acc % | regret vs oracle (pp) |
|---|---|---|---|
| oracle (retrain every build) | 35 | 56.866 | +0.000 |
| never retrain + frozen stats | 0 | 55.168 | +1.698 |
| never retrain + stats refresh | 0 | 56.204 | +0.662 |
| retrain every K=1 builds + refresh | 35 | 56.866 | +0.000 |
| retrain every K=3 builds + refresh | 11 | 56.868 | -0.002 |
| retrain every K=6 builds + refresh | 5 | 56.785 | +0.081 |
| trigger: acc drop > 0.25pp + refresh | 10 | 56.808 | +0.058 |
| trigger: acc drop > 0.5pp + refresh | 6 | 56.774 | +0.092 |
| trigger: acc drop > 1.0pp + refresh | 2 | 56.618 | +0.248 |

## Decomposition

- Stats refresh alone (no retrain): +1.037 pp over frozen stats (56.204 vs 55.168).
- Retraining on top of refresh (oracle vs never-retrain+refresh): +0.662 pp.
