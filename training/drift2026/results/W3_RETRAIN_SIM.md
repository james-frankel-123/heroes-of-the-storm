# W3(g) — Auto-retrain simulation refreshed with the wave-3 arms

Same timeline as W2c (deploy after build 8 = 2.55.3.89754, 36 simulated builds, 1,309,961 games, games-weighted accuracy; regret vs the cumprev always-retrain oracle). Wave-3 policies use the (d)/(e) arms trained at the same cutoffs (added: hybrid, embedrefresh).

| policy | retrains | weighted acc % | regret vs oracle (pp) |
|---|---|---|---|
| oracle cumprev (retrain every build) | 35 | 56.811 | +0.000 |
| never retrain + frozen stats | 0 | 55.128 | +1.682 |
| never retrain + cumprev refresh | 0 | 56.080 | +0.730 |
| retrain every K=3 + cumprev refresh | 11 | 56.782 | +0.029 |
| retrain every K=6 + cumprev refresh | 5 | 56.743 | +0.067 |
| never retrain + hybrid per-signal refresh | 0 | 55.778 | +1.033 |
| never retrain + embed + cumprev refresh | 0 | 56.096 | +0.714 |
| retrain every K=6 + embed + cumprev refresh | 5 | 56.732 | +0.078 |

## Recommended site policy

- **retrain every K=3 + cumprev refresh** (weighted acc 56.782, regret +0.029 pp, 11 retrains over ~2.8 years).
- Best overall: oracle cumprev (retrain every build) (56.811).
