# W2b — policy-level evaluation of the drifted value functions

Greedy drafting (paper-1 machinery) with each arm's deployment-time
stats, opponents = rerun2026 GD pool (sampled), 3 seeds x 500 drafts
per cell. counter/synergy are scored against FUTURE-PERIOD ground
truth (merged post-cutoff per-build counts). Accuracy/sanity columns
are the wave-1 numbers for the same checkpoints.

| cell | future acc | sanity 28 | healer % | degen % | counter | synergy | entropy |
|---|---|---|---|---|---|---|---|
| d2b_allhist | 56.21 ± 0.05 | 24.0 ± 0.0 | 63.2 ± 1.0 | 43.5 ± 1.9 | 0.19 ± 0.03 | 0.50 ± 0.05 | 6.10 ± 0.00 |
| d2b_win6 | 56.53 ± 0.01 | 25.0 ± 0.0 | 56.0 ± 0.7 | 49.7 ± 0.9 | 0.20 ± 0.00 | 0.50 ± 0.07 | 6.03 ± 0.01 |
| d2b_decay90 | 56.48 ± 0.01 | 25.3 ± 0.5 | 56.5 ± 0.5 | 50.1 ± 0.8 | 0.19 ± 0.02 | 0.47 ± 0.06 | 6.05 ± 0.03 |
| d2b_embed | 56.55 ± 0.02 | 25.7 ± 0.5 | 59.9 ± 4.7 | 47.0 ± 4.6 | 0.15 ± 0.05 | 0.51 ± 0.07 | 6.10 ± 0.03 |
| d2c_cumprev | 57.08 ± 0.03 | 24.0 ± 0.0 | 87.6 ± 1.3 | 16.8 ± 0.7 | 0.15 ± 0.02 | 0.23 ± 0.06 | 5.99 ± 0.01 |
| gd (reference) | - | - | 99.4 ± 0.0 | 1.1 ± 0.0 | 0.05 ± 0.00 | -0.13 ± 0.00 | 4.34 ± 0.00 |

## Spreads across regimes (W4 trigger)

- future-accuracy spread: 0.87 pp
- degen-rate spread: 33.27 pp (trigger: > 3 pp)
- sanity-28 spread: 1.67 (trigger: >= 3)
- healer-rate spread: 31.60 pp
- **W4 trigger fired: True** (W4 runs unconditionally per 2026-07-07 decision; the trigger matters for the paper's framing)
