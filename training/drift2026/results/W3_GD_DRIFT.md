# W3/W5 — GD (opponent/behavior model) drift arm

Gap (Max): all drift arms varied only the WP value function while the GD behavioral model stayed frozen (rerun2026 full-snapshot GD, which has seen the future period). W3b showed pick/ban changepoints track community ADAPTATION — behavior drifts too. Same W2 temporal cutoff (train <= 2.55.14.95918, test = last 7 builds).

## (a) Next-pick prediction accuracy on FUTURE-build drafts

1,844,688 next-pick samples (7 post-cutoff builds); train-val = 2% replay-level slice of the <=cutoff period. GenericDraftModel protocol reused unchanged (train_single_model); cutoff/win6 = 2 variants (seeds 42/123), frozen/future = 5.

| GD pool | trained on | n | future top-1 | future top-5 | train-val top-1 | pick top-1 | ban top-1 |
|---|---|---|---|---|---|---|---|
| frozen | full snapshot incl. future (paper-1 pool; non-causal bound) | 5 | 11.71 ± 0.107 | 36.31 | 13.22 | 9.02 | 16.18 |
| cutoff | all builds <= cutoff (deployable) | 2 | 11.08 ± 0.0 | 34.47 | 13.24 | 8.58 | 15.25 |
| win6 | last 6 builds <= cutoff (deployable, recency) | 2 | 11.64 ± 0.015 | 36.91 | 11.87 | 9.04 | 16.00 |
| win30d | last 30 calendar days <= cutoff (within-build rolling window) | 2 | 11.83 ± 0.02 | 36.98 | 7.37 | 8.65 | 17.13 |
| future_oracle | post-cutoff builds only (leaky oracle ref) | 5 | 13.09 ± 0.082 | 40.39 | 9.23 | 9.94 | 18.35 |

- Behavioral-drift gap (frozen - cutoff, future top-1): **+0.63 pp**
- Recency effect (win6 - cutoff, future top-1): **+0.56 pp**
- Within-build effect (win30d - cutoff, future top-1): **+0.75 pp**

Per-build future top-1 (columns = pools):

| build | n samples | frozen | cutoff | win6 | win30d | future_oracle |
|---|---|---|---|---|---|---|
| 2.55.15.96370 | 197,648 | 11.48 | 10.91 | 11.77 | 13.12 | 13.62 |
| 2.55.15.96443 | 10,976 | 11.07 | 10.50 | 11.11 | 12.55 | 13.14 |
| 2.55.15.96477 | 1,040,896 | 11.91 | 11.16 | 11.77 | 12.18 | 13.45 |
| 2.55.16.96846 | 12,944 | 11.53 | 10.88 | 11.46 | 11.05 | 12.81 |
| 2.55.16.96870 | 17,968 | 11.46 | 10.88 | 11.52 | 11.06 | 12.50 |
| 2.55.16.96881 | 407,776 | 11.42 | 10.97 | 11.38 | 10.80 | 12.32 |
| 2.55.16.97039 | 156,480 | 11.50 | 11.16 | 11.40 | 10.61 | 12.14 |

## (b) Policy-level: W2b greedy eval with GD_cutoff opponents/completions

Same W2b protocol (3 seeds x 500 drafts; counter/synergy vs future-truth stats); only the GD pool changes.

| cell | GD pool | n | healer % | degen % | counter | synergy |
|---|---|---|---|---|---|---|
| d2c_cumprev | frozen | 3 | 87.6 ± 1.3 | 16.8 ± 0.7 | +0.15 ± 0.02 | +0.23 ± 0.06 |
| d2c_cumprev | cutoff | 3 | 90.5 ± 1.4 | 14.1 ± 0.7 | +0.19 ± 0.03 | +0.37 ± 0.11 |
| d2b_allhist | frozen | 3 | 63.2 ± 1.0 | 43.5 ± 1.9 | +0.19 ± 0.03 | +0.50 ± 0.06 |
| d2b_allhist | cutoff | 3 | 69.0 ± 0.7 | 40.0 ± 1.6 | +0.12 ± 0.03 | +0.48 ± 0.04 |
| gd | frozen | 1 | 99.4 ± 0.0 | 1.1 ± 0.0 | +0.05 ± 0.00 | -0.13 ± 0.00 |
| gd | cutoff | 1 | 99.3 ± 0.0 | 0.9 ± 0.0 | +0.06 ± 0.00 | -0.13 ± 0.00 |

d2c_cumprev deltas (cutoff - frozen pool): healer +2.93 pp, degen -2.67 pp.

## (c) MCTS arm (conditional)

Trigger (a: frozen-cutoff future top1 >= 1.0pp; b: |d healer| or |d degen| >= 3pp (d2c_cumprev)):
- material (a) accuracy: False (gap +0.63 pp)
- material (b) policy: False
- **fires: False**

Trigger did NOT fire — (a)/(b) effects immaterial by the pre-registered thresholds; MCTS arm skipped.
