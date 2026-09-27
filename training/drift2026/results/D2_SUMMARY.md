# D2 regime / enrichment-sourcing summary (3 seeds per cell)

Future = all builds after 2.55.14.95918; sizable = the 4 headline
test builds. Val = train-period held-out slice. Sanity = paper-1
28-test suite (21-test subset in parens).

| cell | n_train | val acc | future acc | sizable acc | 15.96370 | 15.96477 | 16.96881 | 16.97039 | sanity 28 (21) | epochs |
|---|---|---|---|---|---|---|---|---|---|---|
| d2b_allhist | 3,538,596 | 59.11 ± 0.03 | 56.21 ± 0.05 | 56.20 ± 0.05 | 55.93 | 56.29 | 56.17 | 56.00 | 24.0 (17.0) | 35 |
| d2b_win3 | 275,298 | 58.77 ± 0.06 | 56.37 ± 0.05 | 56.38 ± 0.05 | 55.87 | 56.55 | 56.36 | 56.04 | 24.3 (17.7) | 38 |
| d2b_win6 | 575,126 | 58.97 ± 0.16 | 56.53 ± 0.01 | 56.54 ± 0.02 | 55.96 | 56.69 | 56.50 | 56.28 | 25.0 (18.0) | 32 |
| d2b_win12 | 1,179,048 | 59.41 ± 0.07 | 56.47 ± 0.04 | 56.46 ± 0.04 | 55.89 | 56.65 | 56.34 | 56.21 | 25.0 (18.3) | 32 |
| d2b_decay90 | 3,538,596 | 58.86 ± 0.06 | 56.48 ± 0.01 | 56.48 ± 0.02 | 55.92 | 56.69 | 56.35 | 56.19 | 25.3 (19.0) | 28 |
| d2b_decay365 | 3,538,596 | 59.11 ± 0.02 | 56.36 ± 0.04 | 56.35 ± 0.04 | 55.93 | 56.51 | 56.27 | 56.04 | 25.0 (18.0) | 32 |
| d2b_embed | 3,538,596 | 59.14 ± 0.02 | 56.55 ± 0.02 | 56.55 ± 0.02 | 56.06 | 56.75 | 56.41 | 56.28 | 25.7 (18.7) | 42 |
| d2c_local | 3,538,596 | 70.93 ± 0.06 | 72.77 ± 0.02 | 72.88 ± 0.02 | 76.75 | 70.33 | 76.08 | 76.56 | 22.7 (16.0) | 33 |
| d2c_cumprev | 3,538,596 | 57.16 ± 0.02 | 57.08 ± 0.03 | 57.09 ± 0.04 | 56.71 | 57.21 | 57.11 | 56.78 | 24.0 (18.7) | 58 |
| d2c_frozen | 3,538,596 | 57.35 ± 0.11 | 57.02 ± 0.04 | 57.00 ± 0.04 | 56.70 | 57.06 | 57.03 | 56.87 | 25.0 (18.0) | 45 |

## Reading notes

- **d2c_local is NOT a deployable number.** Per-patch stats include the test
  replay's own game and its patch-mates, so patch-local enriched features
  partially encode the label (comp/pairwise WRs on a 15K-game build are
  computed largely from the very games being predicted — note accuracy is
  HIGHEST on the SMALLEST test builds: 76.8% on 15.96370 [15K games] vs 70.3%
  on 15.96477 [81K]). Treat it as the patch-conditioning ORACLE upper bound;
  a deployable version needs within-patch-causal or leave-one-out stats
  (next wave).
- **d2c_frozen (paper-1 control) is also leaky by design** (end-of-time stats
  span the test patches); it is the control arm, not a deployable regime.
- Deployable ranking on strictly-future data: cumprev 57.08 > embed 56.55 >
  win6 56.53 ~ decay90 56.48 ~ win12 56.47 > decay365 56.36 > win3 56.37 >
  allhist 56.21. Two conclusions: (1) unconditioned all-history is the WORST
  way to use 4 years of data (train-period val acc is highest, future acc
  lowest — classic drift); (2) refreshing the FEATURE AGGREGATES causally
  after the training cutoff (cumprev) recovers ~2.5x more future accuracy
  than the best training-regime change (+0.87 vs +0.34 over allhist),
  without retraining.
