# W2b fill — Table II's missing policy-level cells

d2b_win3 / d2b_win12 / d2b_decay365 evaluated with the identical
W2b protocol (eval_policy_w2b.py, 3 seeds x 500 greedy drafts vs the
rerun2026 GD pool, counter/synergy vs FUTURE-PERIOD ground truth;
checkpoints = the wave-1 models). Accuracy/sanity columns are the
wave-1 numbers for the same checkpoints, as in W2B_SUMMARY.md.
Existing-cell rows are re-aggregated from the same stored per-seed
jsons W2B_SUMMARY used, for one-table reading.

| cell | future acc | sanity 28 | healer % | degen % | counter | synergy | entropy |
|---|---|---|---|---|---|---|---|
| d2b_allhist | 56.21 ± 0.05 | 24.0 ± 0.0 | 63.2 ± 1.0 | 43.5 ± 1.9 | 0.19 ± 0.03 | 0.50 ± 0.05 | 6.10 ± 0.00 |
| d2b_win3 **(new)** | 56.37 ± 0.05 | 24.3 ± 0.9 | 59.9 ± 2.2 | 46.2 ± 1.6 | 0.19 ± 0.01 | 0.50 ± 0.07 | 6.08 ± 0.04 |
| d2b_win6 | 56.53 ± 0.01 | 25.0 ± 0.0 | 56.0 ± 0.7 | 49.7 ± 0.9 | 0.20 ± 0.00 | 0.50 ± 0.07 | 6.03 ± 0.01 |
| d2b_win12 **(new)** | 56.47 ± 0.04 | 25.0 ± 0.8 | 56.3 ± 0.7 | 49.9 ± 0.7 | 0.16 ± 0.01 | 0.56 ± 0.08 | 6.04 ± 0.02 |
| d2b_decay90 | 56.48 ± 0.01 | 25.3 ± 0.5 | 56.5 ± 0.5 | 50.1 ± 0.8 | 0.19 ± 0.02 | 0.47 ± 0.06 | 6.05 ± 0.03 |
| d2b_decay365 **(new)** | 56.36 ± 0.04 | 25.0 ± 0.0 | 60.4 ± 0.8 | 47.5 ± 0.7 | 0.18 ± 0.04 | 0.54 ± 0.04 | 6.09 ± 0.01 |
| d2b_embed | 56.55 ± 0.02 | 25.7 ± 0.5 | 59.9 ± 4.7 | 47.0 ± 4.6 | 0.15 ± 0.05 | 0.51 ± 0.07 | 6.10 ± 0.03 |
| d2c_cumprev | 57.08 ± 0.03 | 24.0 ± 0.0 | 87.6 ± 1.3 | 16.8 ± 0.7 | 0.15 ± 0.02 | 0.23 ± 0.06 | 5.99 ± 0.01 |

## Pipeline reproduction check

- Before the new cells ran, d2b_win6 seed 42 was re-run end-to-end (output redirected outside results/): healer 56.0, degen 49.0, counter +0.201, synergy +0.495, entropy 6.04 — all EXACTLY equal to the stored results/w2b/d2b_win6_s42.json (diff 0.000 on every metric).
