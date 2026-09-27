# W2a — within-patch-causal local enrichment (fixes d2c_local leakage)

Every row's enriched stats = rolling same-build stats from strictly
EARLIER calendar days, blended with the cumulative-to-cutoff prior
(count-weighted shrinkage; kw = prior weight per statistic).
Deployable: test rows never see the row's own day or any post-cutoff
build other than their own build's past. 3 seeds per cell.

| cell | prior weight | val acc | future acc | sizable | sanity 28 |
|---|---|---|---|---|---|
| d2c_causal_merge | true prior counts (~1.8M games) | 59.12 ± 0.02 | 56.26 ± 0.03 | 56.25 ± 0.03 | 25.3 ± 0.5 |
| d2c_causal_k100 | k=100 pseudo-games | 57.48 ± 0.11 | 57.01 ± 0.02 | 57.02 ± 0.02 | 24.0 ± 0.0 |
| d2c_causal_k1000 | k=1000 pseudo-games | 58.44 ± 0.02 | 56.79 ± 0.08 | 56.79 ± 0.08 | 25.3 ± 0.5 |
| d2b_allhist | (cutoff stats only — no local) | 59.11 ± 0.03 | 56.21 ± 0.05 | 56.20 ± 0.05 | 24.0 ± 0.0 |
| d2c_cumprev | (cumulative refresh, no within-build) | 57.16 ± 0.02 | 57.08 ± 0.03 | 57.09 ± 0.04 | 24.0 ± 0.0 |
| d2c_frozen | (paper-1 leaky control) | 57.35 ± 0.11 | 57.02 ± 0.04 | 57.00 ± 0.04 | 25.0 ± 0.0 |
| d2c_local | (LEAKY within-patch oracle) | 70.93 ± 0.06 | 72.77 ± 0.02 | 72.88 ± 0.02 | 22.7 ± 0.9 |

## Verdict

- Best causal-local arm: **d2c_causal_k100** at 57.01% future acc (+0.80 vs the cutoff-stats baseline d2b_allhist, -0.07 vs the cumprev refresh champion, vs the leaky d2c_local oracle at 72.77).

## Within-build depth (accuracy by date tercile, sizable test builds)

Causal-local arms should IMPROVE across terciles (local stats accumulate); cumprev/cutoff should stay ~flat.

| arm | build | early | mid | late | late-early |
|---|---|---|---|---|---|
| causal_k100/d2c_causal_k100 | 2.55.15.96370 | 56.75 | 56.33 | 55.70 | -1.05 |
| causal_k100/d2c_causal_k100 | 2.55.15.96477 | 56.86 | 57.42 | 57.50 | +0.64 |
| causal_k100/d2c_causal_k100 | 2.55.16.96881 | 56.78 | 57.40 | 57.11 | +0.33 |
| causal_k100/d2c_causal_k100 | 2.55.16.97039 | 56.42 | 56.23 | 55.70 | -0.72 |
| causal_merge/d2c_causal_merge | 2.55.15.96370 | 56.22 | 55.56 | 56.04 | -0.18 |
| causal_merge/d2c_causal_merge | 2.55.15.96477 | 56.72 | 56.56 | 55.82 | -0.90 |
| causal_merge/d2c_causal_merge | 2.55.16.96881 | 55.77 | 56.08 | 56.65 | +0.88 |
| causal_merge/d2c_causal_merge | 2.55.16.97039 | 57.45 | 55.55 | 55.86 | -1.59 |
| cumulative_prev/d2c_cumprev | 2.55.15.96370 | 57.03 | 56.53 | 56.65 | -0.38 |
| cumulative_prev/d2c_cumprev | 2.55.15.96477 | 57.53 | 57.30 | 56.76 | -0.77 |
| cumulative_prev/d2c_cumprev | 2.55.16.96881 | 57.41 | 56.89 | 57.20 | -0.21 |
| cumulative_prev/d2c_cumprev | 2.55.16.97039 | 57.75 | 57.19 | 56.19 | -1.56 |
| cutoff/d2b_allhist | 2.55.15.96370 | 56.12 | 55.63 | 56.03 | -0.09 |
| cutoff/d2b_allhist | 2.55.15.96477 | 56.64 | 56.52 | 55.66 | -0.98 |
| cutoff/d2b_allhist | 2.55.16.96881 | 55.88 | 56.11 | 56.36 | +0.48 |
| cutoff/d2b_allhist | 2.55.16.97039 | 57.28 | 55.35 | 55.52 | -1.76 |
