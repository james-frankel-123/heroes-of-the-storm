# W3(d) — Hybrid per-signal enrichment sourcing

hero-WR family from a recent window (target 12,237 games
, from the W3(a) recovery answer; at deployment 1 builds / 31,750 games), role-comp WR from all history, pairwise from cumulative — all strictly causal per row (cumprev-style). 3 seeds; policy metrics = W2b protocol (500 greedy drafts/seed vs GD pool, scored vs future truth).

| cell | val acc | future acc | sizable | sanity 28 | healer % | degen % | counter | synergy |
|---|---|---|---|---|---|---|---|---|
| w3d_hybrid | 57.18 ± 0.07 | 56.91 ± 0.05 | 56.91 ± 0.05 | 25.3 ± 0.5 | 86.0 ± 1.9 | 18.7 ± 2.0 | 0.10 ± 0.03 | 0.21 ± 0.08 |
| d2c_cumprev | 57.16 ± 0.02 | 57.08 ± 0.03 | 57.09 ± 0.04 | 24.0 ± 0.0 | 87.6 ± 1.3 | 16.8 ± 0.7 | 0.15 ± 0.02 | 0.23 ± 0.06 |
| d2b_embed | 59.14 ± 0.02 | 56.55 ± 0.02 | 56.55 ± 0.02 | 25.7 ± 0.5 | 59.9 ± 4.7 | 47.0 ± 4.6 | 0.15 ± 0.05 | 0.51 ± 0.07 |
| d2b_win6 | 58.97 ± 0.16 | 56.53 ± 0.01 | 56.54 ± 0.02 | 25.0 ± 0.0 | 56.0 ± 0.7 | 49.7 ± 0.9 | 0.20 ± 0.00 | 0.50 ± 0.07 |
| d2b_allhist | 59.11 ± 0.03 | 56.21 ± 0.05 | 56.20 ± 0.05 | 24.0 ± 0.0 | 63.2 ± 1.0 | 43.5 ± 1.9 | 0.19 ± 0.03 | 0.50 ± 0.05 |

## Verdict

- w3d_hybrid future acc 56.91 vs d2c_cumprev 57.08 (-0.17 pp): the per-signal window sourcing does **not** beat the uniform cumulative refresh.

Per test build (hybrid): 2.55.15.96370: 56.40, 2.55.15.96443: 56.13, 2.55.15.96477: 57.02, 2.55.16.96846: 56.35, 2.55.16.96870: 57.65, 2.55.16.96881: 56.99, 2.55.16.97039: 56.64
