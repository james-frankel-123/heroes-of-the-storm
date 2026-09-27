# W6 vintage judge matrix (Q2)

Naive-feature WP judges (qm2026 recipe: hero multi-hot ×2 + map + tier one-hots), each trained on the 314,761 most recent ranked games below its cutoff (volume-matched to the smallest window), 10% replay-level holdout. Scored on the 2,000 saved W6 head-to-head drafts: symmetrized P(maintained agent's comp wins).

Consensus reference (existing 4-evaluator W6 number): 0.5084.

| Judge vintage | Train window | Games | Held-out acc | AUC | Maintained WP | SE | Win share |
|---|---|---:|---:|---:|---:|---:|---:|
| QM-2021 | 2021 QM corpus | 91,365 | 0.6111 | 0.6500 | 0.4338 | 0.0033 | 0.322 |
| 2022-07 | 2021-12-07 .. 2022-07-31 | 314,761 | 0.5719 | 0.5985 | 0.4611 | 0.0020 | 0.335 |
| 2023-07 | 2022-08-15 .. 2023-07-31 | 314,761 | 0.5642 | 0.5891 | 0.4927 | 0.0014 | 0.466 |
| 2024-07 | 2023-08-22 .. 2024-07-31 | 314,761 | 0.5654 | 0.5913 | 0.5040 | 0.0019 | 0.534 |
| 2025-07 | 2024-12-04 .. 2025-07-31 | 314,761 | 0.5596 | 0.5820 | 0.4948 | 0.0017 | 0.454 |
| 2026-build | 2025-09-17 .. 2026-02-10 | 314,761 | 0.5697 | 0.5956 | 0.4996 | 0.0019 | 0.492 |

Judges saved under `drift2026/models/vintage_judges/`. Seed 20260713; recipe identical across vintages (only the era of the training window differs). Because sampling is recent-first below each cutoff at fixed volume, the training windows are (nearly) disjoint era slices, not nested supersets.

## Seed robustness (2 extra training seeds: 7, 991)

| Judge vintage | Maintained WP per seed | Mean | SD across seeds |
|---|---|---:|---:|
| 2024-07 | 0.5040 / 0.5049 / 0.5015 | 0.5034 | 0.0018 |
| 2025-07 | 0.4948 / 0.5084 / 0.4981 | 0.5004 | 0.0071 |
| 2026-build | 0.4996 / 0.5202 / 0.5182 | 0.5127 | 0.0114 |

Judge-training seed noise on this metric is ~0.5–1.1pp, larger than the
across-draft SE. Reading the matrix with that in mind:

- The rise from old to new vintages is far outside seed noise and monotone:
  QM-2021 0.434 → 2022-07 0.461 → 2023-07 0.493 → ~0.50–0.51 plateau from
  2024-07 on. Older judges favor the maintained agent decisively less, as
  predicted (the maintained agent optimizes into the 2026 meta, which
  earlier-era judges score as mediocre-to-losing).
- The apparent 2024-07 → 2025-07 dip (0.5040 → 0.4948, single seed) is
  within seed noise (3-seed means 0.5034 vs 0.5004) and should not be read
  as a reversal.
- The 2026-build judge's 3-seed mean maintained WP is **0.5127**, bracketing
  the existing 4-evaluator consensus number 0.5084 (single-seed values
  0.4996 / 0.5202 / 0.5182) — the newest naive judge agrees with the
  consensus that the maintained agent is ahead.
