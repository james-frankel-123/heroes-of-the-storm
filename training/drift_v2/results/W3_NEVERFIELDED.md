# W3(c) — Never-fielded composition set drift

Universe = 252 sorted 5-role tuples; observed at >= 5 games/build (pooled tiers), 28 sizable builds.

Consecutive-build Jaccard of observed comp sets: mean 0.806 (min 0.639); never-fielded set per build has 105-164 comps vs 92 never fielded in the whole corpus — per-build variation is volume-driven (rare comps drop below the observation threshold in smaller builds), not meta-driven.

Per-era tier-2 augmentation targets (cumulative-history never-seen sets per tier) vs the full-corpus target:

| era (through build) | low: target size / sym-diff vs full | mid: target size / sym-diff vs full | high: target size / sym-diff vs full |
|---|---|---|---|
| 2.55.0.86938 | 86 / 69 | 113 / 80 | 140 / 99 |
| 2.55.8.93382 | 25 / 8 | 47 / 14 | 65 / 24 |
| 2.55.14.95918 | 17 / 0 | 33 / 0 | 43 / 2 |
| 2.55.16.97039 | 17 / 0 | 33 / 0 | 41 / 0 |

**Verdict:** the never-fielded comp set is essentially static across builds — per-build observed-set churn is volume-driven, and once ~1 year of history has accumulated, paper-1's synthetic-augmentation target set computed per era differs from the end-of-corpus target by at most 24 comps per tier (0-2 at the D2 cutoff). The augmentation design is drift-robust; the first-era gap in the table is data accumulation, not meta drift. (Expected near-null, confirmed.)
