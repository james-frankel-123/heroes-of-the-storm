# W3(c) — Never-fielded composition set drift

Universe = 252 sorted 5-role tuples; observed at >= 5 games/build (pooled tiers), 28 sizable builds.

Consecutive-build Jaccard of observed comp sets: mean 0.806 (min 0.639); never-fielded set per build has 105-164 comps vs 92 never fielded in the whole corpus — per-build variation is volume-driven (rare comps drop below the observation threshold in smaller builds), not meta-driven.

Per-era tier-2 augmentation targets (cumulative-history never-seen sets per tier) vs the full-corpus target:

| era (through build) | low: target size / sym-diff vs full | mid: target size / sym-diff vs full | high: target size / sym-diff vs full |
|---|---|---|---|
| 2.55.0.86938 | 97 / 76 | 105 / 81 | 121 / 81 |
| 2.55.8.93382 | 31 / 10 | 37 / 13 | 51 / 11 |
| 2.55.14.95918 | 22 / 1 | 24 / 0 | 42 / 2 |
| 2.55.16.97039 | 21 / 0 | 24 / 0 | 40 / 0 |

**Verdict:** the never-fielded comp set is essentially static across builds — per-build observed-set churn is volume-driven, and once ~1 year of history has accumulated, paper-1's synthetic-augmentation target set computed per era differs from the end-of-corpus target by at most 13 comps per tier (0-2 at the D2 cutoff). The augmentation design is drift-robust; the first-era gap in the table is data accumulation, not meta drift. (Expected near-null, confirmed.)
