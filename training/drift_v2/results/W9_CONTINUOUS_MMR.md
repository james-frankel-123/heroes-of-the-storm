# W9 — continuous-MMR input ablation

Champion feature pass `decayed90k100_prev`, D2 protocol (cutoff 2.55.14.95918), 3 seeds [42, 123, 777]. Cell (a) quotes the existing champion runs (results/q7); cells (b)/(c) retrain with a continuous avg_mmr input (z-scored on the train period only). Stats aggregates remain per-tier in all cells; only the model's skill input changes.

## Coverage (avg_mmr non-null)

Overall: 1,943,338/1,949,087 = 99.70%. Zero-valued avg_mmr rows (treated as missing): 0.

| slice | n | coverage |
|---|---|---|
| tier low | 380,100 | 100.00% |
| tier mid | 943,322 | 99.39% |
| tier high | 625,665 | 100.00% |
| 2021 | 33,762 | 100.00% |
| 2022 | 424,390 | 99.99% |
| 2023 | 320,643 | 99.91% |
| 2024 | 342,711 | 99.90% |
| 2025 | 600,869 | 99.83% |
| 2026 | 226,712 | 98.23% |
| train (<= cutoff) | 1,805,406 | 99.89% |
| test (> cutoff) | 143,681 | 97.45% |

Missingness handling: mean-impute (z=0); coverage >= 90% so no indicator column was added. Train-period z-score: mu=2633.49, sd=202.52.

## Results (mean +/- sd over 3 seeds; paired deltas vs cell a)

| cell | input | dim | val acc | future acc | sizable acc | d future (paired) | sanity/28 |
|---|---|---|---|---|---|---|---|
| (a) | tier one-hot (283d, champion, quoted) | 283 | 57.002 ± 0.083 | 57.226 ± 0.024 | 57.233 ± 0.029 | — | 21/25/24 |
| (b) | tier one-hot + avg_mmr | 284 | 57.167 ± 0.015 | 57.247 ± 0.021 | 57.257 ± 0.018 | +0.021 [0.019, -0.018, 0.061] | 25/23/24 |
| (c) | avg_mmr replaces tier one-hot | 281 | 57.178 ± 0.138 | 57.218 ± 0.008 | 57.224 ± 0.009 | -0.008 [0.003, -0.031, 0.005] | 26/23/24 |

## MMR-axis probes

Fixed comps on Cursed Hollow (probe hero + ['Muradin', 'Brightwing', 'Sonya', 'Jaina'] vs ['Diablo', 'Malfurion', 'Falstad', 'Zeratul', 'Li-Ming']); stats lookups fixed at tier=mid so only the model's skill input moves. Values = predicted WP(team0). Grid = train-period avg_mmr quantiles {'p5': 2308.8, 'p25': 2485.6, 'p50': 2633.6, 'p75': 2775.9, 'p95': 2966.9}.

### cell (a) — tier one-hot toggled (stats fixed mid)

| hero | low | mid | high | Δ(last−first) |
|---|---|---|---|---|
| Murky | 0.5075 | 0.5122 | 0.5141 | +0.0066 |
| Abathur | 0.5326 | 0.5410 | 0.5374 | +0.0048 |
| The Lost Vikings | 0.5180 | 0.5313 | 0.5294 | +0.0114 |
| Raynor | 0.5601 | 0.5700 | 0.5498 | -0.0103 |

### cell (b) — tier_plus_mmr: MMR input swept p5..p95 (tier one-hot fixed mid)

| hero | p5 | p25 | p50 | p75 | p95 | Δ(last−first) |
|---|---|---|---|---|---|---|
| Murky | 0.5239 | 0.5155 | 0.5035 | 0.4912 | 0.4760 | -0.0479 |
| Abathur | 0.5558 | 0.5557 | 0.5478 | 0.5328 | 0.5159 | -0.0399 |
| The Lost Vikings | 0.5231 | 0.5265 | 0.5265 | 0.5242 | 0.5195 | -0.0037 |
| Raynor | 0.5709 | 0.5601 | 0.5489 | 0.5349 | 0.5150 | -0.0559 |

### cell (c) — mmr_only: MMR input swept p5..p95 (no tier one-hot)

| hero | p5 | p25 | p50 | p75 | p95 | Δ(last−first) |
|---|---|---|---|---|---|---|
| Murky | 0.5256 | 0.5179 | 0.5096 | 0.5029 | 0.4951 | -0.0305 |
| Abathur | 0.5481 | 0.5503 | 0.5466 | 0.5405 | 0.5306 | -0.0175 |
| The Lost Vikings | 0.5311 | 0.5334 | 0.5342 | 0.5301 | 0.5251 | -0.0060 |
| Raynor | 0.5606 | 0.5520 | 0.5406 | 0.5290 | 0.5199 | -0.0407 |

### Probe reading

All heroes' team0 WP declines as the MMR input rises in (b)/(c) — a global compression of draft-based predictions toward 50% at high MMR (the probe comp favors team0, so shrinking confidence lowers its WP). The hero-specific skill effect is therefore the slope RELATIVE to the skill-neutral Raynor comp:

| cell | axis | Murky | Abathur | The Lost Vikings |
|---|---|---|---|---|
| (a) | low -> high | +0.0169 | +0.0151 | +0.0217 |
| (b) | p5 -> p95 | +0.0080 | +0.0160 | +0.0522 |
| (c) | p5 -> p95 | +0.0102 | +0.0232 | +0.0347 |

The Lost Vikings come out RELATIVELY stronger at high MMR in every cell (consistent with their high skill floor, though the community-stats framing in the work queue expected the opposite direction); Murky and Abathur are near-neutral relative to Raynor with inconsistent signs across cells. The models do not express a strong low-MMR advantage for the cheese heroes — reported as observed, not forced.

## Verdict

Continuous avg_mmr adds nothing measurable over the 3-tier one-hot: appending it (b) moves future-window accuracy by +0.021pp and substituting it for the tier one-hot (c) by -0.008pp, both well inside the +/-0.06pp seed noise of cell (a). The tier grouping is a sufficient skill input for the WP model (and, conversely, a single continuous dim loses nothing vs the 3 tier dims). Robustness row only; the per-tier stats machinery stands.

## Caveats

- avg_mmr is Heroes-Profile-computed near parse time (a post-hoc match-average), cleaner than per-player as-of-parse MMRs but not a strictly pre-game quantity.
- The tier one-hot in cells (a)/(b) and the per-tier stats lookups are derived from the same underlying MMR, so (b) measures the *marginal* value of the continuous signal on top of the tier structure, not a from-scratch comparison.
- Stats aggregates stay per-tier in every cell (kernel-weighted continuous-MMR *stats* would be real machinery, out of scope).

_Generated by drift2026/w9_continuous_mmr.py._
