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
| (a) | tier one-hot (283d, champion, quoted) | 283 | 57.241 ± 0.046 | 57.220 ± 0.060 | 57.235 ± 0.062 | — | 26/25/25 |
| (b) | tier one-hot + avg_mmr | 284 | 57.342 ± 0.061 | 57.227 ± 0.028 | 57.232 ± 0.024 | +0.007 [-0.007, 0.072, -0.044] | 24/23/25 |
| (c) | avg_mmr replaces tier one-hot | 281 | 57.276 ± 0.077 | 57.204 ± 0.027 | 57.207 ± 0.035 | -0.017 [0.039, 0.012, -0.101] | 25/26/26 |

## MMR-axis probes

Fixed comps on Cursed Hollow (probe hero + ['Muradin', 'Brightwing', 'Sonya', 'Jaina'] vs ['Diablo', 'Malfurion', 'Falstad', 'Zeratul', 'Li-Ming']); stats lookups fixed at tier=mid so only the model's skill input moves. Values = predicted WP(team0). Grid = train-period avg_mmr quantiles {'p5': 2308.8, 'p25': 2485.6, 'p50': 2633.6, 'p75': 2775.9, 'p95': 2966.9}.

### cell (a) — tier one-hot toggled (stats fixed mid)

| hero | low | mid | high | Δ(last−first) |
|---|---|---|---|---|
| Murky | 0.5144 | 0.5117 | 0.5202 | +0.0058 |
| Abathur | 0.5131 | 0.5167 | 0.5194 | +0.0063 |
| The Lost Vikings | 0.5266 | 0.5286 | 0.5444 | +0.0178 |
| Raynor | 0.5555 | 0.5599 | 0.5636 | +0.0081 |

### cell (b) — tier_plus_mmr: MMR input swept p5..p95 (tier one-hot fixed mid)

| hero | p5 | p25 | p50 | p75 | p95 | Δ(last−first) |
|---|---|---|---|---|---|---|
| Murky | 0.5486 | 0.5398 | 0.5261 | 0.5076 | 0.4809 | -0.0677 |
| Abathur | 0.5452 | 0.5399 | 0.5296 | 0.5165 | 0.4974 | -0.0478 |
| The Lost Vikings | 0.5661 | 0.5746 | 0.5689 | 0.5528 | 0.5240 | -0.0421 |
| Raynor | 0.5750 | 0.5724 | 0.5564 | 0.5343 | 0.5023 | -0.0726 |

### cell (c) — mmr_only: MMR input swept p5..p95 (no tier one-hot)

| hero | p5 | p25 | p50 | p75 | p95 | Δ(last−first) |
|---|---|---|---|---|---|---|
| Murky | 0.5280 | 0.5247 | 0.5180 | 0.5025 | 0.4732 | -0.0548 |
| Abathur | 0.5330 | 0.5335 | 0.5299 | 0.5193 | 0.5031 | -0.0299 |
| The Lost Vikings | 0.5360 | 0.5403 | 0.5418 | 0.5375 | 0.5203 | -0.0157 |
| Raynor | 0.5636 | 0.5660 | 0.5635 | 0.5500 | 0.5190 | -0.0446 |

### Probe reading

All heroes' team0 WP declines as the MMR input rises in (b)/(c) — a global compression of draft-based predictions toward 50% at high MMR (the probe comp favors team0, so shrinking confidence lowers its WP). The hero-specific skill effect is therefore the slope RELATIVE to the skill-neutral Raynor comp:

| cell | axis | Murky | Abathur | The Lost Vikings |
|---|---|---|---|---|
| (a) | low -> high | -0.0023 | -0.0018 | +0.0097 |
| (b) | p5 -> p95 | +0.0049 | +0.0248 | +0.0305 |
| (c) | p5 -> p95 | -0.0102 | +0.0147 | +0.0289 |

The Lost Vikings come out RELATIVELY stronger at high MMR in every cell (consistent with their high skill floor, though the community-stats framing in the work queue expected the opposite direction); Murky and Abathur are near-neutral relative to Raynor with inconsistent signs across cells. The models do not express a strong low-MMR advantage for the cheese heroes — reported as observed, not forced.

## Verdict

Continuous avg_mmr adds nothing measurable over the 3-tier one-hot: appending it (b) moves future-window accuracy by +0.007pp and substituting it for the tier one-hot (c) by -0.017pp, both well inside the +/-0.06pp seed noise of cell (a). The tier grouping is a sufficient skill input for the WP model (and, conversely, a single continuous dim loses nothing vs the 3 tier dims). Robustness row only; the per-tier stats machinery stands.

## Caveats

- avg_mmr is Heroes-Profile-computed near parse time (a post-hoc match-average), cleaner than per-player as-of-parse MMRs but not a strictly pre-game quantity.
- The tier one-hot in cells (a)/(b) and the per-tier stats lookups are derived from the same underlying MMR, so (b) measures the *marginal* value of the continuous signal on top of the tier structure, not a from-scratch comparison.
- Stats aggregates stay per-tier in every cell (kernel-weighted continuous-MMR *stats* would be real machinery, out of scope).

_Generated by drift2026/w9_continuous_mmr.py._
