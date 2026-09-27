# Q7 — Decayed feature aggregates + shrinkage 2x2

Stats-side recency decay: the enriched-feature AGGREGATES are exponentially
decayed per build (weight 0.5^(age_days/HL), age measured between builds'
max_dates), refreshed at every build under the strictly-causal `_prev`
convention of d2c_cumprev (a row in build N sees stats decayed through build
N-1; build 0 gets empty stats). This is the instrument PARTIAL_REFRESH.md
called for: cumulative aggregates were too dilute to transmit 3-month drift
(mean feature |Δ| 0.054 between cutoff-frozen and refreshed); the decayed
aggregates actually move the model inputs (mean |Δ| vs cumprev on a 1/97 row
sample: decayed365 0.21, decayed90+k100 0.45, decayed90 0.58).

The 2x2 = {cumulative, decayed} x {unshrunk, k=100 count-shrinkage}: the
shrunk cell blends each decayed statistic toward the same-build cumulative
value with k=100 pseudo-games (W2a formula, wr = (g_dec*wr_dec + k*wr_cum) /
(g_dec + k)), availability thresholds gated by CUMULATIVE counts.
(cumulative+k100 collapses to cumulative — shrinking a distribution toward
itself is the identity — so the 2x2 has 3 trained cells + the existing
champion.) Scripts: `drift2026/build_decayed_stats.py` (stats),
`build_drift_features.py --only decayed` (passes `decayed90_prev`,
`decayed365_prev`, `decayed90k100_prev`), `q7_chain.sh` (training; D2
protocol, regime=all, seeds 42/123/777). Raw per-run JSONs:
`results/q7/`; aggregates: `decayed_aggregates.json`.

Future = all 7 builds after 2.55.14.95918 (n_test 287,362 rows); sizable =
the 4 headline test builds; val = train-period held-out slice; n_train =
3,538,596 rows for every cell. Directly comparable to D2_SUMMARY.md.

| cell | val acc | future acc | sizable acc | 15.96370 | 15.96477 | 16.96881 | 16.97039 | sanity 28 (21) | epochs |
|---|---|---|---|---|---|---|---|---|---|
| cumulative (d2c_cumprev, existing champion) | 57.16 ± 0.02 | 57.08 ± 0.03 | 57.09 ± 0.04 | 56.71 | 57.21 | 57.11 | 56.78 | 24.0 (18.7) | 58 |
| decayed HL=365d | 57.17 ± 0.04 | 57.07 ± 0.04 | 57.08 ± 0.04 | 56.48 | 57.26 | 57.00 | 56.80 | 23.3 (18.3) | 60 |
| decayed HL=90d | 57.26 ± 0.07 | 57.19 ± 0.05 | 57.19 ± 0.04 | 56.29 | 57.40 | 57.19 | 56.98 | 25.7 (19.3) | 58 |
| decayed HL=90d + k=100 shrink to cumulative | 57.24 ± 0.05 | 57.22 ± 0.06 | 57.23 ± 0.06 | 56.54 | 57.42 | 57.29 | 56.72 | 25.3 (19.0) | 63 |

Per-seed future acc (seeds 42/123/777):
- d2c_cumprev: 57.086 / 57.114 / 57.044
- q7_decayed365: 57.115 / 57.022 / 57.069
- q7_decayed90: 57.254 / 57.154 / 57.159
- q7_decayed90k100: 57.198 / 57.160 / 57.303

## Verdict — decayed aggregates beat the champion

1. **Both HL=90 cells beat d2c_cumprev's 57.08, and the separation is
   seed-clean**: every decayed90 / decayed90k100 seed (min 57.154) exceeds
   every cumprev seed (max 57.114). decayed90 +0.11 pp, decayed90k100
   +0.14 pp — the new best deployable future accuracy in the suite
   (previous ranking: cumprev 57.08 > embed 56.55 > ...). The shrinkage
   corner is the top cell: recency-decayed values where data is dense,
   pulled toward the long-run mean where the decayed effective sample is
   thin, at cumulative availability.
2. **The gain is a TRAINING-time effect at test-time too**: these cells
   change the per-row feature distribution the model TRAINS under (each row
   sees era-local decayed stats), consistent with PARTIAL_REFRESH.md's
   finding that per-era training stats, not the post-cutoff swap, carry the
   cumprev gain. The decay sharpens exactly that mechanism.
3. **HL=365 is a statistical tie with cumulative** (57.07 vs 57.08): a
   4-year corpus decayed with a 1-year half-life is still ~too dilute; the
   drift-relevant horizon is months, matching the D2a decay curves and the
   d2b sample-weight decay ordering (decay90 56.48 > decay365 56.36 there).
4. Sample-weight decay (d2b_decay90 56.48) vs stats-decay (57.19) is not an
   apples-to-apples pair — d2b used cutoff-frozen features — but the 2x2
   locates the recency gain firmly on the FEATURE side.

## Ordering / pre-registered checks

- decayed365 (57.07) sits between decayed90 (57.19) and cumulative (57.08)
  only up to noise — it lands 0.01 below cumulative (|Δ| << seed sd 0.03-
  0.05). Reading: HL=365 barely changes the features (mean |Δ| 0.21) and
  buys nothing; monotonicity in decay strength holds in the direction that
  matters (90d > both).
- **No effective-sample-size collapse** (the pre-registered failure mode for
  decayed90 < 56): decayed effective games at the cutoff build are mid-tier
  ~144K (vs 872K cumulative); all 270 hero rows (90 heroes x 3 tiers) clear
  the >= 20-game storage threshold at EVERY build, pairs-with->=30-games dip
  at most ~8% below cumulative (46,226 vs 48,024 at the cutoff; worst build
  42,784 vs 47,678). The corpus is dense enough for a 90d half-life.
- **HL=inf convergence control passes exactly**: `build_decayed_stats.py
  --check` (all weights 1) reproduces the cumulative stats files bit-for-bit
  on 45/45 builds (games int-vs-float formatting aside).
- Feature caches are row-aligned with `features_cumulative_prev.npz`
  (identical replay_ids/build_idx, 3,898,174 rows, 0 failed replays).

## Notes / anomalies

- **The first test build 2.55.15.96370 regresses under decay** (56.29
  decayed90 / 56.54 k100 vs 56.71 cumprev) while the other six gain. Its
  feature stats come from the cutoff build itself (prev-build convention),
  so this is not a post-cutoff refresh artifact; plausibly the 90d-decayed
  cutoff stats are a worse description of the FIRST post-patch weeks
  (2.55.15 changed the meta; recency-sharpened pre-patch stats mis-describe
  it more confidently than the dilute cumulative pool). The k=100 shrink
  recovers about half the regression, consistent with that reading.
- The tiny build 2.55.15.96443 (1,712 rows) swings hardest (+1.6 pp for
  decayed90 over cumprev: 56.60 vs 55.00) — low-n noise, not evidence.
- Sanity suite: decayed90 25.7/28 and k100 25.3/28 vs cumprev 24.0/28;
  decayed365 23.3/28. No degradation in the winning cells.
- Training cost is unchanged (~1 min/run, 58-63 epochs).
