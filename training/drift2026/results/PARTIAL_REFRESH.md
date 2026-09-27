# Q5 — Partial-refresh ablation (inference-time stats swap, no retraining)

The three d2c_cumprev value-function checkpoints (seeds 42/123/777) evaluated
on the 7 strictly-future test builds, with the aggregate statistics feeding
the enriched features swapped per SIGNAL CLASS at inference time. Signal
split follows the W3(d) hybrid machinery: hero-WR family =
{hero WR, pick/ban rate, hero-map WR}; pair/comp = {counter WR, synergy WR,
composition WR}. "Refreshed" = cumulative through the row's previous build
(the cumprev convention); "frozen" = cumulative through the training cutoff
2.55.14.95918. Script: `drift2026/phase_q5_partial_refresh.py`; raw numbers:
`partial_refresh.json`.

| cell | hero-WR stats | pair/comp stats | future acc | sizable acc | Δ vs none (paired) | sanity 28 |
|---|---|---|---|---|---|---|
| all_refresh (pos. control) | refreshed | refreshed | 57.081 ± 0.035 | 57.093 ± 0.047 | +0.030 | 24.0 ± 0.0 |
| none_refresh (neg. control) | frozen | frozen | 57.052 ± 0.034 | 57.064 ± 0.041 | — | 24.0 ± 0.0 |
| hero_only | refreshed | frozen | 56.980 ± 0.038 | 56.990 ± 0.042 | **−0.071** | 24.0 ± 1.0 |
| pair_only | frozen | refreshed | **57.121 ± 0.038** | 57.136 ± 0.047 | **+0.069** | 24.3 ± 0.6 |

Per-seed future acc: all 57.086/57.114/57.044 · none 57.038/57.090/57.027 ·
hero_only 56.969/57.023/56.949 · pair_only 57.132/57.152/57.079.
Paired deltas (same test rows, same checkpoint, only the stats source
differs) are same-sign across all 3 seeds for every comparison:
all−none {+0.048, +0.024, +0.017}; hero_only−none {−0.069, −0.067, −0.078};
pair_only−none {+0.094, +0.062, +0.052}; pair_only−all {+0.046, +0.038,
+0.035}.

## Controls

- **Positive control reproduces bit-exactly.** Recomputed-from-stats-files
  all_refresh matches the cached `features_cumulative_prev.npz` evaluation of
  the same checkpoints to 0.000 pp on every seed (57.081 ± 0.035 = published
  d2c_cumprev).
- **Negative control does NOT land at the pre-registered ~56.2 — and that
  expectation was a category error, not a plumbing bug.** 56.21 is the
  accuracy of the d2b_allhist-TRAINED checkpoints. Verified through the same
  eval path: d2b_allhist_s{42,123,777} on the identical cutoff-frozen test
  features give 56.178/56.173/56.274 (mean 56.208 = published 56.21 ± 0.06).
  The cumprev-trained checkpoints fed the very same cutoff-frozen features
  lose only 0.030 pp (57.052). Plumbing further confirmed by two independent
  feature paths agreeing to 0.000 pp on both controls, and by the first test
  build (2.55.15.96370) coming out identical across all 4 cells by
  construction (its previous build IS the cutoff build).

## Verdict — which signal carries it?

1. **The hypothesis is rejected: hero-WR-only refresh does not capture the
   refresh gain — it consistently HURTS** (−0.071 pp vs fully frozen,
   same sign in 3/3 seeds). The pairwise/composition class carries all of
   the (small) inference-time refresh benefit (+0.069 pp), and pair_only
   even edges out refreshing everything (+0.040 pp paired): the hero-WR
   refresh component is what drags all_refresh below pair_only.
2. **The headline reinterpretation: the cumprev arm's +0.87 pp over
   d2b_allhist is a TRAINING-time effect, not a test-time refresh effect.**
   Swapping test-time stats on a fixed cumprev checkpoint moves accuracy by
   at most ±0.07 pp; training with era-appropriate per-row aggregates (vs a
   single cutoff snapshot for all rows) accounts for essentially the entire
   gap. The D2 reading-note framing ("refreshing the feature aggregates
   ... recovers ~2.5x more future accuracy ... without retraining") needs
   this qualification: the checkpoint must have been TRAINED under
   per-era stats for refresh to matter at all, and even then the post-cutoff
   refresh itself is worth only ~0.03 pp at this 3-month horizon.
3. **Mechanism: cumulative aggregates are too dilute to transmit drift at
   this horizon.** On the enriched model-input columns, cutoff-frozen vs
   refreshed features differ by mean |Δ| = 0.054 (median 0.016, p99 0.47) on
   a ~50-scale — 30% of entries are bit-identical — because appending ≤3
   months of games to a 4-year cumulative pool barely moves any estimate.
   This does not contradict the D2a decay curves (hero WR *is* the
   fastest-decaying signal); it shows CUMULATIVE refresh cannot chase that
   decay. The decayed/windowed-aggregate arm (Q7) is the right instrument
   for the refresh-cadence question.
4. Why the sign flip between classes? A candidate mechanism (untested here):
   counter/synergy features are normalized pairwise WRs — pairwise WR minus
   an expectation built from hero WRs. hero_only updates the normalizer but
   not the numerator inside every counter/synergy feature (vintage-mismatched
   in the uninformative direction), while pair_only updates the numerator —
   the signal-bearing part — against the frozen baseline the checkpoint was
   calibrated to at the cutoff. Effects are ≤0.1 pp throughout; treat the
   class-level ordering (pair ≥ none ≥ hero at test time) as the robust
   takeaway, not the magnitudes.

## Notes / anomalies

- Sanity suite is flat across cells (24/28 ± ≤1), as expected for a
  fixed checkpoint under near-identical deployment stats.
- All accuracy deltas are far smaller than the seed sd; they are resolvable
  only because the design is fully paired (identical test rows and weights
  per comparison).
- 3 seeds = 3 training runs of the same arm; the swap deltas are
  deterministic per checkpoint.
