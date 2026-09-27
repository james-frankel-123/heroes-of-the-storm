# W2c — detector-triggered stats-refresh arm (Q4)

Extends the W2C_SUMMARY table: the feature aggregates are refreshed
ONLY when the FDR-surviving WR-shift detector (W3b `wr_bh`) fires at
a sizable-build boundary — honestly lagged by the measured z_naive
sequential-detection latency. Games before the refresh point inside
a fired build are scored with the pre-refresh (stale) stats; the
refresh swaps in stats cumulative through the fired build's
predecessor (identical to what the every-build refresh uses there).
Model fixed at the C0 checkpoint (w2c_cut08_s42); no retraining.
14 fires over the 36-build deployment window;
between fires the stats stay stale.

| policy | retrains | refreshes | weighted acc % | regret vs oracle (pp) |
|---|---|---|---|---|
| oracle (retrain every build) | 35 | 35 | 56.811 | +0.000 |
| never retrain + frozen stats | 0 | 0 | 55.128 | +1.682 |
| never retrain + stats refresh | 0 | 35 | 56.080 | +0.730 |
| retrain every K=1 builds + refresh | 35 | 35 | 56.811 | +0.000 |
| retrain every K=3 builds + refresh | 11 | 35 | 56.782 | +0.029 |
| retrain every K=6 builds + refresh | 5 | 35 | 56.743 | +0.067 |
| trigger: acc drop > 0.25pp + refresh | 11 | 35 | 56.755 | +0.056 |
| trigger: acc drop > 0.5pp + refresh | 7 | 35 | 56.733 | +0.078 |
| trigger: acc drop > 1.0pp + refresh | 3 | 35 | 56.503 | +0.308 |
| never refresh (same C0 model, frozen stats) | 0 | 0 | 55.748 | +1.063 |
| detector-triggered refresh (zero latency) | 0 | 14 | 56.069 | +0.742 |
| detector-triggered refresh (first-detection latency) | 0 | 14 | 56.068 | +0.743 |
| detector-triggered refresh (median-detection latency) | 0 | 14 | 56.066 | +0.745 |
| detector-triggered refresh (fixed 9,391-game latency) | 0 | 14 | 56.065 | +0.746 |

## Detector fire schedule (wr_bh survivors, z_naive latency)

| fired build | detections | first-detection latency (games) | median-detection latency |
|---|---|---|---|
| 2.55.5.92264 | 4 | 2,655 | 4,095 |
| 2.55.6.92665 | 2 | 2,055 | 2,541 |
| 2.55.7.93054 | 3 | 960 | 2,771 |
| 2.55.8.93382 | 1 | 3,250 | 3,250 |
| 2.55.10.93810 | 1 | 6,947 | 6,947 |
| 2.55.10.94189 | 5 | 572 | 1,975 |
| 2.55.10.94387 | 1 | 2,286 | 2,286 |
| 2.55.10.94470 | 1 | 448 | 448 |
| 2.55.12.94786 | 5 | 1,647 | 8,167 |
| 2.55.13.95301 | 3 | 2,529 | 8,645 |
| 2.55.14.95817 | 3 | 4,777 | 7,500 |
| 2.55.14.95918 | 3 | 10,214 | 12,436 |
| 2.55.15.96477 | 11 | 3,509 | 21,482 |
| 2.55.16.96881 | 3 | 1,886 | 5,195 |

## Verdict

Detector-triggered refresh (first-detection latency) lands at 56.068%, 0.013 pp below the every-build refresh arm (56.080%) — WITHIN the ~0.1 pp hypothesis band, with only 14 refreshes instead of 35. Latency symmetry: the outcome shift is detectable at a median 9,391 games into a build (patch-note TPs, naive z), essentially the same scale at which the new build's own hero-WR stats become trustworthy to 1 pp (6.5-12K games: per-tier medians 6,460-8,668, pooled median 10,719, p75 12,237 — W3_RECOVERY / W3_HETEROGENEITY). The closed loop detect-then-refresh therefore costs almost nothing relative to refreshing blindly every build.

