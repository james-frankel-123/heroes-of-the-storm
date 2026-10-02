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
| oracle (retrain every build) | 35 | 35 | 56.866 | +0.000 |
| never retrain + frozen stats | 0 | 0 | 55.168 | +1.698 |
| never retrain + stats refresh | 0 | 35 | 56.204 | +0.662 |
| retrain every K=1 builds + refresh | 35 | 35 | 56.866 | +0.000 |
| retrain every K=3 builds + refresh | 11 | 35 | 56.868 | -0.002 |
| retrain every K=6 builds + refresh | 5 | 35 | 56.785 | +0.081 |
| trigger: acc drop > 0.25pp + refresh | 10 | 35 | 56.808 | +0.058 |
| trigger: acc drop > 0.5pp + refresh | 6 | 35 | 56.774 | +0.092 |
| trigger: acc drop > 1.0pp + refresh | 2 | 35 | 56.618 | +0.248 |
| never refresh (same C0 model, frozen stats) | 0 | 0 | 55.845 | +1.021 |
| detector-triggered refresh (zero latency) | 0 | 14 | 56.192 | +0.674 |
| detector-triggered refresh (first-detection latency) | 0 | 14 | 56.189 | +0.677 |
| detector-triggered refresh (median-detection latency) | 0 | 14 | 56.189 | +0.677 |
| detector-triggered refresh (fixed 9,391-game latency) | 0 | 14 | 56.183 | +0.683 |

## Detector fire schedule (wr_bh survivors, z_naive latency)

| fired build | detections | first-detection latency (games) | median-detection latency |
|---|---|---|---|
| 2.55.5.92264 | 4 | 2,655 | 4,094 |
| 2.55.6.92665 | 2 | 2,055 | 2,541 |
| 2.55.7.93054 | 3 | 958 | 2,762 |
| 2.55.8.93382 | 1 | 3,245 | 3,245 |
| 2.55.10.93810 | 1 | 6,947 | 6,947 |
| 2.55.10.94189 | 5 | 572 | 1,975 |
| 2.55.10.94387 | 1 | 2,281 | 2,281 |
| 2.55.10.94470 | 1 | 448 | 448 |
| 2.55.12.94786 | 5 | 1,642 | 8,159 |
| 2.55.13.95301 | 3 | 2,518 | 8,611 |
| 2.55.14.95817 | 3 | 4,773 | 7,495 |
| 2.55.14.95918 | 3 | 10,198 | 12,410 |
| 2.55.15.96477 | 11 | 3,483 | 21,345 |
| 2.55.16.96881 | 4 | 1,840 | 4,194 |

## Verdict

Detector-triggered refresh (first-detection latency) lands at 56.189%, 0.015 pp below the every-build refresh arm (56.080%) — WITHIN the ~0.1 pp hypothesis band, with only 14 refreshes instead of 35. Latency symmetry: the outcome shift is detectable at a median 9,391 games into a build (patch-note TPs, naive z), essentially the same scale at which the new build's own hero-WR stats become trustworthy to 1 pp (6.5-12K games: per-tier medians 6,460-8,668, pooled median 10,719, p75 12,237 — W3_RECOVERY / W3_HETEROGENEITY). The closed loop detect-then-refresh therefore costs almost nothing relative to refreshing blindly every build.

