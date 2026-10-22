# P3 definitive rerun, build dry1022

Every input was built off the 3090: exports and the merged store on the main box, the A1 cache, merged rows, scoreboard card, model caches and walks on the 3080, fits on the 5090. The pinned manifest (`results/def_dry1022/manifest.json`) lists 107 files with sha256 (39 personalization, 11 upstream, 57 tournament).

## Forward games (log-loss gain over population WP, x1000; paired against the previous main model)

| model | V1 | V2 | V2 vs previous (SE) | OOT | OOT vs previous (SE) | acc V2 | acc OOT |
|---|---|---|---|---|---|---|---|
| today's kernel (A2S-equivalent) | +17.73 | +18.56 | -10.59 (0.50) | +18.39 | -10.07 (0.25) | 60.61% | 60.32% |
| previous main model (full_ms + layer) | +26.43 | +29.15 | +0.00 (0.00) | +28.45 | +0.00 (0.00) | 61.91% | 61.69% |
| **chosen shrinkage model** | +33.38 | +35.61 | +6.46 (0.44) | +36.58 | +8.12 (0.23) | 62.82% | 62.80% |
| chosen model, judge half (full_ms_flip) | +32.78 | +34.32 | +5.17 (0.50) | +35.81 | +7.36 (0.25) | 62.80% | 62.74% |

## Subsets (against today's kernel; gain x1000, SE)

| set | games | today | previous | chosen | chosen − today (SE) |
|---|---|---|---|---|---|
| V2 all | 72,474 | +18.56 | +29.15 | +35.61 | +17.05 (0.67) |
| V2 all-10 pulled | 17,488 | +23.22 | +30.41 | +36.93 | +13.71 (1.20) |
| V2 1+ proven | 9,820 | +33.08 | +40.57 | +47.19 | +14.11 (1.71) |
| V2 3+ proven | 349 | +54.56 | +58.03 | +56.14 | +1.57 (8.37) |
| V2 deep (all 10 with 500+ games) | 601 | +21.95 | +29.85 | +41.49 | +19.54 (5.34) |
| V2 Diamond | 9,188 | +27.51 | +35.22 | +44.02 | +16.51 (1.66) |
| V2 Master | 6,477 | +22.36 | +26.56 | +33.59 | +11.24 (2.09) |
| V2 Diamond+Master, all-10 pulled | 7,272 | +27.09 | +30.25 | +38.98 | +11.89 (1.79) |
| OOT all | 286,016 | +18.39 | +28.45 | +36.58 | +18.19 (0.34) |
| OOT all-10 pulled | 117,332 | +21.53 | +30.11 | +38.89 | +17.36 (0.48) |
| OOT 1+ proven | 35,948 | +30.69 | +36.87 | +46.99 | +16.30 (0.90) |
| OOT 3+ proven | 1,171 | +20.82 | +17.20 | +30.72 | +9.90 (4.65) |
| OOT deep (all 10 with 500+ games) | 1,880 | +15.35 | +17.05 | +24.62 | +9.27 (3.24) |
| OOT Diamond | 33,192 | +24.24 | +32.26 | +42.64 | +18.40 (0.87) |
| OOT Master | 29,843 | +22.08 | +29.45 | +37.98 | +15.90 (0.99) |
| OOT Diamond+Master, all-10 pulled | 36,900 | +25.38 | +32.08 | +41.77 | +16.39 (0.82) |

## Calibration of the chosen model (slope of realized on predicted, V2 + OOT, smooth curve)

| axis | point | previous main | chosen |
|---|---|---|---|
| games | 0 | 1.33 | 1.10 |
| games | 3 | 1.18 | 1.03 |
| games | 10 | 1.10 | 1.01 |
| games | 30 | 0.98 | 0.95 |
| games | 100 | 0.88 | 1.00 |
| games | 300 | 0.76 | 0.92 |
| games | 1000 | 0.80 | 0.84 |
| share | 0.05 | 0.95 | 1.00 |
| share | 0.2 | 0.93 | 0.93 |
| share | 0.4 | 0.92 | 0.91 |
| share | 0.6 | 0.89 | 0.93 |
| share | 0.8 | 1.13 | 1.09 |
| share | 0.95 | 0.98 | 0.97 |
| recency | 0 | 1.04 | 0.99 |
| recency | 10 | 0.99 | 0.97 |
| recency | 30 | 0.97 | 0.97 |
| recency | 100 | 0.99 | 1.00 |
| recency | 365 | 0.91 | 1.04 |
| recency | 1000 | 0.92 | 0.94 |

RMS of (slope − 1), slot-weighted: previous games 0.160, share 0.059, recency 0.036; chosen games 0.058, share 0.043, recency 0.028.

## Display band (D = s2 (1 + exp(-games / 10)))

| window | cells | realized / claimed (95% CI) | 80% coverage |
|---|---|---|---|
| V1 | 8,161 | 0.90 (0.28–1.47) | 0.786 |
| V2 | 8,678 | 0.67 (0.20–1.20) | 0.792 |
| OOT | 51,890 | 0.25 (0.07–0.43) | 0.800 |

Median display sd by games on the hero: 0-0: 5.8pp, 1-9: 4.5pp, 10-29: 3.4pp, 30-99: 2.8pp, 100-299: 2.5pp, 300-1000000000: 2.1pp.

