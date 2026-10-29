# P3 definitive rerun, build def1027

Every input was built off the 3090: exports and the merged store on the main box, the A1 cache, merged rows, scoreboard card, model caches and walks on the 3080, fits on the 5090. The pinned manifest (`results/def_def1027/manifest.json`) lists 113 files with sha256 (42 personalization, 11 upstream, 57 tournament).

## Forward games (log-loss gain over population WP, x1000; paired against the previous main model)

| model | V1 | V2 | V2 vs previous (SE) | OOT | OOT vs previous (SE) | acc V2 | acc OOT |
|---|---|---|---|---|---|---|---|
| today's kernel (A2S-equivalent) | +18.34 | +19.22 | -9.99 (0.50) | +18.49 | -9.82 (0.25) | 60.68% | 60.33% |
| previous main model (full_ms + layer) | +26.69 | +29.22 | +0.00 (0.00) | +28.31 | +0.00 (0.00) | 61.85% | 61.70% |
| **chosen shrinkage model** | +35.80 | +36.53 | +7.31 (0.47) | +37.38 | +9.07 (0.24) | 63.12% | 62.85% |
| chosen model, judge half (full_ms_flip) | +34.52 | +34.95 | +5.74 (0.53) | +36.29 | +7.97 (0.27) | 63.05% | 62.81% |

## Primary contrasts with week-block 95% CIs (log-loss gain x1000; game SE in parentheses)

V2 and OOT were both read during model development (shrinkage study, team shares, role weights), so neither is an untouched holdout. OOT also contains the formerly sealed build 2.55.17.98025 (36,447 games), shown separately below; the 5,000 sealed replays were never read. The previous main model is the pre-stated comparator.

| set | games | chosen − previous | chosen − today |
|---|---|---|---|
| V2 all | 72,474 | +7.31 (0.47) [+6.58, +8.11] | +17.30 (0.69) [+15.91, +18.73] |
| OOT all | 286,016 | +9.07 (0.24) [+8.64, +9.57] | +18.89 (0.35) [+18.25, +19.64] |
| OOT before 2026-09-12 | 247,488 | +9.10 (0.26) [+8.59, +9.60] | +18.91 (0.38) [+18.18, +19.57] |
| OOT from 2026-09-12 | 38,528 | +8.88 (0.64) [+7.80, +9.59] | +18.78 (0.98) [+16.20, +20.76] |
| OOT without 2.55.17.98025 | 249,569 | +9.10 (0.26) [+8.61, +9.60] | +18.92 (0.37) [+18.22, +19.57] |
| OOT 2.55.17.98025 only | 36,447 | +8.84 (0.66) [+7.42, +9.59] | +18.71 (1.01) [+15.39, +20.76] |

## Subsets (against today's kernel; gain x1000, SE, week-block 95% CI)

| set | games | today | previous | chosen | chosen − today |
|---|---|---|---|---|---|
| V2 all | 72,474 | +19.22 | +29.22 | +36.53 | +17.30 (0.69) [+15.91, +18.73] |
| V2 all-10 pulled | 40,348 | +22.07 | +30.46 | +37.90 | +15.82 (0.86) [+14.08, +17.62] |
| V2 1+ proven | 10,123 | +33.46 | +40.53 | +48.02 | +14.56 (1.70) [+11.00, +18.85] |
| V2 3+ proven | 367 | +53.16 | +56.48 | +60.80 | +7.64 (8.46) [-11.95, +25.97] |
| V2 deep (all 10 with 500+ games) | 667 | +20.37 | +28.53 | +40.55 | +20.18 (5.23) [+12.86, +27.96] |
| V2 Diamond | 9,188 | +28.26 | +35.38 | +44.71 | +16.44 (1.68) [+11.77, +21.01] |
| V2 Master | 6,477 | +22.83 | +26.45 | +34.37 | +11.54 (2.13) [+5.75, +17.48] |
| V2 Diamond+Master, all-10 pulled | 11,749 | +26.63 | +32.50 | +41.43 | +14.81 (1.49) [+11.83, +18.39] |
| OOT all | 286,016 | +18.49 | +28.31 | +37.38 | +18.89 (0.35) [+18.25, +19.64] |
| OOT all-10 pulled | 145,922 | +21.61 | +30.48 | +39.77 | +18.15 (0.45) [+17.20, +19.22] |
| OOT 1+ proven | 36,255 | +30.63 | +36.85 | +47.59 | +16.95 (0.91) [+14.66, +18.71] |
| OOT 3+ proven | 1,197 | +22.56 | +18.31 | +31.49 | +8.92 (4.70) [+0.17, +17.90] |
| OOT deep (all 10 with 500+ games) | 1,987 | +15.93 | +17.35 | +27.10 | +11.17 (3.15) [+6.25, +16.77] |
| OOT Diamond | 33,192 | +24.33 | +32.39 | +43.02 | +18.70 (0.88) [+17.27, +20.12] |
| OOT Master | 29,843 | +22.27 | +29.41 | +39.17 | +16.90 (1.02) [+14.38, +18.84] |
| OOT Diamond+Master, all-10 pulled | 44,651 | +24.82 | +31.77 | +41.85 | +17.03 (0.77) [+15.59, +18.18] |
| OOT before 2026-09-12 | 247,488 | +18.55 | +28.36 | +37.46 | +18.91 (0.38) [+18.18, +19.57] |
| OOT from 2026-09-12 | 38,528 | +18.09 | +27.99 | +36.87 | +18.78 (0.98) [+16.20, +20.76] |
| OOT from 2026-09-12, without 2.55.17.98025 | 2,081 | +15.53 | +25.88 | +35.43 | +19.90 (4.06) [+19.90, +19.90] |
| OOT without 2.55.17.98025 | 249,569 | +18.53 | +28.34 | +37.44 | +18.92 (0.37) [+18.22, +19.57] |
| OOT 2.55.17.98025 only | 36,447 | +18.24 | +28.11 | +36.96 | +18.71 (1.01) [+15.39, +20.76] |

## Disclosures

- The drafter fit (full_ms, chosen) and the judge fit (full_ms_flip, chosen_flip) use disjoint player halves but the same V1 games: the shrinkage function, direct channel, role weights, calibration and layer of each arm are fit on V1 slot outcomes, and the two halves share those games' outcomes (teammates and opponents). Their coefficients are fit separately. A game-disjoint variant is a planned sensitivity after this run.
- Filtered states (the judge window, split halves, exclusions) carry no temporal layer, whose residuals come from unfiltered predictions; a poison check of every filter form gates each answer run.
- The tournament reads the base posterior sd (`sd`); the display band (`sd_display`) is for the site only.


## Calibration of the chosen model (slope of realized on predicted, V2 + OOT, smooth curve)

| axis | point | previous main | chosen |
|---|---|---|---|
| games | 0 | 1.31 | 1.05 |
| games | 3 | 1.16 | 0.99 |
| games | 10 | 1.11 | 0.99 |
| games | 30 | 0.99 | 0.95 |
| games | 100 | 0.88 | 1.00 |
| games | 300 | 0.76 | 0.89 |
| games | 1000 | 0.80 | 0.84 |
| share | 0.05 | 0.95 | 0.98 |
| share | 0.2 | 0.93 | 0.92 |
| share | 0.4 | 0.92 | 0.90 |
| share | 0.6 | 0.90 | 0.96 |
| share | 0.8 | 1.12 | 1.10 |
| share | 0.95 | 1.02 | 0.95 |
| recency | 0 | 1.04 | 0.99 |
| recency | 10 | 0.99 | 0.94 |
| recency | 30 | 0.97 | 0.94 |
| recency | 100 | 0.98 | 0.98 |
| recency | 365 | 0.91 | 1.01 |
| recency | 1000 | 0.91 | 0.95 |

RMS of (slope − 1), slot-weighted: previous games 0.153, share 0.059, recency 0.038; chosen games 0.057, share 0.047, recency 0.045.

## Display band (D = s2 (1 + exp(-games / 10)))

| window | cells | realized / claimed (95% CI) | 80% coverage |
|---|---|---|---|
| V1 | 8,161 | 0.83 (0.22–1.43) | 0.785 |
| V2 | 8,678 | 0.70 (0.22–1.25) | 0.792 |
| OOT | 51,890 | 0.26 (0.07–0.44) | 0.799 |

Median display sd by games on the hero: 0-0: 5.8pp, 1-9: 4.4pp, 10-29: 3.4pp, 30-99: 2.8pp, 100-299: 2.5pp, 300-1000000000: 2.1pp.

