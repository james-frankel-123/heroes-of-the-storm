# P3 definitive rerun, build dry1022

Every input was built off the 3090: exports and the merged store on the main box, the A1 cache, merged rows, scoreboard card, model caches and walks on the 3080, fits on the 5090. The pinned manifest (`results/def_dry1022/manifest.json`) lists 112 files with sha256 (42 personalization, 11 upstream, 57 tournament).

## Forward games (log-loss gain over population WP, x1000; paired against the previous main model)

| model | V1 | V2 | V2 vs previous (SE) | OOT | OOT vs previous (SE) | acc V2 | acc OOT |
|---|---|---|---|---|---|---|---|
| today's kernel (A2S-equivalent) | +17.73 | +18.56 | -10.59 (0.50) | +18.39 | -10.07 (0.25) | 60.61% | 60.32% |
| previous main model (full_ms + layer) | +26.43 | +29.15 | +0.00 (0.00) | +28.45 | +0.00 (0.00) | 61.91% | 61.69% |
| **chosen shrinkage model** | +35.02 | +36.34 | +7.19 (0.47) | +37.57 | +9.12 (0.24) | 63.11% | 62.88% |
| chosen model, judge half (full_ms_flip) | +34.55 | +35.16 | +6.01 (0.52) | +36.85 | +8.40 (0.27) | 63.00% | 62.87% |

## Primary contrasts with week-block 95% CIs (log-loss gain x1000; game SE in parentheses)

V2 and OOT were both read during model development (shrinkage study, team shares, role weights), so neither is an untouched holdout. OOT also contains the formerly sealed build 2.55.17.98025 (36,447 games), shown separately below; the 5,000 sealed replays were never read. The previous main model is the pre-stated comparator.

| set | games | chosen − previous | chosen − today |
|---|---|---|---|
| V2 all | 72,474 | +7.19 (0.47) [+6.34, +8.05] | +17.77 (0.68) [+16.65, +19.15] |
| OOT all | 286,016 | +9.12 (0.24) [+8.65, +9.61] | +19.19 (0.35) [+18.57, +19.92] |
| OOT before 2026-09-12 | 247,488 | +9.14 (0.26) [+8.62, +9.66] | +19.17 (0.37) [+18.47, +19.84] |
| OOT from 2026-09-12 | 38,528 | +8.94 (0.64) [+8.03, +9.50] | +19.27 (0.96) [+16.69, +21.05] |
| OOT without 2.55.17.98025 | 249,569 | +9.16 (0.26) [+8.64, +9.68] | +19.19 (0.37) [+18.51, +19.84] |
| OOT 2.55.17.98025 only | 36,447 | +8.85 (0.66) [+7.48, +9.50] | +19.16 (0.99) [+15.68, +21.05] |

## Subsets (against today's kernel; gain x1000, SE, week-block 95% CI)

| set | games | today | previous | chosen | chosen − today |
|---|---|---|---|---|---|
| V2 all | 72,474 | +18.56 | +29.15 | +36.34 | +17.77 (0.68) [+16.65, +19.15] |
| V2 all-10 pulled | 17,488 | +23.22 | +30.41 | +36.78 | +13.57 (1.22) [+11.28, +16.19] |
| V2 1+ proven | 9,820 | +33.08 | +40.57 | +47.76 | +14.68 (1.72) [+10.81, +18.99] |
| V2 3+ proven | 349 | +54.56 | +58.03 | +58.68 | +4.12 (8.69) [-15.14, +22.95] |
| V2 deep (all 10 with 500+ games) | 601 | +21.95 | +29.85 | +40.61 | +18.66 (5.40) [+8.88, +27.31] |
| V2 Diamond | 9,188 | +27.51 | +35.22 | +44.12 | +16.61 (1.69) [+12.56, +20.49] |
| V2 Master | 6,477 | +22.36 | +26.56 | +34.20 | +11.84 (2.13) [+6.13, +17.57] |
| V2 Diamond+Master, all-10 pulled | 7,272 | +27.09 | +30.25 | +38.76 | +11.67 (1.81) [+9.56, +13.74] |
| OOT all | 286,016 | +18.39 | +28.45 | +37.57 | +19.19 (0.35) [+18.57, +19.92] |
| OOT all-10 pulled | 117,332 | +21.53 | +30.11 | +39.68 | +18.15 (0.49) [+17.09, +19.32] |
| OOT 1+ proven | 35,948 | +30.69 | +36.87 | +47.71 | +17.03 (0.91) [+14.88, +18.84] |
| OOT 3+ proven | 1,171 | +20.82 | +17.20 | +29.96 | +9.14 (4.79) [-0.29, +18.21] |
| OOT deep (all 10 with 500+ games) | 1,880 | +15.35 | +17.05 | +25.40 | +10.05 (3.27) [+4.90, +16.02] |
| OOT Diamond | 33,192 | +24.24 | +32.26 | +43.18 | +18.94 (0.88) [+17.57, +20.25] |
| OOT Master | 29,843 | +22.08 | +29.45 | +39.18 | +17.09 (1.01) [+14.59, +19.04] |
| OOT Diamond+Master, all-10 pulled | 36,900 | +25.38 | +32.08 | +42.55 | +17.17 (0.82) [+15.83, +18.36] |
| OOT before 2026-09-12 | 247,488 | +18.43 | +28.46 | +37.60 | +19.17 (0.37) [+18.47, +19.84] |
| OOT from 2026-09-12 | 38,528 | +18.10 | +28.43 | +37.37 | +19.27 (0.96) [+16.69, +21.05] |
| OOT from 2026-09-12, without 2.55.17.98025 | 2,081 | +15.54 | +26.30 | +36.83 | +21.29 (3.97) [+21.29, +21.29] |
| OOT without 2.55.17.98025 | 249,569 | +18.41 | +28.44 | +37.60 | +19.19 (0.37) [+18.51, +19.84] |
| OOT 2.55.17.98025 only | 36,447 | +18.24 | +28.55 | +37.40 | +19.16 (0.99) [+15.68, +21.05] |

## Disclosures

- The drafter fit (full_ms, chosen) and the judge fit (full_ms_flip, chosen_flip) use disjoint player halves but the same V1 games: the shrinkage function, direct channel, role weights, calibration and layer of each arm are fit on V1 slot outcomes, and the two halves share those games' outcomes (teammates and opponents). Their coefficients are fit separately. A game-disjoint variant is a planned sensitivity after this run.
- Filtered states (the judge window, split halves, exclusions) carry no temporal layer, whose residuals come from unfiltered predictions; a poison check of every filter form gates each answer run.
- The tournament reads the base posterior sd (`sd`); the display band (`sd_display`) is for the site only.


## Calibration of the chosen model (slope of realized on predicted, V2 + OOT, smooth curve)

| axis | point | previous main | chosen |
|---|---|---|---|
| games | 0 | 1.33 | 1.08 |
| games | 3 | 1.18 | 1.01 |
| games | 10 | 1.10 | 1.00 |
| games | 30 | 0.98 | 0.95 |
| games | 100 | 0.88 | 0.98 |
| games | 300 | 0.76 | 0.90 |
| games | 1000 | 0.80 | 0.83 |
| share | 0.05 | 0.95 | 0.99 |
| share | 0.2 | 0.93 | 0.92 |
| share | 0.4 | 0.92 | 0.91 |
| share | 0.6 | 0.89 | 0.96 |
| share | 0.8 | 1.13 | 1.15 |
| share | 0.95 | 0.98 | 1.00 |
| recency | 0 | 1.04 | 0.99 |
| recency | 10 | 0.99 | 0.95 |
| recency | 30 | 0.97 | 0.95 |
| recency | 100 | 0.99 | 0.97 |
| recency | 365 | 0.91 | 1.01 |
| recency | 1000 | 0.92 | 0.95 |

RMS of (slope − 1), slot-weighted: previous games 0.160, share 0.059, recency 0.036; chosen games 0.060, share 0.047, recency 0.035.

## Display band (D = s2 (1 + exp(-games / 10)))

| window | cells | realized / claimed (95% CI) | 80% coverage |
|---|---|---|---|
| V1 | 8,161 | 0.78 (0.19–1.36) | 0.786 |
| V2 | 8,678 | 0.64 (0.17–1.17) | 0.793 |
| OOT | 51,890 | 0.24 (0.05–0.41) | 0.800 |

Median display sd by games on the hero: 0-0: 5.8pp, 1-9: 4.5pp, 10-29: 3.4pp, 30-99: 2.8pp, 100-299: 2.5pp, 300-1000000000: 2.1pp.

