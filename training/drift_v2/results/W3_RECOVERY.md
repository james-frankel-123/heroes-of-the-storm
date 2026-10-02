# W3(a) — Post-patch recovery: when can you trust patch-local stats?

Within each sizable build (n=28), rolling estimates at game-count
checkpoints vs the build's final values (spec definition) and vs the
disjoint remainder (deployment-honest). Games-weighted MAE in pp over
keys clearing per-class game minimums; 'expected' = MAE from binomial
sampling alone under within-build stationarity.

## Recovery thresholds (games into the build until MAE <= X)

| signal class | X (pp) | median games (vs final) | p75 | builds reaching | median games (vs rest) |
|---|---|---|---|---|---|
| hero_wr | 1.0 | 10,751 | 12,223 | 28/28 | 17,105 |
| hero_pickrate | 0.25 | 32,870 | 42,990 | 28/28 | never |
| pair_with | 2.0 | 30,565 | 33,188 | 28/28 | never |
| pair_against | 2.0 | 28,691 | 31,157 | 28/28 | never |
| comp_wr | 0.5 | 21,841 | 31,252 | 28/28 | 47,026 |

## Median MAE curves (vs final / expected-from-sampling / vs rest)

| games | hero_wr | hero_pickrate | pair_with | pair_against | comp_wr |
|---|---|---|---|---|---|
| 500 | 4.56 / 4.71 / 4.60 | 1.57 / 1.22 / 1.59 | 5.94 / 6.96 / 6.27 | 7.57 / 6.94 / 7.70 | 2.35 / 2.92 / 2.47 |
| 1,000 | 3.51 / 3.53 / 3.58 | 1.24 / 0.86 / 1.26 | 6.67 / 6.53 / 6.76 | 6.46 / 6.53 / 6.59 | 2.06 / 2.33 / 2.09 |
| 2,000 | 2.56 / 2.51 / 2.63 | 1.00 / 0.60 / 1.02 | 5.98 / 5.88 / 6.31 | 5.88 / 5.83 / 6.15 | 1.73 / 1.89 / 1.80 |
| 3,000 | 2.11 / 2.04 / 2.18 | 0.91 / 0.49 / 0.95 | 5.45 / 5.44 / 5.74 | 5.43 / 5.39 / 5.84 | 1.43 / 1.60 / 1.50 |
| 5,000 | 1.58 / 1.56 / 1.72 | 0.76 / 0.37 / 0.86 | 4.79 / 4.83 / 5.28 | 4.74 / 4.76 / 5.21 | 1.18 / 1.30 / 1.29 |
| 8,000 | 1.20 / 1.20 / 1.42 | 0.67 / 0.29 / 0.77 | 4.16 / 4.18 / 4.81 | 4.08 / 4.08 / 4.69 | 0.94 / 1.05 / 1.07 |
| 12,000 | 0.92 / 0.95 / 1.21 | 0.52 / 0.23 / 0.69 | 3.50 / 3.55 / 4.38 | 3.42 / 3.46 / 4.22 | 0.71 / 0.83 / 0.94 |
| 20,000 | 0.65 / 0.68 / 1.03 | 0.37 / 0.16 / 0.63 | 2.70 / 2.72 / 3.90 | 2.57 / 2.62 / 3.70 | 0.54 / 0.60 / 0.89 |
| 30,000 | 0.50 / 0.51 / 0.88 | 0.30 / 0.12 / 0.55 | 2.06 / 2.09 / 3.42 | 1.99 / 2.00 / 3.24 | 0.44 / 0.45 / 0.84 |
| 50,000 | 0.32 / 0.32 / 0.81 | 0.20 / 0.08 / 0.55 | 1.29 / 1.28 / 3.25 | 1.22 / 1.23 / 3.10 | 0.30 / 0.28 / 0.80 |
| 80,000 | 0.16 / 0.16 / 0.97 | 0.11 / 0.04 / 0.73 | 0.66 / 0.66 / 3.53 | 0.63 / 0.63 / 3.48 | 0.12 / 0.15 / 0.87 |
| 120,000 | 0.12 / 0.12 / 0.88 | 0.07 / 0.03 / 0.53 | 0.52 / 0.51 / 3.40 | 0.48 / 0.49 / 3.22 | 0.10 / 0.12 / 0.63 |

**W3(d) hybrid window answer: 12,223 games** of recent history for the hero-WR family (p75 of the 1pp vs-final crossing across sizable builds).

Reading notes: vs-final converges to 0 mechanically (rolling games are a subset); vs-rest is bounded below by ~sqrt(2)x the one-sided sampling noise. Where observed MAE tracks the expected-from-sampling column, recovery is sampling-limited, not drift-limited — i.e. the estimate is trustworthy as soon as it is statistically stable.
