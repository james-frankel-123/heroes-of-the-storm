# W3(a) — Post-patch recovery: when can you trust patch-local stats?

Within each sizable build (n=28), rolling estimates at game-count
checkpoints vs the build's final values (spec definition) and vs the
disjoint remainder (deployment-honest). Games-weighted MAE in pp over
keys clearing per-class game minimums; 'expected' = MAE from binomial
sampling alone under within-build stationarity.

## Recovery thresholds (games into the build until MAE <= X)

| signal class | X (pp) | median games (vs final) | p75 | builds reaching | median games (vs rest) |
|---|---|---|---|---|---|
| hero_wr | 1.0 | 10,719 | 12,237 | 28/28 | 17,241 |
| hero_pickrate | 0.25 | 32,944 | 43,098 | 28/28 | never |
| pair_with | 2.0 | 30,573 | 33,528 | 28/28 | never |
| pair_against | 2.0 | 28,684 | 31,156 | 28/28 | never |
| comp_wr | 0.5 | 21,875 | 30,500 | 28/28 | 44,667 |

## Median MAE curves (vs final / expected-from-sampling / vs rest)

| games | hero_wr | hero_pickrate | pair_with | pair_against | comp_wr |
|---|---|---|---|---|---|
| 500 | 4.56 / 4.71 / 4.60 | 1.57 / 1.22 / 1.59 | 5.96 / 6.96 / 6.29 | 7.38 / 6.96 / 7.53 | 2.48 / 2.92 / 2.61 |
| 1,000 | 3.51 / 3.53 / 3.58 | 1.24 / 0.86 / 1.26 | 6.68 / 6.52 / 6.77 | 6.48 / 6.53 / 6.55 | 2.06 / 2.33 / 2.09 |
| 2,000 | 2.57 / 2.51 / 2.64 | 1.00 / 0.60 / 1.02 | 5.93 / 5.88 / 6.30 | 5.88 / 5.83 / 6.15 | 1.72 / 1.90 / 1.80 |
| 3,000 | 2.10 / 2.04 / 2.18 | 0.92 / 0.49 / 0.95 | 5.45 / 5.44 / 5.74 | 5.43 / 5.39 / 5.83 | 1.41 / 1.60 / 1.49 |
| 5,000 | 1.57 / 1.56 / 1.72 | 0.76 / 0.37 / 0.86 | 4.77 / 4.83 / 5.29 | 4.75 / 4.76 / 5.21 | 1.15 / 1.30 / 1.32 |
| 8,000 | 1.20 / 1.20 / 1.43 | 0.67 / 0.29 / 0.77 | 4.16 / 4.18 / 4.80 | 4.08 / 4.08 / 4.69 | 0.94 / 1.05 / 1.07 |
| 12,000 | 0.91 / 0.95 / 1.22 | 0.52 / 0.23 / 0.69 | 3.50 / 3.55 / 4.38 | 3.42 / 3.46 / 4.22 | 0.71 / 0.83 / 0.96 |
| 20,000 | 0.65 / 0.68 / 1.03 | 0.37 / 0.16 / 0.63 | 2.70 / 2.72 / 3.90 | 2.57 / 2.62 / 3.70 | 0.54 / 0.60 / 0.89 |
| 30,000 | 0.50 / 0.51 / 0.88 | 0.30 / 0.12 / 0.55 | 2.06 / 2.09 / 3.43 | 1.99 / 2.00 / 3.24 | 0.45 / 0.45 / 0.83 |
| 50,000 | 0.32 / 0.32 / 0.81 | 0.20 / 0.08 / 0.54 | 1.30 / 1.28 / 3.24 | 1.22 / 1.23 / 3.10 | 0.30 / 0.28 / 0.80 |
| 80,000 | 0.16 / 0.16 / 1.04 | 0.10 / 0.04 / 0.73 | 0.66 / 0.66 / 3.53 | 0.62 / 0.63 / 3.43 | 0.11 / 0.15 / 0.92 |
| 120,000 | 0.12 / 0.12 / 0.87 | 0.07 / 0.03 / 0.53 | 0.52 / 0.52 / 3.39 | 0.49 / 0.49 / 3.21 | 0.10 / 0.12 / 0.62 |

**W3(d) hybrid window answer: 12,237 games** of recent history for the hero-WR family (p75 of the 1pp vs-final crossing across sizable builds).

Reading notes: vs-final converges to 0 mechanically (rolling games are a subset); vs-rest is bounded below by ~sqrt(2)x the one-sided sampling noise. Where observed MAE tracks the expected-from-sampling column, recovery is sampling-limited, not drift-limited — i.e. the estimate is trustworthy as soon as it is statistically stable.
