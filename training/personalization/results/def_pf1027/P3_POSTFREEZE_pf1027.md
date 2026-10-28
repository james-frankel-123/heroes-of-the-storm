# P3 post-freeze confirmation (pf1027, coefficients frozen from def1027)

Games: 83,955 complete 2.57 Storm League games (all ten players in the v1 hero set), days 20724..20752.

| model | gain over population WP x1000 (SE) | accuracy | game calibration slope | slot calibration slope |
|---|---|---|---|---|
| today's kernel | +16.27 (0.60) | 60.08% | 1.006 | 0.944 |
| previous main model | +25.62 (0.76) | 61.54% | 0.990 | 1.037 |
| chosen model | +33.47 (0.91) | 62.59% | 0.958 | 0.923 |

| contrast | gain x1000 (game SE) | week-block 95% CI | 3-day-block 95% CI | accuracy |
|---|---|---|---|---|
| chosen-full_ms_layer | +7.84 (0.45) | [+7.23, +8.19] (5 weeks) | [+6.61, +9.08] | +1.05pp |
| chosen-today | +17.20 (0.69) | [+16.29, +17.89] (5 weeks) | [+15.79, +18.63] | +2.51pp |
| full_ms_layer-today | +9.35 (0.49) | [+8.97, +9.76] (5 weeks) | [+8.68, +10.05] | +1.46pp |

Slot calibration slope by games on the hero (smooth curve, 95% player-bootstrap band):

| games | today's kernel | previous main model | chosen model |
|---|---|---|---|
| 0 | 0.93 (0.75-1.12) | 1.07 (0.96-1.17) | 0.85 (0.78-0.94) |
| 3 | 0.91 (0.81-0.99) | 1.02 (0.94-1.08) | 0.89 (0.84-0.94) |
| 10 | 0.98 (0.90-1.08) | 1.05 (0.96-1.12) | 0.94 (0.85-1.01) |
| 30 | 0.83 (0.76-0.88) | 0.91 (0.84-0.98) | 0.93 (0.87-1.00) |
| 100 | 0.74 (0.64-0.84) | 0.81 (0.72-0.90) | 0.94 (0.85-1.01) |
| 300 | 0.85 (0.74-0.99) | 0.83 (0.70-0.96) | 1.01 (0.86-1.12) |
| 1000 | 0.74 (0.48-0.94) | 0.72 (0.48-0.96) | 0.76 (0.50-0.93) |

Pre-freeze windows of this build vs the frozen build (gain x1000):

- today's kernel: V1 +18.340 vs +18.340, V2 +19.225 vs +19.225, OOT +18.489 vs +18.489
- previous main model: V1 +26.687 vs +26.687, V2 +29.218 vs +29.218, OOT +28.313 vs +28.313
- chosen model: V1 +35.801 vs +35.800, V2 +36.532 vs +36.530, OOT +37.381 vs +37.381
