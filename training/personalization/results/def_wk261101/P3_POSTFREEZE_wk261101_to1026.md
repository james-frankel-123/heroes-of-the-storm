# P3 post-freeze confirmation (wk261101, coefficients frozen from def1027)

Games: 84,658 complete 2.57 Storm League games (all ten players in the v1 hero set), days 20724..20752.

| model | gain over population WP x1000 (SE) | accuracy | game calibration slope | slot calibration slope |
|---|---|---|---|---|
| today's kernel | +15.87 (0.59) | 60.20% | 1.005 | 0.938 |
| previous main model | +25.34 (0.76) | 61.59% | 0.991 | 1.032 |
| chosen model | +32.95 (0.91) | 62.69% | 0.956 | 0.915 |

| contrast | gain x1000 (game SE) | week-block 95% CI | 3-day-block 95% CI | accuracy |
|---|---|---|---|---|
| chosen-full_ms_layer | +7.61 (0.44) | [+6.82, +8.07] (5 weeks) | [+6.36, +8.87] | +1.09pp |
| chosen-today | +17.08 (0.69) | [+15.96, +17.94] (5 weeks) | [+15.45, +18.66] | +2.49pp |
| full_ms_layer-today | +9.47 (0.49) | [+8.95, +9.96] (5 weeks) | [+8.53, +10.36] | +1.40pp |

Slot calibration slope by games on the hero (smooth curve, 95% player-bootstrap band):

| games | today's kernel | previous main model | chosen model |
|---|---|---|---|
| 0 | 0.91 (0.76-1.08) | 1.04 (0.95-1.15) | 0.84 (0.77-0.91) |
| 3 | 0.93 (0.83-1.00) | 1.01 (0.95-1.08) | 0.87 (0.83-0.92) |
| 10 | 0.97 (0.86-1.07) | 1.03 (0.96-1.11) | 0.91 (0.85-0.96) |
| 30 | 0.84 (0.75-0.90) | 0.92 (0.83-0.97) | 0.94 (0.88-1.01) |
| 100 | 0.70 (0.62-0.81) | 0.79 (0.69-0.92) | 0.92 (0.83-1.01) |
| 300 | 0.86 (0.72-1.03) | 0.85 (0.75-1.00) | 1.01 (0.84-1.15) |
| 1000 | 0.69 (0.40-0.96) | 0.69 (0.46-0.90) | 0.73 (0.51-0.98) |

Pre-freeze windows of this build vs the frozen build (gain x1000):

- today's kernel: V1 +18.340 vs +18.340, V2 +19.225 vs +19.225, OOT +18.489 vs +18.489
- previous main model: V1 +26.687 vs +26.687, V2 +29.218 vs +29.218, OOT +28.313 vs +28.313
- chosen model: V1 +35.801 vs +35.800, V2 +36.531 vs +36.530, OOT +37.381 vs +37.381
