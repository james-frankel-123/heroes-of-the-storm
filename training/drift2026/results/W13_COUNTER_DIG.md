# W13 counter dig: metric bias, validity, net realized-outcome score

Clean truth games (2.55.16.97039 + 2.55.17.*) split by replay_id parity; statistics from one half, outcomes/scoring on the other.

## even_stats_odd_outcomes

Real games: 149,161. corr(d_hwr, d_ctr) = -0.035, slope -0.026; log-odds counter: corr -0.018, slope -0.0005.

Logistic regression of the real result on team differences (per unit): intercept +0.0071 ± 0.0052, hwr +0.1785 ± 0.0035, syn +0.0320 ± 0.0039, ctr +0.0574 ± 0.0047

| matchup | counter | counter (log-odds) | counter detrended | net realized WP (pp) |
|---|---|---|---|---|
| maintained vs stale-4mo (W6, 5x5 seeds) | -0.370 ± 0.138 (-2.7) | -0.015 ± 0.006 (-2.6) | -0.352 ± 0.138 (-2.6) | +2.796 ± 0.575 (4.9) |
| maintained vs stale-1yr (W7) | -0.372 ± 0.120 (-3.1) | -0.015 ± 0.005 (-3.0) | -0.357 ± 0.119 (-3.0) | +2.449 ± 0.812 (3.0) |
| maintained vs stale-2yr (W7) | -0.575 ± 0.110 (-5.2) | -0.023 ± 0.004 (-5.1) | -0.544 ± 0.110 (-4.9) | +4.325 ± 0.586 (7.4) |
| volume-matched maintained vs stale-2yr (W8a) | -0.270 ± 0.181 (-1.5) | -0.010 ± 0.008 (-1.3) | -0.242 ± 0.185 (-1.3) | +4.487 ± 0.900 (5.0) |
| maintained vs stale-4mo (W10, 15x15 seeds) | -0.433 ± 0.096 (-4.5) | -0.017 ± 0.004 (-4.4) | -0.414 ± 0.095 (-4.3) | +2.804 ± 0.393 (7.1) |
| maintained-d90 vs stale-4mo (W8b) | -0.601 ± 0.101 (-6.0) | -0.024 ± 0.004 (-5.8) | -0.572 ± 0.100 (-5.7) | +4.165 ± 0.284 (14.7) |
| maintained-d90 vs stale-1yr (W8b) | -0.410 ± 0.153 (-2.7) | -0.016 ± 0.006 (-2.6) | -0.388 ± 0.151 (-2.6) | +3.613 ± 0.518 (7.0) |
| maintained-d90 vs stale-2yr (W8b) | -0.536 ± 0.168 (-3.2) | -0.021 ± 0.007 (-3.1) | -0.502 ± 0.167 (-3.0) | +4.959 ± 0.269 (18.5) |
| maintained-d90 vs maintained (W11) | -0.097 ± 0.085 (-1.1) | -0.004 ± 0.003 (-1.1) | -0.089 ± 0.085 (-1.1) | +1.264 ± 0.354 (3.6) |

## odd_stats_even_outcomes

Real games: 149,267. corr(d_hwr, d_ctr) = -0.039, slope -0.030; log-odds counter: corr -0.024, slope -0.0007.

Logistic regression of the real result on team differences (per unit): intercept +0.0012 ± 0.0052, hwr +0.1912 ± 0.0037, syn +0.0285 ± 0.0039, ctr +0.0589 ± 0.0047

| matchup | counter | counter (log-odds) | counter detrended | net realized WP (pp) |
|---|---|---|---|---|
| maintained vs stale-4mo (W6, 5x5 seeds) | -0.163 ± 0.151 (-1.1) | -0.006 ± 0.006 (-1.0) | -0.144 ± 0.151 (-1.0) | +2.675 ± 0.914 (2.9) |
| maintained vs stale-1yr (W7) | +0.078 ± 0.170 (0.5) | +0.004 ± 0.007 (0.5) | +0.094 ± 0.171 (0.5) | +2.580 ± 1.274 (2.0) |
| maintained vs stale-2yr (W7) | -0.524 ± 0.190 (-2.8) | -0.020 ± 0.008 (-2.6) | -0.493 ± 0.190 (-2.6) | +3.933 ± 1.049 (3.8) |
| volume-matched maintained vs stale-2yr (W8a) | -0.625 ± 0.149 (-4.2) | -0.025 ± 0.006 (-4.0) | -0.599 ± 0.150 (-4.0) | +2.886 ± 1.190 (2.4) |
| maintained vs stale-4mo (W10, 15x15 seeds) | -0.192 ± 0.080 (-2.4) | -0.007 ± 0.003 (-2.3) | -0.173 ± 0.079 (-2.2) | +2.610 ± 0.478 (5.5) |
| maintained-d90 vs stale-4mo (W8b) | -0.292 ± 0.091 (-3.2) | -0.011 ± 0.004 (-3.0) | -0.262 ± 0.091 (-2.9) | +4.223 ± 0.416 (10.1) |
| maintained-d90 vs stale-1yr (W8b) | -0.062 ± 0.168 (-0.4) | -0.002 ± 0.007 (-0.3) | -0.042 ± 0.167 (-0.2) | +3.145 ± 0.943 (3.3) |
| maintained-d90 vs stale-2yr (W8b) | -0.627 ± 0.161 (-3.9) | -0.024 ± 0.007 (-3.6) | -0.594 ± 0.161 (-3.7) | +4.032 ± 0.804 (5.0) |
| maintained-d90 vs maintained (W11) | +0.079 ± 0.096 (0.8) | +0.003 ± 0.004 (0.9) | +0.089 ± 0.096 (0.9) | +1.593 ± 0.490 (3.3) |

