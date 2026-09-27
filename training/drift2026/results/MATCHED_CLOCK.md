# Q12 — Matched-clock per-signal decay (reviewer follow-up)

SIGNAL_DECAY/PARTIAL_REFRESH finding: hero-WR is the fastest-decaying
signal but pairwise/comp carries refresh value -> decay each signal
class at its own measured rate. Stats built by
build_matched_clock_stats.py: hero-family aggregates (hero WR,
pick+ban, hero-map) from the Q7 decayed-90d files, pair/comp
(pairwise counter/synergy, comp WR) from the decayed-365d files —
composed at the exact W3(d)/Q5 attribute split; reverse assignment
as the control. Feature passes decayedmc_prev / decayedmcrev_prev
(strictly-causal `_prev` convention, row-aligned with
features_cumulative_prev.npz, 3,898,174 rows, 0 failed). Training =
train_drift_wp.py verbatim (D2 protocol, regime=all, 3 seeds,
n_train 3,538,596 — identical to Q7). counter/synergy features are
genuine hybrids by construction (pairwise WRs normalized by the
other clock's hero WRs), verified column-exact against the parent
passes for the pure groups.

| cell | val acc | future acc | sizable acc | sanity 28 (21) | epochs |
|---|---|---|---|---|---|
| cumulative (d2c_cumprev, reference) | 57.16 ± 0.02 | 57.08 ± 0.03 | 57.09 ± 0.04 | 24.0 (18.7) | 58 |
| uniform decayed 90d + k=100 shrink (Q7 champion, reference) | 57.24 ± 0.05 | 57.22 ± 0.06 | 57.23 ± 0.06 | 25.3 (19.0) | 63 |
| uniform decayed 90d (reference) | 57.26 ± 0.07 | 57.19 ± 0.05 | 57.19 ± 0.04 | 25.7 (19.3) | 58 |
| uniform decayed 365d (reference) | 57.17 ± 0.04 | 57.07 ± 0.04 | 57.08 ± 0.04 | 23.3 (18.3) | 60 |
| MATCHED CLOCK: hero family 90d / pair+comp 365d **(new)** | 57.27 ± 0.07 | 57.07 ± 0.04 | 57.08 ± 0.05 | 24.0 (18.7) | 55 |
| reverse control: hero family 365d / pair+comp 90d **(new)** | 57.21 ± 0.05 | 57.15 ± 0.04 | 57.15 ± 0.04 | 25.7 (19.7) | 62 |

Per-seed future acc (42/123/777):
- d2c_cumprev: 57.086 / 57.114 / 57.044
- q7_decayed90k100: 57.198 / 57.160 / 57.303
- q7_decayed90: 57.254 / 57.154 / 57.159
- q7_decayed365: 57.115 / 57.022 / 57.069
- q12_hero90pair365: 57.060 / 57.131 / 57.023
- q12_hero365pair90: 57.127 / 57.107 / 57.202

## Verdict — does per-class clocking beat uniform decay?

- matched clock 57.07 vs cumulative 57.08 (-0.01 pp), vs uniform decayed90k100 57.22 (-0.15 pp), vs uniform decayed90 57.19 (-0.12 pp).
- reverse control 57.15 (+0.07 pp vs matched): the REVERSE assignment (hero slow / pair fast) is the better of the two — the matched-clock hypothesis fails in its own direction.

**Verdict: NO — per-signal-class clocking does not beat uniform decay.** Both mixed cells sit below uniform decayed90 (57.19) and the decayed90k100 champion (57.22).

Reading: each cell's future accuracy tracks its PAIR/COMP clock, not its hero clock — matched (pair 365d) 57.07 ~= uniform decayed365 57.07, reverse (pair 90d) 57.15 sits near uniform decayed90 57.19. The training-time recency gain of DECAYED_AGGREGATES is therefore carried at least as much by the pair/comp features as by the hero family. That is consistent with PARTIAL_REFRESH (pairwise/comp carries the refresh value at inference) but contrary to the naive recipe "decay each class at its measured signal half-life": hero-WR decaying fastest as a SIGNAL does not mean the hero features are where the fast CLOCK pays. Recommendation for the paper: keep the uniform-90d (k=100 shrunk) champion; report this 2-cell control as the reviewer follow-up.
