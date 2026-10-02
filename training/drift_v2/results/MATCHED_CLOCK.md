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
| cumulative (d2c_cumprev, reference) | 57.02 ± 0.06 | 56.98 ± 0.03 | 56.99 ± 0.03 | 24.0 (17.7) | 62 |
| uniform decayed 90d + k=100 shrink (Q7 champion, reference) | 57.00 ± 0.08 | 57.23 ± 0.02 | 57.23 ± 0.03 | 23.3 (17.3) | 59 |
| uniform decayed 90d (reference) | 57.21 ± 0.04 | 57.17 ± 0.04 | 57.17 ± 0.04 | 25.7 (18.7) | 64 |
| uniform decayed 365d (reference) | 57.00 ± 0.02 | 57.18 ± 0.03 | 57.20 ± 0.03 | 24.3 (17.7) | 53 |
| MATCHED CLOCK: hero family 90d / pair+comp 365d **(new)** | 57.14 ± 0.05 | 57.07 ± 0.03 | 57.08 ± 0.03 | 24.7 (18.3) | 59 |
| reverse control: hero family 365d / pair+comp 90d **(new)** | 57.13 ± 0.06 | 57.14 ± 0.06 | 57.15 ± 0.06 | 24.3 (19.0) | 59 |

Per-seed future acc (42/123/777):
- d2c_cumprev: 57.011 / 56.948 / 56.984
- q7_decayed90k100: 57.205 / 57.259 / 57.214
- q7_decayed90: 57.169 / 57.121 / 57.218
- q7_decayed365: 57.184 / 57.222 / 57.147
- q12_hero90pair365: 57.053 / 57.114 / 57.045
- q12_hero365pair90: 57.091 / 57.119 / 57.221

## Verdict — does per-class clocking beat uniform decay?

- matched clock 57.07 vs cumulative 56.98 (+0.09 pp), vs uniform decayed90k100 57.23 (-0.16 pp), vs uniform decayed90 57.17 (-0.10 pp).
- reverse control 57.14 (+0.07 pp vs matched): the REVERSE assignment (hero slow / pair fast) is the better of the two — the matched-clock hypothesis fails in its own direction.

**Verdict: NO — per-signal-class clocking does not beat uniform decay.** Both mixed cells sit below uniform decayed90 (57.17) and the decayed90k100 champion (57.23).

Reading: each cell's future accuracy tracks its PAIR/COMP clock, not its hero clock — matched (pair 365d) 57.07 ~= uniform decayed365 57.18, reverse (pair 90d) 57.14 sits near uniform decayed90 57.17. The training-time recency gain of DECAYED_AGGREGATES is therefore carried at least as much by the pair/comp features as by the hero family. That is consistent with PARTIAL_REFRESH (pairwise/comp carries the refresh value at inference) but contrary to the naive recipe "decay each class at its measured signal half-life": hero-WR decaying fastest as a SIGNAL does not mean the hero features are where the fast CLOCK pays. Recommendation for the paper: keep the uniform-90d (k=100 shrunk) champion; report this 2-cell control as the reviewer follow-up.
