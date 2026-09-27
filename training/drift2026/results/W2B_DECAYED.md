# W2b greedy evals: decayed-aggregate regimes (thesis-figure check, 2026-07-14)

Motivated by the Fig-1 skeptic reading (accuracy appears to predict degen
because maintained is best on both axes). Protocol identical to W2B
(3 seeds x 500 drafts, deployment stats = decayed{90,90k100} refreshed to
the last completed build).

| cell | future acc | degen % (s42/s123/s777) | mean degen |
|---|---|---|---|
| d2c_cumprev (ref) | 57.08 | — | 16.8 ± 0.7 |
| q7_decayed90 | 57.19 | 19.8 / 16.8 / 20.0 | 18.9 ± 1.8 |
| q7_decayed90k100 | 57.22 | 19.6 / 17.4 / 20.4 | 19.1 ± 1.6 |

VERDICT: no high-degen outlier at the top — the decayed regimes stay
composition-disciplined (~19%). The honest thesis evidence is therefore:
(a) within the seven standard regimes, corr(acc, degen) = +0.78 (accuracy
ranks policies BACKWARDS along the tuning axis); (b) at the top of the
accuracy range, +0.14pp accuracy buys zero-to-slightly-negative
discipline change (+2.3pp degen, inside seed noise) — accuracy gains do
not purchase policy safety anywhere in the design space.
