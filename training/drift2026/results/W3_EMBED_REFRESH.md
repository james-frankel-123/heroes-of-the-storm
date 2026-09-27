# W3(e) — Do the two wave-1 winners stack? (patch embedding x per-build stats refresh)

w3e_embedrefresh = embed regime TRAINED on cumulative_prev (causally refreshed) features; test rows use their build's prev-cumulative stats + the cutoff-clamped embedding. 'embed eval-only' = wave-1 d2b_embed checkpoints (trained on cutoff-stats features) scored on cumprev-refreshed test features without retraining (a deployment shortcut, feature-distribution mismatched by construction).

| arm | val acc | future acc | sizable | sanity 28 | degen % |
|---|---|---|---|---|---|
| w3e_embedrefresh | 57.18 ± 0.10 | 57.09 ± 0.03 | 57.10 ± 0.03 | 25.0 ± 0.8 | 23.3 ± 4.3 |
| d2b_embed | 59.14 ± 0.02 | 56.55 ± 0.02 | 56.55 ± 0.02 | 25.7 ± 0.5 | 47.0 ± 4.6 |
| d2c_cumprev | 57.16 ± 0.02 | 57.08 ± 0.03 | 57.09 ± 0.04 | 24.0 ± 0.0 | 16.8 ± 0.7 |
| d2b_embed eval-only on cumprev feats | - | 56.64 ± 0.01 | 56.65 ± 0.01 | - | - |

## Verdict

- Stacked arm 57.09 vs cumprev alone 57.08 (+0.01) and embed alone 56.55 (+0.54): the winners **stack**.
