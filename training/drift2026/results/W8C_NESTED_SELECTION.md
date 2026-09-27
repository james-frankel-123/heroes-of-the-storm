# W8c — nested-validation champion selection

Inner cutoff 2.55.13.95301 (train <= inner; models never see the pseudo-future or the true test window). Pseudo-future = the 4 pre-cutoff builds 2.55.14.* (295,608 rows, comparable to the true window's 287,362). 4 cells x 3 seeds, existing feature caches, D2 protocol.

| cell | val acc | pseudo-future acc | true-future acc (same inner models) | Q7 full-trained true-future (ref) |
|---|---|---|---|---|
| cumulative (cumprev) | 57.64 ± 0.06 | 56.90 ± 0.04 | 56.99 ± 0.05 | 57.08 |
| decayed HL=365d | 57.50 ± 0.08 | 57.04 ± 0.04 | 57.03 ± 0.06 | 57.07 |
| decayed HL=90d | 57.50 ± 0.08 | 57.12 ± 0.04 | 57.11 ± 0.05 | 57.19 |
| decayed HL=90d + k=100 shrink | 57.47 ± 0.12 | 57.09 ± 0.07 | 57.15 ± 0.06 | 57.22 |

- Selected on pseudo-future (nested, honest): **decayed90**
- Selected on true future with the same inner-trained models: **decayed90k100**
- Selected on true future with the Q7 full-trained models (the original, test-window-selected ranking): **decayed90k100**
- HL=90 family seed-clean above the other cells on pseudo-future: NO (family min 57.04 vs rest max 57.08)

**Verdict: the nested selection TRANSFERS.** The cell picked on a strictly pre-cutoff pseudo-future window (decayed90) matches the test-window-selected champion family (decayed90k100); the nested-selected cell beats cumulative on the TRUE window too (57.11 vs 56.99 with the inner-trained models, 57.19 vs 57.08 full-trained), so a practitioner selecting honestly at the inner cutoff would have shipped a decayed-90d model and realized the gain.

Caveat: WITHIN the HL=90 family the ordering flips between windows (pseudo-future prefers decayed90 57.12 vs decayed90k100 57.09; the true window prefers decayed90k100). The k=100-shrink refinement is within noise of unshrunk HL=90 — the honest claim is "decayed ~90d aggregates", not the specific shrinkage corner. The family-level separation from cumulative/HL=365 is what survives nesting; it is not fully seed-clean on the pseudo-future window (family min 57.04 vs rest max 57.08).
