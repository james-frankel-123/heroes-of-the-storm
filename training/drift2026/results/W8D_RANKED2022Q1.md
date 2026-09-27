# W8d — ranked early-2022 vintage judge (mode/era deconfound)

The QM-2021 point of the W6 vintage matrix is double-confounded: QM games
AND a 2021 era. The ranked corpus window 2021-12-01..2022-03-31 holds
158,709 games (>= the 150K threshold), so one more naive-feature judge was
trained on exactly that window (qm2026 recipe via q2_vintage_judges.
train_judge, seed 20260713; volume below the 314,761 Q2 cap — noted).
Judge: `models/vintage_judges/wp_2022Q1-ranked.pt` (auto-included in all
subsequent w7_rescore vintage rescoring). Raw: `results/w8/
W8D_RANKED2022Q1.json`.

| judge | mode | era | held-out acc | maintained WP on W6 drafts | win share |
|---|---|---|---|---|---|
| QM-2021 | QM | 2021 | 0.6111 | 0.4338 ± 0.0033 | 0.322 |
| **2022Q1-ranked (new)** | ranked | 2021-12..2022-03 | 0.5716 | **0.4688 ± 0.0021** | 0.367 |
| 2022-07 | ranked | ..2022-07 | 0.5719 | 0.4611 | — |
| 2023-07 | ranked | ..2023-07 | 0.5642 | 0.4927 | — |

Reading: holding mode fixed (ranked) and moving to the earliest available
era, the judge still scores the maintained agent BELOW 0.5 (0.4688) — the
era gradient of the vintage matrix is real and not a QM artifact — but the
new point sits ~3.5pp above QM-2021 at an adjacent era, so roughly a third
of the QM-2021 row's extremity is the mode confound, the rest is era. The
ranked-era gradient is monotone from 0.46-0.47 (2022) through ~0.50 (2026
judges) on the W6 drafts.
