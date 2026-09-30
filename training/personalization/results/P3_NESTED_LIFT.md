# P3 experiment #2: nested-model lift (first pass, 2026-09-29)

Setup: `p3_wp_scores.py` scores every snapshot game with paper 2's drift-aware WP (3 `d2c_cumprev` seeds, swap-symmetrized). `p3_nested_lift.py` estimates personal skill from 938,751 games (2024-04-01 to the 2026-02-10 training cutoff), fits combiner weights on the first half of the post-cutoff games (70,974), and tests on the second half (72,474; all ten players labeled). Diagnostics are in `p3_mmr_depth_diag.py`.

Correction 2026-09-29: the first run applied a second sigmoid to the WP outputs (WinProbEnrichedModel already ends in one). Accuracy was unaffected (symmetrized ranking is identical) but probabilities and residuals were compressed. Numbers below are rerun with the fix, including the depth and MMR diagnostics (rerun 2026-09-29, raw output in `p3_mmr_depth_diag.txt`).

Variance components (empirical Bayes): player skill sd 4.3pp (k = 129 games), player x hero beyond player sd 2.2pp (k = 498). Residuals are mean-zero in the estimation window.

| model (test half) | acc | log loss | gain vs M0 (95% CI) |
|---|---|---|---|
| M0 population WP (drift-aware) | 57.19 | 0.67647 | |
| M1 + raw player-hero WR | 57.39 | 0.67567 | +0.0008 (0.0004, 0.0012) |
| M2 + shrunk player residual | 57.32 | 0.67576 | +0.0007 (0.0003, 0.0011) |
| M3 + shrunk player x hero residual | 57.48 | 0.67514 | +0.0013 (0.0007, 0.0018) |
| M4 + raw + player x hero residual | 57.73 | 0.67475 | +0.0017 (0.0011, 0.0023) |

Lift grows with history depth (quartiles of mean estimation games per slot, M4 vs M0): accuracy change +0.08, +0.48, +0.64, +0.95pp; log-loss gain -0.0000, +0.0010, +0.0022, +0.0037.

MMR: `player_mmr` for the game itself is leaked (62.5% accuracy; it is recorded when HP parses the replay, after the game). MMR from the player's latest game at least G days earlier: +1.1pp (G = 1), +0.6 (G = 7), +0.5 (G = 30), and it stacks with M4 (58.50, 58.05, 58.08). The static residual estimate (lag weeks to months) beats 30-day-old MMR on the same games (57.81 vs 57.51; log loss 0.67456 vs 0.67569). Previous-game MMR (no minimum gap): 58.01, and 58.23 on top of M4.

Reading: personal signal is real, modest in aggregate, concentrated in players with deep histories, and recency matters. Residual cleaning beats raw WR at the hero level. Next: online (causal, updated through the previous day) residual estimates; lagged MMR as a baseline arm; the frozen paper-1 WP ablation; party (premade) effects.

Correction 2026-09-30: the lagged-MMR numbers above use the naive lag and are inflated by upload-order leakage at every lag (25 to 42% of sources were uploaded after the game they predict). See P3_SKILL_DRIFT.md, section 0.
