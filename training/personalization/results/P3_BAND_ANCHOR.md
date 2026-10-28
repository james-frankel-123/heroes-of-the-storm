# P3 anchored display-band check (review item C17, 2026-10-28)

The site shows a player's edge on a hero as estimate ± 1.28 √D, with D = s² (1 + exp(−games on the hero / 10)). The earlier check (P3_SHRINKAGE.md section 6) compared D with the error of an average over every prediction inside a player-hero cell. Those predictions keep learning from the cell's own games, so that check flattered the band. Here the estimate and D are frozen at the cell's first scored game and compared with what follows.

## Headline

- **The current band is too narrow by a factor of about 1.6 in sd for the player's edge over the next 10 games.**
  - Realized latent error variance is 2.6 to 3.2 times the claimed D: V1 2.76 (95% CI 2.11 to 3.33), V2 3.17 (2.58 to 3.78), OOT 2.58 (2.41 to 2.77).
  - The gap is largest for thin records: about 3x at 0 to 29 games on the hero, 1.8 to 2.2x at 30 to 299 games.
- **Coverage of the 80% band on the next 10 games' mean result still looks fine (0.77), and it hides the problem.** Game-to-game noise in a 10-game mean (sd 15.6pp) dwarfs the band itself (median sd 2.9 to 3.2pp). Coverage only falls short where the band matters most: 0.71 to 0.72 at 0 games on the hero.
- **Proposed band:** D' = s² exp(1.743 + 0.234 l − 5.170 l²) + (3.8pp)², with l = log1p(games on the hero) / log1p(300). It is fit by maximum likelihood on V1 cells.
  - The prior-driven part shrinks with games: 5.7 s² at 0 games, 2.5 s² at 10, 1.0 s² at 30, about 0 past 100.
  - A floor of 3.8pp remains for every record. A player's edge over the next ten games moves by about that much even with a long record (form, drift, patches).
  - Display sd at the median s²: 10.5pp at 0 games, 5.4pp at 20, 4.3pp at 60, 3.9pp at 200 and above. The current band shows 5.8, 3.4, 2.8, 2.5 and 2.1pp.
  - Held out:
    - realized/claimed 1.13 on V2 (0.89 to 1.32) and 0.90 on OOT (0.85 to 0.95);
    - 80% coverage 0.79 to 0.80 in every games bin;
    - Gaussian log-likelihood per cell better than the current band on both V2 and OOT.

## Method

- **Cells:** (player, hero) pairs with at least K scored games inside a window, using the chosen model's pinned def1027 arm (`results/def_def1027/rcf/p3c/chosen__merged.npz`).
- **Anchor:** the cell's first scored game. The displayed estimate s (slot prediction) and D are those as of the day before it.
- **Target:** the mean realized residual r = y − WP of the cell's first K games (the anchor game and the next K − 1).
- **Error and noise:** e = mean(r) − s. Binomial game noise is Σ wp(1 − wp) / K². This already carries the teammates' and opponents' effects in expectation, since Var(y − wp) ≈ wp(1 − wp).
- **Ratio:** Σ(e² − noise) / Σ D. Player-bootstrap 95% CIs.

## Current band, K = 10

| window | cells | realized / claimed (95% CI) | 80% coverage | median claimed sd |
|---|---|---|---|---|
| V1 | 8,161 | 2.76 (2.11 to 3.33) | 0.775 | 2.9pp |
| V2 | 8,678 | 3.17 (2.58 to 3.78) | 0.768 | 2.9pp |
| OOT | 51,890 | 2.58 (2.41 to 2.77) | 0.772 | 3.2pp |

By games on the hero at the anchor (OOT; V1 and V2 agree within their wider CIs):

| games | cells | realized / claimed (95% CI) | 80% coverage | claimed sd |
|---|---|---|---|---|
| 0 | 8,176 | 2.87 (2.67 to 3.08) | 0.724 | 5.8pp |
| 1-9 | 8,619 | 2.72 (2.33 to 3.07) | 0.759 | 4.4pp |
| 10-29 | 8,822 | 2.48 (1.85 to 3.13) | 0.774 | 3.4pp |
| 30-99 | 12,422 | 2.19 (1.47 to 2.81) | 0.790 | 2.8pp |
| 100-299 | 9,161 | 1.81 (0.67 to 2.87) | 0.790 | 2.5pp |
| 300+ | 4,690 | 0.58 (−1.18 to 2.48) | 0.799 | 2.2pp |

The horizon matters, as drift and new-hero learning would predict. Realized/claimed is 2.2 to 2.5 at K = 5, 2.6 to 3.2 at K = 10, and 3.0 to 4.3 at K = 20.

## Corrected forms, fit on V1, judged on V2 and OOT

| form | parameters | held-out log-lik per cell, V2 / OOT (higher is better) | OOT realized / claimed |
|---|---|---|---|
| current band | 0 | 1.2743 / 1.2771 | 2.58 |
| a · D | 1 | 1.2795 / 1.2813 | 0.94, falling to 0.21 at 300+ |
| a · s² + τ² | 2 | 1.2782 / 1.2803 | 0.98, 1.45 at 0 games |
| s² exp(quadratic in l) | 3 | 1.2796 / 1.2813 | 0.97, 2.1 at 100-299 |
| **s² exp(quadratic in l) + τ²** | 4 | **1.2802 / 1.2815** | **0.90; 0.81 to 1.00 up to 299 games** |

The recommended form is the last row. The 300+ bin rests on few cells with large noise in every form.

## On the site (2026-10-28)

The weekly publish now uses this band (`SITE_BAND=anchored_v1` in `run_site_publish.sh`, band version `anchored_v1` in `p3_rcf_states.BANDS`). The estimates and the base sd are unchanged. The publish check verifies the card band against the formula, computed from the base sd and the games on the hero (max difference 1.5e-08). The model artifact records the version, formula, coefficients and held-out numbers. The cards were republished to Neon and the page copy updated in commit 2d9e973; the live /players API serves the new band.

## Notes

- The empirical-noise variant (the cell's own spread over its K games) gives the same picture: realized/claimed 2.8 to 3.6 at K = 10. In this check it is less stable than the binomial noise, because a 10-game spread is itself very noisy.
- The proposal changes the display only. The tournament reads the base posterior sd (`sd`), not the display band.
- Script: `p3_rcf_band_anchor.py` (`--k`, `BAND_NOISE`). Outputs: `results/def_def1027/rcf/band_anchor_chosen_k{5,10,20}_{binomial,empirical}.json`.
