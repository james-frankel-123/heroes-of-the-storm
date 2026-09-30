# P3: do patches reshuffle hero skill, and do forced roles cost games? (2026-09-30)

Two hypotheses from Max, tested on the same data as P3_SKILL_DRIFT.md: the snapshot, Storm League games from 2024-04 to 2026-05-22. Residuals are taken against the drift-aware WP and adjusted for experience. The fit window is V1 and the test window is V2 (72,474 games). All runs used 4 cores, no GPU.

## Verdicts

- **H1, patches move individual hero skill in a jump:** not supported.
  - Players do not reorder on patched heroes any more than on unpatched heroes at the same boundaries. Stability of player skill across a boundary is 0.67 on patched heroes vs 0.68 on unpatched ones, difference −0.01 (95% CI −0.44 to +0.34).
  - A state-space model with an extra jump at the boundary fits worse at every jump size tried, for patched heroes and placebo heroes alike. A 1pp jump on patched heroes costs 22 log-likelihood units.
  - The one hint: players reshuffle more on heroes whose aggregate win rate moved 2pp or more (7% of boundary-hero cells). A jump placed only on those heroes still fits no better than the same jump on a matched placebo set.
  - Any real jump is below about 1pp sd, against a hero-specific skill spread of 2pp.
- **H2, forced off-role play costs games:** supported, and smaller than the raw data suggests.
  - Off-role means a role that makes up under 10% of the player's earlier games. 9.6% of established players' slots are off-role by Blizzard role, 17.5% by fine role.
  - Raw penalty: −1.7pp (±0.1), after controlling for games on the hero.
  - After the skill model (per-player role, fine-role and hero terms, plus the experience offset), −0.9pp (±0.1) remains in 2025-26. It is −0.6pp (±0.2) in the post-cutoff test window.
  - Last picks are off-role 2.3 times as often as first picks (13.1% vs 5.6%). The penalty per off-role slot is the same at every pick position.
  - At team level, each off-role player costs about 1.3pp of win probability beyond the WP plus skill model. Adding role-familiarity terms improves held-out log loss by +0.0005 (0.0002, 0.0008).

## H1. Patches and individual skill (`p3_ph_patch.py`, `p3_ph_patch_big.py`)

**Boundaries.** I used the 27 sizable build boundaries from paper 2's detector study. Two definitions of a "changed" hero at each boundary:

- **notes:** the patch notes (bug-fix-only and ARAM-only mentions excluded);
- **detector:** heroes the win-rate detector flagged (wr_bh).

All other heroes count as unchanged.

**Population removal.** Each residual has its (build, hero) mean subtracted, so a hero's patch-wide shift is removed, including the WP's one-build lag in catching it. Only the player-specific part is tested.

**Cells.** A cell is a (boundary, player, hero) with at least 10 games on each side, within 90 days of the boundary.

**A. Difference-in-differences** (split-half design, player-cluster bootstrap):

- Each side's games are split into odd and even games in play order. Variance comes from odd × even cross products and cross-boundary covariance from pre × post, so no game-noise model is needed.
- A first version corrected with wp(1 − wp) noise and gave impossible values. In-sample residuals are about 2% less noisy than wp(1 − wp), which is as large as the signal here.

| heroes | cells | var pre (pp²) | var post (pp²) | cov (pp²) | stability | excess change var (pp²) | rank corr (raw) |
|---|---|---|---|---|---|---|---|
| unchanged | 14,588 | 23.8 | 10.6 | 10.8 | 0.68 | 12.9 | 0.056 |
| changed (notes) | 11,569 | 16.4 | 12.2 | 9.4 | 0.67 | 9.8 | 0.046 |
| changed (detector) | 1,522 | 24.0 | 1.8 | 7.0 | (noisy) | 11.9 | 0.048 |

- Notes-changed minus unchanged: stability −0.01 (−0.44, +0.34), excess change variance −3.1 pp² (−12.4, +10.3).
- Detector-changed minus unchanged: excess change variance −1.0 pp² (−24, +26).
- With at least 20 games on each side, the notes difference is −4.2 pp² (−18.4, +9.9).
- Raw rank correlations are near zero everywhere, because 10 to 20 games per side is mostly noise.

**B. Magnitude.** This is the one signal. Cells are grouped by the hero's aggregate win-rate change |ΔWR| across the boundary (all players):

| aggregate shift | cells | stability | excess change var (pp²) |
|---|---|---|---|
| < 1pp | 18,991 | 0.71 | 9.6 |
| 1-2pp | 5,297 | 0.73 | 9.6 |
| 2-4pp | 1,937 | 0.23 | 32.4 |
| 4pp+ | 135 | 0.32 | 51.1 |

The slope is +10.1 pp² of excess change variance per pp of |ΔWR| (1.8, 19.4). Patch-note labels carry nothing once magnitude is known: notes-changed heroes with shifts under 2pp look like unchanged heroes. Reworks are not labeled in the ground truth, so the rework vs number-tweak question cannot be tested directly.

**C. Jump model.** The state-space model from P3_SKILL_DRIFT.md was given an extra jump variance J on a hero's hero-specific state when the player's history crosses a boundary where that hero changed. Other parameters are fixed at the drift fit; the sample is the random 40k players' E-window games. The log-likelihood change vs no jump:

| jump sd | 0.5pp | 1pp | 2pp | 3pp | 5pp |
|---|---|---|---|---|---|
| notes-changed heroes (all boundaries) | −4.2 | −22.1 | −147.0 | −456.6 | |
| detector-changed heroes | −0.4 | −1.5 | −7.5 | −21.3 | |
| placebo: unchanged heroes | −5.7 | −31.4 | −217.4 | −680.1 | |
| heroes with \|ΔWR\| ≥ 2pp (178 boundary-hero pairs) | | −2.6 | −17.9 | −61.1 | −275.9 |
| placebo: 178 pairs with \|ΔWR\| < 1pp | | −3.1 | −15.7 | −45.2 | −184.3 |
| heroes with \|ΔWR\| ≥ 4pp (31 pairs) | | −0.4 | −2.2 | −7.1 | −33.3 |
| placebo: 31 pairs | | −0.4 | −2.3 | −6.9 | −29.0 |

- Maximum-likelihood J sits at the lower bound for the random and heavy samples and all four hero sets.
- On the big-shift heroes the jump fits exactly as badly as on placebo heroes.
- The split-half magnitude trend in B does not show up as a jump in the full model, which uses every game and separates player form from hero skill. I read B as a weak, likely noisy hint (few cells, mostly low-volume heroes), not a finding.

**D. After the 2026-04-20 patch (2.55.16, inside V2).** Posteriors were frozen the day before the patch and compared with each cell's mean over the next 30 days (cells with 10+ games):

- **Patched heroes:** coverage of the 80% intervals is 0.79 with variance ratio 1.41 (1,209 cells).
- **Unpatched heroes:** coverage is 0.80 with variance ratio 1.11 (3,006 cells).

Patched heroes carry a little more post-patch error, which fits the magnitude hint. Coverage is fine without a jump term. The fitted jump is zero, so the jump-aware model's log loss and coverage are identical to the base model's.

**For the merged paper.** Patches move a hero's strength for everyone; the WP absorbs that. They do not detectably reorder which players are good on the hero. The per-player layer can carry skill across patches unchanged.

## H2. Forced roles (`p3_ph_role.py`)

**Role familiarity.** For each slot, the share of the player's earlier games in this hero's Blizzard role and fine role. Earlier games are counted in play-time order (game_date, then replay_id). "Established" means 50+ earlier games in the window; "off-role" means a share under 10%.

**How often.** 9.6% of established slots are off-role by Blizzard role and 17.5% by fine role. By the role being played:

| role played | tank | bruiser | healer | ranged assassin | melee assassin | support |
|---|---|---|---|---|---|---|
| off-role share | 10.3% | 11.7% | 10.0% | 1.9% | 32.0% | 63.4% |

**Role effect or unfamiliar hero?** Mean residual (pp) by role share within bins of games on this hero, 2025-26, established players (Blizzard role). The first two rows are raw residuals vs the WP; the last two are after the skill model.

| games on hero | role share < 5% | 5-10% | 10-20% | 20-35% | 35-50% | 50%+ |
|---|---|---|---|---|---|---|
| raw, 5-19 | −3.8 | −1.6 | −0.4 | +0.3 | +0.6 | +0.2 |
| raw, 20+ | −1.3 | −1.1 | +0.4 | +1.5 | +2.1 | +2.3 |
| after model, 5-19 | −2.5 | −1.3 | −0.5 | −0.1 | 0.0 | −0.5 |
| after model, 20+ | −1.4 | −1.6 | −0.9 | −0.4 | −0.1 | −0.1 |

Even with 20+ games on the hero, a player who rarely plays that role underperforms the model by 1.4 to 1.6pp. It is a role effect, not only an unfamiliar-hero effect.

**Penalty** (off-role < 10% minus on-role ≥ 35%, averaged over hero-familiarity bins):

| | raw | after skill model |
|---|---|---|
| Blizzard role, 2025-26 | −1.7 (±0.1) | −0.9 (±0.1) |
| Blizzard role, post-cutoff | −1.4 (±0.2) | −0.6 (±0.2) |
| fine role, 2025-26 | −1.7 (±0.1) | −0.9 (±0.1) |
| fine role, post-cutoff | −1.7 (±0.2) | −0.8 (±0.2) |

**By the role being played** (after model, 2025-26):

| role | ranged assassin | support | tank | healer | bruiser | melee assassin |
|---|---|---|---|---|---|---|
| penalty | −2.6 (±0.3) | −1.7 (±0.9) | −0.9 (±0.2) | −0.5 (±0.2) | −0.5 (±0.2) | −0.3 (±0.6) |

The roles players complain about being forced into (healer, tank) carry modest penalties. The largest per-slot penalty is for non-assassin players on a ranged assassin, which is rare (1.9% of ranged-assassin slots).

**By pick position within the team** (after model):

| team pick | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| off-role share | 5.6% | 7.4% | 9.5% | 11.7% | 13.1% |
| penalty per off-role slot (pp) | −1.0 (±0.2) | −1.2 (±0.2) | −0.8 (±0.2) | −0.9 (±0.2) | −1.2 (±0.2) |

Forcing shows in frequency: last picks are off-role 2.3 times as often. The per-slot cost does not grow at later picks, so a forced last pick costs about the same as a voluntary off-role first pick.

A side finding for the WP: the mean model residual rises with pick position, from −0.8pp at a team's first pick to +0.2pp at its last. The WP overrates early picks slightly, or undervalues the last pick's counter-pick information.

**Team level** (combiner on logit WP + skill model, fit on V1, test on V2; 72% of V2 teams have no off-role player, 24% one, 3.7% two or more):

| added to WP + skill model | extra log-loss gain (95% CI) | coefficient |
|---|---|---|
| off-role count, Blizzard role | +0.00024 (0.00010, 0.00038) | −0.049 per player |
| off-role count, fine role | +0.00044 (0.00026, 0.00062) | −0.052 per player (about −1.3pp win prob) |
| log role share, Blizzard | +0.00028 (0.00009, 0.00049) | |
| log role share, fine | +0.00042 (0.00015, 0.00065) | |
| all role terms | +0.00052 (0.00024, 0.00078) | |

Alone (without the skill model), the off-role count is worth +0.0013 and log role share +0.0030. Most of that is already captured by the model's per-player role terms and experience offset.

**For the product.** A role-familiarity term is worth adding to the draft-time estimate. It is small in log loss but it is a visible, explainable cost: about 1pp per off-role player. The per-player role blocks in the skill model already shrink toward it. The remaining penalty is the part the model cannot see because the player has too few games in that role.

## Files (training/personalization/)

- `p3_ph_patch.py` (A-D; run with `--ab-only` for the split-half A/B and the jump profile): `results/p3_ph_patch.json/.txt`, `results/p3_ph_patch_ab.json/.txt`.
- `p3_ph_patch_big.py`: jump on big-shift heroes vs placebo (`results/p3_ph_patch_big.json/.txt`).
- `p3_fetch_pickorder.py`, `p3_fetch_gametime.py`: pick order and game timestamps (`cache/pickorder_2024q2.npz`, `cache/gametime_2024q2.npz`).
- `p3_ph_role.py`: forced-role analysis (`results/p3_ph_role.json/.txt`; caches lag-1 static predictions in `cache/sd_pred_static_lag1.npz`).

Note: `results/p3_ph_patch.json` sections A and B come from the first (noise-model) version and are superseded by `p3_ph_patch_ab.json`. Sections C and D there are current.
