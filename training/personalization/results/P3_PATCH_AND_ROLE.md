# P3: do patches reshuffle hero skill, and do forced roles cost games? (2026-09-30)

Two hypotheses from Max, tested on the same data as P3_SKILL_DRIFT.md: the snapshot, Storm League games from 2024-04 to 2026-05-22. Residuals are taken against the drift-aware WP and adjusted for experience. The fit window is V1 and the test window is V2 (72,474 games). All runs used 4 cores, no GPU.

Revised 2026-10-01 after the consolidated audit; changes are logged in `P3_AUDIT_FIXES.md`.

## Verdicts

- **H1, patches move individual hero skill in a jump:** no patch-specific jump detected; jumps up to about 1.25pp sd cannot be ruled out.
  - Split-half DiD (in sample, all 27 boundaries): stability of player skill across a boundary is 0.67 on patched heroes vs 0.68 on unpatched ones, difference −0.01 (95% CI −0.44 to +0.34). The same DiD on the two 2026 boundaries (out of sample): +0.65 (−0.28, +2.45). Neither excludes a sizable difference.
  - Jump model on the demeaned residual with a fitted noise scale, scored on the fit sample, on held-out players and out of time. Small jumps (0.5 to 1pp sd) raise the likelihood a little on patched heroes, and by as much or more on two placebos: unchanged heroes matched by role and volume at the same boundaries, and the same heroes at a date with no patch. Out of time, a 1pp jump lowers the likelihood for the real and placebo sets alike.
  - The best-fitting jump for notes-changed heroes is 0.76pp sd, with a 95% profile upper bound of 1.25pp (1.75pp for detector-flagged heroes and for heroes whose aggregate win rate moved 2pp or more). Hero-specific skill has an sd of about 2pp, so the bound still allows a patch to reshuffle up to about 40% of hero-specific variance.
  - The one hint: players reshuffle more on heroes whose aggregate win rate moved 2pp or more (B). Out of sample it points the same way (stability 0.17 vs 0.54 on unchanged heroes) but is not significant (difference −0.37, CI −1.03 to +0.68).
- **H2, off-role play is associated with losing beyond the skill model:** yes, about 0.5 to 0.8pp per off-role slot in the post-cutoff window.
  - Off-role means a role that makes up under 10% of the player's earlier games. 9.7% of established players' slots are off-role by Blizzard role, 17.7% by fine role.
  - Raw penalty: −1.3pp (SE 0.20) by Blizzard role and −1.7pp (SE 0.21) by fine role, post-cutoff, after controlling for games on the hero.
  - After the skill model (per-player role, fine-role and hero terms, plus the experience offset), post-cutoff: −0.54pp (SE 0.20) by Blizzard role and −0.77pp (SE 0.22) by fine role. SEs are player-cluster bootstrap; game clustering gives 0.19 and 0.21. In V2 alone: −0.48 (SE 0.28) and −0.53 (SE 0.33). The 2025-26 window, in sample for the WP, gives −0.93 (SE 0.09) and −0.89 (SE 0.09).
  - This is an association. Players choose when to go off-role, and the comparison does not separate a role cost from whatever leads players onto unfamiliar roles.
  - Last picks are off-role 2.3 times as often as first picks (13.2% vs 5.8%). The penalty per off-role slot does not differ by pick position.
  - At team level each off-role player (fine role) is associated with about 1.4pp lower win probability beyond the WP plus skill model. Adding role-familiarity terms improves held-out log loss by +0.0005 (0.0003, 0.0008).

## H1. Patches and individual skill (`p3_ph_patch.py`, `p3_ph_patch_big.py`)

**Boundaries.** I used the 27 sizable build boundaries from paper 2's detector study. Two definitions of a "changed" hero at each boundary:

- **notes:** the patch notes (bug-fix-only and ARAM-only mentions excluded);
- **detector:** heroes the win-rate detector flagged (wr_bh).

All other heroes count as unchanged.

**Population removal.** Each residual has its (build, hero) mean subtracted, so a hero's patch-wide shift is removed, including the WP's one-build lag in catching it. Only the player-specific part is tested.

**Cells.** A cell is a (boundary, player, hero) with at least 10 games on each side, within 90 days of the boundary.

**A. Difference-in-differences** (split-half design, player-cluster bootstrap; lag-1 contract, `results/fix/p3_ph_patch_ab.json`):

- Each side's games are split into odd and even games in play order. Variance comes from odd × even cross products and cross-boundary covariance from pre × post, so no game-noise model is needed.
- This avoids a game-noise model. In-sample residuals are about 2% less noisy than wp(1 − wp), which is as large as the signal here.

| heroes | cells | var pre (pp²) | var post (pp²) | cov (pp²) | stability | excess change var (pp²) | rank corr (raw) |
|---|---|---|---|---|---|---|---|
| unchanged | 14,588 | 24.2 | 10.7 | 10.9 | 0.68 | 13.1 | 0.056 |
| changed (notes) | 11,569 | 16.7 | 12.2 | 9.5 | 0.66 | 9.9 | 0.046 |
| changed (detector) | 1,522 | 24.1 | 1.8 | 7.0 | (noisy) | 12.0 | 0.047 |

- Notes-changed minus unchanged: stability −0.01 (−0.44, +0.34), excess change variance −3.1 pp² (−12.6, +10.3).
- Detector-changed minus unchanged: excess change variance −1.1 pp² (−24, +26).
- With at least 20 games on each side, the notes difference is −4.2 pp² (−18.3, +10.0).
- Raw rank correlations are near zero everywhere, because 10 to 20 games per side is mostly noise.

**B. Magnitude.** This is the one signal. Cells are grouped by the hero's aggregate win-rate change |ΔWR| across the boundary (all players):

| aggregate shift | cells | stability | excess change var (pp²) |
|---|---|---|---|
| < 1pp | 18,991 | 0.71 | 9.8 |
| 1-2pp | 5,297 | 0.73 | 9.9 |
| 2-4pp | 1,937 | 0.24 | 32.5 |
| 4pp+ | 135 | 0.32 | 51.0 |

The slope is +10.1 pp² of excess change variance per pp of |ΔWR| (1.8, 19.4). Patch-note labels carry nothing once magnitude is known: notes-changed heroes with shifts under 2pp look like unchanged heroes. Reworks are not labeled in the ground truth, so the rework vs number-tweak question cannot be tested directly.

**C. Jump model (`p3_fix_jump.py`).** The state-space model from P3_SKILL_DRIFT.md gets an extra jump variance J on a hero's hero-specific state when the player's history crosses a boundary where that hero changed. Other parameters are fixed at the drift fit.

- Residual: u = r_adj − mean over (build, hero), with r_adj on the lag-1 count contract.
- Noise: c · wp(1 − wp), c profiled on the fit sample: log-likelihood +7.7, +265, +256, 0 and −483 at c = 0.94, 0.96, 0.98, 1.00 and 1.02. c = 0.96 is used.
- Samples: fit = 40,000 fit-half players' E-window games; held-out players = 40,000 other players' E-window games; out of time = held-out players' full history, scored only on post-cutoff games within 60 days after the 2026 boundaries (2.55.15, 2.55.16).
- Placebos (no patch, same model): (a) matched: for each changed hero at a boundary, the unchanged hero of the same Blizzard role with the closest pre-boundary volume; (b) time: the changed heroes with the jump moved to a build half-way through the pre-boundary segment.

Log-likelihood change vs no jump (pairs = boundary-hero pairs):

| set (pairs) | J = 0.5pp fit / held-out / OOT | J = 1pp fit / held-out / OOT | J = 2pp fit / held-out / OOT |
|---|---|---|---|
| notes-changed (528) | +1.7 / +2.7 / +0.0 | +1.3 / +4.7 / −0.6 | −55 / −51 / −10.7 |
| matched placebo (364) | +1.2 / +1.8 / +0.1 | +1.5 / +3.6 / −0.3 | −32 / −30 / −8.1 |
| time placebo (215) | +0.5 / +1.2 / −0.0 | +1.2 / +3.6 / −0.4 | −8.4 / −1.5 / −4.7 |
| detector-changed (46) | −0.1 / −0.1 / +0.0 | −0.4 / −0.7 / +0.0 | −3.3 / −4.3 / −0.2 |
| matched placebo (46) | +0.2 / −0.0 / +0.0 | +0.7 / −0.2 / +0.0 | +1.6 / −1.9 / −0.1 |
| \|ΔWR\| ≥ 2pp (178) | +0.2 / −0.1 / +0.1 | +0.2 / −1.1 / +0.4 | −6.8 / −12.3 / +0.2 |
| matched placebo (176) | +0.5 / +0.2 / +0.4 | +1.3 / −0.2 / +1.3 | −4.0 / −10.5 / +3.1 |

- Per pair, the real sets gain no more than their placebos at any J and in any sample. A small jump term helps everywhere a little, which points to drift the OU model misses rather than to patches.
- Fit-sample maximum and 95% profile upper bound on J: notes 0.76pp (gain 2.4 units), bound 1.25pp; detector 0.0pp, bound 1.75pp; |ΔWR| ≥ 2pp 0.77pp (gain 0.3), bound 1.75pp.
- Reading: no patch-specific jump is detected in or out of sample. The data do not exclude jumps of 1 to 1.75pp sd, which is a sizable share of the 2pp hero-specific spread. What the data support is "no sign that patches reset skill, and a moderate reset would not have been visible".
- The 2026 DiD (split-half, out of sample): notes-changed minus unchanged stability +0.65 (−0.28, +2.45), excess change variance −23 pp² (−50, +12); |ΔWR| ≥ 2pp minus unchanged stability −0.37 (−1.03, +0.68), excess change variance +11.5 pp² (−27, +53).

**D. After the 2026-04-20 patch (2.55.16, inside V2).** Posteriors were frozen the day before the patch and compared with each cell's mean over the next 30 days (cells with 10+ games):

- **Patched heroes:** coverage of the 80% intervals is 0.79 with variance ratio 1.41 (1,209 cells).
- **Unpatched heroes:** coverage is 0.80 with variance ratio 1.11 (3,006 cells).

Patched heroes carry more post-patch error (variance ratio 1.41 vs 1.11; no CI computed), which fits the magnitude hint and is compatible with a moderate jump. Coverage is fine without a jump term.

**For the merged paper.** Patches move a hero's strength for everyone; the WP absorbs that. We find no sign that they reorder which players are good on the hero, but the test could not detect a jump below about 1.25pp sd. The per-player layer can carry skill across patches for now; a jump term of up to about 1pp sd on heroes with large aggregate shifts would cost little if later data support it.

## H2. Forced roles (`p3_ph_role.py`)

**Role familiarity.** For each slot, the share of the player's earlier games in this hero's Blizzard role and fine role. Earlier games are counted in play-time order (game_date, then replay_id). "Established" means 50+ earlier games in the window; "off-role" means a share under 10%.

**How often.** 9.7% of established slots are off-role by Blizzard role and 17.7% by fine role (lag-1 rerun). By the role being played:

| role played | tank | bruiser | healer | ranged assassin | melee assassin | support |
|---|---|---|---|---|---|---|
| off-role share | 10.4% | 11.9% | 10.1% | 2.0% | 32.4% | 63.9% |

**Role effect or unfamiliar hero?** Mean residual (pp) by role share within bins of games on this hero, 2025-26, established players (Blizzard role). The first two rows are raw residuals vs the WP; the last two are after the skill model.

| games on hero | role share < 5% | 5-10% | 10-20% | 20-35% | 35-50% | 50%+ |
|---|---|---|---|---|---|---|
| raw, 5-19 | −3.8 | −1.6 | −0.4 | +0.3 | +0.6 | +0.2 |
| raw, 20+ | −1.3 | −1.1 | +0.4 | +1.5 | +2.1 | +2.3 |
| after model, 5-19 | −2.5 | −1.3 | −0.5 | −0.1 | 0.0 | −0.5 |
| after model, 20+ | −1.4 | −1.6 | −0.9 | −0.4 | −0.1 | −0.1 |

Even with 20+ games on the hero, a player who rarely plays that role underperforms the model by 1.4 to 1.6pp. It is a role effect, not only an unfamiliar-hero effect.

**Penalty** (off-role < 10% minus on-role ≥ 35%, averaged over hero-familiarity bins; pp, with SE). The raw column's SE is analytic iid; the after-model column gives player-cluster bootstrap SEs (game-cluster SEs are within 0.01 of them). 2025-26 is in sample for the WP.

| | raw | after skill model |
|---|---|---|
| Blizzard role, post-cutoff | −1.3 (SE 0.20) | **−0.54 (SE 0.20)** |
| fine role, post-cutoff | −1.7 (SE 0.21) | **−0.77 (SE 0.22)** |
| Blizzard role, V2 only | | −0.48 (SE 0.28) |
| fine role, V2 only | | −0.53 (SE 0.33) |
| Blizzard role, 2025-26 | −1.6 (SE 0.09) | −0.93 (SE 0.09) |
| fine role, 2025-26 | −1.7 (SE 0.09) | −0.89 (SE 0.09) |

**By the role being played** (after model, 2025-26, in sample; analytic iid SEs, which the cluster check above shows are about right at this window):

| role | ranged assassin | support | tank | healer | bruiser | melee assassin |
|---|---|---|---|---|---|---|
| penalty | −2.6 (SE 0.3) | −1.7 (SE 0.9) | −0.9 (SE 0.2) | −0.6 (SE 0.2) | −0.5 (SE 0.2) | −0.3 (SE 0.6) |

The roles players complain about being forced into (healer, tank) carry modest penalties. The largest per-slot penalty is for non-assassin players on a ranged assassin, which is rare (1.9% of ranged-assassin slots).

**By pick position within the team** (after model):

| team pick | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| off-role share | 5.8% | 7.6% | 9.6% | 11.8% | 13.2% |
| penalty per off-role slot (pp) | −1.0 (SE 0.24) | −1.2 (SE 0.21) | −0.8 (SE 0.20) | −0.8 (SE 0.18) | −1.2 (SE 0.17) |

Forcing shows in frequency: last picks are off-role 2.3 times as often. The per-slot association does not grow at later picks, so a late, more likely forced off-role pick does no worse than an early, more likely voluntary one.

A side finding for the WP: the mean model residual rises with pick position, from −0.8pp at a team's first pick to +0.2pp at its last. The WP overrates early picks slightly, or undervalues the last pick's counter-pick information.

**Team level** (combiner on logit WP + skill model, fit on V1, test on V2; 72% of V2 teams have no off-role player, 24% one, 3.7% two or more):

| added to WP + skill model | extra log-loss gain (95% CI) | coefficient |
|---|---|---|
| off-role count, Blizzard role | +0.00024 (0.00008, 0.00039) | −0.053 per player |
| off-role count, fine role | +0.00047 (0.00027, 0.00067) | −0.055 per player (about −1.4pp win prob) |
| log role share, Blizzard | +0.00027 (0.00008, 0.00048) | |
| log role share, fine | +0.00044 (0.00016, 0.00068) | |
| all role terms | +0.00055 (0.00027, 0.00080) | |

Alone (without the skill model), the off-role count is worth +0.0013 and log role share +0.0027. Most of that is already captured by the model's per-player role terms and experience offset.

**For the product.** A role-familiarity term is worth adding to the draft-time estimate. It is small in log loss but visible and explainable: players on an off-role hero win about 0.5 to 0.8pp less than the model expects (post-cutoff), about 1.4pp per off-role player at team level. Present it as what such players have averaged. The per-player role blocks in the skill model already shrink toward it. The remaining penalty is the part the model cannot see because the player has too few games in that role.

## Files (training/personalization/)

- `p3_ph_patch.py`: A, B (`--ab-only`, rerun on the lag-1 contract: `results/fix/p3_ph_patch_ab.json`) and D (`results/p3_ph_patch.json`, section `oos`).
- `p3_fetch_pickorder.py`, `p3_fetch_gametime.py`: pick order and game timestamps (`cache/pickorder_2024q2.npz`, `cache/gametime_2024q2.npz`).
- `p3_ph_role.py`: forced-role analysis (`results/p3_ph_role.json/.txt`; caches lag-1 static predictions in `cache/sd_pred_static_lag1.npz`). Rerun on the lag-1 count contract: `results/fix/p3_ph_role.json` (cache `cache/fix_sd_pred_static_lag1.npz`).
- `p3_fix_jump.py`: the jump model of section C and the 2026 DiD (`results/fix/p3_fix_jump.json`). `p3_fix_arms.py role`: cluster SEs for H2 (`results/fix/p3_fix_role.json`).

The jump sections of `results/p3_ph_patch.json`, `p3_ph_patch_ab.json` and `p3_ph_patch_big.json` are replaced by `results/fix/p3_fix_jump.json`; see `P3_AUDIT_FIXES.md`.
