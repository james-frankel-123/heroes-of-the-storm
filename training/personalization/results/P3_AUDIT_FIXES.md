# P3 audit fixes (2026-10-01)

This file records what changed in the paper-3 (personalization) write-ups after the consolidated audit of 2026-10-01 (`audits/CONSOLIDATED_AUDIT_2026-10-01.md`, commit 4aff9b0, items P3-01 to P3-26). For each item: what was wrong, what was done, and every number that moved, old → new. All new code is in new files (`training/personalization/p3_fix_*.py`); no existing script was edited, so every pre-audit result can still be reproduced. New outputs are in `results/fix/` and `cache/fix/`.

Policy (Max, 2026-10-01): implementation bugs are fixed and rerun, and the write-ups report the corrected results without describing the bug as a caveat. The write-ups therefore carry only a one-line pointer to this file; this file is the record of what changed.

Resources: at most 4 CPU cores (`nice -n 19 taskset -c 48-63`, threads capped), at most one GPU process on GPU 3 and only when fewer than 4 HotS GPU processes were running. DB read-only. Snapshot data and builds up to 2.55.17.98025 only.

## One count contract (P3-03, P3-04)

**Problem.** Phase 2 counted a player's earlier games with same-day games in replay_id order, which is upload order, not play order. The drafter combiner (b2 = 3.71) was fit on those counts and served lag-1 counts. Role counts were ordered by game end time.

**Fix.** `p3_fix_counts.py`: every count is the number of games on earlier days (lag 1), the experience table is refit on those counts (`prepare_l1`), and role familiarity uses the same rule. Verified against a direct loop on 2,000 random rows; 47.6% of rows had a different count under the old rule. Phase-2 scripts were rerun through `run_fixed` (monkeypatched, outputs in `results/fix/`).

| number | old | new | file |
|---|---|---|---|
| skill model gain, phase 2 (state-space / EB static) | +0.01556 / +0.01557 | +0.01422 / +0.01423 | SKILL_DRIFT |
| paired state-space − static | −0.00001 (−0.00018, +0.00017) | −0.00001 (−0.00017, +0.00017) | SKILL_DRIFT |
| 365-day decay | +0.01546 | +0.01410 | SKILL_DRIFT |
| experience offset only (phase 2) | +0.0098 | +0.0083 | SKILL_DRIFT |
| phase-1 headline (table refit) | +0.0142 (59.95%, 0.66229) | +0.0142 (60.02%, 0.66224) | HERO_STRENGTH |
| experience offset only (phase 1) | +0.0082 | +0.0083 | HERO_STRENGTH |
| experience table, 0 games seen | −2.8pp | −1.5pp | HERO_STRENGTH §7 |
| experience table, 300+ games, never played | −6.9pp | −6.6pp | HERO_STRENGTH §7 |
| VALIDITY skill gain, full / smurf games removed | +0.0156 / +0.0143 | +0.0142 / +0.0130 | VALIDITY |
| party terms on top of skill (partied players / all) | +0.00017 / +0.00025 | +0.00018 / +0.00025 | VALIDITY |
| drafter combiner b2 / b3 | 3.709 / −0.052 | 3.613 / −0.055 | DRAFTER, PERSONAL_GD |
| role: off-role share Blizzard / fine | 9.6% / 17.5% | 9.7% / 17.7% | PATCH_AND_ROLE |
| role: team-level off-role count (fine) | +0.00044, −0.052 per player | +0.00047 (0.00027, 0.00067), −0.055 | PATCH_AND_ROLE |
| momentum: all arms on top of skill | +0.0023 (0.0019, 0.0027) | +0.0025 (0.0020, 0.0029) | SKILL_DRIFT |
| momentum: EWMA residual 100 games | +0.00065 (0.00040, 0.00092) | +0.00071 (0.00045, 0.00098) | SKILL_DRIFT |
| uncertainty: mean × reliability (state-space) | +0.00014 (0.00005, 0.00022) | +0.00012 (0.00004, 0.00019) | SKILL_DRIFT |
| Max-style share of the skill gain | 70% (vs +0.0156) | 76% (+0.0108 vs +0.0142) | SKILL_DRIFT |
| coverage by recency, variance ratio overall (state-space / static) | 1.05 / 1.67 | 1.11 / 1.76 | SKILL_DRIFT |
| safe lagged MMR on top of skill, G = 1 / 7 / 30 | +0.0009 / +0.0009 / +0.0007 | +0.0008 / +0.0008 / +0.0006 | SKILL_DRIFT |

The phase-2 explanation "these gains are higher than phase 1 because the offset counts same-day games known at draft time" is withdrawn: the extra +0.0013 was the upload-order leak.

**P3-04 (availability).** Not rebuilt. "Earlier days" still means every eventually-uploaded game with an earlier date. The write-ups now say so (HERO_STRENGTH setup, SKILL_DRIFT and EXTENSIONS caveats, DRAFTER caveats) and cite the audit's measured cost (about 0.0004 on the headline). Production should filter on ingestion time.

**Same-day rows of SKILL_DRIFT §1b.** `p3_sd_order.py` and `p3_sd_session.py` were rerun on the lag-1 contract (drift hyperparameters from the original `p3_sd_fit.json`). The two same-day rows of the head-to-head table, which mixed same-day upload-order counts into the offset, were removed; 1b carries the same-day values.

| number | old | new |
|---|---|---|
| strict: static same-day vs lag 1 | +0.00070 (0.00060, 0.00079) | +0.00077 (0.00067, 0.00087) |
| strict: drift vs static, same-day | +0.00029 (0.00011, 0.00050) | +0.00037 (0.00018, 0.00057) |
| strict: drift same-day vs lag 1 | +0.00099 (0.00075, 0.00123) | +0.00113 (0.00088, 0.00138) |
| random 40k fast player, rid / strict | 2.82pp, 3.6 d, +24.1 / 2.61pp, 3.8 d, +16.7 | 2.91pp, 3.4 d, +26.8 / 2.74pp, 3.6 d, +19.1 |
| heavy 6k fast player, rid / strict | 1.04pp, 15h, +0.4 / 0.01pp | 1.14pp, 15h, +0.5 / 0.01pp |
| static / drift with same-day games (head-to-head) | +0.01641 / +0.01679 (upload order) | +0.01500 / +0.01536 (strict) |
| per-slot gain by session position, static a day ahead | 1.15 first game, 2.89 game 4+ | 1.11 first game, 1.45 games 2-3, 2.43 game 4+ |

The Kalman drift parameters (`p3_sd_kalman.py`, the component table of SKILL_DRIFT §1) were fit on residuals adjusted with the earlier table and were not refit, for the same reason as the phase-1 kernels below.

## P3-01, P3-15, P3-25: the drafters' realized-agreement check

**Problem.** The check ranks real picks by V, which contains the skill term s, and scores the team against y − WP_pop, which s was fit to predict. It re-tests the skill model. It had been called "real, model-free value" and had spread to the distilled prior and to EXTENSIONS §5(b) and §5(d). The bootstrap resampled the two teams of a game independently (P3-15), and one-step and MCTS used different agreement definitions (P3-25).

**Fix.** `p3_fix_realized.py`: one agreement definition for every drafter (team mean percentile of the real picks in the personalized and population rankings; personal component = the difference), the skill model's own predicted gap V − WP_pop for the same real draft, the remainder = realized − predicted, a game bootstrap, and the regression of realized on agreement and predicted gap. Every realized row in DRAFTER, PERSONAL_GD and EXTENSIONS is relabeled "team-level calibration of the skill model", and "model-free value" is deleted. The non-circular evidence kept: held-out prediction of real opponent actions (imitation, personal GD), the calibration slope of the value function on held-out games, the EXTENSIONS §8 ban natural experiment, and the §2A/§3A adoption designs.

Pre-audit runs (`--old`, pre-audit tables and combiner as the predicted gap), Q5 − Q1 in pp:

| drafter | realized | predicted | remainder | predicted-gap slope |
|---|---|---|---|---|
| one-step, personalized agreement | +7.6 (5.1, 10.4) | +7.6 (7.0, 8.1) | +0.0 (−2.6, +2.8) | 0.98 (0.86, 1.11) |
| one-step, personal component (unified definition) | +9.0 (6.1, 11.7) | +10.2 | −1.2 (−4.0, +1.3) | |
| MCTS 400, BC prior | +4.7 (1.8, 7.3) | +3.2 | +1.6 (−1.4, +4.0) | 0.97 |
| MCTS 400, distilled prior (2,000 lobbies) | +5.1 (0.6, 10.1) | +5.7 | −0.6 (−5.3, +4.2) | 0.91 |
| greedy distilled, personal component | +12.8 (10.1, 15.5) | +10.3 | +2.5 (−0.2, +5.1) | 0.93 |
| MCTS 400, personal-GD opponents (2,000 lobbies) | +4.1 (−1.1, 9.0) | +4.8 | −0.7 (−5.7, +4.0) | 0.92 |

The audit's three reproductions (+7.64 raw, +0.05 remainder) match the first row. Fixed-protocol values are in the drafter section below.

## P3-02: drafter oracle information

**Problem.** Pools included the hero the player actually played (19.2% of slots had that hero added with no earlier games on it), and real bans were forced in the one-step, MCTS and distilled harnesses.

**Fix.** `p3_fix_drafters.py` patches the existing modules at run time: pools = heroes with games on earlier days (empty pool → any free hero; 2.5% of slots), no forced bans (one-step: every ban sampled from GD; MCTS: `lobby()["forced"] = −1`, so our bans are searched and the opponent's come from GD), lag-1 tables, the refit combiner, and lag-1 imitation features. Real-state rankings still add the real hero to the candidate list so it can be ranked (evaluation only). The pick order of players is kept: it is fixed by the lobby before the draft.

One-step (CPU, 1,000 full lobbies, 5,724 realized lobbies):

| number | old | new |
|---|---|---|
| picks that differ, vs GD / vs imitation / self-play | 52% / 53% / 53% | 51% / 51% / 52% |
| gain in V when they differ, vs GD | +3.6 (3.5, 3.7) | +3.4 (3.3, 3.5) |
| WP_pop change when they differ, vs GD | −3.0 | −2.9 |
| whole draft vs GD: gain in V / WP_pop | +5.1 (4.5, 5.9) / −3.1 | +4.9 (4.3, 5.6) / −2.7 |
| whole draft vs imitation: gain in V / WP_pop | +5.8 / −3.2 | +6.6 (5.9, 7.3) / −2.5 |
| mean V, personalized / population, vs GD | 0.733 / 0.682 | 0.738 / 0.689 |
| realized personalized agreement Q5 − Q1 | +7.6 (4.9, 10.5) | +8.5 (6.0, 11.2); predicted +7.1; remainder +1.5 (−1.1, +4.1) |
| realized personal component | +9.6 (6.7, 12.4) (rank of V − WP) | +11.3 (8.3, 14.2) (unified); predicted +9.8; remainder +1.5 (−1.3, +4.5) |
| predicted-gap slope, realized on predicted | (not computed) | 1.00 (0.88, 1.13) |
| collapse: effective pool pers / pop | 10.7 / 12.9 | 10.4 / 12.7 |
| collapse: top-3 share pers (mean) | 28% | 29% |
| self-play effective heroes pers / pop | 69.5 / 63.7 | 69.0 / 65.0 |
| self-play off-role pers / pop | 17.2% / 34.3% | 17.3% / 32.7% |
| real pick = imitation / personalized / population top-1 | 33.3% / 12.4% / 7.7% | 33.0% / 12.2% / 7.6% |
| pick classes: both / outcome only / imitation only / neither | +3.3 / +1.8 / +1.6 / −1.3 | +3.2 / +2.5 / +1.5 / −1.2 (relabeled calibration) |

MCTS, distilled prior and personal GD (GPU 3, one process; real-state searches on the first 2,000 of the 5,724 lobbies, 4,000 teams; the pre-audit MCTS real-state run used all 5,724):

| number | old | new |
|---|---|---|
| MCTS 400: picks that differ / gain when they differ / WP cost | 38% / +3.9 / −1.6 | 39% / +3.9 (3.7, 4.1) / −1.4 |
| MCTS 400 vs GD: gain in V / WP_pop / mean V pers, pop | +5.5 (4.7, 6.4) / −1.9 / 0.698, 0.643 | +6.2 (5.4, 6.9) / −1.1 / 0.687, 0.625 |
| MCTS 400 vs imitation: gain in V / WP_pop | +7.3 / −0.5 | +7.7 (6.9, 8.5) / −0.4 |
| MCTS 400 realized agreement, personalized / personal component | +4.7 / +5.2 | +4.1 / −0.1 realized; predicted +4.0 / +5.9; remainder +0.1 (−4.0, +4.9) / −6.0 (−11.1, −1.0) |
| MCTS self-play effective heroes pers / pop; correlation with real | 42.6 / 41.0; 0.82 / 0.79 | 41.3 / 38.6; 0.65 / 0.62 |
| MCTS off-role, self-play pers / pop | 18.3% / 32.0% | 20.5% / 33.6% |
| MCTS collapse effective pool / top-3 share | 8.9 / 26% | 8.9 / 25% |
| sims curve predicted gain, 25 / 100 / 400 | +2.2 / +2.9 / +5.5 | +2.1 / +4.2 / +6.2 |
| sims curve realized rows | presented as "realized value rises too" | calibration: 100 sims +0.4 realized, remainder −1.8 (−6.4, +2.8); 400 sims see above |
| distilled prior fit: KL / top-1 = MCTS / top-1 = real | 0.174 / 57.0% / 19.3% | 0.171 / 57.7% / 18.3% (log-lik of the real pick dropped: real picks outside the pool get no mass) |
| greedy distilled vs GD: gain in V / WP_pop | +6.1 / −0.2 | +4.6 (4.0, 5.2) / −0.5 |
| MCTS distilled vs GD: gain in V / WP_pop | +6.5 / −4.1 | +6.9 (6.1, 7.6) / −3.3 |
| distilled − BC prior MCTS, paired V / WP_pop | +1.0 / −2.2 | +0.7 (0.1, 1.3) / −2.2 |
| distilled MCTS self-play: effective heroes / off-role | 43.4 / 13.6% | 42.8 / 15.7% |
| distilled realized "+5.1 (0.0, 9.8)", "personal component +12.8 to +14.2 separates them most sharply", "+3.1 vs +0.1 at 100 sims" | presented as merit | calibration: realized matches predicted for every drafter; greedy personal component remainder +1.9 (−0.7, +4.9); MCTS distilled +4.2 (−1.1, +8.8); 100 sims +1.5 (−2.8, +6.3) vs BC −1.8 (−6.4, +2.8) |
| PGD: picks changed / bans changed by opponents as themselves | 24% / 30% | 24% / 29% |
| PGD: mean V R1 / R2; gain R1 / R2 / R3 | 0.691 / 0.616; +6.6 / +8.2 / +12.6 | 0.687 / 0.610; +6.2 / +7.2 / +12.1 |
| PGD ban value R1 / R2 / R3 (one-trick) | +0.53 (+0.62) / +0.63 (+0.83) / +0.83 (+1.53) | +0.51 (+0.62) / +0.62 (+0.78) / +0.82 (+1.49) |
| PGD R3 self-play effective heroes / off-role | 60.9 / 7.5% | 61.1 / 7.5% |
| sims curve 1,500: predicted gain / WP_pop | +7.2 / −3.3 | +7.7 (7.0, 8.4) / −2.7 |
| sims curve realized, 25 / 1,500 sims | −1.3 / +5.3 | −2.2 / +5.8 realized; predicted +0.1 / +6.6; remainder −2.3 (−6.9, +3.2) / −0.8 (−5.5, +3.8) |

**P3-08 (kernel horizon).** Fixed in the kernel sources by commits 1db7df8 and edaac2e (`cuda_personal`, `cuda_prior`, `cuda_pgd` and the ofit lineage), with builds verified but not yet installed. Next (priority 5, after the drift lane's rebuilt site-tier cumprev WP models are available as the P3 baseline): install the four rebuilt .so files, rerun `p3_mcts_verify`, and rerun every MCTS, distilled-prior and personal-GD result above on the fixed kernel. The MCTS rows above are the P3-02/P3-03 reruns on the earlier kernel builds and will be replaced then; the P3_DRAFTER wording on search depth will be corrected with them.

## Phase-1 scripts rerun on the lag-1 table

Under Max's policy (implementation bugs are fixed and rerun, not described as caveats), every phase-1 evaluation script was rerun on the lag-1 contract through `p3_fix_counts.run_fixed`: `p3_hs_eval` (with its kernels file's experience table replaced by the lag-1 refit, `cache/fix_hs_kernels_l1.npz`, and lag-1 counts), `p3_hs_nopool`, `p3_hs_paired`, `p3_hs_cassia`, `p3_hs_calib`, `p3_hs_display`, and `p3_ph_patch --ab-only`. Outputs: `results/fix/`. The write-ups now present these numbers directly; the "first version" remarks were removed from the write-ups and are recorded here.

| number (HERO_STRENGTH) | old | new |
|---|---|---|
| player / player+hero / CF rank 2 gain | +0.0108 / +0.0132 / +0.0142 | +0.0108 / +0.0133 / +0.0142 |
| sample-size table, CF per slot at n = 0 / all | 1.86 / 1.45 | 1.91 / 1.46 |
| CF slope at n = 0 | 1.32 | 1.35 |
| pooling, player+hero − no pooling at n = 0 | +0.50 (0.40, 0.60) | +0.48 (0.38, 0.59) |
| pooling, CF − no pooling at n = 0 | +0.57 (0.45, 0.70) | +0.55 (0.43, 0.68) |
| game level, CF − player+hero / player+hero − no pooling / CF − no pooling | +0.0010 / +0.0017 / +0.0027 | +0.0010 / +0.0016 / +0.0026 |
| decay 180 / 90 days | +0.0135 / +0.0126 | +0.0136 / +0.0128 |
| V1 per-slot, 365 days vs no decay | +1.291 vs +1.273 | +1.298 vs +1.279 |
| calibration scale c / never-played term | 1.20 / 7.04pp | 1.25 / 7.31pp (the latter not displayed, P3-05) |
| display sd at 1-2 / 21-50 / 200+ games | 4.2 / 3.6 / 2.4 | 4.3 / 3.7 / 2.4 |
| 80% band at 1-2 / 21-50 / 200+ games | ±5.4 / ±4.6 / ±3.1 | ±5.5 / ±4.7 / ±3.1 |
| Cassia: CF var ratio; k = 1 R²; raw revealed mean at k = 1 / 5 | 0.56; 0.22; −314 / −81 | 0.57; 0.23; −316 / −82 |
| example profile: Whitemane band; Ana | +0.8 to +8.3; +3.8pp | +0.7 to +8.3; +3.9pp |

| number (PATCH_AND_ROLE §A, B) | old | new |
|---|---|---|
| unchanged / notes cells: var pre, cov, excess | 23.8, 10.8, 12.9 / 16.4, 9.4, 9.8 | 24.2, 10.9, 13.1 / 16.7, 9.5, 9.9 |
| notes − unchanged excess change variance | −3.1 (−12.4, +10.3) | −3.1 (−12.6, +10.3) |
| magnitude bins excess, < 1pp / 1-2 / 2-4 / 4+ | 9.6 / 9.6 / 32.4 / 51.1 | 9.8 / 9.9 / 32.5 / 51.0 |

Not rerun: the kernel hyperparameters and CF factors (`p3_hs_fit.py fit`) were fit on residuals adjusted with the earlier table; refitting them would cascade through every result. The tables differ by at most 1.3pp in a cell and the headline moves by +0.00003 when only the table is refit. PATCH_AND_ROLE section D (`p3_ph_patch.py` full run, coverage after 2.55.16) was not rerun either. Both can be rerun on request.

P3_EXTENSIONS sections 1 to 4 and 6 to 8 were not rerun: their scripts already use lag-1 counts for every evaluation offset (the audit checked this), and the earlier table enters only through the residuals of earlier days, as for the kernels.

## P3-24: imitation features on the lag-1 contract

`p3_fix_counts.recency_features_l1` computes the EWMA shares and days-since from games on earlier days only; `p3_fix_drafters.patch_imitation` swaps it in and refits the imitation weights on V1 (`cache/fix/dr_imitation.npz`). Every fixed drafter run, the distilled prior's features and EXTENSIONS §5(d) use it. The personal GD features (`p3_pgd_feats.py`) were already strictly causal (games that ended before the draft started) and were not changed.

| imitation, V2 picks | old | new |
|---|---|---|
| top-1 / top-3 / top-5 / top-10 | 31.9 / 54.7 / 65.6 / 78.7% | 31.0 / 53.4 / 64.1 / 77.4% |
| log-lik | −2.61 | −2.67 |
| frequency alone top-1 | 26.8% | 25.9% |
| personal history alone top-1 | 27.0% | 26.1% |

## EXTENSIONS §5(b) and §5(d) (P3-01, P3-02, P3-24)

`p3_fix_ext5.py`: both rerun on the fixed one-step protocol with lag-1 imitation features; realized checks reported as calibration.

| number | old | new |
|---|---|---|
| (b) full information − population, V | +5.93 (5.28, 6.60) | +5.79 (5.14, 6.46) |
| (b) teammates only − population, V | +5.75 (4.99, 6.48) | +5.45 (4.74, 6.14) |
| (b) share of value kept | 97% (86%, 107%) | 94% (84%, 105%) |
| (b) realized, teammates-only / full agreement Q5 − Q1 | +6.4 (3.1, 9.9) / +5.7 (2.4, 9.2), as support | +6.5 / +8.5 realized; predicted +6.8 / +7.1; remainder −0.4 (−3.1, +2.3) / +1.5 (−1.2, +3.9) |
| (d) personal component per unit agreement | +12.8 (5.8, 18.9) | +22.8 without the predicted gap; +1.0 (−5.6, +8.7) with it |
| (d) imitation per unit agreement | +31.7 to +36.6 | +24.4 without the predicted gap; +11.4 (3.1, 19.0) with it |
| (d) predicted-gap slope | | 0.95 (0.82, 1.08) |
| (d) tuned λ | 0.002 (by matched residual) | 0.005 (by matched remainder) |
| (d) matched residual, outcome / tuned / imitation | +3.4 / +3.2 / +2.3 | +3.1 / +2.7 / +2.1; remainders +1.0 / +0.8 / +0.6, CIs overlap |

The §5 summary line "imitation +32pp next to +13pp for the personal term" becomes "imitation +11pp beyond the skill model's predicted gap; the personal term adds nothing beyond it". The recommendation (put recency and share into the value function, keep the outcome drafter as recommender) stands on the smaller number. §5(c) and §5(e) were not rerun; §5(c) is a calibration slope, which is the non-circular use.

## P3-05: never-played calibration

**Problem.** Coverage on never-played heroes was measured on cells with 10+ (or 30+) future games, which selects on early outcomes, and the target used each future row's offset rather than the prediction made before the first game. The same selected population gave EXTENSIONS §1's "0.82 on never-played heroes".

**Fix.** `p3_fix_neverplayed.py`: every V2 adoption event (first game on a hero, player with 20+ earlier games; 55,495 events, 20,967 players), scored on its first game with the prediction made before it, player-cluster CIs, and a noise reference from established cells of the same window.

| number | old | new |
|---|---|---|
| calibrated coverage, never-played | 0.79 (cells with 10+ future games) | not a valid test; withdrawn |
| EXT §1 OOT coverage, never-played | 0.82 | withdrawn as evidence (same selection) |
| first-game mean, realized / predicted | | −3.7 / −3.0pp; bias −0.74 (−1.13, −0.33) |
| first-game log loss, WP / + offset / + offset + GP | | 0.67617 / 0.67272 / 0.67200 |
| excess variance vs deep-cell reference, realized / claimed by display with the never-played term | | −14 ± 5 / +65 pp² |
| display rule | 1.20 s² + (7.0pp)² for never-played | 1.25 s² for every hero (calibration refit on the lag-1 contract); the (7.3pp)² term describes players who keep a hero and is not displayed |
| example never-played band | −17 to +3 around −7pp | about ±6pp around −7pp |
| events reaching 3 games: first-game residual | | +7.0pp (rest −5.3pp): the selection that inflated the old band |

## P3-06: causal wording

"A never-played hero costs about 7pp for a veteran" (HERO_STRENGTH), "the first game on a new hero costs about 10pp ... free of selection" (EXT §3A) and "rust costs 1.5 to 2.2pp" (EXT §3B) are restated as averages for players with that history ("players with 300+ games average −6.6pp"), with the selection on hero choice and timing stated. The EXT §8B ban effect keeps "costs" (within-player design with a placebo).

## P3-07: MMR

**Problem.** HERO_STRENGTH's summary still quoted the leaky lagged MMR (60.81%, +0.0201; hero MMR +0.0088).

**Fix.** `p3_fix_mmr.py`: causal at-game MMR (P3_MMR_AT_GAME), same game-level protocol.

| number | old | new |
|---|---|---|
| hero MMR alone | +0.0088 (lagged, leaky) | +0.0025 (0.0021, 0.0030) |
| player / role MMR alone | +0.0043 / +0.0054 (leaky) | +0.0005 / +0.0011 |
| skill + all three MMR | +0.0201 (60.81%) | +0.0149 (60.09%); +0.0006 (0.0004, 0.0009) over skill |
| skill + hero MMR | | +0.0005 (0.0003, 0.0008) over skill |

## P3-08: kernel horizon

The personalized kernels expanded only on our own turns, so the search covered the current own pick block with GD rollouts beyond it. Max decided to fix it; the fix is in the kernel sources (see the drafter section above for status and the rerun plan). No numbers were changed for this item yet.

## P3-09: static and lagged arms

**Problem.** The static@cutoff and lag-7/30 arms reused the lag-1 offset.

**Fix.** `p3_fix_arms.py arms` builds each arm with its own counts.

| arm | old | new |
|---|---|---|
| online, lag 1 | +0.0142 | +0.0142 (replay CI 0.0131, 0.0152; one-way player cluster 0.0131, 0.0154) |
| static at cutoff (state and counts frozen) | +0.0107 | +0.0059 (0.0051, 0.0067) |
| state frozen, counts current | (was labeled static) | +0.0106 (0.0097, 0.0115) |
| lag 7, state and counts | +0.0126 | +0.0112 (0.0102, 0.0121) |
| lag 30, state and counts | +0.0116 | +0.0083 (0.0075, 0.0091) |
| "updating daily adds" | +0.0035 | +0.0083 all updates; +0.0036 (0.0031, 0.0042) GP state with live counts; +0.0047 counts alone |
| lag cost, 7 / 30 days | 0.0016 / 0.0026 | 0.0031 / 0.0059 |

## P3-10: noise model

Not refit everywhere. The write-ups now state that variance fits assume wp(1 − wp) and that in-sample residuals are 2 to 4% less noisy (HERO_STRENGTH caveats, SKILL_DRIFT caveats). The jump redo profiles the noise scale (best 0.96). The never-played variance test uses an empirical noise reference instead of wp(1 − wp).

## P3-11: split date and prevalence denominator

| number | old | new |
|---|---|---|
| V1 | 2026-02-10 to 2026-04-01 | 2026-02-10 to 2026-03-28 |
| V2 | 2026-04-01 to 2026-05-22 | 2026-03-29 to 2026-05-22 (median game day 20541) |
| smurf games, share | 1.2% of all games (÷ 1,949,087 WP index) | 2.24% of the 1,082,432 represented games |
| bias on other players | about −0.4pp | about −0.08pp mean (5/9 opponents × −32, 4/9 teammates × +32, × 2.24%); the issue is noise |

The game counts (70,974 and 72,474) were computed with the median split and were already right.

## P3-12: off-role headline and SEs

`p3_fix_arms.py role`: player- and game-cluster bootstrap SEs by window. The `penalty()` docstring in `p3_ph_role.py` says "bootstrap SE"; the code computes the iid analytic SE. The file was not edited (new files only); the write-up now labels the column correctly.

| off-role penalty after the skill model | old | new (SE, player cluster) |
|---|---|---|
| headline | −0.9pp (±0.1), 2025-26, "forced off-role play costs games" | post-cutoff −0.54 (0.20) Blizzard, −0.77 (0.22) fine; association |
| V2 only | | −0.48 (0.28), −0.53 (0.33) |
| 2025-26 (in sample) | −0.9 (±0.1) | −0.93 (0.09), −0.89 (0.09) |
| raw post-cutoff | −1.4 / −1.7 (±0.2) | −1.3 (0.20) / −1.7 (0.21) |

## P3-13: drift half-lives

`p3_fix_arms.py cf`: profile likelihood over the correlation half-life (random 40k sample).

| statement | old | new |
|---|---|---|
| hero-specific information | "halves in about 4 to 6 years" | drift not detected; rate at bound; 95% profile: half-life ≥ 2 years |
| CF factors | "4 to 7 years" (point fits 2,500 and 1,620 days, no SE) | best 8 years; 95% profile 3 years to no decay |
| state-space variance ratio | "1.05" | 1.11 overall; by days since played 0.62 / 0.37 / 0.68 / 0.89 / 1.11 (1-7 d to 181-365 d), 3.14 never played |

## P3-14: patches and individual skill

**Problem.** The jump filter was fed r_adj, not the demeaned u; the likelihood was in sample with the noise fixed; "any real jump is below about 1pp sd" overstated the test.

**Fix.** `p3_fix_jump.py`: demeaned u on the lag-1 contract, noise scale profiled (0.96), jump grid scored on the fit sample, on 40,000 held-out players and out of time (2026 boundaries 2.55.15 and 2.55.16, post-cutoff games), against matched and time placebos; profile bound on J; split-half DiD on the 2026 boundaries.

| number | old | new |
|---|---|---|
| verdict | "not supported"; "a jump fits worse at every size"; "any jump below about 1pp sd" | no patch-specific jump detected; jumps up to 1.25pp sd (notes) or 1.75pp (detector, \|ΔWR\| ≥ 2pp) not excluded |
| 1pp jump, notes heroes, log-lik | −22.1 (in sample, r_adj) | +1.3 fit / +4.7 held-out / −0.6 OOT; placebos +1.5 / +3.6 / −0.3 (matched) and +1.2 / +3.6 / −0.4 (time) |
| best J, notes | at the lower bound | 0.76pp (gain 2.4 units), 95% bound 1.25pp |
| 2026 DiD, notes − unchanged stability | | +0.65 (−0.28, +2.45) |
| 2026 DiD, \|ΔWR\| ≥ 2pp − unchanged stability | | −0.37 (−1.03, +0.68) |

## P3-16: recurring players

The headline CI is now given with a player-cluster version (one-way, the first team-0 player of each game: 0.0131 to 0.0154) and the audit's full design-effect estimate (about 1.7 in variance: 0.0127 to 0.0157). EXTENSIONS and SKILL_DRIFT headers say replay-bootstrap CIs should be widened by about 1.3x. The drafter calibration tables use game bootstraps.

## P3-17: decay choice

HERO_STRENGTH now says the no-decay choice was made on V2 and names the V1 choice, a 365-day half-life (rerun: 59.85%, +0.0141), as the pre-registered option.

## P3-18: what pooling buys

The summary credited +0.43 to +0.57 per 1,000 slots to "player-specific hero similarity". Restated: most of it is the player's overall level (+0.50 at n = 0, +0.23 to +0.38 at n = 1 to 10), and the learned similarity adds +0.07 (n.s.) at n = 0 and +0.16 to +0.20 at n = 1 to 10, +0.0010 at game level.

## P3-19: multiplicity

`p3_fix_arms.py misc`: Benjamini-Hochberg on the 90 per-hero transfer tests. Only Genji survives at q < 0.05 (Sylvanas, Medivh at q < 0.10); The Butcher q = 0.19; for skill beyond general level nothing survives (smallest q 0.43). HERO_SIMILARITY no longer names The Butcher or Qhira as significant.

## P3-20: smurf exclusion

VALIDITY's exclusion test is labeled retrospective and points to EXTENSIONS §4B (causal purification, −0.00001).

## P3-21

Disputed by the audit; nothing changed.

## P3-22, P3-23

Immaterial (audit: mean-of-logits vs mean-of-probabilities differ by 0.007pp on average; the `_safe_lag` cap touches 0.05% of slots in one superseded arm). Noted here; no change.

## P3-26: tier-label hazard

`p3_fix_tier_guard.py` checks a games file against the old research tier rule (stored league_tier = real + 1; low = Bronze, mid = Silver + Gold + Master, high = Platinum + Diamond) and exits 1 on any mismatch. The current caches pass. EXTENSIONS' rerun guide now warns against refetching without it.

## Files

New scripts (all in `training/personalization/`): `p3_fix_counts.py`, `p3_fix_drafters.py`, `p3_fix_realized.py`, `p3_fix_neverplayed.py`, `p3_fix_jump.py`, `p3_fix_arms.py`, `p3_fix_mmr.py`, `p3_fix_ext5.py`, `p3_fix_tier_guard.py`. Results: `results/fix/*.json` and `.txt`. Caches: `cache/fix/` (input files are symlinks to `cache/`; outputs are never linked).
