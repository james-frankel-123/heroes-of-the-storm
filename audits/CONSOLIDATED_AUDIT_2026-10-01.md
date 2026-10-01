# Consolidated audit fix list (2026-10-01)

This document merges six independent audits written on 2026-09-30 between 16:50 and 17:30. Every finding was rechecked against the repo at HEAD `0c72fb1` and its working tree on 2026-10-01.

| Audit | Scope | Auditor |
|---|---|---|
| `xaudit/codex_p3/AUDIT.md` | personalization (P3) | OpenAI Codex (ran code) |
| `xaudit/grok_p3/AUDIT.md` | P3 | xAI Grok (ran code) |
| `xaudit/gemini_p3/AUDIT.md` | P3 | Google Gemini (read code only) |
| `training/personalization/results/AUDIT_P3.md` | P3 | Claude subagent (ran code) |
| `xaudit/claude_p1/AUDIT.md` | paper 1 revision | Claude |
| `xaudit/claude_drift/AUDIT.md` | drift rebuild | Claude |

`xaudit/` is `/tmp/claude-1000/-home-max-heroes-of-the-storm/a49813e0-7f18-4a49-90f1-0426c355d609/scratchpad/xaudit/`. That is session scratch, so the audit files themselves may not persist.

**Status values.**
- **FIXED:** the commit or file that fixed it is cited.
- **STALE:** the text or result no longer exists.
- **STILL OPEN:** confirmed in the current code or text, with a quote or file:line, and reproduced where it was cheap.
- **DISPUTED:** checked, and the auditor is wrong.

**Severity:** A changes a claim, B wrong number or sentence, C reviewer-bait, D style.

**Owners:** P3 code/text, P1 text, drift text, production/code.

**How this was checked.**
- Read-only throughout. CPU checks were pinned to two cores (`taskset -c 48-49 nice -n 19`). DB access was read-only and mostly avoided.
- No game from a build after 2026-09-27 was used.
- Line numbers refer to the current files: `draft_revision.tex` (P1 "L"), `supplementary_revision.tex` (P1 "S"), and the drift `draft.tex`.

## 1. Counts

| Area | Findings | FIXED | STALE | STILL OPEN | DISPUTED | of which new since the audits |
|---|---|---|---|---|---|---|
| Personalization (4 audits, deduplicated) | 26 | 0 | 0 | 25 | 1 | 1 (P3-26) |
| Paper 1 | 67 | 5 | 7 | 55 | 0 | 4 (P1-N1..N4) |
| Drift paper | 43 | 1 | 0 | 42 | 0 | 3 (D-N1..N3) |
| Cross-cutting, not covered above | 2 | 0 | 0 | 2 | 0 | 2 (X2, X4) |
| **Total** | **138** | **6** | **7** | **124** | **1** | **10** |

STILL OPEN by severity:

| Area | A | B | C | D |
|---|---|---|---|---|
| P3 | 3 | 10 | 9 | 3 |
| P1 | 4 | 25 | 15 | 11 |
| Drift | 3 | 19 | 12 | 8 |
| Cross-cutting | 0 | 2 | 0 | 0 |
| **Total** | **10** | **56** | **36** | **22** |

How the P1 and drift rows were counted:
- **P1:** 51 audit items are still open, plus N1–N4. N1 and N3 are B, N2 and N4 are C.
- **Drift:** the 39 audit items include I1, the internal-consistency bullet that §4 of that audit raised. N2 is B; N1 and N3 are C.

**Why so little is fixed.**
- Between the audits and today, none of the audited P3 code or write-ups was touched. Everything new in P3 was added alongside the old work.
- The drift draft changed in one paragraph only (f02ff4c).
- Paper 1 changed the most. c159669 and 0bf37b2 fixed A3, B4, B5 and B6, and removed seven items. The A-level interpretation items were not touched.

**Agreement across model families.** Four findings were reached independently by at least three auditors from different model families. They are the most robust.

| Finding | Found by |
|---|---|
| P3-01: circular drafter realized value | all four (Codex, Grok, Gemini, Claude). Three reproduced the +7.6pp and got the same near-zero remainder. |
| P3-14: patch "no reset" overstated | Claude, Grok, Codex |
| P3-13: hero-specific half-life | Grok, Claude, Codex |
| P3-12: off-role headline and its SE | Codex, Grok, Claude |

Two cross-family pairs agree as well:
- **P3-09** (frozen arm reuses the lag-1 offset): Codex and Grok.
- **P3-03** (same-day counts): Codex and Claude. Gemini found the role-count ordering variant.

Gemini was the one dissenter on calibration (P3-05 "survives"), and it was wrong. Its unique B finding (P3-21) is DISPUTED.

## 2. Fix first

Ordered by blast radius. Each line points to the full entry below.

1. **X1 / P1 §0 (B, production/code then P1 text): research code reads the live `src/lib/data/compositions.json`, and the nightly sync rewrites it.**
   - Since 2026-10-01 00:02 the file holds patch 2.55 + 2.57 data; the sync log says "Fetching ... (2.55,2.57 ...)".
   - c159669's gN composition audit, gN refits and structure terms all read that file. So did the Table II gN recompute and the leak-free CQL (enr.) refs. That breaks paper 1's "no game from a build released after 2026-09-27" (L82).
   - J_oof s0/s1 and all A_partial/G_base benchmarks were scored with the 10-row table written at 09-30 16:52. G_base gN .636 should be about .638.
   - Fix:
     - pin a copy of the HEAD table, with a hash assert, for all research readers;
     - rerun the listed scorers;
     - then update the text.
2. **P3-01 (A, P3 text): the drafter's "realized value" re-tests the skill model.** Since the audits it has spread into the distilled-prior section of P3_DRAFTER.md and into P3_EXTENSIONS §5(b) and §5(d). Delete "model-free value" (`P3_DRAFTER.md:70`) and relabel every realized-agreement row as team-level calibration.
3. **P1 A1 / A2 / A4 / A5 (A, P1 text).**
   - J_oof landed in outcome (i) of the audit's decision rule: gN −0.007 ± 0.003, RN −0.008 ± 0.003. The search-depth gap is not caused by the leak, yet L35, L59, L273 and L329 still say it is.
   - Also: add the attenuation benchmark, report safety at matched simulations, and either disclose gN's external composition features or remove them.
4. **Drift A1–A3 (A, drift text).** The opponent-model confound in the staleness gradient is still there (the `uoof_gc` runs never ran). So are the p = 0.075 "second property" and the M-cut attribution to a single draw.
5. **P3-05 and P3-14 (A, P3 text/code).** Never-played calibration is measured on outcome-selected cells. The patch "no reset" test runs on `r_adj` instead of the demeaned `u`, using in-sample residuals.
6. **X3 (latent, every area): the 2026-09-30 tier relabel.** Every pinned cache predates it, so current results are consistent. A rebuild from the DB would mix schemes. That includes the pre-registered drift C2 analysis (D-N2). Pin the label source before any rerun.

## 3. Cross-cutting themes

**X1. Live site artifacts used as research inputs.**
- **The mechanism:** `training/sweep_enriched_wp.py:143-149` (`StatsCache._load_compositions`) reads `src/lib/data/compositions.json` at call time.
- **Who reads it:**
  - paper-1 gN and gB, the submitted proxy and the "hp" stats, `comp_gn_folds.py` and `judges_v2.py`;
  - `production_refresh/refresh.py:385,809` ("compositions unchanged");
  - `drift2026/build_drift_features.py:56`, which is superseded by the rebuild;
  - `qm2026/train_qm_wp.py` (summary only);
  - roughly 15 older training scripts.
- **The sync overwrote the file twice:** 09-30 16:52 (10 rows, 2.57 only) and 10-01 00:02 (364 rows, 2.55 + 2.57). The working tree still holds the second version.
- **Tier labels don't match:** the table is in site-scheme tiers, while snapshot-era research rows carry the old labels (P1-N2).
- **Fix:**
  - research code gets a pinned, hashed copy;
  - the site file belongs to the sync alone;
  - `git checkout` the file only if the site should serve HEAD. It probably should not, because the sync owns it.
- The P1 §0 entry covers the paper-1 impact.

**X2. The CUDA MCTS tree never expands past an opponent decision. NEW for paper 1, drift and production (STILL OPEN, B, production/code plus P1 and drift text).**
- **Where:** the parent kernel `training/cuda_mcts/mcts_kernel.cu` has the same rule Codex found in the personalized copy (P3-08).
- **The mechanism:**
  - The root and leaves are expanded only when `scratch.current_team() == scratch.our_team` (lines 229 and 395-397).
  - Selection only walks expanded nodes (line 355).
  - So a child whose next move belongs to the opponent stays a leaf forever, and every visit to it finishes with a GD rollout.
  - The opponent branch at lines 374-390 is reachable only from an expanded node, and expanded nodes are always at our own turn.
- **The effect:** the tree spans only the current own-pick block (one or two picks). Extra simulations refine the estimates over that horizon; they do not search deeper.
- **Kernels affected:**
  - paper 1 (every MCTS agent: B/F/J_oof, the submission's 13 configurations);
  - drift (every MCTS arm);
  - production refresh training;
  - all four P3 kernel copies (P3-08).
- **The web kernel is different:** `src/lib/draft/mcts-search.ts:316-319` already rolls opponent steps forward with GD so that leaves land on our next decision. Its comment says this is to keep the tree from staying "one ply deep". Training and in-browser search therefore run different algorithms.
- **Text affected:**
  - Paper 1 calls the method "AlphaZero-style" and speaks of "search depth" (L59, L273, L329). What it varies is the simulation budget over a one-block horizon.
  - The drift paper's 200-simulation agents are the same kind of search.
- **Fix:**
  - port the web kernel's opponent roll-forward into `mcts_kernel.cu` and the P3 copies, with a CPU trace test that reaches a later own turn;
  - or describe the method as "current-pick-block search with behavioral rollouts" and replace "search depth" with "simulation budget".
- **Severity:** B for the text, since no number changes. It also strengthens P1-A1: more simulations could not have deepened the search.

**X3. The tier relabel is a rerun hazard (latent).**
- **The change:** `4848ca5` relabeled 1,144,953 `replay_draft_data` rows into the site scheme and added `unknown`.
- **Current results are consistent.** These pinned caches predate it:
  - `overfit2026/cache/*.pkl.gz` (09-29);
  - `drift_rebuild/feature_cache/r10_clean_games.json` (09-29 23:50);
  - the drift truth cache (09-27);
  - the P3 `x_post` and `x_side` caches (09-30 16:49 and 16:52).
- **Scripts that would mix schemes on a rebuild:**
  - `overfit2026/data.py build_db` (P1-N4);
  - `drift2026/w12_clean_truth.py`, which is frozen for the prereg; also `w13_counter_dig.py` and `r10_net.py` (D-N2);
  - the P3_EXTENSIONS rerun guide (P3-26).
- **A silent failure mode:** `training/shared.py:163` `tier_to_one_hot` returns an all-zero vector for `unknown`, so no error is raised.
- **Fix:** read the old label from `replay_draft_skill_tier_backup_20260930`, or rebuild it from `league_tier`, in every research fetch. Assert the label distribution on load. Disclose it as a prereg deviation for drift C2.
- **Already on the new scheme:** the expert-study v6 data (`paper1_revision/oct2026_data.py`) and production (`refresh.py`, e16fc2a). Any judge trained on old-label caches that scores their drafts is mixing schemes. That was not checked here and is worth a look before v6 scoring.

**X4. Production's "out-of-fold statistics" do not cover the composition block. NEW (STILL OPEN; B for production, C for drift text).**
- **The code:** `production_refresh/refresh.py:381-385` builds per-fold out-of-fold stats but calls `st._load_compositions()`, documented as "compositions unchanged". The composition features therefore come from the live Heroes Profile aggregate.
- **Why it is a leak:** that aggregate includes the training rows' own outcomes, and it changes nightly. This is the target-leak pattern both papers warn against, and in exactly the dimension paper 1's safety story depends on.
- **Text affected:** drift L863-866 says production uses "90-day decayed statistics computed out of fold for training rows".
- **Fix:** build composition stats per fold from the corpus, as `core.stats_json` and `counts_for` already do for research. Or state the exception in the drift text.

**X5. Patterns that recur across areas.**
- **Circular "realized" validation:** P3-01, plus its new copies in the distilled prior and EXT §5. Any check that ranks real choices by a value containing the predictor, then scores against the residual that predictor was fit to, re-tests the predictor. The same shape appears in paper 1's MCTS checkpoint selection on the optimized value function (P1-C1).
- **Inconsistent visibility contracts:** P3-03, P3-04 and P3-24 use same-day replay_id order, end-time order, lag-1, and "eventually uploaded", side by side. A combiner is fit under one contract and served under another. Production should pick one contract and use it for fitting, evaluation and serving.
- **Bootstraps that ignore the game or player structure:** P3-15, P3-16, P1-C6 (tournament SE covers drafts, not seeds) and drift B14 (normal intervals at df ≈ 3). Default to game-paired or seed-level resampling.
- **Selection on the evaluation instrument:** P3-17 (decay chosen on V2), P1-C4 (operating point chosen on the judges), P1-C5 (consensus structure terms fitted after the agents were scored) and drift A3 (value-function seed best-of-3 on the future window).
- **Numbers with no committed artifact:** P1-C15 and drift C4. Audit scratch directories under `/tmp` are not durable.
- **Causal verbs on observational associations:** "costs" and "sets the rate" in P3-06, P3-12 and drift A2.

The sections below hold the full per-area tables and evidence.

## 4. Personalization pipeline (P3)

Auditors: **Cx** = Codex, **Gk** = Grok, **Gm** = Gemini (read only), **Cl** = Claude (`results/AUDIT_P3.md`).

**Ground truth for this section.**
- None of the audited code or write-ups has changed since the audits. All of `P3_HERO_STRENGTH.md`, `P3_SKILL_DRIFT.md`, `P3_PATCH_AND_ROLE.md`, `P3_VALIDITY.md`, `P3_MMR_AT_GAME.md` and `P3_HERO_SIMILARITY.md`, plus `p3_hs_*`, `p3_sd_*`, `p3_ph_*`, `p3_val_*`, `p3_dr_*`, `p3_mcts_*` and `cuda_personal/*`, are last touched in `8a5a666` or `46f8f1c` (16:33 to 16:43, 2026-09-30, before the audits).
- Post-audit P3 work is all additive:
  - `P3_DRAFTER.md` gained a distilled-prior section (`d08c28d`);
  - `P3_EXTENSIONS.md` and its `p3_x_*` scripts (`2c5935f` .. `89a7871`);
  - `P3_PERSONAL_GD.md` and its `p3_pgd_*` scripts (`9604ec5`, `566f34e`).
- Only `P3_PERSONAL_GD.md` responds to the audit:
  - line 23: the realized check is "not used as evidence (AUDIT_P3 A1)";
  - line 155 and `p3_pgd_mcts.py:160-165`: pools come from pre-day history and no real bans are forced.
- `paper/personalization/PROPOSAL.md` repeats none of the audited claims. Its addendum is dated 2026-07-13.
- `src/` has no product copy for skill or never-played heroes.

So nearly every finding is STILL OPEN. Several have been partly addressed in the new work but not in the original text, and three bug classes were copied into the new work.

### 4. Status table

| ID | Finding | Found by | Status | Sev | Owner |
|---|---|---|---|---|---|
| P3-01 | Drafter "realized value" is the skill model restated (circular agreement-quintile check) | Cx A1, Gk A1, Gm A, Cl A1 (**all four**) | STILL OPEN; **new instances** in d08c28d and P3_EXTENSIONS §5 | A | personalization code/text |
| P3-02 | Drafter oracle information: pool includes the real hero, future real bans forced, realized player order; verify script does not test these paths | Cx A2, Cl C6 | STILL OPEN in one-step, MCTS and distilled harnesses; FIXED only in the personal-GD harness | B | personalization code |
| P3-03 | Phase-2 offsets use replay_id-ordered same-day counts; drafter combiner (b2 3.71) fit on them but served lag-1 counts; role counts ordered by end time | Cx A3, Cl B1, Gm C, Cl C7 | STILL OPEN | B | personalization code |
| P3-04 | Prior-day is not availability (later-uploaded rows enter the state) | Cx A4, Cl C2 | STILL OPEN | C | personalization code / production |
| P3-05 | "80% intervals are calibrated" (coverage vs var_ratio; future-offset target; never-played cells selected on outcomes) | Gk B2, Cx A5, Cl A2; Gm says it survives | STILL OPEN (partly addressed by P3_EXTENSIONS §2A); **new instance** in §1 | A | personalization text |
| P3-06 | −6.9pp offset worded causally ("costs"), product advice | Cx A6, Cl C3, Gm note | STILL OPEN; **new instances** in P3_EXTENSIONS §3 | B | personalization text |
| P3-07 | Stale leaky MMR numbers in the summary; "MMR adds almost nothing" ignores hero MMR alone | Cx A7, Cl B4 (Gk: survives) | STILL OPEN | B | personalization text |
| P3-08 | Personalized MCTS never expands past an opponent decision (search covers only the current own pick block) | Cx B1 | STILL OPEN, in all 4 kernel copies | B | personalization code |
| P3-09 | static@cutoff and lag-7/30 arms reuse the lag-1 offset `mu` | Cx B2, Gk B1 | STILL OPEN (P3_EXTENSIONS §1 does it right for OOT) | B | personalization code/text |
| P3-10 | In-sample WP residuals violate the wp(1−wp) noise assumption used for every variance-component fit | Cx B3, Cl C1 | STILL OPEN (disclosed, not handled) | C | personalization code |
| P3-11 | Split date is 2026-03-29, not 04-01; validity prevalence denominator wrong (1.2% vs 2.24%) | Cx B4 | STILL OPEN | B | personalization text/code |
| P3-12 | Off-role: "bootstrap" SE is analytic; ± is 1 SE; headline −0.9 from the in-sample window; causal "forced ... costs" | Cx B5, Gk C4, Cl B2 | STILL OPEN | B | personalization text |
| P3-13 | "Hero-specific information halves in 4 to 6 years" (rate at bound, about 186 y); CF half-lives without CI; ratio 1.05 averages 0.35 to 2.94 | Gk B3, Cl C5, Cx C1 | STILL OPEN | B | personalization text |
| P3-14 | "Patches don't reset skill / any jump < 1pp" overstated; JumpFilter fed `r_adj`, not demeaned `u` | Cl A3, Gk C1, Cx C1 | STILL OPEN | A | personalization code/text |
| P3-15 | Drafter Q5−Q1 bootstrap resamples the two teams of a game independently | Gk C2, Cl B5, Cx C3 | STILL OPEN; inherited by p3_ds_analyze | C | personalization code |
| P3-16 | CIs ignore recurring players (design effect about 1.7) | Cl B5, Cx C3 | STILL OPEN | C | personalization code |
| P3-17 | No-decay vs 365-day half-life settled on V2 | Gk C3, Cx C2 | STILL OPEN (disclosed at line 94; product rec not flagged) | D | personalization text |
| P3-18 | "Pooling across similar heroes" gain is mostly the player's overall level | Cl B3 | STILL OPEN | B | personalization text |
| P3-19 | Multiplicity: naming The Butcher / Qhira as significant | Cl C4, Cx C2 | STILL OPEN | C | personalization text |
| P3-20 | Smurf-exclusion sensitivity uses future labels | Cx C4 | STILL OPEN in P3_VALIDITY; superseded by causal boundary purification (P3_EXTENSIONS §4B) | C | personalization text |
| P3-21 | `\|sd` / `\|lower80` team features "mathematically meaningless" | Gm B | **DISPUTED** | – | – |
| P3-22 | Kernel mean-of-logits vs combiner fit on mean-of-probabilities | Gm B | STILL OPEN but immaterial | D | personalization code |
| P3-23 | `_safe_lag` `steps < 400` cap | Gm C | STILL OPEN but negligible | D | personalization code |
| P3-24 | Imitation recency features include same-day games | Cl C6 | STILL OPEN (now feeds P3_EXTENSIONS §5d and the distilled prior) | C | personalization code |
| P3-25 | One-step vs MCTS "personal component" defined differently; rows not comparable | Cl C6 | STILL OPEN | C | personalization code/text |
| P3-26 (new) | Rerun hazard after the skill_tier relabel: P3 post and side caches hold old labels, a refetch now returns site labels | this pass | STILL OPEN (latent) | C | personalization code / production |

**Counts:** 25 STILL OPEN (3 A, 10 B, 9 C, 3 D) and 1 DISPUTED. None is fully FIXED or STALE. Partial fixes:
- P3-02 is fixed in the PGD harness only.
- P3-05 is partly addressed by EXT §2A.
- P3-09 is done correctly for OOT in EXT §1.
- P3-20 is superseded by EXT §4B.

### 4. Notes and evidence

**P3-01 (A).**
- **Mechanism, confirmed in code.**
  - `p3_dr_drafter.py` `_value` puts `b2·(S0−S1)` into V, with `S = Σ s`, where `s = mu + m` (`p3_dr_core.py:131-132`).
  - The realized check ranks the real pick by V and regresses `y − WP_pop` on that rank (`p3_dr_analyze.py:209-256`; `p3_mcts_analyze.py:115-160`).
  - `s` was fit to predict exactly that residual.
- **Three independent reproductions agree.**
  - Cx: +7.64 raw, +0.05 remaining after the model's own predicted gap (CI −2.7, +2.8).
  - Cl: +7.64 vs predicted +7.60.
  - Gk: +1.2 (−1.7, +4.5) after residualizing on `s`; MCTS at 1,500 sims is −0.3.
- **Still in the text.** `P3_DRAFTER.md:70` says "real, model-free value". Lines 160, 213-214, 254 and 260 ("Realized value rises too") are unchanged.
- **New instances, all post-audit.**
  1. `P3_DRAFTER.md:337-369` (distilled prior, d08c28d, 17:12), which presents the realized gap as the drafter's merit:
     - "+5.1 (0.0, 9.8)";
     - "personal component ... +12.8 to +14.2pp ... separates them most sharply of any drafter";
     - "realized agreement +3.1 vs +0.1pp".

     The distilled prior's features include `b2·s(p,h)` directly, plus comfort and recency shares (`P3_DRAFTER.md:301`), so this is the same circularity, amplified.
  2. `P3_EXTENSIONS.md` §5(b), realized teammates-only check: +6.4 vs +5.7pp.
  3. `P3_EXTENSIONS.md` §5(d), joint regression with "personal component +12.8pp per unit".

  §5(c) (assignment component slope 0.75 to 0.85) and §8D (ban-value slope 0.92) are framed as calibration slopes. That is the non-circular use the auditors recommend, so they are fine.
- **Fix.**
  - Delete "model-free value".
  - Relabel every realized-agreement row as "team-level calibration of the skill model".
  - Report the remainder after the predicted gap (Cx/Cl table) next to each raw Q5−Q1.
  - Apply this to the distilled section and to EXT §5(b) and §5(d) too.
  - P3_PERSONAL_GD already follows this; copy its line 23 wording.

**P3-02 (B; Cx A, Cl C).**
- **Still in code.**
  - `p3_dr_core.py:139` sets `pool[..., d["hero"][qidx]] = True`.
  - The `p3_mcts_drafter.py` docstring (line 9) and `lobby()` force real bans (`forced[k] = ...` for every ban step) and seat later picks by the real order.
- **Inherited by the distilled-prior runs.** `p3_ds_mcts.py` builds lobbies with `S.lobby(gi)`. `p3_ds_analyze.py` uses `p3_mcts_analyze.realized`, which appends the actual hero to the candidates.
- **Fixed in one place.** `p3_pgd_mcts.py:160-165` builds pre-day pools and sets `forced = -1`.
- **Why B, not A.** EXT §8A measures the pool leak at 5.6% of slots. The PGD rerun under the fixed protocol keeps a predicted gain of similar size (+6.6 to +8.2pp), so the claim's direction survives.
- **Fix.**
  - Port the PGD protocol into `p3_dr_core.personal_tables` and `Setup.lobby`, then rerun the one-step, MCTS and distilled tables.
  - Add a CPU test for pools and forced bans to `p3_mcts_verify.py`.

**P3-03 (B).**
- **The same-day leak is still live.**
  - `p3_hs_core.py:151-171`: `experience_counts` orders by `(replay_id, day)`.
  - The phase-2 users still call `prepare()`: `p3_sd_eval.py:198`, `p3_sd_order.py:185`, `p3_val_validity.py:109`, `p3_ph_role.py:119`.
  - `p3_ph_role.py` scores `mu = table[exp_bins(n_p_rid, n_ph_rid)]`.
- **Combiner train/serve mismatch.** `results/p3_ph_role.json` D_game "skill + off-role count (fine)" has coef `[0.0013, 1.0015, 3.709, −0.052]`. `p3_dr_core.combiner()` serves it to every drafter, while `personal_tables` supplies lag-1 counts. EXT §1 reports b2 = 3.62 when refit on lag-1 counts, which matches Cl.
- **Role counts.** `p3_ph_role.prior_counts` sorts by `(ts, replay_id)`, and `ts` is the game end. That is Gm's point, and it is valid.
- **The new P3_EXTENSIONS scripts do not copy this bug.** They use lag-1 counts for every evaluation offset (`p3_x_oot.py:89-113`, `p3_x_newacct.py:83-84`). They use rid-order counts only to build `r_adj` for earlier days, the phase-1 convention.
- **Numbers to correct.**
  - SKILL_DRIFT +0.0156 becomes about +0.0148 (strict); Cx gets 0.01495 in end-time order.
  - VALIDITY +0.0156/+0.0143 each fall by about 0.0008.
- **Fix.**
  - Use one strict-order or lag-1 count builder everywhere.
  - Refit the D_game combiner on lag-1 counts and re-serve it.
  - Rerun SKILL_DRIFT §1b, VALIDITY and PATCH_AND_ROLE.

**P3-04 (C).**
- Confirmed in `p3_dr_core.py:113-124` (state from every row with `day <= t−1`).
- The same convention is used in EXT §1 ("State comes from every game on earlier days").
- Cl measures the cost at about 0.0004 on the headline.
- **Fix.** Label results "retrospective, all eventually-uploaded past games". For production, rebuild with an ingestion-time filter.

**P3-05 (A for never-played heroes).**
- **Target mismatch.** `p3_hs_eval.py:129-130` sets `r_adj = r − table[counts at each future row]`. So the coverage target uses future offsets, as Cx A5 says.
- **Still claimed.** `P3_HERO_STRENGTH.md:20` ("79% on never-played heroes") and line 225 (the display rule) are unchanged.
- **Partly addressed.** EXT §2A tests the offset level on every first adoption game, without survivorship (−4.3 predicted, −4.5 realized). §2B admits that the survivor band is selected.
- **New instance.** EXT §1 says "0.82 on never-played heroes" at OOT cells with 10+ games, which is the same selected population.
- **Gemini is wrong** that this survives without qualification.
- **Fix.**
  - Score the never-played band with a proper score on the first k = 1 to 3 games of every adoption event.
  - Publish var_ratio next to coverage.
  - Restate the HERO_STRENGTH line 20 and line 225 wording, and the EXT §1 wording.

**P3-06 (B).**
- **Still in the write-up.** `P3_HERO_STRENGTH.md:22`: "Say plainly that a never-played hero costs about 7pp for a veteran."
- **New instances.**
  - EXT §3A: "The first game on a new hero costs about 10pp ... This estimate is free of selection". Game-1 outcome selection is absent, but the choice of hero and timing is not.
  - EXT §3B: rust "costs 1.5 to 2.2pp".
- **Exempt.** The EXT §8B ban effect has a within-player design and a placebo, so "costs" is defensible there.
- **Fix.** Use "players with this recorded history average −7 to −8pp", as Cl wrote, and soften §3's "free of selection".

**P3-07 (B).** `P3_HERO_STRENGTH.md:11` ("60.81% and +0.0201") and line 21 ("+0.0088 ... stacks") are the leaky lagged MMR and are unchanged.
- Clean numbers on V2:
  - hero MMR alone: +0.0028 (Cl) to +0.0037 (Cx);
  - skill + all causal MMR: +0.0008 (Cl) to +0.0010 (Cx) over skill.
- **Fix.** Replace those two lines, and say "player MMR adds almost nothing; hero MMR alone +0.003, +0.001 on top of skill".

**P3-08 (B).**
- **The defect.** `cuda_personal/personal_kernel.cu:536-539` expands only when `scratch.current_team() == scratch.our_team`.
  - Opponent turns are not nodes: the cached-opp branch at lines 513-528 samples and stays on the same node.
  - So a child whose next move is the opponent's is never expanded, and every visit rolls out by GD.
- **Same rule in every copy.** `cuda_personal/ref/mcts_kernel.cu:402`, `cuda_prior/prior_kernel.cu:578` and `cuda_pgd/pgd_kernel.cu:623` all carry it.
- **Text affected.** `P3_DRAFTER.md:245-264` ("deeper search", "the 4,096-node guard binds") describes depth that does not exist.
- **Fix.**
  - Document the search as current-pick-block search with GD rollouts, or add opponent expansion.
  - Add a CPU trace test.
- **Note.** This is the paper-1 kernel lineage. The parent kernel `training/cuda_mcts/mcts_kernel.cu` has the same rule (see X2).

**P3-09 (B).**
- **The defect.** `p3_hs_eval.py:151-160` computes `mu` only when `qidx is None` (the first config) and reuses it for static@cutoff and lag 7/30.
- **Text affected.** `P3_HERO_STRENGTH.md:12` ("freezing ... adds +0.0035") and line 94 (lag costs 0.0016 / 0.0026) are unchanged.
- **Corroboration.** EXT §1's OOT frozen-counts arm (+0.0044 vs +0.0107 with counts current) supports Gk's +0.0062 for count updates.
- **Fix.** Build `mu` per config, and relabel +0.0035 as "GP update with live counts".

**P3-10 (C).** Disclosed only as "probably explains part of the 1.20 scale" (`P3_HERO_STRENGTH.md:235`).
- Variance fits (kernels, Kalman, jump, calibration) still assume wp(1−wp) on in-sample E residuals.
- Cx shows a 2% change in noise moves latent R² from 21% to 15%.
- **Fix.** Profile the noise scale, or cross-fit WP, before quoting latent R² or drift rates.

**P3-11 (B).**
- **Split date.** `p3_hs_eval.py:134` takes the median day, which is 20541 = **2026-03-29**. `P3_HERO_STRENGTH.md:30-31` says 04-01.
- **Prevalence denominator.** `p3_val_validity.py:171` uses `n_games = g.max()+1`, the full WP index space of 1,949,087 games. `P3_VALIDITY.md:25` then says "1.2% of all games", and line 27 builds on it. Cx gets 2.24% of represented games.
- **Fix.** Correct the dates, recompute the denominator from unique represented games, and redo the line-27 arithmetic.

**P3-12 (B).**
- **SE labelled "bootstrap".** `p3_ph_role.py:96-111` `penalty()`: the docstring says "bootstrap SE by row", but the code is the iid analytic SE.
- **What the clustering changes.** Cl and Cx both find player and game clustering leave it unchanged at 2025+ (0.088 vs 0.086 to 0.089). Cx gets a V2-only SE of 0.29, so V2 is −0.57 ± 0.29.
- **Text.** `P3_PATCH_AND_ROLE.md:12-15` leads with −0.9 (±0.1) from the in-sample 2025-26 window and says "forced off-role play costs games".
- **Fix.**
  - Headline the post-cutoff −0.6 (and fine role −0.8) with clustered SE, written as "SE".
  - Fix the docstring.
  - Say "association".

**P3-13 (B).** `P3_SKILL_DRIFT.md:21` "hero-specific information halves in about 4 to 6 years" contradicts the same file's line 92 (the rate at its bound, above about 190 years).
- Line 18 ("4 to 7 years") gives no CI, and line 24's "1.05" variance ratio averages bins.
- **Fix.**
  - "Hero-specific drift not detected (rate at bound)".
  - "CF factor half-life 4.4 to 6.8 y, no SE".
  - Show the per-bin variance ratios.

**P3-14 (A).**
- **The defect.** `p3_ph_patch.py:408/435/451` passes `r_adj` to `JumpFilter`, but the docstring (line 13) and the write-up describe `u = r_adj − mean(build, hero)`, which is computed at line 357 and unused there.
- **Text.** `P3_PATCH_AND_ROLE.md:7-11` still says "Any real jump is below about 1pp sd".
  - That bound comes from an in-sample likelihood under a fixed noise model.
  - The DiD CI is −0.44 to +0.34.
  - Cl finds an OOS variance ratio of 1.41 (patched) vs 1.11 (unpatched) after 2.55.16.
- **Fix.** Rerun the jump test on demeaned `u`, out of sample, with a fitted noise scale. Reword as "no jump detected in-sample".

**P3-15 (C).** `p3_mcts_analyze.py:155-160` and `p3_dr_analyze.py:239-244` resample `top` and `bot` teams independently, although the two teams of a game have residuals of −1.
- `p3_ds_analyze.py` imports the same `realized()`.
- The numbers barely move: game-clustered (4.8 to 5.1, 10.3 to 10.5).
- **Fix.** Use a game bootstrap, or team 0 only.

**P3-16 (C).** Replay bootstraps throughout, including EXT (header line 9: "CIs are replay bootstraps unless stated").
- Cl measures a design effect of 1.65 to 1.71: the headline CI becomes (0.0127, 0.0157).
- **Fix.** Use a player-cluster or two-way bootstrap for the headline and paired contrasts.

**P3-17 (D).** `P3_HERO_STRENGTH.md:94` already says V1 preferred 365 days. Line 223 ships no-decay and calls 365 "equally good".
- **Fix.** One clause saying the choice was made on V2, and use the V1 pick (59.84%, +0.01396) for any pre-registered number.

**P3-18 (B).** `P3_HERO_STRENGTH.md:13-17` attributes the +0.43 to +0.57 per 1,000 (n = 0 to 10) to "player-specific hero similarity", relative to a baseline with no player term.
- Cl: similarity over player+hero is +0.07 (n.s.) at n = 0, +0.16 to +0.20 at n = 1 to 10, and +0.0010 at game level.
- EXT §2A's paired similarity − player-only of +0.19 per 1,000 is consistent with Cl.
- **Fix.** Restate the pooling gain as mostly player level, and give the similarity part separately.

**P3-19 (C).** `P3_HERO_SIMILARITY.md:27,138` name The Butcher as the one significant negative, and line 139 names Qhira. With 90 two-sided tests, about 2 per tail are expected by chance.
- **Fix.** Apply an FDR correction, or drop the names.

**P3-20 (C).**
- `P3_VALIDITY.md:32` still reports the oracle exclusion (+0.0156 → +0.0143) as the answer.
- EXT §4B adds a causal boundary purification (−0.00001), which answers the question properly.
- **Fix.** Point VALIDITY at EXT §4B and label the old row "retrospective".

**P3-21 DISPUTED (Gm B).**
- `p3_sd_eval.py:309-310` builds `|sd` = Σsd(team 0) − Σsd(team 1) and `|lower80` = Σ(m − 1.28 sd) team difference. These are signed team-asymmetry features that ask whether a more uncertain team, or a team with weaker pessimistic estimates, loses more than its mean says.
- A symmetric √(Σ s²) over all 10 players has no sign. It cannot enter a team-difference logit at all, so Gm's proposed "fix" would test nothing.
- The question Gm wants answered (calibration against total lobby uncertainty) is already answered: `P3_SKILL_DRIFT.md:241` reports calibration within quintiles of the summed posterior variance.
- The write-up's conclusion ("uncertainty belongs in the display, not the win probability") stands.

**P3-22 (D).**
- **Confirmed mismatch.**
  - `p3_wp_scores.py:66-71` symmetrizes per seed, then takes the mean of probabilities.
  - `p3_mcts_core.reference_value:141-147` and the kernel take the mean of logits per orientation, then symmetrize.
- **Measured on the 143,681 out-of-sample games in `cache/wp_drift.npz` seeds.**
  - |sigmoid(mean logit) − mean prob|: mean 0.007pp, p99 0.055pp, max 0.49pp.
  - logit-on-logit slope 1.001.
- Immaterial to any number. **Fix (optional).** Align the aggregation for hygiene.

**P3-23 (D).** The cap is real (`p3_sd_eval.py:146`).
- Out of 1,436,577 post-cutoff slots, **710 (0.05%)** have 400+ same-player rows in the 30-day window, and 0 at lags 1 and 7. Only the player-key G = 30 lagged-MMR arm is touched, and that arm is superseded by causal MMR (`P3_MMR_AT_GAME.md`).
- **Fix (optional).** Raise the cap.

**P3-24 (C).**
- **Timing.** `p3_dr_imitation.py:72-90` orders by `(pid, ts, replay_id)` and updates per row, so features include same-day games that ended earlier. The docstring (lines 15-16) says so honestly.
- **Why it matters now.**
  - EXT §5(d) interprets imitation agreement as outcome information ("+32 to +37pp per unit").
  - The distilled prior uses the same EWMA and recency features.
  - Same-day features can carry the win-stay stopping rule (Cl B1: P(another game | win) 51.9% vs 49.0%).
- **Fix.** Recompute the features at lag 1 day, or strictly by start time, and rerun §5(d).

**P3-25 (C).**
- **One-step:** the per-decision rank of (V − WP) (`p3_dr_analyze.py:221-223`).
- **MCTS and distilled:** the team mean of the personalized percentile minus the population percentile (`p3_mcts_analyze.py:145-147`).
- So `P3_DRAFTER.md:214` puts +9.6 next to +5.2, and line 254 puts +7.3 next to one-step, under the same name.
- **Fix.** Use one definition, or rename the rows.

**P3-26 (new, C).**
- `cache/x_post_games.json.gz` and `x_side_games.npz` were fetched at 16:49 and 16:52 on 2026-09-30, before the relabel (`4848ca5`, 20:08). They agree with the snapshot's old labels, so current results are consistent.
- **The hazard.** `p3_pgd_data.py:145-148` and `p3_x_robust.py:178` take `skill_tier` from these caches. The P3_EXTENSIONS "Rerunning" guide (line 852+) re-fetches them, which would now return site-scheme labels and mix schemes with the snapshot. `p3_x_ban_fetch.py:66` hard-codes "stored league_tier is the real tier + 1".
- **Fix.**
  - Pin the fetch to `replay_draft_skill_tier_backup_20260930`, or map site labels back to the research scheme.
  - Assert the scheme on load.

### 4. Cross-cutting (P3)

1. **Circular realized-agreement checks** (P3-01) spread to every new drafter write-up after the audit except P3_PERSONAL_GD. Any check that ranks real picks by a value containing `s` and scores against `y − WP` re-tests the skill model.
2. **Inconsistent visibility contracts** (P3-03, P3-04, P3-24):
   - same-day rows in replay_id order;
   - end-time order;
   - lag-1;
   - "eventually uploaded".

   These are used side by side, and a combiner fit under one is served under another.
3. **Kernel lineage** (P3-08): one expansion rule appears in four P3 kernel copies and its paper-1 parent.
4. **Copied realized helper** (P3-02, P3-15, P3-25): `p3_mcts_analyze.realized` (actual hero added to candidates, team-independent bootstrap) is imported by `p3_ds_analyze.py`.

## 5. Paper 1 revision: the claude_p1 audit checked against the current text (2026-10-01)

**Scope.** Every item in `xaudit/claude_p1/AUDIT.md` was checked against the files as they stand now:
- the text: `paper/paper 1/overleaf/draft_revision.tex` (419 lines; "L" below) and `supplementary_revision.tex` (356 lines; "S" below). Both are the HEAD versions; neither has uncommitted edits.
- the results: `training/paper1_revision/results/`, mainly `comp_rescore.json`, `SUMMARY.json`, `mcts_bench/` and `deferred/`.
- the notes: `REVISION_NOTES.md` §10–§12.
- the commits since the audit: df1d4f6, 65af823, c159669 and 0bf37b2. 0c72fb1 and 0652e78 touch only the expert-study lane.

**How it was checked.** Line numbers are current. "Audit L" refers to the audit's own numbering. The one computation was a scratch rescore of stored benchmark drafts, run under `taskset -c 48-49 nice -n 19`; no repo file was edited.

**Counts** (the audit's 5 A, 25 B, 18 C and 15 D rows, plus §0 and §5, are 65 items; two D rows were marked "fine" with no action and are not counted):

| Status | Count |
|---|---|
| FIXED | 5 |
| STALE | 7 |
| STILL OPEN | 51 |
| DISPUTED | 0 |

There are also 4 new items (N1–N4) that the audit could not have seen.

### 5. Headline items

**§0 / cross-cutting: the live `src/lib/data/compositions.json` leaks into research scoring. STILL OPEN, worse than the audit found.** Severity B, high priority. Owner: production/code, then paper 1 text.

How the file is read:
- `sweep_enriched_wp.StatsCache._load_compositions` reads the live site file at call time.
- Every `feats.stats_from_json(..., compositions=True)` goes through it. That covers:
  - gN and gB featurization (`overfit2026/split.load_stats` → `score._stats`);
  - the submitted proxy and the `"hp"` stats (`feats.paper_stats`). These feed `proxy_sub` / Table VI Train\* and the tournament's unchanged CQL (enr.) and Gourdeau actors;
  - `comp_gn_folds.py` and `judges_v2.py`.
- The QM judges do not read it for features. The revision's own value functions use `deploy_compositions.json`.

The nightly sync overwrote the file twice:

| Write time | Rows (low/mid/high) | Source |
|---|---|---|
| HEAD (paper's table) | 412 (164/127/121) | committed 2026-03 |
| 2026-09-30 16:52 EDT | 10 (2/5/3) | 2.57 only (other session's `nightly.log` 20:52Z) |
| 2026-10-01 00:02 EDT | 364 (132/129/103) | **2.55 + 2.57 combined** (`sync/logs/sync-2026-10-01_000002.log`). The working tree still holds this version. |

Results scored against a non-HEAD table:

- **J_oof s0/s1 (benchmarked 09-30 20:17–20:20) and the whole A_partial/G_base rebench (20:57–21:00) were scored with the 10-row table.**
  - Rescoring the stored drafts with the HEAD table in a scratch run:
    - Control: F_oof s0 (scored 08:19) reproduces to 5e-6.
    - J_oof s0/s1 and every A_partial/G_base seed differ per draft by up to 0.085–0.146.
  - Effect on the means:
    - J_oof gN: −0.0002, so J_oof − F_oof goes from −0.0069 to about −0.0071 and the conclusion is unchanged.
    - A_partial gN: −0.0001.
    - **G_base gN: +0.0017. Table VI G_base gN is .636 and should be about .638.**
  - The Train\* column for A_partial/G_base (`proxy_sub`, "hp" stats) used the same table and was not re-checked.
- **Since 2026-10-01 00:02 every gN computation reads a table built partly from patch 2.57 games**, a build released 2026-09-28, after the 2026-09-27 cutoff. This breaks L82, "No game from a build released after 2026-09-27 is used". It covers c159669's:
  - `comp_audit_{N,SNAP,T17}.npz` (10:39–11:07): the gN gap −1.6 → −0.5 and post-snapshot −1.1;
  - `comp_gn_folds` two-fold gN refits (11:14–11:16), and with them gN's β (−0.04, −0.06, 0.00);
  - the Table II "Ind." gN recompute. REVISION_NOTES §11.4 says the recompute "matches gN to within 0.002"; that 0.002 is this table difference;
  - the leak-free CQL (enr.) refs (`cql_enriched_eval.json`, 01:20; S354 "0.50–0.53").
- The numeric impact on gN is small (4 of 283 dimensions). The independence and cutoff statements are now false, and the per-draft release records for 4 configurations do not reproduce.

Fix:
1. Freeze a copy of the HEAD table under `training/paper1_revision/cache/stats/` and point `_load_compositions` (or `feats.stats_from_json`) at it explicitly; never at `src/lib/data/`.
2. `git checkout src/lib/data/compositions.json`, or let the sync own it and have research code stop reading it.
3. Rerun:
   - `bench_mcts` gN and proxy_sub scoring for J_oof s0/s1 (T0, T1) and A_partial/G_base (all seeds);
   - `comp_audit_data.py`, `comp_gn_folds.py`, `comp_judges.py` and `comp_rescore.py`;
   - `cql_enriched_eval` refs.
4. Add a hash assertion on the composition table to `bench_mcts.py` and `score.py`.

Tier relabel:
- The overfit2026 research caches were built before the relabel (`future_games.pkl.gz` and `backfill_games.pkl.gz` on 09-29; the snapshot is pinned), so they keep the old labels and are unaffected.
- **Risk:** `overfit2026/data.py build` reads `skill_tier` live from the DB. A rebuild would silently mix site-scheme post-snapshot labels with old-scheme snapshot labels. Pin it or assert on the label distribution.
- The HP composition table is in site-scheme tiers both at HEAD and now. See N4.

**§5 PENDING markers: FIXED** (df1d4f6). J_oof is filled in Table VI (L268) and the text (L273). **Decision-rule outcome (i) occurred:**
- v2, T=1: gN −0.0069 ± 0.0029 (2.4 SE); RN −0.0079 ± 0.0031 (2.5 SE).
- QM2026 −0.0009 ± 0.0037; consensus −0.0039 ± 0.0030.
- Degenerate rate 6.6 → 11.1%.

Knock-on edits that were required:
- Done: L273 now says the split experiment's prediction fails at paper scale and gives the degenerate rate. L333 adds the leak-free 400 → 800 sentence. Fig. 2 plots the 800-sim point. L275 says F_oof is at least as good as J_oof.
- Not done: L59 and L329 (see A1).
- Not reported: the T=0 contrast (gN −0.0085 ± 0.0028) and the with/without-resumed-seeds contrast.

**A1: the search-depth gap is still attributed to the leak. STILL OPEN.** Severity A. Owner: paper 1 text.
- Outcome (i) makes these edits mandatory. The body (L273) now says leak-free search also fails past 400, but four other places still credit the leak:
  - L59: "The leak ... produced search-depth and feature gains that independent judges do not confirm";
  - L329: "yet it produced feature and search-depth gains that independent references do not confirm";
  - L273: "Deeper search kept finding drafts the leaky value function overrated";
  - abstract L35: the leak "let search chase its errors". Paper scale does not show this: F_oof − F_400sim gN +0.000.
- Fix: use the audit's wording. The leak made the value function overconfident and inflated its self-reported accuracy; in a controlled split it let search chase its errors; at paper scale, leak-free and leaky agents score the same, and the plateau past 400 simulations appears in both pipelines.

**A2: "about a quarter as large" ignores judge attenuation. STILL OPEN.** Severity A. Owner: paper 1 text.
- L273 still says "the gain is about a quarter as large and is complete by 400" and "removing the leak did not by itself make the optimized and independent gains agree".
- No regression-dilution benchmark was added. REVISION_NOTES has no mention of it.
- With v2, 200 → 400 gives gN +0.018 against own +0.037. That ratio is 0.49, exactly the no-over-optimization slope the audit measured, so 200 → 400 is not over-optimization.
- Fix: add the held-out judge-on-proxy slope (§III-D) and restate 200 → 400 as fully transmitted and 400 → 800 as the over-optimization.

**A3: synthetic data credited in the abstract, contributions and conclusion. FIXED** (0bf37b2).
- The abstract (L35), contribution 3 (L63), §V (L184–187) and the conclusion (L333) now present the mask as the repair and augmentation as a tradeoff.
- Residual: L71, "We express pessimism through features and data instead", still names data as the vehicle. Tracked under D-L71.

**A4: "features buy safety" is not significant at matched search. STILL OPEN.** Severity A. Owner: paper 1 text.
- The text is unchanged at L63 ("18.5% without features, 6.6% at our operating point"), L277 ("What the features buy is safety: 13.9–15.0% vs. 18.5% ... and 6.6% at the operating point") and L310 ("The reliable difference is safety: 7.3% vs. 23.4%").
- The degenerate rates are not affected by the judge rescore. Matched contrast B_oof − K is −4.6pp (13.9 vs 18.5), about 1.3 SE. Most of 18.5 → 6.6 comes from doubling the simulations.
- Fix: as in the audit, report the matched contrast with its SE, attribute the rest to search budget, and caveat or delete "reliable" at L310.

**A5: gN reads the external composition table. STILL OPEN.** Severity A. Owner: paper 1 text, plus code (§0).
- `overfit2026/split.load_stats` still uses `compositions=True`.
- REVISION_NOTES §11.1 itself says gN's features "include role counts and the external composition table".
- Still false in the text:
  - L124: "share no game, statistic, or search with any agent";
  - L35, L51, L59: "share no data with any agent";
  - L110: "The external aggregates no longer enter any value function". The tournament's CQL (enr.) still reads them (C18).
- New aggravation: L126 credits gN's smaller blind spot (−1.6pp vs −5pp) to its role counts alone. Per the notes, the external composition table, the very dimension the correction is about, is also part of the reason.
- Fix: as in the audit (rebuild gN with its own composition table, or disclose), plus the §0 pinning.

### 5. Full table

| ID | Status | Sev | Owner | Current evidence / fix |
|---|---|---|---|---|
| §0 compositions.json | STILL OPEN (worse) | B (priority) | code/production, then P1 text | See above. |
| §5 PENDING | FIXED (df1d4f6) | – | – | Outcome (i). |
| A1 | STILL OPEN | A | P1 text | L59, L329, L273, L35. |
| A2 | STILL OPEN | A | P1 text | L273; no attenuation benchmark. |
| A3 | FIXED (0bf37b2) | – | – | Residual L71 (D-L71). |
| A4 | STILL OPEN | A | P1 text | L63, L277, L310. |
| A5 | STILL OPEN | A | P1 text + code | L124, L35/51/59, L110, L126. |
| B1 | STILL OPEN | B | P1 text | L82: "291,837 later games that no model, statistic, or search here touched". They train gN, gN-naive and RN, and now also fit the structure β. Fix per audit. |
| B2 | STILL OPEN | B | P1 text | L63 and L277: "6.6% at our operating point". The operating point is argmax root (T0): F_oof T0 is 4.6%, K T0 is 15.2% (`comp_rescore.json` configs). |
| B3 | STILL OPEN (partly fixed) | C | P1 text | L130 and L335 now say the tournament plays the policy argmax with no search. L275 still says "Our operating point is F_oof with argmax root" for a setting the tournament never plays. Say "benchmark operating point". |
| B4 | FIXED (c159669 rescore) | – | – | QM2021 v2 greedy scores .497/.488/.480/.493 are all below 0.5. L310 now says "QM2021 places the greedy agents below 0.5", which is true. |
| B5 | FIXED (c159669) | – | – | GD v2 .4396, so "0.44–0.45" (L35, L59, L310, L333) is now correct. |
| B6 | FIXED (c159669) | – | – | L310: 0.558 ± 0.004, gN .562, RN .554, QM26 .530, QM21 .487 all match the `tournament.v2.h2h` "mcts vs k_truebase" record. |
| B7 | STILL OPEN | B | P1 text | L146–147 still .618 and .613. `crosseval.json` by_enriched gives naive .6175 → .617 and hero str. .6125 → .612. |
| B8 | STILL OPEN | B | P1 text | L315: "anchored methods sit at −0.15 to −0.23" mixes Table V (−0.15 to −0.20) with Table VII (GD −.23). |
| B9 (+ new) | STILL OPEN (changed) | B | P1 text | The CQL (enr.) rows of Table V are now leak-free retrains, so several ranges are stale (L227 and L57): "66–71 heroes" should be 66–70; "69–74% agreement" should be 70–73% (the 69/74 came from the old leaky enr. rows); "−0.15 to −0.20" synergy for CQL/BC-CQL/IQL should be −0.15 to −0.19; L57 "CQL cuts degenerate compositions to 2–3%" is now 0.4–3.6% across the shown cells. |
| B10 | STILL OPEN | B | P1 text | L242 caption: "gN seed SE 0.002–0.004". Per-config SE from `mcts_bench` is 0.001–0.005, and M2_relational is 0.018 (one failed seed, 0.573). |
| B11 | STILL OPEN | B | P1 text | L277: N2 0.665 > full 0.654 > M2 0.645. M2 rests on one failed seed; the other four average .662. |
| B12 | STILL OPEN | B | P1 text | L182: "We retrain the HotS win-rate estimator". The checkpoint is reused. |
| B13 | STILL OPEN | B | P1 text | L80: "No reported number is selected on test data." The inherited Gourdeau, GD, CQL and discriminator checkpoints were selected on test. |
| B14 | STILL OPEN | B | P1 text | L35: "all six independent judges". L124 defines five; R17 appears only in the supplement. |
| B15 | STILL OPEN | B | P1 text | S112: "The main paper notes (Section III) that ... snapshot-sensitive". The main text does not say this. |
| B16 | STALE | – | – | Section removed; "directly verifiable" and "validated head" are gone. |
| B17 | STALE | – | – | The old ensemble section is gone. The new S164–201 refers to the 12-strategy tournament correctly. |
| B18 | STALE | – | – | The 57.4–58.0% passage is gone. |
| B19 | STILL OPEN | B | P1 text | S24 and S49: "195 complete training runs". There are 210 with the 15 leak-free runs. |
| B20 | STILL OPEN | B | P1 text | S257 says "57.7% of an in-corpus model" under Table refcal printing 57.9. |
| B21 | STILL OPEN | B | P1 text | S151 naive least-meta quartile is "54.6"; `ngs_quartiles.json` Q4 = 0.54549, so 54.5. |
| B22 | STILL OPEN | B | P1 text | S107: "inflated ... by about 14% and 18%". Relative to the clean values it is 16% and 22%, and the game sets also differ. |
| B23 | STILL OPEN | B | P1 text | L112 "peak and then fall" and S227 "peak at 4,096 ... and then fall". The fall is 0.009/0.004 at SE ~0.005. |
| B24 | STILL OPEN | B (minor) | P1 text | L275 "never hurts" holds for gN only. In v2 T0 − T1, RN for I_600sim is −0.0003 and QM2026 is negative for B_oof, M2 and N2. Write "never lowers gN". The range 0.000–0.007 and the mean +0.003 match v2. |
| B25 | STILL OPEN | B | P1 text | L55: "Three value functions ... disagree on about 90%". Agreement was measured only for naive vs enriched (84% of bans, >90% of picks). |
| C1 | STILL OPEN | C | P1 text | `train_mcts_worker.py:394` saves at the best `avg_wp` under the training WP. The L238 benchmark paragraph does not disclose this. |
| C2 | STILL OPEN (partly fixed) | C | P1 text | L273 discloses that 3 of 5 J_oof seeds were resumed. It says "from their last saved checkpoint"; REVISION_NOTES:336 says best-eval checkpoints (51K/51K/107K), and the ring buffer was not restored. Correct the wording and report J_oof − F_oof without s0/s1/s4. |
| C3 | STILL OPEN | C | P1 text | Value-head pretraining stuck at 0.2500 in 10 of 15 leak-free runs. Not mentioned anywhere. |
| C4 | STILL OPEN | C | P1 text | L282: "Agents were fixed before any tournament score existed" and F was chosen on the judges' benchmark. Say so. |
| C5 | STILL OPEN (now worse) | B | P1 text | L124: the consensus was "fixed before any revised agent was scored". Since c159669, the v2 consensus adds structure terms fitted on 2026-10-01, after every agent was scored, so the sentence is false for the judges every number now uses. |
| C6 | STILL OPEN | C | P1 text | L286 tab:tourn caption gives "SE 0.001" with no seed caveat. It appears only in Limitations (L335). |
| C7 | STILL OPEN | C | P1 text | Table I is unchanged: no slope SDs and no sentence on test slopes above 1. |
| C8 | STILL OPEN | C | P1 text | Table synth (S60–71, moved to the supplement) still lists assigned rates only. The realized label means are 0.049/0.050/0.095/0.299. |
| C9 | STILL OPEN | C | P1 text | S76: "Synthetic rows get out-of-fold features like real rows". Their labels are adjusted with deploy statistics. |
| C10 | STILL OPEN (partly fixed) | C | P1 text | Fig. 2 no longer plots MCTS twice. It now places the constrained MCTS point (Table V, policy argmax, +0.63) next to the MCTS 200/400/800 points (Table VI, T=1 search, +1.13 to +1.34), and the caption does not mention the protocol difference. The CQL points come from the submission's `rerun2026` rich results (the leaky enr. variant), not the new leak-free rows. |
| C11 | STALE | – | – | Table VII no longer has a Counter column. |
| C12 | STILL OPEN (partly fixed) | C | P1 text | S227 now says "2 seeds per cell". Still missing: the runs stopped at about 30% of training; "Own-split val." is the last epoch; the BC prior and GD bootstrap were trained on the full snapshot; the 40-epoch exception (slope 0.40). |
| C13 | STILL OPEN | C | code | `bench_mcts.py:225`: `out["QM2021"] = q.get("QM2022")`. Rename the key. |
| C14 | STILL OPEN | D | P1 text | L300 and S185 say "G\&A estimator"; elsewhere it is "Gourdeau estimator". |
| C15 | STILL OPEN | C | P1 text / release | There is still no `leak_results.json` in `paper1_revision/results` for the L108 figures (38–67%, 58.1 → 57.0). |
| C16 | STILL OPEN | D | P1 text | L187: "Even with a larger 512→256→128 network". |
| C17 | STILL OPEN | C | code | `train_mcts_worker.py` and `paper1_revision/train_mcts.py` have no `random.seed`. |
| C18 | STILL OPEN (now worse) | B | P1 text | The rich table (L194) now carries leak-free CQL (enr.). The tournament still plays the submission's `cql_enr_a2.0` on external statistics (`tournament.py:23–24`, UNCHANGED set). The disclosure is gone, and L282 "Value-function agents use leak-free models" and L110 imply otherwise. Mark the row, or rerun its 22 pairs with the leak-free checkpoint. |
| D-L333a (audit L350) | STILL OPEN | D | P1 text | "Team assembly combines an exponential state space, clustered data, and a terminal evaluation ..." |
| D-L333b (audit L350) | STILL OPEN | D | P1 text | "Out-of-fold statistics, calibration checks, and independent judges should be standard ..." |
| D-L231 (audit L250) | STILL OPEN | D | P1 text | "Imitation, algorithmic pessimism, and manifold discrimination all end at safety by anchoring." |
| D-L44 (audit L45) | STILL OPEN | D | P1 text | Triad: "Sports drafts, roster auctions, and deck drafting". |
| D-L82/L124 (audit L81/L125) | STILL OPEN | D | P1 text | The "no model, statistic, or search" formula. Rewrite with B1. |
| D-L71 (audit L72) | STILL OPEN, upgraded | B | P1 text | "We express pessimism through features and data instead." This is now inconsistent with §12 (data demoted, mask adopted). Suggest "We add structure through features and a rule-based mask." |
| D-L277 (audit L294) | STILL OPEN | D | P1 text | Cleft "What the features buy is safety" (also A4). |
| D-L327 (audit L344) | STILL OPEN | D | P1 text | "Safety is easy: ..." |
| D-S149a (audit S130) | STILL OPEN | D | P1 text | "removing the coordination constraint is not what diversifies drafting; removing strategic discipline is." |
| D-S151 (audit S132) | STILL OPEN | D | P1 text | "a re-weighting within familiar structures, not entry into the never-observed region". |
| D-S139 | STALE | – | – | Section removed. |
| D-S143 | STALE | – | – | Section removed. |
| D-S188 | STALE | – | – | Section removed. |
| D-S49 | STILL OPEN | D | P1 text | "compute-bound rather than overhead-bound". |
| D-S149b (audit S130) | STILL OPEN | D | P1 text | "shaped by social dynamics rather than win maximization". |

Mechanical style checks: no em-dashes in either file ("---" appears only in TeX comments).

### 5. New items (not in the audit)

| ID | Sev | Owner | Issue | Fix |
|---|---|---|---|---|
| N1 | B | P1 text | L333: going from 400 to 800 simulations "lowered every independent one". v2 T=1 J_oof − F_oof is gN-naive +0.0002 ± 0.0039 and QM2026 −0.0009 ± 0.0037; only gN, RN, R17 (and QM2021 at 1.8 SE) fall. L275, "under every independent reference it [F_oof] is at least as good as J_oof", is the same overstatement within noise. | "lowered gN and RN by 0.007–0.008 and left the others within noise". |
| N2 | C | P1 text | L80 says "All models, statistics, and evaluations use the labels consistently". The HP composition table (`compositions.json`) is keyed by the site scheme (Bronze+Silver / Gold+Plat / Diamond+Master) and is looked up with snapshot labels (Bronze / Silver+Gold+Master / Plat+Diamond). This affects the submitted proxies, the "hp" stats and gN's 4 composition features. | Disclose in §III-A, or rebuild gN with its own composition table (A5). |
| N3 | B (minor) | P1 text | S229 says the ensemble disagreement stays "between 0.0053 and 0.0067"; one run is 0.0051 (the audit's §7 number check, unnumbered there). | "0.005–0.007". |
| N4 | C | code | `overfit2026/data.py build_db` reads `skill_tier` from the now-relabeled DB. A cache rebuild would silently put post-snapshot games in the site scheme while the snapshot stays in the old scheme. | Pin the label source or assert on the tier distribution. |

### 5. Checked and correct

These numbers in the post-audit paragraphs match their sources:
- §III-D "Composition correction" (L126): every number matches REVISION_NOTES §11 and `comp_*` results (with the §0 caveat on gN).
- Tournament Table VII, every cell: matches `comp_rescore.json` tournament.v2. So do the Spearman figures (0.94–0.99; top-6 QM 0.83).
- §V (L187): synthetic-data numbers.
- Mask head-to-heads: 0.509/0.511 ± 0.003, and v1 0.499/0.500.
- Table VI v2 cells: spot-checked B_fullwp, F_oof, J_oof, K, N2, M2, C_large, D_deep and G_base, which differs by +0.002 only through §0.
- J_oof contrasts in L273.

## 6. Drift paper (`paper/drift/overleaf/draft.tex`), claude_drift/AUDIT.md re-checked 2026-10-01

**What changed since the audit.** The audit read the draft at dc40672 (16:40). Since then the only commit that touches the draft is f02ff4c, the tier-description paragraph in Sec. 2 (+8 lines, so audit line L is now about L+8 after line 150). No later commit touches `training/drift_rebuild/` or `training/drift2026/` apart from r14_tierfix. The `uoof_gc`/`mref_gc`/`mcut_gc` controls are queued in `jobs_mcts_v1.txt` but were never run: `mcts_runs/` holds only mcut123_rg(5), mref_rg(6), s1oof_era(5), s2oof_era(6) and uoof_rg(6). Production did change: 3df55c8, d36bd75, e16fc2a, e0aed7d and c159669.

**Tier-fix spot check (f02ff4c, REBUILD_NOTES §6).** The draft's numbers match `results/r14_tierfix.json`. Corrected tiers give 57.107 → 57.11 accuracy and a calibration slope of 0.932 → 0.93; the mislabeled run gives 57.081 → 57.08 and 0.929 → 0.93. The per-seed values in §6 match. The described grouping matches the relabel rule in `sync/relabel-skill-tier.ts` (4848ca5): Silver moved mid→low, Platinum high→mid, Master mid→high.

Counts: **1 FIXED, 0 STALE, 0 DISPUTED, 39 STILL OPEN**, plus **3 NEW**.

| ID | Finding | Status | Sev | Current evidence | Fix | Owner |
|---|---|---|---|---|---|---|
| A1 | The staleness gradient is confounded by the opponent-model swap, and the paper credits the swap to the leak fix | OPEN | A | The gc arms were never run. These sentences remain: L118-120 "Agents built with the same recipe one and two years earlier trail by 1.12 and 1.00pp; most of the loss arrives within a year"; L580 "most of it arrives in the first year"; L599-600 "were properties of the leaky stale agent"; L602-603 "The leaky first version put the one-year gap at only +0.56". The abstract (L66) has no caveat. The Fig. 2 caption says only "leaky first version". The conclusion's "one or two years cost about a point" inherits the problem. | Run `uoof_gc` (6 seeds) and rescore. Otherwise apply the audit's text fix: drop "most of the loss ... within a year", qualify "same recipe", add the caveat to the abstract, attribute the change to "the leak fix and the opponent-model change together", and recaption Fig. 2. | drift text |
| A2 | "Two properties set the rate": the second property rests on p = 0.075 | OPEN | A | L301 and L308 are unchanged. Recomputed: 21.2/25.6/30.6 vs 17.8/16.2/16.4 gives Welch t = 3.26, df = 2.14. With the cutoff opponent model, 21.2/25.4/23.2 vs 13.4/15.0/14.0 gives t = 7.03, df = 2.58. | Give df and p for every Welch test. Lead with the cutoff-opponent pair (p ≈ 0.01). Say "History length sets most of the rate". | drift text |
| A3 | "The same-cutoff gap comes from the value-function recipe" rests on one selected value-function draw | OPEN | A | L67 "The same-cutoff gap comes from how the value model was trained"; L116; L571 "The advantage comes from ...". There is no per-seed breakdown and no CI on M vs M-cut. | Say "consistent with". Give refresh as −0.05pp (95% CI −0.30 to +0.19). Add one sentence with the per-value-function-seed rescore, or run `mcut_rg`. | drift text |
| B1 | The production pipeline had never completed a run (see X4: the composition block is not out of fold) | **FIXED** | – | The cron env was fixed in 3df55c8 (absolute `/home/linuxbrew/.linuxbrew/bin/python3`, which imports psycopg2). The full run `production_refresh/2026-09-30/` completed every phase: 5-fold OOF decayed-90 stats, WP slope 1.048 inside [0.8, 1.25], deployed as d36bd75. It trains on known site-scheme tiers only (e16fc2a, `TRAIN_TIERS`). Residuals: (i) that run was started by hand at 20:22, so the first scheduled cron run is 2026-11-01. (ii) The deployed policy was trained at 800 sims and its seed was picked by proxy (`refresh_meta.json` best_seed 1 = max best_wp). The code now does 400 sims with selection by an independent structure-aware judge (e0aed7d, c159669); only a 6K-episode dry run exists (`validate-seedsel-2026-10-01`, gate FAIL on the proxy floor, as expected for a dry run). (iii) The 09-30 refresh postdates the 09-27 registration, so prereg C3 has to log it. | Optional: date the last deployment in Sec. 10, and log the 09-30 event for C3. | production |
| B2 | Rank correlation is over 9 strategies, not 11 | OPEN | B | `qm2026/train_qm_wp.py` CONSENSUS_ORDER still keys `gourdeau_est`/`mcq`; the standings use `gourdeau`/`mcq_t0.5`. Recomputed with the same rank code: both `qm_wp_v0.json` (QM-2021) and `qm2026_judge.json` give **0.917 over 9 and 0.900 over 11** (0.907 with average ranks for ties). L779-780 still say "11 strategies at rank correlation 0.92". `judge_2026.py` imports the same dict. No paper-1 script uses it. | Fix the keys and report 0.90 over 11. | drift text |
| B3 | The 30-day model's per-build story is wrong | OPEN | B | L826-834 are unchanged. W3_GD_DRIFT has three 2.55.15 builds, and win30d is best on all three (96477: 12.18). On 96846 and 96870 it beats all-history (11.05/11.06 vs 10.88). In `r13_rolling_gd.json` the rolling model loses on 96881 (window 37,608) but wins on 96846 (window 35,589, smaller), which contradicts "whose window held fewer games". | Use the audit's wording and drop the window-size explanation. | drift text |
| B4 | "Standard error about 1.9pp" is wrong | OPEN | B | L843 is unchanged. Recomputed from `w2b/d2c_cumprev_s*{,_gdcutoff}.json`: healer +2.93, paired SE 1.23; degeneracy −2.67, paired SE 0.93. `w5_gd_drift.py:308` uses np.std with ddof=0. | Give the correct SEs. | drift text |
| B5 | "Every model in the replay trains on causal statistics with the same seed" | OPEN | B | L333 is unchanged. Table II has frozen-recipe rows with 3 seeds (`r9_replay.json`). | Use the audit's wording. | drift text |
| B6 | "1.08 vs 1.06, mean of three seeds" compares a 3-seed mean with one seed; the noise is 0.07, not 0.05 | OPEN | B | `r9_replay.json`: frozen leak-free 1.118/1.048/1.079 (mean 1.082) vs never-retrain 1.062 (s42 only). L335 and L974 still say "0.05pp". | Use "0.06pp same-seed; seed range 0.07". | drift text |
| B7 | "40 drafts per ordering" is false for the 15×15 files | OPEN | B | L449 is unchanged. M-d90 vs M has 4,500 drafts over 225 pairings, i.e. 10 per ordering; W8B_CHAMPION.md says "10/cell". | State 10 per ordering for the 15×15 matchups. | drift text |
| B8 | "No agent's statistics touch" the 298K scoring games omits the 12,337-game overlap with the selection and opponent window | OPEN | B | L62 and L483 are unchanged. Also L112, "builds no agent has seen", contradicts L470-471 (the opponent model saw the test period). | Disclose the overlap, cite the 2.55.17-only results, and fix L112. | drift text |
| B9 | Net uses 308,375 games, Δ uses 298,428 | OPEN | B | `r10_net.py` fetches live from the DB (builds list, ≤ 2026-09-27); the caption (L491-498) does not say so. | Add a caption note, or recompute on the pinned set. If recomputing, see N2. | drift text |
| B10 | "37 of 59 patches (848)" is on a different base from the 40 mapped patches | OPEN | B | L676 is unchanged. | "22 of the 40 mapped patches (458 mentions)" | drift text |
| B11 | The M-d90 agent uses the shrunk variant, which the nested check did not select | OPEN | B | `w8_hardening.py`: VF = `q7_decayed90k100_s777`, stats `decayed90k100`. L232 and L415-425 do not say this. | State it in the names table and in the text. | drift text |
| B12 | "Combined standard errors 2.5 to 6" is undefined; "up to 0.011" is an SD, not a range | OPEN | B | L756-758 are unchanged. | Define the combination; write "SD up to 0.011 (range up to 0.021)". | drift text |
| B13 | The prereg description omits the analysis date and the C2 rule (z ≥ 1.645 one-sided) | OPEN | B | The draft has no analysis date (only a TeX comment at L908) and paraphrases C2 at L885. PREREG_FORWARD.md:76 has the rule. | Add both verbatim. | drift text |
| B14 | Fig. 2 uses ±1.96·SE despite df ≈ 3-5, and the caption omits the synergy panel | OPEN | B | `gen_figs.py:49,53` use 1.96; the caption (L607-611) is unchanged. | Use t quantiles and caption both panels. | drift text |
| B15a | "near 47% to 50%" | OPEN | B | L320 | "moved the short-history regimes to 46-50%" | drift text |
| B15b | "about a third of the advantage ... came from the leak" | OPEN | B | L538; +0.35 of +0.69 is about half | "a third to a half" | drift text |
| B15c | "fires on 14 of those 20 boundaries" leaves out that one fire is a no-change boundary | OPEN | B | L715 | Qualify. | drift text |
| B15d | "77 of 480 days" | OPEN | B | L725; the span is 506 days, so chance is 0.15 | Correct the denominator. | drift text |
| B15e | "a little over a week" rests on an unsourced 15K/week | OPEN | D | L651 | Cite the inflow source, or use the pinned rate (about 1.7 weeks). | drift text |
| C1 | The detector replay arm uses hindsight | OPEN | C | L705-711 still say "each lagged by its first-detection latency". | Call it a hindsight upper bound, and call detector vs calendar a tie. | drift text |
| C2 | Calibration aging is compared over different window lengths | OPEN | C | `r3_eval_vf.json` n_rows: 287,362 / 1,581,182 / 2,271,002. L627-630 are unchanged. | Score all three models on a common window. | drift text |
| C3 | Inference methods are mixed; W11 z values are pseudo-replicated | OPEN | C | L403-421 still give normal z values. | Rescore W11 with r8_score (t, df) and add "conditional on one VF per arm". | drift text |
| C4 | Numbers with no result artifact (z 2.5/1.7/3.3/3.5/3.1, 0.513-0.537, 0.20/0.22, 77/480, 13/2/81) | OPEN | C | No new files in `results/`. | Commit the scripts and JSON. | drift text |
| C5 | The escalation rule uses a measure the paper calls a lower bound | OPEN | C | L838-848 are unchanged. | One sentence giving 0.89pp rolling. | drift text |
| C6 | Interim looks at M-ref and resumes without the replay buffer are undisclosed | OPEN | C | Only the 1-yr cut is disclosed (L465). | One sentence in Limitations. | drift text |
| C7 | OOF threshold mismatch between train and deploy rows | OPEN | C | `rb_common.py`: K_FOLDS = 5; hero ≥ 20, pair ≥ pair_min, comp/map ≥ 5 on 80% of the counts during training. Magnitude untested. | Report missingness train vs deploy, or rerun one arm with K = 20. | drift text |
| C8 | The vintage section's interpretation rests on the leaky matchup | OPEN | C | L765 "What separates these two agents is largely current-meta knowledge" | Drop it, or rescore M_vs_U.json. | drift text |
| C9 | fig:dose, fig:decay and fig:vintage are never referenced | OPEN | C | Only `\ref{fig:scatter}` (L318) | Add the refs. | drift text |
| C10 | Arm names drift from the names table | OPEN | C | L353 "frozen-stats recipe", L398 "frozen-statistics recipe", L232 "causal per-build statistics" | As in the audit. | drift text |
| I1 | The Limitations example cites a paired comparison (0.029 vs 0.050) as a noise-level ordering | OPEN | B | L973-975 | Use 0.050 vs 0.067. | drift text |
| D1 | "date your judges, report them as a matrix, and put the headline on realized outcomes" | OPEN | D | L784 | Audit rewrite | drift text |
| D2 | "the trigger, the evaluator hygiene, and ..." | OPEN | D | L939 | Audit rewrite | drift text |
| D3 | "and never on a learned judge" | OPEN | D | L127 | Audit rewrite | drift text |
| D4 | "Retraining is the repair ... nearly worthless" | OPEN | D | L376-377 | Audit rewrite (0.02-0.04pp) | drift text |
| D5 | "The claim that survives is weaker and still useful" | OPEN | D | L322 | State the claim | drift text |
| D6 | "the detector is unremarkable" | OPEN | D | L703 | "performs like a calendar" | drift text |
| D7 | AI-disclosure double triad | OPEN | D | L1016, L1024 | Audit rewrite | drift text |

(The other audit §4 bullets duplicate A1, B6, B10, B11, C10 and C3/B14.)

### 6. NEW since the audit

| ID | Finding | Sev | Evidence | Fix | Owner |
|---|---|---|---|---|---|
| N1 | The tier paragraph overclaims: "so no result depends on the grouping" | C | f02ff4c, L156-158. The sensitivity check (`r14_tierfix.json`) retrains only the maintained value function and reports only accuracy and calibration. Degeneracy, agents, the detector and per-tier truth were not rerun. | "...leaves the maintained value function's accuracy and calibration unchanged; we did not rerun the policy-level experiments." | drift text |
| N2 | The tier relabel breaks re-runs of the truth scoring, including the registered C2 | B | The scorers group truth by DB `skill_tier`: `w12_clean_truth.py:64` (frozen for the prereg at 9dbc5df), `w13_counter_dig.py:55` and `r10_net.py:49`. Agents' draft records carry snapshot-scheme tiers. Since the 2026-09-30 relabel, the DB returns the site scheme, so "mid" drafts (Silver+Gold+Master) would be scored against "mid" truth (Gold+Platinum). Current results predate the relabel (truth cache 09-27; r8/r10 at 09-30 16:36-16:38), so they are unaffected. The risk is the forward C2 analysis and any B9 recompute. | Reconstruct the snapshot-scheme label from `league_tier` (or `replay_draft_skill_tier_backup_20260930`) in the forward/truth fetch. Disclose it as a third prereg deviation. | drift text (analysis code) |
| N3 | Net and the "2.3 / about 5 points of team win probability" conversions carry no structure terms | C | `r10_net.py` uses only hero WR, synergy and counter. c159669 found that realized indices without structure terms under-penalize degenerate teams by 4-6pp. The unmaintained side drafts more broken teams (29.7 vs 24.1), so Net probably understates the maintained advantage. That is conservative, but undisclosed. The naive vintage judges belong to the same family. | One sentence, or recompute Net with `gold.StructRealizedIndex`. | drift text |

**Cross-cutting notes (outside drift scope).**
- The `CONSENSUS_ORDER` key bug is limited to `qm2026/` (it is used by both `train_qm_wp.py` and `judge_2026.py`).
- `train_qm_wp.py` reads `src/lib/data/compositions.json` from the working tree, and that file is currently modified. It feeds only the unseen-composition summary there.
- The DB-tier relabel hazard in N2 applies to every research script that fetches `skill_tier` live rather than from a pinned cache.
