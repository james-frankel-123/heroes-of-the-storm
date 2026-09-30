# P3 personalization audit (2026-09-30)

Adversarial audit of the P3 pipeline (`P3_*.md` write-ups, `p3_*.py`, `cuda_personal/`) before paper and product use. Audit scripts and outputs: `/tmp/claude-1000/-home-max-heroes-of-the-storm/a49813e0-7f18-4a49-90f1-0426c355d609/scratchpad/p3audit/` (`base.py`, `e1`–`e8`, JSON outputs). All runs: 2 CPU cores (`taskset -c 48-49`, nice 19), no GPU, cached snapshot only, no DB writes, no existing files modified.

Severity: **A** changes a headline claim, **B** wrong number, **C** fragile or reviewer-bait.

## Verdict by headline claim

| # | claim | verdict |
|---|---|---|
| 1 | Skill estimates raise V2 accuracy 57.19% to 59.95%, log-loss gain +0.014 | **Survives.** Reproduced exactly (59.95%, +0.01418). No leak found in the lag-1 protocol. CI widens with player clustering: (0.0127, 0.0157). Production availability costs about 0.0004 (C2). |
| 2 | Experience offset; veteran −6.9pp on a never-played hero | **Survives as a predictive, pre-game feature.** It replicates out of sample, within player, in a joint 10-player regression and in established lobbies (B-list below). Causal wording ("costs") is not supported. |
| 3 | Pooling across similar heroes helps at small n | **Survives, but mostly mislabeled.** Most of the gain is the player's overall level, not hero similarity (B3). |
| 4 | 80% intervals are calibrated | **Holds for heroes with history. Not supported for never-played heroes** (A2): that coverage is measured on cells chosen by later play, which depends on outcomes. |
| 5 | Hero skill barely drifts; patches don't reset it | **"Barely drifts" is fragile** (C5). **"Patches don't reset it" is overstated** (A3). |
| 6 | Off-role penalty −0.9pp | **Wrong window.** The out-of-sample figure is −0.6 (Blizzard role) / −0.8 (fine role), ±0.2 (B2). |
| 7 | Clean at-game MMR adds almost nothing | **Survives on top of the skill model** (+0.0008). Not true alone: causal hero MMR alone gives +0.0028. The HERO_STRENGTH summary still carries leaky MMR numbers (B1). |
| 8 | Drafter: agreed-with teams beat the population by +7.6pp | **Circular (A1).** The skill model's own prediction for those real drafts is +7.6pp. The drafter adds +0.05pp (−2.8, +2.7) beyond it. |

---

## A. Issues that change a headline claim

### A1. The drafter's "realized value" is the skill model restated, not evidence for the drafter (claim 8)

**Mechanism.**
- Agreement is the percentile of the real pick in the drafter's ranking, which is driven by s(p, h) = experience offset + GP mean.
- The residual it is scored against (won − WP_pop) is exactly what s is trained to predict.
- So high-agreement teams are teams whose real picks the skill model already rates highly. The check re-tests claim 1 on 5,724 lobbies.

**Evidence** (`e5_drafter.py`). For each real V2 team, I computed the combined model's own predicted gap for the real draft, V_real − WP_pop, with the V1 combiner and the drafter's personal tables.

| agreement quintile | Q1 | Q2 | Q3 | Q4 | Q5 | Q5 − Q1 |
|---|---|---|---|---|---|---|
| realized residual (pp) | −3.2 | −2.8 | +0.3 | +1.2 | +4.5 | **+7.64** |
| skill model's predicted gap for the real draft (pp) | −3.9 | −1.6 | +0.4 | +1.3 | +3.7 | **+7.60** |
| never-played real picks per team | 0.46 | 0.32 | 0.25 | 0.22 | 0.15 | |

- **Realized minus predicted, Q5 − Q1:** +0.05pp (game-clustered 95% CI −2.8, +2.7).
- **Regression of the residual on agreement plus predicted gap:** the agreement slope is −0.9pp per unit (−7.2, +6.1); the gap coefficient is 0.98. Alone, the agreement slope is +18.9.
- **"Personal component" agreement:** realized +9.6 vs predicted +12.0; the drafter adds −2.4 (−5.1, +0.4).
- **MCTS** (`e8_mcts.py`):
  - 400 sims: realized +4.7 vs predicted +3.2 (excess +1.6, well within noise);
  - 1,500 sims: realized +5.34 vs predicted +5.35.
  - The "realized value rises with sims" curve therefore only shows that deeper search ranks real picks more like s does.
- **Never-played picks drive much of it.** Teams with 2+ never-played picks realize −10.6pp against a predicted −8.7; teams with none realize +1.5 against a predicted +1.3.

**What survives.**
- The combined model is well calibrated on these lobbies (gap coefficient 0.98). That is a nice confirmation of claim 1 on a high-history subset.
- Nothing in the realized check shows that the drafter's *choices* (the picks it would make and players did not) win more than the model says.
- The fork review found no outcome leak into V or into the ranking, and the kernel looks correct (C6).

**Fix.**
- Drop "model-free value" wording. Report the check as calibration of the skill model at team level.
- For drafter value, report the predicted gain as model-based and bound it by the skill model's calibration along the traded direction. Test calibration in the region the drafter exploits: real teams with high ΔS and low WP_pop, and b2 vs b1 in bins of |ΔS|.
- Build pools from pre-day history only, not the real hero. Do not force future real bans into earlier decisions (C6).

### A2. Coverage for never-played heroes is measured on outcome-selected cells (claim 4)

**Mechanism.** Coverage is scored on cells with ≥10 future games, and the (7.0pp)² never-played term is fit on cells with ≥5. For a newly taken-up hero, whether it gets played again depends heavily on the first outcomes (win-stay). The test population is therefore selected on the outcomes it is scored on.

**Evidence** (V2 cells with no history before the V1/V2 split):

| future games in V2 | cells | first-game raw residual | cell mean r_adj |
|---|---|---|---|
| 1 | 93,294 | −5.9pp | −3.4pp |
| 2-4 | 29,392 | +5.0 | +1.9 |
| 5-9 | 4,670 | +10.6 | +4.7 |
| 10-29 | 1,290 | +13.0 | +5.2 |
| 30+ | 117 | +11.6 | +7.6 |

- The "0.79 calibrated coverage on never-played heroes" is coverage for heroes the player will go on to play 10+ times.
- That group runs about 5 to 13pp above the population of first-time picks. Most of the fitted 7pp extra sd is absorbing this selection, not honest uncertainty.
- The product question "should I pick this?" is asked before any of this is known.

**Fix.**
- Evaluate never-played predictions on every newly played cell, whatever it goes on to do. Use the first k games (k = 1 to 3) with a proper score: binomial log loss, or PIT on the sum of outcomes.
- Refit the display band on that population.
- Report coverage for heroes with history separately. That part (raw 0.76 to 0.83) is much less exposed, although nf ≥ 10 still conditions on continued play.

### A3. "Patches don't reset hero skill" is not supported, only "no jump detected in-sample" (claim 5)

This comes from the fork review of `p3_ph_patch*.py`; the code paths were verified by hand.

- **The jump likelihood test is biased toward zero.**
  - It runs on E-window residuals, which are in-sample for the WP. The residual variance ratio vs wp(1−wp) is 0.983 in E and 1.000 in V (checked).
  - The filter still assumes wp(1−wp) noise, so any added variance is penalized. The placebo heroes are penalized more than the treated ones (−31.4 vs −22.1 log-lik at 1pp), which is a symptom of misspecification, not of power.
  - Section C also passes `r_adj` to `JumpFilter` (`p3_ph_patch.py:408/435/451`), not the (build, hero)-demeaned `u` the docstring and write-up describe.
- **The two discriminating results point the other way.**
  - Magnitude: excess change variance rises +10.1pp² per pp of |ΔWR|, CI (1.8, 19.4), and stability falls to 0.23 at 2 to 4pp shifts.
  - Out of sample after 2.55.16: variance ratio 1.41 on patched vs 1.11 on unpatched heroes. At a claimed sd of about 4.6pp, that is roughly 6pp² of extra variance (about 2.5pp sd). This contradicts "any real jump is below about 1pp".
- **The split-half DiD has almost no power.** The stability difference CI is (−0.44, +0.34).

**Fix.**
- Reword to "no jump detected in-sample; out of sample, post-patch errors on patched heroes are about 1.3x larger, and reshuffling grows with the size of the aggregate shift".
- Rerun the jump test on out-of-sample residuals (post-cutoff boundaries) with the population part removed and a fitted noise scale. Compare treated vs placebo per crossing.

---

## B. Wrong numbers

### B1. Same-day experience counts leak through the stopping rule (SKILL_DRIFT, VALIDITY, PATCH_AND_ROLE absolute gains; drafter combiner)

`prepare()` → `experience_counts` orders same-day games by replay_id, which is upload order. That offset is used in `p3_sd_eval.py:232`, `p3_sd_order.py:240`, `p3_val_validity.py:199` and `p3_ph_role.py:142`. SKILL_DRIFT 1b asserts that counts "carry no outcomes". They do.

**Players continue after wins** (`e2_sameday.py`, V slots):
- P(another game later the same day | win) is 51.9%, against 49.0% after a loss.
- The residual is +1.33pp when a later game exists and −1.35pp when this is the last game of the day.
- Slots whose replay_id count includes games not strictly earlier (16.5% of V slots) have residual +1.24pp, against −0.25pp for the rest.

| same-day rule for counts (V2, game level) | experience only | experience + skill (static) |
|---|---|---|
| lag 1 day (phase-1 headline) | +0.0082 | +0.0142 |
| strict (ended before start and lower replay_id) | +0.0089 | +0.0148 |
| play-time order | +0.0091 | +0.0149 |
| replay_id order (as shipped in phase 2) | +0.0098 | **+0.0156** |
| every same-day game, including later ones (leak control) | +0.0126 | +0.0179 |

**Corrections:**
- SKILL_DRIFT "EB static / state-space +0.0156": **+0.0148**.
- Same-day state-space +0.0168: about **+0.0160** (estimate: shift by the same −0.0008).
- VALIDITY's +0.0156 / +0.0143: about −0.0008 each.
- The paired drift-vs-static differences share the same offset, so they are roughly unaffected.
- The claim "+0.0016 from same-day counts, known at draft time" becomes +0.0007 legitimately.
- The drafter's combiner (b2 = 3.71, from `p3_ph_role`) was fit on the leaky replay_id counts but is served lag-1 counts. Refit on lag-1 counts gives b2 = 3.62. The effect is minor.

**Fix:** use the strict rule for counts everywhere, or lag 1 day, and refit the combiners.

### B2. Off-role penalty headline uses the in-sample window (claim 6)

- The −0.9pp comes from "2025-26", which is mostly E: the WP is in-sample there, and the kernel and offset were fit on it.
- The post-cutoff values in the same table are **−0.6 (±0.2) Blizzard role** and **−0.8 (±0.2) fine role**.
- Those SEs are iid by slot. With the game-level design effect found here (variance ×1.7), they are about ±0.26.
- The "after model" residual also uses the leaky replay_id offset (B1).
- The team-level +0.00044 is fine: it was fit on V1 and tested on V2.

**Fix:** headline −0.6 to −0.8pp with clustered SEs, and recompute with strict counts.

### B3. "Pooling across similar heroes" is mostly pooling to the player's overall level (claim 3)

- The no-pooling baseline has no player term, so "CF − no pooling" (+0.43 to +0.57 per 1000 slots at n = 0 to 10) mixes overall skill with hero similarity.
- The similarity part alone (CF − player+hero):
  - n = 0: +0.07 (−0.02, 0.16), not significant;
  - n = 1 to 10: +0.16 to +0.20 per 1000;
  - game level: +0.0010 (about ±0.0005 after clustering).
- **Fix:** state it as "the player's overall level carries most of the small-n gain; hero similarity adds +0.001".

### B4. Stale leaky MMR numbers remain in the P3_HERO_STRENGTH summary (claim 7)

- The summary bullets still say "Adding lagged MMR brings it to 60.81% and +0.0201" and "hero MMR as of the previous day ... +0.0088". Both are naive lagged MMR, which the file's own section-6 note says is leaky.
- Clean causal MMR from `mmr_at_game.npz` (`e7_mmr.py`, V1 fit, V2 test):

| | gain vs M0 |
|---|---|
| causal player MMR alone | +0.0005 |
| causal role MMR alone | +0.0012 |
| causal hero MMR alone | +0.0028 |
| causal player + role + hero alone | +0.0027 |
| skill model (headline) | +0.0142 |
| skill + causal player MMR | +0.0146 |
| **skill + causal player/role/hero MMR** | **+0.0149 (60.07%)**, i.e. +0.0008 over skill |
| own-game stamp (leak control) | +0.0351 |

- Claim 7 holds on top of the skill model (+0.0008). "Almost nothing" is wrong for hero MMR alone: +0.0028 is about a third of the experience offset.
- The fork suspected that the "upload-order-safe" lagged MMR (+0.0017 at G = 7) still leaks, because parse order is not upload order. At game level it is below clean causal MMR (+0.0027), so I found no evidence of a remaining large leak. It is not ruled out either.
- **Fix:** replace the summary numbers with the causal ones above.

### B5. Confidence intervals ignore shared players

- Game-level CIs are replay bootstraps. The same players recur across V2 games.
- A shared-player sandwich estimate (`e6_cluster.py`) gives variance design effects of 1.65 (offset only) and 1.71 (headline):
  - headline +0.0142: CI (0.0131, 0.0152) becomes **(0.0127, 0.0157)**;
  - a day-block bootstrap gives (0.0132, 0.0152).
- Paired differences (CF vs player+hero, role terms, momentum arms) should widen about 1.3x. None of the main ones flips.
- The drafter Q5 − Q1 bootstraps teams independently, although both teams of a game have mirror residuals. Game-clustered: (4.8, 10.3).

---

## C. Fragile or reviewer-bait

### C1. In-sample WP residuals in E
- Residual variance / wp(1−wp) is 0.983 in E and 1.000 in V.
- Every variance-component fit (kernels, Kalman drift, patch jump, calibration scale 1.20) assumes wp(1−wp) noise on E residuals. That biases signal variances and drift rates toward zero.
- The experience-offset table is not affected materially. Refit on V1 only (out of sample): the 300+,0 cell is −7.2 vs −6.9, and V2 gains are unchanged (+0.0081 / +0.0139).
- **Fix:** fit a noise scale jointly, or refit the variance components on out-of-sample residuals.

### C2. Availability at prediction time
- The lag-1 state uses every stored game from earlier days, including games uploaded after the predicted game. This affects 70% of V slots; the mean count is 245 vs 237 available.
- Restricting counts to games with a lower replay_id: offset alone +0.0078 (vs +0.0082), headline +0.0138 (vs +0.0142). The GP state was not recomputed; expect a small further drop.
- This is not an outcome leak, but the production number should use it.

### C3. What the counts mean
- n_p counts uploaded games since 2024-04, so "veteran" means heavily recorded.
- Uploader proxy: the team of the lobby's most-recorded player beats the WP by +0.36pp (±0.05) in E and +0.32 (±0.13) in V. That is mild upload selection, too small to explain the offset.
- New low-level accounts run +3.5 to +5.9pp (VALIDITY), while the offset gives n_p = 0 −1.6 (V2) to −2.8 (E). The sign is wrong for smurfs.
- Product copy should not say "a never-played hero **costs** 7pp". Say "players with your recorded volume average −7 to −8pp on heroes we have never seen them play".

**Offset robustness (why claim 2 survives).** All values are the veteran never-played cell, or its contrast with 50+ games:

| test | value |
|---|---|
| E table (as shipped) | −6.9 |
| V1 realized | −7.5 (±0.8) |
| V2 realized | −8.0 (±0.7) |
| within player, V2 veterans, n_ph = 0 minus 50+ | −9.6 |
| joint 10-player game-level regression, 0 vs 50+ contrast | −9.3 (E) / −9.9 (V), against the marginal −9.3 |
| lobbies with all 10 players ≥ 20 games (26% of V2) | −8.6 |
| lobbies with all 10 players ≥ 50 games | −9.7 |
| table from 2025+ E only (window-start check) | −7.0; V2 gain unchanged |

- Offset-only game gains in those lobby subsets are larger than overall (+0.0113 and +0.0121).
- The strict test "all 10 long-established" cannot be run: only 0.4% of V2 lobbies have all ten players first seen before 2024-10.
- Teammates, upload selection, window start and WP in-sample fitting do not explain the effect.

### C4. Multiple comparisons and forking paths
- About 60 combiner specs are scored on V2. The headline kernel was pre-chosen on E held-out likelihood, so this is fine for claim 1.
- The calibration form (a never-played term) was chosen after looking at V2 raw coverage by n.
- Per-hero similarity results include "17 of 90 heroes transfer more" (real in aggregate). The single "significant" negative, The Butcher (CI −1.17 to −0.07), and Qhira in the specific version are at chance level for 90 two-sided tests (about 2 expected per tail). Do not name them.

### C5. Drift claims
- The hero-specific rate hit its lower bound (a half-life over 190 years). The CF half-lives (4 to 7 years) are Ornstein-Uhlenbeck extrapolations from a roughly 680-day window, with no CI. Only "correlation 0.90 to 0.93 at 410 days" is identified.
- "No drift" holds after the count-based offset, which is itself a learning or selection curve (−6.9 to +2.3pp over 0 to 50+ games).
- The state-space variance ratio "1.05 overall" averages bins from 0.35 to 2.94.

### C6. Drafter and MCTS harness (fork review; no numeric bug found)
- **Correct:** leaf value, sign and relabeling, pick-to-slot mapping (SL order), and the pick-to-player mapping.
- **Game information leaks into the simulation** (not the outcome):
  - pools include the hero actually played (`p3_dr_core.py`, `pool[..., d["hero"][qidx]] = True`);
  - future real bans are forced at earlier decisions.
- `p3_mcts_verify.py` tests the leaf value and population identity only. Pools, forced bans, decide-only prefix replay, the self-play team switch, the imitation opponent and `Setup.lobby()` relabeling are untested.
- One-step and MCTS "personal component" definitions differ, so those rows are not comparable.
- The imitation recency features include same-day games, although the docs say they use the previous day.
- The combiner and imitation fits are on V1 only, so both are clean.

### C7. Game order
- Lag-1 protocols are safe: history is days ≤ t−1 by UTC `game_date`, which is the game end.
- There are no duplicate-replay leaks: only 1 pair of identical 10-player lobbies ends within 10 minutes. The 902 repeated lobby sets are rematches.
- The ordering problems are confined to the same-day variants (B1) and the role-share counts, which are time-ordered but not strict.

---

## Reproduced numbers

| quantity | write-up | audit |
|---|---|---|
| M0 raw WP, V2 accuracy / log loss | 57.19 / 0.67647 | 57.19 / 0.67647 |
| offset only (lag 1) | +0.0082, 59.01% | +0.00816, 59.01% |
| headline, CF rank 2 online | +0.0142, 59.95% | +0.01418, 59.95% |
| phase-2 static, same-day | +0.01557 | +0.01557 (replay_id counts); +0.0148 strict |
| drafter Q5 − Q1 | +7.6 | +7.64; predicted by the skill model +7.60 |
| MCTS 400 / 1,500 Q5 − Q1 | +4.7 / +5.3 | +4.73 / +5.34; predicted +3.15 / +5.35 |

The GP state is the cached `sd_pred_static_lag1.npz`. Combined with the lag-1 offset, it reproduces the phase-1 headline to four digits.

## Priority fixes before the paper
1. Reframe claim 8 (A1). The realized check validates the skill model, not the drafter.
2. Re-evaluate never-played coverage on unselected cells and refit the band (A2).
3. Reword the patch claim and rerun the jump test on out-of-sample residuals with the demeaned input (A3).
4. Use strict or lag-1 counts everywhere and refit the phase-2 numbers and combiners (B1).
5. Use the out-of-sample off-role number (B2). Replace the stale MMR numbers (B4). Use clustered CIs (B5).
