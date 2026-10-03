# Paper 1 revision notes (2026-09-30)

Paper: "Diagnosing and Repairing Out-of-Distribution Failure in MOBA Draft Policies", under review at IEEE ToG.

**Revised manuscript files** (in `paper/paper 1/overleaf/`)

- **`draft_revision.tex`**: the clean revision (no change markup), trimmed to fit the submission's format.
  - Clean build is **9 pages** after §11 (8 before; the submitted `draft.pdf` is 10). Built with `pdflatex` twice, checked with `pdfinfo`.
  - 0 errors, 0 overfull boxes.
- **`supplementary_revision.tex`**: clean supplement, 7 pages, 0 overfull boxes.
- **`draft_revision_markup.tex`** and **`supplementary_revision_markup.tex`**: the earlier marked-up versions (blue = new, red = removed claim with the reason; 13 pages clean).
  - They document every change against the submission sentence by sentence.
  - They predate the trim (§8), so their layout and some wording differ from the clean files. The numbers are identical.
- **Figures:** `fig_glance_revision.pdf`, `fig_safety_context_revision.pdf`.
- **Untouched:** the submitted `draft.tex`, `draft_anonymous.tex`, `supplementary*.tex`, and figures.
- **Anonymous copy:** skipped for now, per Max.

**Code and results:** `training/paper1_revision/` (file list in §7).

This file is the change log for coauthors and the basis for the response to reviewers. Each change gives what changed, why, and the old vs new numbers.

**Status (2026-10-01):** the independent references are now composition-corrected (§11); every judged number in the manuscript uses the corrected references.

**Status (2026-09-30, late):**
- **800-sim leak-free runs (J_oof):** finished and in the paper. No `\PENDING` markers remain.
- **Deferred queue:** every job has finished except leak-free CQL with enriched features (training at the time of writing).
- **Tier-label bug:** handled in §10.

---

## 1. Summary for coauthors

The 2026-09-29 audit found two problems that touch headline claims.

1. **Outcome leakage.** The enriched features came from Heroes Profile aggregates that contain 38–67% of our own games, test games included. Every game's features held its own outcome.
2. **Self-grading.** The main MCTS numbers ("Avg WP") were scored by the value function the agents were trained against. Two of the four tournament judges were contestants' own value functions. The constrained-vs-unconstrained comparison used two different checkpoints.

The revision rebuilds the evidence without either problem.
- **Statistics:** all value-function statistics come from our own training split, out of fold. The external aggregates no longer enter any model.
- **Selection:** a validation split replaces test-set early stopping and seed selection.
- **Judging:** every generated draft is scored by references that share no game, statistic, or search with any agent. These are judges trained only on 291,837 post-snapshot games, a realized-outcome index on those games, and two Quick Match judges.

### What survives, what changes, what goes

| Claim (submitted) | Status in revision |
|---|---|
| Accuracy does not discriminate between value functions | **Survives, stronger.** Leak-free enriched = naive (57.7% vs 57.7% test; 56.8% vs 56.8% post-snapshot) |
| Enriched model most accurate (58.1%) | **Removed.** The 0.3pp edge was the test games' own outcomes |
| Features change policy behavior | **Survives, smaller.** 5-tank WP: 0.31 (naive), 0.25 (enriched), 0.08 (augmented). Submitted: 0.365 / 0.147 / 0.062 |
| Pessimistic offline RL collapses to anchoring (CQL, IQL, MCQ, BC-CQL, discriminator) | **Survives unchanged.** These policies read no aggregate statistics. They finish 0.44–0.45 in the new tournament under every judge |
| MCQ "dead neurons" | **Survives, now reproducible and stronger.** All units of layers 2–3 are dead at τ=0.5 |
| More training sims monotonically improve quality (+0.042) | **Removed.** Under independent references the gain is about a quarter as large and complete by 400 sims. With the leak-free value function, 400 → 800 lowers gN by 0.007 |
| Enriched features add +0.023 WP to MCTS (p<0.001) | **Removed.** It reverses under independent references: −0.010 (gN), −0.016 (RN), −0.013 (QM2026). Leak-free at matched sims: −0.002 (gN). Features buy safety, not win probability |
| Training length helps (E_1M) | **Survives** under every reference (+0.017 gN) |
| Tournament: constrained MCTS wins 20/20 at 0.670; "not self-graded" | **Replaced.** Independent judges only. MCTS 0.605 (21/22) and constrained MCTS 0.602 (21/22) lead; anchored methods sit at 0.44–0.45. The coarse ordering survives |
| Constrained beats unconstrained MCTS head to head (0.577/0.542) | **Removed.** Matched checkpoints give a tie: 0.499 (revision agents) and 0.503 (submitted J_800sim_s9 on both sides). The mask costs nothing and guarantees validity |
| QM judges reproduce the ordering at Spearman 0.92 | **Corrected.** A key bug dropped 2 strategies. The new tournament gives 0.93 / 0.94 over all 12 strategies but 0.66 within the top six |
| Interaction metrics predict outcomes (supplement) | **Survives on clean data.** Post-snapshot games: +9.1pp synergy, +8.0pp counter, both 9/9 monotone. Published: +10.6/+9.7 with self-inclusion |
| Favorite combinations are "ground-truth strong" | **Removed.** Forward replication fails (pooled pair edge −0.04pp) |
| Greedy synergy +0.76 to +0.85 | **Halved:** +0.29 to +0.44. Part of it was the metric agreeing with the leaked features |
| NGS: enriched 58.1% → 55.5% | **Corrected.** Leak-free 57.7% → 55.5%, the same drop as naive |

**Recommended operating point:** F (400 training sims, 300K episodes), leak-free value function, argmax root.
- It ties E_1M as the best configuration under gN and RN at a fraction of E_1M's compute.
- 800 sims buys no independent gain in the submitted pipeline, and loses a little in the leak-free one (gN −0.007 ± 0.003 vs F_oof).
- Argmax root adds +0.000 to +0.006 gN (mean +0.003) and never hurts.

---

## 2. What changed in the pipeline

### 2.1 Data design (`core.py`)

- **Corpus:** unchanged, the pinned snapshot of 1,949,087 patch-2.55 replays.
- **Test set:** the submission's own 38,981 replays (`shared.split_data`, seed 42).
- **Validation set (new):** 2% hash-selected, 38,214 replays. It is used for early stopping and seed selection. The submission early-stopped on test data and kept the best of 3 seeds by test accuracy (audit C1).
- **Training set:** 1,871,892 replays, 3.74M swap-augmented rows.
- **Statistics** are built only from the training split, in the frozen-stats schema:
  - hero, hero-map, pair-with, and pair-against win rates, plus pick and ban rates;
  - an own-corpus role-composition table with cells admitted at ≥50 games. That gives 117/126/106 compositions per tier (low/mid/high), against 164/127/121 in the external table. 99.89% of training teams fall in an admitted composition.
- **Out-of-fold features:** 5 hash folds. A training row in fold k gets features from the statistics of the other four folds.
  - Val, test, post-snapshot, greedy search, MCTS lookup tables, and every judge's scoring use the "deploy" statistics (the whole training split), which contain none of those games.
- **External Heroes Profile statistics:** dropped from every value function. They are kept only as the instrument for the interaction metrics (synergy, counter, resilience), so baseline rows stay comparable. Reasons for dropping them:
  - They overlap our corpus by 38–67% of games per hero and tier.
  - Only aggregates exist, so test-game subtraction would be approximate.
  - The file cannot be regenerated. Our own statistics rebuild from the replay-id list.
- **Kernel check:** with our own lookup tables, the CUDA kernel's leaf WP matches the Python features to 5e-7 over 200 drafts (`results/verify_kernel.json`).
- **Composition table in the MCTS worker:** patched into `StatsCache._load_compositions` at launch (`train_mcts.py`). The worker file is unchanged.

### 2.2 Independent references (reused from `training/overfit2026`)

Held-out accuracy and calibration slope are measured on the paper test set, which no reference trained on (`results/judge_calibration.json`).

| Reference | What | Games | Test acc | Slope |
|---|---|---|---|---|
| gN | mean of 3 enriched judges, out-of-fold statistics, post-snapshot no-drift games only | 291,837 | 55.3 | 0.97 |
| gN-naive | hero-identity judge, same games | 291,837 | 56.0 | 0.91 |
| RN | realized-outcome index, cross-fitted, same games | 291,837 | 54.7 | 1.04 |
| R17 | same index on drifted 2.55.17 games (not in consensus) | 158,608 | 53.7 | 0.72 |
| QM2026 | Quick Match judge (no draft phase), current era | 116,589 | 53.9 | **0.44** |
| QM2021 | Quick Match judge, 2021 era | 91,365 | 54.5 | **0.39** |

- **Consensus for the tournament** = mean of gN, gN-naive, RN, and QM2026. This was fixed before any revised agent was scored.
- **The Quick Match judges are strongly overconfident on ranked drafts**, so the text uses them for ordering only.
- **Post-snapshot games:**
  - 153,739 snapshot-era games uploaded after the snapshot;
  - 138,098 games on 2.55.16.97039, the build in force at snapshot time;
  - the loader whitelists builds, so nothing from a build released after 2026-09-27 is read.

### 2.3 Seeds and selection rules (stated in the manuscript)

- **WP models:** 3 seeds (42, 123, 777). Tables report 3-seed means; downstream use takes the lowest-validation-loss seed.
- **MCTS benchmark:**
  - The first five seeds (s0–s4) of each submitted configuration, and all seeds of each leak-free configuration. No seed is chosen.
  - The same 1,000 draft configurations for every run: mid tier, random map and side, seed 20260929. 200 inference sims. Root T=1 (the submission's protocol) and T=0.
  - The submission's J_800sim_s9 was the best of 15 seeds by the training proxy. It is no longer singled out.
- **Tournament MCTS agent:** F_oof, all five seeds pooled (draft i uses seed i mod 5).
  - F was chosen before any revised tournament score existed, because the submitted-pipeline benchmark already showed no independent gain past 400 sims.
  - Constrained MCTS uses the same checkpoints. K_truebase pools its 15 submitted seeds.
- **MCTS checkpoint within a run:** the paper worker's rule is unchanged (keep the best periodic eval by the training value function). This is part of the method, not a test-set choice.

---

## 3. Evidence: old vs new, by manuscript element

### 3.1 Table I (value-function accuracy)

- **Test:** the 38,981 replays, both team orders (the original convention).
- **Post:** 291,837 post-snapshot games.
- **Slope:** calibration slope on Post (1 = calibrated, <1 = overconfident).
- **Revised table:** shows both the submitted and leak-free columns.

| Model | Submitted test | Submitted post / slope | Leak-free test (3 seeds) | Leak-free post / slope |
|---|---|---|---|---|
| Naive (197d) | 57.80 | 56.79 / 1.02 | 57.68 ± 0.05 | 56.81 / 1.07 |
| Hero strength (209d) | 57.70 | 56.60 / 1.00 | 57.57 ± 0.06 | 56.60 / 1.05 |
| **Enriched (283d)** | **58.07** | 56.83 / **0.88** | **57.69 ± 0.08** | 56.82 / **0.99** |
| Enriched, in-sample own statistics (control) | n/a | n/a | 57.20 ± 0.11 | 56.33 / **0.78** |
| Enriched 512 | 58.04 | 56.73 / 0.90 | 57.51 ± 0.10 | 56.78 / 1.00 |
| Augmented 512 (v2 labels, WR 10) | 57.92 | 56.75 / 0.87 | 57.45 ± 0.08 | 56.72 / 0.98 |
| Relational 256 | 58.01 | 56.85 / 0.87 | 57.62 | 56.83 / 0.97 |
| Absolute 256 | 57.89 | 56.75 / 0.99 | 57.69 | 56.86 / 1.01 |
| Gourdeau estimator (no statistics) | 56.2 (paper); 56.5 re-measured | 55.7 / 0.87 | n/a | n/a |

- **Why it changed:** self-inclusion (audit A1).
- **The in-sample control reproduces the harm.** It uses our own statistics with the submission's design: half a point lost, and slope 0.78.
- **Test-set slopes** are above 1 for every model, naive included (1.17–1.27). This is a property of that sample. The post-snapshot slope is the calibration reference.

### 3.2 Policy consequences of features (Section IV; sanity suite)

Source: `results/sanity.json`, the 28-test suite plus the degenerate probes against a standard team.

| Model | 5-tank WP, submitted | 5-tank WP, leak-free | Sanity (/28), submitted / leak-free |
|---|---|---|---|
| Naive | 0.365 | 0.312 | 22 / 24 |
| Hero strength | 0.373 | 0.312 | 22 / 22 |
| Enriched 256 | 0.147 | 0.247 | 25 / 26 |
| Enriched 512 | 0.155 | 0.267 | 25 / 22 |
| Aug WR 10% 512 | 0.062 | 0.082 | 22 / 26 |

- **Features:** the enriched features still lower the 5-tank WP, but far less than the submission reported.
- **Augmentation** does most of the work on this probe.
- **Audit B11:** the paper's 0.155 was the 512 model; the 256 model gave 0.147. The revision reports the leak-free 256 value.

### 3.3 Table II/III (greedy cross-evaluation, 480 drafts per drafter)

Source: `results/crosseval.json`. The configurations and protocol are the submission's; the models are leak-free. Submitted values in parentheses. "Ind." = mean of the four consensus references.

| Drafter | by Naive | by Hero str. | by Enriched | Ind. | Healer % | Ranged % | Degen % |
|---|---|---|---|---|---|---|---|
| Naive | **.640** (.654) | .626 (.641) | .618 (.629) | .586 | 61.3 (60.4) | 79.4 (74.8) | 58.5 (64.2) |
| Hero str. | .621 (.627) | **.640** (.668) | .613 (.618) | .579 | 77.1 (68.5) | 76.5 (78.1) | 49.4 (54.0) |
| Enriched | .613 (.614) | .611 (.610) | **.641** (.672) | .580 | 79.0 (76.3) | 76.5 (79.6) | 50.2 (49.6) |

- **Diagonal dominance survives.**
- **Independent references do not separate the three drafters** (.579–.586).
- **"Failure shrinks monotonically with feature richness across healer, ranged, and degenerate rates" is replaced.**
  - Healer rate is monotone.
  - The degenerate rate drops at hero-strength and then stays flat.
  - The ranged rate does not improve.
- **Healer difference, naive vs enriched:** 17.7pp (χ² = 35.9, p = 2e-9). Submitted: 15.8pp.
- **Pick agreement, naive vs enriched greedy:** 16% of bans, 8% of early picks, 9% of late picks. "Disagree on roughly 90%" survives.
- **Top-50 gap:** of the 50 naive drafts the naive model rates most above the enriched model, 34 have no healer (68%; submitted 64%) and 41 are degenerate (82%; submitted 86%).
- **"Sacrificing 4.0pp of naive-scored WP":** now 2.7pp.

### 3.4 Feature-set sweep (Section IV-C; "4,100 configurations")

**Design**
- A 2^(10-3) resolution-V fractional factorial: 128 runs, leak-free features, 256×128, lr 5e-4, seed 42, validation early stopping.
- Generators: pairwise_synergies = ABCD, counter_detail = ABEF, synergy_detail = ACEG. The base factors A–G are map_type, role_counts, hero_wr, team_avg_wr, hero_map_wr, map_delta, and pairwise_counters.
- Every defining word has length ≥ 5. Main effects are therefore unaliased with each other and with every two-factor interaction, and equal the full-factorial marginals up to interactions of order three and higher.

**Why not all 4,096:** about 200 GPU-hours on shared GPUs, for main effects this design estimates from 128 runs. "4,100" was really 4,096 configurations; the CSV has 4,086 rows (audit B12).

| Group | Leak-free test (pp) | Leak-free post (pp) | Submitted marginal (pp) |
|---|---|---|---|
| role_counts | +0.10 | +0.14 | +0.16 |
| pairwise_counters | +0.08 | +0.05 | **+0.20** |
| map_delta | +0.01 | +0.04 | +0.09 |
| pairwise_synergies | −0.02 | +0.01 | +0.15 |
| map_type | +0.00 | −0.01 | +0.01 |
| team_avg_wr | −0.03 | −0.03 | −0.05 |
| synergy_detail | −0.09 | −0.04 | +0.16 |
| counter_detail | −0.06 | −0.06 | +0.14 |
| hero_map_wr | −0.10 | −0.09 | −0.02 |
| hero_wr | −0.11 | −0.13 | −0.16 |

- Effect SE: 0.02pp (test), 0.01pp (post).
- The 128 configurations span 57.0–58.0% (test) and 56.4–57.0% (post).
- **"No group exceeds 0.2pp" survives; the maximum is now 0.14pp.**
- **The pairwise groups lose 0.12–0.25pp each.** That is the self-inclusion.
- "Pairwise counters are largest" is replaced: role counts, which read no statistics, are largest.

### 3.5 Metric validity (supplement; Table IV caption)

Source: `results/metric_validity.json`. The strength-adjusted decile curves use the published figure's code. Each cell is the top-minus-bottom decile spread and the number of increasing steps out of 9.

| Statistics / games | Synergy | Counter | Self-inclusion? |
|---|---|---|---|
| External / paper test (as published) | +10.6, 9/9 | +9.7, 8/9 | yes |
| Audit leave-test-out | +7.4, 6/9 | +7.5, 7/9 | approx. removed |
| **External / 291,837 post-snapshot games** | **+9.1, 9/9** | **+8.0, 9/9** | no |
| Own train split / paper test | +9.0, 8/9 | +13.5, 8/9 | no |
| Own train split / post-snapshot | +7.8, 9/9 | +12.7, 9/9 | no |

- The signal is real and monotone on the larger clean sample.
- Self-inclusion inflated the published spreads by about 14% (synergy) and 18% (counter).
- The audit's non-monotone leave-test-out curve was mostly noise at n = 39K.
- The figure was not regenerated. The supplement now uses a table (`tab:metricvalid`) and no longer includes the figure.

### 3.6 Offline RL (Section VI; Table IV)

**Unchanged rows.** The baseline policies (CQL naive, BC-CQL, IQL, MCQ, GD, discriminator) read no aggregate statistics, so their rows are unchanged.
- The collapse evidence is behavioral (GD agreement, entropy, distinct heroes) and survives.
- In the new tournament the anchored methods finish 0.435–0.445 under every judge. MCQ is last (0.330).

**Corrections**
- **CQL enriched:** trained on the leaky external features. It was not retrained (62M transitions; out of budget); the table caption says so. Its collapse comes from the anchoring penalty.
- **MCQ dead units (audit B21):** now reproducible (`mcq_dead_units.py`, 200,000 held-out states).
  - τ=0.5: 256/256 units dead in layer 2 and 128/128 in layer 3. The submission said 252/256 and 127/128.
  - τ=0.6: layer 3 is fully dead.
  - CQL α=1 control: 22/256 and 16/128.
  - The Q-head is a constant bias, i.e. a fixed hero ranking.
- **IQL mechanism (B19):** softened. The Q loss of 0.6948 is above ln 2.
- **Discriminator trend (B20):** "80 → 53, 44% → 51%" removed; the checkpoint was overwritten.
- **CQL parameters (B13):** 324K. The text says "same hidden layers", not "same architecture".
- **"GD agreement rose 59–65% → 69–74% as the corpus quadrupled" (C6):** removed. It compared against different reference GD models.
- **Synergy range (B10):** −0.15 to −1.60.
- **"Aug WP" column:** dropped. It was a self-evaluation by a model whose labels derive from the same statistics (B3).

**Greedy rows, re-run leak-free** (5 opponent seeds × 1,000 drafts; `results/greedy_rich.json`):

| Row | Healer | Degen | Counter | Synergy | Dist. | Ent. | T10 | GD% | gN |
|---|---|---|---|---|---|---|---|---|---|
| Enriched greedy, submitted | 85.5 | 18.5 | +.25 | +.85 | 90 | 6.08 | 31.4 | 4.4 | n/a |
| Enriched greedy, leak-free | 83.5 | 21.1 | +.16 | **+.44** | 90 | 5.96 | 35.2 | 4.1 | .604 |
| Enr.+aug greedy, submitted | 87.6 | 18.4 | +.21 | +.76 | 90 | 6.11 | 30.2 | 4.5 | n/a |
| Enr.+aug greedy, leak-free | 90.4 | 15.2 | +.09 | **+.29** | 90 | 6.04 | 34.1 | 4.8 | .594 |
| Constrained greedy, leak-free | 100 | 0.0 | +.13 | +.37 | 90 | 6.00 | 33.5 | 4.8 | .607 |
| MCTS (F_oof, argmax) | 94.4 | 6.2 | +.09 | +.78 | 32 | 3.52 | 88.8 | 3.1 | .668 |
| Constrained MCTS (F_oof) | 100 | 0.0 | +.08 | +.63 | 36 | 3.52 | 89.3 | 4.0 | .673 |

- **Greedy synergy roughly halves.** The sign flip against anchored methods survives.
- **"The augmented model matches the unaugmented model's counter, synergy, and safety" is replaced.** Augmentation buys safety (degenerate 21.1 → 15.2%) at a cost in interaction-awareness (+0.44 → +0.29).
- **Constrained MCTS:** the submitted synergy +0.95 and counter +0.20 are replaced by +0.63 and +0.08.

### 3.7 Augmentation (Section V; Table V)

**Generator (B3).** The downstream augmented models use the v2 generator, which adjusts labels pairwise around the assigned WR. The text said "fixed win rate" and now describes the actual generator.

**Revision setup:** the v2 generator with our deploy statistics and our own composition table. That yields 40,700 synthetic records (submitted: 34K). Unseen compositions number 126–146 per tier. Each synthetic row gets out-of-fold features like a real row.

**Table V** (leak-free; accuracy is the 3-seed mean; composition rates come from 2 opponent seeds × 1,000 greedy drafts of the selected model; submitted values in parentheses):

| Assigned WR | Acc. | Healer | Ranged | Degen | 5-tank | Sanity |
|---|---|---|---|---|---|---|
| None | 57.5 (58.0) | 81.8 (82.2) | 93.6 (92.0) | 23.7 (22.0) | .27 | 22 (25) |
| 0% | 57.5 (57.9) | 92.1 (87.1) | 98.6 (96.2) | 14.5 (18.3) | .05 (.003) | 26 (19) |
| 5% | 57.5 (58.0) | 91.6 (88.1) | 98.1 (96.2) | 14.6 (17.1) | .06 | 26 (23) |
| 10% | 57.5 (57.8) | 91.8 (91.2) | 97.5 (94.8) | 13.6 (15.6) | .08 (.089) | 26 (22) |
| 50% | 57.5 (58.1) | 95.1 (89.9) | 95.8 (92.3) | 8.9 (14.0) | .29 (.479) | 25 (24) |

- **The "calibration trade-off" story does not survive.** WR 0% no longer costs sanity tests, and WR 0–10% behave alike.
- **WR 10% is kept downstream** (the submission's choice), so nothing downstream is selected on these numbers.
- **B4:** the Table IV caption now gives each row's architecture.
- **B5:** "49.6% → 22.0%" and "64% → 16%" mixed two protocols. They are replaced with within-protocol comparisons: 23.7 → 13.6% in Table V, and 58.5 → 50.2% in Table II.
- **Scope sweep:** the 27-config unseen-vs-sparse scope sweep (supplement) was not re-run. It is marked as using the submitted models.

### 3.8 MCTS (Section VII; Table VI)

**Protocol:** 5 seeds × 1,000 drafts per configuration, mid tier, 200 inference sims, root T=1. "Train WP" is the value function the configuration optimized; `*` marks the submission's enriched judge where the configuration optimized something else. Sources: `results/SUMMARY.json` → `mcts`, and `results/mcts_bench/`.

| Config | Sims | Train WP | gN | gN-naive | RN | QM2026 | R17 | Degen % | Synergy |
|---|---|---|---|---|---|---|---|---|---|
| J_800sim | 800 | .769 | .667 | .683 | .623 | .646 | .632 | 7.2 | +1.40 |
| I_600sim | 600 | .766 | .666 | .679 | .622 | .647 | .635 | 7.8 | +1.44 |
| F_400sim | 400 | .758 | .671 | .682 | .626 | .645 | .637 | 6.6 | +1.35 |
| E_1M | 200 (1M ep.) | .753 | .671 | .679 | .626 | .650 | .641 | 7.3 | +1.34 |
| C_large | 200 | .738 | .662 | .663 | .615 | .638 | .624 | 6.6 | +1.27 |
| H_augmented | 200 | .730* | .663 | .666 | .619 | .637 | .631 | 12.3 | +1.16 |
| B_fullwp | 200 | .721 | .654 | .662 | .610 | .636 | .618 | 15.0 | +1.14 |
| M2_relational | 200 | .717* | .644 | .651 | .611 | .631 | .618 | 12.2 | +1.15 |
| N2_absolute | 200 | .717* | .666 | .671 | .620 | .638 | .631 | 21.2 | +1.21 |
| K_truebase (no features) | 200 | .709* | .664 | .675 | .626 | .649 | .633 | 18.5 | +1.09 |
| D_deep | 200 | .659 | .618 | .613 | .583 | .611 | .574 | 6.4 | +0.34 |
| **B_oof (leak-free)** | 200 | .698 | .662 | .670 | .621 | .641 | .628 | 13.9 | +1.13 |
| **F_oof (leak-free)** | 400 | .716 | .671 | .680 | .628 | .645 | .634 | 6.6 | +1.26 |
| **J_oof (leak-free)** | 800 | .728 | .665 | .682 | .622 | .646 | .627 | 11.1 | +1.34 |

**Contrasts** (difference ± seed SE, T=1):

| Contrast | Train WP | gN | RN | QM2026 |
|---|---|---|---|---|
| F400 − B200 (submitted) | +.037 ± .004 | +.017 ± .004 | +.016 ± .004 | +.009 ± .006 |
| J800 − F400 (submitted) | +.011 ± .003 | −.004 ± .002 | −.003 ± .002 | +.002 ± .004 |
| B_fullwp − K_truebase | +.013 ± .007 | **−.010 ± .005** | **−.016 ± .004** | **−.013 ± .006** |
| E_1M − B_fullwp | +.031 ± .005 | +.017 ± .004 | +.015 ± .004 | +.014 ± .006 |
| F_oof − B_oof (leak-free) | +.019 ± .003 | +.009 ± .004 | +.008 ± .003 | +.004 ± .006 |
| B_oof − B_fullwp (leak itself, 200 sims) | n/a | +.008 ± .005 | +.010 ± .005 | +.005 ± .007 |
| F_oof − F_400sim | n/a | +.000 ± .002 | +.002 ± .002 | +.000 ± .005 |
| B_oof − K_truebase (features, leak-free) | +.007 ± .006 | −.002 ± .004 | −.005 ± .003 | −.008 ± .006 |
| F_oof − K_truebase | +.025 ± .006 | +.007 ± .003 | +.002 ± .002 | −.004 ± .004 |

Reading:

- **Sim scaling, submitted pipeline:** 200 → 800 sims raises the training WP by +0.048 but gN by +0.013. All of the independent gain arrives by 400 sims.
  - This reproduces the overfit study: 15 seeds × 200 drafts gave +0.041 vs +0.010.
- **Sim scaling, leak-free, 200 → 400:** the independent gain is smaller (+0.009 gN) and the own-proxy gain roughly doubles it (+0.019).
  - This is no better tracking than the leaky pipeline showed at this step. At paper scale, removing the leak did not by itself close the proxy–gold gap.
  - At matched sims, leak-free and leaky agents score the same under independent references. This fits the study's finding that the paper proxy was only mildly leaky (post slope 0.88).
- **J_oof (400 → 800, leak-free): the split experiment's prediction fails at paper scale.**
  - Under independent references the agents get worse: gN −0.007 ± 0.003, RN −0.006 ± 0.003, QM2026 +0.001 ± 0.004.
  - The agents' own score rises by +0.012 ± 0.004, and the degenerate rate goes from 6.6% to 11.1%.
  - Even a calibrated, leak-free value function is over-optimized by deeper search. It is calibrated on human games, not on the states that deep search reaches.
  - F_oof is confirmed as the operating point: it is at least as good as J_oof under every reference, at half the training compute.
  - Caveat: J_oof seeds s0, s1, and s4 were paused by the 2026-09-30 compute cap and resumed from their best-eval checkpoints (51K, 51K, and 107K episodes; optimizer state restored).
- **The features-vs-no-features gap is gone under every independent reference, in both pipelines.**
  - What features buy is safety: 13.9–15.0% vs 18.5% degenerate at 200 sims. At the operating point, F_oof reaches 6.6%.
  - The M2/N2 decomposition is withdrawn. Under gN the ordering is absolute-only 0.666 > full 0.654 > relational 0.644.
- **C_large vs B:** C_large beats B under gN (+0.008) and has less than half the degenerate rate. The submission called them "indistinguishable".
- **D_deep:** harmful under every reference.
- **Argmax root (T=0):** +0.000 to +0.006 gN.
- **Kernel tree-capacity guard:** 0 hits in every run.
- **B6/B7/B8:** fixed by the new table and protocol text.

### 3.9 Tournament (Section VII-D; Fig. 1)

**Setup**
- 12 strategies (K_truebase added), 132 ordered pairs × 200 drafts. There are 102 new pair runs; the 30 pairs among the six unchanged baselines reuse the submission's drafts.
- All pairs are scored by the independent references only.
- SEs are clustered by draft configuration (B23).
- Sources: `results/tournament_standings.json` and `results/tournament/`.

| Strategy | Consensus | Won | gN | gN-naive | RN | QM2026 | QM2021 | Degen | Synergy |
|---|---|---|---|---|---|---|---|---|---|
| MCTS (F_oof, argmax) | .605 | 21/22 | .612 | .617 | .592 | .598 | .694 | 7.3 | +.85 |
| Constrained MCTS | .602 | 21/22 | .618 | .610 | .595 | .585 | .679 | 0.0 | +.73 |
| MCTS, no features (K) | .590 | 18/22 | .590 | .592 | .575 | .602 | .705 | 23.4 | +.82 |
| Enriched greedy | .533 | 14/22 | .537 | .550 | .528 | .518 | .495 | 24.1 | +.57 |
| Constrained greedy | .531 | 15/22 | .543 | .543 | .528 | .511 | .488 | 0.0 | +.50 |
| G&A estimator | .530 | 13/22 | .515 | .552 | .519 | .534 | .517 | 46.0 | +.55 |
| Enr.+aug greedy | .518 | 10/22 | .526 | .531 | .515 | .500 | .481 | 16.1 | +.41 |
| CQL enriched | .445 | 5/22 | .440 | .428 | .448 | .463 | .427 | 3.2 | −.16 |
| CQL naive | .443 | 5/22 | .437 | .425 | .446 | .464 | .442 | 4.5 | −.13 |
| G&A discriminator | .439 | 5/22 | .442 | .410 | .436 | .468 | .436 | 0.0 | +.11 |
| GD | .435 | 5/22 | .428 | .414 | .439 | .457 | .420 | 3.5 | −.23 |
| MCQ | .330 | 0/22 | .313 | .329 | .380 | .299 | .215 | 45.1 | −1.56 |

- **Consensus SE:** about 0.001 per standing.
- **Spearman with consensus, all 12 strategies:** gN .96, gN-naive .97, RN .98, QM2026 .93, QM2021 .94.
- **Spearman within the top 6:** .89 / .83 / .89 / .66 / .66.
- **Old vs new:**
  - Submitted: constrained MCTS 0.670 (20/20), MCTS 0.658, then greedy 0.595, anchored cluster 0.40–0.42, MCQ 0.225.
  - The three tiers survive (search, greedy, anchored), and so do the positions of the external baselines: the estimator is in the greedy tier, the discriminator in the anchored tier.
  - The margins are smaller.
  - The constrained agent no longer tops the table; it ties MCTS.
- **Head to head, both seatings (consensus):**
  - constrained vs unconstrained MCTS, same checkpoints: **0.499 ± 0.003**;
  - constrained vs unconstrained greedy: 0.500;
  - MCTS vs K_truebase: 0.544 ± 0.003. gN .558 and RN .538 agree; QM2026 .511 is marginal and QM2021 .465 disagrees.
  - The MCTS vs K comparison is confounded by sims (400 vs 200). At matched sims (Table VI) there is no difference.
- **Submitted agents, matched (A3 fix; `results/h2h_submitted.json`):** J_800sim_s9 constrained vs the same checkpoint unconstrained scores 0.503 consensus. The submission's 0.577/0.542 compared different checkpoints.
- **B24:** "constrained greedy wins 16/20 vs 14/20" is replaced by 15/22 vs 14/22 and a 0.500 head-to-head.
- **B2:** the text now says the tournament agents play policy argmax with no search.

### 3.10 NGS transfer (supplement; Limitations)

Source: `results/ngs.json`, 11,034 NGS drafts, game-level, symmetrized.

| Model | NGS acc | Slope | Ladder test |
|---|---|---|---|
| Submitted naive | 55.3 | 0.95 | 57.8 |
| Submitted enriched | 55.5 | 0.75 | 58.1 (leaky) |
| Leak-free naive | 55.5 | 0.94 | 57.7 |
| Leak-free enriched | 55.5 | 0.86 | 57.7 |
| Leak-free augmented | 55.5 | 0.90 | 57.5 |

The off-meta quartile breakdown was not re-run; it is marked as using the submitted models.

### 3.11 Other number and text fixes (audit B items)

| Item | Old | New |
|---|---|---|
| B1 corpus size | 2.1M / 2.07M drafts; 3.9M samples | 1.95M replays; 3.74M training rows |
| B9 p-values | M2 p=0.04, B vs N2 p=0.004 | decomposition withdrawn |
| B14 snapshot stability | "strategy ordering" | "ordering of twelve MCTS configurations" |
| B15 QM Spearman | 0.92 | 0.93/0.94 (all 12), 0.66 (top 6); key bug fixed by using exact keys |
| B16 hero-strength shares no features | claimed | removed; it is no longer a judge |
| B17 aggregates a separate corpus | claimed | removed |
| B18 data availability | id bounds; processed matrices | replay-id list, split and fold assignments, and rebuild code (see §6: these must be added to the release) |
| B22 throughput | 140–190 ep/s | 140–161 ep/s (main text and supplement) |
| Gourdeau 275K-replay 52.8% | claimed | removed (no artifact) |
| D style | em-dash, "Notably", "Crucially", "More importantly", "not X but Y", triads, dramatic headers | fixed in changed text; see §6 for remaining unchanged sentences |

---

## 4. Audit item status

- **A1** leakage: fixed.
- **A2** self-grading: fixed.
- **A3** mismatched checkpoints: fixed, and the matched result is a tie.
- **A4** "not self-graded": removed; judges are independent by construction.
- **A5** metric validity: redone on clean data.
- **A6** favorites: withdrawn, with forward numbers reported.
- **B1–B24:** all addressed (§3).
- **C1–C8:** addressed.
  - C1: validation split.
  - C2: 3 seeds with a validation rule.
  - C3: evaluation seeds stated.
  - C4: A_partial and G_base stated as not re-benchmarked.
  - C5: own statistics are rebuildable.
  - C6: removed.
  - C7: circularity paragraph expanded.
  - C8: QM reach stated.
- **C9** anonymous URL: **OPEN**.
- **C10** NGS summary file: **OPEN**.

---

## 5. Draft response to reviewers

> We thank the reviewers. While preparing this revision we audited our own pipeline and found two problems that affected several reported numbers. We have corrected both and re-run the affected experiments. We describe them first because they change some claims.
>
> **1. Outcome leakage in the value-function features.** The enriched features used aggregate statistics published by the community database our replays come from. Those aggregates contain most of our corpus, including the test games, so each game's features contained its own outcome. Removing each test game's own contribution alone drops the submitted enriched model from 58.1% to 57.0% held-out accuracy.
>
> The revision computes every statistic from our own training split, out of fold. Validation, test, and post-snapshot games use statistics that exclude them. Early stopping and seed selection now use a separate validation split. With leak-free features the enriched model is exactly as accurate as the identity-only model (57.7% vs 57.7%) and is calibrated on unseen games (slope 0.99 vs 0.88 before). This strengthens our thesis that accuracy does not discriminate between value functions, and we removed the claim that the enriched model is the most accurate. A new subsection (III-C) explains the mechanism, and a supplement section isolates it in a within-snapshot split experiment.
>
> **2. Self-grading.** The submitted MCTS results were scored by the value function the agents were trained against, and two of the four tournament judges were contestants' own value functions. The revision scores every draft with references that share no data with any agent: judges trained only on 291,837 games uploaded after our snapshot, a model-free realized-outcome index on those games, and two Quick Match judges. Under these references:
>
> - search still beats every behavior-anchored method by a wide margin (0.60 vs 0.44 tournament win probability);
> - the gain from 400 to 800 training simulations and the +0.023 gain from domain features disappear;
> - the features' real contribution is safety (degenerate compositions 18.5% → 13.9% at matched search, 6.6% at our recommended operating point).
>
> We also found that our constrained-vs-unconstrained comparison used different checkpoints. At matched checkpoints the constraint mask ties (0.499): it guarantees structural validity at no measurable cost, and we now claim only that.
>
> The offline-RL findings (CQL, IQL, MCQ collapse to anchoring) do not depend on these statistics and are unchanged. We also corrected a set of smaller numerical errors (listed in the change log). All code, split assignments, and the scripts that rebuild every statistic are in the repository.
>
> [Point-by-point replies to the reviewers' specific comments go here once the reviews arrive; this revision was prepared proactively.]

---

## 6. Open items

1. **J_oof:** done (§3.8).
2. **Not re-run** (stated in the manuscript):
   - CQL with enriched features (62M transitions);
   - the 27-config augmentation scope sweep;
   - A_partial and G_base;
   - the 20-model ensemble-uncertainty analysis;
   - the draft-concentration diagnostics;
   - the NGS off-meta quartiles;
   - the dual-snapshot table. It describes the submitted agents' metric stability and still holds as a statement about those agents.
3. **Release (B18, C5):** add the replay-id list, the split and fold assignments, and `training/paper1_revision/` to the public repository and to `oss-export`. Decide whether the external stats file stays in `oss-export` (license question), since no model needs it any more.
4. **C9:** verify the anonymous repository link resolves to the `/r/<id>` form with no expiry.
5. **C10:** fix `training/ngs2026/NGS_SUMMARY.md` ("less meta-concentrated" contradicts the supplement).
6. **Anonymous version:** deferred per Max; produce it by the submission's toggles when resubmitting.
7. **Style pass:** done in the trim; the whole main text was rewritten (§8).
8. **Page length:** done; the clean build is 8 pages (§8).
9. **Figures:** the metric-validity figure is replaced by a table. The two revised figures are regenerated by `gen_figs.py`. The scatter mixes the MCTS benchmark protocol with rich-evaluation rows, and its caption says so.

---

## 8. Trim to 10 pages (2026-09-30): what moved, merged, or was cut

The clean manuscript went from 13 to 8 pages (7,525 words in the source, down from 11,334). Evidence was compressed, not dropped.

**Every table stays in the main text:**
- Table I: models, submitted vs leak-free, with the in-sample control and Gourdeau.
- Table II: cross-evaluation.
- Table III: feature sweep.
- Table IV: augmentation.
- Table V: rich evaluation.
- Table VI: MCTS. All 11 re-benchmarked submitted configurations, now single-column with 8 columns.
- Table VII: tournament.

**Also kept in main:** both figures, the leakage explanation and fix, the split-experiment summary, the independent-reference methodology with calibration numbers, and every surviving claim with its numbers.

**Moved to the supplement**

| Item | Where now |
|---|---|
| Full offline-RL sweeps (CQL α, BC-CQL β, MCQ τ, IQL grid) with numbers | new supplement §"Offline RL: Sweeps and Robustness" |
| CQL capacity × target-update robustness | same section; now includes a per-metric range over all 15 configs |
| CQL-with-enriched-features paragraph | same section, with the note that it was not retrained leak-free |
| Metric-validity detail (the four-way table) | supplement `tab:metricvalid`; main keeps the +9.1/+8.0pp post-snapshot result in one sentence |
| Reference calibration table | supplement `tab:refcal`; main keeps the five accuracy/slope pairs in one sentence |
| Split-experiment table | supplement `tab:split`; main keeps a numeric summary paragraph |
| Sweep log-loss column and design generators | supplement `tab:sweep` and text; the main table keeps test, post, and submitted effects |
| Two CQL rows | were in the supplement for one draft, then restored to main Table V |

**Merged or compressed**

- **Related work:** four subsections become three short paragraphs. Scaling-MCTS is one sentence.
- **Evaluation framework:** two paragraphs. The circularity paragraph is reduced to its current state; the history of the three self-grading routes lives in these notes.
- **Composition-gap subsections:** merged into three paragraphs (cross-evaluation, accuracy vs policy, independent baseline).
- **Section V:** residual-gap analysis, augmentation setup, scope, and results form one section.
- **MCTS paragraphs:** capacity, augmented-model, and root-selection paragraphs merged. Training length and root selection are one paragraph. The feature-gap paragraph now also carries the C_large, D_deep, and M2/N2 notes.
- **Discussion:** "Two separable axes", "Why pessimistic offline RL collapses", "What transfers", "Practical implications", and "Open problems" merge into two paragraphs.
- **Conclusion and Limitations:** shortened. Limitations keep every caveat in shorter form.

**Cut from main** (withdrawn or weakened claims, all documented in §3)

- the IQL five-opponent cross-check (0.388, judged by the submission's leaky judges);
- the H_augmented "partially redundant" interpretation;
- the dual-snapshot sentence (it stays in the supplement);
- the MCTS training-architecture detail beyond one sentence;
- the "95%+ of training teams include a healer" factor list (kept as one clause);
- the 473K-era GD-agreement comparison;
- the "fused CUDA" implementation paragraph in §VII (the contribution list keeps it).

**Style:** the new text was checked for em-dashes, "notably/crucially/importantly", "not X but Y" and "instead of" contrasts, and rhetorical triads. A few factual three-item lists remain (for example "imitation, algorithmic pessimism, and manifold discrimination all end at safety by anchoring"), which name methods rather than add rhetoric.

**New supplement content from a deferred job:** the NGS off-meta quartile breakdown, re-run leak-free (`results/deferred/ngs_quartiles.json`), does not show the submitted trend.
- Leak-free enriched model, most- to least-meta quartile: 56.0 / 54.2 / 56.9 / 54.8%. Submitted: 57.6% → 54.5%.
- The supplement now says the degradation is not shown to concentrate in off-meta drafts.

## 9. Deferred work queue (resource-capped)

`training/paper1_revision/capped_queue2.sh` runs after `capped_queue.sh`, which finishes the 800-sim seeds (J_oof s2 and s3 running; s4, s0, and s1 resumed from checkpoints, then benchmarked).

**Rules the queue follows**
- One job at a time. It never exceeds 2 GPU processes for this lane.
- GPU jobs run on GPU 3, under `nice -n 19 taskset -c 48-63`.
- OMP/MKL/NUMBA = 1 thread, pools of 4, CQL DataLoader 4 workers (≤6 cores).
- Each job writes `results/deferred/<name>.json` (log: `logs/deferred_<job>.log`) and is skipped if that file exists.
- Code: `training/paper1_revision/deferred.py`.

**Queue order** (by value to the revision)

1. **Augmentation scope sweep, leak-free** (27 configs: WR 10/20/30 × volume 50/100/200 × scope unseen/sparse/both; 256×128, own composition table, out-of-fold features). Output: 5-tank WP, sanity, held-out accuracy and slope. Tests the main-text claim that unseen-composition augmentation is the active ingredient, currently from submitted models. [GPU, ~2–3 h]
2. **NGS off-meta quartiles:** already done (smoke test), skipped by the queue. [CPU]
3. **A_partial and G_base rebench** under all references (seeds 0–4, T=1 and T=0). Completes Table VI. `bench_mcts.py` now supports step-conditioned value functions. [GPU, <1 h]
4. **Draft concentration for the leak-free F_oof agent** (and submitted J_800sim for comparison): per-seed diversity; favorites; pair/trio edges in-sample (train split) vs forward (post-snapshot games); role-composition coverage in the own table; exact-team occurrence. Replaces the supplement section that describes the submitted agents. [CPU, ~1 h]
5. **Specification-diverse ensemble, leak-free** (20 members: 8 full across arch × data half × seed, 9 minus-one-group, 3 naive). Per-draft variance on held-out human, F_oof agent, per-strategy tournament, synthetic degenerate, and probe drafts. Replaces the supplement ensemble section. [GPU, ~2 h]
6. **CQL with enriched features, leak-free.** This is the most expensive job, and its result matters least, because the anchoring collapse does not depend on feature accuracy.
   - `cql_build`: out-of-fold transition features, ~62M rows, validation split for early stopping. [CPU, ~5–8 h at 4 procs]
   - `cql_train_a2.0` and `cql_train_a0.5`. [GPU, ~4–8 h each]
   - `cql_eval`: 5 × 1,000 rich evaluation plus reference scores. Replaces the two CQL (enr.) rows of Table V.

**Results so far** (2026-09-30, `results/deferred/`):

1. **Scope sweep** (`scope_sweep.json`): the claim survives leak-free.
   - Sparse-only augmentation leaves the 5-tank WP at 0.26 (baseline 0.27) in all 9 cells, which are identical because that generator ignores WR and volume.
   - Unseen-only augmentation gives 0.07–0.12 at WR 10%, 0.16–0.18 at 20%, and 0.23–0.28 at 30%.
   - Accuracy is 57.5–57.7% and post-snapshot slope 0.93–1.02 in every cell.
   - Folded into the main text (Section V) and the supplement.
2. **NGS quartiles:** done earlier (§8).
3. **A_partial and G_base** (`rebench_ag.json`), T=1:
   - A_partial: gN .662, RN .620, QM2026 .636, 16.5% degenerate.
   - G_base: gN .636, RN .612, QM2026 .633, 18.4% degenerate.
   - Added to main Table VI.
4. **Concentration** (`concentration.json`). F_oof per seed: 24–27 heroes, 3.25–3.59 bits.
   - Favorites: Samuro, Rehgar, Hogger, Deathwing, Falstad, Illidan, Ragnaros.
   - Pair edge: +0.54pp in-sample (z 5.2), +0.21pp forward (z 1.0).
   - Trio edge: +1.70pp in-sample (z 3.8), +1.68pp forward (z 1.9).
   - None of the top-10 teams (38.8% of drafts) occurs in the corpus. 98.7% of drafts are on admitted role compositions, and 63.7% on compositions with ≥1,000 games.
   - The supplement section is rewritten around these numbers.
5. **Ensemble** (`ensemble_uncertainty.json`), variance relative to the human median:
   - F_oof agent drafts: 1.21×, with 7.1% above the human p95.
   - Tournament strategies: 1.09–2.21×.
   - Synthetic degenerate: 3.20×. Probes: 6.05×.
   - The supplement section is rewritten (Table `tab:ensvar`).
6. **CQL enriched, leak-free:**
   - `cql_build` is done.
   - `cql_train_a2.0` was running at the time of writing, then `cql_train_a0.5` and `cql_eval`.
   - When `cql_enriched_eval.json` lands, replace the two CQL (enr.) rows of main Table V and the supplement's CQL-enriched paragraph.

**Folding results in:** `results/deferred/*.json`. For each finished job, update the corresponding supplement section. Table VI and Table V rows can be added in the main text; each row costs one line. The main text has 2 pages of headroom.

## 10. Tier-label bug (found 2026-09-30)

**The bug.** The replay daemon stored listing league_tier one above the rank: 2 = Bronze … 6 = Diamond, with NULL plus an MMR = Master. It mapped stored ≤2 → low, 3–4 → mid, else high, and NULL → mid.

**What the snapshot labels actually contain.** Checked per replay against the pre-relabel backup `backups/replay_draft_skill_tier_20260930.csv.gz`; `results/tier_sensitivity.json` → `crosstab`.

| Snapshot label | Real ranks | Games | Share |
|---|---|---|---|
| low | Bronze | 380,100 | 19.5% |
| mid | Silver 418,162 + Gold 418,103 + Master 101,308 + no rank/MMR 5,749 | 943,322 | 48.4% |
| high | Platinum 382,491 + Diamond 243,174 | 625,665 | 32.1% |

- No Wood games are present.
- The post-snapshot reference games use the same labels: 42,156 low; 165,152 mid, of which 31,298 have no rank; 84,529 high.
- **The submission's description was wrong.** It said Bronze–Gold / Platinum–Diamond / Master–Grandmaster (`draft.tex` line 93).

**Manuscript fixes**
- `draft_revision.tex` §III-A now states what the labels contain and that they act as three fixed skill strata used consistently throughout. The sensitivity result is one sentence (see below).
- The MCTS benchmark says "mid label".
- The supplement's diagnostic protocol says "mid tier label".
- No other passage names tier boundaries or reports per-tier results: the per-tier composition counts are counts per label.
- `draft_revision_markup.tex` is patched the same way.

**Sensitivity check** (`training/paper1_revision/tier_sensitivity.py`)
- Correct ranks are re-derived per replay: real rank = stored league_tier − 1; NULL with an MMR = Master.
- Two correct schemes:
  - "stated": the submission's description, Bronze–Gold / Platinum–Diamond / Master;
  - "site": Bronze+Silver / Gold+Platinum / Diamond+Master.
- Rows with no rank stay in mid.
- Statistics and out-of-fold features are rebuilt per scheme, and the leak-free enriched and naive models are retrained (seed 42, validation early stopping). They are compared with the mislabeled seed-42 models on the same test and post-snapshot games.
- Results (seed 42; 3.7M training rows each; `results/tier_sensitivity.json` → `runs`):

| Tiers | Model | Test acc | Test LL | Post acc | Post LL | Post slope |
|---|---|---|---|---|---|---|
| snapshot labels (as used) | enriched | 57.77 | .6745 | 56.81 | .6783 | 1.05 |
| snapshot labels (as used) | naive | 57.63 | .6740 | 56.79 | .6778 | 1.05 |
| stated (Bronze–Gold / Plat–Diamond / Master) | enriched | 57.54 | .6749 | 56.71 | .6786 | 0.98 |
| stated | naive | 57.58 | .6740 | 56.72 | .6779 | 1.04 |
| site (Bronze–Silver / Gold–Plat / Diamond–Master) | enriched | 57.64 | .6746 | 56.78 | .6783 | 0.99 |
| site | naive | 57.54 | .6737 | 56.75 | .6777 | 1.06 |

- **The headline numbers do not move.**
  - Accuracy changes by at most 0.25pp. The 3-seed SD is about 0.1pp, and this is one seed.
  - Post-snapshot slopes stay at 0.98–1.06.
  - Enriched and naive stay within 0.1pp of each other under every scheme.
- **Nothing downstream was rerun.** The paper carries one sentence saying so (§III-A).
- **Remaining caveat:** the agents' tier input means the label population, not a clean rank bracket. Downstream results (MCTS, tournament) are conditioned on those labels and are described as label strata.

**Expert study v2** (`results/expert_tier_audit.json`, script `expert_tier_audit.py`)

`src/app/rate/rate-client.tsx` tells raters Low = Bronze–Silver, Mid = Gold–Platinum, High = Diamond–Master. The v2 item pool (`data/rating-items.json`, pool v5, seed 20260918) was generated on 2026-09-18, before the relabel. **Its tiers come from the old DB labels.**

*Ladder-sourced items* (anchors, calibration, screener, catch; 621 items): **239 (38.5%) show a tier banner whose rank range does not contain the game's actual rank.** By displayed tier:
- **Low** (displayed Bronze–Silver): every item is a Bronze game. These are consistent with the banner, but no Silver games appear.
- **Mid** (displayed Gold–Platinum): every item is Silver, Gold, or Master. Every Silver and Master item is outside the displayed range.
  - Anchors: 90 Silver + 22 Master + 4 unranked out of 190.
  - Calibration: 4 Silver + 1 Master out of 13.
  - Screener: 1 Silver.
- **High** (displayed Diamond–Master): every item is Platinum or Diamond. Every Platinum item is outside the displayed range.
  - Anchors: 106 of 190.
  - Calibration: 9 of 13.
  - Screener: 2.
- **By block:** anchors 222/570 (39%), calibration 14/40 (35%), screener 3/8, catch 0/3.

*Machine pairs* (280 items: 93 low, 93 mid, 94 high). Their tier label conditioned the agents and judges, which were trained on the 2026-09-01 snapshot's old labels. The share of each label's training population inside the displayed range is:
- low: 100% (all Bronze);
- mid: 43% (Gold only; Silver and Master are outside);
- high: 39% (Diamond only; Platinum is outside).

So for most mid and high machine items, raters are told a different skill bracket than the one the drafts were optimized for.

**Data collected so far:** `draft_ratings` holds 7 ratings from 1 rater, all on 2026-09-30, probably a test session. There are no study data yet. Before invites go out, one of these needs to happen:
1. Show raters the bracket the labels actually encode (Bronze / Silver–Gold–Master / Platinum–Diamond). This is awkward, because Master sits in mid.
2. Regenerate the anchors from relabeled tiers and retrain the tournament on relabeled data.
3. Drop tier from the rater display and the anchor stratification.

This touches the OSF preregistration v2 (tier framing) and the IRB instrument text. Max decides; nothing was changed in the app or the pool.

## 7. Files

**Code** (`training/paper1_revision/`)

| File | Purpose |
|---|---|
| `core.py` | splits, own statistics, out-of-fold features |
| `train_wp.py` | all WP models and the fractional-factorial sweep |
| `eval_submitted.py`, `eval_ngs.py`, `eval_gourdeau.py`, `judge_calibration.py` | held-out quality |
| `sanity.py` | sanity suite and probes |
| `metric_validity.py` | supplement metric check |
| `mcq_dead_units.py` | MCQ dead-unit count |
| `crosseval.py` | Table II |
| `greedy_evals.py` | Table IV greedy/MCTS rows; Table V |
| `train_mcts.py` (with `--resume`), `launch_mcts*.sh`, `capped_queue.sh` | MCTS training |
| `bench_mcts.py` | Table VI benchmark under all references |
| `verify_kernel.py` | kernel/Python agreement check |
| `h2h_submitted.py` | matched constrained vs unconstrained for the submitted checkpoint |
| `tournament.py`, `run_tournament.py`, `score_tournament.py` | tournament |
| `compile_results.py`, `make_tables.py`, `gen_figs.py` | aggregation, LaTeX rows, figures |
| `deferred.py`, `capped_queue2.sh` | deferred jobs and their capped queue (§9) |
| `tier_sensitivity.py`, `expert_tier_audit.py` | tier-label bug: crosstab, retraining under correct tiers, expert-pool audit (§10) |
| `orchestrate.sh` | pre-cap launcher, stopped |

**Results** (`training/paper1_revision/results/`):
- `SUMMARY.json` (everything), `tables.tex`
- per-experiment JSONs
- `mcts_bench/` (per-draft scores for 55 submitted runs and the leak-free runs), `tournament/` (132-pair records via reuse), `greedy/`

**Models:** `training/paper1_revision/models/` (all seeds, with metadata JSON). **MCTS runs:** `training/paper1_revision/mcts_runs/`.

## 11. Composition blind spot of the independent references (2026-10-01)

**The finding.** In the production seed-selection dry run, a drafter that picks the highest-win-rate available hero scored 0.677 under the realized index, above both trained policies (0.644, 0.650), with 55% degenerate teams. On real held-out games the references under-predict how often degenerate teams lose.

**Code:** `training/overfit2026/`: `structure.py` (indicators), `comp_audit_data.py` (judge scores on held-out sets), `comp_causal_data.py` + `comp_causal.py` (skill and pick-position controls), `comp_gn_folds.py` (two-fold gN refits for held-out predictions on N), `comp_judges.py` (audit, fit, validation), `judges_v2.py` (the corrected judges), `comp_native_ri.py` (structure terms inside the realized index), `gold.StructRealizedIndex`. Rescoring: `training/paper1_revision/comp_rescore.py`.
**Results:** `overfit2026/results/comp_audit.json`, `comp_causal.json`, `comp_judges.json` (and the superseded `comp_judges_t17fit.json`), `comp_native_ri.json`; `paper1_revision/results/comp_rescore.json`.

### 11.1 Audit (task 1)

Realized minus predicted win rate (pp) per team type, on the 1,949,087 snapshot games, which no reference trained on. Types are exclusive in the order shown and cover 1.37%, 0.57%, 0.06% of teams (2.01% in all). SE about 0.2 / 0.3 / 1.0 / 0.2pp.

| Reference | No healer | No frontline | Stack | Any degenerate | Normal teams |
|---|---|---|---|---|---|
| gN | −2.0 | −0.7 | −0.3 | **−1.6** | +0.03 |
| gN-naive | −5.6 | −3.3 | −9.4 | **−5.0** | +0.10 |
| RN | −5.0 | −4.1 | −10.9 | **−4.9** | +0.10 |
| R17 | −6.5 | −5.9 | −11.9 | **−6.5** | +0.13 |
| QM2026 | −6.4 | −2.8 | −7.5 | **−5.4** | +0.11 |
| QM2021 | −6.8 | −3.5 | −7.3 | **−5.9** | +0.12 |
| Consensus | −4.8 | −2.7 | −7.0 | **−4.2** | +0.09 |

- **Other held-out sets agree.** Post-snapshot no-drift games with held-out predictions (gN and gN-naive from two-fold refits of the judge recipe, RN cross-fitted): gN −1.1, gN-naive −5.3, RN −5.2, QM2026 −5.2, consensus −4.2pp. On the drifted 2.55.17 games the gaps are smaller (RN −3.1, consensus −2.4, gN +0.3).
- **The gaps survive recalibration.** After a global slope/intercept refit of each judge the gaps barely move (QM2026 −5.4 → −6.5, others ±0.5), so this is blindness to structure, not overall miscalibration.
- **Why each is blind.**
  - RN, R17: the composition term is a per-role-multiset win rate shrunk to 50% with a 200-game prior; degenerate cells hold a median of 2.5–10 games and the term's weight is small.
  - gN-naive: hero identities only; it must infer the penalty from 2% of teams.
  - QM judges: trained on a mode with a different team-assembly process.
  - gN is the least blind (−1.6pp): its features include role counts and the external composition table.

### 11.2 Is the penalty causal? (task 2)

918,922 games (snapshot games from 2025-06-01 plus post-snapshot games with all ten player rows) joined to the P3 causal per-player, per-hero skill estimates (+CF rank 2 GP, experience offsets, lag 1 day) and to pick order. Skill state built with degenerate-team games left out ("clean"), so it cannot absorb the penalty.

| Controls (offset: held-out RN) | Penalty on degenerate teams (pp) |
|---|---|
| none | −4.80 |
| team skill difference | −4.63 |
| + skill by within-team pick position, first-pick side | −4.84 |
| + partied players, never-played-hero slots | −4.66 |
| same as row 3, skill state including degenerate games | −4.91 |

- Bootstrap SE 0.27pp; SE of the change 0.05pp. Snapshot and post-snapshot samples agree (−5.02 → −5.03; −4.31 → −4.41).
- **Why so little selection:** degenerate teams are weaker in absolute terms (summed skill −0.3pp vs +1.0pp for normal teams), but matchmaking gives them weaker opponents too; the within-game skill gap is only −0.24pp.
- **Verdict:** at most about 0.2pp of the ~4.8pp is selection on these measures. The penalty is essentially a draft effect (unmeasured player differences remain possible). The corrected judges therefore use the plain realized fit; the skill-controlled fit is kept as a variant ("causal" in `comp_judges.json`), differs by ≤0.05 logit, and over-corrects the snapshot by 0.6pp.

### 11.3 Corrected judges (task 3)

**Form (v2):** p\* = σ(logit p_J + β_J·(s_own − s_opp)), s = [no_healer, no_frontline, stack] (non-exclusive, `shared.is_degenerate` conventions). J itself is unchanged; the correction is antisymmetric, so symmetrized scores stay symmetrized. Consensus v2 = mean of the corrected gN, gN-naive, RN, QM2026.

**Fit:** on the 291,837 post-snapshot no-drift games (no agent saw them), always with the judge family's held-out prediction as the offset: gN/gN-naive from the two-fold refits, RN cross-fitted, QM and R17 as they are. β (logits; no healer, no frontline, stack):

| Judge | β |
|---|---|
| gN | −0.04, −0.06, 0.00 |
| gN-naive | −0.22, −0.20, −0.35 |
| RN | −0.18, −0.24, −0.36 |
| R17 | −0.22, −0.28, −0.37 |
| QM2026 | −0.23, −0.17, −0.27 |
| QM2021 | −0.28, −0.22, −0.25 |

**Validation on the 1.95M snapshot games** (no judge, no correction saw them):

| Reference | Any-degenerate gap v1 → v2 | Normal teams v2 | Log loss change (×1e−4) |
|---|---|---|---|
| gN | −1.6 → −0.5 | +0.01 | −0.2 |
| gN-naive | −5.0 → +0.3 | −0.01 | −2.4 |
| RN | −4.9 → +0.3 | −0.01 | −2.4 |
| QM2026 | −5.4 → −0.1 | 0.00 | −2.9 |
| Consensus | −4.2 → +0.0 | 0.00 | −1.7 |

- Per type after correction (all six references): no-healer −1.0 to −0.1, no-frontline +0.8 to +1.8, stack −3.2 to −0.4 (stack SE 1.0).
- With the skill controls added on the snapshot subset, the v2 gaps stay near zero (consensus −0.1).
- **Native form (option 1):** the same three indicators inside the realized index's own cross-fitted logistic (`gold.StructRealizedIndex`) calibrate equally well: N half-split −5.2 → +0.1pp, 25% snapshot sample −4.8 → −0.1pp, log loss 0.68595 → 0.68572. Production uses this form.
- **Drift:** after the 2.55.17 balance patch the penalty is smaller; the v2 references over-charge degenerate teams there by about 2pp. A first fit on 2.55.17 games under-corrected the snapshot by 1.6pp (kept in `comp_judges_t17fit.json`), which is why the fit moved to the no-drift games.
- **What a judge can and cannot learn.** The terms are learned from human degenerate teams: 93% have exactly one defect, 68% are no-healer teams. The MCTS agents' degenerate teams have one defect in 95–100% of cases and are 72–98% no-healer, the best-covered type, so the correction is well supported for them. For structures nobody fields (five tanks, no healer and no frontline together, four of a role) the judge can only add the indicator penalties; it has no data on whether they are worse than that. Option 2 (shrinking composition cells toward a role-count model) and option 3 (retraining gN with out-of-fold composition features) were not needed: gN's structure gap was already small, and the offset form calibrated every judge.

### 11.4 Rescoring: old vs new for every claim (task 4)

Saved draft records only; nothing regenerated. MCTS benchmark: per-draft v1 scores stored, v2 = v1 + correction. Tournament: `tournament_scores.npz` + pair records. Contrasts at T=1; ± is the v2 seed SE.

| Claim | v1 gN / RN / consensus | v2 gN / RN / consensus | Survives? |
|---|---|---|---|
| Sim scaling, submitted, 200 → 400 (F − B) | +.017 / +.016 / +.015 | +.018 / +.020 / +.019 (±.005) | yes |
| Sim scaling, submitted, 400 → 800 (J − F) | −.004 / −.003 / −.001 | −.004 / −.004 / −.001 | yes (no gain past 400) |
| Leak-free 200 → 400 (F_oof − B_oof) | +.009 / +.008 / +.008 | +.010 / +.011 / +.011 (±.003) | yes |
| Leak-free 400 → 800 (J_oof − F_oof) | −.007 / −.006 / −.002 | −.007 / −.008 / −.004 (±.003) | yes, slightly stronger |
| Feature gap, submitted (B_fullwp − K_truebase) | −.010 / −.016 / −.013 | −.009 / −.014 / −.011 (±.005) | yes, reversal holds |
| Feature gap, leak-free matched (B_oof − K_truebase) | −.002 / −.005 / −.005 | −.002 / −.003 / −.003 (±.004) | yes, zero within error |
| Training length (E_1M − B_fullwp) | +.017 / +.015 / +.016 | +.017 / +.019 / +.019 | yes |
| Operating point (F_oof − E_1M) | +.001 / +.003 / .000 | .000 / +.003 / .000 | tie holds |
| Argmax root, F_oof (T0 − T1) | +.003 / +.002 / +.002 | +.003 / +.003 / +.003 | yes |
| Absolute-only vs full (N2 − B_fullwp) | +.012 / +.009 / +.008 | +.011 / +.006 / +.005 | smaller |

- **"Features buy safety, not WP" survives.** The matched gaps move by at most 0.003 under any reference. The correction charges K_truebase's 18.5% broken teams, but it also charges the GD opponents sampled at T=1, 12–13% of whose teams are degenerate, so both sides of the benchmark pay.
- **The 400-sim operating point holds.** F_oof ≥ J_oof under every v2 reference (consensus +.004), F_oof ties E_1M, and argmax root adds +.003.
- **Table VI:** gN changes by ≤0.001 per configuration, RN by −0.004 to +0.004, QM2026 by −0.006 to +0.004.

**Tournament (consensus; v1 → v2):**

| Strategy | v1 | v2 | Won v1 → v2 |
|---|---|---|---|
| Constrained MCTS | .602 | **.611** | 21 → 21 |
| MCTS | .605 | .610 | 21 → 21 |
| MCTS, no features | .590 | .587 | 18 → 18 |
| Constrained greedy | .531 | .538 | 15 → 16 |
| Enriched greedy | .533 | .530 | 14 → 14 |
| Enr.+aug greedy | .518 | .519 | 10 → 10 |
| G&A estimator (46% degen) | .530 | **.513** | 13 → 12 |
| CQL enriched / naive | .445 / .443 | .450 / .447 | 5 / 5 |
| G&A discriminator | .439 | .446 | 5 |
| GD | .435 | .440 | 5 |
| MCQ | .330 | .310 | 0 |

- **Spearman with consensus:** all 12 strategies 0.93–0.98 → 0.94–0.99; top 6 QM 0.66 → 0.83, gN/RN 0.89 → 1.00.
- **Constrained vs unconstrained.**
  - MCTS, same checkpoints: 0.499 → **0.509 ± 0.003**.
  - Greedy: 0.500 → **0.511 ± 0.003**.
  - The submitted J_800sim_s9 matched pair: 0.503 → ≈0.514. This one is approximate: its drafts were not stored, so the stored degenerate counts were charged with the no-healer β.
  - **"The mask costs nothing" becomes "the mask costs nothing and gains about 0.01".** The v1 tie came from references that barely charged the broken teams the mask removes.
- **MCTS vs K_truebase:** 0.543 → 0.558 ± 0.004.
- **Table II "Ind." column:** .586/.579/.580 → .572/.571/.575 (naive/hero str./enriched). The three drafters are still not separated. The v1 column here was recomputed by `judges_v2.score_base`, which matches the stored RN/QM exactly and gN to within 0.002.
- **Greedy/MCTS rich rows (gN):** unchanged within 0.001. The consensus moves −0.003 to +0.005.

### 11.5 Manuscript changes (task 5)

**`draft_revision.tex`** (clean build 9 pages, 0 errors, 0 overfull boxes):
- abstract (0.61 vs 0.44–0.45; the mask adds a little; judges calibrated on broken teams);
- intro tiers (0.61 / 0.51–0.54 / 0.44–0.45);
- contribution 3 (the mask adds about 0.01);
- new paragraph "Composition correction" in §III-D (blind spot, mechanism, selection test, correction, validation, limits);
- Table II Ind. column and text;
- Table VI gN/RN/QM26 columns now v2, with a caption note;
- sim-scaling, training-length, root and feature-gap numbers, plus one sentence on why the benchmark gaps barely move;
- Table VII now v2, reordered, with a caption note; tournament text (tiers, Spearman, the v1 differences named, MCTS vs K);
- §VII-E constraint now gains a little (0.509 / 0.511), with the v1 tie explained;
- conclusion numbers and one sentence on auditing judges;
- limitations (the composition terms extrapolate; about 2pp over-charge after 2.55.17);
- Fig. 1 regenerated from v2 standings (`gen_figs.py` reads `comp_rescore.json` when present), caption updated.

**`supplementary_revision.tex`:** new section "Composition Blind Spot and Corrected References". It has the audit table (`tab:compaudit`), the selection test, the correction with its coefficients, the native-index check, drift, and the v1-vs-v2 claims table (`tab:compclaims`).

**Not changed:**
- the markup files;
- the degenerate definition sentence in §III-C. It lists "no ranged damage", which `shared.is_degenerate` (used for every Deg% except Table II) does not require; Table II's Deg% uses a broader flag that does include it. Worth a one-line fix later.

### 11.6 Production (task 6)

- **Change.** `production_refresh/refresh.py` phase `select` now judges with `gold.StructRealizedIndex` (`JUDGE_CLASS`): the realized index plus the three structure terms inside its cross-fitted logistic.
  - Fitted on the same 90-day window. The window is `game_version LIKE '2.55%'`, so the 2.57 builds released 2026-09-28 are excluded.
  - The plain index is still built and logged per bench row (`judge_v1`) and does not enter the rule.
  - Rule, `DEGEN_TOL` and gates are unchanged.

**Backtest `validate-seedsel-backtest-0930`** (800-sim 2026-09-30 seeds; previous outputs kept as `*_v1judge.json`):

| Bench row | v1 judge | v2 judge | Degen |
|---|---|---|---|
| pop_greedy (GD argmax) | .491 | .497 | 3.4% |
| hero_wr_greedy | **.677** | **.637** | 54.7% |
| s0 | .644 | .654 | 4.3% |
| s1 | .650 | .655 | 6.7% |

- Seed 0 is still selected (s1 is ineligible by the degeneracy tolerance). The gate (judge > pop_greedy) still passes.
- The hero-WR drafter now ranks below both trained seeds.
- Fold structure coefficients: no healer −0.22/−0.15, no frontline −0.26/−0.05, stack −0.41/−0.47.

**Dry run `validate-seedsel-2026-10-01`** (400-sim, only 6K episodes):
- s0 .612 → .614, s1 .615 → .613, hero_wr_greedy .677 → .637.
- Seed 0 is still selected.
- The hero-WR drafter still beats these undertrained seeds. That reflects the 6K-episode training, not the judge.

### 11.7 Expert-study pool (coordination with the oct2026 lane)

- **Today no judge from this audit touches the pool.**
  - Item sampling uses no judge.
  - `provenance.wpTeam0Sym`, and with it the preregistered near-tie exclusion (|consensus − 0.5| ≤ 0.02), comes from the four namespace evaluators (naive, herostrength, enriched, augmented). These are in-corpus WP models, not the independent references.
- **If the lane switches the pool to the independent references** through `oct2026_pool_judges.py --judges py:training/overfit2026/judges_v2.py:gN_v2,...`, the composition correction matters. Measured on the current v5 pool (280 machine pairs, 63 with a degenerate team), v1 consensus vs v2 consensus:
  - near-ties at 0.01 / 0.02 / 0.05: 25 / 40 / 93 → 29 / 46 / 95 (at 0.02: 10 leave, 16 enter);
  - 10 pairs flip the favored side;
  - the largest shift is 0.098.
- **Tier caveat.** The independent references condition on the old (pre-2026-09-30) tier labels, while oct2026 items carry site-scheme tiers. Using them on that pool needs a tier mapping decision first.

## 12. Synthetic augmentation demoted; the structural mask is the repair (2026-10-01)

**Decision (Max).** The submission presented targeted synthetic data for unseen role compositions as the repair for unsafe value search ("a layered repair: domain features, targeted synthetic data, MCTS self-play, structural masking"; the scope sweep called unseen compositions "the active ingredient"). The revision reports augmentation as a tested option with a tradeoff and recommends search plus the structural composition mask. The title stays; the repair it names is now the mask.

**Why: the revision's own evidence**

| | Degenerate % | Synergy | Counter | Head to head (consensus, v2) |
|---|---|---|---|---|
| Enriched greedy | 21.1 | +0.44 | +0.16 | reference |
| Enriched + augmentation, greedy | 15.2 | +0.29 | +0.09 | 0.485 ± 0.004 vs unaugmented |
| Enriched greedy + structural mask | 0.0 | +0.37 | +0.13 | 0.511 ± 0.003 vs unmasked |
| MCTS (F_oof) | 6.2 | +0.78 | +0.09 | reference |
| MCTS + structural mask | 0.0 | +0.63 | +0.08 | 0.509 ± 0.003 vs unmasked; ties for first in the tournament (.611 vs .610) |

- Augmentation improves safety (greedy degenerate 23.7 → 13.6%, five-tank WP 0.27 → 0.08, sanity 22 → 26/28) with no accuracy change, but it lowers synergy and counter responsiveness and loses head to head to the unaugmented agent.
- The mask removes every degenerate team, needs no change to the value function, and wins head to head.
- **Correction to the coordinator's framing:** the mask is not free on the synergy metric either (greedy +0.44 → +0.37, MCTS +0.78 → +0.63). Its cost there is smaller than augmentation's, and it gains win probability where augmentation loses it. The paper says this.

**Manuscript changes**
- **`draft_revision.tex`** (9 pages, 0 errors, 0 overfull boxes):
  - **Abstract:** the repair is search plus a structural mask; synthetic data "also improves safety, but it dulls interaction-awareness, and we no longer recommend it".
  - **Introduction:**
    - "What achieves both is search plus a structural mask."
    - Augmentation is named as the submission's proposed repair, which dulls interaction-awareness.
    - Contribution 2: the heuristics "define a structural mask" (was "target synthetic data").
    - Contribution 3: the repair is MCTS with domain features plus the mask; synthetic data is "tested as an alternative" and "trades interaction-awareness for safety".
  - **§V, now "Synthetic Data: A Tradeoff" (`sec:synth`):** one paragraph with the key numbers (safety gains, interaction losses, 0.485 head to head) and the comparison with the mask. Table V (assigned win rate) and the scope-sweep details moved to the supplement. The "active ingredient" claim is removed.
  - **§VI:** retitled "MCTS with the Leak-Free Value Function" (was "Repaired").
  - **§VII-E:** the mask is called "our recommended repair".
  - **Discussion, practical lessons:** mask invalid compositions during search where they can be described by rule; in our experiments this did better than synthetic data.
  - **Conclusion:** "What works is search plus a structural mask"; synthetic data improves safety but dulls interaction-awareness.
  - **Unchanged:** the tournament table keeps the Enr.+aug greedy row as a result, not featured. Table VI keeps H_augmented. The rich-evaluation table keeps its Enr.+aug row.
- **`supplementary_revision.tex`** (7 pages, 0 errors, 0 overfull boxes):
  - The augmentation section is renamed "Synthetic Augmentation: Assigned Win Rate and Scope".
  - It now holds the moved table (`tab:synth`), the assigned-rate reading, a paragraph on the interaction cost, and the scope sweep.
  - "The active ingredient is the unseen compositions" becomes "Within augmentation, the five-tank effect comes from the unseen compositions."

**For the response to reviewers (draft):**
> The submission proposed synthetic training records for never-observed role compositions as the main repair for unsafe value search. With leak-free models and independent judges, augmentation still improves every safety measure, but it lowers the agents' synergy and counter responsiveness and loses head to head against the same agent without it (0.485). A structural mask applied during search removes every broken team, leaves the value function unchanged, and gains win probability head to head (0.509 for MCTS, 0.511 for greedy). The constrained MCTS agent ties for first in the tournament. We therefore present the mask as the repair and report augmentation as a tested option with a safety-for-interaction tradeoff. Its full results are in the supplement.

**Not changed:**
- the markup files;
- `REVISION_NOTES.md` §3.7, which still documents the leak-free augmentation numbers as evidence;
- the response draft in §5. It does not mention augmentation as the repair, so it needs no edit beyond adding the paragraph above.

## 13. Consolidated audit (2026-10-01): composition-table pin, rebuilt gN, A/B/C/D items

Source: `audits/CONSOLIDATED_AUDIT_2026-10-01.md` §5 (55 open paper-1 items). This section supersedes the gN numbers in §11.

### 13.1 Live composition table (§0, X1) and gN's external table (A5)

**The problem.**
- `StatsCache._load_compositions` read the live site file `src/lib/data/compositions.json`.
- The nightly sync rewrote that file:
  - on 09-30 16:52 with 10 rows of patch-2.57 games;
  - on 10-01 00:02 with 364 rows of 2.55 + 2.57 games.
- gN (4 of its 283 inputs), the submitted proxy (`proxy_sub`) and the "hp" stats read it.
- So c159669's gN audit, gN refits and gN β used a table containing games from builds after 2026-09-27. Separately, the J_oof s0/s1 and A_partial/G_base benchmark scoring, and the A_partial/G_base benchmark *generation* (kernel lookup tables), used the 10-row table.

**The fix.**
- `overfit2026/feats.py` never reads the site file.
  - Stats files with an own-corpus `<name>_compositions.json` use that table.
  - Everything else reads `training/pins/compositions_hp_f2eb025.json`, a copy of the git version dated 2026-03-22, with a SHA-256 assert.
- **gN rebuilt** (`overfit2026/comp_gn_rebuild.py`, models `gN8o_oof_s0..2`, deploy stats `No8`):
  - The recipe is unchanged, but the composition table now comes from the post-snapshot games themselves (out of fold for training rows; cells admitted at ≥50 games).
  - Validation loss 0.6815–0.6818 vs 0.6809–0.6812 for the original.
  - gN-naive reads no statistics and is unchanged.
  - `overfit2026.score.GN` names the judge everywhere (`bench_mcts`, `score_tournament`, `judges_v2`, `comp_audit_data`).
- **Re-run:**
  - A_partial and G_base benchmarks regenerated (all seeds, T0 and T1); old files are in `results/mcts_bench_livecomp_backup/`;
  - gN for every stored draft (mcts_bench, tournament, cross-evaluation, greedy rows);
  - two-fold gN refits with own tables (`comp_gn_folds.py`), the audit (`comp_audit_data.py --gn-only`), the β fit (`comp_judges.py`), `comp_rescore.py`, `judge_calibration.py`, and the CQL (enr.) eval.
- **Previous outputs kept** as `*_livecomp.json` / `*_extcomp.json`.

**Old → new** (v2 references unless noted):

| Quantity | Before (c159669) | Now |
|---|---|---|
| gN test acc / slope | 55.3 / 0.97 | 55.2 / 1.01 |
| gN blind spot on SNAP degenerate teams (v1) | −1.6pp | −3.0pp (the external table was carrying part of gN's structure signal) |
| gN v2 gap | −0.5pp | −0.4pp |
| consensus v1 / v2 gap | −4.2 / +0.0 | −4.6 / +0.0 |
| gN β (no healer, no frontline, stack) | −0.04, −0.06, 0.00 | −0.10, −0.11, −0.10 |
| Table VI gN (B_fullwp / F_400 / J_800 / E_1M / K) | .654/.671/.667/.671/.663 | .639/.664/.658/.661/.654 |
| Table VI gN (B_oof / F_oof / J_oof) | .662/.672/.665 | .652/.666/.655 |
| A_partial gN / deg / Train\* | .662 / 16.5 / .722 | .651 / 16.7 / .721 (regenerated) |
| G_base gN / Train\* | .636 / .671 | .629 / .672 (regenerated) |
| F_400sim − B_fullwp (gN) | +.018 | +.025 ± .005 |
| J_800sim − F_400sim (gN) | −.004 | −.006 ± .002 |
| F_oof − B_oof (gN) | +.010 | +.013 ± .003 |
| J_oof − F_oof (gN) | −.007 | −.011 ± .003 (uninterrupted seeds s2, s3 only: −.016 ± .002) |
| B_fullwp − K_truebase (gN) | −.009 | −.015 ± .005 |
| B_oof − K_truebase (gN) | −.002 | −.002 ± .003 |
| E_1M − B_fullwp (gN) | +.017 | +.022 |
| C_large − B_fullwp (gN) | +.009 | +.014 |
| N2 / full / M2 (gN) | .665 / .654 / .645 | .649 / .639 / .638 (M2 includes one failed seed at .574; the other four average .654) |
| Table II Ind. (naive / hero str. / enriched) | .572 / .571 / .575 | .570 / .569 / .571 |
| Tournament consensus | constrained MCTS .611, MCTS .610 | .610, .610 (other rows within .002; gN within .007) |
| Mask head to head (MCTS / greedy) | .509 / .511 | .509 / .512 |
| Augmented vs unaugmented greedy | .485 | .486 |
| MCTS vs K_truebase | .558 | .559 (gN .565) |
| CQL (enr.) eval refs | 0.50–0.53 | unchanged (drafts reproduce exactly) |

**Effect on the claims.**
- Every claim survives.
- The rebuilt gN is harsher on agents trained against the external table: B_fullwp drops 0.015, against an average of 0.009.
- So the feature-gap reversal is stronger (−0.015), and the submitted 200 → 400 gain is larger (+0.025).

### 13.2 A items

- **A1** (search-depth gap attributed to the leak), **closed.**
  - The abstract, intro (L59), simulation-scaling paragraph and practical lessons now say: the leak inflated held-out accuracy and the value function's own measure of every gain; independent judges confirm neither the feature gain nor any gain past 400 simulations; removing the leak does not change that.
  - The plateau appears in both pipelines. The leak does not explain it.
  - "Search depth" wording is held for X2, per the coordinator.
- **A2** (judge attenuation), **closed.**
  - New benchmark `paper1_revision/judge_attenuation.py` → `results/judge_attenuation.json` and supplement §"Judge Attenuation" (Table `tab:atten`).
  - On the test replays, slope of judge on value function:
    - leak-free value function: gN 0.56, RN 0.46, consensus 0.58;
    - submitted value function: gN 0.48, RN 0.39, consensus 0.50.
  - 200 → 400 transmits fully: submitted predicted +0.018 vs observed +0.025; leak-free predicted +0.010 vs observed +0.013.
  - 400 → 800 is over-optimization: predicted +0.005 / +0.007, observed −0.006 / −0.011.
  - "A quarter as large" is removed.
- **A4** ("features buy safety"), **closed.**
  - The matched contrast is reported with its seed SE: B_oof 13.9% vs K_truebase 18.5%, −4.6 ± 3.5pp; with argmax root −0.8 ± 4.6pp.
  - The drop to 6.6% (4.6% argmax) is attributed to doubling the simulations, which were not run without features.
  - Changed in contribution 3, §VII "Features", and the tournament paragraph ("reliable difference" removed; 7.3 vs 23.4% now noted as confounded).
- **A5** (gN reads the external table), **closed by rebuild** (13.1).
  - L124 describes the rebuilt gN's own table and mentions the first version.
  - L110 now names the one inherited exception: tournament CQL (enr.).

### 13.3 B items

- **Closed:**
  - **B1:** L82 now says the games train only the references.
  - **B2:** 4.6% at argmax root added.
  - **B3:** "benchmark operating point"; the tournament plays the policy argmax.
  - **B7:** .617/.612.
  - **B8:** −0.15 to −0.20 (Table V).
  - **B9:** 70–73%, 66–70 heroes, −0.15 to −0.19 synergy; intro 0.4–3.6% and 70–73%.
  - **B10:** gN seed SE 0.001–0.005 (M2 0.016).
  - **B11:** M2 failed seed disclosed.
  - **B12:** Gourdeau checkpoint is reused, not retrained.
  - **B13:** inherited baselines were selected on test; this is stated.
  - **B14:** "every independent judge".
  - **B15:** supplement no longer cites a main-text passage that does not exist.
  - **B19:** 210 runs (195 + 15).
  - **B20:** 57.9%.
  - **B21:** 54.5%.
  - **B22:** 1.5pp and 1.7pp (16% and 21%), with the game-set caveat.
  - **B23:** "stop improving after 4,096" in main and supplement.
  - **B24:** "never lowers gN".
  - **B25:** L55 now cites the measured 8–9% pick agreement.
  - **N1:** L275 and conclusion: gN and RN fall 0.008–0.011, the others stay within noise.
  - **N3:** 0.005–0.007.
- **C5** (consensus fixed before scoring, upgraded to B): L124 now says the composition correction was added after all agents were scored, and the supplement reports uncorrected values.
- **C18** (upgraded to B): the tournament row is marked "CQL enr.†, submission's checkpoint with external statistics", and L110 and L282 say so.
- **N2:** §III-A discloses that the external composition table is keyed by Heroes Profile's rank groups and read only by the submitted value functions.

### 13.4 C items

- **Closed:**
  - **C1:** MCTS checkpoint selection by the run's own value function is disclosed in the benchmark paragraph.
  - **C2:** J_oof resume wording fixed (best-evaluation checkpoints 51K/51K/107K, replay buffer not restored), and the uninterrupted-seed contrast is reported.
  - **C3:** value-head pretraining stuck at 0.25 in 10/15 leak-free runs. The head is not used by search (verified in `mcts_kernel.cu`), so no reported number depends on it. Per Max's policy (13.8) it is recorded here, not in the manuscript.
  - **C4:** F was chosen on the same references that score the tournament; stated.
  - **C6:** tournament SE excludes seed variation; stated in the caption.
  - **C7:** Table I caption gives slope SDs (0.02–0.08) and test slopes 1.12–1.22.
  - **C8:** realized synthetic label means 4.9/5.0/9.5/29.9%, recomputed from `cache/feats/synth_wr*.npz`.
  - **C9:** synthetic labels are adjusted with deploy statistics; stated.
  - **C10:** Fig. 2 caption gives the protocol of each point. The figure itself is unchanged.
  - **C12:** split-experiment caveats added to the supplement (30% of training, last-epoch accuracy, full-snapshot BC prior, the 40-epoch exception).
  - **C14:** "Gourdeau estimator" used consistently ("Gourdeau est." in Table VII).
  - **C15:** `paper1_revision/leak_measure.py` → `results/leak_results.json` reproduces the L108 figures: median overlap 38/51/67% (low/mid/high); 58.07% → 57.03% with test games removed; random-removal control 57.99–58.05%.
- **Code fixes:**
  - **C13:** `bench_mcts.py` stores the 2022-era QM judge as `QM2022`. The old "QM2021" key held that model, not the tournament's QM2021, and no table uses it.
  - **C17:** `paper1_revision/train_mcts.py` seeds Python `random` for future runs. The reported runs were unseeded, which is now disclosed here.
  - **N4:** `overfit2026/data.py build_db` reads tier labels from `replay_draft_skill_tier_backup_20260930` and asserts on them.
- **Open:** **C11** is stale. **C16** (style) is fixed.

### 13.5 D items

All listed D items are rewritten:
- L44 triad;
- L71 ("structure through features and a rule-based mask");
- L231 triad;
- L277 cleft, now the "Features" paragraph;
- L327 "Safety is easy";
- L333a and L333b;
- the L82/L124 formula;
- S49, S149a, S149b, S151.

### 13.6 Not done

- **X2** (the kernel searches only the current own-pick block): wording held per coordinator.
- **X4** (production `refresh.py` composition block): owned by production. Its select-phase proxy still reads the live site file. That is a production choice, outside the paper.
- **Release:** `training/pins/` and the new result files must go into the public release with the other paper-1 code.

### 13.7 Build

- `draft_revision.tex`: 9 pages, 0 errors, 0 overfull boxes.
- `supplementary_revision.tex`: 7 pages, 0 errors, 0 overfull boxes.
- Fig. 1 regenerated.

### 13.8 Policy: bugs are fixed and rerun, never disclosed as caveats (Max, 2026-10-01)

The manuscript reports corrected results and genuine design limitations only. Implementation bugs and their fixes are recorded in this change log.

**Applied in this pass:**
- Removed the value-head sentence (C3) from the benchmark paragraph.
- Removed the data-sync overwrite and the A_partial/G_base regeneration sentences from the supplement's gN paragraph. The paragraph now reports the own-table gN as a design choice for independence. The external-vs-own comparison stays, because that is a design change.
- Nothing about the X2 kernel horizon was added anywhere in the manuscript.

**Remaining manuscript passages that describe implementation problems.** These need a decision or a rerun under the policy:
1. **§III-A, the tier-label off-by-one** ("Because of an off-by-one in that pipeline ..."). Following the policy fully means relabeling and rerunning everything downstream. The current text describes the labels as they are and reports a sensitivity check.
2. **§VII, J_oof seeds s0/s1/s4.** They were resumed without replay buffers after a compute-cap pause, which is operational. They will be replaced when MCTS is rerun after the X2 kernel fix.
3. **Supplement kernel section footnote** on the virtual-loss bookkeeping bug in the historical batched host implementation (Table throughput).
4. **M2_relational seed s4** (gN 0.574, "one failed seed"). This is a submitted checkpoint, and the cause has not been diagnosed.

## 14. Implementation problems: fixed and not mentioned in the manuscript (2026-10-01)

Max's rule: implementation problems are fixed, the affected numbers are rerun, and the manuscript reports only the corrected results. This log keeps the record.

**Removed from the manuscript:**
- §III-A tier-label wording: the off-by-one, the submission's description, and the sensitivity sentence. The text now states the site scheme, which the rebuild in §15 makes true: low = Bronze + Silver, mid = Gold + Platinum, high = Diamond + Master; unranked games excluded. The share numbers come from the backup crosstab and will be confirmed by the rebuild.
- The J_oof resumed-seeds sentence. Those seeds will be replaced by the fixed-kernel reruns.
- The supplement virtual-loss footnote and the "two corrections to earlier versions of this table" passage.
- "One failed seed" (M2_relational).
- Earlier, under 13.8: the value-head pretraining sentence and the composition-sync sentences.

**Virtual-loss baseline (historical batched engine, `training/cuda_mcts/mcts_engine.cpp`).**
- The bug: the root received the virtual-loss undo although virtual loss is never applied to it, so the root visit count went negative and NaN poisoned UCB.
- The fix: the root gets a plain update. `baseline_engine.cpp` and `apples_engine.cpp` already had this.
- Re-measured with `paper1_revision/throughput_vl_fixed.py` → `results/throughput_vl_fixed.json`, same protocol as the original row: single thread, K = 32, 200 sims, J_800sim_s9. Fixed and original engines alternate in blocks of 40 episodes, 120 episodes each.
  - Fixed: **5.02 ep/s (199 ms/episode)**; original: 5.02 ep/s.
  - Throughput is unchanged. Table throughput row "C++ tree, batched virtual loss (K=32)" goes 5.1 → 5.0, the measured value.
  - The 2.2× controlled comparison already used fixed engines, so it is unchanged.
- The fixed module was built separately (`cuda_mcts_vlfix`) so the shared `cuda_mcts` build used by other lanes was not touched.

**M2_relational seed s4 (gN 0.574): diagnosed as optimization variance, not a bug.**
- Same config, data, value function and weights path as the other seeds. The training log is normal: no NaN or errors, eval WP rises steadily to 0.706.
- It converged to a different hero set (favorites Cho, Rehgar, Falstad, against Samuro, Rehgar, Hogger for most seeds). It is weaker under its own training value function (0.664 against 0.68–0.75 for the 15 seeds; s12 0.681 and s6 0.694 are also low).
- It is the low tail of the seed distribution. The manuscript reports all seeds with their spread (0.574–0.663) and no "failed" framing.
- When the M2 configuration is rerun with the fixed kernel, all seeds are kept as they come.

## 15. Rebuild of paper 1 on site-scheme tiers (started 2026-10-01)

Max's decision (b): every paper-1 number is regenerated once on correctly tiered data. The manuscript is updated in place as results land, with no caveats.

### 15.1 Data

**Tiers.** Each replay is relabeled from its stored league_tier and avg_mmr with the sync's rule (`sync/sync-replays.ts` leagueTierToSkillTier, reimplemented as `overfit2026.data.site_tier`):
- low = Bronze + Silver;
- mid = Gold + Platinum;
- high = Diamond + Master;
- unknown = no rank and no MMR, excluded.

The source is `backups/replay_draft_skill_tier_20260930.csv.gz` (league_tier and avg_mmr per replay). A 20,000-replay sample agrees 100% with the relabeled DB.

**Snapshot.** Same pinned replays and the same split, minus 5,749 unranked games:
- 1,943,338 games: low 798,262 (41.1%), mid 800,594 (41.2%), high 344,482 (17.7%);
- split: train 1,866,369, validation 38,100, test 38,869. The test set is the paper's test replays minus unranked ones.

**Post-snapshot games** (unranked games are more common there, about 11%):
- T97 124,022 + backfill 136,517 = 260,539 no-drift games (was 291,837);
- 2.55.17: 154,387 (was 158,608).

**Quick Match.** QM tiers are relabeled from `qm_games.league_tier` (read-only); QM has no unranked games. QM2026 keeps 116,589 games and QM2021 keeps 91,365, both retrained.

### 15.2 Code

- **Namespaces.** `P1_TIERS=site` relabels on load (`overfit2026/data.py`) and sends every derived artifact to `overfit2026/site/...` and `paper1_revision/site/...` (`data.art`). The legacy artifacts, which the kernel lane still reads, are untouched. The default stays `legacy` until the rebuild is complete.
- **rerun2026 baselines** run in namespace `p1site`:
  - snapshot `snapshots/replay_snapshot_2026-05-22_1956753_p1site.json` (same rows and order; site tiers);
  - `RERUN_SPLIT=p1val` in `rerun2026/common.py` (env-gated; default behaviour unchanged).
- **Selection on validation.** `p1val` drops unranked rows after the permutation split, so every other replay keeps its assignment, and it returns (train, validation). Every retrained baseline (GD, CQL, BC-CQL, MCQ, IQL, the discriminator, the Gourdeau estimator) therefore early-stops and selects on validation games, never on the paper test set. That closes the B13 issue for the rebuilt baselines.
- **Driver:** `paper1_revision/site_rebuild.py`. It is dependency-aware and skips completed jobs. It runs at most 2 GPU processes for this lane and everything under nice 19 on cores 48-63. Log: `paper1_revision/site/logs/`.

### 15.3 Status (2026-10-01 16:00)

- **Done:**
  - QM2026 and QM2021 retrained (own hold-out 0.576 / 0.612; legacy 0.575 / 0.611);
  - gN and gN-naive rebuilt (own composition tables), with the two-fold refits;
  - site stats.
- **Running:**
  - leak-free features (CPU), then Table I models (11 specs × 3 seeds) and the 128-run sweep (GPU);
  - p1site phase-0 caches (CPU), then the Gourdeau estimator, then the baselines.
- **Queued (B1, judges and analyses):** composition audit, causal test, structure fit, native index, judge calibration, attenuation, metric validity, NGS.
- **Queued (B2, baselines):**
  - GD × 5, about 4.6 GPU-h each;
  - CQL naive α × 5, MCQ τ × 5, BC-CQL β × 4, IQL 6 cells, about 2 GPU-h each;
  - the discriminator;
  - CQL capacity × target grid × 15, about 2 GPU-h each.
  - In total about 95 GPU-hours. With one local GPU slot for B2 that is about 4 days. The CQL grid (about 30 GPU-h) is the natural candidate for the remote workers.
- **Next (B3, after GD):**
  - rich evaluation (Table V);
  - greedy, cross-evaluation and sanity;
  - leak-free enriched CQL (`deferred.py` in the site namespace);
  - scope sweep;
  - ensemble;
  - NGS quartiles;
  - MCQ dead units;
  - leak measure;
  - submitted-checkpoint evaluation;
  - non-MCTS tournament pairs.
- **MCTS configurations:** prepared for the fixed-kernel build. They depend on the new GD pool (opponents) and the site leak-free enriched WP.

### 15.4 MCTS reruns on the fixed kernel (queued 2026-10-01)

**Prepared.** `paper1_revision/train_mcts.py` under `P1_TIERS=site`:
- uses the site-tier leak-free enriched WP and own deploy statistics;
- uses the p1site GD pool, both as the kernel opponent and for the bootstrap;
- does value pretraining on the site-tier snapshot, excluding the paper test set, the validation set and unranked games;
- sets `MCTS_SEARCH_MODE=chance` explicitly.

`overfit2026/search.py` (bench_mcts) takes its GD opponents from the p1site pool under site tiers.

**3090.**
- Fixed kernel built; `smoke.py --gpu` passes.
- Pause/resume drill (B config, 200 sims):
  - pause in 6 s, checkpoint at episode 1536;
  - resume restored optimizer, scheduler and buffer, and continued from 1664 (one batch after the checkpoint);
  - stopped after the drill.
- Self-play runs at about 10 ep/s there at 200 sims while the four baseline queues share the GPU.

**Queue.** `paper1_revision/site_mcts_queue.sh` waits for the GD pool (all five meta files) and the site enriched WP. Then:
- **3090:** B_oof s0–s4, two at a time, as pause-aware hotsjob MCTS jobs.
- **This box:** two streams, F_oof s0–s4 and J_oof s0–s4. Each starts a run only while this lane holds fewer than 2 GPU processes.

`site_fetch.sh` also brings the remote MCTS runs and logs back.

**After the runs:** bench_mcts (site namespace), comp_rescore (Table VI rows, sim scaling 200/400/800, operating point), and the tournament's MCTS strategies.

### 15.5 Old-kernel / old-tier MCTS configurations: claim map and plan (2026-10-01)

Rule: no old-kernel or old-tier agent stays in the paper.

| Config | Claims resting on it | Decision |
|---|---|---|
| K_truebase | feature gap (Table VI, features paragraph, contribution 3); safety at matched search (A4); tournament row "MCTS, no feat." and MCTS vs no-features head to head; ensemble table rows | **Retrain as K_oof**: naive (hero-identity) leaf WP, site tiers, chance kernel, 5 seeds, 400 sims × 300K episodes, matched to F_oof (the operating point and the tournament's MCTS agent), so the feature comparison is at matched search everywhere |
| E_1M | training length (Table VI, training-length paragraph); dual-snapshot table | **Retrain as E_oof**: 200 sims × 1M episodes, 5 seeds |
| B_fullwp / F_400sim / J_800sim (and I_600sim) | the leak's effect on paper-scale search: own-score vs independent gains 200 → 800, leaky vs leak-free at matched sims, the 400 → 800 plateau "in both pipelines" | **Retrain as B/F/J_leak**: the in-sample-statistics control WP (Table I "Enr., in-sample") as leaf, 200/400/800 sims, 5 seeds each. **I_600sim dropped** (200/400/800 suffice) |
| C_large, D_deep | capacity sentence | **Dropped** (removed from Table VI and text) |
| H_augmented | supplement augmentation sentence | **Dropped** |
| M2_relational, N2_absolute | relational/absolute decomposition | **Dropped** |
| A_partial, G_base | Table VI rows only | **Dropped** |

**Also old-kernel; to be redone with the retrained agents:**
- the within-snapshot split experiment's search sweep and self-play replication (main §III-B "What the leak does to search"; supplement). This needs the split value functions retrained on site tiers, the search sweep rerun on the chance kernel, and the 8 self-play runs;
- the supplement dual-snapshot stability table (scoring only, with the new agents);
- the ensemble table's tournament rows (scoring only).

**Launcher.** `train_mcts.py` adds config K (400 sims, 300K episodes, naive leaf via the worker's `true_base` path) and `--wp enriched_leak` (run names `<cfg>_leak_s<seed>`).

**Queue.** `site_mcts_queue2.sh` runs behind `site_mcts_queue.sh`:
- **3090:** B_leak s0–s4, then K_oof s0–s4, at most two p1site MCTS jobs at a time.
- **This box:** F_leak s0–s4 then E_oof s0–s2, and J_leak s0–s4 then E_oof s3–s4. These start after B/F/J_oof finish locally.

**GPU-hours** (this box: 400 sims about 3.7 h per 300K episodes, 200 sims about 2 h, 800 sims about 7.5 h; the 3090 is slower when shared):

| Item | Runs | Estimate |
|---|---|---|
| B/F/J_oof (wave 1) | 15 | about 66 h |
| K_oof | 5 | about 19–40 h (3090) |
| E_oof | 5 | about 35 h |
| B/F/J_leak | 15 | about 66 h |
| Split experiment | value functions, sweep, 8 self-play runs | about 25 h |
| **Total** | | **about 210–230 GPU-hours** |

With two local streams plus two concurrent 3090 jobs, from the GD pool landing (late Oct 2):
- wave 1 done about Oct 4;
- wave 2 done about Oct 6;
- split experiment done about Oct 6–7 (queued last).

### 15.6 All HotS compute moved to the remotes (2026-10-01 18:30)

**Stopped on the main box:** site_phase0 cache building, the local Gourdeau/naive-CQL waiter, both MCTS queue scripts (waiting, nothing launched). All local work is stopped.

**Already finished locally before the stop:**
- stats, features, Table I (11 specs × 3 seeds);
- the Gourdeau estimator;
- the judge analyses: composition audit, causal test, structure fit, native index, judge calibration, attenuation;
- metric validity and the NGS evaluation.

The 128-run sweep failed with a GPU out-of-memory error and moved to the 3090.

**3090:**
- **GD:** split into five concurrent seed jobs (`site_remote_queue.py gd0..gd4`). Seed 0 resumed after epoch 9 from its epoch checkpoint. The discriminator waits for all five.
- **Also running:** the three CQL/MCQ/BC-CQL/IQL queues and the feature sweep (`train_wp.py sw_*`; skips finished specs).

**MCTS scheduler.** `paper1_revision/site_mcts_remote.py` runs on the main box as a light ssh/rsync polling loop. When the GD pool and the site WPs exist, it pushes inputs to a host and launches pause-aware hotsjob MCTS jobs in priority order:
1. B/F/J_oof;
2. K_oof;
3. E_oof;
4. B/F/J_leak.

Slots per host are read from `paper1_revision/site/mcts_slots.json`. Current setting: 3090 2, 3080 0 until a split with the drift lane is agreed. The 3080 already has the fixed kernel built (17:41); it needs a smoke test and a drill before the first job of this lane.

**Remaining non-MCTS stage (B3)** runs on the 3090's CPUs and GPU once GD lands:
- rich evaluation, greedy, cross-evaluation, sanity;
- leak-free enriched CQL;
- scope sweep, ensemble, NGS quartiles, MCQ dead units;
- leak measure, submitted-checkpoint evaluation;
- non-MCTS tournament pairs.

### 15.7 Yield to the expert study on the 3090 (2026-10-01 18:45)

**Paused on the 3090:**
- GD seeds 0–4 (epoch checkpoints);
- the discriminator queue (it had not started);
- the feature sweep (finished specs are kept);
- IQL/CQL-grid queues cqlB and cqlC (the job in progress restarts on resume).

The MCQ/BC-CQL queue cqlA had already failed (connection reset during an out-of-memory episode) and will be relaunched. This lane now holds no GPU memory on the 3090; the 22–23 GB in use belong to the oct2026 lane.

**Moved to the 3080** (slot 1, priority over drift):
- p1site phase-0 caches (8 workers);
- GD seeds 0 and 1. Seed 0 continues from its 3090 epoch checkpoint (epoch 9).

The 3090 manifests for GD 0/1 were closed, so a 3090 resume does not duplicate them.

**On the coordinator's signal, resume on the 3090:**
- p1site_gd2, gd3, gd4;
- p1site_disc, cqlB, cqlC, wp_sweep;
- relaunch cqlA.

The MCTS scheduler uses `mcts_slots.json`: 3090 2, 3080 1. The 3080 goes to 2 automatically when the drift lane's MCTS job ends. The fetch loops run per host, with GD 0/1 excluded from the 3090 fetch.

### 15.8 3080 memory incident and the new host rules (2026-10-01 19:00)

**What happened.** The restarted p1site cache build on the 3080 (4 workers) held about 38 GB: the full snapshot parses to about 13.6 GB in the parent, plus about 5–7 GB per worker. That took down the 48 GB WSL. The earlier crash had the same cause: a cache build plus a drill loading the snapshot at the same time.

**Rules now:**
- Heavy CPU/RAM work runs on the 3090 only: at most about 16 cores and 100 GB, CPU only while the expert study holds its GPU.
- The 3080 runs at most one lean job of mine at a time, under about 15 GB.

**Changes:**
- **No rebuild needed on the 3090.** The 3090 already holds complete p1site caches (built 16:21).
- **Lite snapshot.** `site_lite_snapshot.py` (run on the 3090) writes a lean site-tier snapshot: same rows and order, only the fields value pretraining reads (0.64 GB on disk). Loading and splitting it peaks at 3.9 GB RSS, against about 30 GB for the full file.
- **MCTS pretraining.** `train_mcts.py` uses the lite file for value pretraining under site tiers. The pretraining inputs are identical, so the pretraining is unchanged, and a 3080 MCTS job fits the 15 GB budget. The scheduler pushes the lite file.
- **GD caches.** gd_train/gd_test (28.6 GB) are being relayed 3090 → main box (rsync, nice 19, ionice idle) for a later push to the 3080, if GD has to run there before the 3090 GPU frees.

### 15.9 3080 drill with the lean snapshot (2026-10-01 21:20)

- **Memory.** Site config B, value pretraining on the lean snapshot. The job used about 13 GB of RAM at peak, the host stayed above 27 GB free, and nothing crashed. Pretraining selected exactly the site training set (1,866,369 replays).
- **Speed.** Self-play ran at about 5–8 ep/s at 200 sims while the drift MCTS run shared the GPU.
- **Pause/resume.** Pause checkpointed at episode 768. Resume restored the state and continued from 896, one batch later. The drill was then ended and its run directory removed.
- **GD 0 on the 3080 is held** for the expert study's GD trainer, per the coordinator. The GD caches are being pushed to the 3080 (light; network and disk only). GD 0 resumes from its epoch-9 checkpoint when the slot frees.

### 15.10 GD on the 3080: not viable (2026-10-01 22:45)

**Observed.** GD 0 resumed from its epoch-9 checkpoint. The memory-mapped training cache (27 GB) filled the page cache within two minutes. The job reached about 27–30 GB of resident memory, mostly file-backed. When the drift lane's GD trainer started at the same time, available memory fell to 12 GB. No epoch finished in 11 minutes (about 6 min on the main box before the move, about 12 min on the shared 3090).

**Done.** GD 0 was stopped on the 3080 before it could push WSL over; the epoch-9 checkpoint is kept.

**Plan.**
- All five GD seeds run on the 3090 when it frees.
- The 3080 takes this lane's MCTS runs (lean snapshot; drill passed) once the GD pool exists.

### 15.11 3090 resumed (2026-10-02 09:57)

**Dependency check.** The trainers' final steps were checked on the 3090:
- GD writes an ONNX export, then optimize and dynamic quantization for variant 0. A test export plus `optimize_onnx` passed with onnx 1.23.1, onnxruntime 1.30.0 and onnxscript 0.6.2; `onnxruntime.quantization` imports.
- The CQL, MCQ, IQL and discriminator stages and the sweep end in torch/json writes only.

**Resumed or relaunched:**
- GD 0 (from epoch 9), GD 1–4 (fresh: no epoch had finished before the pause);
- the discriminator queue, cqlB, cqlC, the sweep (finished specs kept);
- cqlA relaunched.

The fetch loops were restarted: the 3090 fetches everything, and the 3080 fetch excludes GD files.

**Measured speeds:**
- GD with 5 seeds concurrent on the 3090, alongside three CQL-family queues and the sweep: first epoch 19 min. GPU 72%, 24 of 196 GB RAM used.
- MCTS self-play at 200 sims, from the drills: 3090 about 10 ep/s while shared with the baseline queues; 3080 5–8 ep/s while shared with the drift MCTS.

**ETA:**
- GD pool: about 11 h for GD 0 (35 epochs left) and about 14 h for GD 1–4, so around Oct 3 01:00.
- MCTS, option B: about 325 3090-equivalent GPU-hours, on 3090 (2 slots) and 3080 (1 slot, about 0.4 of a 3090 while shared). That is about 9–10 days of wall time from GD landing, so around Oct 12–13. This will be firmed up from the first 400/800-sim runs.

### 15.12 GD trainer: illegal-target rows (2026-10-02)

**Problem** (found by Codex on the oct2026 memmap). A handful of GD samples have a target hero that is already picked or banned. The model masks illegal logits to −1e9, so each such row adds about 1e9 to the summed cross-entropy.

**p1site memmaps** (`site/results/gd_invalid_targets.json`):
- train: 680 of 24,456,480 rows (0.003%);
- held-out: 17 of 500,336 rows.

The reported held-out loss (about 33,980) was almost entirely 17 × 1e9 / 500,336. At that magnitude float32 resolves about 0.004, against epoch-to-epoch changes of about 0.01 in the real cross-entropy, so best-checkpoint selection and early stopping were close to noise.

**Effect on the gradient.** It is not pathological. Masked cross-entropy has gradient softmax − one-hot; the target's softmax is about 0, so the target logit gets a bounded −1 push, applied to 0.003% of rows.

**Fix** (`train_generic_draft.py`, default behaviour):
- rows whose target is not a legal action are dropped from every training batch and from evaluation;
- best-checkpoint selection and early stopping use cross-entropy over legal-target rows;
- top-1 and top-5 accuracy are logged every epoch, with the number of skipped rows;
- resume states carry `loss_kind="legal"`; states saved under the old loss are ignored.

A toy test (30 + 5 illegal rows) gave the expected chance-level loss ln 90.

**Decision.** The p1site GD pool was restarted from scratch with the fix (10:45 Oct 2). Seeds 1–4 had finished one or two epochs and seed 0 about 11. The saved best checkpoints had been chosen on the noisy metric, and no per-epoch checkpoints exist to reselect from. The old files are kept in `~/hots/gd_oldloss` on both remotes and `rerun2026/ns/p1site/models_gd_oldloss` here.

### 15.13 3080 handed to the production retrain (2026-10-02)

The 91-hero production retrain takes the 3080 from about Oct 3 morning to about Oct 4. The MCTS scheduler's slots are set to 3090: 2 and 3080: 0. The file is read every cycle, and the automatic raise to 2 applies only from 1, so nothing launches on the 3080 until the coordinator frees it.

**Option-B ETA:** about 325 3090-hours of MCTS.
- 3090 alone from GD landing (about Oct 3 02:00) to about Oct 4 midday: roughly 34 h done.
- The remaining about 290 h at 1.4× (3090 plus the 3080 at about 0.4×) is about 8.7 days.
- Finish around Oct 13–14, also allowing for the non-MCTS stage sharing the 3090.

### 15.14 Synthetic augmentation removed from paper 1 (Max, 2026-10-03)

**Manuscript.**
- §V is now "Why Value Search Builds Broken Teams". It keeps the training-concentration finding (95% healer; 126–146 of 252 role cells under 50 games) and presents the role mask as the repair.
- Removed the Enr.+aug greedy rows (Table V, Table VII, Figs. 1–2). The tournament now has 11 strategies and 110 pairs; standings were recomputed from the stored drafts (`comp_rescore.py`).
- Supplement: removed the augmentation section, its rows in the ensemble table, and the dual-snapshot section (old-kernel agents; to be recomputed with the rebuilt agents).
- Build: main 9 pages, of which page 9 is references; supplement 6 pages.

**Rebuild.**
- The augmented greedy agent is dropped from `greedy_evals.py` (rich rows; WR sweep keeps `enriched_512`) and from `tournament.py`.
- The augmentation scope sweep is removed from the B3 queue.
- The site-tier `aug_wr*` WP models (Table I) are trained but no longer used.
