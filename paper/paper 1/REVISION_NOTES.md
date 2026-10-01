# Paper 1 revision notes (2026-09-30)

Paper: "Diagnosing and Repairing Out-of-Distribution Failure in MOBA Draft Policies", under review at IEEE ToG.

**Revised manuscript files** (in `paper/paper 1/overleaf/`)

- **`draft_revision.tex`**: the clean revision (no change markup), trimmed to fit the submission's format.
  - Clean build is **8 pages** (the submitted `draft.pdf` is 10). Built with `pdflatex` twice, checked with `pdfinfo`.
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
