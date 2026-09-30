# Did paper 1's best result overfit the WP model? (2026-09-29)

## Verdict

**Partly, and the cause is the value function's statistics more than search depth.**

1. **About 75% of the sim-scaling gain exists only under the proxy.** From 200 to 800 training sims, the paper's score rises +0.041 (0.726 → 0.768). Independent references rise only +0.008 to +0.010, all of it by 400 sims. From 400 to 800 sims:
   - the proxy gains +0.014 (z = 7.2);
   - post-snapshot judges (gN) gain +0.000;
   - the post-snapshot realized-outcome index (NODRIFT) moves −0.002 (z = −1.1);
   - the index on the drifted 2.55.17 builds (T17) moves −0.006 (z = −3.0).

   That is the Goodhart signature: the proxy keeps rising while gold plateaus.
2. **The +0.023 feature gap (B_fullwp vs K_truebase) disappears under gold:** −0.002 ± 0.003 (gN), −0.008 ± 0.003 (NODRIFT, z = −2.9), −0.008 (QM2026, z = −3.1).
3. **The tournament ordering survives, with margins about a third smaller.** Constrained MCTS still wins 20/20 under the post-snapshot judges: 0.624 vs 0.670 consensus (0.598 on NODRIFT). MCTS stays second, and the rest of the order holds.
4. **The mechanism is label leakage through the aggregate statistics.** In a clean within-snapshot split, a WP whose statistics include each row's own game has a held-out calibration slope of 0.61 (975K training games) or 0.17 (122K). The same model on out-of-fold statistics is calibrated (slope 1.0–1.1) and more accurate (57.0% vs 56.1%).
   - Search widens the leaky proxy's gap to gold at every pressure step. The out-of-fold proxy tracks gold almost one-for-one.
   - In the paper's own self-play pipeline on half A, 200 → 800 training sims:
     - leaky proxy: self-score +0.09, every gold reference −0.02 to −0.05, degenerate teams up to 53%;
     - leak-free proxy: gold +0.01 to +0.03.
5. **The paper's own proxy is only mildly leaky.** Its statistics come from a 3.87M-game aggregate, not its own 1.95M training replays.
   - On unseen games its calibration slope is 0.87–0.89 and its accuracy 56.8–56.9% (vs the 58.07% reported).
   - At paper scale it drafts as well under gold as a leak-free retrain. Its self-reported numbers are inflated more than its drafts are hurt.

**Recommended operating point: F_400sim or E_1M, with argmax root selection.**
- F_400sim is best or tied-best on every no-drift reference, at half of J_800sim's training compute.
- J_800sim is not significantly worse; the extra compute just buys nothing.
- Argmax root on J_800sim_s9 adds +0.011 ± 0.005 (gN) and +0.012 ± 0.004 (NODRIFT).
- The bigger fix is upstream: train the WP on out-of-fold statistics. Then deeper search keeps paying off under gold.

## 1. Gold references and why they are independent

**Proxy:** `wp_enriched_256` with the frozen 2026-05-19 statistics, symmetrized. This is what Table VI reports as "Avg WP".

**References for the saved paper-1 drafts.** None of these shares a game with the proxy's weights or statistics.

| ref | construction | games |
|---|---|---|
| gN | 3 enriched judges with out-of-fold statistics, trained only on post-snapshot no-drift games | 291,837 |
| gN_naive | hero-identity judge on the same games | 291,837 |
| BF | realized index on backfill games (snapshot-era builds, uploaded after the snapshot and the frozen-stats date; no balance drift) | 153,739 |
| T97 | realized index on 2.55.16.97039 games after the snapshot | 138,098 |
| NODRIFT | realized index on BF + T97 | 291,837 |
| T17 | realized index on 2.55.17.{97605, 97650, 97771, 98025} (drifted meta) | 158,608 |
| QM2026 | existing Quick Match judge (different game mode) | 116,589 |

**Realized-outcome index (`gold.py`).** The W13 index from the drift paper plus a role-composition term, cross-fitted:
- statistics from one hash fold (hero WR, synergy and counter residuals, role-comp WR shrunk with a 200-game prior);
- a 4-term logistic fit on the other fold's real results;
- both directions averaged.

Held-out accuracy is 54–56%, so it is a weak but unbiased judge.

**Within-snapshot split (`split.py`).** The snapshot is hash-split into A (proxy side) and B (gold side).
- Proxies, their statistics, early stopping and any search against them use only A.
- Gold on B: gB (3 out-of-fold judges), gB_naive, and RB (the realized index). RN (post-snapshot index) and QM2026 are also applied.

## 2. Paper-1 agents under gold (15 seeds × 200 drafts per config)

Seed SEs are 0.001–0.003.

| config | proxy | gN | NODRIFT | T17 | QM2026 | degen % |
|---|---|---|---|---|---|---|
| J_800sim | .767 | .667 | .624 | .632 | .643 | 7.5 |
| I_600sim | .764 | .666 | .623 | .634 | .643 | 7.6 |
| F_400sim | .753 | .667 | .626 | .638 | .641 | 5.9 |
| E_1M | .752 | .667 | .623 | .637 | .643 | 5.4 |
| B_fullwp | .726 | .657 | .616 | .625 | .633 | 11.0 |
| K_truebase | .703 | .658 | .624 | .630 | .641 | 20.1 |

Contrasts (difference, with z in parentheses):

| contrast | proxy | gN | NODRIFT | T17 | QM2026 |
|---|---|---|---|---|---|
| F400 − B200 | +.027 (7.4) | +.010 (3.1) | +.010 (3.7) | +.013 (4.5) | +.008 (2.3) |
| J800 − F400 | +.014 (7.2) | .000 (0.0) | −.002 (−1.1) | −.006 (−3.0) | +.002 (0.7) |
| E_1M − J800 | −.015 (−5.9) | +.001 | −.001 | +.005 | .000 |
| B − K_truebase | +.023 (5.0) | −.002 (−0.5) | −.008 (−2.9) | −.005 (−1.5) | −.008 (−3.1) |

Other findings on the paper-1 agents:
- **Inference-time sims (100–800) on fixed checkpoints** change nothing on any reference.
- **Root exploration** (Dirichlet noise and T = 1.5) hurts.
- **Direct real-outcome check (`stage_c`).** Real unseen games are weighted by how often each config uses the same 3-hero cores.
  - The real win rate of the favored cores goes 0.571 (B) → 0.575 (F) → 0.574 (J).
  - The proxy's prediction for those same teams goes 0.579 → 0.589 → 0.590.
  - So the proxy's excess error on the teams the agents favor grows from +0.8pp to +1.6pp (± 0.6). This shows the plateau with no model involved.

## 3. Mechanism: the split experiment

Held-out quality on half B. "Own-split val acc" is what a leaky pipeline would report about itself.

| proxy | games | own-split val acc | held-out acc | calib slope |
|---|---|---|---|---|
| leak, 1/8 of A | 122K | 68.4% | 53.2% | 0.17 |
| leak, all of A | 975K | 59.4% | 56.1% | 0.61 |
| out-of-fold, all of A | 975K | 56.3% | 57.0% | 1.10 |
| hero identity only | 975K | 56.5% | 57.1% | 1.05 |
| paper proxy (post-snapshot games) | — | 58.07% | 56.8–56.9% | 0.87–0.89 |

Width, depth, dropout, weight decay and epochs each move held-out accuracy by less than 0.2pp. The one exception is 40 epochs with no regularization (slope 0.40). Leakage dominates every capacity knob.

**Search sweep.** Behavioral-cloning prior distilled from GD (no outcome information), argmax root, 400 paired drafts per point, SE about 0.005.

| proxy | sims | proxy | gB | RB | RN | QM2026 |
|---|---|---|---|---|---|---|
| leak | 1024 | .780 | .631 | .576 | .541 | .550 |
| leak | 4096 | .832 | .651 | .598 | .564 | .566 |
| leak | 16384 | .848 | .666 | .611 | .579 | .591 |
| out-of-fold | 1024 | .673 | .661 | .589 | .550 | .589 |
| out-of-fold | 4096 | .721 | .708 | .633 | .593 | .613 |
| out-of-fold | 16384 | .734 | .713 | .640 | .605 | .623 |
| leak, 1/8 data | 4096 | .910 | .613 | .572 | .556 | .558 |
| leak, 1/8 data | 16384 | .932 | .604 | .567 | .555 | .565 |
| out-of-fold, 1/8 data | 16384 | .702 | .650 | .642 | .617 | .620 |

- **Positive control:** the small leaky proxy's gold peaks at 4096 sims and then falls while its own score climbs to 0.93. The method detects overoptimization when it is present.
- **Leakage costs more than data:** a leak-free proxy trained on 1/8 of the data beats a leaky proxy with 8× the data on both realized indices.

**Self-play replication (the paper's pipeline on half A).** Unchanged training worker, value pretraining restricted to A, 2 seeds per cell; benchmarked at 200 inference sims with argmax root.

| proxy | training | proxy | gB | RB | RN | QM2026 | degen |
|---|---|---|---|---|---|---|---|
| leak | 200 sims, 300K episodes | .757 | .695 | .652 | .617 | .640 | 27% |
| leak | 800 sims, ~95K episodes (stopped early) | .827 | .630 | .629 | .588 | .603 | 53% |
| out-of-fold | 200 sims, 300K episodes | .683 | .709 | .655 | .633 | .647 | 3.5% |
| out-of-fold | 800 sims, ~87K episodes (stopped early) | .709 | .722 | .663 | .635 | .655 | 5% |

At matched ~100K episodes, 200 → 800 training sims moves the leaky proxy's gold down 0.02–0.05 and the out-of-fold proxy's gold up 0.01–0.03.

**Paper scale (`fullscale.json`, 4096 sims, post-snapshot gold):**

| leaf value | self | gN | RN | QM2026 |
|---|---|---|---|---|
| paper proxy | .767 | .631 | .590 | .608 |
| out-of-fold retrain | .738 | .633 | .590 | .609 |
| own-stats leaky retrain | .791 | .619 | .580 | .592 |

## 4. Other signals and candidate fixes

- **Ensemble disagreement is not a Goodhart detector here.** The per-draft std of the 4-seed ensemble stays between 0.0053 and 0.0067 at every pressure level. Seeds trained on the same leaky statistics share the same errors.
- **Pessimistic (LCB) search does nothing.** It was implemented exactly in a kernel copy and verified against Python to 1e-7. At 1024 sims, λ from 0 to 8 moves gB by at most ±0.007 and RN by at most ±0.003.
- **A flat prior is worse than the behavioral prior.** At 4096 sims it yields 43% degenerate teams and lower gold. The behavioral prior acts as a regularizer.
- **Kernel bug.** The CUDA kernel has no guard on its 4096-node tree.
  - Paper runs are unaffected: at 800 sims with a trained prior the tree peaks around 590 nodes, and output is bit-identical to a guarded copy.
  - A flat prior overflows by 800 sims.
  - The guarded copy in `cuda_ofit/` hits the guard at 16384 sims, so treat those points with caution.

## 5. What each paper should say

**Paper 1:**
- Report sim scaling alongside an independent reference. Suggested wording: "On 291,837 post-snapshot games, 200 → 800 sims gains +0.010 vs +0.041 under the training WP, complete by 400 sims."
- Present F_400sim or E_1M as the operating point, or say that J_800sim's extra compute buys no measurable real quality.
- Drop or requalify the +0.023 feature-gap claim.
- State separately what features do buy: safety (degenerate rate 20% for K_truebase vs 8–11%), which the realized references barely penalize.
- Keep the tournament claim, with the gold standings as a robustness row.
- Restate Table I accuracy as measured on unseen games (56.8–56.9%, slope 0.87–0.89), with one sentence on target leakage through aggregate statistics.

**Paper 2:** the split experiment is a clean, drift-free confirmation of the leakage finding. Leaky vs out-of-fold statistics at matched data change calibration (0.61 → 1.1) and search outcomes on every gold reference, which supports causal statistics as the default.

## 6. Caveats

- The gold references are models or weak indices. The realized index can't judge compositions nobody plays and barely penalizes degenerate teams. The evidence is agreement across seven differently built references, not any one of them.
- gB shares the proxy's model class. The references that don't (RB, RN, QM2026) agree with it.
- Post-snapshot games differ in population (late uploads). BF and T97 remove balance drift but not population shift.
- The split overstates the leak relative to the paper: half-A proxies have smaller statistics (slope 0.61 vs 0.87–0.89). The paper-scale table calibrates for this.
- The self-play replication is thin: 2 seeds per cell, and the 800-sim runs stopped at about 30% of training.
- Search below about 64 sims is dominated by a kernel artifact: unvisited children get Q = 0, and ties go to the lowest hero index.
- The "12/12, consensus 0.703" in the brief matches neither tournament. The paper reports constrained MCTS at 0.670 (20/20); the September refresh has MCTS at 0.690 (16/16). The July paper artifacts were used here.

## 7. Next experiments

1. Finish the 800-sim self-play runs to 300K episodes with 3+ seeds per cell (about 12 GPU-hours).
2. Retrain J and F with the full-scale out-of-fold WP (`pS8_oof`) and rescore on post-snapshot gold.
3. Use the expert study as the only available check on degenerate compositions.
4. Make held-out calibration slope a standard diagnostic for every value function. Here it predicted overoptimization better than ensemble uncertainty did.

## Files

- **Code:** `data.py`, `gold.py`, `stage_b_saved.py`, `stage_b2_judges.py`, `analyze_paper.py`, `stage_c_calib.py`, `leak_probe_paper.py`, `split.py`, `bc_prior.py`, `search.py`, `cuda_ofit/`, `stage_e.py`, `analyze_split.py`, `train_mcts_split.py`, `stage_f_selfplay.py`, `stage_g_fullscale.py`
- **Results** (`results/`): `paper_gold.json`, `stage_b.json`, `stage_b2.json`, `stage_c.json`, `split_heldout.json`, `split_search/`, `split_summary.json`, `selfplay/`, `fullscale.json`, `leak_probe_paper.json`
- **Figures** (`figs/`): `paper_pressure_curves.png`, `split_pressure_curves.png`, `split_proxy_vs_gold.png`, `split_lcb.png`
