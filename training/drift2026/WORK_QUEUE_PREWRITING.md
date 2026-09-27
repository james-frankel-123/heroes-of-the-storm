# Drift paper — pre-writing work queue (2026-07-13, ToG target)

Adopted from external review of COAUTHOR_SUMMARY.md. Every item lands in
`results/` with an .md + .json pair, same conventions as existing waves.
GPU assignments noted to avoid contention. Item 6 (terminology pass) from the
review is deferred to writing time by design.

## Q1 — W6 judge-free rescoring [me, GPU3, TODAY]
Score the saved W6 terminal drafts (results/w6_head2head.json `records`)
against FUTURE-PERIOD ground-truth stats (phase_w4's `future_truth_stats()`
protocol): counter and synergy deltas per side + existing degen/healer rates.
These become the headline damage numbers (no learned judge → no vintage
confound). Output: results/W6_JUDGEFREE.md.

## Q2 — Vintage judge matrix [agent, GPU1]
Train NAIVE-feature WP judges (hero multi-hot ×2 + map + tier one-hots; the
qm2026/train_qm_wp.py MLP recipe, NOT the enriched pipeline — naive features
need no stats and are era-portable) on ranked replay_draft_data windows at
~4 vintages: train-through cutoffs ending near 2022-07, 2023-07, 2024-07,
2025-07, and the W2 cutoff build 2.55.14.95918 (2026). ~identical volume per
judge (cap at the smallest window's game count for comparability, sampled
recent-first below each cutoff). Then score the W6 records (both agents)
with every judge + the QM-2021 judge (qm2026/results/qm_wp_v0.pt) →
**vintage matrix**: judge era × (maintained WP). Expect monotone: older
judges favor the maintained agent less / reverse. Output:
results/W6_VINTAGE_MATRIX.{json,md} + the trained judges under
models/vintage_judges/.

## Q3 — W6 crossed-random-effects inference [me or agent, CPU]
Per-draft maintained-side outcomes with crossed random effects for
(seed_maintained, seed_frozen): lme4 glmer on the binary win indicator AND
lmer on the continuous consensus score; plus the conservative 5-disjoint-
pairings estimate; report all SEs with units (continuous-score SE vs
binomial). Output: results/W6_INFERENCE.md.

## Q4 — Detector-triggered refresh arm in the W2c grid [agent, GPU0]
Extend the W2c deployment simulation with arms where stats refresh fires
when the FDR-surviving outcome-shift detector (results/w3_changepoints.json)
fires — honestly lagged: the refresh takes effect ~9.4K games into the build
(the measured median detection latency), not at the boundary. Compare rows:
never / every-build / detector-triggered / accuracy-triggered (existing).
If detector-triggered lands within ~0.1pp of every-build, findings 4+7 fuse
into a closed-loop system. Include the latency-symmetry sentence data:
detection ~9.4K games vs local-stats trustworthy at 6.5-12K. Output:
results/W2C_DETECTOR_ARM.md + updated w2c table.

## Q5 — Partial-refresh ablation [agent, GPU2]
Inference-time stats swap on the d2c_cumprev checkpoints (no retraining):
refresh ONLY per-hero-WR aggregates vs ONLY pairwise/comp aggregates vs ALL
(existing) vs NONE (cutoff stats). Future-window accuracy + the 28-test
sanity suite per cell, 3 checkpoints (seeds) each. Decay curves predict
hero-WR refresh carries most of the gain — confirming links findings 2+3
mechanistically. Output: results/PARTIAL_REFRESH.md.

## Q6 — Staleness gradient [me, GPU3 tonight]
VFs trained at two additional cutoffs (~1yr stale ≈ builds through 2025-07,
~2yr ≈ 2024-07; d2b_allhist recipe, 3 seeds each), then MCTS per arm
(phase_w4 recipe, 5 seeds, 200 sims, 300K episodes), then W6-protocol
head-to-head vs the maintained agent, judged by (a) future-truth stats
(judge-free) and (b) the Q2 vintage matrix. Deliverable: dose-response
curve "pp lost per year of neglect" — monotonicity is the artifact-killer.
Output: results/W7_STALENESS_GRADIENT.md.

## Q7 — Decayed-aggregates arm + shrinkage 2×2 [me, GPU3 after Q6 trains]
Stats-side recency decay: exponentially decayed feature AGGREGATES
(half-life 90d / 365d) refreshed per build, vs cumulative (existing champion
57.08). 2×2 with count-shrinkage strength (the W2a k=100 prior machinery):
{decayed, cumulative} × {k=100, unshrunk}. 3 seeds/cell, D2 protocol.
Output: results/DECAYED_AGGREGATES.md.

## Q8 — Warm-start finetune row [me, GPU3 queue]
One cadence: at each W2c retrain point (every-3-builds schedule), warm-start
from the previous checkpoint, 5 epochs on the trailing 6-build window, vs
the existing cold full-retrain row. One table row; converts the most
predictable reviewer objection. Output: row in W2C summary + note.

## Bookkeeping (writing time, not now)
Arm-name disambiguation table ("frozen" ×3 meanings), window-labeled
accuracies (56.08 deployment-weighted vs 57.08 future-window), games-vs-
samples units. Venue: ToG (decided 2026-07-13); TMLR fallback if the
gradient is strong; benchmark packaging deferred (needs Zemill licensing
conversation).

# W8 — second-review hardening (2026-07-13, priority order)

## W8a — volume-matched gradient control [HIGHEST PRIORITY]
The corpus grows over time → stale-2yr trained on less data → "+1.16pp at
2yr" conflates staleness with training-set size. Control: maintained-config
VF (cumprev features, regime=all) trained on a random subsample matched to
stale-2yr's n_train (read it from results/w7/w7_stale_2.55.4.91418_s42.json),
3 seeds; MCTS 5 seeds (same W4/W7 recipe, MCTS_STATS_BUILD = last completed
build); head-to-head vs w7_stale2yr; judge-free scoring. Also report
per-cutoff training volumes for all gradient points. Verdict: does the
volume-matched maintained agent hold its edge?

## W8b — champion-config maintained agent + seed power
The recommended policy is decayed90k100 (57.22) but W6/W7's maintained agent
is cumprev. Re-run the maintained side with the champion config: export best
Q7 decayed90k100 VF → MCTS **15 seeds** (worker may need a stats-kind
override so MCTS_STATS_BUILD loads patch_stats/decayed90k100/ — check
train_mcts_worker.py stats loading, extend minimally). Also extend
w4_d2b_allhist (stale-4mo) 5→15 seeds. Head-to-heads: champion vs stale-4mo
at 15×15 (reduce drafts/cell to ~10), champion vs stale-1yr/2yr at 5×5
(40/cell). Judge-free + vintage rescoring + crossed-RE inference (extend
n-seeds handling). Degenerate-rate trend test across the gradient pools.
Kills the z=1.8 problem and removes the "you didn't test your own
recommendation" objection in one pass.

## W8c — nested-validation champion selection [analysis-mostly]
The 2×2 champion was selected on the test window. Repair: inner cutoff a few
builds before the true cutoff (pseudo-future = the pre-cutoff builds between
them, comparable volume); retrain the 4 cells × 3 seeds with existing
feature caches (train mask ≤ inner cutoff, select on pseudo-future); report
whether the selection transfers to the true test window (it should:
seed-clean separation). Report the full 2×2 both windows.

## W8d — ranked early-2022 vintage judge [optional]
The QM-2021 matrix point is mode+era double-confounded. If the ranked corpus
window 2021-12..2022-03 holds ≥150K games, train one more vintage judge on
it and add the row; otherwise skip (the acknowledgment sentence is already
in the summary).

## W9 — continuous-MMR input ablation [after W8 drains; cheap]
The 3-tier grouping (low/mid/high) was a design choice inherited from paper
1; replay_draft_data carries continuous avg_mmr. Cells (D2 protocol, 3
seeds each, champion decayed90k100 features): (a) tier one-hot (existing
champion, reference); (b) + normalized avg_mmr appended; (c) continuous
avg_mmr replacing the tier one-hot. Also probe: does predicted WP for
known skill-dependent heroes (Murky, Abathur, The Lost Vikings) vary
correctly along the MMR axis in (b)/(c)? Note the caveat that avg_mmr is
Heroes-Profile-computed near game time (cleaner than the per-player
as-of-parse MMRs). Coverage check first: fraction of rows with non-null
avg_mmr per tier/era. Scope note: this is a robustness/reviewer-preempt
row for paper 2 (stats aggregates STAY per-tier — kernel-weighted
continuous-MMR stats would be real machinery, not this cell); if (b) shows
a real gain it also feeds personalization, which conditions on MMR anyway.
