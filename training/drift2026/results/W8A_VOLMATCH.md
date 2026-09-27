# W8a — volume-matched staleness-gradient control

The corpus grows over time, so the stale-2yr arm trained on fewer rows than the maintained arm; the W7 "+1.16pp at 2yr" could conflate staleness with training-set size. Control: the maintained config (cumulative_prev features, regime=all) trained on a seeded random subsample matched to stale-2yr's n_train, then the full W4 MCTS recipe (5 seeds) and the W6 head-to-head protocol vs the same stale-2yr agent.

## Per-cutoff training volumes (rows; every VF sees 2x-augmented rows of its era's replays)

| arm | train rows |
|---|---:|
| maintained (cutoff 2.55.14.95918, 2026-02) | 3,538,596 |
| stale-1yr (cutoff 2.55.9.93613, 2025-02) | 2,270,654 |
| stale-2yr (cutoff 2.55.4.91418, 2024-02) | 1,594,630 |
| volume-matched maintained (subsample) | 1,594,630 |

## Value-function future accuracy (3 seeds)

- maintained, full volume: 57.08 ± 0.04
- maintained config, volume-matched to 2yr: 56.73 ± 0.04
- stale-2yr: 55.27 ± 0.01

## Head-to-head vs stale-2yr (crossed-RE over 5x5 seed pairings)

| maintained side | n | consensus WP (z vs .5) | binary share | Δ future hero WR pp | Δ synergy | Δ degen pp | degen % (m/f) |
|---|---|---|---|---|---|---|---|
| volume-matched | 2000 | 0.5129 ± 0.0182 (z 0.71) | 0.565 ± 0.088 | +0.82 ± 0.23 (z 3.6) | -0.63 ± 0.40 (z -1.6) | -14.4 ± 13.5 (z -1.1) | 22.5 / 37.0 |
| full-volume (W7 ref) | 2000 | 0.5164 ± 0.0133 (z 1.24) | 0.581 ± 0.059 | +0.96 ± 0.14 (z 6.7) | +0.08 ± 0.39 (z 0.2) | -7.9 ± 7.4 (z -1.1) | 19.7 / 27.6 |

## Vintage matrix

| judge | volmatch vs 2yr | full vs 2yr (ref) |
|---|---|---|
| 2022-07 | 0.4418 | 0.4409 |
| 2022Q1-ranked | 0.4691 | 0.4650 |
| 2023-07 | 0.4884 | 0.4875 |
| 2024-07 | 0.5139 | 0.5051 |
| 2025-07 | 0.5177 | 0.5132 |
| 2026-build | 0.5134 | 0.5175 |
| QM-2021 | 0.4521 | 0.4392 |

## Verdict

**The 2yr point survives volume matching: this is a staleness story, not a
data-scale story.** The volume-matched maintained agent retains 86% of the
full-volume agent's judge-free edge over stale-2yr (+0.82 ± 0.23 vs +0.96 ±
0.14 pp future-hero-WR; z 3.6 vs 6.7), and its consensus WP is statistically
indistinguishable from the full-volume agent's (0.5129 ± 0.018 vs 0.5164 ±
0.013). Of the 1.81 pp VF-accuracy gap between maintained and stale-2yr,
only 0.35 pp (~19%) is attributable to training volume (57.08 → 56.73 when
subsampled); the remaining ~1.45 pp is era. Note the volume-matched agent's
higher own-side degen rate (22.5% vs 19.7%) — less data degrades comp
discipline slightly — yet its stale opponent still degenerates far more
(37.0%).

## Notes / provenance

- The full-volume reference row uses the REGENERATED
  `w6_head2head_stale2yr.json` (main session, 2026-07-13). The original W7
  run of that file was contaminated by the periodic-checkpoint gating
  pitfall (stale2yr seeds s3/s4 were read mid-training), which had inflated
  the maintained edge to 0.5500 consensus / +1.16 pp hero-WR. The clean
  numbers (0.5164 / +0.96) are the citable ones; W7_STALENESS_GRADIENT.md
  should be read against this correction. The stale-1yr point's original
  h2h read its s4 seed at ~85% training (near-final; not regenerated).
- Volume-matched VFs: `w8_volmatch_s{42,123,777}` (results/w8/), seeded
  row subsample via `train_drift_wp.py --max-train-rows 1594630`; best seed
  42 exported to `models/w8_vf_volmatch.pt`; MCTS runs
  `mcts_runs/w8_volmatch_s0..4` (W4 recipe, cumulative stats @ last
  completed build). Inference: `results/w8/inference_w8_volmatch.json`.
