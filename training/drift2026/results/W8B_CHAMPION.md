# W8b — champion-config maintained agent + seed power

Maintained side = the RECOMMENDED policy (Q7 champion decayed90k100, 57.22 future acc; VF q7_decayed90k100_s777, deploy stats decayed90k100 @ last completed build), MCTS x15 seeds. Stale-4mo (d2b_allhist) extended 5 -> 15 seeds (w4_ + w8_ runs pooled). All inference = crossed-RE over (seed_maintained, seed_stale).

| head-to-head | n | consensus WP (z vs .5) | binary share | Δ future hero WR pp | Δ synergy | Δ degen pp | degen % (champ/stale) |
|---|---|---|---|---|---|---|---|
| champion vs stale-4mo (15x15, 10/cell, staggered) | 4500 | 0.4986 ± 0.0067 (z -0.20) | 0.484 ± 0.030 | +0.92 ± 0.09 (z 9.7) | +0.75 ± 0.14 (z 5.3) | -8.5 ± 5.3 (z -1.6) | 22.7 / 31.1 |
| champion vs stale-1yr (5x5, 40/cell) | 2000 | 0.5070 ± 0.0086 (z 0.81) | 0.531 ± 0.042 | +0.87 ± 0.12 (z 7.0) | +0.64 ± 0.27 (z 2.3) | -6.6 ± 9.9 (z -0.7) | 23.5 / 30.1 |
| champion vs stale-2yr (5x5, 40/cell) | 2000 | 0.5056 ± 0.0140 (z 0.40) | 0.542 ± 0.063 | +0.97 ± 0.09 (z 10.3) | -0.15 ± 0.29 (z -0.5) | -4.5 ± 12.4 (z -0.4) | 27.7 / 32.2 |

## Vintage matrix (maintained WP by judge era)

| judge | champion vs stale-4mo (15x15, 10/cell, staggered) | champion vs stale-1yr (5x5, 40/cell) | champion vs stale-2yr (5x5, 40/cell) |
|---|---|---|---|
| 2022-07 | 0.4501 | 0.4445 | 0.4367 |
| 2022Q1-ranked | 0.4624 | 0.4611 | 0.4680 |
| 2023-07 | 0.4903 | 0.4866 | 0.4857 |
| 2024-07 | 0.5062 | 0.5013 | 0.5036 |
| 2025-07 | 0.4883 | 0.5136 | 0.5059 |
| 2026-build | 0.5017 | 0.5376 | 0.5170 |
| QM-2021 | 0.4098 | 0.4287 | 0.4423 |

## Degenerate-rate trend across the gradient (stale side, seed-level OLS)

| arm | staleness (yr) | stale-side degen % | n stale seeds |
|---|---|---|---|
| champion vs stale-4mo (15x15, 10/cell, staggered) | 0.33 | 31.1 | 15 |
| champion vs stale-1yr (5x5, 40/cell) | 1.00 | 30.1 | 5 |
| champion vs stale-2yr (5x5, 40/cell) | 2.00 | 32.2 | 5 |

OLS slope: +0.51 pp/year (SE 3.20, z 0.16, 25 seed-level points).

## The z=1.8 degen question

W6 (cumprev vs 4mo, 5x5): degen delta z = -1.82 (n.s.). W8b (champion vs 4mo, 15x15): degen delta z = -1.61 — still not individually significant at 15x15.

## Interpretation

1. **Consensus vs judge-free divergence.** The champion agent is
   statistically EVEN with every stale arm on the 4-evaluator consensus
   (0.4986 / 0.5070 / 0.5056, all |z| < 1), while the judge-free
   future-truth metrics favor it decisively (hero-WR delta z 7-10, synergy
   z 5.3 at 15x15). The consensus evaluators penalize the champion's
   elevated own-side degen rate (22.7-27.7% vs the cumprev-maintained
   agent's 14.5% in W6); the future-truth metrics reward its pick quality.
   The two scoreboards disagree about the champion, not about staleness.
2. **The recommended VF config does not dominate as an agent.** decayed90k100
   is the best VF (57.22 future acc) but its MCTS agent trades away counter
   quality (counter delta negative, z -3.2 at 15x15) and comp discipline
   (degen up ~8pp vs cumprev-maintained) relative to what W6's cumprev agent
   showed. Recency-sharpened deploy stats appear to make thin-cell
   degenerate comps look better to the in-kernel value function. "Best
   future accuracy" and "best downstream agent" are different selections —
   an honest caveat for the recommendation section.
3. **The z=1.8 question is NOT resolved; it dissolved.** W6's degen delta
   (-20.8pp, z -1.82) was cumprev-vs-4mo. With the champion at 15x15 the
   delta shrinks to -8.5pp (z -1.61) because the champion itself degenerates
   more; power went up (SE 5.3pp vs 11.4pp) but the effect went down. The
   robust degen claim is about the CUMPREV maintained agent, not the
   champion config.
4. **No degen trend across staleness** (stale side 31.1 / 30.1 / 32.2%,
   slope +0.5 pp/yr, z 0.16): degeneration is a property of the d2b/stale
   training configuration, not of how stale it is.
5. **Seed power worked.** 15x15 cut the crossed-RE consensus SE from 0.0112
   (W6 5x5) to 0.0067 and the hero-WR-delta SE from 0.20 to 0.09; the seed
   components, not draft sampling, were the binding variance.

## Notes / provenance

- 15x15 used `--stagger-configs` (cell-offset map x tier indexing) so
  10 drafts/cell still covers all 42 map-tier combos across the grid;
  configs remain identical for both orderings and both sides (paired).
- The stale-4mo pool = w4_d2b_allhist_s0..4 + w8_d2b_allhist_s5..14 (same
  VF `w4_vf_d2b_allhist.pt` and env recipe; multi-prefix loader "w4,w8" in
  w6_head2head.py).
- Champion runs: `mcts_runs/w8_champion_s0..14`, VF
  `models/w8_vf_champion.pt` (= q7_decayed90k100_s777), deploy LUT stats
  decayed90k100 @ 2.55.16.96881 via the new MCTS_STATS_KIND override in
  train_mcts_worker.py (verified: identical LUT byte-layout, mean |delta|
  1.20 vs cumulative; end-to-end smoke run).
- A first run of the two 5x5 head-to-heads was discarded: it read
  w8_champion_s4's PERIODIC checkpoint ~8 min into training (draft_policy.pt
  is saved during training, so file existence is not run completion). All
  tables above use post-completion policies, gated on the pool driver's
  DONE log lines. The same pitfall previously inflated W7's stale-2yr
  head-to-head (regenerated 2026-07-13, consensus 0.5500 -> 0.5164).
- Inference files: `results/w8/inference_w8_champ{4mo,1yr,2yr}.json`
  (crossed-RE machinery `drift2026/w8_inference.py`, verified to reproduce
  results/w6_inference.json exactly).
