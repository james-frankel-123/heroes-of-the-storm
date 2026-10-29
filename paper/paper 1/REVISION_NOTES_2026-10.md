# Paper 1: what changed since the last coauthor version (2026-10-29)

The last coauthor version was the site-tier rebuild of 2026-10-12/13 (commits 5b13eaf,
4163770, c0b4ef4). Main paper 10 pages (was 9), supplement 7 (was 6); no overfull boxes.
`draft_revision_markup.pdf` shows every change against the submitted version.

## New: expert study (main Section VIII, Table VII; supplement last section)

- **Design.** Registered blinded study (https://osf.io/fspuc).
  - The ten tournament strategies without K, retrained with the same recipe on ranked games before 2026-09-01.
  - 280 machine pairs, 570 real games with known winners (190 per tier), 40 calibration items, 11 attention checks.
  - 14 paid raters with fixed 240-item sets.
- **Collection.** Closed 2026-10-26 with 11 of 14 complete; the three remaining raters did not respond to a reminder (one sentence in the paper).
  - Nine raters pass the gates; two fail the screener.
  - The three partial raters are excluded (gates as written). Holding them only to the items they reached changes nothing.
- **Results.**
  - Agreement with the models on generated pairs: 0.50 (0.43-0.57), so the primary test fails. It is the same under every sensitivity.
  - Where anchored strategies met searching ones, experts chose the anchored draft 71% of the time (agreement 0.33, 0.21-0.48).
  - Expert ranking vs tournament order: rho = -0.19.
  - Real winners: experts 60% at low tier, 49% at mid, 53% at high. The model label scores 60/54/56% on the same games (descriptive, post hoc; about ±7pp per tier).
  - Sliders flat; the low-rank-experience group no more accurate; agreement does not fall with unfamiliarity.
- **Framing (Max).** We tested whether expert judgment can check the models. It cannot: the experts are at chance on real games above the lowest tier, so their disagreement neither confirms nor refutes the ordering, and the claims rest on the outcome-based checks.
  - The abstract, introduction, conclusion and limitations say this in one sentence each.
  - The old line "a blinded expert study ... is in progress" is gone.
- **Supplement.** Design, gates, the nine analysis choices the plan left open (approved 2026-10-26), every registered test with its sensitivities, the post hoc model-accuracy table, and the patch-2.57 exclusions.

## Corrected numbers (checked against the result files)

All 32 MCTS runs behind the paper were trained on the fixed kernel (chance nodes at opponent turns), with the corrected composition fallback; benchmarks ran on the same builds.

| Where | Was | Now |
|---|---|---|
| Abstract, introduction | 1.95M drafts | 1.94M (unranked games excluded) |
| Composition correction | 1.95M snapshot games | 1.94M |
| Gourdeau estimator, five tanks | 45.5% | 44.9% |
| Table I caption, leak-free slopes on Test | 1.08-1.22 | 1.03-1.22 (Gourdeau estimator included) |
| CQL setup | "62M transitions with team swaps" | "16 transitions per replay, doubled with team swaps" (count not traced) |
| Leak control, consensus at 800 sims | -0.016 | -0.015 |
| E - B, consensus | -0.006 | -0.005 |
| Pairwise feature groups | lose 0.12-0.21pp | 0.12-0.22pp (main and supplement) |
| Supplement, MCTS runs | 210 (195 + 15) | 227 (195 + 32) |
| Supplement, forward trios | z = 1.0 | z = 0.9 |
| Supplement, self-including spreads | about 28% and 20% | 30% and 20% |
| Supplement, masked greedy head to head | ±0.004 | ±0.003 (as in main) |

## Follow-up (2026-10-29, after the coordinator's review)

1. **Kernel throughput re-measured** on one RTX 5090. The original RTX PRO 6000 cards were busy. The protocol is the submission's (128 concurrent episodes per launch, one warm-up, 768 timed episodes, 200 simulations), with the paper's F agent and leak-free leaf WP (`training/paper1_revision/throughput_fixed_kernel.py`, `site/results/throughput_fixed_kernel.json`).
   - The revision's search runs 63 ep/s at 200 sims and 33 at 400.
   - The submission's search on the same GPU runs 216 and 112, so the deeper search costs a factor of 3.4.
   - The 2.2× controlled comparison against the host pipeline covers only the submission's search, the only one the host pipeline implements. It is not repeated: on the 5090 machine's 16 cores the host pipeline is CPU-bound (9 ep/s).
   - Contribution 4 and the supplement's kernel section now say all this. The submission's history table is labeled as such. The stale "GD cache is dead code" sentence now says the revision's search uses the cache.
2. **57.6 vs 57.8.** These are different quantities: 57.6% is the main text's three-seed mean, 57.8% the selected seed in the supplement's reference table. The supplement now labels it.
3. **Untraced numbers, now traced or softened.** Traced by a cheap recount from the saved split and snapshot:
   - 51,090 no-healer teams in the training split, win 42.4%;
   - 92.6% of human degenerate teams have one defect, 68.2% no healer;
   - the 0.677 / 0.644 / 0.650 / 55% greedy-win-rate drafter (`production_refresh/validate-seedsel-backtest-0930/seed_selection_v1judge.json`);
   - the NGS descriptive statistics (`ngs2026/meta_analysis_p1site.json`).

   Corrected:
   - degenerate multisets in the post-snapshot games: median 4-10 team-games by tier (was 3-9);
   - MCTS degenerate teams with one defect: 93-100% (was 94-100%).

   Dropped:
   - the untraceable IQL advantage weights (0.995-0.999);
   - r = -0.87 (the qualitative statement stays).

   Inherited by design: the submitted sweep column and the kernel history table.
4. **Split experiment kernel:** confirmed. All 8 self-play runs report search_mode chance in `kernel_info.json` and the training log (`overfit2026/site/results/split_selfplay_kernel_info.json`).
5. **Results committed.** The site-tier result files behind the revision are committed: `training/paper1_revision/site/results`, `training/overfit2026/site/results`, `training/rerun2026/ns/p1site/results`.
   - The scan found no player identifiers (battletags, ids or names).
   - The 20 per-draft dumps in `paper1_revision/site/results/greedy/rich_*.json` (172 MB) stay out of git; their sha256 are in `greedy/LARGE_FILES.sha256`.
