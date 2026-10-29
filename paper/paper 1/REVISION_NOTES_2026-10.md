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

## Open items for the authors

1. **Kernel throughput (contribution 4 and the supplement kernel section).** The figures (140-161 episodes/s, 2.2× the host pipeline) were measured on the submission's kernel. The fixed kernel searches the opponent's turns and runs about 3-4× slower at equal simulations. They need re-measuring on the current kernel, or must be stated as the old kernel's numbers.
2. **Two in-corpus accuracies.** The naive model is 57.6% in the main text (three-seed mean) and 57.8% in the supplement (the selected seed). Pick one.
3. **Not traced to a saved result.** These are from the submission or were computed ad hoc:
   - 51,090 no-healer games;
   - the median of 3-9 games per degenerate multiset;
   - 93% one-defect;
   - r = -0.87;
   - the IQL advantage weights;
   - the 0.677 / 55% greedy-win-rate drafter;
   - the NGS descriptive statistics;
   - the submitted sweep column.
4. **Unverified kernel for the split experiment.** Its self-play runs were trained on a remote host, and their run directories are not on the main box, so their kernel could not be checked locally. The code refuses any other kernel.
5. **Results not in git.** The site-tier result files behind the revision are untracked.
