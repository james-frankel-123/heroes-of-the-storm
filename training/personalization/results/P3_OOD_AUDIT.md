# P3 out-of-support audit: does the drafter feed on thin player-hero cells?

**Question.** Paper 1's MCTS found hero combinations the judges over-rate. The personalized drafter (`x-combo1-dr-chance`) maximizes b2·(S_ours − S_theirs) with b2 = 4.24, S = posterior mean, no sd penalty, over a pool that includes every hero a teammate has played, and assigns heroes by the best of 120 assignments. That is a max over noisy estimates. Does it put players on thin cells (a 3–0 record), and do those cells disappoint?

**Short answer: no.**
- The drafter mostly drafts deep cells. 74% of its cells have 20+ earlier games, against 65% for the humans' own picks in the same lobbies and 41% for paper 1's MCTS.
- When it does use a thin cell, it picks the hot ones. On cells with 2–4 games, the average past record is +12.4 points, against −0.5 for the humans' picks. Thin cells are 7% of its cells, though, and its hot-thin share (3.1%) is no higher than the humans' (3.4%).
- The value function has a mild winner's curse. Selected cells come in 0.4 to 1.3 points below prediction, and most of these gaps are within 2 SE.
- The judge-free +4.8 record comes from deep cells. Cells with 20+ games contribute +4.3 of the +4.8. Cells with 1–4 games contribute +0.0, and hot-thin cells contribute −0.1.

Script: `p3_ood_audit.py` (stages prep, verify, exposure, curse, judges, record). Numbers: `results/p3_ood_audit.json`. All edges are in WP points on the y − WP_pop scale. Throughout, n is the merged games on the hero before the as-of day. Bins are 0, 1, 2–4, 5–19 and 20+. A cell is "hot thin" if 1 ≤ n ≤ 5 and its past mean edge is ≥ +0.20 or its win share is ≥ .80.

## Data and windows

- **Drafts and scores.** These are the headline tournament files: 300 lobbies, every arm's drafts and score files. They are copied from the 3080 (`cache_def_def1027_headline/p3t/{lobbies.npz,tourney,scores}`, 220 MB). The multi-GB state files were not copied.
- **Player states.** These were rebuilt on the main box.
  - The drafter state comes from new query files answered by `p3_rcf_states.answer` (half 0, unfiltered, as of the day before each lobby). Every query is a row of the definitive `judge_q/p3t_data_states__0.npz`.
  - The judge states are the existing answer files `judge_a_{orig,gd,v2}/p3t_data_states__3.npz`.
  - **Check:** the rebuild reproduces the headline score files exactly. The judge-free per-player record and J's personal sum match with difference 0, and the drafter's own V matches to 2e-16, on five pairs.
- **Realized edges, parts 2 and 4.** These use the judge window, 2026-07-01 to 09-11: slot rows, post-snapshot, not the sealed build 2.55.17.98025 and no sealed id (162,657 games). Every drafter state, combiner and model weight was fixed before July 1, so this window is outcome-independent of the drafter's s. It is the window the judge-free record already uses.
- **Out-of-sample window for the judges, part 3.** This is 2.57 Storm League, 2026-09-28 to 10-26, from the weekly build wk261101's slot table (84,656 games). That build excludes the sealed list, and `p3_sealed` refilters it here; no sealed game is read. The def1027 table has almost nothing between 09-12 and 09-27 that is not the sealed build.
- **Selection caveat (parts 2–4).** We only see outcomes for cells a player chose to play later. Players keep heroes that go well and drop ones that go badly. So "played" cells are selected on the player's own later form. The tables report what share of selected cells were played at all.

## 1. Exposure: what the drafter's cells look like

Cell shares by games on hero, under the drafter's own assignment (b2·S + b3·O on its state). Each arm's drafts are pooled over both seats against the other nine arms.

| arm | n=0 | 1 | 2–4 | 5–19 | 20+ | median n | hot thin | mean s |
|---|---|---|---|---|---|---|---|---|
| humans' actual picks | 3.2 | 2.6 | 7.3 | 21.7 | 65.3 | 41 | 3.4 | +0.02 |
| **selected design** | 5.1 | 1.6 | 4.2 | 15.6 | **73.6** | 60 | 3.1 | +2.13 |
| aMCTS-pp | 6.3 | 1.7 | 4.6 | 16.0 | 71.4 | 54 | 3.1 | +1.72 |
| DraftRec | 2.2 | 1.2 | 3.7 | 19.6 | 73.3 | 55 | 2.3 | +1.23 |
| Real (arm) | 6.6 | 2.3 | 6.8 | 20.3 | 64.1 | 39 | 4.0 | +0.23 |
| GD | 11.2 | 3.0 | 7.4 | 21.5 | 56.9 | 28 | 4.3 | −0.95 |
| **paper 1's MCTS** | **25.3** | 4.2 | 8.6 | 21.0 | 41.0 | 12 | 5.1 | −2.51 |
| aMCTS-cur | 24.5 | 4.0 | 8.3 | 21.1 | 42.2 | 12 | 5.1 | −2.34 |

- **Selected design minus paper 1's MCTS** (lobby-bootstrap SE):
  - hot thin −2.0 (0.3)
  - n 1–4 −7.0 (0.4)
  - n = 0 −20.2 (0.5)
  - 20+ +32.6 (0.6)
- **Selected design minus the humans' picks:**
  - hot thin −0.4 (0.4)
  - n 1–4 −4.2 (0.5)
  - n = 0 +1.9 (0.4)
  - 20+ +8.4 (0.8)
- **Within the thin cells it uses, the drafter does select on hot records.** By n bin, its cells' past edge versus the humans' picks:

| n bin | selected design | humans' picks |
|---|---|---|
| 1 | +6.4 | −0.2 |
| 2–4 | +12.4 | −0.5 |
| 5–19 | +9.1 | +0.5 |

  44% of its n ≤ 5 cells are hot, against 27% for the humans'. This is the winner's-curse pattern, but on few cells.
- **About 5% of its cells are heroes the player never played** (n = 0). The pool is the team pool, not the player's. These cells carry s ≈ −2.7, so the drafter takes them only when forced, not because they look good.
- **The assignment rules agree on 82% of cells.** The drafter's own assignment puts the same hero on the same player as the record's most-games rule in 82% of cells.

## 2. Winner's curse on later games (judge window)

Players are every tournament-window player (60,335). s is the drafter-fit state as of 2026-06-30. The outcome is the game-weighted mean y − WP_pop in July to September. Slope is realized regressed on predicted; gap is realized minus predicted. SEs are clustered by player.

| cells | games | slope (SE) | gap pp (SE) | gap, n=1 | gap, n 2–4 | gap, n 5–19 | gap, 20+ |
|---|---|---|---|---|---|---|---|
| all played cells | 1,155,493 | 0.77 (0.01) | +0.64 (0.05) | +0.97 (0.22) | +0.50 (0.16) | +0.44 (0.10) | +0.42 (0.06) |
| each player's top-s hero (39% played) | 172,965 | 0.81 (0.04) | −0.16 (0.12) | −1.25 (1.06) | −0.67 (0.67) | −0.84 (0.34) | −0.02 (0.13) |
| top-s among n 1–4 cells (14% played) | 22,101 | 0.82 (0.07) | −0.41 (0.35) | +0.49 (0.75) | −0.65 (0.40) | | |
| hot-thin cells (10% played) | 56,623 | 0.88 (0.04) | +0.24 (0.22) | +0.61 (0.32) | −0.13 (0.32) | +0.17 (0.69) | |

**Headline lobbies, the cells each drafter actually drafted.** Here s is the state at the lobby.

| cells | played later | gap pp (SE) | slope (SE) |
|---|---|---|---|
| selected design | 66% | +0.06 (0.27) | 0.63 (0.08) |
| paper 1's MCTS | 40% | +0.52 (0.36) | 0.82 (0.10) |
| humans' picks | 83% | +0.28 (0.33) | 0.69 (0.09) |

The selected design's thin bins hold 59 to 226 cells each, with SEs of 2 to 4.5 points. They are not informative on their own.

**Reading:**
- The value function is too spread out everywhere, not only in thin cells: slope 0.77 overall, 0.76 to 0.85 by bin. The over-trust at 1–4 games that the post-freeze check found (slope 0.84 to 0.87) shows here as a level shift. Played cells beat their prediction by +0.4 to +1.0, most at n = 1, which is the selection on later play.
- Picking the max costs about 0.6 to 1.1 points relative to all played cells. The top-s gap is −0.16 against +0.64 for all cells; for top-s among thin cells it is −0.41 against about +0.5 to +1.0. Part of this comes from the selection caveat as well as the curse.
- Hot-thin cells are not over-predicted (+0.24, SE 0.22): the shrinkage already discounts a 3–0.

## 3. Judge bias by sparsity

Each cell's personal term is in WP points at p = 0.5: (c2/4)·s_judge. J_cal multiplies by cal_k. CONS has no personal term. Judge states use July to September evidence only, as of 09-11.

- **In the judges' own window (in-sample)** the comparison is mechanical. A cell with one window game has realized −6.6 against a predicted −2.0, because one-and-done heroes are mostly losses. Slopes are 2.2 to 4.5, so the judges shrink hard. This shows no over-crediting of thin cells, but it is not a test.
- **Out of sample (2.57, 09-28 to 10-26)** every judge has a slope of 0.74 to 0.76, flat across sparsity:

| judge | slope (SE) | slope, 1 window game | 2–4 | 5–19 | 20+ | gap pp (SE) |
|---|---|---|---|---|---|---|
| J | 0.75 (0.03) | 0.72 (0.08) | 0.77 (0.07) | 0.69 (0.06) | 0.79 (0.10) | +1.31 (0.09) |
| J_cal | 0.75 (0.03) | 0.71 | 0.76 | 0.68 | 0.79 | +1.32 |
| chosen_flip_gd | 0.74 (0.03) | 0.75 | 0.79 | 0.71 | 0.82 | +1.37 |
| chosen_v2flip | 0.76 (0.03) | 0.76 | 0.87 | 0.74 | 0.91 | +0.76 (0.09) |
| CONS | — | | | | | +0.40 (0.09) |

- **No judge credits thin cells more than deep ones.** The judges are about 25% too spread out at every depth. The positive gap is the selection on later play plus the patch change.
- **Under J's assignment, the selected design's cells are mostly ones the judge saw played.** 24% have no window games and 46% have 5+, against 54% and 20% for paper 1's MCTS. Its average personal term is +0.15 points per cell, against −2.44 for paper 1's MCTS.

## 4. Where the +4.8 judge-free record comes from

The paired record against paper 1's MCTS reproduces at +4.77 (SE 1.27). Contributions add to the total.

| selected design's cell | share of covered players | contribution (SE) | mean diff per player |
|---|---|---|---|
| n = 0 | 0.3% | −0.09 (0.07) | |
| 1 | 0.5% | −0.05 (0.10) | |
| 2–4 | 1.8% | +0.05 (0.16) | |
| 5–19 | 12.7% | +0.53 (0.45) | +3.5 |
| 20+ | 84.8% | **+4.33 (1.18)** | +5.6 |
| hot-thin cells | 1.0% | −0.12 (0.08) | |

- **By relation:**
  - The design's hero is deeper than paper 1's for 60% of covered players, contributing +5.27 (1.07).
  - It is shallower for 18%, contributing −0.30 (0.67).
  - It is the same hero for 21%, contributing 0.
- **Against the other arms the pattern holds.** 20+ cells carry the record and thin cells add nothing:

| vs | total (SE) | from 20+ cells |
|---|---|---|
| Real | +3.07 (1.06) | +3.16 |
| GD | +3.62 (1.09) | +2.98 |
| DraftRec | +2.10 (0.83) | +1.90 |
| aMCTS-pp | +0.67 (0.83) | +0.68 |

- **The record's own rule does not hide thin-cell picks.** The record places players on heroes by most earlier games. Scored with the drafter's own assignment instead, the record is +5.24 (1.28) vs paper 1's MCTS and +3.69 (1.02) vs real drafts.
- **Caveat: coverage.** Only 37% of players are covered under both arms against paper 1's MCTS (design 77%, paper 1's MCTS 44%). Thin cells are rarely covered, because players seldom go on to play them. The record cannot see the thin cells paper 1's MCTS uses, but it would see thin cells the design drafted, and there are few of them.

## Caveats

- **Selection on later play.** Outcomes exist only for cells players chose to play. This raises realized edges in thin bins, most at n = 1, and can mask a winner's curse. The played shares are reported next to each block.
- **The judges' in-window numbers are in-sample.** The out-of-sample test is a new patch, 2.5 to 6.5 weeks later. Part of the level gap is the patch.
- **Small thin bins.** The headline-lobby bins below 5 games have under 250 cells each, and their SEs are large.
- **Not a test of hero-combination support.** This audits player-hero cells only, not paper 1's question about hero combinations under the population judges. CONS still prefers paper 1's MCTS (44.3%).
