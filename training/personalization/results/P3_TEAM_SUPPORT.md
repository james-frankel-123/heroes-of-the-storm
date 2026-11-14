# P3 team support: how far the drafter's teams leave human compositions, and whether it matters

**Question.** Paper 1's MCTS wins its judges partly with hero combinations that humans rarely play. How often does paper 2's drafter (`x-combo1-dr-chance`) do the same? Does it matter for its real-outcome edge?

**Short answer.**
- The drafter is out of support on about a third of its teams. Paper 1's MCTS is out on two thirds, and the humans in the same lobbies on 6%.
- The 17% previously reported for "Real" is not the humans. It belongs to the Real *arm*, which falls back to GD whenever the opponent arm has taken the human's pick. The humans' own teams are at 6.3%, close to paper 1's held-out 5%.
- A corpus from the lobbies' own weeks changes nothing. At equal corpus size, every arm's support and every gap match paper 1's corpus within about 2 points.
- No arm shows a support-vs-record frontier. Arms that stay in support (`aMCTS-bc`, DraftRec) reach judge-free records close to the drafter's.
- On real games, the personal term is about as reliable on rare teams as on common ones. Its slope is 0.75 to 0.84 in every support bin.
- Nearly every out-of-support team could have stayed in support. A legal in-support hero existed at the breaching pick in 99% of cases. The hindsight price of taking one is a median 1.6 points of the drafter's own leaf value.

Script: `p3_team_support.py`, stages base, boot, ablation, corpus, matched, residual and regret. Numbers: `results/p3_team_support.json`. Support is paper 1's yardstick. min3 is the corpus count of a team's rarest 3-hero subset, and the floor is 25, which is the 5th percentile of paper 1's held-out human teams. PMI is the mean pair PMI, where higher means heroes co-occur more than their popularity predicts. SEs are lobby bootstrap over the 300 headline lobbies unless stated otherwise.

## 1a. Support per arm, with SEs (paper 1's corpus)

Each arm's teams are pooled over both seats against the other nine arms (5,400 teams). "Humans" are the actual final teams of the 300 lobbies (600 teams).

| arm | below 25 | median min3 | mean PMI | seen exactly |
|---|---|---|---|---|
| humans (actual teams) | **6.3% (1.0)** | 201 (14) | −0.099 (0.004) | 66% |
| DraftRec | 3.0% (0.5) | 302 (12) | −0.070 (0.002) | 74% |
| Real arm | 17.4% (1.1) | 135 (8) | −0.160 (0.004) | 50% |
| GD | 18.5% (0.8) | 99 (6) | −0.259 (0.004) | 59% |
| p1step-cur | 28.9% (0.9) | 60 (3) | −0.199 (0.002) | 30% |
| a1step-cur | 31.6% (1.0) | 47 (2) | −0.222 (0.003) | 24% |
| **selected design** | **37.6% (1.2)** | 34.5 (1.1) | −0.270 (0.004) | 20% |
| aMCTS-pp | 39.5% (1.1) | 33 (1) | −0.281 (0.004) | 19% |
| aMCTS-cur | 66.9% (1.1) | 15 (2) | −0.298 (0.003) | 7% |
| paper 1's MCTS (pMCTS-cur) | 68.0% (1.2) | 14 (1) | −0.294 (0.003) | 6% |

**Selected design minus another arm** (same bootstrap draws):

| vs | below 25 (pp) | median min3 | PMI |
|---|---|---|---|
| paper 1's MCTS | −30.4 (1.7) | +20.5 (1.6) | +0.025 (0.005) |
| aMCTS-pp | −1.9 (0.8) | +1.5 (0.9) | +0.011 (0.003) |
| Real arm | +20.3 (1.4) | −100 (8) | −0.110 (0.006) |
| humans | +31.3 (1.4) | −166 (13) | −0.171 (0.006) |

**What the Real arm is.** It replays each human action while that action is still legal. When the opponent arm has already taken or banned the hero, it falls back to GD. 48% of its teams have at least one fallback pick.
- Teams without a fallback sit at 7.7% (1.3) below 25.
- Teams with a fallback sit at 27.9% (1.5).
- Its 17.4% is therefore mostly GD splicing, not human drafting.
- Real arm minus humans: +11.1 pp (0.7).

## 1b. All arms on common opponents: support vs the judge-free record

Arms play different schedules, so a fair comparison needs a common opponent set. Two sets are used:
- **(A)** both seats against `aMCTS-pp`. This covers every design-study row, 42 arms.
- **(B)** both seats against each of `pMCTS-cur`, GD and Real. This covers 38 arms. Record and J are against `pMCTS-cur`.

The record is the judge-free paired record in pp per covered player. Its SEs are lobby-only and carry no seed term, so they are tighter than P3_TOURNAMENT's CIs.

Set B, selected rows (all rows are in the JSON):

| arm | below 25 | PMI | record vs pMCTS-cur | J vs pMCTS-cur |
|---|---|---|---|---|
| DraftRec | 2.9% (0.5) | −0.070 | +3.0 (1.3) | 52.8 |
| `aMCTS-bc` | 8.7% (0.7) | −0.122 | +4.0 (1.3) | 51.3 |
| `pMCTS-bc` | 10.2% (0.8) | −0.111 | −1.0 (1.6) | 46.6 |
| Real arm | 18.8% (1.3) | −0.163 | +1.6 (1.2) | 46.7 |
| `a1step-cur` | 31.6% (1.2) | −0.224 | +4.1 (1.4) | 52.9 |
| **selected design** | 36.0% (1.4) | −0.265 | **+4.8 (1.3)** | **59.0** |
| `x-combo1` | 36.8% (1.4) | −0.265 | +3.3 (1.4) | 59.2 |
| `aPol-pp` | 37.2% (1.5) | −0.294 | +5.8 (1.3) | 57.3 |
| `aMCTS-pp` | 37.8% (1.3) | −0.277 | +5.8 (1.1) | 58.2 |
| `x-combo3` | 42.4% (1.5) | −0.263 | +4.0 (1.4) | 58.7 |
| `aMCTS-sp` | 59.1% (1.4) | −0.302 | +4.2 (1.0) | 56.1 |
| `pPol-cur` | 64.6% (1.3) | −0.297 | −0.1 (0.5) | 50.2 |
| `aMCTS-cur` | 64.6% (1.2) | −0.302 | +0.3 (0.5) | 50.9 |
| paper 1's MCTS | 72.8% (1.5) | −0.294 | — | — |

- **No frontier.** Across arms, the rank correlation of the below-25 share with the record is +0.10 in set A and −0.24 in set B.
- **The record can be had in support.** `aMCTS-bc` gets +4.0 with 8.7% of its teams out of support, and DraftRec gets +3.0 with 2.9%. The selected design gets +4.8 with 36%.
- **What the in-support arms lack is J.** They score 51 to 53% against paper 1's MCTS; the selected design scores 59%.
- **The x-pp and x-combo design variants all sit at 36 to 46% below 25.** Their records against `aMCTS-pp` range from −1.6 to +3.1. The selected design is at +0.7 (0.8). The rich-prior variants (`x-rich-*`) are at 13 to 37%, with records of −2.2 to −0.2.
- **Set A shows the same picture.** The out-of-support rate is a property of the search family, not of one design choice.

## 1c. A time-matched human corpus

Three outcome-free count corpora are built from the slot table's Storm League teams. Each excludes the sealed build, the sealed ids and the 300 headline games:

| corpus | window | teams |
|---|---|---|
| lobby window | 2026-05-23 to 06-30 | 169,062 |
| post | 2026-05-23 to 09-11 | 494,376 |
| pre | 2026-02-11 to 05-22 | 308,307 |

Paper 1's corpus has 3.73M teams. A raw floor of 25 on a corpus 4 to 22 times smaller is not comparable. Each matched corpus is therefore scored at the size-equivalent floor, 25 × N/3.73M: 1.1 for the lobby window and 3.3 for post. The reference is paper 1's corpus binomially thinned to the same size and scored at the same floor.

Share below the size-equivalent floor, lobby-window corpus vs thinned paper 1:

| arm | lobby-window corpus | paper 1, thinned | mean PMI, lobby window (paper 1) |
|---|---|---|---|
| humans | 12.7% (1.3) | 14.3% | −0.097 (−0.099) |
| Real arm | 25.0% (1.3) | 26.9% | −0.160 (−0.160) |
| selected design | 52.6% (1.1) | 54.4% | −0.269 (−0.270) |
| aMCTS-pp | 55.2% (1.1) | 56.9% | −0.280 (−0.281) |
| paper 1's MCTS | 78.5% (1.0) | 78.4% | −0.305 (−0.294) |

- **Matched in time, nothing moves.** The selected-design-minus-humans gap is +39.9 pp (1.5) on the lobby-window corpus and +40.1 (1.4) on thinned paper 1. The corpus closes 0.4% of the gap. On the post corpus it closes −1%.
- **PMI closes 0 to 4% of the gap.** Selected design minus humans is −0.172 on the lobby window and −0.171 on paper 1. Against the Real arm it is −0.109 and −0.110.
- **Humans at the matched corpus's own 5th-percentile floor.** On the post corpus that floor is 3: the humans sit at 5.8% (0.9), the selected design at 33.4% (1.1) and paper 1's MCTS at 42.6% (1.3). On the lobby-window corpus the 5th percentile is 0, so that floor is degenerate.
- **Was 17% corpus drift? No.** It was the Real arm's GD fallbacks. The humans are at about 5–6% on either corpus.
- **One drift-like signal, in paper 1's MCTS.** On the pre corpus (Feb to May, the end of its training data) its teams look more supported: 53.7% vs 76.9% thinned. Against the lobby window they look exactly as unsupported as on paper 1's corpus (78.5 vs 78.4). Its compositions fit the late-snapshot meta better than the all-years corpus suggests, but not the lobbies' own weeks.

## 2. Personal residual vs the real team's support (judge window)

**Data.** Real games from 2026-07-01 to 09-11 (162,657 games), excluding the sealed build and sealed ids. Each game gives one row per team, and SEs are clustered by game.
- r = y − WP_pop.
- P = the drafter's personal term: sig(b1·logit WP + b2·dS + b3·dO) − sig(b1·logit WP). It uses the headline combiner (b2 = 4.24) and the drafter state as of 06-30, on the actual player-hero cells.
- e = r − P.
- 71% of player slots have a state. The rest get s = 0, i.e. no personal term. The all-known subset (16,165 games) agrees within its wider SEs.

Slope of r on P (1 = the personal term is right on average), by the team's min3 on paper 1's corpus:

| min3 | share of teams | slope of r on P | mean P (pp) | mean e (pp) |
|---|---|---|---|---|
| 0–4 | 0.5% | 0.84 (0.10) | −0.27 | +1.6 (1.2) |
| 5–24 | 5.7% | 0.74 (0.03) | −0.14 | +0.0 (0.3) |
| 25–99 | 22.9% | 0.74 (0.02) | −0.12 | +0.6 (0.2) |
| 100–499 | 51.5% | 0.80 (0.01) | +0.01 | +0.1 (0.1) |
| 500+ | 19.4% | 0.83 (0.02) | +0.17 | −1.1 (0.2) |

- **Rare minus common slope.**
  - Paper 1's corpus, below 25: −0.04 (0.03).
  - Lobby-window corpus, min3 below 2 (about the same rate): −0.06 (0.02).
  - The personal term is about 5–7% more over-trusted on rare teams. Every bin shows the overall over-spread of 0.79, which matches the OOD audit's 0.77.
- **The residual level does not fall on rare teams. It rises.**
  - e trends −0.47 pp (0.07) per log unit of min3 on paper 1's corpus and −0.21 (0.08) on the lobby-window corpus.
  - Most of this is in r itself (−0.37 and −0.10): WP_pop slightly under-rates rare human teams and over-rates the most common ones.
  - PMI shows no trend in e: −0.7 (0.9) pp per PMI unit.
- **Reading.** Player-hero skill carries over to rare compositions about as well as to common ones. There is no sign that the personal term is badly over-trusted on rare teams. Selection caveat: these are compositions humans chose, not the drafter's.

## 3. Regret audit: could the out-of-support teams have stayed in support?

2,033 of the selected design's 5,400 teams (37.6%) fall below 25.
- **Where the breach happens.** The first breaching pick is the team's 3rd pick in 390 cases, its 4th in 777 and its 5th in 866.
- **The alternatives.** At the breaching pick, the alternatives are the heroes in the drafter's own action set: the team-pool legal mask plus paper 1's structural mask, which contains every actual pick. The median action set has 73 heroes.

| | share of sub-floor teams |
|---|---|
| a hero kept every triple ≥ 25 at the breaching pick | **99.4% (0.3)**; median 38 such heroes |
| also unused later, and the whole substituted team ≥ 25 (rest of draft fixed) | **87.1% (1.3)** |
| the best such fix adds a player on a cell with n < 5 | 19.4% (1.2) |

Cost by the drafter's own leaf V: vintage WP plus b2·S + b3·O, with the best of 120 assignments. It reproduces the score files' own V to 1e-7. These are hindsight swaps: one pick changes, and the rest of the draft stays as it happened.

| | median ΔV (pp) | mean ΔV (pp) | population WP part | b2·ΔS (logit) |
|---|---|---|---|---|
| best in-support swap vs actual pick | +0.44 (0.16) | +0.45 (0.16) | +1.4 | −0.053 |
| best swap of any hero vs actual pick | +2.0 (0.1) | +2.3 (0.1) | +3.9 | −0.075 |
| **floor cost: best any (or actual) minus best in support** | **1.6 (0.1)** | **2.5 (0.1)** | | |

- **In hindsight the searched pick is the best leaf only 29% of the time.** A max over about 70 alternatives with the future fixed always beats a pick made under uncertainty, so the absolute ΔVs flatter the alternatives.
- **The floor cost is the clean number.** Staying in support costs a median 1.6 points of the drafter's own value: zero for 30% of teams, 3.7 at the 75th percentile and 6.6 at the 90th.
- **The in-support fix trades personal value for population WP.** It gains WP_pop and loses personal term: b2·ΔS ≈ −0.05 logit, about −1.3 points at p = 0.5.
- **Where the personal value comes from.** Part of the drafter's personal edge rides on heroes that complete rare compositions. The deep-cell part is small: 19% of the fixes move a player to a thin cell.

## Caveats

- **1a/1b.** Arms in 1a face slightly different opponent sets: each faces the other nine. The common-opponent tables in 1b fix this. Record SEs here omit the seed term.
- **1c.** The matched corpora are much smaller than paper 1's, so the comparison uses a rate-equivalent floor and a thinned reference rather than the raw 25. Thinning is per triple, which keeps each triple's distribution but not the correlation among a team's triples.
- **Selection in part 2.** Part 2 uses human teams in real games, and humans pick rare compositions for their own reasons.
- **Part 3 holds the rest of the draft fixed.** The opponent and the drafter's later picks would respond to a different pick, so the costs are a static, hindsight bound and not a re-search.
