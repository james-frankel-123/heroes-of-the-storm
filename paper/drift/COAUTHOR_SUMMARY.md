# Paper 2 (Drift) — summary for coauthors


> **Update 2026-09-27 (pre-submission review, details in REVIEW_2026-09-27.md).**
> The head-to-head scoring overlapped the maintained agent's own statistics.
> Rescored on 298K games from five later builds that no agent touched: the
> hero-timing deficit holds (+0.66pp vs the same-cutoff unmaintained agent,
> z 4.3; +0.56 at 1yr; +1.10 at 2yr; the volume control keeps 91%). The
> four-month synergy gap shrinks from +0.76 to +0.23. Stale agents counter
> slightly better, which gives back about a sixth of the timing gain. The net
> realized team win-probability gap is +2.5 to +4.1pp. On clean data the
> decayed-stats agent beats the cumulative one on timing (+0.31, z 3.6),
> where before it looked tied. The "stale-4mo" arm is renamed "unmaintained"
> (same training cutoff as maintained). The settling thresholds below are
> per-tier slice counts (low 6.5K, high 7.8K, mid 8.7K games); the pooled
> median is 10.7K. The numbers below predate this update.
Title: "Maintaining Draft Policies Under Meta Drift: Data-Side Refresh
Beats Retraining" (2026-07-24: dropped the "in Adversarial Combinatorial
Assembly" tail; the program is now framed as four data-regime conditions
rather than a branded class, and paper 1 was retitled to "Diagnosing and
Repairing Out-of-Distribution Failure in MOBA Draft Policies").

Max, Ernest, James, Hod — all experiments are complete (including the full
post-review hardening queue, 2026-07-13); this is the pre-writing digest.
Venue: **ToG** (series coherence with paper 1; TMLR is the fallback if we
later want the MLOps/continual-learning audience). Infra:
`training/drift2026/` — every table regenerable via MANIFEST.md.

**Name glossary** (the word "frozen" is overloaded in our internal files;
the paper will use these names):
- **maintained** — model trained once, feature aggregates refreshed per build
  (internally `d2c_cumprev`)
- **stale-4mo / stale-1yr / stale-2yr** — same recipe, everything frozen at a
  cutoff that far before the test window (`d2b_allhist`, `w7_stale*`)
- **leaky-frozen** — the paper-1 control whose stats saw the future (never a
  result, only a control)

## Headline: what maintenance is worth

One system difference — refresh the feature aggregates instead of freezing
them at deployment — measured three ways:

1. **Win rate, head-to-head, judge-free (W6/W7).** The maintained agent vs
   agents of increasing staleness, 2,000 direct matchup drafts per point,
   scored against the *actual future meta's outcomes* (no learned judge),
   inference with crossed seed effects (5×5 pairings):

   | opponent staleness | deficit in future-meta hero WR | z |
   |---|---|---|
   | ~4 months | +0.80 ± 0.20 pp | 4.1 |
   | ~1 year | +0.74 ± 0.18 pp | 4.2 |
   | ~2 years | +0.96 ± 0.14 pp | 6.7 |

   (The 2-year row was corrected on 2026-07-13: an earlier +1.16 figure was
   contaminated by a pipeline flaw — two stale-side seeds were still
   mid-training when the head-to-head first ran; details under Status.)
   Shape, stated as the data shows it: **the cost arrives within months
   (~0.8pp) and is essentially flat through two years** — the 2-year point
   is directionally higher but the rise over the plateau is not
   significant (+0.19 ± 0.19). The sharpest part of the corrected result:
   at two years the deficit is *entirely hero-meta timing* — the stale
   agent's synergy-vs-future-truth is intact (+0.08, vs +0.76–0.80 gaps at
   the earlier points where composition discipline also differed).
   Interaction knowledge doesn't rot even at two years; hero-timing
   knowledge rots within months. That is finding 3's signal-class
   decomposition showing up in an end-to-end agent. **The volume-matched
   control (W8a) passed: a maintained agent trained on the stale-2yr
   agent's exact row count keeps +0.82 ± 0.23pp (z = 3.6) of the
   full-volume +0.96 — 86% of the edge survives volume matching, and only
   ~19% of the VF-accuracy gap is data scale. The gradient is a staleness
   story, cleanly.** The maintained side also drafts higher-synergy teams (+0.76 vs
   future ground truth, z = 2.4) and, at full power (W10: 15 seeds/side,
   4,500 drafts), drafts degenerate comps at **22.0% vs the stale agent's
   36.9% — a paired −15.0pp ± 6.7 under crossed seed effects, z = 2.2,
   now confirmatory**. (Honest note: 15 seeds moved the point estimates
   from the 5-seed values 14.5/35.3, so the ratio claim is ~1.7×, not
   2.4×.) The same 4,500 drafts reproduce the headline judge-free deltas
   at tripled seed count: hero WR +0.72, synergy +0.69. Degen trend
   across the stale pools: flat (+0.5 pp/yr, z = 0.16) — comp discipline
   doesn't decay with staleness; hero timing does. One
   pre-emptive honesty note on the judge-free metric: future-meta hero WR
   is realized outcomes, not model opinion, but it is a *marginal* (per-
   hero) quantity — which is exactly why it is always paired with the
   synergy-vs-future-truth and degenerate-rate numbers rather than carrying
   the claim alone.
2. **Accuracy.** Over a 2.8-year simulated deployment (games-weighted):
   frozen-everything 55.13% vs refresh-only 56.08% (+0.95pp), within 0.73pp
   of the retrain-every-build oracle. On the fixed future test window the
   best maintenance variant reaches **57.22%** vs 56.21% for stale-4mo.
   Edge-over-chance framing, labeled precisely: the frozen system's 5.13pp
   edge grows to **6.08pp with stats refresh alone (+19%)** and to 6.81pp
   with refresh plus near-free periodic retrains (+33%, the near-oracle) —
   the thesis clause "the story is the stats" rests on the 6.08 number,
   not the 6.81. (Deployment-weighted and future-window accuracies are
   different windows; the paper labels every number's window to avoid a
   fake contradiction.)
3. **Policy quality (the real damage).** Greedy degenerate-composition rate
   43.5% → 16.8%; under MCTS 17.6% → 9.3%. Sub-1pp accuracy differences
   hide 2.5× differences in how often the policy drafts a broken team —
   paper 1's accuracy–policy disconnect, now on the time axis.

**The evaluator-drift exhibit (vintage matrix).** Learned judges trained at
five era cutoffs score the same 2,000 maintained-vs-stale-4mo drafts
monotonically in judge vintage: 0.434 (QM-2021) → 0.461 (2022) → 0.493
(2023) → ~0.50–0.51 (2024–2026; newest judges agree with the consensus).
A 2021 judge *reverses* the verdict — not evidence against maintenance but
drift biting the evaluator itself: the whole difference between these
agents is current-meta knowledge, which an old judge cannot see. (Caveat
on the leftmost point, now *quantified* — W8d added a ranked-2022Q1 judge:
it scores the same drafts at 0.469 vs QM-2021's 0.434, so roughly a third
of the QM point's extremity is mode confound; the within-ranked 2022→2026
gradient (0.47 → 0.50) carries the exhibit and is clean.) The same
old judges reproduce paper 1's tournament ordering (ρ = 0.92) because those
contrasts are composition-structural (era-stable per our decay curves),
while this one is meta-timing (the fastest-decaying class). Consequence for
practice: learned-judge scalars are only reported through this matrix,
never standalone — the headline rests on the judge-free metrics above.

**QM-2026 judge results (2026-07-21, `qm2026/judge_2026.py`).** The
matrix's missing cell is filled: a judge trained on 116,589 recent-era
(2025-07+) Quick Match games — current-era AND out-of-family — scores
the same 2,000 maintained-vs-stale drafts at **0.508 ± 0.002 for
maintained (win share 0.549)**. It agrees with the realized-outcome
verdict and with the same-era ranked judges (0.50-0.51), against
QM-2021's reversed 0.434. Within the QM family the era contrast is now
isolated from the mode contrast: same corpus mode, 2021 vs 2026, moves
the verdict 0.434 → 0.508. Paper-1 tournament rescore (exploratory,
panel frozen in prereg): Spearman 0.917 vs consensus — identical
agreement to the QM-2021 fifth evaluator. Degenerate probes all judged
losing (5-mage 0.38, 5-tank 0.41 vs standard). Now IN both papers
(2026-07-24): paper 2's Fig 3 shows both QM diamonds (0.434 and 0.508
bracketing the ranked curve) with the era-not-mode passage in
§Evaluators Drift Too and vintage counts updated to eight judges at
seven vintages; paper 1 carries the tournament rescore as the
out-of-family robustness passage.

**Blind patch-timing inference (2026-07-15, new §Operating Policy
passage).** Can match data alone reveal WHEN the environment changed?
A multi-scale changepoint detector (6 window scales, pick/ban/WR
families, per-hero self-normalization), tuned on pre-2025 boundaries and
evaluated held out on 2025-26: only the largest patches are recoverable
(9.4x/7.5x baseline, day-exact timing); held-out P 0.18 / R 0.27
overall; most boundaries score below ambient weekly variation. Paper
takeaway (4 sentences in the detector passage): blind detection cannot
reconstruct the event calendar, and our trigger never needs one — it
fires on deployed-stat shift magnitude, the property that transfers to
unlabeled domains. Also scoped into the transfer proposal's W4.

**The recommendation is now deployed (2026-07-15).** Our live drafter
retrains all three models monthly on the full corpus with 90-day
decayed aggregates (`training/production_refresh/`), and the site
serves the same decayed statistics the value function trains on
(closing a train/serve stats mismatch that predated the paper). First
refresh: WP 57.43% test acc, policy eval WP 0.796 vs the previous
production 0.777. The paper states this in one sentence at the end of
the W8b/W11 resolution passage.

## Core findings

1. **Drift is real but accuracy hides it.** All-history training: best
   validation (59.1%), worst future accuracy (56.2%); the best regimes beat
   it by under 1pp — but at the policy level the gap explodes (finding on
   the time axis of paper 1's accuracy–policy disconnect). Report accuracy
   as edge-over-chance (Max's rule): +0.87pp on a 6.21pp edge is 14%
   relative; vs fully-frozen the edge grows 33%. Never call this small.
   (Finding 1's edge numbers carry the same window labels as the headline:
   +0.87pp is future-window over stale-4mo; 5.13→6.08 refresh-only /
   →6.81 near-oracle are deployment-weighted.) Survives deep search (W4:
   17.6% vs 9.3% degen, 5 seeds/arm).
2. **Refresh the stats, retrain rarely — and decay the aggregates.** Data-
   side refresh beats every retraining regime. Best deployable variant:
   **90-day-half-life decayed aggregates + k=100 count-shrinkage, 57.22%
   future accuracy (+0.14 over cumulative refresh, seed-clean: every
   decayed seed beats every cumulative seed)**. Mechanism (partial-refresh
   ablation): cumulative aggregates are too dilute to transmit short-
   horizon drift — test-time cumulative refresh moves a fixed checkpoint
   ≤±0.07pp over 3 months (cumprev's +0.87 over all-history is largely a
   *training-time* effect); decayed aggregates move the inputs ~10× more,
   which is where the +0.14 comes from. The pairwise/composition class —
   not hero-WR — carries the small test-time gain (hero-WR-only refresh
   slightly hurts). Honest wrinkle: the first post-patch build regresses
   under sharp decay (recency-sharpened stats mis-describe a brand-new
   meta; shrinkage recovers half) — dilute priors are safest right after a
   patch, which ties into finding 4's latency story. Also-rans: causal
   patch-local stats 57.01, hybrid per-signal sourcing 56.91,
   embed+refresh 57.09. Selection hygiene (W8c): a nested pre-cutoff
   pseudo-future window selects the same HL=90 family, so the champion is
   not test-window-tuned (the within-family k=100/unshrunk choice flips
   between windows — reported as family-level selection, honestly).
   **The W8b twist, RETRACTED after W10/W11 (2026-07-14):** the
   "champion agent has worse composition discipline (22.7% vs 14.5%)"
   comparison was a seed-count artifact — champion@15seeds vs
   cumprev@5seeds; at matched 15 seeds cumprev degens at 22.0%. The
   settling experiment (W11: direct champion-vs-cumprev head-to-head,
   15×15, 4,500 drafts): **statistically tied on every policy metric**
   (hero timing z = 1.4, degen z = 0.7, consensus z = 1.3). The
   recommendation simplifies: **use decayed aggregates** — best
   predictions, agent quality indistinguishable. The residual finding is
   cleaner than the twist: maintenance-*variant* choice moves policy
   quality ~0 while maintenance-vs-none moves it +0.7–0.8pp — flavor is
   a prediction-level decision, refresh-at-all is the policy-level one.
   And the meta-lesson stands harder than before: claims about agents
   need agent-level seed budgets.
3. **Signal-class decay is heterogeneous — and decay speed does NOT equal
   refresh value.** Role-composition WR essentially static (r ≈ 0.95 at lag
   16 builds); pick rates very stable; per-hero WR the fastest-decaying
   reliable signal (0.94 → 0.62); net pairwise interaction
   weak-but-persistent. The general lesson, corrected by our own ablation:
   decay curves tell you *what is changing*; the partial-refresh ablation
   tells you *what is worth refreshing* — and they are not the same.
   Refreshing hero-WR (the fastest-decaying class) slightly *hurts*; the
   refresh value lives in the pairwise/composition class, because refresh
   value tracks what the model actually leans on, and mostly at training
   time. Cross-paper echo worth citing explicitly: in paper 1's
   4,100-config feature sweep, per-hero WR was the one feature group with
   *negative* marginal accuracy — hero-WR is simultaneously the
   fastest-drifting and least-load-bearing signal, a coherent two-paper
   story.
4. **Balance-change detection from outcomes alone works — and closes the
   loop** (likely the most novel section). Ground truth scripted from
   official patch notes (40 mapped patches + 5 no-change controls; all 27
   sizable boundaries validated). FDR-surviving WR-shift detections:
   precision 0.79 — outcome shifts almost always correspond to published
   changes; pick/ban detectors instead track community *adaptation*
   (recall 0.68, precision 0.32). **Detector-triggered refresh — with
   honest per-detection latency — matches blind every-build refreshing to
   within 0.013pp using 14 refreshes instead of 35.** Detect → refresh is a
   demonstrated closed-loop system. Latency symmetry: detection confirms at
   a median ~9.4K games into a build; the build's own stats become
   trustworthy at 6.5–12K games — by the time you know something changed,
   the repair data is ready.
5. **Recovery + heterogeneity.** Trust patch-local hero-WR after ~6–12K
   games (tier-dependent). Meta stability ordering by tier is measurable;
   comp structure stable everywhere. The never-fielded composition set is
   volume-driven, not meta-driven (consecutive-build Jaccard 0.81): paper
   1's augmentation targets would have been ~the same in any era.
6. **Behavior (opponent-model) drift is real but modest; rolling-fresh wins
   (W5, Max-identified arm).** The GD next-pick model loses +0.63pp future
   top-1 to behavioral drift — under the pre-registered 1.0pp materiality
   trigger, so the MCTS retraining arm was skipped by rule. Best deployable
   pool: a **30-day rolling window** (11.83, beating even the leaky frozen
   pool) that itself goes stale across later builds — behavior wants
   *continuous* refresh. Completes the three-speed maintenance picture:
   **stats every build (or detector-triggered), opponent model on a ~30-day
   rolling window, value model rarely.**
7. **Deliverable (running on hotsfever): the operating policy.** Regret vs
   the retrain-every-build oracle over 2.8 simulated years: frozen
   everything +1.68pp; stats-refresh-only +0.73pp; refresh + retrain every
   3 builds +0.03pp; detector-triggered refresh +0.74pp on 40% of the
   refreshes; warm-start finetune at K=3 +0.089pp. With fresh stats, any
   reasonable retrain style sits within ~0.1pp of oracle — the story is the
   stats, not the weights. ("Refresh the feature store, not the weights.")

## What we ran (arms, implementation, protocols)

All arms share paper 1's enriched WP architecture (283-dim, 256→128, 3
seeds/cell unless noted), a hard temporal cutoff at build 2.55.14.95918
(train ≤ cutoff; test = the 7 strictly-future builds, ~1.8M games), and
causal features throughout (a row's stats never include its own day or any
post-cutoff data).

- **Training regimes (D2, 8 arms + 2 controls):** all-history; recency
  windows of 3/6/12 builds; sample-decay at 90d/365d half-lives; learned
  patch embedding; cumulative-refresh (cumprev). Controls: leaky-frozen
  (paper-1 stats) and a within-build local-stats oracle (72.8%, quantifies
  leakage; never reported as a result).
- **Decayed aggregates 2×2 (Q7):** stats-side exponential decay (90d/365d
  half-lives) × count-shrinkage (k=100) — the new champion cell; HL=∞
  verified to reproduce the cumulative files exactly.
- **Within-build causal enrichment (W2a):** rolling same-build stats from
  strictly earlier days, shrunk toward the cumulative prior — adds nothing
  over refresh (57.01 vs 57.08); the leaky version of this (72.8%) is how
  we caught and removed a leakage bug.
- **Policy-level evaluation (W2b):** every regime drafts greedily, 3 seeds ×
  500 drafts vs the GD pool; compositions scored against future-period
  ground-truth stats, not model opinion.
- **Deployment simulation (W2c):** 36 builds / 2.8 years replayed forward;
  retrain-policy grid (never / every build / every 3 / every 6 / accuracy-
  triggered / detector-triggered with honest latency / warm-start finetune)
  × stats refresh on/off; games-weighted accuracy and regret vs oracle.
- **Drift structure (W3 family):** per-signal decay curves over 45 builds;
  changepoint detection vs patch-note ground truth (FDR-corrected);
  post-patch recovery thresholds by tier; per-tier heterogeneity;
  never-fielded stability; hybrid per-signal sourcing; embed+refresh.
- **Deep-search transfer (W4):** MCTS self-play (200 sims, 300K episodes,
  fused CUDA kernel, 5 seeds/arm) on best/worst regime VFs, benchmarked vs
  future-meta GD opponents.
- **Opponent-model drift (W5):** GD next-pick pools (frozen / causal-cutoff
  / 6-build window / 30-day rolling / future oracle) on 1.84M future
  samples, plus policy-level cells swapping pools; pre-registered
  materiality trigger for an MCTS arm (did not fire).
- **Head-to-head + staleness gradient (W6/W7):** maintained vs stale-4mo/
  1yr/2yr MCTS agents, 2,000 drafts per point (5×5 seed pairings, both
  orderings, paper-1 tournament protocol); scored judge-free against
  future-truth stats AND by the five-vintage judge matrix; inference via
  two-way crossed random effects. Full tables:
  training/drift2026/results/W7_STALENESS_GRADIENT.md.
- **Partial-refresh ablation (Q5):** inference-time stats swap on fixed
  checkpoints — refresh only hero-WR vs only pairwise/comp vs all vs none;
  positive control reproduced the champion bit-exactly.
- **Continuous-MMR ablation (W9, reviewer-preempt row):** the 3-tier skill
  grouping is a *sufficient* skill input — appending continuous avg_mmr
  adds +0.007pp (noise), and replacing the 3 tier dims with one continuous
  dim loses only −0.017pp (noise). Coverage 99.7%. Probe curiosity for the
  discussion: predictions compress toward 50% as the MMR input rises
  (drafts discriminate outcomes less among skilled players), and The Lost
  Vikings rate *stronger* at high MMR relative to baseline — consistent
  with their skill floor.

**Deliberately not run (so we can say so in reviews):** online/continual-
learning methods (EWC etc. — our point is that data-side refresh makes them
unnecessary at this timescale); cross-game generalization (Gourdeau's DOTA2
data remains an optional arm); player-skill drift (explicitly the
personalization paper's roadmap).

## Framing

Non-stationarity at three timescales (meta / talents / player skill) — this
paper builds the machinery; the talents and personalization papers consume
it. Thesis: *drift damages policies far more than accuracy metrics reveal,
and the cheapest repair is data-side (refresh — and decay — the
aggregates), not model-side (retraining), with balance changes detectable
from outcomes alone fast enough to trigger the repair.* Every clause is
individually demonstrated, including the trigger (detector-triggered ≈
every-build refresh at 40% of the refreshes). For the ML-facing framing:
the differentiator vs the classifier-centric concept-drift literature is
measuring drift damage at the *policy* level of a combinatorial decision
system (accuracy hides a 2.5× policy failure), the multi-year controlled
maintenance-policy comparison, and evaluator drift.

## Status / asks

- Both hardening queues complete as of 2026-07-13: the first (judge-free
  rescoring, vintage matrix, crossed-seed inference, detector-triggered
  closed loop, partial-refresh mechanism, staleness gradient, decayed
  aggregates, warm-start finetune) and the second (W8: volume-matched
  control PASSED at 86% edge retention; champion-agent runs at 15 seeds
  producing the best-VF≠best-agent finding; nested selection TRANSFERS at
  family level; ranked-2022Q1 judge quantifying the QM mode confound).
  Full tables: training/drift2026/results/W8*_*.md.
- Remaining optional items only: ~10 extra cumprev-agent seeds to make the
  degen contrast fully confirmatory (one GPU-afternoon; Max to call), and
  the W9 continuous-MMR ablation (running). **Nothing blocks drafting.**
- **Correction log (2026-07-13):** the MCTS worker saves its policy file
  periodically *during* training, so any pipeline step gated on file
  existence can consume half-trained policies. This contaminated the first
  stale-2yr head-to-head (two stale seeds ~30–60% trained → the stale side
  looked artificially weak: +1.16pp, consensus 0.550 z = 2.7). Re-run with
  completed checkpoints: **+0.96 ± 0.14 (z = 6.7), consensus 0.5164
  (z = 1.2, n.s.)** — the numbers now in the headline. All other published
  points verified clean against checkpoint timestamps (stale-1yr, stale-4mo,
  W4). All W8 stages now gate on the pool's DONE events, not file
  existence.
- Writing starts next. Known writing-time chores: the name glossary above
  applied everywhere; every accuracy window-labeled; games-vs-samples units
  stated.
- Reviewers will ask: single game, single major patch (same honest scoping
  as paper 1). Suggestions for missing arms welcome now — GPUs are free
  again.
- Figure candidates: the staleness dose-response (headline), the vintage
  matrix heatmap (evaluator drift), decay-lag curves by signal class, the
  changepoint precision/recall table, the W2c regret table with the
  closed-loop row.
