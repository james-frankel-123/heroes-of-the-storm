# Hypothesis-driven win probability: framing spec

Date: 2026-10-13. This is the build spec for the shared harness and the first two weeks. It consolidates three independent reviews (Fable 5.1, Opus 5.5, Astra), each with a Round 1 report (R1) and a Round 2 cross-critique (R2), plus the project lead's binding requirements. Citations such as "Opus R2 §5.3" point to those reports. Database counts come from small count-only queries on the research role, run today. No outcome was read.

## 0. Summary in numbers

- **What we build.** A win-probability (WP) model made of named additive terms. LLM sessions propose hypotheses, and a harness fits and tests them, so the functional form is chosen by tests instead of gradient descent through a black box. The model exports as lookup tables for the MCTS leaf.
- **Fresh data is the binding constraint.** Of 2,485,469 ranked drafts, everything before patch 2.57 has already been explored. The only fresh outcomes are 2.57: 37,023 games so far, arriving at 18K to 21K a week (Opus R2 §0, Astra R2 §5).
- **Every look is charged.** All data access goes through a broker. Each session gets 120 bits on its dev tranche per two-week cycle, at most 8 bits per cell group, and at most 40 hypothesis cards. For scale, one exact exploratory query ("Stitches with Medivh on each of 14 maps") leaks about 45 bits, while a 100-test Bonferroni family costs 6.6 bits (Opus R2 §0).
- **Success is OOD first.** The model must stay within 0.7pp accuracy and 0.004 log loss of the neural WP on a new build. It must also beat the neural WP on at least 2 of 4 OOD tests and halve its exploitability under MCTS.
- **First wave.** One infrastructure session runs days 1 to 6. Then three kit-anchored sessions run in parallel: named heroes, hero-blinded, and shuffled outcomes. The LLM-free baselines (state-space additive, EBM) and the frozen neural contrast are fit and scored by the harness itself.

## 1. Goal and success criteria

**Goal.** The model should beat the neural WP where it matters for drafting:

- transfer to new builds, rebalanced heroes and new heroes;
- calibration on rare compositions;
- resistance to exploitation by search.

It should give up little on ordinary games, and every term should carry a theory and a recorded test.

**The reference model, N.** N is the drift-aware neural WP. Out of time on June to September 2026 it scored 56.94% (calibration slope 0.93), and 59.70% with the P3 personal layer (+0.0149 log loss, slope 0.99) (Opus R1 §2.1). Two paper-1 findings bear on search: a leak-free neural WP rates five tanks at about 0.28 against a standard team, and greedy drafting with it fields 47.5% degenerate teams (Fable R2 §4).

**OOD suite.** The six tests below are frozen in the evaluator before any outcome access.

| ID | Test | Data |
|---|---|---|
| O1 | New build: train through 2.55.17, score on 2.57. Weighted equally by sub-build, and also by game. | Tranche C |
| O2 | Patch onset: the first 1K and 10K games after each build boundary, with models trained before it. Scored with frozen weights and with the declared update policy. | Backtests on S; the 2.57 onset in C |
| O3 | Leave one hero out: 20 heroes with at least 20K games each, chosen by keyed hash. Metric is the share of hero-identity log-loss value recovered. | Backtests on S |
| O4 | New hero: Xal'atath games in C (about 3.5K now, about 7K by Oct 26), scored at 0, 1K, 2K and 5K games of her history. | C |
| O5 | Rare compositions: a team's role multiset at or below 0.2% of training teams, or a within-team pair with fewer than 100 training games. Rarity is defined from training data only. | C |
| O6 | Search behavior: degenerate-team rate under greedy and under MCTS at 400 and 1,000 simulations, with no mask; five-tank and no-healer probes; optimism O(W); split-half pick agreement. O(W) is the mean, over the model's own MCTS drafts, of p_W − p_J, where J is an independent NN-free judge fit on a disjoint time block (Opus R2 §4.8). | Simulated |

Log loss is the gate metric. Accuracy has lower power for small terms (Fable R1 §3.1, Opus R1 §3.1). Accuracy, slope and intercept are reported alongside.

**Pilot go/no-go, decided on day 14 on C.** C is about 40K fresh 2.57 games after carve-outs (section 4). Each criterion compares the best NN-free model with N and with the EBM.

| # | Criterion | Go | No-go |
|---|---|---|---|
| G1 | Parity on O1 | Log-loss excess vs N at most 0.002 (point estimate) and at most 0.004 (95% upper bound); accuracy within 0.7pp; slope 0.90 to 1.10 | Excess above 0.004, or accuracy worse by more than 1.2pp |
| G2 | OOD advantage | Beats N or the EBM by the stated margin on at least 2 of 4: O2 by +0.001; O3 recovers at least 30%; O4 by +0.002; O5 slope at least 0.1 closer to 1, with log loss no more than 0.002 worse | None of the four |
| G3 | Search safety (O6) | MCTS-1000 degenerate rate at most 5% and at most N's; optimism at most half of N's; five-tank probe at most 0.25; split-half agreement at least 40% | Optimism worse than N's |
| G4 | Loop yield | LLM-accepted terms add at least +0.0005 log loss on C over the same backbone without them, and beat the EBM by 0.0005; pooled retention (C gain divided by dev gain) at least 0.5 | Added gain at or below 0, or retention below 0.3 |
| G5 | False accepts | Shuffled session accepts at most 1 card; 0 of 12 null canaries accepted; at least 4 of 6 known-true canaries accepted | 2 or more false accepts |
| G6 | Harness | Every planted leak rejected; zero unlogged access; ledger reconciles; swap error below 1e-8 | Any failure |
| G7 | Replays | Parse success at least 98%; mediator signal-to-noise ratio (SNR) measured | Parse success below 90% |

**Decision rules.**

- **If G1 to G6 pass,** cycle 2 runs as planned.
- **If the backbone passes but G4 fails,** keep the glass-box model, run automated term search, and limit the LLM to documentation and patch priors.
- **If the backbone fails G1 against the EBM too,** the fitter is at fault. Fix it and rerun.
- **If the backbone beats the EBM but fails against N,** report the interpretability cost and decide it as a product question. A neural hybrid is excluded by requirement 1.

**Power note.** Whole-model comparisons on 40K games have an SE of 0.0007 to 0.001. C can therefore show a 0.003 shortfall, but it cannot certify a 0.001 tie. Certifying a tie needs F1. An underpowered result is inconclusive, and it does not count as a failure (Opus R2 §7.5, Astra R2 §5).

## 2. Where the three reviewers agree

- **The idea is sound, and the project already has a small working instance.** The P3 personal layer is a readable model built one tested hypothesis at a time. Most of its candidate terms were nulls, and it is still the project's largest gain (Fable R1 §0, Opus R1 §0, Astra R1 §2).
- **Accuracy against the NN.** All three forecast that a fitted additive model lands 0 to 0.7pp from N: Fable 0.2 to 0.4pp, Opus a central 0.25pp, Astra −0.7 to +0.3pp (R1 §2.3, R1 §2.3, R1 §4).
  - The only measurement is a floor. Opus's unfitted marginal-rate additive model scored 55.00% on the four 2.55.17 builds, against N's 56.94% on an overlapping window (Opus R1 §2.1).
  - The leak-free hero-identity MLP ties the enriched MLP (57.6% vs 57.5%), which hints at little high-order signal (Fable R1 §2.1). Astra notes this does not prove the problem is additive (R2 §2).
- **The self-inclusion trap.** With hero-by-map cells, the same additive model read 57.4% in-sample but 55.0% out of time on the next builds (Opus R1 §2.1).
  - The hero-by-map cells added 1.5pp in-sample and only 0.35pp out of sample.
  - The link scale also shifted (k = 1.0 in-sample, 0.7 out of sample).
  - Astra notes the cohort also changed between the two scores, so 2.4pp is an upper bound on self-inclusion (R2 §2).
  - The lesson holds either way. A loop that scores hypotheses on statistics containing the scored games will "discover" noise in sparse cells. Section 6.3 bans this in code.
- **Power.** A term that shifts win probability by δ on a fraction f of games scores z = δ√(fn) and gains about 2fδ² in log loss (Fable R1 §3.1). So a 1pp effect on every game is worth 0.0002 and reaches z = 3.9 at 150K games.
  - Whole-model comparisons are much noisier: paired SD 0.08 to 0.144, so the minimum detectable effect is near 0.001 at 100K games (Opus R2 §1, Astra R2 §5).
  - Single pairs cannot be tested: 4,005 ally pairs at about 285 games per pair-map-tier cell gives a 3pp SE (Astra R1 §2). Counters must be pooled.
- **The theorems do not reach our effect sizes.**
  - Thresholdout's accuracy at n = 87K is about 0.019. Opus corrected his own R1 figure in R2 §1, and Astra confirmed the correction.
  - The Ladder's error at 150K is about 0.04.
  - Our effects are near 0.0002.
  - What survives from that literature is its structure: one answer per registered query, coarsened releases, budgets counted in released information, and fresh data for confirmation. With no usable bound, overfitting has to be measured with controls (all three, R2).
- **Budget sizes.**
  - Fable: about 300 bits per 150K tranche, and about 570 released answers a year (R2 §5.3).
  - Opus: 300 bits per 115K tranche, 8 bits per cell family, and a margin of √(2 ln2 · I) SE once a family has absorbed I bits, from Russo and Zou (R2 §5.3).
  - Astra: 200 scalars, 30 recipes and 200 inspected games per pilot. Astra also notes that exported coefficients and inspection packets carry millions of bits of channel capacity (R2 §5).
- **Agreed design facts.**
  - P3's personal state is built on neural residuals, so it must be rebuilt NN-free (Astra R1 §4; conceded in both other R2s).
  - The LLM proposes structure, and ordinary fitting sets the numbers.
  - Replays do not add independent games.
  - An LLM-free EBM is the mandatory comparator.
  - LLM program synthesis ranks last.

## 3. Remaining disagreements and how the pilot settles them

| # | Disagreement | Position here | Settled by |
|---|---|---|---|
| 1 | Fitted additive gap: 0.2pp (Fable), 0.25pp (Opus), up to 0.7pp (Astra) | Unknown above a 1.9pp floor | G1 on C. The EBM is the tie-breaker: if it also trails by 0.7pp, the gap is architectural. |
| 2 | Dev gate: FWER at z ≥ 3.3 (Fable), FDR at 10% (Opus), or small frozen batches (Astra) | Dev is a screen at z ≥ 3.0 plus half the information margin (5.4). Claims come only from C and F, with Holm at 0.05. | The ledger records each rule's decision on the same z. Compare false accepts per rule in the shuffled session and on canaries. |
| 3 | Worst-case information margin vs empirical calibration | Half margin on dev. C and F take no interactive queries at all. | If pooled null trials show false accepts near zero, keep half margin. Otherwise use full margin in cycle 2. |
| 4 | Astra R2 §1 read NN-free as "no LLM at all" | LLM proposers are allowed, because requirement 1 targets neural predictive guidance. | The arms, plus two LLM-free comparators (EBM, grammar enumeration), measure what the LLM adds. |
| 5 | LLM knowledge: a leak or a prior | Both. It does not bias refit values, but it inflates discovery counts. | Arm a vs b, and named vs blinded (section 8). |
| 6 | Do mechanism checks buy power? Fable 2 to 5x, Opus 3 to 5x, Astra unknown | Only if mediator SNR is at least 1.5x the outcome's | The day-2 SNR measurement, then retention of mechanism-validated vs unvalidated terms |
| 7 | Kit attributes learned (Fable, Opus) or fixed (Astra) | Fixed in the pilot | O3 and O4. Below 20% identity recovery, revisit. |
| 8 | Is 2.55.17 fresh? | Dev only; confirmation on 2.57 and later | By design |
| 9 | Download a census of new games (Opus R2 §6.4) or interesting subsets plus controls | A census is impossible: files live about 28 days, and uploads run 35K to 40K a week against a 25K quota (`training/replays/README.md`). | Settled by the measured retention and quota |
| 10 | Value of narratives | Inputs to hypothesis generation only | Yield test (9.3) |
| 11 | Additive refresh vs retrain: the same (Fable R1) or not (Opus, Astra) | Not the same: variances, link scale and the combiner need refitting | O2 backtest, refresh-only vs full refit |
| 12 | Is an additive leaf safe under MCTS by construction? | No: bonus stacking and rule cliffs are exploitable (Astra R2 §2) | O6 search-budget curves |
| 13 | Can one shuffled arm certify a low false-accept rate? | One arm cannot: 0 accepts in 30 still allows 9.5% (Astra R2 §2) | Pool the shuffled arm with null canaries, about 52 null trials in cycle 1 (40 shuffled cards plus 12 canaries; 6% upper bound if none pass) and about 100 by the end of cycle 2 (3%) |

## 4. Data and tranches

### 4.1 Tranches

Tranches are cut by build. Counts are from the research DB through 2026-10-12. Only 6,919 games predate October 2021.

| Tranche | Builds | Dates | Games | Access |
|---|---|---|---|---|
| **S** sandbox | 2.55.14 and earlier | 2019-03 to 2026-02-10 | 1,946,097 | Broker, sandbox granularity, capped and logged. Never supports a claim. |
| **Dpre** pre-cutoff dev | 2.55.15.*; 2.55.16.96846 to 96881 | 02-10 to 05-11 | 173,764 | Dev gate for arms a and c. Broker only, charged in bits. |
| **G** fit-only gap | 2.55.16.97039 | 05-11 to 07-20 | 155,191 | Harness fit jobs only (parameters, personal state). Straddles the proposers' June 2026 training cutoff. |
| **Dpost** post-cutoff dev | 2.55.17 (4 builds) | 07-20 to 09-29 | 173,394 | Dev gate for arm b, the blinded arm, and every non-arm variant |
| **M** replay dev | 2.57 games with parsed timelines (cohort B plus interesting cohorts) | from 09-28 | about 26K by Oct 26, then 8K to 15K a week | Keyed-hash split: M-explore (30%) for inspection and narratives; M-test (70%) for aggregate mechanism queries only |
| **R** sealed replays | 2.57.0.98285 to 98348 | uploads up to replay ID 65,956,438 | 5,000 files (about 4,200 with DB rows) | Fetch and checksum only. DB rows quarantined from every view, cache, history and online state update (Opus R2 §5.1, Astra R2 §5). Unlocks only by Max's recorded decision. |
| **C** pilot confirmation | 2.57.* | played 09-28 to 10-26 | about 75K in DB, about 40K after carve-outs | One frozen batch of at most 8 comparisons on day 14. No interactive access. |
| **F1, F2, …** prospective | 2.57 onward | 4-week blocks from 10-27 | about 75K raw, about 45K net each | Each block scores everything frozen before it began, once. A keyed-hash 10% of arrivals forms monthly same-era audit blocks (Astra R2 §5). |

Recent weekly volume has run 16,171 to 21,048 games. The proposers are Opus 5.5 with a June 2026 cutoff, so Dpost, C and F all fall after the cutoff. The lead's example post-cutoff builds, 2.55.17 and 2.57, are exactly Dpost and C.

### 4.2 Carve-outs from fresh data

Every 2.57 game lands in exactly one of R, M, the interesting pool, C or F. The rules that decide this never read outcomes, with one exception: selection inside the pool.

- **R** follows the frozen rule in `training/replays/cohorts/sealed_rule.json`.
- **Cohort B** follows the frozen tier-by-map stratified rule (`SELECTION_RULES.md` §B). Its games move to M.
  - B ignores outcomes, so the C that remains is skewed only in its tier-by-map mix.
  - The evaluator post-stratifies C and F to the full 2.57 tier-by-map counts and reports n_eff.
- **The interesting pool** is a keyed-hash 10% of non-sealed 2.57 games. Interesting cohorts draw only from this pool, because selecting by surprise reads outcomes (Fable R2 §6.5, Opus R2 §6.4).

**Cost.** At 8,300 a week, cohort B removes about 40% of new games from confirmation. After the pilot, `window_caps.stratified_b` in the live `cohorts.json` should drop to about 4,000 a week. That is a config value, separate from the frozen rule. Keep the higher cap only if variant M earns it.

### 4.3 What unlocks when

| Event | Condition | Release |
|---|---|---|
| Cycle close (every 2 weeks) | Budgets reset | Accepted cards become visible to every session. The cycle's dev data joins fit data. |
| Dpost retires | After cycle 2 | Becomes fit data. The next dev tranche is the oldest F block already scored. |
| C | Day-13 freeze complete: model hashes and the comparison list are in the ledger | One batch, to Max and the coordinator. Sessions see only their own accept and retain lines. |
| F_k | 4 days after the block's last game; every scored model frozen before the block began | One batch; the block then becomes fit and dev data |
| R | Max's entry in `/home/max/hots-replays/sealed/README.md` | Recommended at the end of cycle 2: four declared mechanism implications, the narrative-validity check, and draft log loss. At that point, draw R2 from later uploads by the same rule. |

## 5. The query broker and the information budget

### 5.1 Enforcement

Proposer sessions run as a separate Unix user, `hypagent`. That user has no database credentials, no label files and no SSH keys to the workers. It holds only its own worktree and the `hyp` CLI.

The broker runs as the data owner. It does six jobs:

- holds `DATABASE_URL_RESEARCH` (never Neon) and the label files;
- dispatches fit jobs;
- applies noise and coarsening;
- charges the budget;
- caches answers;
- writes the ledger.

Refused queries are logged too, because they show intent (Opus R2 §5.4).

### 5.2 Query classes

| Class | What | Where | Release | Charge |
|---|---|---|---|---|
| Q0 covariate | counts, pick and ban rates, co-occurrence, map mix | S, Dpre, Dpost, M | Exact; cells of at least 50 | 0 bits, logged |
| Q1 outcome cell | win rate, or the incumbent's mean residual, by at most 2 grouping dimensions and 20 cells | S | Cells of at least 200; rounded to 0.1pp | 1 of the 5,000 sandbox numbers |
| | | Dpre, Dpost | Cells of at least 500; Gaussian noise σ = 2·SE; rounded to 0.5pp; smaller cells refused | 0.16 bits per number (0.5·log2(1 + 1/κ²) at κ = 2), charged to each touched cell group |
| Q2 nested test | gain of a card's term over the incumbent | Dpre, Dpost | Accept or reject, plus gain rounded to 0.0001 with noise σ = SE | 1.5 bits |
| Q3 mediator cell | mean of a declared early mediator by cell | M-test | Cells of at least 200; σ = 2·SE | 0.16·ρ² bits, where ρ is the mediator-to-win correlation measured on S XP curves (default 0.5) |
| Q4 incumbent score | whole-model dev log loss, ladder style | Dpre, Dpost | Released only if it beats the best so far by 0.0005; 0.0005 bins | 3.7 bits; at most 3 per cycle |
| Q5 game record | one game's timeline or narrative | M-explore | Full record | 1 of 300 inspection slots. The game is spent and never enters a test. |
| Q6 fit job | fit on allowed data; diagnostics on S rolling origins | S, G, the dev fit side | At most 20K coefficients per snapshot (16-bit), at most 30 snapshots | Logged. Dev-fit coefficient groups are charged as Q1 releases. |

Three rules hold across all classes:

- Identical queries return the cached answer, so noise cannot be averaged away (Astra R2 §5).
- A top-k list is charged for its whole candidate universe.
- C, F and R have no query class at all. Only the evaluator touches them, and only at unlock.

**Example: "Stitches with Medivh by map."**

- **On S:** there are 9,883 same-team games over 14 maps. The per-map cells, about 700 games each, come back exact to 0.1pp and cost 14 sandbox numbers.
- **On Dpost:** there are 956 games over 12 maps, with at most 130 per map. Every per-map cell is refused. The pooled cell (SE 1.6pp) comes back with 3.2pp noise and costs 0.16 bits.
- **"Does any hero counter this duo" on Dpost** asks for 89 cells of about 100 games each. All are refused, and the broker points the session to a kit-level query instead (5 to 10 cells). Fine-grained exploration belongs on S.

### 5.3 Budgets per session per cycle

| Item | Cap |
|---|---|
| Sandbox outcome numbers | 5,000 |
| Dev bits (Q1, Q2 and Q4 on the session's dev tranche) | 120 |
| Bits per cell group (a hero, pair, map × axis, structure term) | 8; after that the group is burned on the tranche |
| Cards tested | 40, plus 6 canaries injected by the harness |
| Mediator bits (Q3) | 60 |
| Inspected games | 300, of which at most 100 narrated |

A typical cycle spends 40 Q2 tests (60 bits), about 300 noisy dev cells (48 bits) and 3 ladder reads (11 bits).

If all 120 bits were aimed at one statistic, the worst-case bias would be √(2 ln2 · 120) ≈ 12.9 SE. The 8-bit group cap holds that to 3.3 SE, or 1.7 SE at the half margin used below.

Whole-model claims across the project are limited to about 8 every four weeks, because each C or F block carries one frozen list of at most 8 comparisons. Everything else is screening. This is Fable's "query-starved by design" made concrete (R2 §5.3).

### 5.4 Dev acceptance gate

A Q2 test on cell group g passes if

> z ≥ 3.0 + 0.59·√(I_g)

- I_g is the bits charged to g on that tranche before the test.
- 0.59 is half the Russo and Zou worst case of 1.18 (Opus R2 §5.3).
- z uses the empirical paired SD, with a block bootstrap by week and by player cluster. That bootstrap builds in the design effect of about 1.7 (Opus R1 §3.1).

The gate rewards cards written from theory and the sandbox. An untouched group needs z ≥ 3.0. A group already probed for 4 bits needs z ≥ 4.2.

On Dpost, a true 1pp effect on every game has an expected z of about 3.2 after the design effect, so it passes about 60% of the time. A 1.5pp effect passes about 95% of the time. Smaller effects must be tested as pooled families: a k-parameter family is one Q2 with a k-degree-of-freedom likelihood-ratio threshold.

A rejected card returns only in a materially different form, on the next tranche. A near miss (z of 2.0 up to the threshold) buys one retry there at double charge (Fable R1 §3.4).

### 5.5 Ledger schema

The ledger is append-only JSONL owned by the broker. Each record carries `prev_hash` and `hash` (sha256 of the canonical record plus the previous hash). It is copied nightly to `/archive-backup/`. A session reads only its own released answers, through `hyp ledger`.

| Field | Content |
|---|---|
| `seq, ts, prev_hash, hash` | Hash chain |
| `session_id, arm, variant, family, card_id` | Who, and under which contract |
| `kind` | One of: query, refusal, fit, test, inspect, narrate, canary, accept, reject, freeze, unlock, budget_reset |
| `tranche, snapshot_id, code_hash` | Which data and which code |
| `template, params_json, canonical_hash` | The query; `canonical_hash` is also the cache key |
| `cells` | List of (cell_group, n_games) |
| `released, raw_ref, noise_seed` | What was shown; a pointer to the private raw store |
| `bits_charged, group_bits_after, session_bits_after` | Budget state |
| `gate_z, threshold, decision, alt_decisions` | What the FWER, FDR and full-margin rules would have decided |

Cards are hashed into the ledger before the first query that names their cell groups. That timestamp is the entire "registration". It lives in code, and no document is involved.

### 5.6 How exploration is charged

Exploration uses the same ledger and the same formulas as testing. There is no free "just looking" class.

- **On S,** exploration spends sandbox numbers. It cannot bias dev tests, because the data are disjoint. The cap exists to keep exploration hypothesis-led.
- **On dev,** exploration spends bits on the touched cell groups, which raises the bar for every later test on those groups.
- **On M,** exploration spends mediator bits or inspection slots.
- **Interesting-cohort selection** is a frozen harness rule. It charges no session, and the games it selects become M-explore.

## 6. The model language

### 6.1 Form

Every model has the form

> η(A, B) = T(A | ctx) − T(B | ctx) + X(A, B) − X(B, A),  P(A wins) = σ(η)

T is team value, X is cross-team value, and ctx is map, tier, build and date. Swapping the teams negates η, and no context effect can favor one side by itself (Astra R1 §4).

Terms come only from the term library. A new primitive is a library extension, and only the infrastructure owner can make one.

| Type | Form | Example |
|---|---|---|
| lookup | θ by categorical keys, hierarchically shrunk | hero × map class × tier |
| state | random walk across builds; per-hero jump variance from patch notes | θ(hero, build) |
| multiset | role-multiset effect, shrunk | 2 tanks + 1 healer |
| indicator | capped, monotone structural flag | no healer; no frontline |
| attribute sum | f_k(Σ_h a_hk), shape-constrained (monotone, concave, capped threshold) | diminishing waveclear |
| bilinear | A_Aᵀ W A_B over named kit axes; at most 30 nonzero entries in W | burst vs low mobility |
| product | attribute sum × context class | global presence × three-lane map |
| pair family | pooled pair residuals, shrunk to zero, entered only as a family | within-team pairs with at least 500 S games |
| decay | history aggregate with a named half-life, computed as-of | player-hero experience, rust |
| rule | at most 3 predicates; \|β\| ≤ 0.15 logit; at least 5,000 S games across at least 3 builds | no waveclear on a three-lane map |

**Caps.** At most 20K parameters in total, and at most 30 nontrivial interaction or rule terms (Astra R1 §6).

**Inputs** are limited to what is visible at draft time:

- heroes, bans and pick order;
- map, tier, build and date;
- party structure;
- as-of player histories;
- the dated kit table.

Nothing from the scored game's own replay, scoreboard, talents or parse-time MMR may enter.

### 6.2 Fitting

- **Default fitter.** Penalized MLE (L2 plus group penalties) on sparse, swap-augmented designs, solved with L-BFGS. Shrinkage variances come from empirical Bayes (Laplace). A fit takes minutes on 4 to 8 cores.
- **Bayesian backend.** Laplace or SVI, used for the state-space hero strengths and the personal block. Posterior SDs feed the lower-confidence-bound MCTS leaf (section 11).
- **EBM.** InterpretML, used for the control only.
- **Hyperparameters.** Shrinkage, half-lives, link scale and K are chosen on S rolling origins. Patch-note jump variances are set before the build's games are seen.
- **What a card buys.** An accepted card buys a form plus a refit rule. Values are always refit causally (Opus R2 §4.0). Meta facts such as "hero X is strong in 2.57" are parameter values and are never tested as hypotheses.

### 6.3 Ban on self-inclusion, enforced in code

1. **As-of contract.** Any statistic or state used for game g comes only from games that ended, and that our ingestion first saw, before g started.
   - Historical first-seen time is the game date plus one day, the P3 convention, and is labeled retrospective.
   - From now on, a daily snapshot diff records true first-seen times, because `fetched_at` is overwritten on upsert (Astra R1 §7).
2. **Fit-score disjointness.** Every fit job writes the row IDs it read for statistics, coefficients, variances, calibration and link scale. The evaluator refuses to score any row in that set.
3. **Fixtures on every library change.**
   - Delete-game invariance.
   - Late-upload invariance.
   - Rejection of planted future-scoreboard and parse-time-MMR features.
   - Swap antisymmetry to 1e-8 (Astra R1 §6).
4. **No in-sample number is ever released.** Opus's 57.4% against 55.0% is the reason.

## 7. The hypothesis lifecycle

### 7.1 Stages

1. **Proposal.** The session writes a YAML card:
   - `id`, `family` (7.3), and `tag` (structural or meta);
   - `theory`, in 2 to 5 sentences;
   - `mechanism`: the intermediate, its direction, its time window, and one negative-control mediator;
   - `expected_effect`: sign, size in pp and in log loss, and the fraction of games affected;
   - `term`, written in the library DSL;
   - `sources`: ledger entries, patch notes, kit fields or inspected games;
   - `alternatives_considered` (Fable R1 §3.5, Opus R1 §3.4).

   The card is hashed into the ledger before any query on its cell groups.
2. **Implementation.** The harness compiles the DSL and runs the fixtures. A card that fails a fixture is rejected, and the rejection is logged.
3. **Fit and invariance.** The term is fit on S plus G, plus the dev fit side when the contract allows it. Its sign must hold in at least 4 of the 6 most recent S builds, and Cochran's Q must not reject at 0.05 (Opus R2 §4).
4. **Test.** A Q2 run on the session's dev tranche, and a Q3 run on M-test if the card declares a mechanism. Safety probes (five-tank, no-healer, greedy degenerate rate) read no outcomes and are free.
5. **Gate.** A card is accepted only if all of these hold:
   - z clears the 5.4 threshold;
   - the sign is as predicted;
   - invariance holds;
   - no safety probe gets worse;
   - the complexity caps hold;
   - the mechanism passes, or the term is accepted with an `unexplained` flag.

   A size outside the predicted range counts as a calibration miss against the card. If more than a third of a session's accepts are flagged `unexplained`, the session is reviewed for leaks (Fable R2 §6.4).
6. **Ledger and incumbent.** Every stage writes a ledger record. An accepted term joins the session's incumbent, and its coefficients are cross-fit on the next tranche before any scoring (Fable R1 §3.4).

### 7.2 Who can change what

| Actor | May change | May not change |
|---|---|---|
| Variant session (`hypagent`) | Its own cards, DSL instances and notes | The harness, broker, evaluator, ledger, tranche manifests, library core, OOD suite, canaries and kit table |
| Infrastructure owner (the infra session, later the coordinator) | Library extensions (logged), bug fixes | Tranche membership after freeze; past records |
| Max | Unlocking R; future cohort caps; next-cycle budgets | Past records. A correction is always a new record. |

The evaluator directory is read-only to `hypagent`, and a pre-commit hook blocks harness edits from session worktrees. Sessions never see each other's ledgers within a cycle. The coordinator sees only accept and reject summaries until the cycle closes.

### 7.3 Hypothesis families

Each session splits its 120 dev bits and 60 mediator bits across eight families. All families express their terms through the kit axes of 10.1.

| Family | Candidate terms | Replay measurements | Mechanism test (negative control in brackets) | Dev / M bits |
|---|---|---|---|---|
| 1. Composition and strategy | Role multiset; no-healer, no-frontline and stacked-role flags; identity scores (poke, dive, sustain, push as fixed axis combinations) with diminishing returns; coherence (spread of members' identity loadings); 4×4 identity-vs-identity W | Fight count and timing; isolated backline deaths; objective contest rate; first-fort time | "Dive beats poke" predicts more early isolated deaths of the poke team's ranged heroes [minion-XP share unchanged] | 20 / 10 |
| 2. Raw hero strength | θ(hero, build) as a state-space walk; tier slope; map-class deviation; kit link θ_h = a_h·β + residual, which gives new and reworked heroes a prior. Cards test structure (jump-variance rules, pooling); values are refit. | Per-hero damage per minute, death rate and XP contribution in minutes 0 to 10 | A patch-note prior shows in matching behavior within 2K games, e.g. a damage nerf lowers damage per minute [untouched heroes] | 12 / 4 |
| 3. Solo-lane matchup | Laner vs laner on sustain, waveclear, burst and range, weighted by P(hero solos \| composition, map) and lane structure | Lane assignment from 15 s positions in minutes 1 to 5; solo laner's XP difference at 5 and 7 minutes; lane deaths; fort and keep damage; first fort in that lane | Predicts the early solo-lane XP lead and fort timing in that lane [the other side lane] | 14 / 10 |
| 4. Macro | Waveclear × lane count; global presence; merc-clear capacity × camp value; siege × map | XP by source for minutes 3 to 10; mercs captured before 10 minutes; first objective; structures lost by minute 15; rotation latency to objective spawns | Waveclear predicts a higher minion-XP share and an earlier level 10 [hero-XP share] | 14 / 10 |
| 5. Team fight | Engage (hard CC) × follow-up (burst, AoE); peel; sustain vs burst across teams; sustain threshold | Fights (2+ deaths within 15 to 20 s in one region); fight winner by death difference; fight-win rate after level 10; objective conversion within 60 s; CC followed by death within 3 s | Predicts fight-win rate and conversion, with the fight rate reported because fights are self-selected [early lane XP] | 16 / 10 |
| 6. Ganking | Mobility + burst + stealth + global vs enemy heroes with poor escapes | Isolated deaths before 10 minutes; deaths away from lanes; first-blood side and time | Predicts more early isolated enemy deaths [5v5 fight-win rate] | 10 / 6 |
| 7. Scaling | Early-vs-late axis; talent-tier and level-20 power; scaling difference × the opponent's tempo | Level lead by minute; time to levels 10 and 20; lead reversals; game length. Also the HP XP curves on S (about 180K games from 2022 and 2023) | Predicts the win rate vs game length curve, modeled as cumulative incidence of each team's wins over time (competing risks), never as win rate among games that lasted (Fable R2 §6.4, Astra R1 §5). The scaling team's share of wins rises with time, and its level lead changes sign. | 14 / 6 |
| 8. Others | Map objectives (map × axis); talent-build tendencies from past games; NN-free personalization (experience, player × hero, Kalman skill, rust, role familiarity); smurf prior; party size; counter-pick timing in the draft order | Objective events; action rates; talent pick times | Party predicts fewer isolated deaths; a smurf prior predicts extreme individual stats from game 1; a late counter-pick predicts a larger matchup term | 20 / 4 |

Personalization carries the largest known gain (+0.0149). Player skill varies with an SD of 4.3pp per slot, against draft effects of 0.5 to 1pp (Opus R1 §2.3). When bits run short, family 8 gets them first.

## 8. Controls and prior-knowledge arms

Every arm runs the kit-anchored variant K on the same harness, as its own session with its own ledger. All frozen models are scored on the same C and F games.

| Arm | Prompt | Names | Dev gate | Labels |
|---|---|---|---|---|
| **a** no restriction | Standard; free use of game knowledge | Named | Dpre | Real |
| **b** post-cutoff confirmation | Standard | Named | Dpost | Real |
| **c** prompted debiasing | Standard plus three instructions: do not recall win rates or tier lists; tag each card as coming from memory of statistics or from mechanics; reason from mechanism first | Named | Dpre | Real |
| **blinded** | Standard | Heroes H01 to H91 and maps M01 to M14 under a per-session key; roles and kit vectors kept; patch notes withheld | Dpost | Real |
| **shuffled** | Standard; the session is not told | Named | Dpost copy | Whole game records (outcome plus trajectory) permuted within (map, tier, build) blocks on S, Dpost and M |

**What each comparison measures.**

- **a vs b:** the retention gap (C gain divided by dev gain) shows how much gating on pre-cutoff data inflates acceptance. Dpre and Dpost differ in size by only 370 games, so their power matches.
- **c vs a:** what prompting alone recovers.
- **b vs blinded:** the confirmed gain owed to knowing hero and map names. If that gain is near zero, production should use the blinded arm, because it is cleaner (Opus R2 §7.5).
- **shuffled:** every acceptance is false, and accepted terms should show zero gain on real C. These trials are pooled with the null canaries.
- **Re-identification probe.** At the end of the cycle the blinded session is asked to name 20 H-labels. If it names more than half, the kit vectors have leaked identity and the blinded arm measures little.

**Canaries.** Each session gets 6 harness-written cards per cycle, indistinguishable from its own:

- 4 null canaries: terms built on a keyed hash of replay_id, or a kit axis permuted within role;
- 2 known-true canaries: no-healer penalty variants that are stable across S builds.

Canaries measure false accepts and power continuously (Opus R2 §5.5).

## 9. Replay machinery

### 9.1 Extraction targets

What already exists:

- the downloader, with frozen cohorts and the sealed set (commits 5839037 and 6f2b483);
- `parse_replay.py` (commit 99a7c44), which produces about 14 KB of gzipped timeline per game in about 1.5 s on 3090 cores 24 to 35, validated against the DB.

Files expire after about 28 days, so nothing uploaded before 2026-09-15 can be fetched. Historical mechanism data is therefore limited to the HP per-minute XP curves stored for about 10% of drafts: 12% of 2022 games, 41% of 2023, about 1% of 2024 and 2025, and about 10% of 2026 (Fable R2 §2.1, Opus R2 §2.1).

The replay feature store turns timelines into one row per game and team, with deterministic, versioned code:

- **Economy:** level lead at minutes 5, 7, 10 and 15; time to levels 10 and 20; XP by source to minute 10; lead reversals.
- **Deaths:** first blood; deaths before 10 minutes; isolated deaths; deaths by map region and by solo lane.
- **Fights:** death clusters; fight winner; fight-win rate before and after level 10; conversion to an objective or structure within 60 to 90 s.
- **Structures, mercs and objectives:** first fort by lane, and structures lost by minute 15; merc and objective events normalized to (instance, team, time, contested), with a per-map dictionary validated by map family.
- **Lanes:** assignment from 15 s positions. Interpolated positions are flagged and never treated as observed.
- **Talents and game events:** talent pick times; CC followed by a death within 3 s; action rates. Action rates feed only the personal priors of later games.
- **End-of-game scoreboard:** kept as description only, never as a mediator.

Per-hero aggregates computed on M, such as early XP contribution, may validate kit axes. They may enter later games' features as-of, and never the same game's.

### 9.2 Mechanism-consistency tests

- **Declaration first.** The card declares its positive mediators (direction, time window, minimum standardized effect), at least one negative control, and the outcome sign.
- **Stage 1.** A draft-to-mediator model is fit on M-test from draft-time inputs only. Evidence across mediators is combined as an e-value product, the POPPER pattern (Opus R1 §1.3). The negative controls must not reject.
- **Forbidden:**
  - conditioning on game length or late state;
  - testing among winners only;
  - controlling for the realized lead when estimating draft value;
  - using end-of-game totals as mediators (all three, R2).
- **Day-2 SNR check.** For 10 known structural terms (no healer, no frontline, double support, two globals and others), compute z_mediator / z_outcome on the S XP curves and on parsed cohort-B games.
  - If the ratio is at least 1.5 for at least 5 of the 10 terms, build variant M.
  - Otherwise mechanisms become narrative-only, and the M budget drops to 20 bits (Opus R2 §6.3).
- **No assumed gain.** The false-accept reduction from mechanism tests is whatever the null canaries show. We assume no multiplication of error rates (Astra R2 §2).

### 9.3 Narrative layer

The narrative pipeline has four steps.

1. Build a deterministic timeline with event IDs.
2. Fit an NN-free in-game WP: an additive GAM over minute, with level lead, XP lead by source, structure difference, objective state and recent deaths, using the pre-game WP as an offset. It is fit on S XP curves and M-explore, and scored on M-test.
3. Rank episodes by the change in in-game WP over 60 s, excluding core kills. These are labeled "largest predicted swings", with no claim about cause.
4. The LLM writes 3 to 6 sentences from the selected events only.
   - Every factual clause cites an event ID.
   - Interpretation is kept separate from observation.
   - Summaries stop at fixed cutoffs (minute 5 or 10), never at a "first half" of unknown length (Astra R2 §6).

| Check | Target |
|---|---|
| Faithfulness, checked mechanically per clause | At least 98% of clauses supported; no invented heroes or outcomes |
| Stability: top swing across two runs | Within 60 s in at least 80% of games |
| Human check (Max or one strong player, 30 games, once) | At least 70% of top-3 episodes accepted |
| Hypothesis yield (the decisive test) | Narrative-cited cards confirm at 1.5x or more the rate of table-only cards; otherwise cut the narrative step |

Narratives are capped at 100 games per session per cycle, and the corpus is never narrated in full.

### 9.4 Interesting subsets with controls

1. **Pool only.** Interesting cohorts are drawn only from the 10% pool, never from C, F or R.
2. **Frozen surprise model.** A frozen NN-free model, A0 trained through 2.55.17, scores surprise, and its version is logged per game. Trajectory surprise (the draft prior disagrees with the HP XP curve at minute 10) is preferred over outcome surprise, which carries only about 0.1 bit about mechanism (Fable R2 §6.5).
3. **Deadline.** The rule is written into `SELECTION_RULES.md` §C by day 4, before the first fetch. The first 2.57 files expire from about Oct 27.
4. **Two-phase sampling.** Inclusion probability is π_i = 0.2 + 0.4·1[top surprise quintile], drawn independently and recorded. Cohort B supplies stratified random controls at known within-cell rates. Population estimates use Horvitz-Thompson weights (1/π) and report n_eff (Opus R2 §6.4, Astra R2 §6).
5. **Use of interesting games.** They go to M-explore for reading only, and no effect is estimated on them without weights. A mechanism found among surprising games must replicate on the M-test controls.
6. **Target population.** Download and parse failures are logged by stratum. The stated target is "downloadable Heroes Profile Storm League games". The DB holds about 84% of listed games, and M and C inherit that coverage (`SELECTION_RULES.md` §A and §B).

## 10. Variants to run in parallel

### 10.1 Shared kit table

The kit table, `kit_<patch>.json`, is dated and hashed. It has 12 axes:

- hard CC; soft CC and peel;
- healing throughput; self-sustain and shields;
- burst; sustained damage; range;
- mobility; waveclear; siege; global presence;
- early-vs-late power.

The team identities (poke, dive, sustain, push) are fixed combinations of these axes.

How the table is built and checked:

- **Source.** Ability and talent text from the pinned heroes-talents reference (`training/replays/ref/heroes_talents_ref.json`, current to 2.55.3), plus patch-note deltas through 2.57, which covers Xal'atath.
- **Coding.** Hero names are stripped from the text. Three independent LLM codings are averaged under a rubric that forbids any judgment of strength or win rate. Max spot-checks 15 heroes.
- **Fixed during fits.** Kit values do not move when models are fit in the pilot.
- **Behavioral check.** Each axis must predict its signature on M. For example, waveclear must predict minion-XP share, and hard CC must predict CC-then-death events. An axis that fails is renamed "unnamed factor k" (Fable R2 §4, Opus R2 §4).
- **Permutation check.** Each axis must beat 95% of within-role permutations of its values (Opus R1 §4).

### 10.2 Variants

**K. Kit-anchored attribute model.** Wave 1. Ranked first by Opus R2, second by Fable and Astra.

- **Form:**

  > η = Σ state-space hero strength (kit-linked prior) + w·(A_A − A_B) + Σ_k [f_k(A_Ak) − f_k(A_Bk)] + A_Aᵀ W A_B − A_Bᵀ W A_A + map class × axis + role structure + NN-free personal block

  - A = Σ a_h over the 12 axes.
  - About 200 structural parameters are fitted beyond the hero states and the personal block.
- **Hypothesis space:** all eight families, expressed through the axes. A card can add a W entry with its sign, set the shape of an f_k, add a map × axis product, add a coherence term, define a solo-lane matchup on the axes, or set a patch-note jump rule.
- **Budget:** standard, per arm (section 8).
- **Expected:**
  - O1 ties A within ±0.0005;
  - O3 recovers 30 to 60% of identity value (Opus R2 §4);
  - O4 gains +0.001 to +0.006 on Xal'atath's first 2K games, shrinking as her data accrues (Fable R2 §4, Opus R2 §4);
  - O5 slope 0.9 to 1.1;
  - five-tank probe at or below 0.20.

**A. State-space hierarchical additive logit.** A0 is built by infra in week 1, and an LLM session joins in wave 2. Ranked first by Fable and Astra, second by Opus.

- **Form:** θ(hero, build) as a random walk with patch-note jump variance; a tier offset; hero × map class, shrunk; monotone role structure; pooled pair families; draft order; party; account status; the NN-free personal block. 5K to 20K parameters.
- **Hypothesis space:** families 2, 1 (structure only), 7 (half-lives) and 8. No attribute terms.
- **Budget:** standard, arm b.
- **Expected:**
  - −0.5 to 0.0pp vs N in steady state, and −0.2 to +0.3pp on 2.57, where N is stale;
  - +0 to +0.002 on O2 with patch priors;
  - O5 slope 0.85 to 1.05;
  - no O3 recovery by design (Opus R2 §4).

**M. Replay mechanism model.** Wave 2, and only if the SNR check passes. All three reviewers rank it third.

- **Form:** K or A with multi-task heads. The loss is L_win + Σλ_k L_k over 4 declared early mediators, with fixed λ:
  - level lead at 10 minutes;
  - first to level 10;
  - structures lost by 10 minutes;
  - mercs captured by 10 minutes.

  Inference uses draft inputs only, and a direct outcome head is kept (Astra R2 §4).
- **Hypothesis space:** families 3 to 7.
- **Budget:** 120 dev bits and 100 mediator bits.
- **Expected:** no draft-time accuracy gain. The falsifiable claim is that mechanism-validated terms retain 0.7 to 0.9 of their dev gain, against 0.4 to 0.7 for unvalidated terms, and that patch effects appear in mediators within 1K to 2K games, against 5K to 10K for win rate (Opus R2 §4).

**R. Support-constrained rule layer.** Wave 3.

- **Form:** K or A plus at most 20 rules (at most 3 predicates each, capped at 0.15 logit, at least 5,000 S games across 3 or more builds). Rules come from three sources, competing in one L1 fit:
  - LLM cards;
  - shallow trees fit on the backbone's own S residuals;
  - classical grammar enumeration with no LLM, which serves as Astra's LLM-free symbolic baseline.
- **Budget:** 40 dev bits (at most 2 bundles), 20 mediator bits.
- **Expected:** a cost of 0 to 0.002 log loss (Astra R2 §4), the best O6 behavior, and plain-language draft advice.

**E. EBM/GA2M, LLM-free control.** Built by infra in week 1.

- **Form:** InterpretML on hero difference indicators (91), map, tier, role counts, kit sums and NN-free personal summaries, with at most 30 FAST pairs, antisymmetrized. The configuration is frozen on S before any dev query.
- **Budget:** no proposer queries; scored only in the C and F batches.
- **Expected:** within 0.3pp of A in steady state, and 0.001 to 0.003 worse after patches (Opus R2 §4).
- **Role:** if the LLM arms cannot beat E, the LLM's job shrinks to documentation and patch priors.

**N. Neural contrast.** The one NN in the system.

- **Form:** the frozen drift-aware MLP plus P3.
- **Use:** scored on the C and F batches and the OOD suite. Results go to Max and the coordinator at cycle close, and never into any proposal context, residual table or selection rule.
- **Expected:** 56.0 to 56.8% on 2.57; O5 slope 0.7 to 0.9; exploitable without the paper-1 mask (Opus R2 §4).

**Dropped.** LLM program synthesis is dropped: all three reviewers rank it last, because its search pressure on a noisy evaluator is the largest. Mixture-of-experts survives only as a card type inside A.

### 10.3 Session organization and order

Sessions are organized by **variant × arm**, with families as budget lines inside each session. With only 2 or 3 sessions, family-split sessions would share one incumbent model and one set of cell-group budgets. They would collide on the kit axes (team fight and composition both touch CC and burst) and contaminate each other's information accounting. A self-contained ledger per variant session is what keeps the arms comparable.

| Wave | Days | Parallel sessions | Question |
|---|---|---|---|
| 0 | 1 to 6 | Infra (one session with subagents). The harness fits A0, K0 and E and scores N on the remote workers. | Is the harness sound? |
| 1 | 7 to 11 | K-b, K-blinded, K-shuffled | Does the loop add confirmed gain over K0 and E at a certified false-accept rate, and how much of it comes from name knowledge? |
| 2 | 15 to 25 | K-a, K-c, A-b, and M-b if SNR passes | Prior-knowledge arms, the backbone loop, mechanisms |
| 3 | from 29 | The winning variant as **variant × family group** (3 sessions: {composition, team fight, ganking}; {strength, solo lane, scaling}; {macro, maps, others}), plus R | Throughput, once the false-accept rate is known |

In wave 3 the family-group sessions share one incumbent through a harness merge queue. Acceptances are serialized and rescored against the merged model, and the cell groups are split so that no two sessions touch the same group.

## 11. Shared infrastructure to build first

Effort is given in focused days of one Opus 5.5 session. Heavy compute runs on 3090 cores 24 to 35 and on the 5090's spare CPU threads, at nice 19, with resumable checkpoints every 10 minutes or less. The main box only answers broker aggregates and runs git.

| # | Component | Interface | Days |
|---|---|---|---|
| 1 | Snapshot and loaders | `hyp_data.snapshot(as_of)` writes parquet per tranche from the research role. Labels are stored apart and readable only by the broker. A daily append assigns tranches and records first-seen times. | 1.5 |
| 2 | Tranche manager | `tranche(replay_id)` returns S, Dpre, G, Dpost, M, pool, R, C, F_k or audit. Keyed hashes stay with the broker. R and the pool are quarantined everywhere. | 0.5 |
| 3 | Broker and CLI | `hyp query`, `hyp card submit`, `hyp test`, `hyp inspect`. Implements Q0 to Q6, noise, caching, refusals and budgets, over a Unix socket. | 2 |
| 4 | Ledger and card registry | Hash-chained JSONL; `hyp ledger --mine`; nightly NAS copy; reconciliation | 0.5 |
| 5 | As-of feature store | `features(game_ids, as_of_rule)` returns a sparse matrix. Carries the 6.3 fixtures. Reuses the P3 and paper-1 builders under one contract. | 1.5 |
| 6 | Term library and fitters | DSL compiler; penalized MLE with empirical Bayes; Laplace/SVI; EBM wrapper. Every fit writes its row-ID read set. | 2 |
| 7 | NN-free personal block | The P3 estimator rerun on A0 residuals (10 to 30 core-hours) | 1.5 |
| 8 | Kit table | `kit_<patch>.json` plus the validation hooks of 10.1 | 1 |
| 9 | Evaluator | Paired LL, accuracy, slope; block bootstrap; the 5.4 gate; O1 to O6; post-stratification; canaries; a C/F/R batch runner that refuses unfrozen models | 2 |
| 10 | Remote dispatch | Built on `training/remote_workers/hotsjob.py`. `hyp_job.submit(spec)` is resumable and pinned with `taskset 24-35`. Ships label-free extracts plus a broker label token; dev scoring jobs return only released numbers. | 1 |
| 11 | MCTS-leaf adapter | `leaf.evaluate(drafts[N,10], map, tier, build)` returns logit, sd and per-term contributions. Exports `leafpack.npz` and `leafpack.json` (term IDs, card hashes). Includes a Python reference and a CUDA additive kernel beside `training/cuda_mcts/device_forward.cuh`. The leaf value is σ(η − κ·sd), with κ set on S. Also computes O6 optimism against the judge. | 2 |
| 12 | Replay feature store | `replay_features(game_ids)` builds the mediator table from `timelines/*.json.gz`: fights, isolated deaths, lanes, XP sources, objective dictionary. Checked with `validate_timelines.py`. | 2 |
| 13 | Session kit | Arm prompts, blinding relabeler, permuted-copy builder, card template, re-identification probe | 1 |
| | **Total** | About 19 serial days, or about 6 days on the critical path with subagents | |

**Cycle-1 compute.**

| Job | Core-hours |
|---|---|
| A0, K0 and E fits, plus S backtests | 5 to 15 |
| Personal rebuild | 10 to 30 |
| Three sessions × 46 cards × about 3 fits each | 30 to 60 |
| Parsing about 25K replays | 10 |

This fits easily on 12 cores over two weeks. The only GPU work is the O6 MCTS runs, which use the existing kernel on the 5090.

## 12. Two-week pilot schedule and decision points

Day 1 is Wednesday, 2026-10-14. Tennis borrows part of the 3090 until about Oct 18, and HotS keeps cores 24 to 35.

| Day | Date | Work |
|---|---|---|
| 1 | Oct 14 | Snapshot as of Oct 13; tranche manifest; R and pool quarantine; broker skeleton; 100-file parse benchmark |
| 2 | Oct 15 | Broker Q0 to Q4, ledger, fixtures. Mediator SNR check on 10 structural terms (charged to the harness). |
| 3 | Oct 16 | Feature store, term library, fitter. A0 fit (remote). Kit coding. |
| 4 | Oct 17 | Personal rebuild (remote); E and K0 fits; the trajectory-surprise rule written into `SELECTION_RULES.md` §C |
| 5 | Oct 18 | Evaluator, OOD suite, safety probes, leaf adapter. N scored into the sealed results store. Interesting-cohort downloads start, ahead of the Oct 27 expiry. |
| 6 | Oct 19 | Session kit, canaries, permuted copies; a dry-run card end to end; S backtests (O2, O3, O5) of A0, K0 and E |
| **D1** | Oct 19 | **Decision 1.** Start wave 1 if G6 and G7 pass and A0 and K0 land within 0.5pp of E in the S backtests. The SNR result decides whether M joins wave 2. |
| 7 to 11 | Oct 20 to 24 | Wave 1: K-b, K-blinded and K-shuffled, each with up to 40 cards plus 6 canaries. Infra handles library extensions. 100 narratives with faithfulness and stability checks. |
| 12 | Oct 25 | Freeze: model hashes; predicted C gains (dev gain shrunk by the running optimism); the 8 C comparisons, all written to the ledger |
| 13 | Oct 26 | Last C games played. Ledger reconciled, canaries tallied, re-identification probe run. |
| **D2** | Oct 27 to 28 | **C batch** once uploads catch up. **Go/no-go** by the 1.3 table. Set the cohort-B cap. Choose half or full margin for cycle 2. |

**The eight C comparisons** are tested with Holm at familywise 0.05:

1. K-b vs N on O1.
2. K-b vs E on O1.
3. K-b vs K0 on O1 (the loop's yield).
4. K-blinded vs K0.
5. Best NN-free model vs N on O4.
6. Best NN-free model vs N on O5.
7. Best NN-free model vs N on the 2.57 onset (O2).
8. K-shuffled vs K0. Any gain here is a harness failure.

O3 and O6 come from backtests and simulation and are reported alongside.

## 13. Risks and open questions

1. **Replay cohorts eat confirmation data.** At the current cap, cohort B removes about 40% of new games from C and F. Lowering it to about 4K a week returns most of them. This is Max's call.
2. **C is small.** It can show a 0.003 shortfall, and nothing finer. Tie claims wait for F1 (Oct 27 to Nov 23). If 2.57 volume fades after launch, the F blocks get longer.
3. **The kit table may carry LLM priors.** Hero-blind coding, fixed values, behavioral validation and the permutation test all narrow this channel, but none closes it. The blinded arm sees kit vectors, so it may re-identify heroes; the probe measures how much.
4. **The NN-free personal rebuild may lose part of the +0.0149.** Below +0.0100, the personalized G1 is at risk whatever the draft model does.
5. **Mediator SNR is unmeasured.** If it is low, the replay-heavy families (solo lane, macro, team fight, ganking) fall back to outcome tests with a 1pp floor.
6. **DB coverage is about 84% of listed games.** If the missing 16% is not random, C, F and M share the bias. Check listing marginals at each unlock.
7. **Lane assignment and fight detection are new code on sparse positions** (median 15 s). Validate both on 50 games by hand before any family-3 or family-5 test counts.
8. **The half-margin rule is a judgment call.** The roughly 52 pooled null trials in cycle 1 decide it for cycle 2.
9. **Isolation depends on the Unix user split.** If sessions run as `max`, nothing stops them from reading labels. The `hypagent` user and the read-only harness are the real enforcement.
10. **Drift confounds retention.** Same-era audit blocks and rescoring on every F block separate drift from overfitting roughly, but never exactly (Astra R2 §5).
11. **Readability does not make search safe.** O6 must be measured. Prospective policy evaluation on real drafts is outside this spec.
12. **Open: the drift paper's registered 2.57 tests** also use 2.57 outcomes. Confirm with that lane that C's frozen batch does not interfere.
13. **Open: what "fully understandable" means.** Opus proposes a concrete check (R2 §8): a reviewer reading only the term cards predicts the sign of the model's change for 10 patched heroes, with a target of 8 of 10. Adopt it at the end of cycle 2.
