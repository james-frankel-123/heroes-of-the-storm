# Paper 1 (Diagnosing and Repairing OOD Failure in MOBA Draft Policies) — summary for advisors

Max, Ernest, James, Hod — dense informal digest, same format as
paper/drift/COAUTHOR_SUMMARY.md. **Status: submission-ready for IEEE ToG**
(11 pages, last half-page is references, + 5-page supplement,
double-anonymous variant via a toggle). **Submission proceeds WITHOUT the
expert study (Max, 2026-07-24)**: the survey timeline slipped, the paper's
Limitations already frame it as an in-progress validation tier, and the
study stays prepped in the background for a reviewer request. In its
place, the out-of-family Quick Match judges are now IN the paper
(2026-07-24): a passage in the tournament section (2021-era and
current-era QM judges, disjoint draft-free corpus, both reproduce the
consensus ordering at Spearman 0.92; degenerate probes scored losing),
one clause in the intro's not-self-graded sentence, and the Limitations
now cite it as the out-of-corpus bound alongside the 20-model ensemble's
in-corpus bound. Manuscript: `paper 1/overleaf/draft.tex`; all experiments
regenerable from `training/rerun2026/` (MANIFEST.md). Open source:
github.com/maxsegan/adversarial-combinatorial-assembly (curated export; the
paper cites an anonymized mirror — repo rename pending, see title note).
Deployed agent: hotsfever.com/draft.
(Original early-CQL working notes preserved in
notes_archive_early_cql_2026-06.md — fun archaeology: hypothesis H1 there,
"CQL is just expensive GD," became the paper's core finding.)

**Pre-submission pass (2026-07-24, advisor round):** abstract cut 250 →
181 words (cap 200). Revision-growth plan, decided now: if the expert
study lands at revision, its results go to the SUPPLEMENT with a
3-sentence main-text summary, and the main-text cut (if a page must be
found) is the 27-config synthetic-sweep scope paragraph (its conclusion,
"unseen-only augmentation is the active ingredient," survives in one
line; details move to supplement). External-validity number (11,034
coordinated-league drafts, 58.1→55.5%) already surfaces in main-text
Limitations. Anonymous mirror + GitHub both verified resolving
(HTTP 200); Max should confirm the 4open.science expiry is set to
"forever" in its settings since review can run many months.

**Title note (2026-07-24):** retitled from "Adversarial Combinatorial
Assembly: ..." — the research program reframed from a branded problem
class to four data-regime conditions (learned-proxy judge, biased
behavior policy, uncoverable action space, changing game). The paper now
describes the draft structure plainly and never coins ACA; the capstone
paper owns the program framing.

## The problem structure and the thesis

**The structure (formerly branded "ACA"):** rivals alternately claim
rivalrous elements from a shared pool; set value is dominated by
interactions within and against the opposing set; outcomes come from a
noisy downstream process evaluated only by a learned proxy. MOBA drafting
is the instance: 90 heroes, 7.2×10¹⁴ matchups, our corpus of 1,949,087
ranked drafts (pinned May-2026 snapshot, patch 2.55) covers a vanishing,
clustered fraction — OOD behavior dominates policy quality.

Thesis in one line: *search exploits learned value functions' OOD optimism
(assembling structurally broken teams rated as competitive), pessimistic
offline RL does not repair this — it collapses to behavioral anchoring —
and the repair that works is layered and domain-structured: enriched
features, targeted synthetic data, MCTS self-play, and structural
constraint masking.*

## Headline

**Constrained MCTS wins all 20 of its tournament matchups at 0.670
consensus win probability** (four evaluators; margin stable under each
individually: 0.657–0.678), is structurally valid *by construction* (the
mask), and preserves the interaction-awareness that behavioral anchoring
destroys (synergy +0.95). Degenerate compositions: 64% under naive
value-guided search → 8% with the layered repair → 0 by rule under the
mask.

## Core findings

1. **Accuracy doesn't discriminate; policy does.** All WP model variants
   sit at 57.7–58.1% accuracy (0.3pp spread); a 4,100-config feature sweep
   shows no group's marginal accuracy exceeds 0.2pp — per-hero WR is
   actually *negative*. Yet the same features move the policy massively:
   5-tank WP 0.365 (naive) → 0.155 (enriched) → 0.062 (augmented), and
   greedy degenerate rates separate by double digits. This
   accuracy-vs-policy disconnect is the recurring motif of the series
   (the drift paper reproduces it on the time axis).
2. **Anchoring collapse is a property of the problem class.** With
   terminal-only rewards, Monte-Carlo returns leave pessimism exactly one
   pathway: anchor to the behavior policy. CQL collapses to behavioral
   cloning — from hero-identity inputs it *appears* to subsume the whole
   feature/augmentation pipeline (99.8% healer at α=2), but it is
   imitating humans, not evaluating compositions. IQL's mechanism is the
   cleanest: terminal-only MC returns → advantages ≈ 0 → AWR weights
   0.998–0.999 → literal BC. MCQ collapses to a near-deterministic
   ten-hero policy. All three families anchor, and anchoring destroys
   interaction-awareness.
3. **The decomposition: composition safety vs interaction-awareness.**
   Safety = avoiding structurally broken teams; interaction-awareness =
   responding to synergies/counters vs the opposing set. Anchored methods
   are safe but blind (they inherit human comp discipline, ~zero synergy
   exploitation); naive value search is interaction-aware but unsafe. The
   metric suite measures both; the final agent is the first method strong
   on both axes at once.
4. **The layered repair, and what each layer is for.** Enriched domain
   features give the value function vocabulary for compositional quality
   (+0.023 WP under deep search, p<0.001 — down from +0.070 at 473K
   replays: data scale closes the *value* gap, not the OOD gap; features
   stay critical for shallow search and OOD behavior). Targeted synthetic
   data (34K records, unseen role-comps only, WR=10%) repairs *greedy*
   deployment (degen 22→16%) but is **not in the final agent** — under
   MCTS it is partially redundant and slightly costly (self-play learns
   implicit composition regularization). MCTS self-play converts the VF
   into a policy that is safe and interaction-aware simultaneously.
   Structural constraint masking (reachability-based role masking)
   removes residual invalid picks by rule without anchoring — the
   constrained agent beats its own unconstrained variant head-to-head.
5. **Tournament (the adjudication).** 11 strategies, all 110 ordered pairs
   × 200 drafts, scored by all four WP evaluators + the full metric
   suite. Standings: constrained MCTS 0.670 (20/20) > MCTS 0.658 >
   constrained greedy ≈ enriched greedy 0.595 > enriched+augmented 0.580
   > Gourdeau estimator 0.519 > CQL variants ≈ Gourdeau discriminator ≈
   GD (0.404–0.420) > MCQ 0.225. The two external reimplementations land
   **by mechanism, not authorship**: the estimator — the only baseline
   that searches a value function — finishes above 0.5, ahead of every
   anchored method; the discriminator falls inside the anchored cluster.
6. **Systems.** Fused CUDA MCTS at ~150 episodes/s/GPU — 2.2× a *tuned
   batched-evaluation* host pipeline at equal search quality (the honest
   apples-to-apples; the eye-catching 117× is vs the Python prototype and
   is labeled as such) — and the complete agent (search + policy + value
   + opponent models) runs in-browser at interactive latency in ~20MB.

## Why the tournament verdict is trustworthy (the circularity portfolio)

The predictable attack: "your own models judged your own agents." The
defense is layered; each item answers a different version of it:

- **Feature-sharing:** the naive and hero-strength evaluators share no
  features with the agents beyond hero identity; the margin is stable
  under each evaluator alone.
- **Corpus-sharing (strongest recent addition):** a WP model trained on
  91K **Quick Match** games — different mode, different composition
  distribution (degenerate comps are in-distribution there), different
  era, hero-identity features only — reproduces the tournament ordering
  at **ρ = 0.917**, constrained MCTS first, anchored cluster at the
  bottom. Scope note learned in the drift paper: era-mismatched judges
  are valid for composition-structural contrasts like these (era-stable)
  but not for meta-timing contrasts — measured, not assumed.
- **Ground truth where it exists:** the agent's favorite cores are
  real-outcome validated (Hogger+Samuro pairs/trios win +3.9pp, n=2,065,
  significant); 99.63% of agent comps use role structures that real
  winning teams use; trio-level empirical analysis backs the synergy
  metric.
- **Epistemic honesty:** a 20-model ensemble (seed / architecture /
  feature-subset / data-half diversity) shows tournament verdicts are
  issued in the models' *most-trusted* region — agent drafts sit at
  1.16–2.21× human ensemble variance vs 3.2–3.7× for degenerate probes,
  and the anchored-vs-non stratum (the headline contrast) is the *least*
  OOD.
- **Robustness:** dual-snapshot stability; NGS organized-league data
  (11,034 games: WP 58.1→55.5, degradation localized to the
  least-ladder-like quartile; coordinated play is *more*
  meta-concentrated, not less).
- **The one remaining epistemic category — human judgment — is the expert
  study** (next section): every current validator learns from win/loss
  outcomes of largely uncoordinated play; only experts judge drafts *as
  drafts*.

## The expert rating study (in flight; the last edit before submission)

Preregistered, blinded, paid instrument at hotsfever.com/rate (v4 after
two adversarial design reviews): **14 raters × a fixed 240-item set**
($100 on completion, performance-independent) — 8 hidden screener items
interleaved among 48 opening positions (inclusion gate ≥7/8; random
passes 3.5%), 60 machine-pair items each (all 280 tournament pairs
covered ×3 raters via Latin square), 129 real-game anchors (all
post-training-snapshot, so no anchor exists in any model's training
data), 3 late-session catch items at fixed positions. Confirmatory
endpoints pre-registered with power analysis (primary: expert–model
agreement on 255 effective machine pairs, worst-case z ≈ 3.2 at true
0.60), an OOD moderation test with frozen ensemble-variance covariates, a
named anchored-stratum endpoint, and a **pre-committed interpretation
map** — every outcome, including experts disagreeing with the models, has
a written consequence for the paper, so the study cannot "fail."
Sequence: recruiting + the IRB exempt determination (Hod) → OSF
registration (the binding timestamp) → invites. The instrument is live
and end-to-end verified; the prereg text is frozen pending those steps.

## What we ran (infrastructure, for the record)

Everything regenerated on the pinned snapshot via `training/rerun2026/`
(phases 0–4 + tournament, zero failures): the 4,100-config feature sweep;
the model zoo; CQL α-sweep + enriched-CQL + MCQ + IQL + BC; GD pools; the
MCTS config grid (15 configs × 15 seeds; headline config J_800sim); the
110-pair round-robin; constrained-search variants; ensemble uncertainty
(incl. the all-11-strategies table); ground-truth favorites / role-comp /
trio validation; the NGS study; the QM out-of-family evaluator
(training/qm2026/); throughput apples-to-apples. A 2026-era QM judge
arrives ~2026-07-21 (recent-era QM fetch completing now); we will run
the tournament rescore when the data lands. Methodology lessons kept
honest in the paper: the original relational-vs-absolute ablation via
inference-time weight zeroing was invalid (co-adapted networks) and was
redone with genuinely retrained variants (M2 0.719 vs N2 0.714, ns); a
historical virtual-loss bug and a production 1-ply search bug were found
and fixed during the rerun.

## Status / asks

- **Hod:** the IRB exempt-determination question is the one blocker on
  the survey chain (expert adults rating anonymized game artifacts, $100
  compensation — textbook exemption, but it cannot be granted
  retroactively, and IEEE may ask).
- Max recruiting 14 raters — low-rank-experienced raters especially (the
  pre-registered skill contrast is confirmatory only with ≥3 of them,
  designated in writing before invites go out).
- After survey results land: insert (one section + supplement), submit to
  ToG, then the non-anonymous arXiv version.
- Series context in research_focus_v2.md: drift (paper 2 — experiments
  complete, summary in paper/drift/COAUTHOR_SUMMARY.md, writing starting),
  personalization (paper 3 — proposal with Ernest,
  paper/personalization/PROPOSAL.md), talents (paper 4 — spec'd,
  paper/talents/SUMMARY.md), transfer (capstone).
