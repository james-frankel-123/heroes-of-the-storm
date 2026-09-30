# Rating study: statistical power for per-rater accuracy (calibration + extended anchor arms)

Scope: this note quantifies what the **known-outcome ladder anchors** buy us —
i.e. our ability to estimate, per rater and per skill tier, how often an expert
picks the team that actually won the real game. Anchors are the only items with
a ground-truth label, so they (not the machine-pair items) drive accuracy power.
Counts below match the frozen seed-20260710 pool (`data/rating-items.json`):
40 shared calibration anchors + 100 core items (45 anchors) + 700 extended
items (555 anchors).

## Anchor supply per rater

| Arm | Anchors | Per tier (low/mid/high) |
| --- | --- | --- |
| Calibration (shared, rated by every rater first) | 40 | 14 / 13 / 13 |
| Core (latin square, 30 items/rater) | 45 total in pool; 9–19 land in any one slot's 30 | 15 / 15 / 15 in pool |
| Extended (uncapped) | 555 distinct (185/185/185) | 185 / tier |
| **Max distinct anchors one rater can reach** | **up to 614** (40 + up to 19 + 555) | **~205 / tier** |

A rater who exhausts the extended arm sees every extended anchor plus the
calibration 40 and their slot's core anchors. The extended machine-pair items
(145) carry no ground truth and are used for expert-vs-model agreement, not
accuracy. Pool totals: 640 anchors (214 low / 213 mid / 213 high).

## The shared calibration block (n = 40 per rater)

All raters rate the identical 40 anchors before their core 30, so per-rater
accuracy on this block is directly comparable across raters (no
item-assignment confounding). It doubles as the pre-registered quality gate:
**exclude a rater whose calibration accuracy is ≤ 50% (≤ 20/40)**. Exact
binomial operating characteristics:

| True accuracy | P(pass gate, ≥ 21/40) |
| --- | --- |
| 0.50 (coin flipper) | 0.437 (**excluded w.p. 0.563**) |
| 0.60 (marginal honest rater) | **0.870** |
| 0.65 | 0.964 |
| 0.70 | 0.994 |

At n = 40 the calibration block is a gate and comparability anchor, not a
precise estimate: 95% CI half-width at p = 0.5 is ±15.5 pp (see table below).

## Per-rater accuracy CI vs. number of anchors rated

95% CI half-width for a single accuracy proportion, worst case p = 0.5 (widest);
Wald `1.96·sqrt(p(1-p)/n)`. Wilson intervals, which we use in practice, are
slightly tighter and better-behaved near 0/1.

| n anchors | ±half-width @ p=0.5 | @ p=0.65 | @ p=0.75 |
| --- | --- | --- | --- |
| 40 (calibration only) | ±15.5 pp | ±14.8 pp | ±13.4 pp |
| 100 | ±9.8 pp | ±9.3 pp | ±8.5 pp |
| 250 | ±6.2 pp | ±5.9 pp | ±5.4 pp |
| 500 | ±4.4 pp | ±4.2 pp | ±3.8 pp |

Headline resolutions: **≈ ±16 / ±10 / ±6 / ±4.4 pp at n = 40 / 100 / 250 /
500**. With the expanded pool a single high-volume rater can now reach n = 614
distinct anchors, so n = 500 is attainable without pooling across raters.

## Primary-endpoint precision (machine-pair agreement)

The confirmatory endpoint pools core machine-pair judgments: 55 pairs × 3
ratings, minus near-ties at |consensus − 0.5| ≤ 0.02 (8 pairs in the frozen
pool) → **47 items / 141 planned judgments**. At a true agreement rate of
0.60, a simple binomial 95% CI has half-width **±8.1 pp at n = 141**; because
judgments are clustered 3-per-item, the effective n lies between 47 items
(±14.0 pp) and 141, and the pre-registered mixed logistic model accounts for
this. Sensitivity thresholds {0.01, 0.05} retain 52 / 40 items (156 / 120
judgments; ±7.7 / ±8.8 pp).

## Per-tier splits

Accuracy is expected to depend on tier (the whole point — see the pre-specified
contrast). Splitting a rater's anchors three ways cuts n per cell to ~n/3:

| Anchors rated | per-tier n | per-tier ±half-width @ p=0.5 |
| --- | --- | --- |
| 100 | ~33 | ±17 pp |
| 250 | ~83 | ±11 pp |
| 614 (full pool) | ~205 | ±6.8 pp |

Single-rater per-tier accuracy at moderate volume is only coarsely resolved;
**tier-level conclusions come from pooling raters within a tier**, where the
effective n is (raters × per-rater per-tier anchors). With, say, 6 raters each
completing the extended arm, a tier cell holds ~6 × 200 ≈ 1,200 anchor
judgments → ±2.8 pp per tier.

## Two-rater comparison power

To compare two raters' accuracy (difference of independent proportions),
`SE_diff = sqrt(p(1-p)(1/n1 + 1/n2))`. For equal n per rater at p = 0.5,
`SE_diff = sqrt(0.5/n)`; minimum detectable difference (MDE) at 80% power,
two-sided α = 0.05 is `≈ 2.8·SE_diff`:

| n per rater | SE_diff | MDE (80% power) |
| --- | --- | --- |
| 100 | 7.1 pp | ≈ 20 pp |
| 250 | 4.5 pp | ≈ 12 pp |
| 500 | 3.2 pp | ≈ 8.8 pp |

Pairwise rater-vs-rater differences are thus only detectable when large (≥ ~12 pp
even at 250 anchors each). Fine-grained rater ranking is **not** a goal we are
powered for; the anchors are powered for (i) calibrating each rater's accuracy
band and (ii) the group contrast below.

## Pre-specified contrast: low-rank experience × low-tier anchors

Hypothesis (registered in advance): raters **with low-rank ladder experience**
(designated in writing at invite time, before any of that rater's responses
are unblinded) are **more accurate on LOW-TIER anchors** than raters without
that experience, whose expertise is concentrated at high MMR. This directly
operationalizes the paper's caveat that professional judgment may itself be
tier-limited.

Test (per the prereg): mixed-effects logistic regression `correct ~ group +
(1|rater) + (1|item)` on **low-tier anchors only** (214 in the pool: 14
calibration + 15 core + 185 extended; a high-volume rater can cover ~205 of
them), one-sided group effect (A > B), α = .05. Unclustered two-proportion
approximations bracket the MDE:

- Group A = one high-volume rater (~205 low-tier anchors) vs Group B pooled at
  ~160: `SE_diff = sqrt(0.25(1/205 + 1/160)) = 5.3 pp`, MDE ≈ **15 pp**.
- Same Group A vs Group B contributing only calibration + core low-tier
  anchors (~72 pooled, 4 raters × ~18): `SE_diff = 6.8 pp`, MDE ≈ **19 pp**.
- Both groups at ~205: `SE_diff = 4.9 pp`, MDE ≈ **14 pp**.

The mixed model is expected to be modestly more conservative than these
approximations (item and rater clustering). A tier-limitation effect large
enough to matter for the paper's argument (experts near-chance, ~50%, on
low-tier drafts vs. low-rank raters at ~65–70%) is a 15–20 pp gap — detectable
at 80% power when Group B extends beyond core, and comfortably detected if the
true gap is larger. We therefore pre-commit to: (1) reporting each rater's
per-tier accuracy with Wilson 95% CIs; (2) the low-experience-vs-rest contrast
on low-tier anchors as the primary anchor-derived result; (3) treating
single-rater per-tier point estimates as descriptive, with inference done on
pooled groups.

## Design implications baked into the study

- A shared 40-anchor calibration block (14/13/13 by tier) is rated by every
  rater first, giving an identical-items basis for the quality gate and for
  cross-rater comparability.
- Extended anchors are tier-stratified 185/185/185 so per-tier n grows evenly
  as a volunteer keeps rating.
- Extended items are served in a stable per-rater order with under-coverage as
  tiebreak, so coverage across the 555 extended anchors spreads out and no
  tier is starved for a high-volume rater.
- `block` ('calibration' | 'core' | 'extended') on each rating separates the
  gate block and the pre-registered core estimate (unchanged: 100 items, 3
  ratings each) from the volunteer accuracy arm, so each arm can be analyzed
  as a distinct, pre-specified component.
