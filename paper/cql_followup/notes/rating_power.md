# Rating study: statistical power for per-rater accuracy (extended anchor arm)

Scope: this note quantifies what the **known-outcome ladder anchors** buy us —
i.e. our ability to estimate, per rater and per skill tier, how often an expert
picks the team that actually won the real game. Anchors are the only items with
a ground-truth label, so they (not the machine-pair items) drive accuracy power.

## Anchor supply per rater

| Arm | Anchors | Per tier (low/mid/high) |
| --- | --- | --- |
| Core (latin square, 30 items/rater) | 45 total in pool; ~14 land in any one rater's assigned 30 | ~4–5 / tier |
| Extended (uncapped) | 300 distinct (100/100/100) | 100 / tier |
| **Max distinct anchors one rater can reach** | **345** | **~115 / tier** |

A rater who exhausts the extended arm sees every one of the 345 distinct
anchors. The extended machine-pair items (100) carry no ground truth and are
used for expert-vs-model agreement, not accuracy.

## Per-rater accuracy CI vs. number of anchors rated

95% CI half-width for a single accuracy proportion, worst case p = 0.5 (widest);
Wald `1.96·sqrt(p(1-p)/n)`. Wilson intervals, which we use in practice, are
slightly tighter and better-behaved near 0/1.

| n anchors | ±half-width @ p=0.5 | @ p=0.65 | @ p=0.75 |
| --- | --- | --- | --- |
| 100 | ±9.8 pp | ±9.3 pp | ±8.5 pp |
| 250 | ±6.2 pp | ±5.9 pp | ±5.4 pp |
| 500 | ±4.4 pp | ±4.2 pp | ±3.8 pp |

So the headline resolutions are **≈ ±10 / ±6 / ±4.4 pp at n = 100 / 250 / 500**.
Note the current pool caps a single rater at 345 distinct anchors, so n = 500 is
reachable only by (a) pooling anchors across raters within a tier, or (b) a
future anchor-pool expansion; it is tabulated for completeness and as the target
resolution for the aggregated per-tier estimates below.

## Per-tier splits

Accuracy is expected to depend on tier (the whole point — see the pre-specified
contrast). Splitting a rater's anchors three ways cuts n per cell to ~n/3:

| Anchors rated | per-tier n | per-tier ±half-width @ p=0.5 |
| --- | --- | --- |
| 100 | ~33 | ±17 pp |
| 250 | ~83 | ±11 pp |
| 345 (full pool) | 115 | ±9.1 pp |
| 500 (pooled) | ~167 | ±7.6 pp |

Single-rater per-tier accuracy is therefore only coarsely resolved (±9–17 pp);
**tier-level conclusions come from pooling raters within a tier**, where the
effective n is (raters × per-rater per-tier anchors). With, say, 6 raters each
completing the extended arm, a tier cell holds 6 × 100 = 600 anchor judgments →
±4.0 pp per tier.

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
(e.g. Fan, who plays and knows the Bronze–Silver meta) are **more accurate on
LOW-TIER anchors** than raters without that experience, whose expertise is
concentrated at high MMR. This directly operationalizes the paper's caveat that
professional judgment may itself be tier-limited.

Test: two-group difference of accuracy on the **low-tier anchors only** (up to
115 distinct low-tier anchors in the pool; a high-volume rater like Fan can
cover all 115). Group A = low-rank-experienced raters; Group B = the rest.

- If Fan alone (n ≈ 115 low-tier anchors) is compared against a pooled Group B of
  k raters × their low-tier anchors, the Group B arm is the wider one only if k
  is small. With Group B ≈ 4 raters × ~40 low-tier anchors each (160), the
  comparison has `SE_diff = sqrt(0.25(1/115 + 1/160)) = 6.1 pp`, MDE ≈ **17 pp**.
- If both groups reach ~115 low-tier anchors (Fan + one more low-rank rater vs. a
  pooled high-rank group of similar size), `SE_diff = 5.9 pp`, MDE ≈ **16 pp**.

A tier-limitation effect large enough to matter for the paper's argument (experts
being near-chance, ~50%, on low-tier drafts they don't understand, vs. low-rank
raters at ~65–70%) is a 15–20 pp gap — right at the edge of what the low-tier
anchors detect at 80% power, and comfortably detected if the true gap is larger.
We therefore pre-commit to: (1) reporting each rater's per-tier accuracy with
Wilson 95% CIs; (2) the low-experience-vs-rest contrast on low-tier anchors as
the primary anchor-derived result; (3) treating single-rater per-tier point
estimates as descriptive, with inference done on pooled groups.

## Design implications baked into the study

- Extended anchors are tier-stratified 100/100/100 so per-tier n grows evenly as
  a volunteer keeps rating.
- Extended items are served least-blocked / stable-per-rater, so coverage across
  the 345 anchors spreads out and no tier is starved for a high-volume rater.
- `block='extended'` on each rating separates the pre-registered core estimate
  (unchanged: 100 items, 3 ratings each) from the volunteer accuracy arm, so the
  extended data can be analyzed as a distinct, pre-specified secondary arm.
