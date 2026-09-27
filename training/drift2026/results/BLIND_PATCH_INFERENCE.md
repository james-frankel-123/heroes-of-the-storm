# Blind patch-timing inference from match data (transfer-paper probe)

2026-07-15. Question: without version labels, can the match stream alone
tell us WHEN the environment changed? W3(b) answers "did stats change
across this KNOWN boundary"; unlabeled domains (NBA, MLB, Best Ball) need
the blind version. This domain has ground truth to validate it.

Script: `drift2026/blind_patch_inference.py`. Daily per-hero pick/WR
series from the pinned snapshot; +/-7-day window shifts per hero;
aggregate top-10 heroes per day; peak-find over threshold sweep; validate
against the 27 sizable-build release dates at +/-3-day tolerance. Three
statistic families tried: two-proportion z, raw pp effect size,
self-normalized (per-hero own-history) effect size. All three tell the
same story.

## Findings

1. **Only major patches separate cleanly.** Two boundaries score 2.5-3.5x
   the ambient baseline (2.55.14.95817 of 2025-12-05, 2.55.16.96881 of
   2026-04-21) and are unambiguous blind detections. The median boundary
   scores 1.37x baseline: above the noise floor, but inside the noise
   distribution's upper tail.
2. **Best operating point: precision 0.23 / recall 0.26** (threshold
   1.5x). Sweeping the threshold trades a few high-precision detections
   against missing nearly everything.
3. **When it fires, timing is nearly exact**: matched detections land
   0-2 days from the true release date.
4. **Blind peak height does not track patch-note size** (Spearman rho =
   0.13 vs. number of heroes in the notes). The top-scored days are all
   real multi-hero patches, but many large patches are blind-invisible.
5. Interpretation consistent with the paper: between-patch drift here is
   substantially CONTINUOUS (population mix, meta adaptation, free
   rotation), and most patch steps are the same order as a week of
   ambient movement at this resolution.

## Transfer implications

- In unlabeled domains, expect blind detection to recover only MAJOR
  regime changes; do not plan analyses that require a recovered event
  calendar.
- The capstone domains mostly have external calendars anyway (NBA/MLB
  rule changes, season boundaries, CBA changes) — the analogue of the
  patch index. W3(b)-style boundary-anchored detection transfers given
  those.
- The paper-2 REFRESH detector does not need patch timing at all: it
  monitors deployed-model-relevant statistics continuously and fires on
  magnitude. That machinery transfers to unlabeled domains as-is; only
  the evaluator-vintage analyses need dated boundaries.

## v2: tuned multi-scale detector (blind_patch_multiscale.py)

Six window scales (2-14 days), three families (pick, ban, WR), per-hero
self-normalization, scale-consistency combining; 270-config grid tuned by
F1 on the 16 pre-2025 boundaries, evaluated held-out on the 11 from
2025-26.

- Tuned config (pick+ban, top-10, scale-mean, th 1.4): train P 0.75 /
  R 0.375; **held-out P 0.18 / R 0.27** (timing errors 0, -3, 0 days).
  Sweeping the threshold on the held-out period never beats P 0.67 /
  R 0.18. Tuning does not transfer across periods.
- What multi-scale DOES do is sharpen the visibility tiers: the two major
  patches hit **9.4x and 7.5x** baseline (unambiguous; single-scale had
  3.5x/3.0x), a middle tier (June 2025 builds, Dec 2025 second build,
  Feb 2026) sits at 2.4-3.1x (marginal), and the rest sit at 1.0-1.8x —
  genuinely below ambient weekly variation, not detector weakness.

Conclusion unchanged and now robust: blind changepoint detection recovers
MAJOR regime changes with day-level timing and misses most patches, even
tuned. Refresh triggering therefore cannot rely on inferring the event
calendar; the paper's detector monitors deployed-stat magnitude directly
and needs no calendar, which is the design property that transfers to
unlabeled domains.
