# Expert study v6.2 analysis: implementation notes

Pipeline for prereg v3 (`oss-export/docs/prereg_expert_study_v3.md`, sha256 `90c2a40c…`,
registered https://osf.io/fspuc). Run `bash run_all.sh <out_dir>` (read-only Neon export,
then everything) or `bash run_all.sh <out_dir> <ratings.csv>` on a frozen export. Mixed
models run in R 4.4.1 / lme4 1.1-35.5 / lmerTest 3.1-3 (docker image `hots-r-lme4:4.4.1`,
built from `rocker/r-ver:4.4.1`); the rest in Python.

Rater codes only. Code-to-slot and code-to-group come from the written designations and
are in `config.json`; the designations file itself is never committed.

## Choices the prereg does not settle

1. **Incomplete raters (§3.8 says their ratings are "retained"; the gates assume 240
   items).** Primary: the gates are applied as written, so a rater who never reached a
   gate's items cannot pass it (a rater who stopped before position 121 fails the catch
   gate; one who stopped before position 48 fails the screener); the speeding median is
   over the items rated. Sensitivity (`partials_lenient`): a partial rater is held only to
   the gate items they reached (screener must be complete; catch at the 2/3 rate on the
   catch items rated, vacuous if none). Both are reported.
2. **"Complete" for the S3 confirmatory condition** ("at least 3 Group-A raters complete
   the 240-item set"): counted as 240 ratings, regardless of gates. Reported together with
   the Group-A raters actually included.
3. **Convergence failure** (the trigger for every fallback chain): the fit errors or lme4
   reports a convergence message. A singular (boundary) fit is a valid estimate, is
   recorded, and is not a failure. No optimizer switching.
4. **S1 has no pre-specified fallback.** If a tier's fit fails to converge, the H1 chain's
   first step is used (drop the rater random effect) and labeled.
5. **Majority-vote fallback (H1, S5):** items whose ratings split evenly (possible with 2
   or 4 ratings) carry no majority and are dropped.
6. **H2 permutation fallback:** 100,000 random permutations, seed 20261026.
7. **S4 tournament ordering** ("round-robin tournament win-rate ordering"): each
   strategy's mean structure-corrected consensus P(win) over every tournament draft it
   played (round-robin and constrained-search files), from its own side. The uncorrected
   ordering is reported alongside. Bradley-Terry by the MM algorithm, log-strengths
   centred at 0; exact one-sided permutation over all 10! orderings.
8. **H2 z-scoring** is over the analysis items (each item once), recomputed for every
   subset (patch-2.57 variants included).
9. **Patch-2.57 variants** are applied to H1, H2, S1 and S5 at the primary 0.02
   threshold only.
10. **Item checks:** every rating must be an item of the rater's slot under the app's
    assignment (abort otherwise); the stored block must equal the pool's.

## Stopping-rule deviation (draft sentence for the paper's methods)

"We closed data collection on 2026-10-26, about three weeks before the pre-registered
six-week limit, with 11 of the 14 rater slots complete, after the three remaining raters
did not respond to a reminder; no outcome results were examined before closing."

(Check before use: the reminder outcome, and that the pipeline dry run on 2026-10-25 was
inspected only for structure, which is how it was done.)
