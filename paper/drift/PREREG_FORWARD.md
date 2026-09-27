# Pre-registered prospective forward test (paper 2, drift)

**Status: DRAFT v2 (2026-09-27), not yet registered.** It becomes binding when
it is submitted as an OSF registration. The OSF submission timestamp is the
registration date. The frozen code is identified by git commit hash and by
SHA-256 (the table at the bottom is filled at registration). Results are
reported in the paper's prospective section whatever the outcome.

## Why register, and what this does not cover

Every analysis in the paper is retrospective: the cutoff was chosen, then the
future was scored. This registration adds tests on builds that do not yet
exist, with code and decision rules fixed in advance.

**Excluded builds.** Everything up to and including 2.55.17.98025 (shipped
2026-09-12) is excluded from the claims below. That covers the paper's
post-cutoff builds and the builds 2.55.16.97039, 2.55.17.97605, 97650, 97771
and 98025. Those last five were already used retrospectively as the
out-of-sample scoring set (W12, `training/drift2026/results/W12_CLEAN_TRUTH.md`,
run 2026-09-27, before this registration). The detector has not been run on
the 2.55.17 boundaries. If they are reported, they appear as descriptive,
non-confirmatory results.

**Covered builds.** Every build whose first game date is after the
registration date ("new builds"), up to the analysis date.

## Setting and power

- Inflow: about 14-16K new ranked (Storm League) games per game-date week,
  April-September 2026, measured from `replay_draft_data` on 2026-09-27. An
  earlier figure of ~10K/week was an undercount caused by upload lag.
- Sizable builds have historically shipped every 6-9 weeks (4 builds between
  2026-05-11 and 2026-09-12). A six-month review window should contain 3-4 new
  builds.
- The paper submits without waiting for these results. They go into the
  revision.
- **Data dependency:** the Heroes Profile v1 API shuts down 2027-01-01. The
  daily sync must be migrated to the v2 client (`sync/hp-api.ts`, `HP_API=v2`)
  before then, or the revision window loses its data.

## Registered claims

**C1: detector fire/no-fire.** For each boundary between consecutive new builds
(and between 2.55.17.98025 and the first new build), once the later build has
at least 12,000 games:
- Detector: `w3_changepoints.py` unchanged. Two-proportion test on each hero's
  win rate, at least 200 games per side, Benjamini-Hochberg FDR at q = 0.05
  within the boundary. The boundary **fires** if at least one hero survives.
- Ground truth: official patch notes, extracted by `fetch_patch_notes.py`
  unchanged. Bug-fix-only and ARAM-only hero mentions are excluded, as in the
  retrospective analysis.
- **Change boundary:** the notes list at least one hero balance change.
  **No-change boundary:** the notes list zero hero changes, or no notes exist
  and the build is a maintenance build.
- Prediction: the detector fires on change boundaries and stays silent on
  no-change boundaries.
- Boundaries whose listed changes are all individually minor are reported
  descriptively and neither confirm nor falsify C1. Minor means no change to
  a hero's base stats or abilities, only talent-level or tooltip changes.
  Retrospectively these are the detector's misses (16/21 fires; the misses
  include two roster-wide cleanup patches). The reviewer-visible reference
  rates are 16/21 fire and 5/6 silence.
- Reported: per-boundary fire/no-fire, the hero flags, and the classification
  of every boundary.

**C2: maintained advantage on unseen builds.** Once the new builds pool at
least 20,000 games:
- Rescore the existing W10 head-to-head drafts (maintained vs unmaintained,
  15x15 seeds, 4,500 drafts, `results/w6_head2head_w10_degen.json`) against the
  new builds' realized statistics. Use the W12 protocol unchanged
  (`w12_clean_truth.py` with the truth set replaced by the new builds' games),
  with crossed seed random effects.
- Primary metric: the paired future-meta hero win-rate delta (maintained minus
  unmaintained).
- **Confirmed** if the estimate is positive with z >= 1.645 (one-sided
  alpha = 0.05). **Falsified** if the estimate is <= 0. Otherwise
  **inconclusive**.
- Secondary (descriptive, no decision rule): the same for the W6, W7 and W8a
  files; synergy, counter and degeneracy deltas; and the W13 net
  win-probability score.
- Neither agents nor drafts are retrained or regenerated. Everything scored
  already exists and predates the new builds.

**C3: production log.** Every stats-refresh event in the production deployment
between registration and the analysis date is logged with its trigger source
(build boundary, detector fire, or the monthly cadence) and reported verbatim,
matched against C1's fires.

## Freezes and deviations

- Frozen: the files listed below at the registered commit. No threshold
  retuning, no metric substitution, no build exclusions beyond those above.
- Deviations: any deviation is disclosed with its reason, for example a build
  that never reaches its volume gate, or a hero added to or removed from the
  pool.
- Analysis date: fixed at submission, in the paper and on OSF.

## Frozen files (filled at registration)

| file | git commit | SHA-256 |
|---|---|---|
| training/drift2026/w3_changepoints.py | 9dbc5df | 6c95e608ac11b7a810c6184dacb3aa268859b978a059e8a0ae707f058cba2e28 |
| training/drift2026/fetch_patch_notes.py | 9dbc5df | a047d188b22b06eb6480a96acb7f716e95ae95104fc02eca27170a4e0171fe62 |
| training/drift2026/w12_clean_truth.py | 9dbc5df | 23079b63ab208ab3f79fb12e7d3c4e7a34aee0852b45a335aa220ee10aed96c0 |
| training/drift2026/w8_inference.py | 9dbc5df | f539cbf3b0690bebbddeb8749c274dcde45c66e8cf62241324d75193b1d95578 |
| training/drift2026/build_patch_stats.py | 9dbc5df | eb6314ff44d75295ab8ee12fb345aa24de32797bc07f30c0f5015bdee325fb30 |
| training/drift2026/common.py | 9dbc5df | fb01c5297a03006d09dceebd9cc48a2a1e0501befe7651801410061c5be1d37f |
| training/drift2026/results/w6_head2head_w10_degen.json | 9dbc5df | ce500a1036fc7e4ff7a4a728397110518f89c0fb988b40b932e25ed1c1d4fe1d |

Command: `sha256sum <files>` at the registered commit.

## Sign-off

- [x] Max approves text (2026-09-27)
- [x] Code committed and pushed (9dbc5df, branch drift-prereg-freeze); hashes filled in above
- [ ] Submitted on OSF (Max; embargo on, anonymized view-only link generated for review)
- [ ] Paper macros `\preregdate` / `\prereglink` filled in
