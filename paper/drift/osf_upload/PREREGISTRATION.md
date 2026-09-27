# Prospective test of outcome-based balance-change detection and draft-policy maintenance on future Heroes of the Storm builds

Forward pre-registration accompanying a paper under review on meta drift and draft-policy maintenance. Every analysis in the paper is retrospective. The claims below cover only game builds released after this registration.

## Research questions

- **RQ1.** Does an outcome-shift detector, with its decision rule fixed in advance, separate game builds that ship balance changes from builds that do not, on builds released after registration?
- **RQ2.** Does a draft agent whose aggregate statistics were maintained keep its advantage over an unmaintained agent on builds that neither agent, nor any analysis, has seen?
- **RQ3.** How do maintenance events in a production deployment line up with the detector's fires?

## Hypotheses

- **H1 (C1).** Take each boundary between consecutive new builds, and between the last excluded build (2.55.17.98025) and the first new build, once the later build has at least 12,000 ranked games. The detector fires on change boundaries and stays silent on no-change boundaries.
- **H2 (C2).** The existing 4,500 maintained-versus-unmaintained head-to-head drafts (created before registration) are rescored against the new builds' realized statistics. The maintained side's mean future-meta hero win rate exceeds the unmaintained side's.
- **RQ3 (C3)** is descriptive and has no hypothesis.

## Foreknowledge of data

Selected level: **Authors' limited observation of the data could not influence their analysis decisions.**

The outcome data for every claim (ranked games and patch notes from builds released after registration) does not exist yet. Two parts of the analysis already exist and have been observed:

1. **The C2 draft set.** This fixed artifact was created before registration and analyzed retrospectively against earlier builds. It acts as the test instrument. Its outcome on new builds cannot be known.
2. **The earlier side of C1's first boundary** (build 2.55.17.98025). We used its games, pooled with four other builds, as a scoring set in the paper. We have not run the detector on it or on any 2.55.17 boundary.

**Steps taken to limit influence:**
- All thresholds were fixed by the retrospective analysis on builds up to February 2026: q = 0.05, 200 games per side, the 12,000 and 20,000 game gates, and the one-sided z = 1.645 criterion.
- All builds already used are excluded from confirmation.
- The code is frozen by commit hash and SHA-256.
- Every outcome will be reported.

## Study type and design

- **Type:** observational. There is no manipulation, and blinding does not apply.
- **Design:** a prospective test on game builds released after the registration date. The agents and drafts in C2 are fixed. Nothing is retrained or regenerated.

## Data

**Sources:**
- Ranked (Storm League) replays from the Heroes Profile public API, collected by the authors' ongoing daily sync.
- Official patch notes from Blizzard's news feed, extracted by the frozen script `fetch_patch_notes.py`.

**Covered builds:** every build whose first game date is after the registration date, up to the analysis date.

**Excluded builds:** everything up to and including 2.55.17.98025 (released 2026-09-12). This includes 2.55.16.97039 and 2.55.17.97605, 97650, 97771 and 98025, which the paper uses retrospectively as its out-of-sample scoring set.

**Expected sample:**
- New ranked games arrive at about 14,000 to 16,000 per week, so a new build reaches the C1 gate in about a week.
- Sizable builds have shipped every 6 to 9 weeks, so a six-month window should contain three or four new builds.

**Stopping rule:** the analysis date is fixed at paper submission.
- C1 is evaluated for each boundary whose later build reaches 12,000 games by that date.
- C2 is evaluated once the new builds pool at least 20,000 games.
- A build that never reaches its gate is reported as not decidable.

## Variables

- **C1:**
  - Per hero, per boundary: win-rate shift (a two-proportion test).
  - Per boundary: fire or no-fire, plus a change class from the patch notes:
    - **Change:** at least one hero balance change listed.
    - **No-change:** zero hero changes listed, or a maintenance build with no notes.
    - **Minor-only:** only talent-level or tooltip changes, with no change to base stats or abilities.
- **C2:** per-draft paired deltas (maintained minus unmaintained) in:
  - Mean future-meta hero win rate. This is the primary metric.
  - Synergy, counter and degenerate-composition rate. These are secondary and descriptive.
- **C3:** production refresh events and their trigger source (build boundary, detector fire, or monthly cadence).

## Analysis plan

**C1:**
- Test: `w3_changepoints.py` unchanged. A two-proportion test on each hero's win rate across the boundary, with at least 200 games per side and Benjamini-Hochberg correction at q = 0.05 within the boundary.
- Decision: a boundary fires if at least one hero survives.
- Ground truth: the notes are extracted by `fetch_patch_notes.py` unchanged. Bug-fix-only and ARAM-only hero mentions are excluded.
- Retrospective reference rates: fired on 16 of 21 change boundaries; silent on 5 of 6 no-change boundaries.

**C2:**
- Protocol: `w12_clean_truth.py`, with the truth set replaced by the pooled games of the new builds (same thresholds, statistics derived by `build_patch_stats.py` and `common.py`).
- Drafts: `w6_head2head_w10_degen.json` (15 x 15 agent seeds, 4,500 drafts).
- Inference: crossed seed random effects (`w8_inference.py`).

**Inference criteria:**
- **H1:**
  - Reported per boundary.
  - Supported when change boundaries fire and no-change boundaries stay silent.
  - Minor-only boundaries are reported descriptively and neither support nor refute H1.
- **H2:**
  - **Confirmed** if the primary estimate is positive with z >= 1.645 (one-sided alpha = 0.05).
  - **Falsified** if the estimate is <= 0.
  - **Inconclusive** otherwise.
- **C3:** descriptive.

**Data exclusions:** the excluded builds above, plus bug-fix-only and ARAM-only patch-note mentions.

**Missing data:** any build that falls short of its gate is reported, not dropped.

**Deviations:**
- No thresholds are retuned, no metrics substituted, and no builds excluded beyond those listed.
- Any deviation, for example a hero added to or removed from the pool, is disclosed with its reason.

## Frozen code

Attached in `frozen/`, with `SHA256SUMS`. Git commit `9dbc5df3b70517da7818d7998e4332da5509c8e1`.

| file | SHA-256 |
|---|---|
| w3_changepoints.py | 6c95e608ac11b7a810c6184dacb3aa268859b978a059e8a0ae707f058cba2e28 |
| fetch_patch_notes.py | a047d188b22b06eb6480a96acb7f716e95ae95104fc02eca27170a4e0171fe62 |
| w12_clean_truth.py | 23079b63ab208ab3f79fb12e7d3c4e7a34aee0852b45a335aa220ee10aed96c0 |
| w8_inference.py | f539cbf3b0690bebbddeb8749c274dcde45c66e8cf62241324d75193b1d95578 |
| build_patch_stats.py | eb6314ff44d75295ab8ee12fb345aa24de32797bc07f30c0f5015bdee325fb30 |
| common.py | fb01c5297a03006d09dceebd9cc48a2a1e0501befe7651801410061c5be1d37f |
| w6_head2head_w10_degen.json | ce500a1036fc7e4ff7a4a728397110518f89c0fb988b40b932e25ed1c1d4fe1d |

## Other

The Heroes Profile v1 API ends 2027-01-01. Data collection migrates to the v2 client before then.
