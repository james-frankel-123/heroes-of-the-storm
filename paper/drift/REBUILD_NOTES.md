# Drift paper rebuild notes (2026-09-29/30)

This rebuild answers `AUDIT_2026-09-29.md`. The pre-rebuild manuscript is saved as
`overleaf/draft_pre_rebuild.tex`. All new code and outputs are in
`training/drift_rebuild/` (nothing in `training/drift2026/` was edited, and no
pre-registration-frozen file was touched). No game from a build released after
2026-09-27 was analyzed, and the detector was not run on any 2.55.17 boundary.

## 1. What was rerun, and how

| Step | Script | Output |
|---|---|---|
| Leak-free ("out-of-fold") cutoff feature passes at 2.55.14.95918, 2.55.9.93613, 2.55.4.91418, 2.55.3.89754. Five hash folds; a training row's statistics are the cutoff statistics minus its own fold's games; deployment rows keep the unmodified cutoff statistics. Verified: built from the full counts, the code reproduces the stored cutoff stats file exactly; deployment-row features are bit-identical to the leaky pass. | `r1_oof_features.py` | `feature_cache/features_oof_<build>.npz` |
| Retrain every cutoff-frozen value function on the leak-free passes (7 Table I regimes x 3 seeds, stale-1yr and stale-2yr x 3 seeds, the replay's frozen-at-C0 arm x 3 seeds), with `drift2026/train_drift_wp.py` unchanged | `r2_train_vf.py` | `models/r2_*`, `results/d2/r2_*.json` |
| Accuracy, log loss and calibration slope on the in-period validation slice and the future window, for leaky and leak-free models | `r3_eval_vf.py` | `results/r3_eval_vf.json` |
| W2b greedy degeneracy protocol (imported unchanged; reproduces `d2b_allhist_s42` = 43.4 exactly) for all leak-free regimes | `r4_w2b.py` | `results/r4_w2b/` |
| Era-matched opponent models (GD trained only on games up to 2.55.4.91418 / 2.55.9.93613) and era value-pretraining exclusion lists | `r5_era_gd.py` | `models/gd_era_*`, `models/exclude_after_*.json` |
| MCTS agents (W4 recipe unchanged) for the new arms; VF seeds cycled across agent seeds, never selected on the test window | `r6_mcts.py`, `run_queue.py` | `mcts_runs/r6_*` |
| Head-to-head (W6 protocol; reproduces the stored W6 drafts 2000/2000 exactly) | `r7_h2h.py` | `results/h2h/*.json` |
| Out-of-sample scoring with the frozen `w12_clean_truth` / `w8_inference` code by import, plus Satterthwaite df and t-reference p (reproduces W12 +0.658 ± 0.154 exactly) | `r8_score.py` | `results/r8_scores.json`, `R8_SCORES.md` |
| Replay extensions: retrain-without-refresh cadences, blind refresh schedules, leak-free frozen recipe (pipeline reproduces stored matrix cells exactly) | `r9_replay.py` | `results/r9_replay.json`, `R9_REPLAY.md` |
| W13 Net score for new matchups (games restricted to the five W12 builds and game_date < 2026-09-28; reproduces +2.70 / +2.44 / +4.00 vs paper +2.7 / +2.5 / +4.1; the DB now holds 308K rather than 298K such games because of upload lag) | `r10_net.py` | `results/r10_net.json` |
| Table I assembly | `r12_table1.py` | `results/r12_table1.json`, `R12_TABLE1.md` |
| Vintage SEs clustered by seed pairing (audit scratch `vintage_se.py`, rerun) | | `results/vintage_pairing_se.json` |

## 2. Headline claims, old vs new

| Claim (pre-rebuild) | Old | New (leak-free) | Verdict |
|---|---|---|---|
| Title: "Data-side refresh beats retraining" | refresh-only +0.730 regret beats "every retraining schedule" | never maintained 1.062; refresh-only 0.730 (recovers 0.33pp, 31%); retrain every 3 without refresh 0.050, with refresh 0.029; retrain every 6: 0.106 / 0.067 | **Reversed.** Retraining is the repair; refresh adds 0.02-0.04pp once retraining is scheduled. New title: "What Retraining and Refresh Are Worth". |
| "Two thirds recipe, one third refresh" | recipe 0.62 (1.682 vs 1.063), refresh 0.33 | leak-free frozen recipe 1.05-1.12 (3 seeds) vs 1.062: recipe about 0.02pp in the replay | **Retracted.** The replay's recipe term was the leak. At the 2026 cutoff the recipe is 0.46 of 0.49pp future-window accuracy (refresh 0.03). |
| Table I: accuracy spread vs degeneracy spread, "2.5x", r = +0.78, "accuracy ranks policies backwards" | future 56.21-57.08; degen 43.5 vs 16.8; r +0.78 | future 56.48-57.08; degen 16.8-49.7; r = -0.46 (n = 7, n.s.); all-history 25.8 vs maintained 16.8 (t 3.3); patch embed 57.05 / 31.9 vs maintained 57.08 / 16.8 (t 10.9) | **Weakened, survives in a different form**: at equal accuracy policies differ twofold. "Backwards" retracted; the 2.5x was mostly the leak (43.5 -> 25.8). |
| Leaky val column | 58.8-59.4 | 56.1-57.3 (leak inflated val by ~2pp); future calibration slope 0.69-0.73 leaky vs 0.93-1.03 leak-free | **New finding** (Sec. 3). |
| Head-to-head maintained vs unmaintained (Table III row 1) | +0.66 +- 0.15 (5x5; 15x15: +0.69 +- 0.10), Net +2.7 | paper's maintained agent (15 MCTS seeds, one test-selected VF): +0.43 +- 0.13 (15x6, t 3.3, df 10, p 0.008), Net +2.3, +0.41 / +0.46 on the two truth halves. With VF seeds varied on both sides (new maintained replication, 6 agents = 2 per VF seed): +0.34 +- 0.17 (6x6, t 2.0, df 10, p 0.07), Net +1.6 | **Shrinks by a third to a half; positive in every estimate, significant only with the larger (single-VF) seed budget.** |
| What the head-to-head gap is | "maintenance: recipe plus three months of refresh" | maintained vs maintained-at-cutoff -0.05 +- 0.11 (t -0.5); maintained-at-cutoff vs unmaintained +0.52 +- 0.13 (t 3.9) | **Clarified (A2)**: the same-cutoff gap is the training recipe; refresh adds nothing detectable at the agent level. |
| Staleness gradient | +0.56 (1 yr), +1.10 (2 yr), volume-matched keeps 91% | pure 2-yr staleness (same recipe): +1.00 +- 0.16 (t 6.3), Net +4.5; maintained vs 2-yr: +1.40 +- 0.09 (t 15.9), Net +6.3; all-VF-seed maintained vs 2-yr +1.28 +- 0.15 (t 8.7), Net +5.6; volume-matched maintained +1.28 (91%); S1-PENDING | **Survives and grows.** Stale agent now has an era-matched opponent model (A6). |
| "Synergy knowledge does not rot" (2-yr synergy +0.07 +- 0.16) | +0.07 | +0.21 +- 0.08 (t 2.8) vs maintained; +0.27 +- 0.12 (t 2.2) vs unmaintained | **Retracted.** Two years do cost synergy. |
| "Stale agents counter slightly better" | -0.31 +- 0.08 (15x15), -0.56 (2 yr) | +0.12 +- 0.07 (same cutoff), -0.08 +- 0.10 (2 yr); leak-free vs leaky unmaintained -0.39 +- 0.09 (t -4.3) | **Retracted**: a property of the leaky value function. |
| Head-to-head degeneracy benefit | 22.0 vs 36.9 (15.0pp, z 2.2) | 24.1 vs 29.7 (-5.6 +- 7.6, n.s.) | **Retracted at the agent level** (greedy protocol still 16.8 vs 25.8). |
| Decayed aggregates | 57.22, "every decayed seed beats every cumulative seed" | 57.22 / 57.19; nested window +0.19/+0.21 (z 3.3/3.5); paired game-clustered z 2.5/1.7 | **Survives**, evidence restated. Agent: +0.31 vs maintained (z 3.6), +0.77 vs unmaintained (t 7.0). |
| Detector precision | 0.94 boundaries, 0.79 heroes | fires 16/21 changed vs 1/6 unchanged (Fisher p 0.015); precision 16/17 vs base 0.78 (binomial p 0.08); heroes 0.79 any-mention / 0.67 balance-only vs chance 0.30 | **Restated.** |
| Detector as trigger: "matches blind per-build refresh at 40% of refreshes" | 0.743 vs 0.730, 14 vs 35 | 0.743 vs 14 evenly spaced blind 0.733, every 3rd build (11) 0.738 | **Retracted as a selling point**: no better than a calendar. |
| Latency / settling "same timescale" | 7,000 latency; 10,700 settling | first flag per boundary 2,500; per-flag 7,000 uncorrected, 18,900 Bonferroni (7/46 never); settling vs remaining games 17,200 | **Retracted** ("by the time you can prove ... the data already exists" dropped). |
| 30-day opponent window | "rolling", best pool | trained once: best early, worst by 2.55.16; actually rolled (new r13): 11.97% top-1 vs 11.83 once vs 11.08 all-history | **Corrected; rolling now measured.** |
| Production "refresh every build or on detector fire, opponent on a rolling month" | | production retrains everything monthly on 90-day decayed, out-of-fold statistics with a calibration-slope gate | **Corrected** to match `refresh.py`. |
| Vintage matrix | 0.42-0.51, "none saw the scoring period" | same values, one estimator, pairing-clustered SEs 0.005-0.013; QM-2026 pool 32% post-cutoff, capped retrain 0.513-0.537 | **Survives**, disclosures added. |


## 3. Changes to the manuscript, section by section

Line numbers refer to `draft_pre_rebuild.tex`.

**Title and abstract (L39, L47-71).** "Data-Side Refresh Beats Retraining" became "What Retraining and Refresh Are Worth". The pre-rebuild title was contradicted by its own Table IV and is contradicted more strongly by the leak-free rerun (A3). The abstract now leads with the replay decomposition (never maintained 1.06pp; retrain every 3 recovers 1.01pp; refresh alone 0.33pp; refresh between retrains +0.02-0.04pp), then the accuracy-vs-policy result at equal accuracy, the head-to-head, the detector against the calendar, and evaluator drift (the leak is not in the abstract, per the owner). Removed: "Drift damages policies far more than accuracy reveals ... 2.5x", "refreshing ... beats every retraining schedule", "0.94 of flagged builds", "40% of the refreshes", "synergy knowledge does not rot".

**Introduction (L78-161).** Rewritten around the two maintenance actions and four findings, with one sentence on out-of-fold statistics. Removed the three "answers" framing, "the loop closes", "refresh the feature store before you reach for the training cluster", and the production claim that contradicted `refresh.py`. The intro/body judge boundary mismatch (audit B15) is gone: 2023 everywhere.

**Section 2, setting (L163-244).** The false causality sentence (L186-189, audit A1/B1) is replaced by a per-arm statement of where statistics come from. "Five documented no-change builds" (B5) is now "five have no published notes and are treated as no-change builds"; "21 contain true balance changes" is now "the notes for 21 of them mention at least one hero". "45 balance builds" (B23) is "45 game builds". The names table adds `maintained-at-cutoff` and `unmaintained-leaky`, redefines `unmaintained` as the leak-free agent, adds the era opponent model to the stale arms, and drops the false "its drafting statistics are three months older" (A2/B2). The instruments paragraph now says agents act greedily from their policy heads without statistics.

**Leak disclosure (A1), folded into Sec. 2 at the owner's request (2026-09-30).** The leak is no longer its own section or a headline finding: the abstract and introduction drop it (the introduction keeps one sentence saying statistics are computed causally or out of fold, and why), and Sec. 2 has one paragraph with the mechanism, the out-of-fold fix and the headline evidence (val 59.11 -> 57.12, future calibration slope 0.69 -> 0.93, greedy degeneracy 43.5 -> 25.8). Leaky versions remain as labeled controls in Tables I, III, IV. The stale-model calibration decline (0.93 / 0.84 / 0.80) moved to Sec. 6 (what drifts). Earlier draft of this section, for the record: Mechanism, the out-of-fold rebuild, and the evidence: val 59.11 -> 57.12, future 56.21 -> 56.59, future calibration slope 0.69 -> 0.93 (maintained 0.93), leaky weights on leak-free features still 0.69-0.73, stale leaky 0.50-0.57 vs leak-free 0.75-0.86, replay C0 leaky 0.44 vs 0.77, greedy degeneracy 43.5 -> 25.8. The replay's "two thirds recipe" is shown to be leakage (1.682 vs leak-free 1.05-1.12 vs same-model never-refresh 1.062). Cites Kaufman et al. 2012 and CatBoost (new bib entries).

**Section 3 (old Section 3), Table I (L246-352).** Table I now shows leak-free Val / Future / calibration slope / Degen for all 7 regimes x 3 seeds plus the leaky first-version Val / Future / Degen. The "2.5x", "+0.78 ... accuracy ranks policies backwards", "Nowhere in this design space ..." and the Fig. 1 caption "The paper's thesis in one picture" are gone (C3, D). The new claim: leak-free, the patch-embedding regime is within 0.03pp of maintained's accuracy yet drafts broken teams at 31.9% vs 16.8% (Welch t 10.9); across the 7 regimes r = -0.46 (n = 7, n.s.). The two drivers are history length (short-history regimes 40-50%) and era-aligned training statistics (25.8 vs 16.8, t 3.3). The causal opponent pool rerun (C4) is cited: leak-free all-history 23.3%, maintained 14.1%. Dropped: the W4 "17.6% vs 9.3%" deep-search sentence (it compared the leaky agent); the "edge" paragraph (5.13 -> 6.08, "19% larger edge", audit B3); the 72.8% local-leak control paragraph and the causal_k100 claim (B10; the k100 prior carries the A1 leak at weight 100).

**Section 4 (old Sections 4 and 9 merged), replay (L354-414, L765-821).** Table IV rebuilt: leak-free frozen recipe (3 seeds), blind every-3rd and 14-even refresh rows (A4), retrain-without-refresh rows at K = 3 and 6 (new experiment, A3 "no replay arm retrains without refreshing"), K = 9/12, "reference" instead of "oracle" (B8), caption states one seed and what refresh means (C5). The text states: refresh alone recovers 0.33 of 1.06 (31%), not "more than half" (B3); retraining adds 0.73 over refresh; with retraining on schedule refresh adds 0.021 (K=3) / 0.039 (K=6); cadences up to 6 builds within 0.11pp and K=9/12 at 0.20/0.22 (B.L157). The 2026-cutoff split of the 0.49pp recipe gap (0.46 training-time, 0.03 refresh) is stated next to the replay's ~0.02pp recipe effect. Decayed aggregates: the nested pseudo-future window now leads (+0.19/+0.21, z 3.3/3.5; B27), with game-clustered paired z 1.7/2.5 and the 3-seed separation caveat; 57.19 reported with 57.22; the 365d "wash" sentence corrected to window-dependent (B13); the "ten times" feature-delta sentence dropped (B14); "the first post-cutoff build is the one place decay loses" dropped (B12); degeneracy 18.9/19.1 vs 16.8 now "about 2 points higher, not resolvable at 3 seeds" (B11); decay granularity disclosed (C6). The MCTS champion paragraph now discloses best-of-3 VF selection on the test window (A5) and gives the per-build W11 split (B22). The partial-refresh paragraph is rewritten per A10 (normalization confound stated; "what drifts is not what matters" framing and "least load-bearing" dropped).

**Section 5 (old Section 7), head-to-head.** See section 2 of these notes. Protocol paragraph now says agents act greedily without statistics and that agent-level maintenance is a retrain (A2); inference uses t with Satterthwaite df (C1); primary contrast named (C2).

**Section 6 (old Section 5), what drifts (L416-482).** Decay numbers labeled disattenuated with raw values (B25), "pick and ban rates" -> "pick rates"; settling now uses the deployment-honest disjoint-remainder 17,200 with the vs-final 10,700 explained as accumulation (A11); tier recovery ranking dropped (B26); never-fielded paragraph corrected (B24: Jaccard of observed sets, within 13 comps per tier from ~3 years, within 2 at the cutoff, earliest eras up to 81, r = -0.98 volume).

**Section 7 (old Section 6), detection (L484-545).** Both truth variants reported (B4): hero precision 46/58 = 0.79 any-mention, 39/58 = 0.67 balance-only, chance about 0.30; boundary fire rate 16/21 vs 1/6 (Fisher p = 0.015) and 15/19 vs 2/8 (p = 0.014); precision 16/17 against base rate 0.78 (binomial p = 0.08) (A9); controls few and clustered in 2.55.3. Latency: per-boundary first detection median 2,500; per-flag uncorrected 7,000; Bonferroni 18,900 with 7/46 never crossing; fire decisions use the full build (A11, B6, B7). Detector vs calendar: +0.743 vs +0.733 (14 even) / +0.738 (every 3rd) / +0.730; latency costs 0.004pp, sparsity 0.012pp (B7b); "40%" dropped; "14 of 20 sizable boundaries" (B7b). "honestly lagged", "the loop closes", "Outcome statistics detect the patch ..." dropped. "one lists 60 separate fixes" dropped (C12); "0.90" now "(17 of 19 flags)". Blind changepoint: chance precision about 0.16 stated, tuned family named (C7).

**Section 8, evaluators (L652-710).** Judge values use one estimator (single seed; 2026 three-seed mean given separately) (B16); "up to 0.011" (B18); intervals in Fig. 5 clustered by seed pairing and QM-2021 moved to Dec 2021 (B18, C9); the drafts are identified as the first-version (leaky) matchup; judges' relation to the agents' training data stated (C9); the QM-2026 pool's post-cutoff games disclosed with the capped retrain 0.513-0.537 (B17); "none saw the scoring period" and "disjoint from both agents'" removed; rank correlation over 11 strategies (C9). "Nothing is wrong with the old judges except their era" dropped (D).

**Section 9, opponent model (L712-763).** Table relabeled (B19); variant-matched gap 0.68 and win30d lead +0.08 (B20); "30-day rolling window" now "30 days before the cutoff, trained once", with per-build decay and pick/ban split (A8); materiality rule described as set in code before running, not publicly registered, with the SE of 2.9 vs 3.0 (C8). RESULT OF r13 ROLLING WINDOW: see section 2.

**Section 10, operating policy (L765-821).** Rewritten to match the data (retrain on a schedule; refresh is a small extra) and to describe production as it is: `refresh.py` retrains all models monthly on 90-day decayed statistics computed out of fold for training rows, with a calibration-slope deploy gate (verified in `training/production_refresh/refresh.py`, L54-79, L343-377) (A8 production).

**Section 11, prospective (L823-853).** Kept. Added the two disclosures: the frozen `w3_changepoints.load_ground_truth` counts bug-fix-only mentions although the registration text says they are excluded (reported as a deviation with C1, both variants reported), and C2's registered drafts are the leaky first-version matchup (reported as registered, leak-free matchups descriptive).

**Related work.** Added Chen et al. 2021 (JueWu-Draft) and Gourdeau & Archambault 2021 (both IEEE ToG), Berner et al. 2019 (OpenAI Five patch "surgery"), Kaufman et al. 2012, Prokhorenkova et al. 2018 (C11). Dropped claims that the cheapest adaptation "touches no weights".

**Limitations and conclusion.** Rewritten: one seed in the replay, multiplicity named (C2), the agent-level caveats, and the three analysis errors including the leak; the "each correction made the results smaller and the claims stronger" sentence and the closing aphorism are gone (D).

**Figures.** `gen_figs.py` updated (pre-rebuild copy at `training/drift_rebuild/gen_figs_pre_rebuild.py`): `fig_scatter` now plots leak-free vs leaky regimes from `r12_table1.json`; `fig_vintage` uses pairing-clustered SEs and places QM-2021 at Dec 2021; `fig_decay` labels disattenuated r and the raw net-pair series; `fig_dose` rebuilt from the new head-to-heads.


## 4. Still open

Compute was cut mid-run by the owner's machine cap (HotS lane: at most 2 GPU processes on GPU 3, 6 cores, nice 19). What that cut, and what else remains:

1. **Cut: the leak-free one-year-stale agent** (`r6_s1oof_era`, 6 seeds). The paper now has a two-point leak-free staleness gradient (same cutoff, two years) and reports the leaky one-year value (+0.56) only as a control. The era opponent model and exclusion list for it exist (`models/gd_era_2.55.9.93613`, `models/exclude_after_2.55.9.93613.json`); `jobs_mcts_nogc.txt` has the six jobs.
2. **Cut: opponent-model sensitivity arms** (`r6_uoof_gc`, `r6_mref_gc`: 2026 agents trained against the cutoff-era opponent model instead of the paper-1 model that saw the test period). Without them, part of the two-year gap may come from the opponent-model asymmetry; the paper says so (Sec. 5 and Limitations).
3. **The all-VF-seed maintained replication** (`r6_mref_rg`) was first cut to 3 agent seeds. Then restored to 6 seeds when GPU 3 freed up. Seeds 0 and 1 were stopped by the cap and resumed from their checkpoints on GPU 3 with `MCTS_FRESH=0`; the resume restores weights, optimizer and scheduler but not the replay buffer, a small deviation from the recipe. Result: +0.34 +- 0.17 vs unmaintained (p 0.07), +1.28 +- 0.15 vs 2-yr.
4. The paper's `maintained` agent (15 seeds) and `maintained-d90` agent still rest on value functions chosen best-of-3 on the future window (disclosed in Sec. 4-5). The volume-matched agent (W8a) is reused as is.
5. The vintage matrix still scores the first-version drafts (maintained vs leaky unmaintained). The paper now says so; as a demonstration of judge era it does not depend on which agents drafted.
6. Latency (2,529 / 7,026 / 18,873 among the 39 that cross, 7 of 46 never) was re-derived here from `w3_changepoints.json`. Numbers taken from the audit's recomputation rather than rerun here: 848 bug-fix mentions in 37 of 59 patches; chance hero rate ~0.30; never-fielded 13/2/81 comps; QM-2026 capped retrain 0.513-0.537; variant-matched opponent numbers (11.755, 0.68, +0.075); rank correlation n = 11; blind changepoint chance 0.16. The Fisher/binomial/precision numbers were rerun (`cp_recount.py`) and match.
7. The Net column for new matchups uses 308,375 clean games (the DB now holds more pre-2026-09-28 games in those five builds because of upload lag) against W13's 298,428; on the paper's own files it reproduces +2.70 / +2.44 / +4.00 against the published +2.7 / +2.5 / +4.1.
8. Audit A10's suggested extra cell (refresh pair win rates together with their normalizing hero win rates) was not run; the text is softened instead.
9. `COAUTHOR_SUMMARY.md` still carries pre-audit numbers (audit C10). Not touched here.
10. Carried over from the first review: the anonymous-repo link is dead, and ToG's blind-review policy is undecided.
11. Pre-registration: when C1 and C2 are reported, disclose (a) the frozen ground-truth code counts bug-fix-only mentions, contrary to the registration text, and (b) C2's registered drafts are the leaky first-version matchup. Both disclosures are already in Sec. 11.
12. Abstract, introduction and conclusion are functional but not polished, because the owner expects to merge this paper with the personalization paper; body sections are self-contained.


## 5. For a merged drift + personalization paper: what survives, and the timescales

(Added at the owner's request, 2026-09-30. Body sections of `draft.tex` are written to stand alone: Sec. 2 setting (with the out-of-fold paragraph), Sec. 3 regimes, Sec. 4 replay, Sec. 5 head-to-head, Sec. 6 what drifts, Sec. 7 detection, Sec. 8 evaluators, Sec. 9 opponent model.)

Findings that survive the leak fix and would anchor the meta-drift half:

1. **Retraining is the repair; refresh is a small extra.** Replay (2.8 yr, 36 builds): never maintained 1.06pp regret; refresh-only 0.73; retrain every 3 builds 0.05 without refresh, 0.03 with; K <= 6 within 0.11pp; K = 9/12 at 0.20/0.22. Agent level: maintained vs maintained-at-cutoff -0.05 +- 0.11pp (refresh adds nothing detectable), maintained-at-cutoff vs unmaintained +0.52 +- 0.13 (the recipe carries the advantage).
2. **Out-of-fold / causal statistics are mandatory**, and the check is calibration on rows the statistics never contained (leaky future slope 0.69 vs 0.93; stale leaky 0.5 vs 0.8). The personalization side uses causal per-hero skill estimates, which is the same discipline.
3. **Accuracy understates policy differences** at equal accuracy (patch embed 57.05 vs maintained 57.08; greedy broken teams 31.9% vs 16.8%).
4. **Recency-decayed statistics (90-day half-life) predict and draft better** (nested window +0.19/+0.21pp, z 3.3/3.5; agent +0.31pp vs maintained, +0.77pp vs unmaintained).
5. **Evaluators drift**: judges trained before 2023 reverse a 2026 verdict.
6. **Detector**: fires on 16/21 changed vs 1/6 unchanged boundaries (Fisher p = 0.015); useless as a refresh scheduler.

Timescales (for matching against individual-skill timescales):

| quantity | value | source |
|---|---|---|
| sizable-build cadence | median 60 days between sizable builds (28 builds, 4.5 yr) | `patch_index.json` |
| hero win-rate self-correlation (disattenuated) | 0.94 at 1 build (60 d), 0.78 at 8 builds (487 d), 0.62 at 16 builds (975 d); raw 0.85 -> 0.57; about -0.02 per sizable build | `SIGNAL_DECAY.md` |
| role-composition win rate | 0.98 / 0.95 / 0.95 at 1 / 8 / 16 builds (static) | `SIGNAL_DECAY.md` |
| hero pick rate | 0.98 / 0.94 / 0.89 | `SIGNAL_DECAY.md` |
| games until a new build's hero win rates are within 1pp of the build's remaining games (deployment-honest settling) | median 17,241 games (about 8 days at 15K ranked games/week) | `W3_RECOVERY.md` |
| same, against the build's final value (mechanical, accumulation) | median 10,719 (p75 12,237) | `W3_RECOVERY.md` |
| first detector flag per firing boundary | median about 2,500 games into the build | audit recount |
| per-flag latency, uncorrected / Bonferroni | 7,000 / 18,900 games (7 of 46 never cross) | audit recount |
| best decay half-life for aggregate statistics | 90 days (365 d ties cumulative on the future window) | `DECAYED_AGGREGATES.md`, W8c nested |
| opponent (behavior) model window | 30 days before each build, retrained per build: 11.97% top-1 vs 11.08% all-history | `r13_rolling_gd.json` |
| value-model calibration slope vs staleness (leak-free) | 0.93 at cutoff, 0.84 one year stale, 0.80 two years stale | `r3_eval_vf.json` |
| retraining cadence that keeps accuracy regret < 0.11pp | every 1-6 builds (replay builds average 29 days apart, so monthly to twice a year); 9-12 builds cost 0.20-0.22pp | `R9_REPLAY.md` |
