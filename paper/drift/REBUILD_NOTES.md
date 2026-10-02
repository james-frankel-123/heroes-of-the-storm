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
| Leaky val column | 58.8-59.4 | 56.1-57.3 (leak inflated val by ~2pp); future calibration slope 0.69-0.73 leaky vs 0.93-1.03 leak-free | **Disclosed** in the Sec. 2 out-of-fold paragraph; slope column kept in Table I. |
| Head-to-head maintained vs unmaintained (Table III row 1) | +0.66 +- 0.15 (5x5; 15x15: +0.69 +- 0.10), Net +2.7 | paper's maintained agent (15 MCTS seeds, one test-selected VF): +0.43 +- 0.13 (15x6, t 3.3, df 10, p 0.008), Net +2.3, +0.41 / +0.46 on the two truth halves. With VF seeds varied on both sides (new maintained replication, 6 agents = 2 per VF seed): +0.34 +- 0.17 (6x6, t 2.0, df 10, p 0.07), Net +1.6 | **Shrinks by a third to a half; positive in every estimate, significant only with the larger (single-VF) seed budget.** |
| What the head-to-head gap is | "maintenance: recipe plus three months of refresh" | maintained vs maintained-at-cutoff -0.05 +- 0.11 (t -0.5); maintained-at-cutoff vs unmaintained +0.52 +- 0.13 (t 3.9) | **Clarified (A2)**: the same-cutoff gap is the training recipe; refresh adds nothing detectable at the agent level. |
| Staleness gradient | +0.56 (1 yr), +1.10 (2 yr), volume-matched keeps 91% | pure 2-yr staleness (same recipe): +1.00 +- 0.16 (t 6.3), Net +4.5; maintained vs 2-yr: +1.40 +- 0.09 (t 15.9), Net +6.3; all-VF-seed maintained vs 2-yr +1.28 +- 0.15 (t 8.7), Net +5.6; volume-matched maintained +1.28 (91%); 1-yr (leak-free, era opponent, 3 seeds): +1.12 +- 0.23 vs unmaintained (t 4.8, df 5), +1.34 +- 0.16 vs maintained | **Survives and grows; most of the loss arrives within the first year** (leaky 1-yr was only +0.56). Stale agent now has an era-matched opponent model (A6). |
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

1. **Reduced: the leak-free one-year-stale agent** (`r6_s1oof_era`) to 3 agent seeds (one per value-function seed) instead of 6, after the compute cap. Its df are small (3-5) though the effect is large (t 4.8-8.6). Seeds 3-5 remain in `jobs_mcts_nogc.txt` if a fuller estimate is wanted.
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

## 6. Tier-label correction (2026-09-30)

**Bug.** The replay daemon stored `league_tier` one level too high and labeled Master games (`league_tier` NULL) as "mid". In the pinned snapshot the research tiers are therefore low = Bronze and below, mid = Silver + Gold + Master, high = Platinum + Diamond. The pre-rebuild and rebuilt drafts both said "Bronze-Gold, Platinum-Diamond, Master-Grandmaster", which was wrong. The live DB was relabeled on 2026-09-30; the snapshot keeps the old labels, and every model and statistic in the paper uses them.

**Manuscript.** Sec. 2 now states the actual grouping and the sensitivity check below. No per-tier finding remains in the rebuilt draft: the tier ordering of settling and stability ("low 6,500, high 7,800, mid 8,700"; "meta stability is tier-ordered") was already dropped in this rebuild (audit B26), so nothing tier-ordered needed relabeling. The "within 13 compositions per tier" sentence uses the tiers only as bins. The underlying artifacts `training/drift2026/results/W3_HETEROGENEITY.md` and `W3_RECOVERY.md` (tier rows) still carry the old names and should be read with the mapping above; they are no longer cited per tier.

**Sensitivity check** (`training/drift_rebuild/r14_tierfix_features.py`, results `results/r14_tierfix.json`). Corrected tiers in the site's scheme (real = league_tier - 1, NULL = Master; low = Bronze + Silver, mid = Gold + Platinum, high = Diamond + Master); 907,710 of 1,949,087 games (47%) change label. The maintained (cumulative_prev) feature cache was rebuilt with corrected tiers for both the per-tier statistics and the tier one-hot, and the value function retrained (3 seeds, same protocol):

| | future acc | future calibration slope | future log loss |
|---|---|---|---|
| mislabeled (paper) | 57.08 (57.09 / 57.11 / 57.04) | 0.93 (0.92 / 0.92 / 0.95) | 0.6777 |
| corrected tiers | 57.11 (57.12 / 57.04 / 57.17) | 0.93 (0.91 / 0.96 / 0.93) | 0.6776 |

Nothing moves beyond seed noise, so no other experiment was rerun; the paper reports this in one sentence.

## 7. Consolidated audit (2026-10-01), drift items

Source: `audits/CONSOLIDATED_AUDIT_2026-10-01.md` §6 (4aff9b0). New scripts are in `training/drift_rebuild/audit_fixes/`, and their results are in `training/drift_rebuild/results/`.

**Compute caveat.** At the coordinator's request, all MCTS training in this lane stopped on 2026-10-01 at about 15:20. The X2 bug: the CUDA kernel expands nodes only at our own turn, so the tree never searches past the current own-pick block. Max ruled it a must-fix. Every drift agent (drift2026 w4/w7/w8 and drift_rebuild r6_*) is old-kernel and must be retrained once the fixed kernel is ready; `mcts_runs/OLD_KERNEL.md` records this. The A1 control below uses the four U-era agents (`r6_uoof_gc_s0-3`) that finished before the stop. They are old-kernel, the same as every other agent in the paper.

| Item | What changed | Numbers / artifact |
|---|---|---|
| A1 opponent-model confound | **Control run (partial).** U-era is the leak-free unmaintained VF trained against the cutoff-era opponent model (`drift2026/models/gd_cutoff`), 4 seeds. Results: U-era vs S1 +0.97 ± 0.15 (t 6.4, df 2.7); U-era vs S2 +0.64 ± 0.14 (t 4.8, df 8); U vs U-era +0.24 ± 0.21 (n.s.), i.e. the paper-1 opponent model's value to the 2026 agent. **Claims changed:** the staleness headline is now era-matched (0.97 / 0.64), with the mismatched rows shown separately. "Most of the loss arrives within a year" is dropped (the 1- and 2-year gaps cannot be told apart). "Two years do cost synergy" is retracted: with matched opponents, synergy is −0.06 ± 0.08 at 2 years and −0.17 ± 0.07 at 1 year, and counters favor the newer agent (+0.27 ± 0.12). Abstract, intro, Sec. 5, Fig. 2 (redrawn: era-matched vs mismatched), conclusion and limitations are all updated. Still pending: two more U-era seeds, and the M-ref/M-cut era-matched arms, all to be redone on the fixed kernel. | `results/h2h/{Ugc_vs_S1,Ugc_vs_S2,U_vs_Ugc,M_vs_Ugc}.json`, `results/r8_scores.json` |
| A2 second degeneracy property | Softened to "History length accounts for most of the spread"; era-aligned statistics "may lower it further". The cutoff-opponent pair now leads (23.3 vs 14.1, t 7.0, df 2.6, p 0.01), followed by the original pair (25.8 vs 16.8, p 0.08). The paper states that the difference is not detected at the agent level. All Welch tests now give df and p. | `audit_fixes/a2_welch.py`, `results/a2_welch.json` |
| A3 one VF draw | "Comes from" changed to "consistent with". M vs M-cut now carries a 95% CI (−0.30 to +0.19). Per-VF-seed means of the replication are +0.35 / +0.38 / +0.30. The split is stated as conditional on one VF draw. The planned `mcut_rg` (all VF seeds) arm was not run (kernel stop); queue it on the fixed kernel. | `results/r8_scores.json` |
| B1 production | Dated: the first full run completed and deployed 2026-09-30 (slope 1.05); scheduled runs begin 2026-11-01. Sec. 11 logs it as the first C3 event. X4 (compositions not out of fold) is on hold and untouched. | |
| B2 | 0.92 over 9 strategies → **0.90 over 11** for both QM judges (0.91 / 0.90 with tied ranks). The CONSENSUS_ORDER key bug lives in `qm2026/` and was not edited there. | `audit_fixes/b2_rank_corr.py`, `results/b2_rank_corr.json` |
| B3 | 30-day per-build story rewritten from W3_GD_DRIFT (best on the three 2.55.15 builds, beats all-history on the first two 2.55.16 builds, worst on the last two); the window-size explanation is dropped. | |
| B4 | Paired seed SEs are 1.2 (healer) and 0.9 (degeneracy); degeneracy missed its threshold by 0.3pp at about 2.9 SE. | |
| B5, B6, I1 | The replay rows' seeds are described accurately. Noise is now 0.07pp (the frozen-recipe seed range). The recipe comparison gives the 3-seed mean and the same-seed 0.06pp. The Limitations example uses 0.050 vs 0.067. | `results/r9_replay.json` |
| B7 | 10 drafts per ordering for the 15×15 first-version matchups. | |
| B8 | "Touch" → "include". The 12,337-game overlap with the selection and opponent window is disclosed, and the 2.55.17-only result is cited (M vs U +0.46, p 0.007). Intro "builds no agent has seen" is fixed. | `results/r8_scores.json` T_255_17 |
| B9, N3 | The Net column is recomputed with the structure-aware realized index (`overfit2026.gold.StructRealizedIndex`: hero, synergy, counter, composition + no-healer / no-frontline / stack). The caption states its game set: 308,375 games from the r10 cache fetched 2026-09-29, before the relabel, so snapshot-scheme tiers. Values move slightly: M vs U 2.3 → 2.7, M vs S2 6.3 → 6.1, U vs S2 4.5 → 3.4. | `r15_net_struct.py`, `results/r15_net_struct.json` |
| B10 | "22 of the 40 mapped patches (458 mentions)". | recomputed from `patch_notes_ground_truth.json` |
| B11 | The names table and text now say M-d90 uses the shrunk variant chosen on the future window, while the nested check prefers the unshrunk one. | |
| B12 | The judge error is now defined as pairing-clustered SE combined in quadrature with judge-seed SD (up to 0.011; range up to 0.021). | |
| B13 | The C2 rule is stated verbatim (z ≥ 1.645 one-sided; ≤ 0 falsified). The analysis date is a macro, `\analysisdate`, which must be set at submission. | |
| B14 | Fig. 2 now uses t quantiles with each contrast's Satterthwaite df, and the caption covers both panels. | `paper/drift/scripts/gen_figs.py` |
| B15a-e | 46-50%; "a third to a half"; 14 of 20 fires including the no-change 2.55.10.94470; 507 held-out days, chance 0.15; inflow is the snapshot's early-2026 rate of about 11,000/week (~1.5 weeks; from `patch_sidecar.npz`) plus the prereg's 14-16K/week (~8 days). | `results/c4_misc.json` |
| C1 | The detector arm is labeled a hindsight upper bound and called a tie with the calendar; the 0.748 alternative placement is noted. | |
| C2 | Calibration aging is rescored on a common window (the 7 post-2026-cutoff builds): **0.93 / 0.79 / 0.73** (was 0.93 / 0.84 / 0.80 on unequal windows). | `audit_fixes/c2_calib_common.py`, `results/c2_calib_common.json` |
| C3 | W11 is rescored with t/Satterthwaite (t 3.6, df 31; 1.9; 4.1; degeneracy 0.7), "conditional on one VF per arm". The decayed-aggregate z values now have an artifact and reproduce exactly (future 2.47 / 1.74; nested 3.28 / 3.47; 365d nested 3.10). | `audit_fixes/c3_decayed_z.py`, `results/c3_decayed_z.json` |
| C4 | Artifacts: K = 9/12 regrets (0.199 / 0.222); chance 0.152; capped QM-2026 retrain (0.515 / 0.537 / 0.524 → text "0.515 to 0.537", was 0.513); never-fielded 13 / 2 / 81 (already in `drift2026/results/w3_neverfielded.json` era_target_rows). | `audit_fixes/c4_misc.py`, `c4_qm2026_capped.py`, `results/c4_*.json` |
| C5 | Sentence added: the rule's measure is a lower bound; the rolling window's 0.89pp also misses 1.0pp. | |
| C6 | Limitations now discloses the interim looks at M-ref (3 and 5 seeds), the resumes without the replay buffer, the 1-yr cut, and the stopped U-era seeds. | |
| C7 | Train/deploy lookup-rate mismatch measured on 60K fold-0 rows: availability differs by at most 0.005 percentage points (synergy 0.99998 vs 0.99999, comp 0.99992 vs 0.99997). One sentence added in Sec. 2. | `audit_fixes/c7_oof_missingness.py`, `results/c7_oof_missingness.json` |
| C8 | **Vintage matrix rescored on the leak-free M vs U drafts** (7,200). Ranked 0.462 / 0.479 / 0.487 / 0.488 / 0.512 / 0.525 (2022Q1 → 2026); QM 0.428 / 0.432 / 0.487 / 0.522. Pre-2023 judges reverse the verdict by 1.9-5.4 combined SE; 2023-2024 lean the same way by about 1; 2025-2026 side with the realized outcome by 1.1-2.1. **Claim changed:** "from 2023 on, judges call it roughly even" → "2025-2026 judges agree with realized outcomes". The interpretive "current-meta knowledge" sentence is dropped. Fig. 4 is redrawn on the leak-free drafts; the first-version numbers reproduce exactly. | `audit_fixes/c8_vintage_leakfree.py`, `results/c8_vintage_leakfree.json` |
| C9, C10 | All figures are referenced. Arm names: maintained definition, frozen-statistics recipe row, U-era row; "frozen-statistics recipe" is used consistently. | |
| D1-D7 | Applied as suggested. | |
| N1 | "So no result depends on the grouping" → "We did not rerun the policy-level experiments with the corrected tiers." | |
| N2 | Third prereg deviation disclosed in Sec. 11: a rerun of the frozen truth scorer must reconstruct snapshot-scheme tiers from `league_tier` (or the 2026-09-30 backup table). **Not yet implemented in code**; the forward C2 analysis needs it. | |

**Holds respected.** The X2 kernel wording ("AlphaZero-style", Sec. 2) is unchanged. Limitations does gain one factual sentence saying the agents were trained with the old kernel and will be rerun; remove it if Max prefers to handle X2 wording in one place. The X4 production-composition sentence (Sec. 10, "computed out of fold for training rows") is unchanged.

**Page count:** 11 pages (was 10), no overfull boxes.

**Policy (Max, 2026-10-01): implementation bugs are never disclosed as paper limitations; they are fixed, rerun, and the corrected results reported.** The coordinator removed the kernel caveats (c650374). Following the same rule, I also removed "two of its seeds resumed from checkpoints without their replay buffer" from Limitations; those agents are old-kernel and will be rerun anyway. Kept, because they are statistical disclosures rather than bugs: the interim looks at the maintained replication and the one-year seed cut (audit C6), the leak-free/out-of-fold paragraph, and the prereg deviations, which the registration requires us to disclose. For the owner to decide: the Sec. 2 tier sentence still says "Because of a labeling error in our replay collection". The text is written as the coordinator requested on 2026-09-30, but it is a bug disclosure under this policy.

## 8. v2: full rebuild on site-tiered data (2026-10-01, in progress)

**Decision (Max):** fix the tier labels properly and regenerate every number once. Implementation problems are fixed, not mentioned in the paper.

**Data.** `training/drift_rebuild/v2/v2env.py` relabels the pinned snapshot with the site rule (`sync/relabel-skill-tier.ts`): league_tier NULL → high if avg_mmr is set, else unknown; ≤3 low; ≤5 mid; else high. Unknown rows (5,749) are dropped, leaving 1,943,338 games. The out-of-sample truth (2.55.16.97039 + four 2.55.17 builds, game_date ≤ 2026-09-27) was re-fetched read-only and relabeled the same way: 292,902 games, the same set for Δ and Net (`v2/r16_v2_truth.py`). Quick Match judges are relabeled from qm_games.league_tier, where Master = 0 (`v2/v2_judges.py`). All outputs go to `training/drift_v2/`; nothing in `drift2026/` or the frozen prereg files changed. Frozen scripts are run unmodified through `v2/run.py`, which applies v2env and redirects paths. `detector_curves.py` hard-codes v1 verification values, so a v2 copy (`v2/detector_curves_v2.py`) replaces that gate with a printed comparison.

**Design change.** In the W2b greedy protocol and in every MCTS agent, the opponent model is now era-matched by construction: v2 `gd_cutoff` for 2026 agents, `gd_era_<b>` for stale agents. The future-inclusive paper-1 opponent model is no longer used anywhere, so the A1 asymmetry disappears rather than needing a control.

**Done (non-MCTS):** patch stats, decayed and matched-clock stats; all feature caches (cumprev, cutoff, decayed×3, matched-clock×2, C0, OOF×4); all value functions (Table I 7 regimes × OOF/leaky × 3 seeds, cumprev, q7×3, stale OOF, C0 OOF/leaky, W2c replay 35, W8c nested 12, Q12 6, finetune chain 11, volmatch s42); r3 calibration; c2 common-window calibration; r9 replay; W2c summary and detector arm; detector curves; Q5 partial refresh; W8c nested; matched clock; c3 decayed z; c4 misc; c7 missingness; signal decay; recovery; changepoints + `detector_counts.json`; blind changepoint; never-fielded (+ volume r); v2 opponent models cutoff (v0, v1), full (v0), era 2.55.4.91418; v2 judges (10).

**Running on the 3080:** W2b greedy degeneracy (51 cells, 4 CPU queues, about 2 h); then volmatch s123/s777, W9 MMR cells and report, and the opponent-model pools era 2.55.9.93613, win6, win30d, future and roll30 (gated to start after W2b to stay within 47 GB RAM). One MCTS agent, `v2_M_s0`, is finishing (fixed kernel, chance mode, about 4.5 h).

**Manuscript status:** Secs. 4 (replay, decay, refresh), 6 and 7 and the Setting numbers are on v2. Sec. 3, the Setting degeneracy figure, fig_scatter and the abstract/intro accuracy-level numbers are on v2 too. Still pending: Sec. 9 (waiting on the opponent pools) and Secs. 5 and 8 plus the abstract/intro head-to-head numbers (waiting on MCTS).

**Claims changed by v2 so far:**
- Replay: never maintained 1.06 → 1.02; refresh recovers 0.36 (35%, was 31%); retrain every 3: 0.03 without refresh / 0.00 with; every 6: 0.15 / 0.08; refresh between retrains adds 0.03-0.07; the finetune now matches a full retrain (−0.005, was +0.089).
- Decayed aggregates: all three half-lives beat cumulative on the future window (+0.19 / +0.24 / +0.20, z 3.0-4.2), but the nested check no longer prefers the 90-day family (−0.02 / +0.03 / +0.06, |z| ≤ 1.3). The paper now recommends decay without a best half-life.
- Recipe gap at the 2026 cutoff: 0.30pp (was 0.49), of which refresh is 0.035.
- **Patch-embedding regime is now above maintained in accuracy:** 57.10 vs 56.98, and it drafts broken teams at 24.0% vs 12.7% (Welch t 6.9, 3.7 df, p 0.003). The claim gets stronger: the most accurate regime is not the least degenerate. Abstract/intro now say "most accurate of eight, 0.12 points above maintained, 24% vs 13%".
- Table I (W2b, era-matched cutoff opponent, 51 cells): OOF all-history 20.7% (was 25.8), maintained 12.7% (was 16.8; was 14.1 with the cutoff opponent), all-history vs maintained t 3.7, 3.0 df, p 0.04 (was p 0.01 / 0.08 by opponent). Short-history regimes 40-48%; all/most-history 21-30%. Spread 12.7-48.3%, nearly 4x. corr(future acc, degen) over the 7 cutoff regimes: OOF −0.38, leaky +0.36 (was −0.46 / +0.78). Leaky all-history 41.8% (was 43.5); leaky vs OOF t 8.9, 3.6 df, p 0.001. The two-opponent comparison paragraph is gone because v2 uses the era-matched opponent only.
- Operating policy: "Prefer 90-day decayed statistics" → "Prefer decayed statistics; our data do not pick a half-life between 90 and 365 days" (follows the nested check).
- Calibration aging (common window): 0.96 / 0.79 / 0.75 (was 0.93 / 0.79 / 0.73).
- Detector: 46/59 (0.78) and 39/59 (0.66); boundary tests unchanged; replay arm 0.677 vs blind 0.675 / 0.685 vs every-build 0.662; sweep within 0.05pp.

**MCTS plan (priority 4, behind paper 1 on the 3080):** 29 queued agents in `drift_v2/mcts_remote_order.txt`, priority order M, U, S2, S1, Mcut, Md90, Mvol, two per VF seed. Specs are in `drift_v2/mcts_jobs.json` (local paths) and `mcts_jobs_remote.json` (3080 paths). Launch with `drift_rebuild/v2/launch_mcts_remote.sh <host> <name>`, which is pause-aware via hotsjob. `v2_M_s3` has a resume_state at episode 22,144 on the 3080 (copied from the local pause). On the 3080 one agent saturates the GPU at about 16-18 ep/s, roughly 4.5-5 h per agent, so 29 agents is about 5-6 GPU-days. The drift scheduler is stopped until capacity is allocated.
