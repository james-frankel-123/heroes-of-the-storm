# drift2026 — Paper 2 (patch drift) experiment suite

Serves `research_focus_v2.md` § "Paper 2 — Drift: balance patches as repeated
natural experiments". Built on rerun2026 conventions: same pinned snapshot
(2026-05-22, 2.55-filtered, 1,949,087 replays via `rerun2026.common.load_data`),
same GPU pool (`rerun2026.common.run_pool`, logs redirected to
`drift2026/logs/`), paper-1 training protocol and sanity suite reused by
import. Full replays are never refetched.

## Patch unit (key design decision)

The **build** (full `game_version`, e.g. `2.55.14.95918`), not the minor
patch. The 2.55 corpus has only **17 minor versions** (2.55.11 never shipped)
but **45 builds, 28 with >= 20K games** — the spec's "~50 minor patches" are
these builds. Minor patches are too coarse (2.55.3 = 17 months / 8 builds).
See `PATCH_INVENTORY.md`.

## Temporal split (D2)

- Train cutoff: builds through **2.55.14.95918** (last game 2026-02-10),
  ~1.80M replays (positions 0..37 of 45).
- Strictly-future test: everything after (all of 2.55.15 + 2.55.16, ~144K
  replays, 7 builds); headline "sizable" test builds: 2.55.15.96370 (15K),
  2.55.15.96477 (81K), 2.55.16.96881 (32K), 2.55.16.97039 (12K).
- Early stopping uses a replay-level 2% validation slice of the TRAIN period
  (fixed seed 42), never the future test set.

## Leakage-avoidance rules

1. Per-patch stats use only that build's games; cumulative-through-N uses
   builds 0..N only (`build_patch_stats.py` running merge).
2. D2(b) features — including comp_wr/meta_strength aggregates, which come
   from OUR corpus, not the HP API — use cumulative-through-cutoff stats for
   BOTH train and test rows.
3. `cumulative_prev` features use only data available before the row's own
   patch began (build 0 gets empty/default stats).
4. Frozen 2026-05-19 end-of-time stats appear only as the explicit paper-1
   control arm (`d2c_frozen`).
5. Patch-embedding regime: future test rows are clamped to the cutoff
   build's embedding (the only causal choice for a deployed model).

## Pipeline

```
set -a && source .env && set +a
python drift2026/fetch_patch_data.py               # D0 fetch (DB, once)  [DONE]
python drift2026/fetch_patch_data.py --inventory   # PATCH_INVENTORY.md   [DONE]
python drift2026/build_patch_stats.py              # D1 (~8 min, CPU)     [DONE]
python drift2026/signal_decay.py                   # D2a (~1 min, CPU)    [DONE]
bash  drift2026/run_d2_pipeline.sh                 # D2b+c features + GPU grid [DONE]
python drift2026/summarize_d2.py                   # results/D2_SUMMARY.md    [DONE]
```

## Experiment -> outputs -> research_focus_v2 bullet

| Experiment | Script | Outputs | Spec bullet served |
|---|---|---|---|
| D0 patch sidecar | `fetch_patch_data.py` | `patch_sidecar.npz` (replay_id -> build/date, 4.5 MB), `patch_index.json`, `PATCH_INVENTORY.md` | prerequisite for all of Paper 2 |
| D1 per-patch aggregates | `build_patch_stats.py` | `patch_stats/per_patch/<build>.json.gz`, `patch_stats/cumulative/<build>.json.gz` (45 each; StatsCache drop-in via `common.load_patch_stats`), `patch_stats/patch_counts.pkl.gz` | "Per-minor-patch aggregate statistics feeding the enriched features" |
| D2a signal decay | `signal_decay.py` | `results/signal_decay.json`, `results/SIGNAL_DECAY.md` | "post-patch recovery speed per signal class (per-hero WR decays fast; role structure barely moves; pairwise in between — measure, don't assume)" |
| D2b regime comparison | `build_drift_features.py` (`features_cutoff.npz`) + `phase_d2.py` -> `train_drift_wp.py` | `results/d2/d2b_{allhist,win3,win6,win12,decay90,decay365,embed}_s{42,123,777}.json`, `models/d2b_*.pt` | "Training regimes: all-history vs. recency-window vs. time-decay-weighted vs. patch-conditioned (patch embedding)" |
| D2c enrichment sourcing | `build_drift_features.py` (`features_{local,cumulative_prev,frozen}.npz`) + `phase_d2.py` | `results/d2/d2c_{local,cumprev,frozen}_s*.json` (+ `d2b_allhist_s*` = cumulative-to-cutoff arm) | "hypothesis: enrichment improves further with patch-local stats; the frozen-snapshot design of paper 1 becomes the control arm" |

Every D2 cell reports: future-test accuracy (overall / per test build /
sizable-only), train-period val accuracy, and the paper-1 sanity suite
(21- and 28-test pass counts + degenerate-comp WP scores) computed with that
arm's deployment-time stats.

## D2a headline (already computed)

Games-weighted cross-build correlations over the 28 sizable builds
(r_adj = disattenuated for binomial sampling noise; lag in sizable builds):

| signal | r_adj@lag1 | r_adj@lag8 | r_adj@lag16 | notes |
|---|---|---|---|---|
| role-comp WR | 0.983 | 0.952 | 0.947 | essentially static (fit floor 0.95) |
| hero pick rate | 0.984 | 0.939 | 0.887 | meta habits very stable |
| hero WR | 0.941 | 0.783 | 0.622 | the fastest-decaying reliable signal |
| pairwise raw WR | 0.938 | 0.773 | ~0.61 | tracks its component hero WRs |
| pairwise net-of-solo | (raw r ~0.15-0.18, flat) | | | interaction term is SMALL but persistent; mostly sampling noise within a patch even at >=300 games/pair |

So: "per-hero decays fast, role structure slow" confirmed; the pairwise
*interaction* component (what counter_detail etc. encode net of solo WR)
behaves differently than assumed — it is weak-but-stable rather than
medium-decay.

## Wall-time (4x RTX PRO 6000, 64 cores)

| Stage | Estimate | Actual |
|---|---|---|
| D0 fetch + inventory | ~5 min | ~5 min |
| D1 stats | ~10 min | 8 min |
| D2a decay | ~1 min | 1 min |
| D2 feature passes (4x) | 2-3 h CPU | **8 min** (phase0's hours are CQL-dominated; the full-draft pass is ~3.9M extract calls on 56 workers) |
| D2 training grid (30 jobs) | 2-4 h on 4 GPUs | **~12 min** (30/30 OK; ~40s/job — full-GPU-resident tensors, early stop at 28-58 epochs) |

Total ~35 min wall — far under the 24h budget; no trimming of seeds/windows
was needed (3 seeds, 3 windows, 2 half-lives as specced). Marginal cost of
extra cells is trivial; adding seeds/regimes is cheap.

## D2 grid results (see results/D2_SUMMARY.md for the full table)

Strictly-future test accuracy (mean over 3 seeds; train-period val in parens):

- d2b_allhist 56.21 (59.11) — most data, best val, WORST future: drift is real.
- d2b_win3/6/12: 56.37 / 56.53 / 56.47 — recency windows help; ~6 builds best.
- d2b_decay90/365: 56.48 / 56.36.
- d2b_embed 56.55 (59.14) — best training regime: keeps all-history val acc
  AND wins on the future.
- d2c_cumprev 57.08 — temporally-matched causal feature stats, refreshed
  through each test build's predecessor: beats every training regime by ~2.5x
  the best regime's gain, with NO retraining. Site-facing implication for the
  auto-retrain deliverable: refresh aggregates continuously; retrain (with
  patch conditioning) second.
- d2c_local 72.77 — **within-patch self-leakage, oracle only** (per-patch
  stats contain the predicted games; accuracy highest on smallest test
  builds). A deployable patch-local arm needs within-patch-causal stats.
- d2c_frozen 57.02 — paper-1 end-of-time control (leaky by design).

Sanity suite: 24-26/28 for all cells except d2c_local (22.7).

## Wave 2 (2026-07-07): W2a causal-local, W2b policy eval, W2c auto-retrain, W4 MCTS

| Experiment | Script | Outputs |
|---|---|---|
| W2a within-patch-causal local stats | `build_causal_local_features.py` (passes `causal_{merge,k100,k1000}`: rolling same-build stats from strictly earlier calendar days, count-weighted shrinkage against the cumulative-to-cutoff prior; day-1 rows == `features_cutoff` rows [verified]) + `phase_w2.py` -> `train_drift_wp.py` | `feature_cache/features_causal_*.npz`, `results/d2/d2c_causal_*_s*.json`, `results/W2A_SUMMARY.md` (+ `results/w2a_depth.json` within-build tercile accuracies) |
| W2b policy-level eval of drifted VFs | `eval_policy_w2b.py` (paper-1 greedy drafting vs GD pool; the drafter extracts features with the ARM'S deployment stats; counter/synergy scored vs FUTURE-period ground truth merged from `patch_counts.pkl.gz`) via `phase_w2.py`; 5 regime cells x 3 seeds x 500 drafts + GD reference (1500) | `results/w2b/*.json`, `results/W2B_SUMMARY.md`, `results/w2b_summary.json` (contains the W4 trigger evaluation) |
| W2c auto-retrain policy simulation | `phase_w2.py` (one `train_drift_wp.py --cutoff-build` job per build position 8..43, cumulative_prev features, regime=all, seed 42, sanity skipped; position 37 reuses `d2c_cumprev_s42`; plus a frozen-stats-at-C0 arm on the new `features_cutoff_2.55.3.89754.npz` pass) -> `summarize_w2.py` simulates never/K-cadence/trigger policies on the resulting acc[cutoff][build] matrix | `results/w2c/*.json` (matrix rows), `results/W2C_SUMMARY.md`, `results/w2c_policies.json` |
| W4 MCTS arm (UNCONDITIONAL per 2026-07-07 decision; the W2b trigger is reported for framing only) | `phase_w4_mcts.py`: best-vs-worst regime VFs (wave-1 future acc: d2c_cumprev vs d2b_allhist), 5 seeds x 200 sims x 300K episodes via `train_mcts_worker.py` (new opt-in `MCTS_STATS_BUILD` env: kernel LUT built from the arm's cumulative patch-stats file instead of the frozen snapshot; value-head pretraining excludes ALL post-cutoff replays via `w4_exclude_ids.json`); benchmark = 200 kernel drafts vs 5 FUTURE-GD opponents (GenericDraft retrained on post-cutoff replays only, `models/gd_future/`), metrics vs future-truth stats, terminal WP by the neutral rerun2026 wp_enriched_256 judge | `mcts_runs/w4_*/draft_policy.pt`, `results/w4_mcts_results.json`, `results/W4_SUMMARY.md`; logs `logs/phase_w4.log`, `logs/w4_mcts_*.log` |

Wave-2 pipeline:

```
python3 drift2026/build_causal_local_features.py                    # W2a passes (~2 min)
python3 drift2026/build_drift_features.py --cutoff-at 2.55.3.89754  # W2c frozen-at-C0 pass
python3 drift2026/phase_w2.py                                       # 62 GPU jobs (idempotent)
python3 drift2026/summarize_w2.py                                   # W2A/W2B/W2C summaries
nohup python3 -u drift2026/phase_w4_mcts.py > drift2026/logs/phase_w4.log 2>&1 &  # overnight
```

Headline results: `results/W2A_SUMMARY.md`, `results/W2B_SUMMARY.md`,
`results/W2C_SUMMARY.md`, `results/W4_SUMMARY.md` (W4 written by the detached
driver when it finishes; monitor with
`tail -f drift2026/logs/phase_w4.log drift2026/logs/w4_mcts_*.log`).

## Wave 3 (2026-07-08): recovery, changepoints, hybrid sourcing, stacking, sim refresh

| Task | Script | Outputs |
|---|---|---|
| W3a post-patch recovery | `w3_recovery.py` (per sizable build: chronological walk, rolling estimates at log-spaced game checkpoints vs the build's final values AND vs the disjoint remainder, games-weighted MAE + binomial-expected MAE per signal class; also writes `daily_hero_counts.pkl.gz`, the per-build per-day hero pick/win/ban cache consumed by W3b/W3f) | `results/w3_recovery.json` (incl. `hybrid_window_games`, the W3d window answer), `results/W3_RECOVERY.md` |
| W3b hero changepoint detection | `w3_changepoints.py` (per sizable-build boundary x hero: two-proportion tests on WR/pick/ban with BH FDR; detection latency via day-boundary sequential z [naive + Bonferroni] and Wald SPRT at ~95%; validation P/R vs `patch_notes_ground_truth.json` — written by the patch-notes fetch agent (`fetch_patch_notes.py`), gracefully skipped if missing/empty) | `results/w3_changepoints.json`, `results/W3_CHANGEPOINTS.md` |
| W3c never-fielded comp drift | `w3_neverfielded.py` (per-build observed role-comp sets from D1 counts, consecutive Jaccard, per-era tier-2 augmentation targets vs full-corpus) | `results/w3_neverfielded.json`, `results/W3_NEVERFIELDED.md` — near-null confirmed: target sets differ by 0-2 comps/tier at the D2 cutoff |
| W3d hybrid per-signal features | `w3_build_hybrid.py` (hero-WR family from the smallest recent-build window reaching W3a's `hybrid_window_games`; comp WR all-history; pairwise cumulative; strictly causal per row -> `patch_stats/hybrid/*.json.gz` + `feature_cache/features_hybrid.npz` via the new `hybrid` pass in `build_drift_features.py`) + `phase_w3.py` -> `train_drift_wp.py` (`w3d_hybrid` x 3 seeds, deploy stats = hybrid file of the last build) + W2b policy evals (cells added to `eval_policy_w2b.py`) | `results/d2/w3d_hybrid_s*.json`, `results/w2b/w3d_hybrid_s*.json`, `results/W3_HYBRID.md` |
| W3e embed+refresh stack | `phase_w3.py` (`w3e_embedrefresh` x 3 seeds: regime=embed TRAINED on cumulative_prev features — the two wave-1/2 winners stacked; plus an eval-only check of the wave-1 `d2b_embed` checkpoints on cumprev test features in `summarize_w3.py`) | `results/d2/w3e_embedrefresh_s*.json`, `results/w2b/w3e_embedrefresh_s*.json`, `results/W3_EMBED_REFRESH.md` |
| W3f tier/map heterogeneity | `w3_heterogeneity.py` (D2a decay per skill tier and per map [hero-per-map WR], + within-build 1pp hero-WR recovery per slice from the daily cache) | `results/w3_heterogeneity.json`, `results/W3_HETEROGENEITY.md` |
| W3g retrain-sim refresh | `phase_w3.py` stage 3 (never-retrain-at-C0 trains for both new arms: `w3g_{hybrid,embedrefresh}_cut08_s42`; K=6-cadence trains at cutoffs 14/20/26/32/38 for the (d)/(e) winner IF it beats d2c_cumprev) -> `summarize_w3.py` | `results/w2c/w3g_*.json`, `results/w3g_policies.json`, `results/W3_RETRAIN_SIM.md` (refreshed regret table + recommended site policy) |

Wave-3 pipeline (self-driving, detached):

```
nohup bash drift2026/w3_chain.sh > drift2026/logs/w3_chain.log 2>&1 &
# stages: w3_recovery -> w3_heterogeneity -> w3_neverfielded ->
#         w3_changepoints -> w3_build_hybrid -> phase_w3 (GPU, --num-gpus 2
#         politeness while the rerun2026 diversity diagnostic shares the
#         pool) -> summarize_w3 -> changepoint re-validation if the patch-
#         notes ground truth landed late
# status: tail -f drift2026/logs/w3_chain.log
```

## Wave 5 (2026-07-09): GD/opponent-model drift arm (Max-identified gap)

Every prior drift arm varied only the WP value function; the GD behavioral
model stayed frozen (rerun2026 full-snapshot pool, which has seen the future
period). W3b showed pick/ban changepoints track community *adaptation* — so
the opponent/behavior model drifts too. Same W2 temporal cutoff.

| Part | Script | Outputs |
|---|---|---|
| (a) GD behavioral prediction under drift | `w5_gd_drift.py` (`--stage cache/train-one/eval`): GD memmap caches with per-sample build idx (`feature_cache/gd_w5_*`); GenericDraftModel protocol reused unchanged (`train_single_model`, replay-level 98/2 split, variants 0-1 = 2 seeds) for `gd_cutoff` (<= cutoff), `gd_win6` (last 6 builds, d2b_win6 window) and `gd_win30d` (rolling last 30 calendar days — within-build behavioral drift); next-pick top-1/top-5 on the 7 future builds (overall / per build / pick vs ban) vs frozen full-snapshot GD (non-causal bound) and W4 future-GD (leaky oracle ref) | `models/gd_{cutoff,win6,win30d}/generic_draft_{0,1}.pt`, `results/w5_gd_eval.json` |
| (b) policy-level opponent staleness | `eval_policy_w2b.py --gd-pool cutoff` (new flag; pool = GD_cutoff x2 as opponents AND rollout completions): d2c_cumprev + d2b_allhist x 3 seeds x 500 drafts + GD reference (1500), vs the existing frozen-pool W2b cells | `results/w2b/*_gdcutoff.json` |
| (c) MCTS arm, CONDITIONAL | trigger (pre-registered in `w5_gd_drift.py`): frozen-cutoff future top-1 gap >= 1 pp OR \|d healer\|/\|d degen\| >= 3 pp on (b). If fired: 3 seeds x 200 sims x 300K episodes, cumprev VF (`w4_vf_d2c_cumprev.pt`), `MCTS_GD_PATH` = GD_cutoff; benchmark identical to W4 (vs future-GD), so `w4_d2c_cumprev_s*` is the frozen-opponent arm | `results/w5_gd_trigger.json`, `mcts_runs/w5_cumprev_gdcut_s*/`, `results/w5_gd_mcts.json` |

Self-driving detached pipeline (idempotent, resumable):

```
nohup python3 -u drift2026/w5_gd_drift.py > drift2026/logs/w5_gd_drift.log 2>&1 &
# caches -> 6 GD train jobs (pool) -> next-pick eval -> 7 W2b jobs (pool)
# -> trigger -> [conditional 3 MCTS jobs + W4-protocol benchmark]
# -> results/W3_GD_DRIFT.md
# re-render summary only: python3 drift2026/w5_gd_drift.py --stage summarize
```

Headline results: `results/W3_GD_DRIFT.md` (written by the detached driver).

## Paper-revision compute items (2026-07-14): detector curves, Table II fill, matched clock

| Item | Script | Outputs |
|---|---|---|
| Q11 detector operating curves | `detector_curves.py` (BH FDR level q swept over 9 points on the W3b `wr_bh` rule, recomputed from the daily cache with the identical `w3_changepoints` primitives; P/R vs patch notes per point + full W2c detector-triggered replay per fire schedule via the `w2c_detector_arm` machinery, first-detection latency; 4 hard verification gates reproduce the stored 0.793-precision / 14-refresh / +0.743 numbers exactly) | `results/detector_curves.json`, `results/DETECTOR_CURVES.md`, `paper/drift/overleaf/fig_detector_curves.pdf` |
| W2b fill (Table II missing cells) | `eval_policy_w2b.py` (cells `d2b_win3`, `d2b_win12`, `d2b_decay365` added; identical W2b protocol, 3 seeds x 500 drafts; pipeline repro-checked by re-running `d2b_win6_s42` to exact equality first) + `summarize_w2b_fill.py` | `results/w2b/d2b_{win3,win12,decay365}_s*.json`, `results/w2b_fill.json`, `results/W2B_FILL.md` (W2B_SUMMARY.md untouched) |
| Q12 matched-clock per-signal decay | `build_matched_clock_stats.py` (`patch_stats/decayedmc/` = hero family decayed90 + pair/comp decayed365; `decayedmcrev/` = reverse; composed at the W3d/Q5 attribute split, verified exact) -> `build_drift_features.py` passes `decayedmc_prev`/`decayedmcrev_prev` (column-exact vs parents on pure groups) -> `train_drift_wp.py` D2 protocol x 3 seeds (`results/q12/`) + `summarize_matched_clock.py` | `results/q12/*.json`, `results/matched_clock.json`, `results/MATCHED_CLOCK.md` |

## Deferred (explicitly out of this wave)

- ~~Within-patch-causal "local" arm~~ — DONE in wave 2 (W2a).
- ~~Policy-level evaluation~~ — DONE in wave 2 (W2b, via
  `experiment_rich_evaluation` machinery rather than the crosseval worker).
- **Augmentation metadata under drift** (never-fielded composition set per
  patch) — spec bullet 3; pure computation over `patch_counts.pkl.gz`, not
  yet written.
- ~~Auto-retrain policy deliverable~~ — DONE in wave 2 (W2c).
- **Crosseval matrix per regime** (Table II analogue under drift) — W2b covers
  the composition/counter/synergy side; cross-scoring drafts across regime VFs
  remains open.
- **Future-GD-in-the-loop MCTS** — W4 trains vs the standard GD opponent and
  benchmarks vs future-GD; training with future-GD in the env would need
  opponents unavailable at the cutoff (deliberately out of scope).
- Python 3.14 forkserver hazard: D2 loads feature npz fully into RAM (no
  DataLoader workers, no memmap pickling), so the rerun2026 `__getstate__`
  pattern is not needed here.
