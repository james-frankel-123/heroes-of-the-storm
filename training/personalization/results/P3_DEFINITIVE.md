# P3 definitive rerun: off-3090 build, pinned manifest, dry run (2026-10-22)

The definitive personalization rerun cannot use 3090-built inputs: the 3090's RAM fault silently changed drafter states. This lane now builds every definitive input on the main box, the 3080 and the 5090, pins them in one manifest with sha256, and runs the chosen shrinkage model (P3_SHRINKAGE.md) with its display band.

**Status (Oct 22).** The full-history download is not finished.
- Tranche 2 is 44,638 of 76,904 accounts done, and the puller is benched until Oct 23 06:00 UTC.
- About 5 days of pulling remain, so it should finish around **Oct 28**.

The whole chain has been dry-run end to end on the current panel (build `dry1022`). A waiter starts the definitive build by itself once the download is complete. The numbers below are from the dry run. The definitive page will be `results/def_<D>/P3_DEFINITIVE_<D>.md`.

## The chain (`remote/p3_def.sh <D> <stage>`)

| stage | host | what |
|---|---|---|
| export | main box | frozen DB exports: stamps, games, scoreboard (read-only research role, game_date < 2026-09-28, no 2.57) |
| store | main box | merged as-of history store `full_history/<D>/history.npz`. HP shards and panels hold battletags and never leave this box |
| upstream | main box | pins the window-cache inputs the chain links rather than rebuilds |
| ship_a1 | main box → 3080 | code, exports, history.npz, upstream inputs; the 3080 verifies the manifest before anything runs |
| a1 | 3080 (cores 0-3, ≤ 20 GB) | stamps parse, A1 cache (players, hero MMR, game times, MMR stamps), hs_slots, today's kernels, mmr_at_game, x_slots_ext, versioned merged rows `merged_rows_<D>_<slot fingerprint>.npz`, scoreboard card a_shrink, compact aug and metric caches, fit inputs |
| fit | 5090 GPU (≤ 11.5 GB) | full_ms (drafter half) and full_ms_flip (judge half, other players) |
| walks | 3080 | baselines (today, previous main), the chosen model on both halves, calibration curves, display band, state self-check, subset report |
| collect | 3080 → main | hashes every definitive file into the manifest and verifies it |
| pin_tournament | main | copies the tournament's pinned fixed inputs and verifies them on the 3080 and 5090 |
| report | main | `p3_def_report.py`: the numbers page |
| answer | 3080 | answers a tournament walk-query directory with chosen-model states (`p3_rcf_states.py`) |

Every stage is idempotent: the A1 build and the walks skip finished steps.

## Manifest (`p3_manifest.py add | verify | show`)

Each entry records: path (relative to `training/`), role (personalization, upstream or tournament), sha256, bytes, producer host, stage, git commit, created and format.

The dry-run manifest holds 107 files: 39 personalization, 11 upstream and the tournament's 57. `verify` passes on the 3080 and the 5090. The tournament pipeline refuses to run on any mismatch.

**Merged rows.** The Oct 21 replacement on the 5090 and 3080 was this lane's local rebuild. Its arrays are identical to the 3090 copy; only an embedded path string differs. The definitive build has exactly one versioned file per build, named in the manifest.

## Interface with the tournament

- **Paths.** The tournament reads `cache_def_<D>/` and `results/def_<D>/` through `P3T_BUILD=<D>` (its commit 487efe4).
- **Fits it uses from this build:**
  - the judge's other-half fit: `results/def_<D>/rcf/params/full_ms_flip.npz`;
  - the old-model layer parts: `cache_def_<D>/rcf/parts_full_ms.npz`;
  - the layer meta: `results/def_<D>/p3_c_tests_def.json`.
- **New-model player states.**
  - The tournament records its walk queries (`P3T_WALK_RECORD`), and this lane answers them (`p3_def.sh <D> answer <qdir> <adir>`).
  - Each answer gives s, sd and n over 90 heroes, in fractions of win probability. The MMR term is excluded and the layer is included.
  - Self-check: at the played hero, the state equals the chosen arm's own slot prediction to 1.5e-08 on 3,000 random V2 slots.
  - Columns are in shared.HEROES order. n is the full-history merged count, checked by brute force.
  - The evidence filter (ev_lo, parity, exclude window) applies to the state-space record, the shrinkage features, the direct scoreboard channel and the layer's residual terms.
  - Full-history counts drive the experience offset, the new-hero flag and n.
  - The tournament's format check caught two bugs in the first answer file: slot-table hero order and windowed n. Both are fixed, and its replay now refuses any n mismatch.
  - The dry-run query files are answered on the 3080 in `/home/max/hots/p3t_dry/walk_answers/`.

## Dry-run numbers (build dry1022, current panel)

Log-loss gain over population WP (x1000), paired by game against the previous main model:

| model | V1 | V2 | V2 vs previous (SE) | OOT | OOT vs previous (SE) | acc V2 | acc OOT |
|---|---|---|---|---|---|---|---|
| today's kernel (A2S-equivalent) | +17.73 | +18.56 | −10.59 (0.50) | +18.39 | −10.07 (0.25) | 60.61% | 60.32% |
| previous main model (full_ms + layer) | +26.43 | +29.15 | 0 | +28.45 | 0 | 61.91% | 61.69% |
| **chosen shrinkage model** | +33.38 | **+35.61** | **+6.46 (0.44)** | **+36.58** | **+8.12 (0.23)** | 62.82% | 62.80% |
| chosen model, judge half | +32.78 | +34.32 | +5.17 (0.50) | +35.81 | +7.36 (0.25) | 62.80% | 62.74% |

**Subsets.** Chosen model minus today's kernel:
- all games: +17.1 (0.7) on V2, +18.2 (0.3) on OOT;
- all-10 pulled: +13.7 (1.2) on V2, +17.4 (0.5) on OOT;
- 1+ proven: +14.1 (1.7) on V2, +16.3 (0.9) on OOT;
- Diamond: +16.5 (1.7) on V2, +18.4 (0.9) on OOT;
- Master: +11.2 (2.1) on V2, +15.9 (1.0) on OOT.

**Calibration.** RMS of (slope − 1), slot-weighted:

| axis | previous main | chosen |
|---|---|---|
| games on the hero | 0.160 | 0.058 |
| hero share | 0.059 | 0.043 |
| recency | 0.036 | 0.028 |

Slope by games on the hero, chosen model: 1.10 at 0 games, 1.03 at 3, 1.01 at 10, 0.95 at 30, 1.00 at 100, 0.92 at 300, 0.84 at 1000.

**Display band** (D = s² (1 + exp(−games / 10))):

| window | realized / claimed | 80% coverage |
|---|---|---|
| V1 | 0.90 | 0.786 |
| V2 | 0.67 | 0.792 |
| OOT | 0.25 | 0.800 |

Median sd by games on the hero: 5.8pp at 0, 4.5 at 1-9, 3.4 at 10-29, 2.8 at 30-99, 2.5 at 100-299, 2.1 at 300+.

These match the full_1017 study within the panel difference. The dry1022 store has about 74K pulled players against 42K in full_1017.

## Fixes the dry run forced

- **Worker venvs have no pandas.** The scoreboard cards now parse the export with the csv module in 1M-row chunks. Output is identical to the pandas build; peak 11.8 GB.
- **The chosen walk exceeded 20 GB on the 3080.** Three changes:
  - the walk is sharded by player (`P3_RCF_SHARDS`). Players are independent in the walk, so this is exact: max |Δ| 1.9e-9;
  - the shrinkage function is evaluated in chunks;
  - direct-metric features are kept only for scored slots.

  Peak after the walk fell from 21 GB to 12 GB.
- **Missing offset term in the state writer.** It dropped the experience offset that the arm adds outside the shrinkage function. The self-check caught it, and the state writer is fixed.
- **3080 cores.** By agreement with the tournament agent, this lane's jobs run on cores 0-3, away from the nightly parse on 4-15.

## Tournament format check (dry run, 2026-10-22)

The tournament agent re-recorded its walk queries on the dry1022 build, and this lane answered them on the 3080:

| file | queries |
|---|---|
| states__1 | 60,335 |
| states__0 | 869,026 |
| judgefit__0 | 1,626,570 |

Its replay passed end to end:
- echoed fields are identical;
- n equals its merged counts in every cell of every file;
- the rebuilt drafter and judge states, the judge-fit inputs and the off-role flags show no errors.

Old model (full_ms) against the chosen model, on tournament-window lobbies:

| state | correlation of s, all heroes / played heroes | spread of s on played heroes |
|---|---|---|
| drafter | 0.930 / 0.921 | 3.95 → 4.00pp |
| judge | 0.933 / 0.929 | 3.27 → 4.10pp |

Judge fit on 144,414 judge-window games, with the consensus WP as the offset:

| model | personal coefficient c2 | log loss | gain over the consensus, ×1000 |
|---|---|---|---|
| old | 4.62 (0.07) | 0.66024 | +19.98 (0.51) |
| chosen | 3.15 (0.04) | 0.65737 | **+22.85 (0.55)** |

The judge gets about 14% more personal signal. c2 drops because the new s is spread wider, so the same information carries a smaller coefficient. The tournament refits c2 and its combiners on whatever states it receives, so no constants carry over. Its attenuation figure will be recomputed.

Their scripts: `/home/max/hots/p3t_dry1022/dry1022_replay.sh` on the 3080, and the check tool `p3t_check_answers.py` (commit 610c2f6).

## Review fixes before the definitive run (2026-10-24)

An external plan review (C-numbered findings) led to these changes before the build:

- **Filtered states carry no temporal layer (C1).** The layer's residuals u = r − pred come from unfiltered predictions, so the judge, split and exclusion states inherited excluded outcomes through them. Any query with ev_lo ≥ 0, a parity filter or an exclusion window now gets no layer, matching the old judge contract. A poison check gates every selfcheck and answer run: for 50 players, every outcome and scoreboard value outside each filter form (judge window, even-day split, odd-day split, exclusion only) is perturbed, along with every layer residual. The filtered answers must not change at all, the unfiltered controls must change, and perturbing evidence inside the filter must change the filtered answers. Each filtered walk must also equal a walk over a physically filtered copy of the history. On dry1022 it passes; with the old behaviour switched back on, it fails (judge state moved by 0.038).
- **`sd` is the base posterior sd again (C12).** Answer files carry `sd` = sqrt(var) (the old sqrt(var_all) meaning) and `sd_display` = the site band. The tournament reads `sd`. Neither is a validated sd of the full estimator: the shrinkage function, direct channel and calibration change only the mean.
- **Failures stop the chain (C4, C5).** `def_walks.sh` and the collect step run under `set -e`. The selfcheck covers both halves and raises unless every difference is finite and below 1e-6. The report fails on missing subsets. Job polling retries unknown status for up to 2 hours instead of failing on one ssh error.
- **Waiter completion (C6).** The download counts as complete only if the status command succeeds, both panels show 0 left, the row count parses and is stable over two polls, and no account has status "error". On 2026-10-24 there were 0 error records (88,228 ok, 229 not found).
- **Manifest fails closed (C7).** verify fails on a missing or empty manifest, on zero matching entries, and (with `--complete`) on a missing stage inventory per role. Each host's code bundle (personalization `.py`, `remote/*.sh`, `shared.py`, built kernel `.so`) is pinned per build. Entries made outside a git checkout carry `git = src:<source digest>`.
- **No inherited derived caches (C9).** The definitive cache links regular files and three input directories only (export, full_history, gd_trainonly); `rcf/` and `p3t/` are listed as derived. dry1022 had real `rcf/` and `p3t/` directories, so it was not affected. The aug cache key now includes the merged-rows file's content, and the metric caches are keyed by the aug key and the card file's content.
- **Reporting (C13, C22).** The definitive report prints week-block 95% CIs for the primary contrasts and splits OOT at 2026-09-12 and by the formerly sealed build 2.55.17.98025. It states that V2 and OOT were read during development.
- **Disclosed, sensitivity later (C18).** The drafter and judge fits use disjoint player halves but share V1 game outcomes, which fit the shrinkage function, direct channel, role weights, calibration and layer of each arm (coefficients fit separately). A game-disjoint variant is a planned sensitivity after the run.

## Files

In `training/personalization/`:
- `p3_manifest.py`
- `p3_rcf_states.py`
- `p3_rcf_band_def.py`
- `p3_def_report.py`
- `p3_rcf_cards.py`, `p3_rcf.py`, `p3_rcf_tails.py`: memory-lean, pandas-free
- `remote/p3_def.sh`, `def_env.sh`, `def_build_a1.sh`, `def_fit.sh`, `def_walks.sh`, `p3_def_waiter.py`
- `results/def_dry1022/`: manifest, calibration, band, subsets, the numbers page `P3_DEFINITIVE_dry1022.md`
