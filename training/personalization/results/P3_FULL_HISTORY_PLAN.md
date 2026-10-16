# P3 full-history rerun plan

Written 2026-10-16. Nothing here has been run yet except the store build and its stats (step 0). The big rerun starts after the player backfill finishes. At 05:15 UTC on Oct 16, 51,073 replays were left below the cursor, falling about 1,400 an hour, so the backfill waiter crosses its 5K trigger around Oct 17.

## What we have

**Merged as-of history store** (`p3_full_history.py`, built on the oct05 export):

| | value |
|---|---|
| Rows | 43,975,356 (22.5M HP-only, 11.2M in both sources, 10.3M corpus-only) |
| Players | 578,142, of which 29,577 have a full HP history |
| Agreement on 11.16M rows in both sources | 0 winner and 0 hero disagreements |
| Rows dropped by the 2026-09-28 cap | 37,442 HP rows, dropped unread |
| File | `cache/full_history/oct05/history.npz` (1.6 GB) plus `meta.json` |
| Build | 5.5 min on 4 cores of the main box, 10 GB peak |

**Coverage gain for the 29,577 pulled players** (all games before the cap):

| | full | corpus | ratio |
|---|---|---|---|
| Career games, median | 712 | 267 | 2.47 per player (IQR 1.74 to 3.55); 3.01 pooled |
| Corpus share of career, before 2025 | | | 16% |
| Corpus share of career, since 2025 | | | 69% |
| As-of games before a scored game, median | 824 | 214 | 3.43 |
| As-of games on the slot's hero, median | 38 | 12 | 2.75 |
| Scored slots with 50+ prior games on the hero | 43.8% | 18.5% | |

The as-of rows come from a 2M-slot sample of P3 scored slots whose player was pulled. Those are 63% of all 10.8M scored slots.

**Lobbies with all 10 players pulled** (the key comparison set):

| window | games | all 10 pulled now | after tranche 2 |
|---|---|---|---|
| V1 | 70,974 | 5,594 | 37,554 |
| V2 | 72,474 | 4,721 | 40,590 |
| OOT (unsealed builds) | 249,569 | 3,153 | 135,895 |
| OOT sealed 2.55.17.98025 (counted only, not scored) | 36,447 | 214 | 11,743 |

**Tranche-2 pull.** `panel2.tsv` holds 76,904 accounts not yet pulled that have 20+ SL games in our corpus since 2025-01-01, most recent last game first. The cron picked it up at 05:20 UTC on Oct 16 with no crontab change. It runs at 5.1 calls/min, about 1.004 calls and about 400 rows per account.

Budget arithmetic (the 45K rolling week is unchanged):
- 45K calls by about Oct 22;
- a short bench until the oldest hours roll off;
- the last 32K calls by about Oct 27-28.

Status: `npx tsx sync/fetch-mmr-history.ts --status`.

Note that the 20-game threshold counts games in our corpus. At 69% corpus coverage since 2025, that is about 29 real games.

## Two data issues found while building the store

**1. Corpus clock bug.** This is fixed in the store but not in the DB.

**Update 2026-10-16:** the DB is now corrected and the parse is fixed (`sync/docs/timezone-fix-2026-10.md`). The bug follows the fetch time, not the id: the first shifted row is 64,264,412 (64,264,402 to 64,264,408 were stored by the old daemon and are right). Exports made after 2026-10-16 11:09 UTC are already UTC, and `p3_full_history.merge()` now shifts corpus-only rows only when the pairs show the export still carries the offset. `p3_export.py` needs no shift.

For replay ids from 64,264,402 on (games from about 2026-07-09), `replay_draft_data.game_date` is the true UTC end plus the US Eastern offset: 4 h in summer, 5 h in winter. Evidence:
- On the 11.16M (player, match) pairs in both sources, the HP and corpus times agree exactly on every pair below that id.
- From that id on, they differ by exactly the Eastern offset on 99.98% of pairs.
- Our own fetch time comes before the corpus end on 20.6% of those rows, which is impossible. Against HP's time it happens on 0.03%.

The likely cause is `new Date(replay.game_date)` (`sync/sync-replays.ts:434`) parsing HP's naive UTC string as local time on a host set to America/New_York.

What the store does: a pair takes HP's time, and 1,779,196 corpus-only rows from that id on are shifted back.

What it touches outside the store:
- Every P3 table that reads `end_ts` or `game_date` from the export for games from July 2026 on: `gametime_2024q2.npz`, the lag-1 day of a game near midnight UTC, and `p3_mmr_at_game`'s parse-before-start rule.
- Most of OOT.

Fix the writer and the stored dates in the sync lane, or at least correct the export, before the rerun (step 1).

**2. Late uploads.** HP replay ids are assigned at upload. The store estimates each match's upload time as the 95th-percentile game end among the 2,000 neighbouring ids. A bin spans a median 12.6 h, made monotone in the id.
- Across 9.84M replays, the median gap is 11 days; 54% are over 7 days and 11% over a year. Corpus replays alone: 52% over 7 days.
- The estimate agrees with HP's parse stamps on the over-7-day call for 93.5% of corpus replays (rank correlation 0.85).
- For 2026 games we fetched within a day of the game, the estimate never says over 7 days.

For a scored game, 7.3% of its as-of history rows (rule "game": played before, uploaded whenever) were not yet uploaded when it started. This counts only gaps of 1+ days. The share is 20% for corpus-history rows, because recent rows are the ones still in flight.

So the current P3 corpus history already counts late uploads, and the HP histories do not make this worse. The store carries `seen_ts`, and rule "seen" drops those rows, as a sensitivity arm (R5).

## As-of contract

- History for a scored game uses rows with game end strictly before the game's start. P3's GP state adds rows on whole days (day <= game day - 1), which is stricter.
- The game's own row is always excluded by replay_id. That matters because an HP copy can carry a different time.
- `post_mmr` is HP's post-game rating of that row. It may only describe later games.
- Fixture tests in `tests/test_full_history.py` all pass (7 of 7). They include a leaky reader planted to fail F1 and F2.

| test | checks |
|---|---|
| F1 | planted future rows, at the game's time and a day later, change no feature; a leaky reader (<= and no own-row exclusion) is caught on 60 of 60 probes |
| F2 | own label: flipping the outcome of the scored game changes none of its features, even when its HP row is timed 10 minutes before the query; the leaky reader is caught on 60 of 60 |
| F3 | a late upload under rule "seen" equals deleting it; under rule "game" it counts (the documented gap) |
| F4 | vectorized counts equal brute force on a random store with time ties |
| F5 | dedup (a match in both sources counted once, corpus values win) and the 2026-09-28 cap in the shard parser |
| F6 | `p3_rows` gives only HP-only rows of P3 players, in the slot table's hero order |
| F7 | clock fix by replay id, July and January offsets |

Run with `cd training && python3 personalization/tests/test_full_history.py`.

## Arms

All arms score the same games, combiner on V1, scored on V2 and unsealed OOT. The sealed build is scored once, with `--final`, after every choice.

| arm | history | built by |
|---|---|---|
| A0 window | corpus rows from 2024-04-01 (the current headline, +0.0142 on V2) | existing |
| A1 corpus-full | all corpus rows after the backfill (the waiter's `p3_full`) | existing, auto-launched |
| A2 merged | A1 plus HP-only rows of pulled players | new, `p3_c_merged.py` (to write) |
| A2-seen | A2 with rule "seen" for HP-only rows | flag of A2 |

A1 against A0 separates corpus coverage from estimation, as already planned. A2 against A1 on all-10-pulled lobbies is the key comparison: same games, same model, history thin vs full.

**How A2 feeds P3.** The HP-only rows have no draft, so they have no WP score. They enter in two ways:
- **Counts:** `n_p` and `n_ph` (the experience offset and every count-based term) come from `Store.asof_counts`.
- **Skill state:** HP-only rows are added to the GP state with r = y - wp_proxy and v = wp_proxy (1 - wp_proxy). wp_proxy is the hero's win rate in E for that build era, from corpus games; 0.5 is the fallback arm.

Matchmaking aims at 50%, so the proxy residual is noisier but not biased per player unless a player's drafts are systematically strong. R4 tests that on corpus games, where both are known.

## Steps and commands

All Python from `training/`. Main box: `nice -n 19 taskset -c 48-51`. 3090: through `remote/p3_job.sh` (cores 28-31 by default, 8 threads, `--mem-max 48G`).

**0. Done (Oct 16).** The stats above come from the 05:40 UTC build (29,577 pulled players). The store is rebuilt as tranche-2 accounts arrive (29,654 at 06:20 UTC).
```
nice -n 19 taskset -c 48-51 python3 personalization/p3_full_history.py build --tag oct05
nice -n 19 taskset -c 48-51 python3 personalization/p3_full_history.py stats --tag oct05
```

**1. Clock fix before any rerun** (DONE 2026-10-16, see the update above) (sync lane, about 1 h). Fix the `game_date` parse (UTC, not local) and correct stored rows from replay id 64,264,402 on. The research role cannot write, so the sync owner runs it. Until it is fixed, `p3_export.py` must apply the same shift as `p3_full_history.CLOCK_FIX_RID`. Add that to the stamps and games queries.

Check: re-export and confirm that fetched_ts < end_ts is about 0% in every id band.
```
nice -n 19 python3 personalization/p3_export.py stamps --tag full_MMDD
```
The waiter does this export itself at trigger time. If the fix lands after the trigger, re-export by hand and relaunch A1.

**2. A1 corpus-full (automatic, about Oct 17).** `remote/p3_backfill_waiter.py` (PID 2419065) exports, ships, and launches `p3_full` on the 3090. That covers `p3_c_history.py`, then `p3_pipe.py full`, 12 stages, results in `results/oct26_full`.
- Estimate: 4-8 h on cores 28-31 at 48G or less. The 45-stage window chain took about 4.5 h; full history has about 2.5x the slots.
- If step 1 is not done when it triggers, pause it on the 3090 with `python ~/hots/repo/training/remote_workers/hotsjob.py pause p3_full` (explicit job name, not a pattern; finished stages keep their done markers). Then fix, delete the `_done` markers of the stages that read game times (`hs_slots` onward), and relaunch:
```
P3_CORES=28-31 P3_MEM=48G personalization/remote/p3_job.sh max-windows-3090 p3_full bash personalization/remote/run_full.sh full_MMDD
```

**3. Store on the post-backfill export** (main box, 6 min, 4 cores):
```
nice -n 19 taskset -c 48-51 python3 personalization/p3_full_history.py build --tag full_MMDD
nice -n 19 taskset -c 48-51 python3 personalization/p3_full_history.py stats --tag full_MMDD
```
Ship `history.npz` only; it holds no battletags. Never ship the shards or panels.
```
rsync -rlt personalization/cache/full_history/full_MMDD/ max-windows-3090:/home/max/hots/repo/training/personalization/cache_full/full_history/full_MMDD/
```

**4. Write `p3_c_merged.py`** (code, about half a day; owner: P3 lane). It builds `cache_merged` like `p3_c_history.py` (links A1's inputs) and adds:
- `merged_counts.npz`: `asof_counts` per slot at slot start;
- `merged_rows.npz`: `p3_rows`, keyed to `hs_slots` pids, with hero in slot order.

Patches:
- `p3_fix_counts.lag1_counts` reads `merged_counts.npz` when `P3_HISTORY=merged`.
- `p3_x_common.online_predict` and `p3_hs_core.online_state` add `merged_rows` to the state with the day-lag rule.

Tests: extend `tests/test_full_history.py` with an end-to-end planted-future check through `online_predict` on a 50-player fixture. Add a `merged` chain to `p3_pipe.py` with the same 12 stages as `full`.

**5. A2 run.** First pass on the panel-1 store right after A1 (V2 has 4,721 all-10 lobbies). Definitive run after tranche 2 lands, about Oct 28 (40,590 V2 lobbies, 135,895 OOT).
```
P3_CORES=28-31 P3_MEM=48G personalization/remote/p3_job.sh max-windows-3090 p3_merged \
  env P3_CACHE=/home/max/hots/repo/training/personalization/cache_merged P3_HISTORY=merged \
  P3_RESULTS=/home/max/hots/repo/training/personalization/results/oct26_merged P3_EXPORT_TAG=full_MMDD \
  python personalization/p3_pipe.py merged
```
- Estimate: 6-10 h per arm. Slots stay the same, but the GP state gets about 3x more rows for pulled players.
- Memory: the (players x 90) state is unchanged; the extra rows add about 2 GB.
- A2-seen is the same command with `P3_HISTORY=merged_seen` (results `oct26_merged_seen`).
- The two 3090 jobs run one after the other, never alongside paper 1's MCTS beyond the 2-slot limit.

**6. Rerun tests** (Max, 2026-10-16), as `p3_c_tests.py` (to write, about half a day). Run each on A1 and A2, on 3090 CPU, about 1-2 h each.

- **R1 Calibration by observed record.** Cells (player x hero) as of each scored game. Bins:
  - games on the hero: 1-9, 10-19, 20-49, 50-99, 100+;
  - MMR-adjusted residual win rate to date: below -5pp, -5 to 0, 0 to +5, +5 to +10, over +10.

  Per bin, compare the mean realized residual on the next games with the model's shrunk prediction, with week-block bootstrap CIs. Over-shrinkage shows as realized above predicted in the top-record, high-n bins. Max's examples (about .500 on Diablo, .622 on Ana over many games) are cells in the 100+ bins.
- **R2 Heavy-tailed prior.** Replace the Gaussian cell effect (sd about 4pp) with two variants:
  - a Student-t, df fit by marginal likelihood on E;
  - a two-component specialist mixture, with the share and the wide component's sd fit on E.

  Score overall lift, and lift on lobbies with 1+ and 3+ proven-hero players.
- **R3 Subset lifts.** Lobbies where 1+ or 3+ players are on a proven hero (50+ prior games and posterior over +5pp). Also deep-history lobbies (all 10 with 500+ prior games), Diamond and Master, and all-10-pulled lobbies. Each with week-block bootstrap CIs, reported for A0, A1, A2.
- **R4 Meaningful-game weighting.** Weight history rows in S and P by w = w_recency x w_mode x w_opp x w_tier:
  - recency half-life 365 d (V1 already preferred 365);
  - SL 1.0, QM 0 or 0.3 (QM rows only from the corpus; HP histories are SL only);
  - opponent strength from corpus lobbies (residual vs expected via the lobby's causal MMR);
  - tier weight by Diamond or Master at the time.

  Choose weights on V1 and score on V2. Also validate wp_proxy (0.5 vs hero-era win rate) against WP_drift on corpus rows.
- **R5 Late-upload sensitivity.** A2 against A2-seen on V2. If the lift moves by more than its CI, report A2-seen as the headline (it is what a live system could have seen).

**7. Report.** Write `results/P3_FULL_HISTORY.md`:
- headline lift for A0, A1, A2 on V2 and OOT, overall and on all-10-pulled lobbies;
- R1 to R5;
- then the sealed build once, `--final`.

## Compute summary

| step | host | cores / RAM | time |
|---|---|---|---|
| store build + stats (each export) | main | 4 cores (48-51), 10 GB peak (measured) | 6 min + 10 min |
| A1 corpus-full | 3090 | 28-31, 48G cap | 4-8 h |
| A2 merged, A2-seen | 3090 | 28-31, 48G cap | 6-10 h each |
| R1-R5 | 3090 | 28-31 (or 32-33 when free), 48G cap | 1-2 h each, A1 and A2 |
| tranche-2 pull | main (cron) | nice 19, 48-63 | to about Oct 28 |

No GPU is needed. The 3080 and the 3090's cores 0-23 are not used.

## Data rules

- The local research store only, never Neon.
- The store drops every game from 2026-09-28 on while parsing, so no 2.57 game reaches it: the scilearn C, F, R and pool tranches and the sealed replays stay out.
- Shards, panels and done marks hold battletags and stay under the gitignored `training/personalization/cache/`.
