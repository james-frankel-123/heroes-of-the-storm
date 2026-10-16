# Timezone fix, 2026-10-16

Some stored game times were 4 or 5 hours late. On 2026-10-16 we fixed the parser and
corrected three tables in the local store. The corrected player history was then
published to Neon.

## Cause

Heroes Profile sends `game_date` as a UTC string with no zone, for example
`"2026-07-09 20:14:30"`. Our writers called `new Date(s)` on it. JavaScript reads a string
like that as host local time. Once the host was on America/New_York, every game was
stored late by the US Eastern offset: 4 h in summer, 5 h in winter.

Why it started in July:

- This box ran on UTC until 2026-05-09. On that day (about 13:50 UTC) `/etc/localtime` was
  pointed at America/New_York. `sync/logs/cron.log` stamps runs `+00:00` up to May 9 and
  `-04:00` from May 10.
- Node fixes the zone when a process starts. The replay daemon was a long-running process
  started under UTC, so its parse stayed correct until it restarted.
- It restarted on 2026-07-09 at 20:22:44 UTC for the Phase 1 player deploy (commit
  94168c5; `sync/logs/replay-daemon.log` begins at that moment). Every row it stored from
  then on was shifted.
- `fetch-qm` first ran on 2026-07-10, after the zone change, so every `qm_games` row was
  shifted.
- `sync-players` starts fresh every night, so `player_match_history` rows written from
  2026-05-10 on were shifted. So were a few earlier manual runs from an Eastern shell: the
  first pulls for two players on 2026-03-09 and some rows on 2026-03-25. An upsert also
  rewrites `game_date`, so the shift follows the last write, not `created_at`.

The v1 client (`sync/api-client-v2.ts`) passes HP's string through unchanged. The parse
happened in the workers, so the bug was the same on both APIs.

## Code fix

- `sync/hp-dates.ts`: `parseUtcDate` / `requireUtcDate`. A zone-less string is read as UTC
  (append `Z`). A string with an offset or `Z` keeps it. The result never depends on the
  host zone.
- Every place we parse HP dates now uses it:
  - `sync/sync-replays.ts` (daemon, `replay_draft_data`)
  - `sync/fetch-qm.ts` (`qm_games`)
  - `sync/sync-players.ts` (`player_match_history`)
- `fetch-qm.ts` `versionDateMap` also parsed DB timestamps read back as text with
  `new Date()`. It now uses the same helper.
- Writers checked and not affected:
  - `refetch-players` and `player-store` store no game time.
  - `fetch-mmr-history` writes HP's raw strings to files.
  - `fetch-replay-files.py` copies strings.
  - `scripts/generate-rating-items.ts` reads `to_char(game_date)`.
- Test: `sync/__tests__/hp-dates.test.ts` sets `TZ=America/New_York` inside the file. It
  fails with the old `new Date(s)` parse even on a UTC host. We checked this by putting the
  old parse back: 3 of 6 tests fail.

## Which rows were wrong

Ground truth is HP's per-player MMR-history pull
(`training/personalization/cache/mmr_history/shards`). It covers 9.90M replays with one
`game_date` each, and the players never disagree (0 conflicts).

`replay_draft_data` (2,512,732 rows; 2,391,669 also in the HP pull):

- The bug follows fetch time, not replay id.
  - Every row fetched before 2026-07-09 20:22:44 UTC agrees with HP exactly: 2,018,957 of
    2,018,957 matched rows.
  - Every row fetched after it is off by exactly the Eastern offset at the true game time:
    372,712 of 372,712 matched rows.
  - No row fetched after the restart is correct.
- The last correct id is 64,264,408 (stored 20:19 by the old process). The first wrong id
  is 64,264,412.
  - The id 64,264,402 used earlier was the refetch cursor at that time. Ids 64,264,402 to
    64,264,408 are correct.
  - No lower id was written after the restart, because the backfill had already reached
    its floor. So in this table the id boundary really is sharp at 64,264,412.
- DST is handled. Among matched rows, 324,835 are off by 4 h and 47,877 by 5 h. The 5 h rows
  are late uploads of winter games. Games around the 2026-11-01 change do not exist yet.
- Affected: 429,711 rows (every row with `fetched_at >= 2026-07-09 20:22:44`):
  - 372,712 take HP's time.
  - 56,999 get the time-zone reversal.

`qm_games`: all 669,323 rows. None are in the HP pull, which is Storm League only.

`player_match_history`: 16,031 of 26,425 rows were wrong.
- 15,914 take HP's time.
- 17 take the corrected corpus time.
- 100 rows (aaron#12738, 2019 to 2021, not in HP or the corpus) came from one write batch.
  Its 826 HP-checked rows were all shifted, so these 100 got the reversal.
- 2 rows from a batch whose HP-checked rows were all correct were left as they were.

No other table stores HP game times. `replay_players` and `replay_extras` have none.
`mmr_date_parsed` in `raw_extras` is HP's own Eastern string and was not touched.

## Correction

The reversal undoes exactly what JavaScript did. It reads the stored value as a UTC
instant and takes its New York wall-clock time
(`(game_date AT TIME ZONE 'UTC') AT TIME ZONE 'America/New_York'`). It handles DST both
ways and needs no guess in the November repeated hour.

Its one blind spot is a true time inside the spring-forward hour (02:00 to 03:00 on the
March change day). That wall-clock time does not exist in New York, so it comes back 1 h
late.

Dry run on the 372,712 HP-matched rows: the reversal reproduced HP on 372,697. The other 15
are all in that spring hour. Rows without an HP match that may carry this 1 h error:
- 1 in `replay_draft_data`
- 143 in `qm_games`

Steps:

1. **Backup.** `pg_dump -Fc` of `replay_draft_data`, `qm_games` and
   `player_match_history`, taken under the backup lock.
   - We restored it into a scratch DB. Row counts and `game_date` checksums matched live.
   - Copied to the NAS at `/archive-backup/hots-db/manual/tzfix_2026-10-16T1054Z.tar` and
     checked by sha256.
   - The plan CSVs, SQL and logs are in `tzfix_2026-10-16T1054Z_correction_plan.tar`
     beside it. A staging copy is in `/home/max/pgdata/staging/{backup,tzfix}`.
2. **Apply.** One transaction as the owner role, run 2026-10-16 at 11:09 UTC.
   - Held under the backfills and backup locks, with the three tables locked.
   - Before writing, it checked that the affected set still matched the plan and that
     every row still held its old value.
   - Each UPDATE matched on both the key and the old value. It checked afterwards that
     every planned row had its new value.
   - Updated rows: `replay_draft_data` 429,711, `qm_games` 669,323,
     `player_match_history` 16,031.
3. **Incremental backups.** The `updated_at` triggers mean the next incremental backup
   carries the corrected rows.

## Verification (after commit)

- `replay_draft_data` against HP: 0 disagreements in 2,391,669 overlapping rows.
- `player_match_history`:
  - Against HP: 0 disagreements in 26,306 rows.
  - Against the corpus: 0 disagreements in 6,989 rows.
- Fetched before the game time, which is impossible:

  | Rows | Before | After |
  |---|---|---|
  | `replay_draft_data`, the 429,711 corrected rows | 78,115 (18.2%) | 88 (0.020%) |
  | `qm_games` | 18,256 (2.7%) | 2 |

  - Baseline before the bug (rows fetched April 1 to July 9): 136 of 1,609,416 (0.008%).
  - 84 of the 88 remaining `replay_draft_data` rows use HP's own time.
- QM against HP's processing stamps (10% sample, 59,688 games):
  - HP had processed the game before its stored time in 14.8% of games before the fix and
    0.23% after.
  - Storm League rows before the bug: 0.50%.

## Neon

- `player_match_history` on Neon was identical to the local table, so it was wrong in the
  same 16,031 rows.
- We republished it with `sync/publish-site.ts`, the normal path. That is one transaction,
  with DELETE and INSERT on that table only. Afterwards the counts and `game_date`
  checksums match local.
- No other site table has a game time.

The expert study tables were not modified. All 621 real `rating_items` have a
`provenance.gameDate`, and all 621 are the stored DB value from before the fix, so they are
4 h late.
- After correction, none would leave the pool window [2026-09-01, 2026-09-28).
- The training snapshot cut was made on the same shifted clock, so no rated game was in
  training.
- `scripts/check-rating-pool.ts` checks that provenance equals the DB value. That check
  will now fail on these 621 items. It is expected: the item is unchanged, and the DB is now
  in UTC.

## Downstream notes

- **Paper 3, scilearn `snap_20261013` (frozen, not modified).**
  - 394,759 rows in `games.npz` and 3,302 in `pool.npz` carry the old clock in `game_ts`.
  - Their `first_seen_ts` is retrospective (`game_ts` + 1 day), so it is 4 to 5 h late
    too. It still comes after the game.
  - Tranches are set by build. The only date-based boundary is C versus F/audit at
    2026-10-27, and the latest game in this snapshot is 2026-10-12. So no tranche
    assignment changes.
  - The `game_date < 2026-10-13` cut did not bind either, because the latest stored game
    is 2026-10-12 00:43. So no row was left out.
  - Snapshots built from now on use UTC. The same replay will have a `game_ts` 4 to 5 h
    earlier than in `snap_20261013`. Compare across snapshots by replay id, not time.
  - `first_seen.npz`: 402,448 retrospective entries for affected rows keep the late
    value. That is conservative. True first-seen times from the daily diff do not depend
    on `game_date`.
- **Personalization.**
  - The waiter's export (`p3_export.py`) reads the local table directly, so it now gets
    UTC end times. Its `game_date < '2026-09-28'` cap now applies to true time.
  - `p3_full_history.merge()` used to shift corpus-only rows from the bug id on. That
    would double-correct a new export, so it now shifts only when the pairs show the
    export still carries the offset. `CLOCK_FIX_RID` is now 64,264,412.
  - `gametime_2024q2.npz` (built 2026-09-30) still carries the offset and keeps its shift.
  - The "clock skew up to ~3h" between `mmr_date_parsed` and the stored end in
    `p3_mmr_at_game.py` was most likely this bug.
  - Results from the oct05 export used the shifted clock for games after 2026-07-09.
- **Production refresh.**
  - The 90-day decay weights move by under 0.2% for a 5 h shift. The deployed models
    (October 1 run) are fine to keep.
  - The November 1 run will read the corrected table. It dumps with `force_refresh`, so
    stale caches are not used.
  - Any `.replay_cache.json` from before 11:09 UTC on 2026-10-16 has the old clock.
- **Replay-file cohorts.**
  - The selection rules use ids, builds, salts and the fetcher's own clock. None of them
    use `game_date`, so no cohort changes.
  - `manifest.jsonl` stores a `game_date` string per row. Sealed rows copy HP's listing
    string, which is correct. Rows from the DB written before the fix carry the old clock.
  - When comparing cohort B to the listing by date, shift DB-sourced rows from before the
    fix, or re-read the date from the DB.
