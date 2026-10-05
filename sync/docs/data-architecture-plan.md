# Data architecture plan: local primary store, small Neon, incremental NAS backups

Status: Phase 1 (audit and plan), 2026-10-05. Nothing has been changed on Neon.
All Neon numbers below come from catalog queries (`pg_stat_user_tables`,
`pg_total_relation_size`) and a 0.05% `TABLESAMPLE` for row widths.

## 1. Summary

- Neon holds 72.5 GB. The website reads 31 MB of it, plus 1.5 MB for the expert study.
- 99.9% of Neon is research corpus (`replay_players` 63.3 GB, `replay_draft_data` 6.8 GB,
  `replay_extras` 1.2 GB, `replay_fetch_queue` 0.85 GB). The deployed site never reads it.
- The weekly `backup_db.py` run pulled about 62 GB of row text from Neon per run
  (about 265 GB a month). Research rebuilds pulled about 3.6 GB per corpus read, four or more times
  last week.
- Plan: run Postgres 17 on this box (local NVMe RAID) as the primary store. Every sync writer
  and every research or production reader uses it. A nightly publish step pushes the 31 MB site
  subset to Neon. Backups go to the NAS: a weekly local base dump plus daily watermark increments,
  with no Neon traffic.
- One-time migration egress: about 1.3 GB if done before the HP quota resets (around
  2026-10-08 04:30 UTC), because we reuse the 2026-10-04 NAS dump and copy only the delta.
  A fresh full pull would cost about 62 GB.
- After the change: Neon storage about 35 MB (from 72.5 GB). Neon egress is site reads plus a
  1.5 MB daily study backup, probably well under 5 GB a month.
- Effort: about 3 to 4 working days. Everything can land well before the Nov 1 production refresh.

## 2. What the website reads

The site uses `drizzle-orm/neon-http` (`src/lib/db/index.ts`). Every DB read goes through
`src/lib/data/queries.ts`, `src/components/layout/site-footer.tsx`,
`src/app/api/player-search/route.ts` and `src/app/api/ratings/{route,items/route}.ts`.
Pages `/`, `/heroes`, `/maps` and `/draft` use ISR with `revalidate = 3600`. `/rate` and the
ratings API are dynamic.

| Table | Rows | Size on Neon | Read by | Rows and columns used |
|---|---:|---:|---|---|
| hero_pairwise_stats | 49,208 | 14.0 MB | heroes, draft pages (synergies, counters, all pairs) | all rows, all columns |
| player_match_history | 26,357 | 13.0 MB | dashboard, player-search API | all rows (tracked players only); search uses `ILIKE` on battletag |
| hero_talent_stats | 5,884 | 1.5 MB | heroes page | all rows |
| player_hero_map_stats | 3,332 | 1.2 MB | dashboard, draft personal stats | all rows |
| hero_map_stats_aggregate | 3,276 | 1.2 MB | maps, heroes, power picks | all rows |
| player_hero_stats | 452 | 0.2 MB | dashboard, draft personal stats | all rows |
| hero_stats_aggregate | 273 | 0.2 MB | all pages, footer "last updated" | all rows |
| tracked_battletags | 7 | 0.1 MB | all pages | all rows |
| map_stats_aggregate | 36 | <0.1 MB | maps page, map list | all rows |
| **Site subset** | | **31.4 MB** | | |
| rating_items (study) | 901 | 1.2 MB | /rate, ratings API | all rows; `provenance` is server-only |
| draft_ratings (study) | 742 | 0.3 MB | ratings API (written by raters from Vercel) | all rows |

These tables are already small aggregates, so no further aggregation is needed.

The site reads **nothing** from `replay_players`, `replay_draft_data`, `replay_extras`,
`replay_fetch_queue`, `qm_games` or any queue or state table. The draft tool's model inputs
(hero stats, map win rates, synergies, counters) come from the static artifact
`src/lib/data/draft-stats-decayed.json`, which `training/production_refresh/refresh.py`
builds and deploys with the ONNX models. Compositions come from the static
`src/lib/data/compositions.json`. The only site aggregates derived from a big table are
`map_stats_aggregate` and `hero_map_stats_aggregate`, which `sync/derive-map-stats.ts`
builds nightly from recent `replay_draft_data`. In the new layout that query runs locally,
and the publish step pushes the result.

Tables on Neon that the site does not use: `users` (1 row), `sync_log`, `player_privacy`,
`privacy_feed_state`, `player_history_marks`, and all queue and state tables.

## 3. Who writes and reads what

Volumes come from `pg_stat_user_tables`, the NAS manifests (2026-09-27 to 2026-10-04 deltas)
and the sync logs. Row text width comes from the sample: `replay_players` 1.46 KB/row,
`replay_draft_data` 3.88 KB, `replay_extras` 0.43 KB, `qm_games` 0.13 KB,
`replay_fetch_queue` 0.08 KB.

### Writers (all run on this box)

| Job | Schedule | Writes | Reads | Volume |
|---|---|---|---|---|
| `run-sync.sh` -> `sync/index.ts` | daily 00:00 | hero_stats_aggregate, hero_talent_stats, hero_pairwise_stats, hero_map_stats_aggregate, map_stats_aggregate, player_match_history, player_hero_stats, player_hero_map_stats, tracked_battletags, users, sync_log; also `compositions.json` (file) | `replay_draft_data` (derive-map-stats, aggregated server-side), player_match_history | about 30 MB of small tables rewritten or upserted |
| `run-backfills.sh` -> `replay-daemon --cron` | every 4 h at :30 | replay_fetch_queue, replay_sync_state, replay_draft_data, replay_players, replay_extras | queue | about 49K drafts/week into `replay_draft_data` (Sep 27 to Oct 4) |
| `run-backfills.sh` -> `refetch-players --cron` | same | replay_players, replay_extras, player_refetch_state; drains player_fetch_queue | replay ids below the cursor | up to about 48K replays/day, about 480K player rows/day (0.7 GB text) while quota lasts |
| `run-backfills.sh` -> `fetch-qm --cron` | same | qm_games, replay_players, replay_extras, qm_fetch_state | | about 18K QM games/week |
| `privacy-sync.ts` | every 6 h at :45 | player_privacy, privacy_feed_state; deletes from player_match_history, player_hero_stats, player_hero_map_stats; nulls sync_log battletag. In `pseudonymize` mode it also rewrites `replay_players.blizz_id`/`battletag` (current mode is `retain`) | | a few rows per run |
| `fetch-mmr-history.ts` | hourly :20 | files only (`training/personalization/cache/mmr_history/`) | `panel.tsv` | no DB traffic |
| `enqueue-player-histories.ts` | cron paused since 2026-08-28 | player_fetch_queue, player_history_marks | replay_players, replay_draft_data | |
| `recompute-mawp.ts`, `resync-matchups.ts`, `relabel-skill-tier.ts` | manual / one-off | player tables, hero_pairwise_stats, replay_draft_data (+ backup table) | | |
| `backup_db.py` | weekly Sun 03:00, **disabled 2026-10-05** | | every table | about 62 GB per run |
| Vercel `/api/ratings` | live | draft_ratings | rating_items | KB |
| `scripts/seed-rating-items.ts`, `generate-rating-items.ts` | manual | rating_items | | KB |

`replay_players` upserts set `fetched_at = excluded.fetched_at`, so `fetched_at` already acts
as a modification watermark for that table and for `replay_extras`. `replay_draft_data` has
in-place updates that do not touch `fetched_at`: the talents upsert in `sync-replays.ts` and the
2026-09-30 skill-tier relabel. Its 1.87M lifetime updates come mostly from that relabel.

### Readers off the site (all on this box)

| Reader | Schedule | Reads | Egress per run |
|---|---|---|---|
| `training/production_refresh/refresh.py` (via `cadence.sh`) | monthly, 1st at 04:00; next run **Nov 1** | full `replay_draft_data`, 13 columns (`shared.load_replay_data(force_refresh=True)`, plus `phase_dump`) | about 3.6 GB (the `.replay_cache.json` it writes is 3.6 GB) |
| `training/shared.py` users (train_*, sweep_enriched_wp, drift, overfit2026) | ad hoc; 24 h JSON cache | same corpus | about 3.6 GB per cache miss |
| Snapshot builders (`rerun2026/make_snapshot.py`, `paper1_revision/oct2026_data.py`, `overfit2026/data.py`, `drift2026/*`) | ad hoc | `replay_draft_data` slices | 0.6 to 3.6 GB each |
| `personalization/p3_*` | ad hoc | `replay_players` joined to `replay_draft_data` | up to several GB |
| `sync/export_qm_jsonl.py`, `overfit2026/site_qm.py` | ad hoc | qm_games + QM `replay_players` rows | about 0.1 to 1 GB |
| Study analysis (`rerun2026/tournament_refresh.py`, `rating_items_ood.py`, `paper1_revision/oct2026_pool_judges.py`, `scripts/check-rating-pool.ts`) | ad hoc | rating_items, draft_ratings | KB; **must keep reading Neon** |

Last week's large corpus files written from Neon: `.replay_cache.json` (Sep 30, 3.6 GB),
`snapshots/replay_snapshot_2026-09-01_sitetiers` (Oct 1, 3.6 GB),
`replay_snapshot_2026-05-22_p1site` (Oct 1, 2.9 GB), the production-refresh corpus (Oct 2).
That is about 15 GB, on top of two weekly backups (about 124 GB).

### Other lane (HP API client)

`git log` shows no sync commits after 2026-10-02 and no uncommitted changes under `sync/`, so
nothing is in flight that this plan conflicts with today. The overlap point is `sync/db.ts`:
this plan swaps its driver, and every sync script imports it. The plan does not touch
`api-client*.ts`, `hp-api.ts` or the worker logic. Coordinate the `db.ts` change as one small
commit.

## 4. Proposed architecture

### What exists on this box

- **Postgres: not installed.** There is no `psql`, `pg_dump` or server binary. Docker is
  running (one unrelated container). Neon runs PostgreSQL 17.11 with `wal_level=replica`, so
  logical replication from Neon is off. Turning it on would need a Neon setting change, so this
  plan does not rely on it.
- **Local disk:** `/` is `md0`, a RAID0 of 4x 4 TB Samsung 9100 PRO NVMe. Size 15 TB, 2.9 TB
  free, about 1.2 GB/s sequential write (measured, 2 GB fdatasync). RAID0 has **no redundancy**:
  losing one drive loses the volume. That is why the NAS copy matters.
- **NAS:** `10.10.10.2:/mnt/tank/backup` on `/archive-backup` (NFS 4.2, hard mount), 48 TB free.
  Measured 590 MB/s write and 330 MB/s read. The existing full dumps take 19.7 GB
  (`hots-db/2026-07-21` 2.8 GB, `2026-09-27` 7.9 GB, `2026-10-04` 9.0 GB).

### Layout

```
 HP API ──> sync jobs (cron, this box) ──> LOCAL Postgres 17  (primary, full data, ~73 GB)
                                             │   PGDATA on local NVMe RAID
                                             │
                    research / production ───┤ read-only role, or frozen snapshot files
                                             │
                    publish (nightly, after run-sync and privacy-sync)
                                             │   31 MB allowlist of site tables
                                             v
                                           NEON  (site subset + live expert study, ~35 MB)
                                             ^
                                             └── Vercel site reads; raters write draft_ratings

 LOCAL Postgres ──> NAS /archive-backup/hots-db/   weekly base + daily increments (local reads only)
 NEON study tables ──> NAS                         daily full copy, about 1.5 MB
```

**Primary store: local Postgres 17 on the RAID, not on the NAS.** Running PGDATA over NFS costs
fsync latency and makes the database hang whenever the NAS mount hangs (hard mount). The NAS
works better as the backup target and the long-term file store. Same major version as Neon
(17), so row text and checksums match during verification and `pg_dump` output loads on either
side.

- Install: the `postgres:17` Docker image with `--restart unless-stopped`, port bound to
  `127.0.0.1:5433`, PGDATA at `/home/max/pgdata/hots17`, and the container pinned to cores
  48-63 at low CPU priority (per the resource cap). An apt PGDG install would also work. Docker
  avoids touching system packages on a shared box and ships `pg_dump`/`psql` inside the image.
- Settings: `shared_buffers` 16 GB, `effective_cache_size` 64 GB, `maintenance_work_mem` 4 GB,
  `max_wal_size` 16 GB, `wal_compression=zstd`. A small share of a 1 TB machine.
- Roles: `hots_sync` (read/write), `hots_research` (read-only on the corpus), `hots_publish`.
- Local-only additions (applied by a SQL file, not by `schema.ts`): an `updated_at timestamptz`
  column with a `BEFORE UPDATE` trigger on `replay_draft_data`, `replay_players`,
  `replay_extras` and `qm_games`. Then every big table has a reliable modification watermark,
  including the talents upsert and any future relabel. Adding a column with a `now()` default is
  a catalog-only change in PG 17.

**Env switch.** On this box, `.env` `DATABASE_URL` points to the local store, and a new
`NEON_DATABASE_URL` is used only by the publish job, the study backup and the study scripts.
The Vercel env does not change. That moves all 21 `sync/` entry points and about 35 training
scripts at once, with no per-script edits. Code changes:

1. `sync/db.ts`: pick the driver by URL. Use `drizzle-orm/node-postgres` (`pg` is already a
   dependency) unless the host is `*.neon.tech`.
2. `sync/privacy-sync.ts`: its two `db.batch([...])` calls are neon-http only. Wrap them in
   `db.transaction`. Also apply the three site-table `DELETE`s to Neon right away, so a privacy
   purge reaches the live site within 6 hours instead of waiting for the next publish.
3. Study scripts (`scripts/*rating*.ts`, `scripts/check-rating-pool.ts`, the three training
   study scripts) read `NEON_DATABASE_URL`. They fail loudly if it is unset.
4. `training/shared.py` and `refresh.py` need no change beyond the env. Optional: point
   `HOTS_CORPUS_PATH` at a NAS snapshot for frozen paper runs.

### Publish step (`sync/publish-site.ts`, new)

- Runs at the end of `run-sync.sh` (after derive-map-stats and compute-derived), and after each
  `privacy-sync` run.
- Hard-coded allowlist: the 9 site tables in section 2. It refuses any other table, so it can
  never touch `rating_items` or `draft_ratings`.
- Per table, in one Neon transaction: create a staging table, `COPY` the local rows in,
  `DELETE` and `INSERT` from staging, commit. Readers see the old or the new data, never a mix.
  About 15 MB of writes per night. That is ingress to Neon, not egress.
- Gate: skip a table, and log loudly, if its local row count fell by more than 20% since the
  last publish. A broken sync then cannot blank the site.

### Incremental backup to the NAS (`sync/backup_local.py`, replaces `backup_db.py`)

All reads come from the local store, so none of this costs Neon egress.

```
/archive-backup/hots-db/
  base/<YYYY-MM-DD>/            weekly pg_dump -Fd -j4 -Z zstd of the local store (+ manifest.json)
  incr/<table>/<YYYY-MM-DDTHH>.csv.zst   rows with updated_at in (last_wm - 10 min, run_start]
  small/<YYYY-MM-DD>/<table>.csv.zst     full copy of each table under 50 MB (state, queues, site tables)
  neon-study/<YYYY-MM-DD>/{rating_items,draft_ratings}.csv.zst   daily, from Neon
  watermarks.json                         last committed watermark per table
  legacy/2026-07-21, 2026-09-27, 2026-10-04   old CSV full dumps (moved, not deleted)
```

- **Append and upsert tables** (`replay_players`, `replay_draft_data`, `replay_extras`,
  `qm_games`): every 4 hours, after `run-backfills.sh`, export rows where `updated_at` is past
  the last watermark (with a 10 minute overlap; restore is an idempotent upsert). Expected size
  at full quota: about 2.5M `replay_players` rows a week, about 220 bytes/row gzipped, so about
  0.6 GB a week on the NAS.
- **Mutable or deleting tables:** the queues (`replay_fetch_queue` 0.5 GB text, about 70 MB
  compressed; `player_fetch_queue`), the state tables, `player_privacy` and all site tables are
  small. Copy them in full once a day. That covers deletes without tombstones.
- **Base:** every Sunday, a local `pg_dump -Fd -j4` (zstd) to `base/`. About 9 GB, about
  15 minutes at the measured NAS speed. The old cron slot (Sun 03:00) works, with the same
  `nice 19`, `taskset 48-63` and a lock.
- **Retention:** 4 weekly bases plus 6 monthly bases (first base of each month); increments
  back to the oldest kept base. About 150 GB total on a 48 TB volume.
- **Manifest per run:** row counts, watermark range, file sha256, and a per-table
  `count(*), sum(hashtextextended(t::text, 0))` for the base. A run writes to `.partial` and
  renames on success, like today.
- **Privacy:** purges are replayed at restore from `player_privacy` (see below). The legacy
  full dumps hold `player_match_history` rows for players who later went private. Max should
  decide how long to keep them (decision 5).
- **Optional (decision 2):** add continuous WAL archiving with pgBackRest to the NAS. That
  gives point-in-time restore with an RPO of minutes instead of up to 4 hours. Low effort,
  but one more moving part.

### Restore procedure (target: under 1 hour for the full store)

1. Start a fresh `postgres:17` container with an empty PGDATA.
2. `pg_restore -j8 -d hots base/<latest>/` (about 20 to 30 minutes including index builds).
3. For each big table, apply `incr/<table>/*` newer than the base's watermark in time order:
   `COPY` into a temp table, then `INSERT ... ON CONFLICT (pk) DO UPDATE`.
4. For each small table, `TRUNCATE` and load the latest `small/<date>/` copy.
5. Re-apply privacy: for each `player_privacy` row with state `private`, delete site rows. If
   `applied_mode = 'pseudonymize'`, pseudonymize `replay_players` again.
6. Verify row counts and checksums against the latest manifest.
7. Point `DATABASE_URL` at it and re-enable cron.
8. Neon side: re-run the publish job. If Neon itself is lost, also load
   `neon-study/<latest>/` into a new Neon branch.

Rehearse the restore once into a scratch container before shrinking Neon, then quarterly.

## 5. Migration plan

### Step A: build the local store from the 2026-10-04 dump (no Neon egress)

The 2026-10-04 dump was taken in one REPEATABLE READ snapshot that started
2026-10-04 06:59:52 UTC (`backup-db.log`; first file written 02:59:52 EDT). It holds 9.6 GB gz
and 34,570,407 `replay_players` rows. Every big file decompresses cleanly, and its line count
matches the manifest (checked 2026-10-05).

1. Create the schema with `drizzle-kit push` against the local URL, plus the local-only SQL.
2. `COPY` each CSV in (about 62 GB of text). Load before building secondary indexes. Expect
   2 to 3 hours at nice 19 on cores 48-63.

### Step B: copy the delta from Neon

Watermark: `T0 = 2026-10-04 06:00 UTC`, about an hour before the dump snapshot. The overlap is
harmless because loads are upserts.

| Table | Delta query on Neon | Rows (est.) | Egress |
|---|---|---:|---:|
| replay_players | `fetched_at >= T0` (one seq scan on Neon, compute only) | about 525K | 0.77 GB |
| replay_draft_data | `fetched_at >= T0` | about 11K | 45 MB |
| replay_extras | `fetched_at >= T0` | about 51K | 22 MB |
| qm_games | `fetched_at >= T0` | about 5K | 1 MB |
| replay_fetch_queue | full (it mutates) | 5.66M | 0.48 GB |
| all tables under 20 MB (site, state, privacy, study) | full | | about 20 MB |
| **Total** | | | **about 1.35 GB** |

The row estimates are `n_live_tup` now minus the manifest counts. The HP Replay/Data allowance
is spent until about 2026-10-08 00:30 UTC (see section 8), so the delta stays near 1.35 GB until
then. After quota returns it grows by about 0.7 GB a day. **Do this step before Oct 8**, during
the natural write pause.

### Step C: verify

Run the same query on Neon and on the local store, with `SET TimeZone = 'UTC'` on both:

```sql
SELECT replay_id / 100000 AS bucket, count(*), sum(hashtextextended(t::text, 0))
FROM <table> t GROUP BY 1 ORDER BY 1;
```

This returns about 660 buckets per table (a few KB of egress), but it costs one full scan per
big table in Neon compute. Same PG major version, so `jsonb` and `real` render identically. A
mismatched bucket gets re-copied (about 100K replay ids, so at most about 150 MB for
`replay_players`). This also catches in-place `replay_draft_data` updates since the dump, such as
the talents upsert, which the `fetched_at` delta would miss. Small tables: compare a full row
hash.

### Step D: switch writers

1. Coordinator comments out the `run-backfills.sh`, `run-sync.sh` and `privacy-sync` cron lines
   for the cutover window (about 30 minutes; I will not edit crontab myself).
2. Re-run Step B for anything written since, then re-verify the small tables.
3. Deploy the `db.ts` driver switch, the privacy-sync transaction change and the study-script
   URL change. Flip `.env` `DATABASE_URL` to local and add `NEON_DATABASE_URL`.
4. Re-enable cron. Add `publish-site` to `run-sync.sh` and the new backup jobs.
5. Run publish once by hand and diff the Neon site tables against the previous values.
6. Run Neon and local in parallel for at least 7 days. Neon big tables are frozen and unused.
   Watch the site, the study and the sync logs.

Cost after the switch: zero big-table reads from Neon. Corpus pulls for research and production
read the local store over a socket.

### Step E: shrink Neon (only after Max approves; no action taken)

Drop these tables on Neon:

| Table | Size on Neon |
|---|---:|
| replay_players | 63.27 GB |
| replay_draft_data | 6.82 GB |
| replay_extras | 1.22 GB |
| replay_fetch_queue | 0.85 GB |
| replay_draft_skill_tier_backup_20260930 | 0.13 GB |
| qm_games | 0.12 GB |
| player_fetch_queue | 31 MB |
| player_history_marks | 0.5 MB |
| sync_log, player_privacy, users, qm_fetch_state, player_refetch_state, replay_sync_state, privacy_feed_state | about 0.9 MB |
| **Total** | **about 72.4 GB** |

Keep on Neon: the 9 site tables (31 MB) plus `rating_items` and `draft_ratings` (1.5 MB).
Neon goes from 72.5 GB to about 35 MB plus history. Neon keeps the dropped data in its PITR
history until the retention window passes, so billed storage falls after that window.

Dropping the small bookkeeping tables too means a writer still pointed at Neon fails loudly
instead of quietly writing to the wrong place. Guard: `drizzle-kit push` against Neon would
recreate dropped tables from `schema.ts`. Add a `tablesFilter` (site and study tables only) to
a Neon-specific drizzle config before the drop.

## 6. Estimates

| | Now | After |
|---|---:|---:|
| Neon storage | 72.5 GB | about 35 MB (plus PITR history until it ages out) |
| Neon egress, backups | about 62 GB/week | 0 (local), plus 1.5 MB/day study copy |
| Neon egress, research and production | about 3.6 GB per corpus read, about 15 GB last week | 0 |
| Neon egress, site | small; bounded by about 31 MB of tables times ISR regenerations | unchanged, likely under 5 GB/month |
| Neon writes (ingress) | sync writes, about 0.7 GB/day at full quota | publish, about 15 MB/day |
| One-time migration egress | | about 1.35 GB before Oct 8, about 0.7 GB/day more after |
| Local disk | | about 75 GB now, +4 GB/week at full quota; 2.9 TB free |
| NAS | 19.7 GB of full dumps | +9 GB/week of bases (rotating), +0.6 GB/week of increments |

I could not read Neon's own transfer meter: there is no Neon API key in `.env`. The Neon
console's usage page will confirm the before and after numbers.

Effort:

| Work | Time |
|---|---|
| Local Postgres + load from dump (Step A) | 0.5 day, plus 2 to 3 hours of loading |
| Delta copy + verification (B, C) | 0.5 day |
| `db.ts` driver switch, privacy-sync transaction, study scripts on `NEON_DATABASE_URL`, tests | 0.5 to 1 day |
| `publish-site.ts` with gates | 0.5 day |
| `backup_local.py` (base, increments, small, study) + restore rehearsal | 1 day |
| Shrink Neon (after approval) | 15 minutes |

Risks and mitigations:

- **Expert study (live).** The study tables never leave Neon. Publish has an allowlist that
  excludes them. Study scripts move to an explicit `NEON_DATABASE_URL`. The study has no local
  copy, so nothing can read stale ratings by accident. Daily NAS copy from Neon.
- **Site uptime.** The site keeps reading Neon throughout. Publish swaps tables in one
  transaction, and the row-count gate stops a bad sync from blanking pages. If this box or the
  local store dies, the site keeps serving the last published data (stale, not down).
- **Sync continuity.** Cutover needs about 30 minutes with cron paused. The workers are
  checkpointed and idempotent (`--cron` mode, queue and upserts). Best done while they are
  quota-benched (now until about Oct 8).
- **Nov 1 production refresh.** `cadence.sh` reads `DATABASE_URL` from `.env`, so after the
  switch it reads the local store with no code change. If the switch slips past Nov 1, the
  refresh still works against Neon at about 3.6 GB of egress. Plan to switch by Oct 20 and do a
  dry run of `refresh.py data` against local before Nov 1.
- **RAID0.** The primary sits on a volume with no redundancy. Mitigated by the 4-hourly NAS
  increments (RPO 4 hours, or minutes with WAL archiving) and the restore rehearsal.
- **Resource cap.** Postgres is pinned to cores 48-63 with modest memory. Loads and base dumps
  run at nice 19 with `ionice -c3`.
- **Two copies drifting during the parallel week.** After the switch, Neon's big tables are
  frozen. Nothing writes them, and Step E drops them.

## 7. Decisions needed from Max

1. **Primary store:** local Postgres 17 in Docker on the NVMe RAID, with the NAS as backup and
   file store (recommended). The alternative, PGDATA on the NAS, is not recommended.
2. **Backup depth:** watermark increments every 4 hours plus a weekly base (recommended), with
   or without continuous WAL archiving (pgBackRest) for minute-level restore.
3. **Timing:** run Steps A to C before about Oct 8 to keep migration egress near 1.35 GB, and
   switch writers by about Oct 20, ahead of Nov 1.
4. **Neon shrink list:** approve dropping the tables in Step E (about 72.4 GB), after 7 days of
   parallel running and a successful restore rehearsal.
5. **Legacy full dumps on the NAS** (2026-07-21, 09-27, 10-04): keep them until the first base
   and restore rehearsal succeed, then keep only 2026-10-04? They contain site rows for players
   who later opted out under the HP privacy feed.
6. **Research access:** read-only role on the local store for ad hoc work, plus frozen snapshot
   files on the NAS for paper runs (recommended), or snapshots only.

## 8. Side finding: refetch-players exits in about 2 seconds

**Not broken and not done: it is out of HP API quota.** Since 2026-10-05 05:19 UTC every run
logs `v2: Replay/Data quota exhausted`, then `All API keys quota-benched — checkpointing and
exiting (cron mode)`. The replay daemon and `fetch-qm` log the same thing. The checkpoint is
intact (cursor 46,633,655, 2,522,264 replays processed, 0 errors).

The single v1 key's Replay/Data allowance is 250K calls per rolling 7 days. Live v1 calls began
around 2026-10-01 00:30 UTC. From Oct 1 to Oct 4, refetch-players alone ran 24 batches of 2,000
replays a day (about 48K/day), and the daemon and QM worker shared the same pool. Together they
spent the week's allowance in about 4.2 days. Calls should come back around 2026-10-08
00:30 UTC; the first backfills run after that is 2026-10-08 04:30 UTC. Nothing needs changing.
(The `ETA 250K: -342h` figure in the log is a cosmetic display bug.)
