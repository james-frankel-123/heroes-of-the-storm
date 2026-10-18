# Replay files: download, cohorts, timelines

Full `.StormReplay` files from Heroes Profile, parsed into per-game timelines. Since
2026-10-18 they feed personalization (paper 2): per-player timeline features such as deaths
before 10:00, death positions, early takedown participation and APM. They were first
collected for the hypothesis-driven win-probability system (paper 3), which was cancelled
on 2026-10-17. The pipeline has five parts.

| part | file | runs on |
|---|---|---|
| downloader, cohorts, NAS backup, retention probe | `sync/fetch-replay-files.py`, `sync/run-replay-files.sh` | this box, cron, nice 19, cores 48-63 |
| selection rules (frozen) and cohort config | `training/replays/cohorts/` | - |
| daily ids batch for cohort `p2_panel_v1` | `build_p2_panel.py` | this box, cron 07:45, nice 19, cores 48-51 |
| parser to per-game timelines | `parse_replay.py`, `run_parse.py` | 3090 cores 24-35 (`remote_parse.sh`), or locally |
| validation against our DB | `validate_timelines.py` | this box (local research role) |

The downloader is Python, not TypeScript: it sits beside `sync/backup_local.py` (same tar
bundle style), needs only `requests` and `psycopg2`, and the parser is Python anyway.

## Storage

Everything lives on the RAID under `/home/max/hots-replays/` (never in git):

```
raw/<build>/<replayID>.StormReplay          every non-sealed cohort (B, surprise_v1, p2_panel_v1, ...)
sealed/files/<build>/<replayID>.StormReplay cohort a (fetch + checksum only)
sealed/{README.md, sealed_rule.json, sealed_ids.txt, universe.jsonl}
manifest.jsonl     one line per download attempt or cohort link
state.json         last quota headers, window end
timelines/<build>/<replayID>.json.gz, timelines/index.jsonl, timelines/validation.json
retention_probes.jsonl
```

Manifest fields: `replay_id, cohort, reason, status (ok | linked | http_403 | http_404 | bad_file | ...),
charged, window, fetched_at, game_version, build, game_map, game_date, region, league_tier,
skill_tier, size, sha256, path`. `linked` means the file was already on disk for another
cohort, so cohort membership is exact without a second download.

## Quota and schedule

- replay_download pool: 25,000 per 7-day window that starts at the first call
  (`x-hp-quota-reset` counts down to its end), 120 per minute. The downloader paces at
  about 90 per minute, stops at 1,000 remaining, honours `Retry-After` on 429, and resumes
  from the manifest.
- 403 (outside retention) and 404 are recorded and never retried; neither was charged in
  testing.
- Caps per window are in `cohorts/cohorts.json`. Week 1 (window ending 2026-10-20 13:04
  UTC): sealed 5,000, cohort B 15,000, surprise_v1 508, then p2_panel_v1 takes what is left
  under 23,000 (about 2,490). From 2026-10-20: p2_panel_v1 8,000 per week, B and
  surprise_v1 0, 24,000 total with the 1,000 margin, so about 16,000 a week is headroom
  for ad hoc cohorts (add a cohort and a cap to `cohorts.json`, and a dated rule to
  `SELECTION_RULES.md` before its first fetch).
- Cron: `run-replay-files.sh run` every 2 hours (`flock -n`, 115-minute timeout), and
  `run-replay-files.sh backup` on Sundays at 06:00 (bundle, then a retention probe).
  `build_p2_panel.py` runs daily at 07:45 and appends one batch to
  `cohorts/interesting/p2_panel_v1.txt`, which the next download run fetches.

## Cohorts

| cohort | rule | status (2026-10-18) |
|---|---|---|
| `sealed` | `SELECTION_RULES.md` A | 5,000 fetched. Paper 2's final holdout: never parsed, opened or selected on |
| `stratified_b` | B | 15,000 fetched and parsed; closed (cap 0) |
| `surprise_v1` | C.1 | 895 (508 fetched, 387 linked to B), parsed; closed |
| `p2_panel_v1` | C.2 | about 1,140 a day from 2026-10-18 (first batch 2,400) |

`p2_panel_v1` picks new 2.57+ Storm League games from our DB (played in the last 21 days,
not yet on disk, not sealed) with the most players whose full Heroes Profile history is
pulled (`training/personalization/cache/mmr_history/panel*.tsv` plus `done.jsonl`), k of
10, at least 3, highest k first, oldest upload first among ties. It reads no outcomes.
Every batch is logged with its pool and k histograms in `p2_panel_v1_batches.jsonl`.

With paper 3 gone, its tranche rules (M-explore, M-test, pool, C, F) no longer restrict
anything: every non-sealed timeline is open to personalization. The timeline files are
plain files under `timelines/`; nothing in `training/scilearn` gates reading them (its
quarantine lives in the broker and its private store). One thing to change when timeline
features are built: `training/personalization/p3_full_history.py` drops every row with a
game end on or after 2026-09-28 (`CAP_TS`), which keeps all 2.57 games out of the history
store. That cap was there for paper 3's tranches. A timeline channel must read timelines
directly, or the cap must become a sealed-ID filter (the HP shards hold rows for sealed
games, which must stay out).

## Backups

`fetch-replay-files.py backup` writes one tar per week to `/archive-backup/hots-replays/`:
`raw_<stamp>.tar` (every new non-sealed file) and, when there are new ones,
`sealed_<stamp>.tar`. Each tar holds `bundle_manifest.json` (path, sha256, size per file)
and a copy of the manifest and frozen files under `meta/`. A `<archive>.json` sidecar has
the archive's own sha256. Each file's sha256 is checked as it goes into the tar. No trees
of small files on the NAS.

## Retention window (measured 2026-10-13)

Heroes Profile keeps files for about **28 days after upload**. Retention follows the upload
order (replayID), not the game date: on 2026-10-13 at 13:10 UTC, every listed replay with ID
>= 65,189,949 was downloadable and every one below was not. The boundary replays were
uploaded at about 2026-09-15 13:10, exactly 28 days earlier. A second probe 11 minutes
later found the boundary at 65,189,979 (upload 2026-09-15 13:21), so the window rolls
continuously. Late uploads of old games
(game dates in 2024 and 2025) above the boundary are still downloadable. The listing's
`downloadable` flag agrees with the download endpoint (10 of 10 probes), so
`fetch-replay-files.py probe-retention` finds the boundary with about 20 listing calls and
no download quota. The weekly backup job logs a probe to `retention_probes.jsonl`.

What it means:

- There is no backfill. Anything before 2026-09-15 uploads is gone, including all of 2.55.
- New Storm League uploads run at about 1,000 per 4,300 replayIDs, roughly 35-40K per
  week, more than the 25K download quota. We cannot keep everything, so the cohort rules
  decide what survives.
- The first 2.57 games (uploaded from 2026-09-29) expire from about 2026-10-27. The sealed
  set fetches oldest first and finishes in week 1. Interesting cohorts that need early-patch
  games must be defined before then.

## Timelines

`parse_replay.parse(path)` returns one dict per game (about 14 KB gzipped):

- `players`: pid (1-10), team (0/1), toon_id (= `replay_players.blizz_id`), hero, win,
  `talents` (tier, internal id, display name, time), `levels` (level, time),
  `regen_globes`, `positions` (hero position every 15 s), `end` (score screen,
  end-of-game XP breakdown, time dead, talent choices). No names.
- `team_levels`, `xp_periodic` (per team, per minute: minion/creep/structure/hero/trickle XP).
- `deaths`: time, victim pid, killer pids, position.
- `structures`: forts, keeps, towers, wells, gates, walls, core (time, type, owner team, killer, position).
- `mercs`: camp captures (time, camp, type, team, position).
- `map_events`: every other stat event (map objectives: tributes, shrines, altars, immortals and so on), raw fields.
- `notable_units`: born/died of objective, boss, merc and similar units (for example GardenTerror).
- `unit_type_counts`: born counts per unit type per team, for later discovery.
- `actions_per_min`: per player rates of commands, retargets, camera moves, key presses,
  pings, chat, from game events (about 1 s of the 1.5 s parse; `--no-actions` skips it).
- Header: version, build, protocol used, map_id, length_s, gates_open_s, winner_team, bans, picks.

Times are seconds since replay start (`gameloop / 16`).

Protocol: PyPI heroprotocol 2.55.4.91756 plus a shim for the removed `imp` module. GitHub
master stops at 96477, which is identical to 91756 apart from comments, and 91756
decodes 2.57 (builds 98285-98348) fully.

Hero and talent display names come from `ref/heroes_talents_ref.json`, built by
`build_reference.py` from heroespatchnotes/heroes-talents (MIT) at a pinned commit (data to
2.55.3). Newer talents fall back to matching the DB name inside the internal id.

## Running the parse

```bash
training/replays/remote_parse.sh setup            # once: venv with heroprotocol on the 3090
training/replays/remote_parse.sh push             # code + cohort files (default p2_panel_v1)
training/replays/remote_parse.sh run              # taskset 24-35, nice 19, 12 workers
training/replays/remote_parse.sh status
training/replays/remote_parse.sh pull             # timelines back to the RAID
set -a; . ./.env; set +a; python3 training/replays/validate_timelines.py
```

`remote_parse.sh cycle` does all of that; cron runs it daily at 22:40 for cohort
`p2_panel_v1` (`flock`, `sync/logs/replay-parse.log`). B and surprise_v1 are fully parsed
(15,508 timelines), so their cycles were dropped on 2026-10-18; `run` still re-lists every
pushed cohort and skips files already parsed. On the 3090's cores 24-35 a game takes about 7 s per worker
(those are busy SMT siblings), so 12 workers do about 6,000 games per hour.

Each worker holds one replay at a time (about 25 MB). Locally:
`nice -n 19 taskset -c 48-63 <venv>/python training/replays/run_parse.py --workers 4`.

## Validation against our DB (first 617 cohort-b games, 2026-10-13)

| check | agreement |
|---|---|
| games found in `replay_draft_data` | 617 / 617 |
| winner | 100% |
| hero sets per team | 100% |
| players matched on toon_id = `replay_players.blizz_id` | 6,133 / 6,170 (the other 37 are missing from `replay_players`: those games have 8 or 9 rows) |
| per player hero, team, win | 100%, 100%, 100% |
| talent count per player (non-empty DB talents) | 99.64% |
| talent names, by tier | 99.80% of 39,740 (38,778 via the reference, 881 by name inside the internal id) |
| internal talent id maps to one DB name | 100% of 1,880 ids |
| game length | replay length = DB `game_length` + 70 s, offset between 70.0 and 70.9 s in 100% of games |

The 81 talent misses are all reference gaps, not parse errors: Muradin's reworked
`MuradinExtraCharges` (DB: Mountain King) and three Xal'atath talents, which postdate
the reference data. `validate_timelines.py` writes the full numbers to
`timelines/validation.json`.
