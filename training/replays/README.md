# Replay files: download, cohorts, timelines

Full `.StormReplay` files from Heroes Profile feed a future hypothesis-driven,
interpretable win-probability system. The pipeline has four parts.

| part | file | runs on |
|---|---|---|
| downloader, cohorts, NAS backup, retention probe | `sync/fetch-replay-files.py`, `sync/run-replay-files.sh` | this box, cron, nice 19, cores 48-63 |
| selection rules (frozen) and cohort config | `training/replays/cohorts/` | - |
| parser to per-game timelines | `parse_replay.py`, `run_parse.py` | 3090 cores 24-35 (`remote_parse.sh`), or locally |
| validation against our DB | `validate_timelines.py` | this box (local research role) |

The downloader is Python, not TypeScript: it sits beside `sync/backup_local.py` (same tar
bundle style), needs only `requests` and `psycopg2`, and the parser is Python anyway.

## Storage

Everything lives on the RAID under `/home/max/hots-replays/` (never in git):

```
raw/<build>/<replayID>.StormReplay          cohorts b, c, ...
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
- Caps per window are in `cohorts/cohorts.json`. Week 1 (to 2026-10-20): sealed 5,000,
  then cohort b up to 20,000 in total. From week 2: cohort b 8,300 per week, 24,000 total,
  leaving the rest for "interesting" cohorts.
- Cron: `run-replay-files.sh run` every 2 hours (`flock -n`, 115-minute timeout), and
  `run-replay-files.sh backup` on Sundays at 06:00 (bundle, then a retention probe).

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
uploaded at about 2026-09-15 13:10, exactly 28 days earlier. Late uploads of old games
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
training/replays/remote_parse.sh push             # code + cohort b files (rsync)
training/replays/remote_parse.sh run              # taskset 24-35, nice 19, 12 workers
training/replays/remote_parse.sh status
training/replays/remote_parse.sh pull             # timelines back to the RAID
set -a; . ./.env; set +a; python3 training/replays/validate_timelines.py
```

Each worker holds one replay at a time (about 25 MB). Locally:
`nice -n 19 taskset -c 48-63 <venv>/python training/replays/run_parse.py --workers 4`.
