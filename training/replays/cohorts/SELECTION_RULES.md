# Replay-file cohorts: selection rules

Written 2026-10-13, before any cohort file was downloaded. Sections A and B are frozen:
changing them means defining a new cohort with a new name, not editing these. Section C
is the interface for cohorts that come later.

Downloader: `sync/fetch-replay-files.py`. Live config: `cohorts.json` (this directory).
Files: `/home/max/hots-replays/` on this box's RAID. Manifest: `/home/max/hots-replays/manifest.jsonl`.

## Shared facts

- Source: Heroes Profile v1 `GET /download/replay?replayID=<id>` (replay_download pool,
  25,000 per rolling week, 120 per minute). Listing: `GET /replays?after=<id>` (replay_index pool).
- Retention: files are kept for about 28 days after upload (measured 2026-10-13; see
  `training/replays/README.md`). Anything we do not fetch inside that window is gone.
- "Current patch" means `game_version` starting `2.57.`.
- Every cohort except A excludes the sealed IDs. The code enforces this; it is not a
  convention.

## A. Sealed holdout (cohort `sealed`)

Purpose: the final, untouched test set for the hypothesis-driven win-probability system.

Rule:

1. Universe U = every replay in the Heroes Profile `/replays` listing with
   `game_type == "Storm League"`, `game_version` starting `2.57.`, `deleted == 0`,
   `downloadable == true`, and replayID in `(65450000, freeze_max]`, where `freeze_max` is the
   listing's `max_replay_id` at freeze time.
2. A 32-byte random salt is drawn once at freeze time.
3. `rank(id) = sha256("<salt>:<id>")` as a hex string.
4. Sealed set = the 5,000 members of U with the lowest rank.

The salt, `freeze_max`, the universe size and the sha256 of both ID lists are in
`sealed_rule.json` (written by `freeze-sealed`, never edited). The ID list is in
`sealed_ids.txt`. U is defined from the Heroes Profile listing, not our own database, so
our own fetch history does not shape it.

Handling:

- Files go to `/home/max/hots-replays/sealed/files/`, apart from everything else.
- The pipeline fetches them and checks the sha256 and the 4-byte MPQ magic. Nothing else.
  No parsing, no opening, no summary statistics, no joins of their database rows into
  any analysis for this system.
- They unlock only by an explicit decision of the project lead (Max Segan), recorded in
  `/home/max/hots-replays/sealed/README.md` with a date.
- Fetch order is oldest replayID first, because those expire first.

Known selection notes: our local DB holds draft rows for about 84% of listed Storm League
games in the same ID range (checked on 1,000 listing rows: 836 present). The sealed set is
drawn from the listing, so it also includes games our DB lacks; when it unlocks, its tier
labels must come from a fresh fetch or from the DB rows that exist.

## B. Random stratified control pool (cohort `stratified_b`)

Purpose: the unbiased-by-design comparison pool that any "interesting" cohort is measured
against.

Rule:

1. Pool = rows of `replay_draft_data` in the local research DB with `game_version` starting
   `2.57.`, minus the sealed IDs, minus IDs that returned 403/404.
2. Cells = (`league_tier`, `game_map`), with NULL tier (Master in the site scheme) as its own
   cell. That is 6 tiers by 12 maps, 72 cells.
3. Within a cell, order by `sha256("<salt_b>:<id>")` (salt in `cohorts.json`).
4. Selection is greedy: the next download is the lowest-ranked unfetched game from the
   cell with the fewest cohort-b members so far (ties broken by a salted hash of the cell
   key). At any stopping point the cells are balanced to within one game, until a cell runs
   out, after which the rest stay balanced.
5. Volume: about a third of the weekly quota (8,300) from week 2 on. Week 1 (until
   2026-10-20) allows 15,000 so that sealed plus B reaches about 20,000.

Known selection notes:

- Equal cells over-represent rare cells (Master, small maps) relative to the ladder. Any
  population estimate from B must reweight by the cell counts in the pool, which the
  manifest records (`league_tier`, `game_map` on every row).
- B inherits the DB's coverage (about 84% of listed games). If the DB's missing 16% is not
  random (for example, games that were not parsed yet when the daemon passed), B carries
  that bias. Compare B to the listing marginals (map, region, date) when it matters.
- The pool grows as the replay daemon adds games, so later B draws come from a larger pool.
  The manifest records `fetched_at` and the cell rank, so the draw order is reconstructable.

## C. "Interesting" cohorts (criteria to come)

Interface only. A new cohort is one entry in `cohorts.json`:

```json
{"name": "comebacks_v1", "kind": "sql", "priority": 20,
 "sql": "SELECT replay_id, 'why this game' AS reason FROM replay_draft_data WHERE ...",
 "target": 3000}
```

or, for a fixed list:

```json
{"name": "hand_picked_v1", "kind": "ids", "priority": 20,
 "ids_file": "interesting/hand_picked_v1.txt", "target": 500}
```

- `sql` runs read-only against `DATABASE_URL_RESEARCH` (local only) and must return a
  `replay_id` column; an optional `reason` column is stored per row in the manifest.
- `ids_file` is one replayID per line, optionally followed by a tab and a reason. Paths
  are relative to this directory.
- `priority` orders cohorts (lower first). `window_caps` in `cohorts.json` can cap a cohort
  per quota week. `target` caps the cohort's total size.
- Sealed IDs are dropped automatically. A game already fetched for another cohort is not
  downloaded again: the manifest gets a `linked` row, so cohort membership stays exact and
  overlap with B is measurable.
- Write the criteria into this file (new subsection, dated) before the cohort's first fetch.
