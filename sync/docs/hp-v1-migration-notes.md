# Heroes Profile API v1 migration notes (2026-08-24)

Deadline: old API dies 2027-01-01. Activation of live data on the account
page EXPIRES THE OLD KEY IMMEDIATELY — it is the last step, after the port
is tested against fixtures. Until activation, v1 returns fixture data
(header `X-HP-Data-Source: fixture`) at no quota cost.

Base URL: https://www.heroesprofile.com/api/external/v1
Auth: `Authorization: Bearer $HEROES_PROFILE_V2_API_KEY` (never ?api_token=)

## Endpoint mapping for OUR integration

| Ours (old) | v1 | Parsing changes |
|---|---|---|
| Replay/Data (refetch, fetch-qm, daemon) | `GET /replay/{replayID}` (id in PATH) | One match object: `players` is array-of-array (per team!), plus `replay_bans`, `draft_order`, `winner`; `game_map` is an OBJECT; `game_length` always seconds; full battletags incl. discriminator |
| Replay/Min_id | `GET /replays?after=` | Cursor is EXCLUSIVE (`after` = last seen id; old min_id was inclusive). `{replays, next_after, max_replay_id}`; stop when next_after null. Listing has `parsed`/`deleted`/`downloadable` — NO `valid` field (drop that check). Page size observed 25 in fixtures |
| Replay/Max | folded into `/replays` as `max_replay_id` | no separate call |
| Player/Replays (enumerator) | `GET /players/matches` | Laravel pagination: rows under `data`, `pagination_page` param, `per_page`/`total` present. Replaces our 6-month date-window chunking entirely |
| Heroes/Stats (daily stats sync) | `GET /heroes/stats` | Named object with `data` rows + averages; **may answer 202 + job** (see below) |
| Heroes/Matchups | `GET /heroes/matchups` | Split into `ally` / `enemy` / `combined`; heroes are nested objects; may 202 |
| Player, Player/Hero/All, Player/MMR | `/players`, `/players/heroes`, `/players/mmr` | Named objects; hero is nested object not id; league_tier now carries sub-tier ("Diamond 2") |
| Patches | `/patches` | Flat list `{patches:[...]}` with season; only stats-queryable patches listed |

## Cross-cutting gotchas

- **Per-endpoint allowances.** Quota is now per endpoint family (replay_data,
  replay_index, heroes_stats, player_match_history, ...). Error example in
  docs: "Weekly limit of 1,000 calls reached for this endpoint." ACTUAL
  numbers are plan-dependent — read them off the Billing page BEFORE
  activation and rewrite training/QUOTA_ALLOCATION_PLAN.md accordingly.
- **202 + job pattern** (global stats only): 202 → follow `Location`, honor
  `Retry-After` (poll free), 200 body = the data; 404 job = restart call;
  500 job = query failed. Treat 202 as success-in-progress.
- **Errors**: real status codes + envelope `{error:{code,message}}`
  (401 no key, 403 not in plan, 422 bad param, 429 quota/rate).
  Old API returned errors as HTTP-200 prose — all our "Max calls"/
  "non-JSON" string matching must become status-code checks.
- Names everywhere (hero/map/game_type/region names, not ids);
  `sl`/`Storm League` both fine.
- game_length: seconds everywhere now.
- Rate limit headers present (`x-ratelimit-limit: 120` seen in fixture mode;
  real per-key rate TBD from Billing page).

## New v1 endpoints valuable to the research (no old equivalent)

- `/players/mmr/history` (+ /heroes, /roles): rating over time, ONE ENTRY
  PER MATCH. Directly fixes the personalization caveat that replay_players
  MMRs are as-of-parse, not at game time.
- `/players/friendfoe`: per-teammate/opponent aggregates (duo stats
  server-side).
- `/compositions`, `/draft` (pick order/position stats), `/party`,
  `/leaderboard`, talent builder.

## Port plan

1. `sync/api-client-v2.ts`: Bearer auth, base URL, error-envelope handling,
   202-job helper, rate headers. Callers behind `HP_API=v2` env flag.
2. Adapt parsers: replay detail (per-team players array), replays cursor,
   players/matches pagination, heroes stats/matchups shapes.
3. Fixture test suite asserting `X-HP-Data-Source: fixture` + shape
   invariants for every endpoint we use.
4. Read real allowances from Billing → rewrite quota plan.
5. Activate live data (kills old key) → flip flag → monitor first cron
   cycle end-to-end.

Timing: port this week against fixtures; ACTIVATE after the dense-window
backfill completes on the old key (~first week of September).
