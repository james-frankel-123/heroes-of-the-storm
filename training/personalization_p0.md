# Personalization P0 — player-identity data feasibility (definitive)

Probe run 2026-07-09 (5 Replay/Data calls, read-only otherwise). Verdict up front: **GO.**
The ladder API returns full 10-player identities per replay; a 250-500K-game player-labeled
starter corpus is achievable in 1.2-2.3 weeks with existing infra (first ~250K in well under a
week — this week's Replay/Data budget is unspent, so the fetch is front-loaded); full-corpus
refetch is ~9-10 weeks (Max's 8-week estimate is nominal-quota math; realized throughput says
9-10).

> **STATUS UPDATE 2026-07-09: Phase 1 is LAUNCHED.** `replay_players` + `replay_extras` are
> migrated (schema differs from the sketch below — Max's call: "save everything we can this
> time," so the full per-player `scores` payload is stored as `scoreboard` jsonb, plus talents
> and a `raw_extras` jsonb; replay-level discards go to `replay_extras`). Worker:
> `sync/refetch-players.ts`, running detached; status: `npx tsx sync/refetch-players.ts --status`.
> The daemon's normal new-game path now also writes players going forward.
> Every landed row joins `replay_draft_data` on `replay_id` immediately — the starter corpus is
> usable the moment rows arrive, no end-of-phase reprocessing.

## (a) What we have today — confirmed

- `replay_draft_data` (2,082,991 rows; 2,075,325 on patch 2.55; replay_id 41,046,538-64,264,295;
  2019-03 → today): **no player identity fields.** Schema stores draft order, comps, bans, winner,
  map/tier/MMR, talents keyed by hero only (`src/lib/db/schema.ts:265`). The daemon's parser
  (`sync/sync-replays.ts:325`) iterates the per-player entries and **discards the keys, which are
  the battletags** — the identities have been flowing through our pipeline the whole time.
- `player_match_history` covers only ~8-10 tracked battletags (Lumi#1912: 10,657 games;
  SirWatsonII#1400: 5,397; AzmoDonTrump#1139: 3,243; Maballsies, Django, aaron, Mestupidum, ...).

## (b) Probe results — exactly what Replay/Data returns per player

Fetched 5 known replay_ids spanning the corpus (41,046,538 / 2021 r1; 42,329,711 / 2022 r2;
62,450,141 / 2025 r2; 62,379,752 / 2026 r1; 64,264,295 / today r1). Identical structure in all
five, both regions, oldest to newest:

- Top level: `fingerprint, game_type, game_date, game_length, game_map, game_version, region,
  draft_order, <10 player entries>, experience_breakdown`.
- **Each player entry is keyed by full battletag** (e.g. `"Atreyu209#1875"`) and contains:
  - `blizz_id` (stable numeric account id — the join key we should canonicalize on; battletags can change)
  - `hero`, `team` (0/1), `winner` (bool), `hero_level`, `mastery_taunt`
  - `party` (0 = solo; shared non-zero value marks premade groups — duo-chemistry ready)
  - `player_mmr`, `hero_mmr`, `role_mmr` + `mmr_date_parsed` — **caution: MMR is as-of-HP-parse,
    not at game time** (the 2021 replay carries a 2024 mmr_date). Usable as a static skill covariate,
    not a time series.
  - `talents` (per-tier), `scores` (~40 performance stats: takedowns, hero damage, healing, XP, ...)
- Region is top-level (per-replay), not per-player. No per-player slot field: map players to draft
  picks via hero name (unique per game) — `draft_order.player_slot` covers pick order.

So: battletag ✓, blizz_id ✓, hero mapping ✓, region ✓ (replay-level), plus party, MMR triple,
and full scoreboard we didn't even ask for.

## (c) Quota verification and refetch math

**Memory's "25K/week/key" is stale.** Key 1 was upgraded to a developer account:
`sync/replay-daemon.ts` (commit 1d8b4d4, 2026-04-07) runs it at 180/min citing a **250K/wk
Replay/Data quota**; key 2 is an Intermediate account (55/min, 25K/wk). Public docs no longer publish
quota tables (404/403), so the best evidence is empirical: **peak realized weeks of 225,308 /
219,411 / 215,026 rows stored** (weeks of 2026-03-16, 05-11, 05-04) — impossible under 50K/wk
combined, consistent with ~275K/wk nominal minus discovery calls and invalid replays.

- **Full refetch:** 2.083M replays ÷ ~215-225K/wk realized ≈ **9.3-9.7 weeks**, call it **~10**
  after reserving ~15-19K/wk for the daemon's new-game intake (current SL arrival rate).
  Max's ~8 weeks = 2.08M ÷ 275K nominal = 7.6; correct to **9-10 weeks**.
- **Starter corpus:** 250K in ~1.2 wk; ~430-450K in 2 wk. At today's ~15-19K/wk game arrival,
  430K recent games reach back ~5-6 months (builds 2.55.14-2.55.16) — two-plus builds of
  fresh-meta data.

### Incremental plan (recommended)

**Order: most-recent-first** (`ORDER BY replay_id DESC`). Rationale: (i) freshest meta matches the
deployed MAWP models and the hotsfever deliverable; (ii) active players recur, so per-player panel
density accrues automatically; (iii) "highest-density-players-first" is not directly orderable —
Replay/Data can't target players, so density ordering only exists *after* a seed pass (and is then
better served by Player/Replays, see (d)).

**Mechanism: reuse the talent-backfill pattern** (`backfillTalents` in `sync/sync-replays.ts`):
re-queue rows lacking player coverage into `replay_fetch_queue`; `fetchReplayData` already
re-fetches with `onConflictDoUpdate`. Extend it to also upsert players, and make the daemon's
normal new-game path write players going forward (free ~15-19K/wk accrual, zero extra quota).

**Storage: sidecar table, not a jsonb column** — per-player panel queries need an index on
blizz_id; a jsonb column on the 2M-row hot table gives neither indexability nor cheap backfill.
The sidecar also means **labels join existing drafts on `replay_id` the moment they land** —
no touch to `replay_draft_data`. Original sketch (now superseded by the migrated schema in
`src/lib/db/schema.ts` — see status box above):

```ts
export const replayPlayers = pgTable('replay_players', {
  replayId: integer('replay_id').notNull(),          // FK → replay_draft_data
  blizzId:  bigint('blizz_id', { mode: 'number' }).notNull(),
  battletag: varchar('battletag', { length: 64 }).notNull(),
  team: smallint('team').notNull(),
  hero: varchar('hero', { length: 40 }).notNull(),
  winner: boolean('winner').notNull(),               // denormalized for panel queries
  party: bigint('party', { mode: 'number' }),        // 0 = solo
  heroLevel: integer('hero_level'),
  playerMmr: real('player_mmr'), heroMmr: real('hero_mmr'), roleMmr: real('role_mmr'),
  mmrDate: timestamp('mmr_date'),                    // as-of-parse, NOT game time
}, (t) => ({
  pk: primaryKey({ columns: [t.replayId, t.blizzId] }),
  playerIdx: index('replay_players_blizz_idx').on(t.blizzId, t.replayId),
  tagIdx: index('replay_players_tag_idx').on(t.battletag),
}))
```

~10 rows/replay → 4.3M rows for the 2-week starter, ~21M full — fine for Neon. The sketch's
"omit `scores`" call was **overruled at review** ("save everything we can this time"): the
migrated table stores `talents` jsonb, the full ~40-stat `scores` payload as `scoreboard` jsonb,
leftover per-player fields in `raw_extras` jsonb, and discarded replay-level fields
(fingerprint, experience_breakdown, ...) in a `replay_extras` sidecar.

## (d) Cheaper per-player endpoint — yes, and it changes the plan

`Player/Replays` (already wrapped: `getPlayerReplays` in `sync/api-client.ts:202`) returns a
player's match history — replay_id, hero, map, win, date, KDA/damage/healing/XP, talents, rank.
Evidence: Lumi#1912's 10,657-game history sits in `player_match_history` from routine syncs.
Two corrections from quota verification: (1) it is **not** "entire history in one call" at scale —
**the endpoint times out on large histories** (Lumi#1912's full-history request 500-errors;
routine syncs succeeded because they pass `start_date` windows). Use **chunked date-window
requests**, which multiplies calls per heavy player. (2) Quota is **5K calls/wk on the dev key +
500/wk on the Intermediate key** — a **separate pool from Replay/Data**, so panels don't compete
with the refetch. Draft context is recovered by joining replay_id to `replay_draft_data` — our
corpus is a near-census of SL in the ID range, so join rates should be high.

**Cost comparison per player-game observation:**

| | Replay/Data refetch | Player/Replays panel |
|---|---|---|
| Yield per call | 1 replay = 10 player-games, full roster | 1 player's history window (chunked; times out unchunked on heavy players) |
| Needs identities up front | no | **yes** (battletag + region) |
| Roster/party/opponents | full | target player only |
| MMR fields | yes (snapshot) | no (rank string only) |
| Quota | ~275K/wk nominal combined, 215-225K realized (binding) | **5K/wk (dev) + 500/wk (Intermediate)** — separate pool from Replay/Data |

**Recommended hybrid:** Phase 1 — recent-first Replay/Data refetch (identity census + full rosters
+ party ids): first ~250K in **under a week** (front-loaded, current week's budget unspent),
250-450K within 1-2 weeks. Phase 2 — rank harvested blizz_ids by frequency; fetch the top
~10-20K players via Player/Replays over **~2-4 weeks** (5.5K calls/wk combined, more calls per
heavy player due to date-window chunking) → multi-year longitudinal panels for exactly the
players the models will serve, likely millions of joined player-games. Because the quota pools
are separate, **Phase 2 runs interleaved with Phase 1 from day one**, not after it. Targeted
spot-refetch of a specific player's older games (enumerate their replay_ids, prioritize in the
Replay/Data queue) covers "deep history, fully rostered" for individual players on demand.
Phase 3 (only if an experiment demands full rosters corpus-wide) — continue the backfill to
~10 weeks total.

## Launch checklist — resolved 2026-07-09

Formerly "do not start yet"; all four gates cleared: (1) Player/Replays quota confirmed
(5K/wk dev + 500/wk Intermediate, separate pool; endpoint needs chunked date windows on heavy
players); (2) `replay_players` + `replay_extras` migrated; (3) canonicalization decided —
**blizz_id primary** (PK is (replay_id, blizz_id)), battletag stored for display/joins but
non-unique across accounts, so opponent-identity features must disambiguate via blizz_id;
(4) refetch runs as a **standalone detached worker** (`sync/refetch-players.ts`) rather than
inside the daemon — the daemon keeps new-game priority untouched, and its new-game path now
writes `replay_players` rows too (free ~15-19K/wk accrual). Status command:
`npx tsx sync/refetch-players.ts --status`.
