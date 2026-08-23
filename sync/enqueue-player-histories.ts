/**
 * Player-history enumerator (personalization longitudinal depth).
 *
 * Spends the Player/Replays quota (its OWN weekly pool: ~5K calls on key 1
 * + 500 on key 2, separate from Replay/Data) to enumerate the historical
 * replay ids of high-value panel players, and inserts the ones we have not
 * yet covered into player_fetch_queue — which sync/refetch-players.ts
 * drains BEFORE its blind descending scan. Each queued replay contains at
 * least one target player by construction (and often several, since
 * high-count players cluster), so panel depth per Replay/Data call is a
 * multiple of blind scanning. See training/QUOTA_ALLOCATION_PLAN.md.
 *
 * The endpoint 500s on large histories, so requests are CHUNKED by date
 * window (6-month slices back to 2021-12); a chunk that still fails is
 * halved once, then skipped with a log line.
 *
 * Target selection: panel players with >= MIN_GAMES games in replay_players
 * (ranked-joined), skipping players already enumerated (marked in
 * player_history_marks). Two modes:
 *   default      — heaviest players first (the right SEED for the queue)
 *   --stratified — equal counts from each activity-tier quartile (used for
 *                  the 3-5K expansion: heavy players are systematically
 *                  unrepresentative — more engaged, better, flatter learning
 *                  curves — so personalization models must not train
 *                  disproportionately on the heavy tail)
 *
 * Usage:
 *   npx tsx sync/enqueue-player-histories.ts [--limit 500] [--min-games 40] [--stratified]
 *   npx tsx sync/enqueue-player-histories.ts --status
 *
 * Safe to run weekly; it stops on quota exhaustion and resumes where it
 * left off (enumerated players are marked in player_history_marks).
 */
import { sql } from 'drizzle-orm'
import { HeroesProfileApi } from './api-client'
import { createDb, SyncDb } from './db'
import { log } from './logger'

const CHUNK_MONTHS = 6
const HISTORY_START = '2021-12-01'

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function ensureMarksTable(db: SyncDb) {
  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS player_history_marks (
      battletag varchar(100) PRIMARY KEY,
      blizz_id bigint,
      region integer,
      replays_found integer NOT NULL DEFAULT 0,
      enqueued integer NOT NULL DEFAULT 0,
      marked_at timestamp NOT NULL DEFAULT now()
    )`)
}

interface Target {
  battletag: string
  blizz_id: number
  region: number
  games: number
}

async function selectTargets(
  db: SyncDb, limit: number, minGames: number, stratified: boolean,
): Promise<Target[]> {
  const rows = stratified
    ? await db.execute(sql`
        WITH counts AS (
          SELECT p.battletag, p.blizz_id, max(p.region) AS region, count(*) AS games,
                 ntile(4) OVER (ORDER BY count(*)) AS tier
          FROM replay_players p
          WHERE EXISTS (SELECT 1 FROM replay_draft_data d WHERE d.replay_id = p.replay_id)
            AND NOT EXISTS (SELECT 1 FROM player_history_marks m WHERE m.battletag = p.battletag)
          GROUP BY p.battletag, p.blizz_id
          HAVING count(*) >= ${minGames}
        ), ranked AS (
          SELECT *, row_number() OVER (PARTITION BY tier ORDER BY random()) AS rn
          FROM counts
        )
        SELECT battletag, blizz_id, region, games FROM ranked
        WHERE rn <= ${Math.ceil(limit / 4)}
        ORDER BY tier, rn
        LIMIT ${limit}`).then(r => r.rows)
    : await db.execute(sql`
        SELECT p.battletag, p.blizz_id, max(p.region) AS region, count(*) AS games
        FROM replay_players p
        WHERE EXISTS (SELECT 1 FROM replay_draft_data d WHERE d.replay_id = p.replay_id)
          AND NOT EXISTS (SELECT 1 FROM player_history_marks m WHERE m.battletag = p.battletag)
        GROUP BY p.battletag, p.blizz_id
        HAVING count(*) >= ${minGames}
        ORDER BY count(*) DESC
        LIMIT ${limit}`).then(r => r.rows)
  return rows.map(r => ({
    battletag: String(r.battletag),
    blizz_id: Number(r.blizz_id),
    region: Number(r.region ?? 1),
    games: Number(r.games),
  }))
}

/** Player/Replays for one battletag+window; returns replay ids or null on failure. */
async function fetchWindow(
  api: HeroesProfileApi, t: Target, from: string, to: string,
): Promise<number[] | null> {
  try {
    const raw: any = await api.fetch('Player/Replays', {
      battletag: t.battletag,
      region: String(t.region),
      mode: 'json',
      game_type: 'Storm League',
      start_date: from,
      end_date: to,
    })
    // Response shape: { "Storm League": { "<replayId>": {...}, ... } } —
    // replay ids are the KEYS of the object nested under the game_type.
    const inner = raw?.['Storm League'] ?? Object.values(raw ?? {})[0] ?? {}
    const entries: any[] = Array.isArray(inner)
      ? inner.map((r: any) => r.replayID ?? r.replay_id ?? r.replayId)
      : Object.keys(inner)
    return entries
      .map(Number)
      .filter((n: number) => Number.isFinite(n) && n > 0)
  } catch (err) {
    const msg = String(err)
    if (msg.includes('Max calls') || msg.includes('non-JSON response')) throw err // quota: stop run
    return null // window-level failure (timeout/500): caller may halve
  }
}

function* dateWindows(): Generator<[string, string]> {
  const start = new Date(HISTORY_START)
  const now = new Date()
  let a = new Date(start)
  while (a < now) {
    const b = new Date(a)
    b.setMonth(b.getMonth() + CHUNK_MONTHS)
    yield [a.toISOString().slice(0, 10), (b < now ? b : now).toISOString().slice(0, 10)]
    a = b
  }
}

async function main() {
  const args = process.argv.slice(2)
  const db = createDb()
  await ensureMarksTable(db)

  if (args.includes('--status')) {
    const [q] = await db.execute(sql`SELECT count(*) AS n FROM player_fetch_queue`).then(r => r.rows)
    const [m] = await db.execute(
      sql`SELECT count(*) AS players, COALESCE(sum(replays_found),0) AS found, COALESCE(sum(enqueued),0) AS enq
          FROM player_history_marks`).then(r => r.rows)
    console.log(`queue pending: ${Number(q.n).toLocaleString()} replay ids`)
    console.log(`players enumerated: ${Number(m.players).toLocaleString()} ` +
      `(${Number(m.found).toLocaleString()} history rows found, ${Number(m.enq).toLocaleString()} enqueued)`)
    return
  }

  const key1 = process.env.HEROES_PROFILE_API_KEY
  if (!key1) { log.error('HEROES_PROFILE_API_KEY required'); process.exit(1) }
  const key2 = process.env.HEROES_PROFILE_API_KEY2
  const limit = Number(args[args.indexOf('--limit') + 1]) || 500
  const minGames = Number(args[args.indexOf('--min-games') + 1]) || 40
  // Player/Replays shares each key's per-minute rate with the other workers;
  // keep this modest — the weekly pools (key1 ~5K, key2 ~500 calls) are the
  // real limit. On key1 quota exhaustion we fall through to key2 and squeeze
  // its pool too; the run stops only when EVERY key is spent.
  const clients = [{ name: 'key1', api: new HeroesProfileApi(key1, 20, 3) }]
  if (key2) clients.push({ name: 'key2', api: new HeroesProfileApi(key2, 15, 3) })
  let ci = 0

  const stratified = args.includes('--stratified')
  // TRACKED battletags are unconditional priority targets (Max, 2026-08-17):
  // they are the personalization paper's focal players, but their thin
  // early coverage meant the game-count-based selection never drew them.
  // Enumerate any not-yet-marked tracked player FIRST, every run.
  const tracked = await db.execute(sql`
    SELECT DISTINCT t.battletag, COALESCE(t.region, 1) AS region,
           COALESCE(p.blizz_id, 0) AS blizz_id,
           COALESCE(p.games, 0) AS games
    FROM tracked_battletags t
    LEFT JOIN (SELECT battletag, max(blizz_id) blizz_id, count(*) games
               FROM replay_players GROUP BY battletag) p ON p.battletag = t.battletag
    WHERE NOT EXISTS (SELECT 1 FROM player_history_marks m WHERE m.battletag = t.battletag)`)
    .then(r => r.rows.map(x => ({
      battletag: String(x.battletag), blizz_id: Number(x.blizz_id),
      region: Number(x.region), games: Number(x.games),
    })))
  if (tracked.length > 0) {
    log.info(`Priority: ${tracked.length} tracked battletags not yet enumerated: ` +
      tracked.map(x => x.battletag).join(', '))
  }
  const selected = await selectTargets(db, limit, minGames, stratified)
  const trackedTags = new Set(tracked.map(x => x.battletag))
  const targets = [...tracked, ...selected.filter(s => !trackedTags.has(s.battletag))]
  log.info(`Enumerating histories for ${targets.length} players ` +
    `(>=${minGames} panel games, ${stratified ? 'stratified by activity quartile' : 'heaviest first'})`)

  let calls = 0
  for (let ti = 0; ti < targets.length; ti++) {
    const t = targets[ti]
    const api = clients[ci].api
    let found = 0
    let enq = 0
    try {
      for (const [from, to] of dateWindows()) {
        let ids = await fetchWindow(api, t, from, to)
        calls++
        if (ids === null) {
          // halve once, then give up on this window
          const mid = new Date((new Date(from).getTime() + new Date(to).getTime()) / 2)
            .toISOString().slice(0, 10)
          const first = await fetchWindow(api, t, from, mid); calls++
          const second = await fetchWindow(api, t, mid, to); calls++
          ids = [...(first ?? []), ...(second ?? [])]
          if (first === null || second === null) {
            log.warn(`${t.battletag}: window ${from}..${to} failed even halved — skipping remainder`)
          }
        }
        if (ids.length === 0) continue
        found += ids.length
        // Enqueue only ids we have not covered and are not already queued.
        // ids are validated finite numbers, safe to inline (drizzle does not
        // serialize a JS array as a PG array literal for unnest).
        const idList = sql.raw(`ARRAY[${ids.join(',')}]::int[]`)
        const res = await db.execute(sql`
          INSERT INTO player_fetch_queue (replay_id, source)
          SELECT x.id, ${'history:' + t.battletag}
          FROM unnest(${idList}) AS x(id)
          WHERE NOT EXISTS (SELECT 1 FROM replay_players p WHERE p.replay_id = x.id)
          ON CONFLICT (replay_id) DO NOTHING`)
        enq += Number((res as any).rowCount ?? 0)
      }
      await db.execute(sql`
        INSERT INTO player_history_marks (battletag, blizz_id, region, replays_found, enqueued)
        VALUES (${t.battletag}, ${t.blizz_id}, ${t.region}, ${found}, ${enq})
        ON CONFLICT (battletag) DO UPDATE SET replays_found = ${found}, enqueued = ${enq}, marked_at = now()`)
      log.info(`${t.battletag}: ${found} history rows, ${enq} enqueued (${calls} calls so far)`)
    } catch (err) {
      const msg = String(err)
      if (msg.includes('Max calls') || msg.includes('non-JSON response')) {
        if (ci + 1 < clients.length) {
          ci++
          log.warn(`${clients[ci - 1].name} quota exhausted after ${calls} calls — ` +
            `switching to ${clients[ci].name}, retrying ${t.battletag}`)
          ti-- // re-enumerate this player on the next key (unmarked, no loss)
        } else {
          log.warn(`All Player/Replays quotas exhausted after ${calls} calls — stopping (resume next week)`)
          break
        }
      } else {
        log.warn(`${t.battletag}: unexpected error, skipping: ${msg.slice(0, 200)}`)
      }
    }
    await sleep(250)
  }
  log.info(`Done: ${calls} Player/Replays calls this run.`)
}

main().catch(err => {
  log.error('Fatal enumerator error', err)
  process.exit(1)
})
