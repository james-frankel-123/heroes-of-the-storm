/**
 * Phase 1 player-identity refetch worker (paper 3, personalization).
 *
 * Walks replay_draft_data.replay_id DESCENDING (recent first, freshest meta
 * first) and re-fetches Replay/Data for every known replay, storing the full
 * per-player payload into replay_players (+ replay-level leftovers into
 * replay_extras) via the shared storeReplayPlayers helper.
 *
 * - Both API keys at their real rates: key 1 (dev) 180/min, key 2
 *   (Intermediate) 55/min, each with its own small worker pool so the slower
 *   key can't throttle the faster one (round-robin would cap at 2x slower key).
 * - Resumable: cursor checkpoint in player_refetch_state (single row, id=1).
 *   Batches are selected with NOT EXISTS(replay_players), so a killed batch
 *   resumes idempotently and already-fetched replays are never re-bought.
 * - Graceful on deleted/404 replays: counted as skipped, cursor moves on.
 * - Weekly quota exhaustion ("non-JSON response"): the affected key sleeps
 *   60 min then re-probes; the other key keeps working.
 *
 * Usage:
 *   npx tsx sync/refetch-players.ts             # run the refetch loop
 *   npx tsx sync/refetch-players.ts --status    # progress + rate + ETA
 *   npx tsx sync/refetch-players.ts --restart   # reset cursor to top (sweep pass)
 *
 * Detached launch:
 *   nohup npx tsx sync/refetch-players.ts >> sync/logs/refetch-players.log 2>&1 &
 */
import { sql, eq } from 'drizzle-orm'
import { HeroesProfileApi } from './api-client'
import { createDb, SyncDb } from './db'
import { log } from './logger'
import { storeReplayPlayers } from './player-store'
import { playerRefetchState } from '../src/lib/db/schema'

const BATCH_SIZE = 2000
const KEY1_RATE = 180 // dev account
const KEY2_RATE = 55  // Intermediate account
const KEY1_WORKERS = 4
const KEY2_WORKERS = 2
const QUOTA_BACKOFF_MS = 60 * 60_000 // exhausted key re-probes hourly

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// ── Checkpoint state ─────────────────────────────────────────────────

interface RefetchState {
  cursor: number
  processedCount: number
  playerRowsUpserted: number
  skippedCount: number
  errorCount: number
  startedAt: Date
}

async function loadState(db: SyncDb): Promise<RefetchState | null> {
  const rows = await db.select().from(playerRefetchState).limit(1)
  if (rows.length === 0) return null
  const r = rows[0]
  return {
    cursor: r.cursor,
    processedCount: r.processedCount,
    playerRowsUpserted: r.playerRowsUpserted,
    skippedCount: r.skippedCount,
    errorCount: r.errorCount,
    startedAt: r.startedAt,
  }
}

async function initState(db: SyncDb, cursor: number): Promise<RefetchState> {
  const state: RefetchState = {
    cursor,
    processedCount: 0,
    playerRowsUpserted: 0,
    skippedCount: 0,
    errorCount: 0,
    startedAt: new Date(),
  }
  const existing = await db.select().from(playerRefetchState).limit(1)
  if (existing.length === 0) {
    await db.insert(playerRefetchState).values({ ...state })
  } else {
    await db.update(playerRefetchState)
      .set({ ...state, updatedAt: new Date() })
      .where(eq(playerRefetchState.id, existing[0].id))
  }
  return state
}

async function saveState(db: SyncDb, state: RefetchState) {
  await db.update(playerRefetchState).set({
    cursor: state.cursor,
    processedCount: state.processedCount,
    playerRowsUpserted: state.playerRowsUpserted,
    skippedCount: state.skippedCount,
    errorCount: state.errorCount,
    updatedAt: new Date(),
  })
}

// ── Batch selection ──────────────────────────────────────────────────

/** Next batch of replay_ids below the cursor that have no player rows yet. */
async function nextBatch(db: SyncDb, cursor: number): Promise<number[]> {
  const rows = await db.execute(
    sql`SELECT d.replay_id
        FROM replay_draft_data d
        WHERE d.replay_id < ${cursor}
          AND NOT EXISTS (SELECT 1 FROM replay_players p WHERE p.replay_id = d.replay_id)
        ORDER BY d.replay_id DESC
        LIMIT ${BATCH_SIZE}`
  ).then(r => r.rows)
  return rows.map(r => Number(r.replay_id))
}

// ── Worker pool ──────────────────────────────────────────────────────

interface KeyedClient {
  name: string
  api: HeroesProfileApi
  exhaustedUntil: number // epoch ms; 0 = healthy
}

interface BatchCounters {
  processed: number
  playerRows: number
  skipped: number
  errors: number
}

async function processBatch(
  db: SyncDb,
  clients: KeyedClient[],
  ids: number[],
  counters: BatchCounters,
  stop: () => boolean,
): Promise<void> {
  let idx = 0
  const retries: number[] = []
  const retried = new Set<number>()

  const takeId = (): number | undefined => {
    if (idx < ids.length) return ids[idx++]
    return retries.shift()
  }

  const handleOne = async (client: KeyedClient, replayId: number) => {
    try {
      const raw = await client.api.getReplayData(replayId)
      const replay = raw[String(replayId)] || raw
      if (!replay || typeof replay !== 'object') {
        counters.skipped++
        return
      }
      const res = await storeReplayPlayers(db, replayId, replay)
      if (res.playerRows > 0) {
        counters.processed++
        counters.playerRows += res.playerRows
      } else {
        counters.skipped++ // replay exists but carried no player entries
      }
    } catch (err) {
      const msg = String(err)
      const isQuota = msg.includes('non-JSON response') || msg.includes('Max calls')
      const isPermanent = /API error 4\d\d/.test(msg) || msg.includes('returned error')
      if (isQuota) {
        // Weekly quota on this key — bench it, give the id back to the pool.
        client.exhaustedUntil = Date.now() + QUOTA_BACKOFF_MS
        log.warn(`${client.name}: Replay/Data quota exhausted — benching for ${QUOTA_BACKOFF_MS / 60000} min`)
        retries.unshift(replayId)
        retried.delete(replayId) // quota bounce doesn't consume the retry
      } else if (isPermanent) {
        // Deleted/404/invalid replay — skip forever, cursor moves past.
        counters.skipped++
      } else if (!retried.has(replayId)) {
        // Transient (rate-limit/5xx/network/DB) even after client retries —
        // one more in-batch attempt, then leave for a --restart sweep.
        retried.add(replayId)
        retries.push(replayId)
      } else {
        counters.errors++
        if (counters.errors % 25 === 1) {
          log.warn(`Refetch error for replay ${replayId}: ${msg.slice(0, 300)}`)
        }
      }
    }
  }

  const workerLoop = async (client: KeyedClient) => {
    while (!stop()) {
      if (client.exhaustedUntil > Date.now()) {
        const healthy = clients.some(c => c.exhaustedUntil <= Date.now())
        if (!healthy) log.warn('All API keys quota-benched — sleeping')
        await sleep(Math.min(client.exhaustedUntil - Date.now(), 60_000))
        continue
      }
      const id = takeId()
      if (id === undefined) return
      await handleOne(client, id)
    }
  }

  const workers: Promise<void>[] = []
  for (const client of clients) {
    const n = client.name === 'key1' ? KEY1_WORKERS : KEY2_WORKERS
    for (let i = 0; i < n; i++) workers.push(workerLoop(client))
  }
  await Promise.all(workers)
}

// ── Status ───────────────────────────────────────────────────────────

async function printStatus(db: SyncDb) {
  const state = await loadState(db)
  if (!state) {
    console.log('Refetch has not been started (no player_refetch_state row).')
    return
  }
  const [playerCount] = await db.execute(
    sql`SELECT count(*) AS c, count(DISTINCT replay_id) AS r FROM replay_players`
  ).then(r => r.rows)
  const [remaining] = await db.execute(
    sql`SELECT count(*) AS c FROM replay_draft_data WHERE replay_id < ${state.cursor}`
  ).then(r => r.rows)

  const replaysCovered = Number(playerCount.r)
  const elapsedMin = (Date.now() - state.startedAt.getTime()) / 60_000
  // Rate from actual coverage (processedCount only checkpoints per batch and
  // misses the daemon's contribution).
  const rate = replaysCovered / Math.max(elapsedMin, 1) // replays/min
  const to250k = Math.max(0, 250_000 - replaysCovered)
  const etaTo250k = rate > 0 ? to250k / rate / 60 : Infinity
  const etaFull = rate > 0 ? Number(remaining.c) / rate / 60 : Infinity

  console.log('── Player refetch status ─────────────────────────────')
  console.log(`  cursor (descending):    ${state.cursor}`)
  console.log(`  replays processed:      ${state.processedCount.toLocaleString()}`)
  console.log(`  replays skipped:        ${state.skippedCount.toLocaleString()} (deleted/404/no players)`)
  console.log(`  transient errors left:  ${state.errorCount.toLocaleString()} (recoverable via --restart sweep)`)
  console.log(`  player rows in DB:      ${Number(playerCount.c).toLocaleString()} across ${replaysCovered.toLocaleString()} replays`)
  console.log(`  remaining below cursor: ${Number(remaining.c).toLocaleString()}`)
  console.log(`  running since:          ${state.startedAt.toISOString()} (${elapsedMin.toFixed(0)} min)`)
  console.log(`  observed rate:          ${rate.toFixed(1)} replays/min`)
  console.log(`  ETA to 250K replays:    ${isFinite(etaTo250k) ? etaTo250k.toFixed(1) + ' h' : 'n/a'} (quota permitting)`)
  console.log(`  ETA full corpus:        ${isFinite(etaFull) ? (etaFull / 24).toFixed(1) + ' d' : 'n/a'} (nominal; weekly quotas gate this)`)
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  const key1 = process.env.HEROES_PROFILE_API_KEY
  const key2 = process.env.HEROES_PROFILE_API_KEY2
  if (!process.env.DATABASE_URL) { log.error('DATABASE_URL required'); process.exit(1) }
  const db = createDb()

  if (process.argv.includes('--status')) {
    await printStatus(db)
    return
  }

  if (!key1) { log.error('HEROES_PROFILE_API_KEY required'); process.exit(1) }
  const clients: KeyedClient[] = [
    { name: 'key1', api: new HeroesProfileApi(key1, KEY1_RATE, 3), exhaustedUntil: 0 },
  ]
  if (key2) clients.push({ name: 'key2', api: new HeroesProfileApi(key2, KEY2_RATE, 3), exhaustedUntil: 0 })
  log.info(`Player refetch: ${clients.length} key(s) at ${key2 ? `${KEY1_RATE}+${KEY2_RATE}` : KEY1_RATE}/min, ` +
    `${KEY1_WORKERS}+${key2 ? KEY2_WORKERS : 0} workers, batch=${BATCH_SIZE}`)

  let state = await loadState(db)
  if (!state || process.argv.includes('--restart')) {
    const [maxRow] = await db.execute(
      sql`SELECT COALESCE(MAX(replay_id), 0) AS m FROM replay_draft_data`
    ).then(r => r.rows)
    const top = Number(maxRow.m) + 1
    log.info(`${state ? '--restart: resetting' : 'Initializing'} cursor to ${top} (descending scan)`)
    state = await initState(db, top)
  } else {
    log.info(`Resuming from cursor ${state.cursor} (${state.processedCount} processed so far)`)
  }

  let stopping = false
  const requestStop = (sig: string) => {
    log.warn(`${sig} received — finishing in-flight calls, then checkpointing`)
    stopping = true
  }
  process.on('SIGINT', () => requestStop('SIGINT'))
  process.on('SIGTERM', () => requestStop('SIGTERM'))

  const runStart = Date.now()
  let processedThisRun = 0

  while (!stopping) {
    const ids = await nextBatch(db, state.cursor)
    if (ids.length === 0) {
      log.info('Refetch complete — no unfetched replays below cursor. Exiting.')
      break
    }

    const counters: BatchCounters = { processed: 0, playerRows: 0, skipped: 0, errors: 0 }
    await processBatch(db, clients, ids, counters, () => stopping)

    // Only advance the cursor past a FULLY attempted batch. On a mid-batch
    // stop, leaving the cursor put means the unprocessed remainder is
    // re-selected on resume (processed ids are excluded by NOT EXISTS).
    if (!stopping) {
      state.cursor = ids[ids.length - 1] // batch is DESC; last id is the lowest
    }
    state.processedCount += counters.processed
    state.playerRowsUpserted += counters.playerRows
    state.skippedCount += counters.skipped
    state.errorCount += counters.errors
    await saveState(db, state)

    processedThisRun += counters.processed
    const rate = processedThisRun / Math.max((Date.now() - runStart) / 60_000, 0.1)
    log.info(
      `Batch done: +${counters.processed} replays (${counters.playerRows} player rows, ` +
      `${counters.skipped} skipped, ${counters.errors} errors) | cursor=${state.cursor} | ` +
      `total ${state.processedCount.toLocaleString()} | ${rate.toFixed(0)}/min | ` +
      `ETA 250K: ${((250_000 - state.processedCount) / Math.max(rate, 1) / 60).toFixed(1)}h`
    )
  }

  await saveState(db, state)
  log.info('Checkpoint saved. Bye.')
}

main().catch(err => {
  log.error('Fatal refetch worker error', err)
  process.exit(1)
})
