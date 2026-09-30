/**
 * Continuous replay sync daemon for Draft Insights Hyper Pro Max.
 *
 * Alternates between discovery (Replay/Min_id) and fetch (Replay/Data)
 * phases, using two API keys to maximize throughput.
 *
 * Key 1 = developer account (180/min safe limit, higher weekly quota)
 * Key 2 = standard account (55/min)
 *
 * Usage: set -a && source .env && set +a && npx tsx sync/replay-daemon.ts
 *        --fresh    Reset cursor to most recent replays (skip old queue)
 */
import { MultiKeyApi, ReplayApiPool, SingleKeyPool } from './api-client'
import { createHpApi, isV2 } from './hp-api'
import { createDb } from './db'
import { isHpAccessPaused } from './hp-errors'
import { log } from './logger'
import { discoverReplays, discoverBackfill, fetchReplayData, backfillTalents, getReplayStats } from './sync-replays'
import { replaySyncState, replayFetchQueue } from '../src/lib/db/schema'
import { eq, sql } from 'drizzle-orm'

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function main() {
  if (!process.env.DATABASE_URL) { log.error('DATABASE_URL required'); process.exit(1) }
  const db = createDb()

  let api: ReplayApiPool
  if (isV2()) {
    api = new SingleKeyPool(createHpApi('key1', 110, 3))
    log.info('Using Heroes Profile v1 API (HP_API=v2), one key')
  } else {
    const key1 = process.env.HEROES_PROFILE_API_KEY
    const key2 = process.env.HEROES_PROFILE_API_KEY2
    if (!key1) { log.error('HEROES_PROFILE_API_KEY required'); process.exit(1) }
    // Key 1 = dev account (180/min). Key 2 = standard account (55/min), used
    // as fallback when key 1's weekly Replay/Data quota exhausts.
    const keys = [key1, ...(key2 ? [key2] : [])]
    const rates = key2 ? [180, 55] : [180]
    api = new MultiKeyApi(keys, rates, 3)
    log.info(`Using ${keys.length} API key(s) (rates: ${rates.join('/')} /min)`)
  }

  // --cron: one-shot mode — exit when caught up (or quota-blocked) instead
  // of sleeping, so the Neon endpoint can suspend between scheduled runs.
  const cron = process.argv.includes('--cron')

  // --fresh flag: reset cursor to near max, clear old unfetched queue
  const fresh = process.argv.includes('--fresh')
  if (fresh) {
    log.info('--fresh: Resetting to most recent replays...')
    const maxApi = api.next()
    const maxId = await maxApi.getReplayMax()
    // Start discovery 200K IDs back from current max (roughly 1-2 weeks of replays)
    const newCursor = maxId - 200_000
    await db.update(replaySyncState).set({
      discoveryCursor: newCursor,
      maxKnownId: maxId,
    }).where(eq(replaySyncState.id, 1))
    // Clear old unfetched queue entries
    const cleared = await db.delete(replayFetchQueue).where(eq(replayFetchQueue.fetched, false))
    log.info(`  Cursor reset to ${newCursor} (max: ${maxId})`)
    log.info(`  Cleared old unfetched queue entries`)
  }

  log.info('╔══════════════════════════════════════════════╗')
  log.info('║  Replay Sync Daemon — Hyper Pro Max         ║')
  log.info('╚══════════════════════════════════════════════╝')

  // v1 meters /replays at 20K/wk shared with fetch-qm; run-backfills fires
  // ~42x/wk, so ~350 listing pages per run leaves headroom.
  const DISCOVERY_BATCH = isV2() ? 250 : 2000
  const BACKFILL_BATCH = isV2() ? 100 : 500 // Backfill discovery calls per cycle
  const FETCH_BATCH = 5000      // Increased to use more of our 250K/wk Data quota
  const CYCLE_PAUSE_MS = 10_000 // 10s pause between cycles

  let cycle = 0
  let consecutiveErrors = 0
  while (true) {
    cycle++
    log.info(`\n=== Cycle ${cycle} ===`)

    try {
      // Phase 1a: Discover new replay IDs (forward scan)
      const discovered = await discoverReplays(api, db, DISCOVERY_BATCH)

      // Phase 1b: Backfill older replays (backward scan)
      const backfilled = await discoverBackfill(api, db, BACKFILL_BATCH)

      // Phase 2: Fetch full data for queued replays
      const fetched = await fetchReplayData(api, db, FETCH_BATCH)

      // Phase 3: Backfill talent data for older replays (only when queue is empty)
      const talentBackfilled = await backfillTalents(db, 2000)

      const caughtUp = discovered === 0 && backfilled === 0 && fetched === 0 && talentBackfilled === 0

      // Report stats every 15th cycle (count(*) on multi-GB tables is a full
      // scan — too expensive to run every 10s) and whenever we're caught up.
      if (cycle % 15 === 1 || caughtUp) {
        const stats = await getReplayStats(db)
        log.info(`Stats: ${stats.draftDataRows} drafts stored, ${stats.pendingInQueue} pending, ` +
          `cursor gap: ${stats.gapRemaining}, total API calls: ${api.getTotalCallCount()}`)
      }

      // If we're caught up on everything (including talent backfill), slow down
      if (caughtUp) {
        if (cron) {
          log.info('Caught up (or quota-blocked) — exiting (cron mode)')
          break
        }
        log.info('Caught up — waiting 5 minutes before next cycle')
        await sleep(300_000)
      } else {
        await sleep(CYCLE_PAUSE_MS)
      }
      consecutiveErrors = 0
    } catch (err) {
      if (isHpAccessPaused(err)) {
        log.error(`${err.message} — stopping; accept the terms / fill in project details on the HP account page`)
        if (cron) process.exit(1)
        await sleep(30 * 60_000)
        continue
      }
      log.error(`Cycle ${cycle} error:`, err)
      consecutiveErrors++
      if (cron && consecutiveErrors >= 3) {
        log.error('3 consecutive cycle errors — exiting (cron mode)')
        process.exit(1)
      }
      await sleep(60_000) // Wait 1 min on error
    }
  }
}

main().catch(err => {
  log.error('Fatal daemon error', err)
  process.exit(1)
})
