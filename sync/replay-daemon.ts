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
import { MultiKeyApi } from './api-client'
import { createDb } from './db'
import { log } from './logger'
import { discoverReplays, fetchReplayData, getReplayStats } from './sync-replays'
import { replaySyncState, replayFetchQueue } from '../src/lib/db/schema'
import { eq, sql } from 'drizzle-orm'

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function main() {
  const key1 = process.env.HEROES_PROFILE_API_KEY
  if (!key1) { log.error('HEROES_PROFILE_API_KEY required'); process.exit(1) }
  if (!process.env.DATABASE_URL) { log.error('DATABASE_URL required'); process.exit(1) }

  const db = createDb()
  // Key 1 = developer account (200/min limit, use 180 safe)
  const api = new MultiKeyApi([key1], 180, 3)
  log.info('Using dev API key (180/min)')

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

  const DISCOVERY_BATCH = 2000
  const FETCH_BATCH = 1000
  const CYCLE_PAUSE_MS = 10_000 // 10s pause between cycles

  let cycle = 0
  while (true) {
    cycle++
    log.info(`\n=== Cycle ${cycle} ===`)

    try {
      // Phase 1: Discover new replay IDs
      const discovered = await discoverReplays(api, db, DISCOVERY_BATCH)

      // Phase 2: Fetch full data for queued replays
      const fetched = await fetchReplayData(api, db, FETCH_BATCH)

      // Report stats
      const stats = await getReplayStats(db)
      log.info(`Stats: ${stats.draftDataRows} drafts stored, ${stats.pendingInQueue} pending, ` +
        `cursor gap: ${stats.gapRemaining}, total API calls: ${api.getTotalCallCount()}`)

      // If we're caught up (no pending and cursor near max), slow down
      if (discovered === 0 && fetched === 0) {
        log.info('Caught up — waiting 5 minutes before next cycle')
        await sleep(300_000)
      } else {
        await sleep(CYCLE_PAUSE_MS)
      }
    } catch (err) {
      log.error(`Cycle ${cycle} error:`, err)
      await sleep(60_000) // Wait 1 min on error
    }
  }
}

main().catch(err => {
  log.error('Fatal daemon error', err)
  process.exit(1)
})
