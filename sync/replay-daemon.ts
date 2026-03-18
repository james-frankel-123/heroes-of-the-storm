/**
 * Continuous replay sync daemon for Draft Insights Hyper Pro Max.
 *
 * Alternates between discovery (Replay/Min_id) and fetch (Replay/Data)
 * phases, using two API keys in round-robin to maximize throughput.
 *
 * Usage: set -a && source .env && set +a && npx tsx sync/replay-daemon.ts
 *
 * Designed to run as a systemd service or long-lived process.
 */
import { MultiKeyApi } from './api-client'
import { createDb } from './db'
import { log } from './logger'
import { discoverReplays, fetchReplayData, getReplayStats } from './sync-replays'

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function main() {
  const key1 = process.env.HEROES_PROFILE_API_KEY
  if (!key1) { log.error('HEROES_PROFILE_API_KEY required'); process.exit(1) }
  if (!process.env.DATABASE_URL) { log.error('DATABASE_URL required'); process.exit(1) }

  const db = createDb()
  // Key1 is developer account (200/min)
  const api = new MultiKeyApi([key1], 180, 3)

  log.info('╔══════════════════════════════════════════════╗')
  log.info('║  Replay Sync Daemon — Hyper Pro Max         ║')
  log.info('╚══════════════════════════════════════════════╝')

  // Discovery: 200 calls/batch, Fetch: 300 calls/batch
  // At ~110 calls/min combined, each cycle takes ~5 min
  const DISCOVERY_BATCH = 2000
  const FETCH_BATCH = 300
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
