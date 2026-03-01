import { eq } from 'drizzle-orm'
import { syncLog } from '../src/lib/db/schema'
import { HeroesProfileApi } from './api-client'
import { createDb } from './db'
import { syncGlobalStats } from './sync-global'
import { syncPlayerData, BattletagConfig } from './sync-players'
import { computeDerivedStats } from './compute-derived'
import { log } from './logger'

// ── Configuration ────────────────────────────────────────────────────

// Hardcoded battletags for initial development (spec: Django#1458, AzmoDonTrump#1139, SirWatsonII#1400)
// TODO: Switch to reading from tracked_battletags table once site is live
const BATTLETAGS: BattletagConfig[] = [
  { battletag: 'Django#1458', region: 1 },
  { battletag: 'AzmoDonTrump#1139', region: 1 },
  { battletag: 'SirWatsonII#1400', region: 1 },
]

// ── Sync log helpers ─────────────────────────────────────────────────

async function logSyncStart(db: ReturnType<typeof createDb>, syncType: string, battletag?: string) {
  const result = await db.insert(syncLog).values({
    syncType,
    battletag: battletag ?? null,
    status: 'running',
    matchesProcessed: 0,
    startedAt: new Date(),
  }).returning({ id: syncLog.id })

  return result[0].id
}

async function logSyncSuccess(db: ReturnType<typeof createDb>, id: number, matchesProcessed = 0) {
  await db.update(syncLog)
    .set({
      status: 'success',
      matchesProcessed,
      completedAt: new Date(),
    })
    .where(eq(syncLog.id, id))
}

async function logSyncError(db: ReturnType<typeof createDb>, id: number, error: string) {
  await db.update(syncLog)
    .set({
      status: 'error',
      errorMessage: error.slice(0, 2000),
      completedAt: new Date(),
    })
    .where(eq(syncLog.id, id))
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  const startTime = Date.now()
  log.info('╔══════════════════════════════════════════════╗')
  log.info('║     HotS Fever — Data Sync Starting         ║')
  log.info('╚══════════════════════════════════════════════╝')

  // Validate environment
  const apiKey = process.env.HEROES_PROFILE_API_KEY
  if (!apiKey) {
    log.error('HEROES_PROFILE_API_KEY environment variable is required')
    process.exit(1)
  }
  if (!process.env.DATABASE_URL) {
    log.error('DATABASE_URL environment variable is required')
    process.exit(1)
  }

  const db = createDb()
  const api = new HeroesProfileApi(apiKey)

  // ── Phase 1: Global aggregate stats ──
  let globalLogId: number | undefined
  try {
    globalLogId = await logSyncStart(db, 'aggregate')
    await syncGlobalStats(api, db)
    await logSyncSuccess(db, globalLogId)
  } catch (err) {
    log.error('Global stats sync failed', err)
    if (globalLogId) {
      await logSyncError(db, globalLogId, err instanceof Error ? err.message : String(err))
        .catch(e => log.error('Failed to log sync error', e))
    }
  }

  // ── Phase 2: Player match history ──
  for (const config of BATTLETAGS) {
    let playerLogId: number | undefined
    try {
      playerLogId = await logSyncStart(db, 'player', config.battletag)
      await syncPlayerData(api, db, [config])
      await logSyncSuccess(db, playerLogId)
    } catch (err) {
      log.error(`Player sync failed for ${config.battletag}`, err)
      if (playerLogId) {
        await logSyncError(db, playerLogId, err instanceof Error ? err.message : String(err))
          .catch(e => log.error('Failed to log sync error', e))
      }
    }
  }

  // ── Phase 3: Derived stats ──
  let derivedLogId: number | undefined
  try {
    derivedLogId = await logSyncStart(db, 'derived')
    await computeDerivedStats(db, BATTLETAGS.map(b => b.battletag))
    await logSyncSuccess(db, derivedLogId)
  } catch (err) {
    log.error('Derived stats computation failed', err)
    if (derivedLogId) {
      await logSyncError(db, derivedLogId, err instanceof Error ? err.message : String(err))
        .catch(e => log.error('Failed to log sync error', e))
    }
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1)
  log.info(`Sync complete in ${elapsed}s. Total API calls: ${api.getCallCount()}`)
}

main().catch(err => {
  log.error('Fatal sync error', err)
  process.exit(1)
})
