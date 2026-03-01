/**
 * Recompute derived stats (MAWP, trends, map/hero-map aggregates).
 * No API calls — reads from existing player_match_history in the DB.
 *
 * Usage: npx tsx sync/recompute-mawp.ts
 */
import 'dotenv/config'
import { computeDerivedStats } from './compute-derived'
import { createDb } from './db'
import { log } from './logger'

const BATTLETAGS = ['Django#1458', 'AzmoDonTrump#1139', 'SirWatsonII#1400']

async function main() {
  if (!process.env.DATABASE_URL) {
    log.error('DATABASE_URL required')
    process.exit(1)
  }

  const db = createDb()
  log.info('Recomputing derived stats with updated MAWP formula...')
  await computeDerivedStats(db, BATTLETAGS)
  log.info('Done.')
}

main().catch(err => {
  log.error('Fatal error', err)
  process.exit(1)
})
