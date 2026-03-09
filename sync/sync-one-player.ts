/**
 * One-shot script to sync a single player's data.
 * Usage: set -a && source .env && set +a && npx tsx sync/sync-one-player.ts "NotoriousPig#11231"
 */
import { HeroesProfileApi } from './api-client'
import { createDb } from './db'
import { syncPlayerData, BattletagConfig } from './sync-players'
import { computeDerivedStats } from './compute-derived'
import { log } from './logger'

async function main() {
  const battletag = process.argv[2]
  if (!battletag) {
    log.error('Usage: npx tsx sync/sync-one-player.ts "BattleTag#1234"')
    process.exit(1)
  }

  const apiKey = process.env.HEROES_PROFILE_API_KEY
  if (!apiKey) { log.error('HEROES_PROFILE_API_KEY required'); process.exit(1) }
  if (!process.env.DATABASE_URL) { log.error('DATABASE_URL required'); process.exit(1) }

  const db = createDb()
  const api = new HeroesProfileApi(apiKey)

  const config: BattletagConfig = { battletag, region: 1 }

  log.info(`Syncing player data for ${battletag}...`)
  await syncPlayerData(api, db, [config])

  log.info(`Computing derived stats (MAWP)...`)
  await computeDerivedStats(db, [battletag])

  log.info(`Done. Total API calls: ${api.getCallCount()}`)
}

main().catch(err => {
  log.error('Fatal error', err)
  process.exit(1)
})
