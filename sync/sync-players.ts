import { sql, eq, desc } from 'drizzle-orm'
import { playerMatchHistory, trackedBattletags } from '../src/lib/db/schema'
import { HeroesProfileApi } from './api-client'
import { SyncDb } from './db'
import { log } from './logger'

/** Battletag config for sync. */
export interface BattletagConfig {
  battletag: string
  region: number
}

function num(val: any, fallback = 0): number {
  if (val === null || val === undefined) return fallback
  const n = typeof val === 'string' ? parseFloat(val) : Number(val)
  return isNaN(n) ? fallback : n
}

/**
 * Get the most recent game date we have for a battletag.
 * Returns ISO date string or undefined if no history exists.
 */
async function getLastSyncDate(db: SyncDb, battletag: string): Promise<string | undefined> {
  const result = await db
    .select({ gameDate: playerMatchHistory.gameDate })
    .from(playerMatchHistory)
    .where(eq(playerMatchHistory.battletag, battletag))
    .orderBy(desc(playerMatchHistory.gameDate))
    .limit(1)

  if (result.length > 0 && result[0].gameDate) {
    // Format as YYYY-MM-DD for the API
    return result[0].gameDate.toISOString().split('T')[0]
  }
  return undefined
}

/**
 * Parse replay data from the API into rows for player_match_history.
 * Handles both array and object formats.
 */
function parseReplays(data: any, battletag: string): Array<{
  battletag: string
  replayId: string
  hero: string
  map: string
  win: boolean
  gameDate: Date
  gameLength: number | null
  kills: number
  deaths: number
  assists: number
  heroDamage: number
  siegeDamage: number
  healing: number
  experience: number
  talents: any
  gameMode: string | null
  rank: string | null
}> {
  if (!data) return []

  // Actual API format: { "Storm League": { "replayId": { ...fields } } }
  // Unwrap the game type wrapper and replay ID keys
  let replayEntries: Array<[string, any]> = []

  if (Array.isArray(data)) {
    replayEntries = data.map((item, i) => [String(item.replayID ?? item.replay_id ?? i), item])
  } else if (typeof data === 'object') {
    // Check if top-level keys are game types (e.g., "Storm League")
    for (const [key, val] of Object.entries(data)) {
      if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
        // Check if this looks like a game type wrapper (value is keyed by replay IDs)
        const innerKeys = Object.keys(val as Record<string, any>)
        const looksLikeReplayIds = innerKeys.length > 0 && innerKeys.every(k => /^\d+$/.test(k))

        if (looksLikeReplayIds) {
          // { "Storm League": { "12345": {...}, "12346": {...} } }
          for (const [replayId, replayData] of Object.entries(val as Record<string, any>)) {
            replayEntries.push([replayId, replayData])
          }
        } else if (/^\d+$/.test(key)) {
          // { "12345": { ...replayData } }
          replayEntries.push([key, val])
        }
      }
    }
  }

  if (replayEntries.length === 0) {
    log.warn(`Could not parse replay data: ${typeof data}, keys=${typeof data === 'object' ? Object.keys(data).slice(0, 3) : 'N/A'}`)
    return []
  }

  return replayEntries
    .map(([replayId, r]: [string, any]) => {
      const hero = r.hero || r.Hero || ''
      const map = r.game_map ?? r.map ?? r.Map ?? ''

      // Winner is boolean in the actual API
      let win = false
      const winField = r.winner ?? r.win ?? r.result
      if (typeof winField === 'boolean') win = winField
      else if (typeof winField === 'number') win = winField === 1
      else if (typeof winField === 'string') win = winField.toLowerCase() === 'win' || winField === '1' || winField.toLowerCase() === 'true'

      const dateStr = r.game_date ?? r.gameDate ?? r.date ?? ''
      const gameDate = dateStr ? new Date(dateStr) : new Date()

      // Build talent data from level_X fields (actual API format)
      const talents: Record<string, string | null> = {}
      for (const [level, tierNum] of [['level_one', 1], ['level_four', 4], ['level_seven', 7], ['level_ten', 10], ['level_thirteen', 13], ['level_sixteen', 16], ['level_twenty', 20]] as const) {
        talents[String(tierNum)] = r[level] ?? null
      }
      // Also check for pre-structured talent data
      const talentData = r.talents ?? talents

      return {
        battletag,
        replayId,
        hero,
        map,
        win,
        gameDate,
        gameLength: r.game_length ? num(r.game_length) : null,
        kills: num(r.kills),
        deaths: num(r.deaths),
        assists: num(r.assists),
        heroDamage: num(r.hero_damage ?? r.heroDamage),
        siegeDamage: num(r.siege_damage ?? r.siegeDamage),
        healing: num(r.healing),
        experience: num(r.experience_contribution ?? r.experience ?? r.xp),
        talents: talentData,
        gameMode: r.game_type ?? r.game_mode ?? 'Storm League',
        rank: r.rank ?? r.league_tier ?? null,
      }
    })
    .filter(r => r.replayId && r.hero)
}

/**
 * Sync match history for a single battletag.
 * Uses incremental sync — only fetches games since last known game date.
 */
async function syncBattletag(
  api: HeroesProfileApi,
  db: SyncDb,
  config: BattletagConfig,
): Promise<number> {
  const { battletag, region } = config

  // Determine start date for incremental sync
  const lastDate = await getLastSyncDate(db, battletag)
  log.info(`  Last synced game date: ${lastDate ?? 'never (full sync)'}`)

  const raw = await api.getPlayerReplays(battletag, region, lastDate)
  const replays = parseReplays(raw, battletag)

  if (replays.length === 0) {
    log.info(`  No new replays for ${battletag}`)
    return 0
  }

  log.info(`  Parsed ${replays.length} replays for ${battletag}`)

  // Batch upsert replays
  const BATCH_SIZE = 100
  for (let i = 0; i < replays.length; i += BATCH_SIZE) {
    const batch = replays.slice(i, i + BATCH_SIZE)
    await db.insert(playerMatchHistory)
      .values(batch)
      .onConflictDoUpdate({
        target: [playerMatchHistory.battletag, playerMatchHistory.replayId],
        set: {
          hero: sql.raw('excluded.hero'),
          map: sql.raw('excluded.map'),
          win: sql.raw('excluded.win'),
          gameDate: sql.raw('excluded.game_date'),
          gameLength: sql.raw('excluded.game_length'),
          kills: sql.raw('excluded.kills'),
          deaths: sql.raw('excluded.deaths'),
          assists: sql.raw('excluded.assists'),
          heroDamage: sql.raw('excluded.hero_damage'),
          siegeDamage: sql.raw('excluded.siege_damage'),
          healing: sql.raw('excluded.healing'),
          experience: sql.raw('excluded.experience'),
          talents: sql.raw('excluded.talents'),
          gameMode: sql.raw('excluded.game_mode'),
          rank: sql.raw('excluded.rank'),
        },
      })
  }

  // Update tracked_battletags.lastSynced if the row exists
  await db.update(trackedBattletags)
    .set({ lastSynced: new Date() })
    .where(eq(trackedBattletags.battletag, battletag))

  return replays.length
}

/**
 * Sync all tracked battletags.
 * Continues even if individual battletags fail.
 */
export async function syncPlayerData(
  api: HeroesProfileApi,
  db: SyncDb,
  battletags: BattletagConfig[],
): Promise<void> {
  log.info(`=== Starting player sync for ${battletags.length} battletags ===`)

  let totalReplays = 0
  let successCount = 0
  let failCount = 0

  for (const config of battletags) {
    log.info(`Syncing ${config.battletag} (region=${config.region})`)
    try {
      const count = await syncBattletag(api, db, config)
      totalReplays += count
      successCount++
    } catch (err) {
      failCount++
      log.error(`Failed to sync ${config.battletag}`, err)
    }
  }

  log.info(`=== Player sync complete: ${successCount} succeeded, ${failCount} failed, ${totalReplays} total replays ===`)
}
