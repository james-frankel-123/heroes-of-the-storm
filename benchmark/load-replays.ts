/**
 * Load replay draft data for benchmark simulation.
 */

import pg from 'pg'
import * as fs from 'fs'
import * as path from 'path'

const CACHE_DIR = path.join(new URL('.', import.meta.url).pathname, 'cache')

export interface ReplayDraft {
  replayId: number
  gameMap: string
  skillTier: string
  draftOrder: { hero: string; type: string; pick_number: number; player_slot: number }[]
  team0Heroes: string[]
  team1Heroes: string[]
  winner: number
}

export async function loadReplays(
  limit = 1000,
  tier?: string,
): Promise<ReplayDraft[]> {
  // Try cache
  const cacheKey = `replays-${tier || 'all'}-${limit}`
  const cachePath = path.join(CACHE_DIR, `${cacheKey}.json`)
  if (fs.existsSync(cachePath)) {
    const age = Date.now() - fs.statSync(cachePath).mtimeMs
    if (age < 24 * 60 * 60 * 1000) {
      return JSON.parse(fs.readFileSync(cachePath, 'utf-8'))
    }
  }

  const dbUrl = process.env.DATABASE_URL
  if (!dbUrl) throw new Error('DATABASE_URL required')

  const client = new pg.Client({ connectionString: dbUrl })
  await client.connect()

  let query = `
    SELECT replay_id, game_map, skill_tier, draft_order,
           team0_heroes, team1_heroes, winner
    FROM replay_draft_data
    WHERE draft_order IS NOT NULL
      AND jsonb_array_length(draft_order) = 16
  `
  const params: any[] = []
  if (tier && tier !== 'all') {
    params.push(tier)
    query += ` AND skill_tier = $${params.length}`
  }
  // Use a random sample from the most recent 100K replays (test set)
  query += ` ORDER BY replay_id DESC LIMIT ${limit * 3}`

  const result = await client.query(query, params)
  await client.end()

  const replays: ReplayDraft[] = []
  for (const r of result.rows) {
    const draftOrder = typeof r.draft_order === 'string'
      ? JSON.parse(r.draft_order)
      : r.draft_order
    if (!Array.isArray(draftOrder) || draftOrder.length !== 16) continue

    const t0h = Array.isArray(r.team0_heroes) ? r.team0_heroes : JSON.parse(r.team0_heroes || '[]')
    const t1h = Array.isArray(r.team1_heroes) ? r.team1_heroes : JSON.parse(r.team1_heroes || '[]')
    if (t0h.length !== 5 || t1h.length !== 5) continue

    replays.push({
      replayId: r.replay_id,
      gameMap: r.game_map,
      skillTier: r.skill_tier,
      draftOrder,
      team0Heroes: t0h,
      team1Heroes: t1h,
      winner: r.winner,
    })
  }

  // Shuffle and take requested limit
  for (let i = replays.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [replays[i], replays[j]] = [replays[j], replays[i]]
  }
  const selected = replays.slice(0, limit)

  fs.mkdirSync(CACHE_DIR, { recursive: true })
  fs.writeFileSync(cachePath, JSON.stringify(selected))
  console.log(`  Cached ${selected.length} replays to ${cachePath}`)

  return selected
}
