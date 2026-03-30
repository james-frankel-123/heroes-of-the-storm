/**
 * Load DraftData from the database for standalone Node.js benchmark use.
 * Mirrors the server-side getDraftData() but without Next.js dependencies.
 */

import pg from 'pg'
import type { DraftData } from '@/lib/draft/types'
import type { SkillTier } from '@/lib/types'
import { computeBaselineCompWR } from '@/lib/draft/composition'
import * as fs from 'fs'
import * as path from 'path'

const CACHE_DIR = path.join(new URL('.', import.meta.url).pathname, 'cache')

export async function loadDraftData(tier: SkillTier): Promise<DraftData> {
  // Try cache first
  const cachePath = path.join(CACHE_DIR, `draft-data-${tier}.json`)
  if (fs.existsSync(cachePath)) {
    const age = Date.now() - fs.statSync(cachePath).mtimeMs
    if (age < 24 * 60 * 60 * 1000) { // 24h
      return JSON.parse(fs.readFileSync(cachePath, 'utf-8'))
    }
  }

  const dbUrl = process.env.DATABASE_URL
  if (!dbUrl) throw new Error('DATABASE_URL required')

  const client = new pg.Client({ connectionString: dbUrl })
  await client.connect()

  // Hero stats
  const heroStatsResult = await client.query(
    `SELECT hero, win_rate, pick_rate, ban_rate, games FROM hero_stats_aggregate WHERE skill_tier = $1`,
    [tier]
  )
  const heroStats: DraftData['heroStats'] = {}
  for (const r of heroStatsResult.rows) {
    heroStats[r.hero] = {
      winRate: parseFloat(r.win_rate),
      pickRate: parseFloat(r.pick_rate || 0),
      banRate: parseFloat(r.ban_rate || 0),
      games: parseInt(r.games),
    }
  }

  // Hero-map win rates
  const mapResult = await client.query(
    `SELECT hero, map, win_rate, games FROM hero_map_stats_aggregate WHERE skill_tier = $1`,
    [tier]
  )
  const heroMapWinRates: DraftData['heroMapWinRates'] = {}
  for (const r of mapResult.rows) {
    if (!heroMapWinRates[r.map]) heroMapWinRates[r.map] = {}
    heroMapWinRates[r.map][r.hero] = {
      winRate: parseFloat(r.win_rate),
      games: parseInt(r.games),
    }
  }

  // Pairwise stats
  const pairResult = await client.query(
    `SELECT hero_a, hero_b, relationship, win_rate, games FROM hero_pairwise_stats WHERE skill_tier = $1`,
    [tier]
  )
  const synergies: DraftData['synergies'] = {}
  const counters: DraftData['counters'] = {}
  for (const r of pairResult.rows) {
    const entry = { winRate: parseFloat(r.win_rate), games: parseInt(r.games) }
    if (r.relationship === 'with') {
      if (!synergies[r.hero_a]) synergies[r.hero_a] = {}
      synergies[r.hero_a][r.hero_b] = entry
    } else {
      if (!counters[r.hero_a]) counters[r.hero_a] = {}
      counters[r.hero_a][r.hero_b] = entry
    }
  }

  // Compositions from static JSON
  const compPath = path.join(new URL('.', import.meta.url).pathname, '..', 'src', 'lib', 'data', 'compositions.json')
  let compositions: DraftData['compositions'] = []
  if (fs.existsSync(compPath)) {
    const raw = JSON.parse(fs.readFileSync(compPath, 'utf-8'))
    const tierComps = raw[tier] || []
    compositions = tierComps.map((c: any) => ({
      roles: c.roles,
      winRate: c.winRate,
      games: c.games,
      popularity: c.popularity ?? c.games,
    }))
  }

  await client.end()

  const data: DraftData = {
    heroStats,
    heroMapWinRates,
    synergies,
    counters,
    playerStats: {},
    playerMapStats: {},
    compositions,
    baselineCompWR: computeBaselineCompWR(compositions),
  }

  // Cache
  fs.mkdirSync(CACHE_DIR, { recursive: true })
  fs.writeFileSync(cachePath, JSON.stringify(data))
  console.log(`  Cached DraftData (${tier}) to ${cachePath}`)

  return data
}
