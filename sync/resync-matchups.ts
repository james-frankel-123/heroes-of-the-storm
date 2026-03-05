/**
 * One-shot script to re-sync hero matchups with tier-specific data.
 * Previously matchups were fetched without a league_tier param and
 * duplicated across all tiers. This fetches per-tier data.
 *
 * Usage: set -a && source .env && set +a && npx tsx sync/resync-matchups.ts
 */
import { sql } from 'drizzle-orm'
import { HERO_ROLES } from '../src/lib/data/hero-roles'
import { heroPairwiseStats } from '../src/lib/db/schema'
import { HeroesProfileApi } from './api-client'
import { createDb } from './db'
import { getCurrentPatch } from './sync-global'
import { log } from './logger'

type SkillTier = 'low' | 'mid' | 'high'

const TIER_MAPPING: Array<[SkillTier, string]> = [
  ['low', '1,2'],
  ['mid', '3,4'],
  ['high', '5,6'],
]

const ALL_HEROES = Object.keys(HERO_ROLES)
const CONCURRENCY = 15

function num(val: any, fallback = 0): number {
  if (val === null || val === undefined) return fallback
  const n = typeof val === 'string' ? parseFloat(val) : Number(val)
  return isNaN(n) ? fallback : n
}

async function pMap<T, R>(
  items: T[],
  fn: (item: T, index: number) => Promise<R>,
  concurrency: number,
): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let nextIndex = 0
  async function worker() {
    while (nextIndex < items.length) {
      const i = nextIndex++
      results[i] = await fn(items[i], i)
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, items.length) }, () => worker()),
  )
  return results
}

interface MatchupRow {
  heroA: string
  heroB: string
  relationship: 'with' | 'against'
  games: number
  wins: number
  winRate: number
}

function parseMatchupData(hero: string, data: any): MatchupRow[] {
  const rows: MatchupRow[] = []
  if (!data || typeof data !== 'object') return rows

  let heroMatchups = data
  if (data[hero] && typeof data[hero] === 'object') heroMatchups = data[hero]

  for (const [heroB, md] of Object.entries(heroMatchups)) {
    if (typeof md !== 'object' || md === null) continue
    const m = md as Record<string, any>

    if (m.ally && typeof m.ally === 'object') {
      const w = num(m.ally.wins_with ?? m.ally.wins)
      const l = num(m.ally.losses_with ?? m.ally.losses)
      if (w + l > 0) {
        rows.push({
          heroA: hero,
          heroB,
          relationship: 'with',
          games: w + l,
          wins: w,
          winRate: num(m.ally.win_rate_as_ally ?? m.ally.win_rate),
        })
      }
    }

    if (m.enemy && typeof m.enemy === 'object') {
      const opponentWins = num(m.enemy.wins_against ?? m.enemy.wins)
      const opponentLosses = num(m.enemy.losses_against ?? m.enemy.losses)
      const gamesA = opponentWins + opponentLosses
      const opponentWR = num(m.enemy.win_rate_against ?? m.enemy.win_rate)

      if (gamesA > 0) {
        // API returns the OPPONENT's wins/WR, so we invert to get hero A's perspective
        rows.push({
          heroA: hero,
          heroB,
          relationship: 'against',
          games: gamesA,
          wins: gamesA - opponentWins,
          winRate: Math.round((100 - opponentWR) * 100) / 100,
        })
      }
    }
  }
  return rows
}

async function main() {
  const apiKey = process.env.HEROES_PROFILE_API_KEY
  if (!apiKey) { log.error('HEROES_PROFILE_API_KEY required'); process.exit(1) }
  if (!process.env.DATABASE_URL) { log.error('DATABASE_URL required'); process.exit(1) }

  const db = createDb()
  const api = new HeroesProfileApi(apiKey, 55, 2)
  const patch = await getCurrentPatch(api)
  log.info(`Patch: ${patch.version}`)

  for (const [tier, leagueTier] of TIER_MAPPING) {
    log.info(`\n=== Syncing matchups for tier=${tier} (league=${leagueTier}) ===`)
    let ok = 0, fail = 0

    await pMap(ALL_HEROES, async (hero) => {
      try {
        const raw = await api.getHeroMatchups(hero, patch.type, patch.version, leagueTier)
        const rows = parseMatchupData(hero, raw)

        if (rows.length === 0) {
          ok++
          return
        }

        const dbRows = rows.map(r => ({
          heroA: r.heroA,
          heroB: r.heroB,
          relationship: r.relationship,
          skillTier: tier,
          games: r.games,
          wins: r.wins,
          winRate: r.winRate,
          updatedAt: new Date(),
        }))

        // Batch upsert in chunks
        for (let i = 0; i < dbRows.length; i += 100) {
          const batch = dbRows.slice(i, i + 100)
          await db.insert(heroPairwiseStats).values(batch).onConflictDoUpdate({
            target: [heroPairwiseStats.heroA, heroPairwiseStats.heroB, heroPairwiseStats.relationship, heroPairwiseStats.skillTier],
            set: {
              games: sql.raw('excluded.games'),
              wins: sql.raw('excluded.wins'),
              winRate: sql.raw('excluded.win_rate'),
              updatedAt: sql`now()`,
            },
          })
        }

        ok++
        if (ok % 20 === 0) log.info(`  tier=${tier} progress: ${ok}/${ALL_HEROES.length}`)
      } catch (err) {
        fail++
        log.warn(`  Failed ${hero} tier=${tier}: ${err}`)
      }
    }, CONCURRENCY)

    log.info(`tier=${tier} done: ${ok} succeeded, ${fail} failed`)
  }

  log.info(`\nDone. Total API calls: ${api.getCallCount()}`)
}

main().catch(err => {
  log.error('Fatal error', err)
  process.exit(1)
})
