import { sql } from 'drizzle-orm'
import {
  heroStatsAggregate,
  heroTalentStats,
  heroPairwiseStats,
} from '../src/lib/db/schema'
import { HERO_ROLES } from '../src/lib/data/hero-roles'
import { HeroesProfileApi } from './api-client'
import { SyncDb } from './db'
import { log } from './logger'

type SkillTier = 'low' | 'mid' | 'high'

const TIER_MAPPING: Array<[SkillTier, string]> = [
  ['low', '1,2'],
  ['mid', '3,4'],
  ['high', '5,6'],
]

const ALL_HEROES = Object.keys(HERO_ROLES)

// ── Staleness thresholds ─────────────────────────────────────────────

// ── Helpers ──────────────────────────────────────────────────────────

/** Run an async function over items with bounded concurrency. */
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
    Array.from({ length: Math.min(concurrency, items.length) }, () => worker())
  )
  return results
}

/** Extract a numeric value from a response field, handling strings and nulls. */
function num(val: any, fallback = 0): number {
  if (val === null || val === undefined) return fallback
  const n = typeof val === 'string' ? parseFloat(val) : Number(val)
  return isNaN(n) ? fallback : n
}

/**
 * The API may return hero data as either:
 * - An object keyed by hero name: { "Abathur": { ... }, "Ana": { ... } }
 * - An array of objects with a name/hero field
 * This normalizes to [{ name, ...stats }]
 */
function normalizeHeroData(data: any): Array<{ name: string; [k: string]: any }> {
  if (Array.isArray(data)) {
    return data.map(item => ({
      ...item,
      name: item.name || item.hero || item.Hero || 'Unknown',
    }))
  }
  if (typeof data === 'object' && data !== null) {
    return Object.entries(data).map(([name, stats]: [string, any]) => ({
      ...(typeof stats === 'object' ? stats : {}),
      name,
    }))
  }
  throw new Error(`Unexpected hero data format: ${typeof data}`)
}

/**
 * Batch upsert helper. Splits values into batches to avoid oversized queries.
 */
async function batchUpsert(
  db: SyncDb,
  table: any,
  values: any[],
  conflictTarget: any[],
  updateSet: Record<string, any>,
  batchSize = 100,
) {
  for (let i = 0; i < values.length; i += batchSize) {
    const batch = values.slice(i, i + batchSize)
    await db.insert(table).values(batch).onConflictDoUpdate({
      target: conflictTarget,
      set: updateSet,
    })
  }
}

// ── Patch resolution ─────────────────────────────────────────────────

interface PatchInfo {
  type: 'major' | 'minor'
  version: string
}

export async function getCurrentPatch(api: HeroesProfileApi): Promise<PatchInfo> {
  const data = await api.getPatches()

  // The Patches endpoint might return various formats.
  // Try to find the most recent major patch.
  let patches: Array<{ version: string; type?: string; game_version?: string }> = []

  if (Array.isArray(data)) {
    patches = data
  } else if (typeof data === 'object' && data !== null) {
    // Might be keyed by version or type
    for (const [key, val] of Object.entries(data)) {
      if (Array.isArray(val)) {
        patches.push(...(val as any[]).map((v: any) => ({ ...v, version: v.version || v.patch || key })))
      } else if (typeof val === 'object') {
        patches.push({ ...(val as any), version: (val as any).version || (val as any).patch || key })
      } else {
        patches.push({ version: String(val), type: key })
      }
    }
  }

  log.debug(`Found ${patches.length} patches, sample: ${JSON.stringify(patches.slice(0, 3))}`)

  // Find the most recent major patch
  // Major patches typically look like "2.55" while minor ones look like "2.55.3.90670"
  const majorPatches = patches.filter(p => {
    const v = p.version || p.game_version || ''
    const isMajor = p.type === 'major' || (v.split('.').length <= 2)
    return isMajor
  })

  if (majorPatches.length > 0) {
    // Sort by version string descending (lexicographic works for "2.XX" format)
    majorPatches.sort((a, b) => {
      const va = a.version || a.game_version || ''
      const vb = b.version || b.game_version || ''
      return vb.localeCompare(va, undefined, { numeric: true })
    })
    const latest = majorPatches[0]
    return { type: 'major', version: latest.version || latest.game_version || '' }
  }

  // Fall back to latest patch of any type
  if (patches.length > 0) {
    const latest = patches[patches.length - 1]
    const version = latest.version || latest.game_version || ''
    const type = version.split('.').length <= 2 ? 'major' : 'minor'
    return { type: type as 'major' | 'minor', version }
  }

  throw new Error('No patches found from API')
}

// ── Hero stats sync ──────────────────────────────────────────────────

async function syncHeroStatsForTier(
  api: HeroesProfileApi,
  db: SyncDb,
  tier: SkillTier,
  leagueTier: string,
  patch: PatchInfo,
) {
  log.info(`Syncing hero stats for tier=${tier} (league=${leagueTier})`)

  const raw = await api.getHeroStats(patch.type, patch.version, leagueTier)
  const heroes = normalizeHeroData(raw)

  log.info(`  Got ${heroes.length} heroes`)
  if (heroes.length > 0) {
    log.debug(`  Sample hero data: ${JSON.stringify(heroes[0]).slice(0, 300)}`)
  }

  const rows = heroes.map(h => ({
    hero: h.name,
    skillTier: tier,
    games: num(h.games_played ?? h.games),
    wins: num(h.wins),
    winRate: num(h.win_rate ?? h.winrate),
    banRate: num(h.ban_rate ?? h.banrate, 0),
    pickRate: num(h.popularity ?? h.pick_rate ?? h.pickrate, 0),
    avgKills: num(h.kills ?? h.avg_kills, 0),
    avgDeaths: num(h.deaths ?? h.avg_deaths, 0),
    avgAssists: num(h.assists ?? h.avg_assists, 0),
    avgHeroDamage: num(h.hero_damage ?? h.avg_hero_damage, 0),
    avgSiegeDamage: num(h.siege_damage ?? h.avg_siege_damage, 0),
    avgHealing: num(h.healing ?? h.avg_healing, 0),
    avgExperience: num(h.experience ?? h.avg_experience, 0),
    patchTag: patch.version,
    updatedAt: new Date(),
  }))

  await batchUpsert(
    db,
    heroStatsAggregate,
    rows,
    [heroStatsAggregate.hero, heroStatsAggregate.skillTier],
    {
      games: sql.raw('excluded.games'),
      wins: sql.raw('excluded.wins'),
      winRate: sql.raw('excluded.win_rate'),
      banRate: sql.raw('excluded.ban_rate'),
      pickRate: sql.raw('excluded.pick_rate'),
      avgKills: sql.raw('excluded.avg_kills'),
      avgDeaths: sql.raw('excluded.avg_deaths'),
      avgAssists: sql.raw('excluded.avg_assists'),
      avgHeroDamage: sql.raw('excluded.avg_hero_damage'),
      avgSiegeDamage: sql.raw('excluded.avg_siege_damage'),
      avgHealing: sql.raw('excluded.avg_healing'),
      avgExperience: sql.raw('excluded.avg_experience'),
      patchTag: sql.raw('excluded.patch_tag'),
      updatedAt: sql`now()`,
    },
  )

  log.info(`  Upserted ${rows.length} hero stats for tier=${tier}`)
}

// ── Hero-map stats sync ──────────────────────────────────────────────
// NOTE: The Heroes Profile API's group_by_map parameter does NOT return
// per-map breakdowns — it returns the same format as regular hero stats.
// Hero-map aggregate stats will be populated from player replay data instead.
// The hero_map_stats_aggregate and map_stats_aggregate tables are left empty
// for aggregate data until a per-map API endpoint is found or we accumulate
// enough replay data to compute meaningful aggregates.

// ── Talent stats sync ────────────────────────────────────────────────

async function syncTalentStatsForTier(
  api: HeroesProfileApi,
  db: SyncDb,
  tier: SkillTier,
  leagueTier: string,
  patch: PatchInfo,
) {
  log.info(`Syncing talent stats for tier=${tier}`)

  let raw: any
  try {
    // Try without hero param first (bulk fetch)
    raw = await api.getTalentDetails(patch.type, patch.version, leagueTier)
  } catch (err) {
    log.warn(`Bulk talent fetch failed, will try per-hero: ${err}`)
    raw = null
  }

  if (raw) {
    const talentRows = parseTalentData(raw)
    if (talentRows.length > 0) {
      await upsertTalentRows(db, talentRows, tier)
      log.info(`  Upserted ${talentRows.length} talent stats for tier=${tier}`)
      return
    }
  }

  // Fallback: fetch per hero (expensive, rate-limited)
  log.info(`  Falling back to per-hero talent fetch for tier=${tier}`)
  let totalTalents = 0

  for (const hero of ALL_HEROES) {
    try {
      const heroRaw = await api.getTalentDetails(patch.type, patch.version, leagueTier, hero)
      const talentRows = parseTalentData(heroRaw, hero)
      if (talentRows.length > 0) {
        await upsertTalentRows(db, talentRows, tier)
        totalTalents += talentRows.length
      }
    } catch (err) {
      log.warn(`  Failed to fetch talents for ${hero}: ${err}`)
    }
  }

  log.info(`  Upserted ${totalTalents} talent stats (per-hero) for tier=${tier}`)
}

interface TalentRow {
  hero: string
  talentTier: number
  talentName: string
  games: number
  wins: number
  winRate: number
  pickRate: number
}

const TALENT_TIERS = [1, 4, 7, 10, 13, 16, 20]

/**
 * Parse talent data from the API.
 * Actual API format: { "HeroName": { "1": { "TalentName": { games_played, wins, losses, win_rate, popularity } }, ... } }
 * Talent tiers are keyed by tier number, and within each tier, talents are keyed by name.
 */
function parseTalentData(data: any, defaultHero?: string): TalentRow[] {
  const rows: TalentRow[] = []

  if (!data || typeof data !== 'object') return rows

  if (Array.isArray(data)) {
    for (const t of data) {
      rows.push({
        hero: t.hero || t.name || defaultHero || 'Unknown',
        talentTier: num(t.level ?? t.tier ?? t.talent_tier),
        talentName: t.title || t.talent_name || t.name || 'Unknown',
        games: num(t.games_played ?? t.games),
        wins: num(t.wins),
        winRate: num(t.win_rate ?? t.winrate),
        pickRate: num(t.popularity ?? t.pick_rate, 0),
      })
    }
    return rows
  }

  // Check if top-level keys are talent tiers (single hero) or hero names (multi-hero)
  const topKeys = Object.keys(data)
  const keysAreTiers = topKeys.every(k => TALENT_TIERS.includes(parseInt(k, 10)))

  if (keysAreTiers && defaultHero) {
    // Single hero: { "1": { "TalentName": { stats } }, ... }
    parseTalentTiers(data, defaultHero, rows)
  } else {
    // Multi-hero: { "HeroName": { "1": { "TalentName": { stats } }, ... }, ... }
    for (const [heroName, heroData] of Object.entries(data)) {
      if (typeof heroData !== 'object' || heroData === null) continue
      parseTalentTiers(heroData as Record<string, any>, heroName, rows)
    }
  }

  return rows
}

function parseTalentTiers(tierData: Record<string, any>, hero: string, rows: TalentRow[]) {
  for (const [tierStr, talents] of Object.entries(tierData)) {
    const talentTier = parseInt(tierStr, 10)
    if (isNaN(talentTier)) continue

    if (Array.isArray(talents)) {
      // Array format: [{ name, games_played, ... }]
      for (const t of talents) {
        rows.push({
          hero,
          talentTier,
          talentName: t.title || t.talent_name || t.name || 'Unknown',
          games: num(t.games_played ?? t.games),
          wins: num(t.wins),
          winRate: num(t.win_rate ?? t.winrate),
          pickRate: num(t.popularity ?? t.pick_rate, 0),
        })
      }
    } else if (typeof talents === 'object' && talents !== null) {
      // Object format: { "TalentName": { games_played, wins, losses, win_rate, popularity } }
      for (const [talentName, stats] of Object.entries(talents as Record<string, any>)) {
        if (typeof stats !== 'object' || stats === null) continue
        rows.push({
          hero,
          talentTier,
          talentName,
          games: num(stats.games_played ?? stats.games),
          wins: num(stats.wins),
          winRate: num(stats.win_rate ?? stats.winrate),
          pickRate: num(stats.popularity ?? stats.pick_rate, 0),
        })
      }
    }
  }
}

async function upsertTalentRows(db: SyncDb, rows: TalentRow[], tier: SkillTier) {
  const dbRows = rows.map(r => ({
    hero: r.hero,
    skillTier: tier,
    talentTier: r.talentTier,
    talentName: r.talentName,
    games: r.games,
    wins: r.wins,
    winRate: r.winRate,
    pickRate: r.pickRate,
    updatedAt: new Date(),
  }))

  await batchUpsert(
    db,
    heroTalentStats,
    dbRows,
    [heroTalentStats.hero, heroTalentStats.skillTier, heroTalentStats.talentTier, heroTalentStats.talentName],
    {
      games: sql.raw('excluded.games'),
      wins: sql.raw('excluded.wins'),
      winRate: sql.raw('excluded.win_rate'),
      pickRate: sql.raw('excluded.pick_rate'),
      updatedAt: sql`now()`,
    },
  )
}

// ── Matchup stats sync ───────────────────────────────────────────────

const MATCHUP_CONCURRENCY = 20

async function syncMatchupForHero(
  api: HeroesProfileApi,
  db: SyncDb,
  hero: string,
  patch: PatchInfo,
): Promise<boolean> {
  const raw = await api.getHeroMatchups(hero, patch.type, patch.version)
  const rows = parseMatchupData(hero, raw)

  if (rows.length === 0) return true

  // Store for all 3 tiers since we're fetching overall stats
  for (const tier of ['low', 'mid', 'high'] as SkillTier[]) {
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

    await batchUpsert(
      db,
      heroPairwiseStats,
      dbRows,
      [heroPairwiseStats.heroA, heroPairwiseStats.heroB, heroPairwiseStats.relationship, heroPairwiseStats.skillTier],
      {
        games: sql.raw('excluded.games'),
        wins: sql.raw('excluded.wins'),
        winRate: sql.raw('excluded.win_rate'),
        updatedAt: sql`now()`,
      },
    )
  }
  return true
}

async function syncMatchups(
  api: HeroesProfileApi,
  db: SyncDb,
  patch: PatchInfo,
) {
  log.info(`Syncing hero matchups (${ALL_HEROES.length} heroes, concurrency=${MATCHUP_CONCURRENCY})`)

  let successCount = 0
  let failCount = 0

  await pMap(ALL_HEROES, async (hero) => {
    try {
      await syncMatchupForHero(api, db, hero, patch)
      successCount++
    } catch (err) {
      failCount++
      log.warn(`Failed to fetch matchups for ${hero}: ${err}`)
    }
  }, MATCHUP_CONCURRENCY)

  log.info(`Matchup sync complete: ${successCount} heroes succeeded, ${failCount} failed`)
}

interface MatchupRow {
  heroA: string
  heroB: string
  relationship: string
  games: number
  wins: number
  winRate: number
}

/**
 * Parse matchup data from the API.
 * Actual format: { "Ana": { "HeroB": { "ally": { wins_with, losses_with, win_rate_as_ally }, "enemy": { wins_against, losses_against, win_rate_against } }, ... } }
 */
function parseMatchupData(hero: string, data: any): MatchupRow[] {
  const rows: MatchupRow[] = []
  if (!data || typeof data !== 'object') return rows

  // Unwrap the hero key if present
  let heroMatchups = data
  if (data[hero] && typeof data[hero] === 'object') {
    heroMatchups = data[hero]
  }

  for (const [heroB, matchupData] of Object.entries(heroMatchups)) {
    if (typeof matchupData !== 'object' || matchupData === null) continue
    const md = matchupData as Record<string, any>

    // Parse ally (synergy) data
    if (md.ally && typeof md.ally === 'object') {
      const ally = md.ally
      const winsW = num(ally.wins_with ?? ally.wins)
      const lossesW = num(ally.losses_with ?? ally.losses)
      const gamesW = winsW + lossesW
      const winRateW = num(ally.win_rate_as_ally ?? ally.win_rate)

      if (gamesW > 0) {
        rows.push({
          heroA: hero,
          heroB,
          relationship: 'with',
          games: gamesW,
          wins: winsW,
          winRate: winRateW,
        })
      }
    }

    // Parse enemy (counter) data
    if (md.enemy && typeof md.enemy === 'object') {
      const enemy = md.enemy
      const winsA = num(enemy.wins_against ?? enemy.wins)
      const lossesA = num(enemy.losses_against ?? enemy.losses)
      const gamesA = winsA + lossesA
      const winRateA = num(enemy.win_rate_against ?? enemy.win_rate)

      if (gamesA > 0) {
        rows.push({
          heroA: hero,
          heroB,
          relationship: 'against',
          games: gamesA,
          wins: winsA,
          winRate: winRateA,
        })
      }
    }
  }

  return rows
}

// ── Main export ──────────────────────────────────────────────────────

export async function syncGlobalStats(api: HeroesProfileApi, db: SyncDb) {
  log.info('=== Starting global stats sync ===')

  const patch = await getCurrentPatch(api)
  log.info(`Current patch: ${patch.version} (${patch.type})`)

  // 1. Hero stats — 3 tiers in parallel (fast endpoint, ~30s each)
  await Promise.all(
    TIER_MAPPING.map(([tier, leagueTier]) =>
      syncHeroStatsForTier(api, db, tier, leagueTier, patch)
        .catch(err => log.error(`Failed to sync hero stats for tier=${tier}`, err))
    ),
  )

  // 2. Talent stats — sequential (this endpoint is very slow and can't
  //    handle parallel requests without timing out)
  for (const [tier, leagueTier] of TIER_MAPPING) {
    try {
      await syncTalentStatsForTier(api, db, tier, leagueTier, patch)
    } catch (err) {
      log.error(`Failed to sync talent stats for tier=${tier}`, err)
    }
  }

  // 3. Hero matchups — 88 heroes with concurrency pool
  try {
    await syncMatchups(api, db, patch)
  } catch (err) {
    log.error('Failed to sync matchups', err)
  }

  log.info('=== Global stats sync complete ===')
}
