import { sql, eq, desc } from 'drizzle-orm'
import {
  playerMatchHistory, playerHeroStats, playerHeroMapStats,
  mapStatsAggregate, heroMapStatsAggregate,
  users, trackedBattletags,
} from '../src/lib/db/schema'
import { SyncDb } from './db'
import { log } from './logger'

// ── MAWP Formula (from spec) ─────────────────────────────────────────
//
// Estimates a player's current win probability for a hero via three
// mechanisms:
//
// 1. Game-count weighting — recent games matter more:
//    w_games(i) = 1.0                         if rank <= 30
//    w_games(i) = exp(-λ_g * (rank - 30))     otherwise
//    λ_g = ln(2) / 30
//
// 2. Time-decay blending — old game outcomes blend toward 50%:
//    w_time(i) = 1.0                          if days <= 180
//    w_time(i) = exp(-λ_t * (days - 180))     otherwise
//    λ_t = ln(2) / 90
//    effectiveOutcome(i) = outcome(i) * w_time(i) + 0.5 * (1 - w_time(i))
//
// 3. Bayesian padding — if games < 30, pad with (30 - games) phantom
//    50% observations at full weight.
//
// MAWP = (Σ(w_games(i) * effectiveOutcome(i)) + phantomPadding)
//      / (Σ(w_games(i)) + phantomCount)

const LAMBDA_G = Math.LN2 / 30
const LAMBDA_T = Math.LN2 / 90
const CONFIDENCE_THRESHOLD = 30

interface MatchRecord {
  win: boolean
  gameDate: Date
}

function computeMAWP(matches: MatchRecord[]): number {
  if (matches.length === 0) return 0.5

  // Sort newest first
  const sorted = [...matches].sort((a, b) => b.gameDate.getTime() - a.gameDate.getTime())

  const now = new Date()
  let weightedSum = 0
  let weightSum = 0

  for (let i = 0; i < sorted.length; i++) {
    const rank = i + 1 // 1-based
    const match = sorted[i]

    // Game count factor
    const wGames = rank <= 30
      ? 1.0
      : Math.exp(-LAMBDA_G * (rank - 30))

    // Time factor — blends outcome toward 50% rather than reducing weight
    const daysDiff = (now.getTime() - match.gameDate.getTime()) / (1000 * 60 * 60 * 24)
    const wTime = daysDiff <= 180
      ? 1.0
      : Math.exp(-LAMBDA_T * (daysDiff - 180))

    const outcome = match.win ? 1 : 0
    const effectiveOutcome = outcome * wTime + 0.5 * (1 - wTime)

    weightedSum += wGames * effectiveOutcome
    weightSum += wGames
  }

  // Bayesian padding: add phantom 50% games to reach confidence threshold
  if (sorted.length < CONFIDENCE_THRESHOLD) {
    const phantomCount = CONFIDENCE_THRESHOLD - sorted.length
    weightedSum += phantomCount * 0.5
    weightSum += phantomCount
  }

  return weightSum > 0 ? weightedSum / weightSum : 0.5
}

// ── Compute per-hero stats for a battletag ───────────────────────────

interface HeroAggregation {
  hero: string
  games: number
  wins: number
  winRate: number
  mawp: number
  recentWinRate: number | null
  trend: number | null
  avgKills: number
  avgDeaths: number
  avgAssists: number
}

function aggregateByHero(matches: Array<{
  hero: string
  win: boolean
  gameDate: Date
  kills: number | null
  deaths: number | null
  assists: number | null
}>): HeroAggregation[] {
  // Group by hero
  const grouped: Record<string, typeof matches> = {}
  for (const m of matches) {
    if (!grouped[m.hero]) grouped[m.hero] = []
    grouped[m.hero].push(m)
  }

  return Object.entries(grouped).map(([hero, heroMatches]) => {
    const games = heroMatches.length
    const wins = heroMatches.filter(m => m.win).length
    const winRate = games > 0 ? (wins / games) * 100 : 0

    // MAWP
    const mawp = computeMAWP(heroMatches) * 100

    // Recent win rate (last 20 games)
    const sorted = [...heroMatches].sort((a, b) => b.gameDate.getTime() - a.gameDate.getTime())
    const recent20 = sorted.slice(0, 20)
    const recentWinRate = recent20.length >= 5
      ? (recent20.filter(m => m.win).length / recent20.length) * 100
      : null

    // Trend (already in percent after above changes)
    const trend = recentWinRate !== null ? recentWinRate - winRate : null

    // Average stats
    const avgKills = heroMatches.reduce((s, m) => s + (m.kills ?? 0), 0) / games
    const avgDeaths = heroMatches.reduce((s, m) => s + (m.deaths ?? 0), 0) / games
    const avgAssists = heroMatches.reduce((s, m) => s + (m.assists ?? 0), 0) / games

    return {
      hero,
      games,
      wins,
      winRate,
      mawp,
      recentWinRate,
      trend,
      avgKills,
      avgDeaths,
      avgAssists,
    }
  })
}

// ── Compute per-hero-per-map stats ───────────────────────────────────

interface HeroMapAggregation {
  hero: string
  map: string
  games: number
  wins: number
  winRate: number
}

function aggregateByHeroMap(matches: Array<{
  hero: string
  map: string
  win: boolean
}>): HeroMapAggregation[] {
  const grouped: Record<string, { hero: string; map: string; games: number; wins: number }> = {}

  for (const m of matches) {
    const key = `${m.hero}|${m.map}`
    if (!grouped[key]) {
      grouped[key] = { hero: m.hero, map: m.map, games: 0, wins: 0 }
    }
    grouped[key].games++
    if (m.win) grouped[key].wins++
  }

  return Object.values(grouped).map(g => ({
    ...g,
    winRate: g.games > 0 ? (g.wins / g.games) * 100 : 0,
  }))
}

// ── Main computation ─────────────────────────────────────────────────

async function computeForBattletag(db: SyncDb, battletag: string): Promise<void> {
  // Fetch all match history for this battletag
  const matches = await db
    .select({
      hero: playerMatchHistory.hero,
      map: playerMatchHistory.map,
      win: playerMatchHistory.win,
      gameDate: playerMatchHistory.gameDate,
      kills: playerMatchHistory.kills,
      deaths: playerMatchHistory.deaths,
      assists: playerMatchHistory.assists,
    })
    .from(playerMatchHistory)
    .where(eq(playerMatchHistory.battletag, battletag))
    .orderBy(desc(playerMatchHistory.gameDate))

  if (matches.length === 0) {
    log.info(`  No matches found for ${battletag}, skipping derived stats`)
    return
  }

  log.info(`  Computing derived stats from ${matches.length} matches for ${battletag}`)

  // 1. Per-hero stats with MAWP
  const heroStats = aggregateByHero(matches)

  for (const hs of heroStats) {
    await db.insert(playerHeroStats)
      .values({
        battletag,
        hero: hs.hero,
        games: hs.games,
        wins: hs.wins,
        winRate: hs.winRate,
        mawp: hs.mawp,
        avgKills: hs.avgKills,
        avgDeaths: hs.avgDeaths,
        avgAssists: hs.avgAssists,
        recentWinRate: hs.recentWinRate,
        trend: hs.trend,
        updatedAt: new Date(),
      })
      .onConflictDoUpdate({
        target: [playerHeroStats.battletag, playerHeroStats.hero],
        set: {
          games: sql.raw('excluded.games'),
          wins: sql.raw('excluded.wins'),
          winRate: sql.raw('excluded.win_rate'),
          mawp: sql.raw('excluded.mawp'),
          avgKills: sql.raw('excluded.avg_kills'),
          avgDeaths: sql.raw('excluded.avg_deaths'),
          avgAssists: sql.raw('excluded.avg_assists'),
          recentWinRate: sql.raw('excluded.recent_win_rate'),
          trend: sql.raw('excluded.trend'),
          updatedAt: sql`now()`,
        },
      })
  }

  log.info(`  Upserted ${heroStats.length} hero stats for ${battletag}`)

  // 2. Per-hero-per-map stats
  const heroMapStats = aggregateByHeroMap(matches)

  const BATCH_SIZE = 100
  for (let i = 0; i < heroMapStats.length; i += BATCH_SIZE) {
    const batch = heroMapStats.slice(i, i + BATCH_SIZE)
    await db.insert(playerHeroMapStats)
      .values(batch.map(hm => ({
        battletag,
        hero: hm.hero,
        map: hm.map,
        games: hm.games,
        wins: hm.wins,
        winRate: hm.winRate,
        updatedAt: new Date(),
      })))
      .onConflictDoUpdate({
        target: [playerHeroMapStats.battletag, playerHeroMapStats.hero, playerHeroMapStats.map],
        set: {
          games: sql.raw('excluded.games'),
          wins: sql.raw('excluded.wins'),
          winRate: sql.raw('excluded.win_rate'),
          updatedAt: sql`now()`,
        },
      })
  }

  log.info(`  Upserted ${heroMapStats.length} hero-map stats for ${battletag}`)
}

// ── Aggregate map stats from player_match_history ───────────────────

async function computeMapAggregates(db: SyncDb): Promise<void> {
  const rows = await db.execute(
    sql`SELECT map, count(*) as games FROM player_match_history GROUP BY map`,
  )

  const SKILL_TIERS = ['low', 'mid', 'high'] as const

  for (const row of rows.rows) {
    for (const tier of SKILL_TIERS) {
      await db.insert(mapStatsAggregate)
        .values({
          map: row.map as string,
          skillTier: tier,
          games: Number(row.games),
          updatedAt: new Date(),
        })
        .onConflictDoUpdate({
          target: [mapStatsAggregate.map, mapStatsAggregate.skillTier],
          set: {
            games: sql.raw('excluded.games'),
            updatedAt: sql`now()`,
          },
        })
    }
  }

  log.info(`  Upserted map_stats_aggregate for ${rows.rows.length} maps × 3 tiers`)
}

// ── Aggregate hero-map stats from player_match_history ──────────────

async function computeHeroMapAggregates(db: SyncDb): Promise<void> {
  const rows = await db.execute(
    sql`SELECT hero, map, count(*) as games, sum(win::int) as wins FROM player_match_history GROUP BY hero, map`,
  )

  const SKILL_TIERS = ['low', 'mid', 'high'] as const
  const BATCH_SIZE = 100

  const allValues: Array<{
    hero: string
    map: string
    skillTier: 'low' | 'mid' | 'high'
    games: number
    wins: number
    winRate: number
    updatedAt: Date
  }> = []

  for (const row of rows.rows) {
    const games = Number(row.games)
    const wins = Number(row.wins)
    const winRate = games > 0 ? (wins / games) * 100 : 0
    for (const tier of SKILL_TIERS) {
      allValues.push({
        hero: row.hero as string,
        map: row.map as string,
        skillTier: tier,
        games,
        wins,
        winRate,
        updatedAt: new Date(),
      })
    }
  }

  for (let i = 0; i < allValues.length; i += BATCH_SIZE) {
    const batch = allValues.slice(i, i + BATCH_SIZE)
    await db.insert(heroMapStatsAggregate)
      .values(batch)
      .onConflictDoUpdate({
        target: [heroMapStatsAggregate.hero, heroMapStatsAggregate.map, heroMapStatsAggregate.skillTier],
        set: {
          games: sql.raw('excluded.games'),
          wins: sql.raw('excluded.wins'),
          winRate: sql.raw('excluded.win_rate'),
          updatedAt: sql`now()`,
        },
      })
  }

  log.info(`  Upserted hero_map_stats_aggregate: ${allValues.length} rows`)
}

// ── Seed tracked battletags ─────────────────────────────────────────

async function seedTrackedBattletags(db: SyncDb, battletags: Array<{ battletag: string; region: number }>): Promise<void> {
  // Upsert system user
  const result = await db.insert(users)
    .values({ email: 'system@hotsfever.local', displayName: 'System' })
    .onConflictDoUpdate({
      target: [users.email],
      set: { displayName: sql.raw('excluded.display_name') },
    })
    .returning({ id: users.id })

  const userId = result[0].id

  // Upsert tracked battletags
  for (const { battletag, region } of battletags) {
    await db.insert(trackedBattletags)
      .values({ userId, battletag, region })
      .onConflictDoUpdate({
        target: [trackedBattletags.userId, trackedBattletags.battletag],
        set: { region: sql.raw('excluded.region') },
      })
  }

  log.info(`  Upserted ${battletags.length} tracked battletags for system user (id=${userId})`)
}

/**
 * Compute derived statistics for all given battletags.
 * Reads from player_match_history and writes to player_hero_stats, player_hero_map_stats,
 * map_stats_aggregate, hero_map_stats_aggregate, and tracked_battletags.
 */
export async function computeDerivedStats(db: SyncDb, battletags: string[]): Promise<void> {
  log.info(`=== Computing derived stats for ${battletags.length} battletags ===`)

  for (const battletag of battletags) {
    try {
      await computeForBattletag(db, battletag)
    } catch (err) {
      log.error(`Failed to compute derived stats for ${battletag}`, err)
    }
  }

  // Aggregate stats across all players
  log.info('Computing map and hero-map aggregates...')
  await computeMapAggregates(db)
  await computeHeroMapAggregates(db)

  // Seed tracked battletags
  log.info('Seeding tracked battletags...')
  await seedTrackedBattletags(db, battletags.map(b => ({ battletag: b, region: 1 })))

  log.info('=== Derived stats computation complete ===')
}
