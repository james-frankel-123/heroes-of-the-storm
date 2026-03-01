/**
 * Data access layer — backed by Drizzle ORM / Neon Postgres.
 *
 * Notes:
 *   - avg_damage_soaked, avg_merc_captures, avg_self_healing, avg_time_dead
 *     columns were added to the schema but never migrated to the actual DB.
 *     We default these to 0 in the application layer.
 *   - KDA / damage averages in hero_stats_aggregate are always 0 because the
 *     upstream API (Heroes Profile /Heroes/Stats) does not return them.
 */

import { db } from '@/lib/db'
import {
  heroStatsAggregate,
  heroMapStatsAggregate,
  mapStatsAggregate,
  heroTalentStats as heroTalentStatsTable,
  heroPairwiseStats as heroPairwiseStatsTable,
  playerMatchHistory as playerMatchHistoryTable,
  playerHeroStats as playerHeroStatsTable,
  playerHeroMapStats as playerHeroMapStatsTable,
  trackedBattletags,
} from '@/lib/db/schema'
import { eq, and, desc, asc, sql, inArray } from 'drizzle-orm'

import type {
  SkillTier,
  HeroStats,
  MapStats,
  HeroMapStats,
  HeroTalentStats,
  HeroPairwiseStats,
  PlayerHeroStats,
  PlayerMatch,
  PlayerHeroMapStats,
  TrackedBattletag,
} from '@/lib/types'

import type { DraftData } from '@/lib/draft/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Round to 1 decimal place */
function r1(n: number): number {
  return Math.round(n * 10) / 10
}

/** Shape of what we select from hero_stats_aggregate */
type HeroStatsRow = {
  hero: string
  skillTier: string
  games: number
  wins: number
  winRate: number
  banRate: number | null
  pickRate: number | null
  avgKills: number | null
  avgDeaths: number | null
  avgAssists: number | null
  avgHeroDamage: number | null
  avgSiegeDamage: number | null
  avgHealing: number | null
  avgExperience: number | null
  patchTag: string | null
}

/** Map a hero_stats_aggregate row to our HeroStats type, defaulting missing cols */
function toHeroStats(row: HeroStatsRow): HeroStats {
  return {
    hero: row.hero,
    skillTier: row.skillTier as SkillTier,
    games: row.games,
    wins: row.wins,
    winRate: r1(row.winRate),
    banRate: r1(row.banRate ?? 0),
    pickRate: r1(row.pickRate ?? 0),
    avgKills: r1(row.avgKills ?? 0),
    avgDeaths: r1(row.avgDeaths ?? 0),
    avgAssists: r1(row.avgAssists ?? 0),
    avgHeroDamage: Math.round(row.avgHeroDamage ?? 0),
    avgSiegeDamage: Math.round(row.avgSiegeDamage ?? 0),
    avgHealing: Math.round(row.avgHealing ?? 0),
    avgExperience: Math.round(row.avgExperience ?? 0),
    // These columns don't exist in the actual DB yet — default to 0
    avgDamageSoaked: 0,
    avgMercCaptures: 0,
    avgSelfHealing: 0,
    avgTimeDead: 0,
    patchTag: row.patchTag ?? null,
  }
}

// ---------------------------------------------------------------------------
// Aggregate queries
// ---------------------------------------------------------------------------

/** All hero stats for a skill tier, sorted by win rate desc */
export async function getHeroStats(tier: SkillTier): Promise<HeroStats[]> {
  const rows = await db
    .select({
      hero: heroStatsAggregate.hero,
      skillTier: heroStatsAggregate.skillTier,
      games: heroStatsAggregate.games,
      wins: heroStatsAggregate.wins,
      winRate: heroStatsAggregate.winRate,
      banRate: heroStatsAggregate.banRate,
      pickRate: heroStatsAggregate.pickRate,
      avgKills: heroStatsAggregate.avgKills,
      avgDeaths: heroStatsAggregate.avgDeaths,
      avgAssists: heroStatsAggregate.avgAssists,
      avgHeroDamage: heroStatsAggregate.avgHeroDamage,
      avgSiegeDamage: heroStatsAggregate.avgSiegeDamage,
      avgHealing: heroStatsAggregate.avgHealing,
      avgExperience: heroStatsAggregate.avgExperience,
      patchTag: heroStatsAggregate.patchTag,
    })
    .from(heroStatsAggregate)
    .where(eq(heroStatsAggregate.skillTier, tier))
    .orderBy(desc(heroStatsAggregate.winRate))

  return rows.map((r) => toHeroStats(r))
}

/** Single hero stats across all tiers */
export async function getHeroStatsByName(hero: string): Promise<HeroStats[]> {
  const rows = await db
    .select({
      hero: heroStatsAggregate.hero,
      skillTier: heroStatsAggregate.skillTier,
      games: heroStatsAggregate.games,
      wins: heroStatsAggregate.wins,
      winRate: heroStatsAggregate.winRate,
      banRate: heroStatsAggregate.banRate,
      pickRate: heroStatsAggregate.pickRate,
      avgKills: heroStatsAggregate.avgKills,
      avgDeaths: heroStatsAggregate.avgDeaths,
      avgAssists: heroStatsAggregate.avgAssists,
      avgHeroDamage: heroStatsAggregate.avgHeroDamage,
      avgSiegeDamage: heroStatsAggregate.avgSiegeDamage,
      avgHealing: heroStatsAggregate.avgHealing,
      avgExperience: heroStatsAggregate.avgExperience,
      patchTag: heroStatsAggregate.patchTag,
    })
    .from(heroStatsAggregate)
    .where(eq(heroStatsAggregate.hero, hero))

  return rows.map((r) => toHeroStats(r))
}

/** All map stats for a skill tier */
export async function getMapStats(tier: SkillTier): Promise<MapStats[]> {
  const rows = await db
    .select()
    .from(mapStatsAggregate)
    .where(eq(mapStatsAggregate.skillTier, tier))
    .orderBy(desc(mapStatsAggregate.games))

  return rows.map((r) => ({
    map: r.map,
    skillTier: r.skillTier as SkillTier,
    games: r.games,
  }))
}

/** All map names */
export async function getAllMaps(): Promise<string[]> {
  const rows = await db
    .selectDistinct({ map: mapStatsAggregate.map })
    .from(mapStatsAggregate)
    .orderBy(asc(mapStatsAggregate.map))

  return rows.map((r) => r.map)
}

/** Hero performance on a specific map for a tier */
export async function getHeroMapStats(
  tier: SkillTier,
  map?: string,
  hero?: string
): Promise<HeroMapStats[]> {
  const conditions = [eq(heroMapStatsAggregate.skillTier, tier)]
  if (map) conditions.push(eq(heroMapStatsAggregate.map, map))
  if (hero) conditions.push(eq(heroMapStatsAggregate.hero, hero))

  const rows = await db
    .select()
    .from(heroMapStatsAggregate)
    .where(and(...conditions))
    .orderBy(desc(heroMapStatsAggregate.winRate))

  return rows.map((r) => ({
    hero: r.hero,
    map: r.map,
    skillTier: r.skillTier as SkillTier,
    games: r.games,
    wins: r.wins,
    winRate: r.winRate,
  }))
}

/** Talent stats for a hero at a tier */
export async function getTalentStats(
  hero: string,
  tier: SkillTier
): Promise<HeroTalentStats[]> {
  const rows = await db
    .select()
    .from(heroTalentStatsTable)
    .where(
      and(
        eq(heroTalentStatsTable.hero, hero),
        eq(heroTalentStatsTable.skillTier, tier)
      )
    )
    .orderBy(
      asc(heroTalentStatsTable.talentTier),
      desc(heroTalentStatsTable.winRate)
    )

  return rows.map((r) => ({
    hero: r.hero,
    skillTier: r.skillTier as SkillTier,
    talentTier: r.talentTier,
    talentName: r.talentName,
    games: r.games,
    wins: r.wins,
    winRate: r.winRate,
    pickRate: r.pickRate ?? 0,
  }))
}

/**
 * Pairwise stats: synergies or counters for a hero.
 * The DB stores both directions (A→B and B→A), so we only query heroA = hero
 * to avoid returning each partner twice.
 */
export async function getPairwiseStats(
  hero: string,
  relationship: 'with' | 'against',
  tier: SkillTier
): Promise<HeroPairwiseStats[]> {
  const rows = await db
    .select()
    .from(heroPairwiseStatsTable)
    .where(
      and(
        eq(heroPairwiseStatsTable.heroA, hero),
        eq(heroPairwiseStatsTable.relationship, relationship),
        eq(heroPairwiseStatsTable.skillTier, tier)
      )
    )
    .orderBy(desc(heroPairwiseStatsTable.winRate))

  return rows.map((r) => ({
    heroA: r.heroA,
    heroB: r.heroB,
    relationship: r.relationship as 'with' | 'against',
    skillTier: r.skillTier as SkillTier,
    games: r.games,
    wins: r.wins,
    winRate: r.winRate,
  }))
}

// ---------------------------------------------------------------------------
// Meta / Dashboard queries
// ---------------------------------------------------------------------------

/** Top N heroes by win rate for a tier */
export async function getTopHeroes(
  tier: SkillTier,
  limit = 10
): Promise<HeroStats[]> {
  const all = await getHeroStats(tier)
  return all.filter((h) => h.games >= 50).slice(0, limit)
}

/** Bottom N heroes by win rate for a tier */
export async function getBottomHeroes(
  tier: SkillTier,
  limit = 10
): Promise<HeroStats[]> {
  const all = await getHeroStats(tier)
  return all
    .filter((h) => h.games >= 50)
    .reverse()
    .slice(0, limit)
}

/** Top synergy pairs for a tier */
export async function getTopSynergies(
  tier: SkillTier,
  limit = 10
): Promise<HeroPairwiseStats[]> {
  const rows = await db
    .select()
    .from(heroPairwiseStatsTable)
    .where(
      and(
        eq(heroPairwiseStatsTable.relationship, 'with'),
        eq(heroPairwiseStatsTable.skillTier, tier)
      )
    )
    .orderBy(desc(heroPairwiseStatsTable.winRate))
    .limit(limit)

  return rows.map((r) => ({
    heroA: r.heroA,
    heroB: r.heroB,
    relationship: r.relationship as 'with' | 'against',
    skillTier: r.skillTier as SkillTier,
    games: r.games,
    wins: r.wins,
    winRate: r.winRate,
  }))
}

/** Top counter matchups for a tier */
export async function getTopCounters(
  tier: SkillTier,
  limit = 10
): Promise<HeroPairwiseStats[]> {
  const rows = await db
    .select()
    .from(heroPairwiseStatsTable)
    .where(
      and(
        eq(heroPairwiseStatsTable.relationship, 'against'),
        eq(heroPairwiseStatsTable.skillTier, tier)
      )
    )
    .orderBy(desc(heroPairwiseStatsTable.winRate))
    .limit(limit)

  return rows.map((r) => ({
    heroA: r.heroA,
    heroB: r.heroB,
    relationship: r.relationship as 'with' | 'against',
    skillTier: r.skillTier as SkillTier,
    games: r.games,
    wins: r.wins,
    winRate: r.winRate,
  }))
}

/** Power picks: hero+map combos with high win rate for a tier */
export async function getPowerPicks(
  tier: SkillTier,
  threshold = 55,
  limit = 15
): Promise<HeroMapStats[]> {
  const rows = await db
    .select()
    .from(heroMapStatsAggregate)
    .where(
      and(
        eq(heroMapStatsAggregate.skillTier, tier),
        sql`${heroMapStatsAggregate.games} >= 100`,
        sql`${heroMapStatsAggregate.winRate} >= ${threshold}`
      )
    )
    .orderBy(desc(heroMapStatsAggregate.winRate))
    .limit(limit)

  return rows.map((r) => ({
    hero: r.hero,
    map: r.map,
    skillTier: r.skillTier as SkillTier,
    games: r.games,
    wins: r.wins,
    winRate: r.winRate,
  }))
}

// ---------------------------------------------------------------------------
// Personal player queries
// ---------------------------------------------------------------------------

/** All tracked battletags */
export async function getTrackedBattletags(): Promise<TrackedBattletag[]> {
  const rows = await db.select().from(trackedBattletags)
  return rows.map((r) => ({
    battletag: r.battletag,
    region: r.region ?? 1,
    lastSynced: r.lastSynced,
  }))
}

/** Per-hero stats for a battletag */
export async function getPlayerHeroStats(
  battletag: string
): Promise<PlayerHeroStats[]> {
  const rows = await db
    .select()
    .from(playerHeroStatsTable)
    .where(eq(playerHeroStatsTable.battletag, battletag))
    .orderBy(desc(playerHeroStatsTable.games))

  return rows.map((r) => ({
    battletag: r.battletag,
    hero: r.hero,
    games: r.games,
    wins: r.wins,
    winRate: r1(r.winRate),
    mawp: r.mawp != null ? r1(r.mawp) : null,
    avgKills: r1(r.avgKills ?? 0),
    avgDeaths: r1(r.avgDeaths ?? 0),
    avgAssists: r1(r.avgAssists ?? 0),
    recentWinRate: r.recentWinRate != null ? r1(r.recentWinRate) : null,
    trend: r.trend != null ? r1(r.trend) : null,
  }))
}

/** Per-hero-per-map stats for a battletag */
export async function getPlayerHeroMapStats(
  battletag: string,
  hero?: string
): Promise<PlayerHeroMapStats[]> {
  const conditions = [eq(playerHeroMapStatsTable.battletag, battletag)]
  if (hero) conditions.push(eq(playerHeroMapStatsTable.hero, hero))

  const rows = await db
    .select()
    .from(playerHeroMapStatsTable)
    .where(and(...conditions))
    .orderBy(desc(playerHeroMapStatsTable.winRate))

  return rows.map((r) => ({
    battletag: r.battletag,
    hero: r.hero,
    map: r.map,
    games: r.games,
    wins: r.wins,
    winRate: r1(r.winRate),
  }))
}

/** Match history for a battletag */
export async function getPlayerMatchHistory(
  battletag: string,
  limit = 50
): Promise<PlayerMatch[]> {
  const rows = await db
    .select()
    .from(playerMatchHistoryTable)
    .where(eq(playerMatchHistoryTable.battletag, battletag))
    .orderBy(desc(playerMatchHistoryTable.gameDate))
    .limit(limit)

  return rows.map((r) => ({
    battletag: r.battletag,
    replayId: r.replayId,
    hero: r.hero,
    map: r.map,
    win: r.win,
    gameDate: r.gameDate,
    gameLength: r.gameLength ?? 0,
    kills: r.kills ?? 0,
    deaths: r.deaths ?? 0,
    assists: r.assists ?? 0,
    heroDamage: r.heroDamage ?? 0,
    siegeDamage: r.siegeDamage ?? 0,
    healing: r.healing ?? 0,
    experience: r.experience ?? 0,
    talents: r.talents,
    gameMode: r.gameMode ?? 'Unknown',
    rank: r.rank ?? null,
  }))
}

/** Players who are strong on a particular hero (for draft assistant) */
export async function getPlayersStrongOnHero(
  battletags: string[],
  hero: string
): Promise<PlayerHeroStats[]> {
  if (battletags.length === 0) return []

  const rows = await db
    .select()
    .from(playerHeroStatsTable)
    .where(
      and(
        inArray(playerHeroStatsTable.battletag, battletags),
        eq(playerHeroStatsTable.hero, hero)
      )
    )
    .orderBy(desc(playerHeroStatsTable.winRate))

  return rows
    .map((r) => ({
      battletag: r.battletag,
      hero: r.hero,
      games: r.games,
      wins: r.wins,
      winRate: r1(r.winRate),
      mawp: r.mawp != null ? r1(r.mawp) : null,
      avgKills: r1(r.avgKills ?? 0),
      avgDeaths: r1(r.avgDeaths ?? 0),
      avgAssists: r1(r.avgAssists ?? 0),
      recentWinRate: r.recentWinRate != null ? r1(r.recentWinRate) : null,
      trend: r.trend != null ? r1(r.trend) : null,
    }))
    .filter((r) => r.games >= 10 && r.winRate >= 52)
}

// ---------------------------------------------------------------------------
// Map-level personal queries
// ---------------------------------------------------------------------------

/** Aggregate a player's win rate per map (across all heroes) */
export async function getPlayerMapStats(
  battletag: string
): Promise<{ map: string; games: number; wins: number; winRate: number }[]> {
  const rows = await db
    .select({
      map: playerMatchHistoryTable.map,
      games: sql<number>`count(*)::int`,
      wins: sql<number>`sum(case when ${playerMatchHistoryTable.win} then 1 else 0 end)::int`,
    })
    .from(playerMatchHistoryTable)
    .where(eq(playerMatchHistoryTable.battletag, battletag))
    .groupBy(playerMatchHistoryTable.map)
    .orderBy(desc(sql`count(*)`))

  return rows.map((r) => ({
    map: r.map,
    games: r.games,
    wins: r.wins,
    winRate: r.games > 0 ? Math.round((r.wins / r.games) * 1000) / 10 : 0,
  }))
}

/** Player matches filtered to a specific map */
export async function getPlayerMatchesOnMap(
  battletag: string,
  map: string,
  limit = 50
): Promise<PlayerMatch[]> {
  const rows = await db
    .select()
    .from(playerMatchHistoryTable)
    .where(
      and(
        eq(playerMatchHistoryTable.battletag, battletag),
        eq(playerMatchHistoryTable.map, map)
      )
    )
    .orderBy(desc(playerMatchHistoryTable.gameDate))
    .limit(limit)

  return rows.map((r) => ({
    battletag: r.battletag,
    replayId: r.replayId,
    hero: r.hero,
    map: r.map,
    win: r.win,
    gameDate: r.gameDate,
    gameLength: r.gameLength ?? 0,
    kills: r.kills ?? 0,
    deaths: r.deaths ?? 0,
    assists: r.assists ?? 0,
    heroDamage: r.heroDamage ?? 0,
    siegeDamage: r.siegeDamage ?? 0,
    healing: r.healing ?? 0,
    experience: r.experience ?? 0,
    talents: r.talents,
    gameMode: r.gameMode ?? 'Unknown',
    rank: r.rank ?? null,
  }))
}

// ---------------------------------------------------------------------------
// Draft-specific queries
// ---------------------------------------------------------------------------

/** Build the full DraftData bundle for a given tier and map */
export async function getDraftData(
  tier: SkillTier,
  map: string,
  battletags: string[]
): Promise<DraftData> {
  // Hero aggregate stats
  const heroStatsList = await getHeroStats(tier)
  const heroStats: DraftData['heroStats'] = {}
  for (const h of heroStatsList) {
    heroStats[h.hero] = {
      winRate: h.winRate,
      pickRate: h.pickRate,
      banRate: h.banRate,
      games: h.games,
    }
  }

  // Hero-map win rates
  const heroMapList = await getHeroMapStats(tier, map)
  const heroMapWinRates: DraftData['heroMapWinRates'] = {}
  for (const h of heroMapList) {
    heroMapWinRates[h.hero] = { winRate: h.winRate, games: h.games }
  }

  // Pairwise synergies and counters — fetch all for the tier
  const [synRows, counterRows] = await Promise.all([
    db
      .select()
      .from(heroPairwiseStatsTable)
      .where(
        and(
          eq(heroPairwiseStatsTable.relationship, 'with'),
          eq(heroPairwiseStatsTable.skillTier, tier)
        )
      ),
    db
      .select()
      .from(heroPairwiseStatsTable)
      .where(
        and(
          eq(heroPairwiseStatsTable.relationship, 'against'),
          eq(heroPairwiseStatsTable.skillTier, tier)
        )
      ),
  ])

  // DB stores both directions (A→B and B→A), so just map directly
  const synergies: DraftData['synergies'] = {}
  for (const p of synRows) {
    if (!synergies[p.heroA]) synergies[p.heroA] = {}
    synergies[p.heroA][p.heroB] = { winRate: p.winRate, games: p.games }
  }

  const counters: DraftData['counters'] = {}
  for (const p of counterRows) {
    if (!counters[p.heroA]) counters[p.heroA] = {}
    counters[p.heroA][p.heroB] = { winRate: p.winRate, games: p.games }
  }

  // Player personal stats
  const playerStats: DraftData['playerStats'] = {}
  const playerMapStats: DraftData['playerMapStats'] = {}

  for (const bt of battletags) {
    const heroStatsForPlayer = await getPlayerHeroStats(bt)
    playerStats[bt] = {}
    for (const h of heroStatsForPlayer) {
      playerStats[bt][h.hero] = {
        games: h.games,
        wins: h.wins,
        winRate: h.winRate,
      }
    }

    const heroMapForPlayer = await getPlayerHeroMapStats(bt, undefined)
    playerMapStats[bt] = {}
    for (const h of heroMapForPlayer.filter((hm) => hm.map === map)) {
      playerMapStats[bt][h.hero] = {
        winRate: h.winRate,
        games: h.games,
      }
    }
  }

  return {
    heroStats,
    heroMapWinRates,
    synergies,
    counters,
    playerStats,
    playerMapStats,
  }
}

/** Check if a battletag is registered (exists in tracked battletags or match history) */
export async function isRegisteredBattletag(battletag: string): Promise<boolean> {
  // Check tracked_battletags first
  const tracked = await db
    .select({ id: trackedBattletags.id })
    .from(trackedBattletags)
    .where(eq(trackedBattletags.battletag, battletag))
    .limit(1)

  if (tracked.length > 0) return true

  // Fallback: check if they have match history
  const matches = await db
    .select({ id: playerMatchHistoryTable.id })
    .from(playerMatchHistoryTable)
    .where(eq(playerMatchHistoryTable.battletag, battletag))
    .limit(1)

  return matches.length > 0
}
