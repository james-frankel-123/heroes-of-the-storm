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
import { eq, and, desc, asc, sql, inArray, gte } from 'drizzle-orm'

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

import type { DraftData, CompositionData } from '@/lib/draft/types'
import compositionsJson from '@/lib/data/compositions.json'
import { computeBaselineCompWR } from '@/lib/draft/composition'

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

/** All hero-map stats for a tier in one query, grouped by hero */
export async function getAllHeroMapStats(
  tier: SkillTier
): Promise<Record<string, HeroMapStats[]>> {
  const rows = await db
    .select()
    .from(heroMapStatsAggregate)
    .where(eq(heroMapStatsAggregate.skillTier, tier))
    .orderBy(desc(heroMapStatsAggregate.winRate))

  const result: Record<string, HeroMapStats[]> = {}
  for (const r of rows) {
    const entry: HeroMapStats = {
      hero: r.hero,
      map: r.map,
      skillTier: r.skillTier as SkillTier,
      games: r.games,
      wins: r.wins,
      winRate: r.winRate,
    }
    if (!result[r.hero]) result[r.hero] = []
    result[r.hero].push(entry)
  }
  return result
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

/** All talent stats for a tier in one query, grouped by hero */
export async function getAllTalentStats(
  tier: SkillTier
): Promise<Record<string, HeroTalentStats[]>> {
  const rows = await db
    .select()
    .from(heroTalentStatsTable)
    .where(eq(heroTalentStatsTable.skillTier, tier))
    .orderBy(
      asc(heroTalentStatsTable.hero),
      asc(heroTalentStatsTable.talentTier),
      desc(heroTalentStatsTable.winRate)
    )

  const result: Record<string, HeroTalentStats[]> = {}
  for (const r of rows) {
    const entry: HeroTalentStats = {
      hero: r.hero,
      skillTier: r.skillTier as SkillTier,
      talentTier: r.talentTier,
      talentName: r.talentName,
      games: r.games,
      wins: r.wins,
      winRate: r.winRate,
      pickRate: r.pickRate ?? 0,
    }
    if (!result[r.hero]) result[r.hero] = []
    result[r.hero].push(entry)
  }
  return result
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

/**
 * All pairwise stats for a tier in one query, grouped by hero.
 * Returns { synergies: hero → HeroPairwiseStats[], counters: hero → HeroPairwiseStats[] }
 * Only includes heroA rows (not B→A duplicates).
 */
export async function getAllPairwiseStats(
  tier: SkillTier
): Promise<{
  synergies: Record<string, HeroPairwiseStats[]>
  counters: Record<string, HeroPairwiseStats[]>
}> {
  const rows = await db
    .select()
    .from(heroPairwiseStatsTable)
    .where(eq(heroPairwiseStatsTable.skillTier, tier))
    .orderBy(desc(heroPairwiseStatsTable.winRate))

  const synergies: Record<string, HeroPairwiseStats[]> = {}
  const counters: Record<string, HeroPairwiseStats[]> = {}

  for (const r of rows) {
    const entry: HeroPairwiseStats = {
      heroA: r.heroA,
      heroB: r.heroB,
      relationship: r.relationship as 'with' | 'against',
      skillTier: r.skillTier as SkillTier,
      games: r.games,
      wins: r.wins,
      winRate: r.winRate,
    }
    if (r.relationship === 'with') {
      if (!synergies[r.heroA]) synergies[r.heroA] = []
      synergies[r.heroA].push(entry)
    } else {
      if (!counters[r.heroA]) counters[r.heroA] = []
      counters[r.heroA].push(entry)
    }
  }

  return { synergies, counters }
}

// ---------------------------------------------------------------------------
// Meta / Dashboard queries
// ---------------------------------------------------------------------------

/**
 * Merge Cho and Gall into a single "Cho'gall" entry.
 * They're one hero played by two people — aggregate stats are identical
 * (same games, same wins, same WR). We keep whichever row has more games
 * (they should be equal) and rename to Cho'gall.
 */
function mergeChoGall(heroes: HeroStats[]): HeroStats[] {
  const cho = heroes.find((h) => h.hero === 'Cho')
  const gall = heroes.find((h) => h.hero === 'Gall')
  const merged = heroes.filter((h) => h.hero !== 'Cho' && h.hero !== 'Gall')

  // Use whichever has data (they should be identical); prefer Cho's row
  const base = cho ?? gall
  if (base) {
    merged.push({ ...base, hero: "Cho'gall" })
  }

  return merged
}

/** Top N heroes by win rate for a tier */
export async function getTopHeroes(
  tier: SkillTier,
  limit = 10
): Promise<HeroStats[]> {
  const all = await getHeroStats(tier)
  const merged = mergeChoGall(all)
  return merged
    .filter((h) => h.games >= 50)
    .sort((a, b) => b.winRate - a.winRate)
    .slice(0, limit)
}

/** Bottom N heroes by win rate for a tier */
export async function getBottomHeroes(
  tier: SkillTier,
  limit = 10
): Promise<HeroStats[]> {
  const all = await getHeroStats(tier)
  const merged = mergeChoGall(all)
  return merged
    .filter((h) => h.games >= 50)
    .sort((a, b) => a.winRate - b.winRate)
    .slice(0, limit)
}

/**
 * Merge Cho/Gall references in pairwise stats into "Cho'gall".
 * The DB has separate rows for Cho+X and Gall+X with identical stats.
 * Rename both to Cho'gall, drop the Cho+Gall self-pair, then dedup
 * by keeping the first occurrence of each (heroA, heroB) pair.
 */
function mergeChoGallPairwise(pairs: HeroPairwiseStats[]): HeroPairwiseStats[] {
  const renamed = pairs.map((p) => ({
    ...p,
    heroA: p.heroA === 'Cho' || p.heroA === 'Gall' ? "Cho'gall" : p.heroA,
    heroB: p.heroB === 'Cho' || p.heroB === 'Gall' ? "Cho'gall" : p.heroB,
  }))

  // Drop self-pairs (Cho'gall + Cho'gall)
  const filtered = renamed.filter((p) => p.heroA !== p.heroB)

  // Dedup: keep first occurrence of each pair (order-independent key,
  // since Cho→Fenix and Fenix→Gall both become Cho'gall↔Fenix after rename)
  const seen = new Set<string>()
  return filtered.filter((p) => {
    const [a, b] = [p.heroA, p.heroB].sort()
    const key = `${a}|${b}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

/** Top synergy pairs for a tier (deduped: only heroA < heroB alphabetically) */
export async function getTopSynergies(
  tier: SkillTier,
  limit = 10
): Promise<HeroPairwiseStats[]> {
  // Fetch extra rows to account for Cho/Gall dedup reducing count
  const rows = await db
    .select()
    .from(heroPairwiseStatsTable)
    .where(
      and(
        eq(heroPairwiseStatsTable.relationship, 'with'),
        eq(heroPairwiseStatsTable.skillTier, tier),
        sql`${heroPairwiseStatsTable.heroA} < ${heroPairwiseStatsTable.heroB}`
      )
    )
    .orderBy(desc(heroPairwiseStatsTable.winRate))
    .limit(limit + 10)

  const mapped = rows.map((r) => ({
    heroA: r.heroA,
    heroB: r.heroB,
    relationship: r.relationship as 'with' | 'against',
    skillTier: r.skillTier as SkillTier,
    games: r.games,
    wins: r.wins,
    winRate: r.winRate,
  }))

  return mergeChoGallPairwise(mapped)
    .sort((a, b) => b.winRate - a.winRate)
    .slice(0, limit)
}

/** Top counter matchups for a tier (deduped: only heroA < heroB alphabetically) */
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
        eq(heroPairwiseStatsTable.skillTier, tier),
        sql`${heroPairwiseStatsTable.heroA} < ${heroPairwiseStatsTable.heroB}`
      )
    )
    .orderBy(desc(heroPairwiseStatsTable.winRate))
    .limit(limit + 10)

  const mapped = rows.map((r) => ({
    heroA: r.heroA,
    heroB: r.heroB,
    relationship: r.relationship as 'with' | 'against',
    skillTier: r.skillTier as SkillTier,
    games: r.games,
    wins: r.wins,
    winRate: r.winRate,
  }))

  return mergeChoGallPairwise(mapped)
    .sort((a, b) => b.winRate - a.winRate)
    .slice(0, limit)
}

/** Power picks: hero+map combos with high win rate for a tier */
export async function getPowerPicks(
  tier: SkillTier,
  threshold = 55,
  limit = 15
): Promise<HeroMapStats[]> {
  // Fetch extra to account for Cho/Gall dedup
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
    .limit(limit + 10)

  const mapped = rows.map((r) => ({
    hero: r.hero === 'Cho' || r.hero === 'Gall' ? "Cho'gall" : r.hero,
    map: r.map,
    skillTier: r.skillTier as SkillTier,
    games: r.games,
    wins: r.wins,
    winRate: r.winRate,
  }))

  // Dedup Cho'gall entries on the same map (identical stats)
  const seen = new Set<string>()
  return mapped
    .filter((p) => {
      const key = `${p.hero}|${p.map}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
    .slice(0, limit)
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
    .orderBy(desc(playerHeroStatsTable.mawp))

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
    .filter((r) => r.games >= 10 && (r.mawp ?? r.winRate) >= 52)
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

/** Per-hero aggregate stats for a battletag since a given date (season snapshot) */
export async function getPlayerHeroStatsSince(
  battletag: string,
  since: Date,
): Promise<{ hero: string; games: number; wins: number }[]> {
  const rows = await db
    .select({
      hero: playerMatchHistoryTable.hero,
      games: sql<number>`count(*)::int`,
      wins: sql<number>`sum(case when ${playerMatchHistoryTable.win} then 1 else 0 end)::int`,
    })
    .from(playerMatchHistoryTable)
    .where(and(
      eq(playerMatchHistoryTable.battletag, battletag),
      gte(playerMatchHistoryTable.gameDate, since),
    ))
    .groupBy(playerMatchHistoryTable.hero)
    .orderBy(desc(sql`count(*)`))

  return rows.map((r) => ({ hero: r.hero, games: r.games, wins: r.wins }))
}

/** Per-year breakdown of hero and map stats for a battletag */
export async function getPlayerSeasonBreakdown(
  battletag: string,
): Promise<{
  year: number
  heroStats: { hero: string; games: number; wins: number; winRate: number }[]
  mapStats: { map: string; games: number; wins: number; winRate: number }[]
}[]> {
  const heroRows = await db
    .select({
      year: sql<number>`EXTRACT(YEAR FROM ${playerMatchHistoryTable.gameDate})::int`,
      hero: playerMatchHistoryTable.hero,
      games: sql<number>`count(*)::int`,
      wins: sql<number>`sum(case when ${playerMatchHistoryTable.win} then 1 else 0 end)::int`,
    })
    .from(playerMatchHistoryTable)
    .where(eq(playerMatchHistoryTable.battletag, battletag))
    .groupBy(sql`1`, playerMatchHistoryTable.hero)
    .orderBy(desc(sql`1`), desc(sql`count(*)`))

  const mapRows = await db
    .select({
      year: sql<number>`EXTRACT(YEAR FROM ${playerMatchHistoryTable.gameDate})::int`,
      map: playerMatchHistoryTable.map,
      games: sql<number>`count(*)::int`,
      wins: sql<number>`sum(case when ${playerMatchHistoryTable.win} then 1 else 0 end)::int`,
    })
    .from(playerMatchHistoryTable)
    .where(eq(playerMatchHistoryTable.battletag, battletag))
    .groupBy(sql`1`, playerMatchHistoryTable.map)
    .orderBy(desc(sql`1`), desc(sql`count(*)`))

  const years = new Set([...heroRows.map(r => r.year), ...mapRows.map(r => r.year)])
  return [...years].sort((a, b) => b - a).map(year => ({
    year,
    heroStats: heroRows
      .filter(r => r.year === year)
      .map(r => ({
        hero: r.hero, games: r.games, wins: r.wins,
        winRate: r.games > 0 ? Math.round((r.wins / r.games) * 1000) / 10 : 0,
      })),
    mapStats: mapRows
      .filter(r => r.year === year)
      .map(r => ({
        map: r.map, games: r.games, wins: r.wins,
        winRate: r.games > 0 ? Math.round((r.wins / r.games) * 1000) / 10 : 0,
      })),
  }))
}

/** Per-map aggregate stats for a battletag since a given date (season snapshot) */
export async function getPlayerMapStatsSince(
  battletag: string,
  since: Date,
): Promise<{ map: string; games: number; wins: number; winRate: number }[]> {
  const rows = await db
    .select({
      map: playerMatchHistoryTable.map,
      games: sql<number>`count(*)::int`,
      wins: sql<number>`sum(case when ${playerMatchHistoryTable.win} then 1 else 0 end)::int`,
    })
    .from(playerMatchHistoryTable)
    .where(and(
      eq(playerMatchHistoryTable.battletag, battletag),
      gte(playerMatchHistoryTable.gameDate, since),
    ))
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

  // Hero-map win rates — load all maps in one query
  const allHeroMapStats = await getAllHeroMapStats(tier)
  const heroMapWinRates: DraftData['heroMapWinRates'] = {}
  for (const [hero, mapStats] of Object.entries(allHeroMapStats)) {
    for (const s of mapStats) {
      if (!heroMapWinRates[s.map]) heroMapWinRates[s.map] = {}
      heroMapWinRates[s.map][hero] = { winRate: s.winRate, games: s.games }
    }
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
        mawp: h.mawp,
      }
    }

    const heroMapForPlayer = await getPlayerHeroMapStats(bt, undefined)
    playerMapStats[bt] = {}
    for (const h of heroMapForPlayer) {
      if (!playerMapStats[bt][h.map]) playerMapStats[bt][h.map] = {}
      playerMapStats[bt][h.map][h.hero] = {
        winRate: h.winRate,
        games: h.games,
      }
    }
  }

  // Composition data from static JSON
  const allComps = compositionsJson as Record<string, CompositionData[]>
  const compositions = allComps[tier] ?? []
  const baselineCompWR = computeBaselineCompWR(compositions)

  return {
    heroStats,
    heroMapWinRates,
    synergies,
    counters,
    playerStats,
    playerMapStats,
    compositions,
    baselineCompWR,
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
