/**
 * Data access layer.
 * Currently backed by mock data. When DB is ready, swap implementations
 * to drizzle queries — calling code doesn't change.
 */

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

import {
  mockHeroStats,
  mockMapStats,
  mockHeroMapStats,
  mockTalentStats,
  mockPairwiseStats,
  mockPlayerHeroStats,
  mockPlayerHeroMapStats,
  mockPlayerMatchHistory,
  mockTrackedBattletags,
  MAPS,
} from '@/lib/mock/data'

// ---------------------------------------------------------------------------
// Aggregate queries
// ---------------------------------------------------------------------------

/** All hero stats for a skill tier, sorted by win rate desc */
export async function getHeroStats(tier: SkillTier): Promise<HeroStats[]> {
  return mockHeroStats
    .filter((h) => h.skillTier === tier)
    .sort((a, b) => b.winRate - a.winRate)
}

/** Single hero stats across all tiers */
export async function getHeroStatsByName(hero: string): Promise<HeroStats[]> {
  return mockHeroStats.filter((h) => h.hero === hero)
}

/** All map stats for a skill tier */
export async function getMapStats(tier: SkillTier): Promise<MapStats[]> {
  return mockMapStats.filter((m) => m.skillTier === tier)
}

/** All map names */
export async function getAllMaps(): Promise<string[]> {
  return MAPS
}

/** Hero performance on a specific map for a tier */
export async function getHeroMapStats(
  tier: SkillTier,
  map?: string,
  hero?: string
): Promise<HeroMapStats[]> {
  let result = mockHeroMapStats.filter((h) => h.skillTier === tier)
  if (map) result = result.filter((h) => h.map === map)
  if (hero) result = result.filter((h) => h.hero === hero)
  return result.sort((a, b) => b.winRate - a.winRate)
}

/** Talent stats for a hero at a tier */
export async function getTalentStats(
  hero: string,
  tier: SkillTier
): Promise<HeroTalentStats[]> {
  return mockTalentStats
    .filter((t) => t.hero === hero && t.skillTier === tier)
    .sort((a, b) => a.talentTier - b.talentTier || b.winRate - a.winRate)
}

/** Pairwise stats: synergies or counters for a hero */
export async function getPairwiseStats(
  hero: string,
  relationship: 'with' | 'against',
  tier: SkillTier
): Promise<HeroPairwiseStats[]> {
  return mockPairwiseStats
    .filter(
      (p) =>
        (p.heroA === hero || p.heroB === hero) &&
        p.relationship === relationship &&
        p.skillTier === tier
    )
    .sort((a, b) => b.winRate - a.winRate)
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
  return all.filter((h) => h.games >= 200).slice(0, limit)
}

/** Bottom N heroes by win rate for a tier */
export async function getBottomHeroes(
  tier: SkillTier,
  limit = 10
): Promise<HeroStats[]> {
  const all = await getHeroStats(tier)
  return all.filter((h) => h.games >= 200).reverse().slice(0, limit)
}

/** Top synergy pairs for a tier */
export async function getTopSynergies(
  tier: SkillTier,
  limit = 10
): Promise<HeroPairwiseStats[]> {
  return mockPairwiseStats
    .filter((p) => p.relationship === 'with' && p.skillTier === tier)
    .sort((a, b) => b.winRate - a.winRate)
    .slice(0, limit)
}

/** Top counter matchups for a tier */
export async function getTopCounters(
  tier: SkillTier,
  limit = 10
): Promise<HeroPairwiseStats[]> {
  return mockPairwiseStats
    .filter((p) => p.relationship === 'against' && p.skillTier === tier)
    .sort((a, b) => b.winRate - a.winRate)
    .slice(0, limit)
}

/** Power picks: hero+map combos with >= threshold win rate */
export async function getPowerPicks(
  tier: SkillTier,
  threshold = 60,
  limit = 15
): Promise<HeroMapStats[]> {
  return mockHeroMapStats
    .filter(
      (h) => h.skillTier === tier && h.winRate >= threshold && h.games >= 30
    )
    .sort((a, b) => b.winRate - a.winRate)
    .slice(0, limit)
}

// ---------------------------------------------------------------------------
// Personal player queries
// ---------------------------------------------------------------------------

/** All tracked battletags */
export async function getTrackedBattletags(): Promise<TrackedBattletag[]> {
  return mockTrackedBattletags
}

/** Per-hero stats for a battletag */
export async function getPlayerHeroStats(
  battletag: string
): Promise<PlayerHeroStats[]> {
  return mockPlayerHeroStats
    .filter((p) => p.battletag === battletag)
    .sort((a, b) => b.games - a.games)
}

/** Per-hero-per-map stats for a battletag */
export async function getPlayerHeroMapStats(
  battletag: string,
  hero?: string
): Promise<PlayerHeroMapStats[]> {
  let result = mockPlayerHeroMapStats.filter((p) => p.battletag === battletag)
  if (hero) result = result.filter((p) => p.hero === hero)
  return result.sort((a, b) => b.winRate - a.winRate)
}

/** Match history for a battletag */
export async function getPlayerMatchHistory(
  battletag: string,
  limit = 50
): Promise<PlayerMatch[]> {
  return mockPlayerMatchHistory
    .filter((m) => m.battletag === battletag)
    .sort((a, b) => b.gameDate.getTime() - a.gameDate.getTime())
    .slice(0, limit)
}

/** Players who are strong on a particular hero (for draft assistant) */
export async function getPlayersStrongOnHero(
  battletags: string[],
  hero: string
): Promise<PlayerHeroStats[]> {
  return mockPlayerHeroStats
    .filter(
      (p) =>
        battletags.includes(p.battletag) &&
        p.hero === hero &&
        p.games >= 15 &&
        (p.mawp ?? p.winRate) >= 52
    )
    .sort((a, b) => (b.mawp ?? b.winRate) - (a.mawp ?? a.winRate))
}

// ---------------------------------------------------------------------------
// Map-level personal queries
// ---------------------------------------------------------------------------

/** Aggregate a player's win rate per map (across all heroes) */
export async function getPlayerMapStats(
  battletag: string
): Promise<{ map: string; games: number; wins: number; winRate: number }[]> {
  const matches = mockPlayerMatchHistory.filter(
    (m) => m.battletag === battletag
  )
  const byMap: Record<string, { games: number; wins: number }> = {}
  for (const m of matches) {
    if (!byMap[m.map]) byMap[m.map] = { games: 0, wins: 0 }
    byMap[m.map].games++
    if (m.win) byMap[m.map].wins++
  }
  return Object.entries(byMap)
    .map(([map, { games, wins }]) => ({
      map,
      games,
      wins,
      winRate: Math.round((wins / games) * 1000) / 10,
    }))
    .sort((a, b) => b.games - a.games)
}

/** Player matches filtered to a specific map */
export async function getPlayerMatchesOnMap(
  battletag: string,
  map: string,
  limit = 50
): Promise<PlayerMatch[]> {
  return mockPlayerMatchHistory
    .filter((m) => m.battletag === battletag && m.map === map)
    .sort((a, b) => b.gameDate.getTime() - a.gameDate.getTime())
    .slice(0, limit)
}

// ---------------------------------------------------------------------------
// Draft-specific queries
// ---------------------------------------------------------------------------

import type { DraftData } from '@/lib/draft/types'

/** Build the full DraftData bundle for a given tier and map */
export async function getDraftData(
  tier: SkillTier,
  map: string,
  battletags: string[]
): Promise<DraftData> {
  // Hero aggregate stats
  const heroStatsList = mockHeroStats.filter((h) => h.skillTier === tier)
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
  const heroMapList = mockHeroMapStats.filter(
    (h) => h.skillTier === tier && h.map === map
  )
  const heroMapWinRates: DraftData['heroMapWinRates'] = {}
  for (const h of heroMapList) {
    heroMapWinRates[h.hero] = { winRate: h.winRate, games: h.games }
  }

  // Pairwise synergies and counters
  const synergies: DraftData['synergies'] = {}
  const counters: DraftData['counters'] = {}

  const pairwise = mockPairwiseStats.filter((p) => p.skillTier === tier)
  for (const p of pairwise) {
    if (p.relationship === 'with') {
      if (!synergies[p.heroA]) synergies[p.heroA] = {}
      if (!synergies[p.heroB]) synergies[p.heroB] = {}
      synergies[p.heroA][p.heroB] = { winRate: p.winRate, games: p.games }
      synergies[p.heroB][p.heroA] = { winRate: p.winRate, games: p.games }
    } else {
      if (!counters[p.heroA]) counters[p.heroA] = {}
      if (!counters[p.heroB]) counters[p.heroB] = {}
      // A counters B: A has high WR against B
      counters[p.heroA][p.heroB] = { winRate: p.winRate, games: p.games }
      // B is countered by A: invert the win rate
      counters[p.heroB][p.heroA] = {
        winRate: Math.round((100 - p.winRate) * 10) / 10,
        games: p.games,
      }
    }
  }

  // Player personal stats
  const playerStats: DraftData['playerStats'] = {}
  const playerMapStats: DraftData['playerMapStats'] = {}

  for (const bt of battletags) {
    const heroStatsForPlayer = mockPlayerHeroStats.filter(
      (p) => p.battletag === bt
    )
    playerStats[bt] = {}
    for (const h of heroStatsForPlayer) {
      playerStats[bt][h.hero] = {
        games: h.games,
        winRate: h.winRate,
        mawp: h.mawp,
      }
    }

    const heroMapForPlayer = mockPlayerHeroMapStats.filter(
      (p) => p.battletag === bt && p.map === map
    )
    playerMapStats[bt] = {}
    for (const h of heroMapForPlayer) {
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

/** Check if a battletag is registered (exists in tracked battletags) */
export async function isRegisteredBattletag(battletag: string): Promise<boolean> {
  return mockTrackedBattletags.some(
    (bt) => bt.battletag.toLowerCase() === battletag.toLowerCase()
  )
}
