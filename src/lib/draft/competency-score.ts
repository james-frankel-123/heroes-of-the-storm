/**
 * Player Competency Scoring System
 *
 * Calculates how competent a player is with each hero based on:
 * - Win rate (skill level)
 * - Games played (experience)
 * - Map performance (map-specific bonus)
 */

import { HeroStats, MapStats, PlayerData } from '@/types'

export interface HeroCompetency {
  hero: string
  winRate: number
  games: number
  competencyScore: number
  mapBonus: boolean  // True if hero is strong on selected map
  role?: string
}

export interface PlayerCompetency {
  battletag: string
  slot: number
  topHeroes: HeroCompetency[]
  totalGames: number
  overallWinRate: number
}

/**
 * Calculate competency score for a single hero
 *
 * Formula: (winRate / 100) * log(games + 1) * mapMultiplier
 *
 * - Win rate normalized to 0-1
 * - Log scale for games prevents over-weighting of high game counts
 * - Map multiplier rewards heroes player performs well on selected map
 */
export function calculateHeroCompetency(
  hero: string,
  winRate: number,
  games: number,
  selectedMap?: string,
  mapStats?: MapStats[]
): HeroCompetency {
  if (games === 0) {
    return {
      hero,
      winRate: 0,
      games: 0,
      competencyScore: 0,
      mapBonus: false
    }
  }

  // Check if hero performs well on selected map (60%+ win rate with 3+ games)
  let mapBonus = false
  let mapMultiplier = 1.0

  if (selectedMap && mapStats) {
    const mapData = mapStats.find(m => m.map === selectedMap)
    if (mapData) {
      const heroOnMap = mapData.heroes.find(h => h.hero === hero)
      if (heroOnMap && heroOnMap.games >= 3 && heroOnMap.winRate >= 60) {
        mapBonus = true
        mapMultiplier = 1.2
      }
    }
  }

  // Calculate competency score
  const normalizedWinRate = winRate / 100
  const experienceFactor = Math.log(games + 1)
  const competencyScore = normalizedWinRate * experienceFactor * mapMultiplier

  return {
    hero,
    winRate,
    games,
    competencyScore,
    mapBonus
  }
}

/**
 * Calculate competency for all heroes a player has played
 */
export function calculatePlayerCompetency(
  battletag: string,
  slot: number,
  playerData: PlayerData | null,
  selectedMap?: string
): PlayerCompetency {
  if (!playerData) {
    return {
      battletag,
      slot,
      topHeroes: [],
      totalGames: 0,
      overallWinRate: 0
    }
  }

  // Calculate competency for each hero
  const heroCompetencies = playerData.heroStats.map(heroStat =>
    calculateHeroCompetency(
      heroStat.hero,
      heroStat.winRate,
      heroStat.games,
      selectedMap,
      playerData.mapStats
    )
  )

  // Sort by competency score descending
  const sortedHeroes = heroCompetencies
    .filter(h => h.games > 0)
    .sort((a, b) => b.competencyScore - a.competencyScore)

  return {
    battletag,
    slot,
    topHeroes: sortedHeroes,
    totalGames: playerData.totalGames,
    overallWinRate: playerData.overallWinRate
  }
}

/**
 * Get top N heroes for a player
 */
export function getTopHeroes(
  playerCompetency: PlayerCompetency,
  count: number = 5
): HeroCompetency[] {
  return playerCompetency.topHeroes.slice(0, count)
}

/**
 * Find which player is most competent with a specific hero
 */
export function findBestPlayerForHero(
  hero: string,
  playerCompetencies: PlayerCompetency[]
): {
  player: PlayerCompetency | null
  competency: HeroCompetency | null
} {
  let bestPlayer: PlayerCompetency | null = null
  let bestCompetency: HeroCompetency | null = null
  let highestScore = 0

  for (const player of playerCompetencies) {
    const heroComp = player.topHeroes.find(h => h.hero === hero)
    if (heroComp && heroComp.competencyScore > highestScore) {
      bestPlayer = player
      bestCompetency = heroComp
      highestScore = heroComp.competencyScore
    }
  }

  return { player: bestPlayer, competency: bestCompetency }
}

/**
 * Get minimum games threshold for "competent" classification
 */
export function getMinimumCompetencyGames(): number {
  return 5  // Must have at least 5 games to be considered competent
}

/**
 * Get minimum win rate threshold for "good" classification
 */
export function getMinimumGoodWinRate(): number {
  return 50  // Must have 50%+ win rate to be considered good
}

/**
 * Check if player is competent with a hero
 */
export function isCompetentWith(
  hero: string,
  playerCompetency: PlayerCompetency
): boolean {
  const heroComp = playerCompetency.topHeroes.find(h => h.hero === hero)
  if (!heroComp) return false

  return (
    heroComp.games >= getMinimumCompetencyGames() &&
    heroComp.winRate >= 45  // Slightly below 50% is acceptable for experience
  )
}

/**
 * Check if player is good with a hero
 */
export function isGoodWith(
  hero: string,
  playerCompetency: PlayerCompetency
): boolean {
  const heroComp = playerCompetency.topHeroes.find(h => h.hero === hero)
  if (!heroComp) return false

  return (
    heroComp.games >= getMinimumCompetencyGames() &&
    heroComp.winRate >= getMinimumGoodWinRate()
  )
}

/**
 * Format competency for display
 */
export function formatCompetency(competency: HeroCompetency): string {
  return `${competency.hero} (${competency.winRate}% WR, ${competency.games}g)`
}

/**
 * Format competency with map bonus indicator
 */
export function formatCompetencyWithMap(competency: HeroCompetency): string {
  const base = formatCompetency(competency)
  return competency.mapBonus ? `${base} ⭐` : base
}
