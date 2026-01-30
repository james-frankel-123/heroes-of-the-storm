import { HeroStats, MapStats, PlayerData, PowerPick, Insight } from '@/types'
import { TEAM_COMPOSITIONS, DuoStats } from '@/lib/data/team-compositions'
import { generatePowerPicks } from '@/lib/data/transform'

// Format commentary for display
export function formatCommentary(text: string): string {
  return text.trim()
}

// Parse streaming SSE response
export async function* parseStreamingResponse(response: Response): AsyncGenerator<string> {
  const reader = response.body?.getReader()
  const decoder = new TextDecoder()

  if (!reader) {
    throw new Error('No response body')
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          if (data === '[DONE]') continue
          yield data
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

// Hero Commentary Payload Types
export interface HeroCommentaryPayload {
  hero: string
  role: string
  winRate: number
  games: number
  wins: number
  losses: number
  mapPerformance: { map: string; winRate: number; games: number }[]
  playerContext: {
    totalGames: number
    overallWinRate: number
    topHeroes: { hero: string; winRate: number; games: number }[]
    roleStats: Record<string, { winRate: number; games: number }>
    bestMaps: { map: string; winRate: number }[]
    knownSynergies: { hero1: string; hero2: string; winRate: number }[]
  }
}

// Create hero commentary payload with full player context
export function createHeroPayload(
  heroStats: HeroStats,
  playerData: PlayerData
): HeroCommentaryPayload {
  // Extract hero's map performance
  const mapPerformance: { map: string; winRate: number; games: number }[] = []

  playerData.mapStats.forEach(mapStat => {
    const heroOnMap = mapStat.heroes?.find(h => h.hero === heroStats.hero)
    if (heroOnMap && heroOnMap.games > 0) {
      mapPerformance.push({
        map: mapStat.map,
        winRate: heroOnMap.winRate,
        games: heroOnMap.games
      })
    }
  })

  // Get top heroes for comparison
  const topHeroes = playerData.heroStats
    .sort((a, b) => b.winRate - a.winRate)
    .slice(0, 5)
    .map(h => ({
      hero: h.hero,
      winRate: h.winRate,
      games: h.games
    }))

  // Find synergies involving this hero
  const knownSynergies = TEAM_COMPOSITIONS
    .filter(comp => comp.heroes.includes(heroStats.hero))
    .map(comp => {
      const [hero1, hero2] = comp.heroes.split(' + ')
      return {
        hero1,
        hero2,
        winRate: comp.winRate
      }
    })

  return {
    hero: heroStats.hero,
    role: heroStats.role,
    winRate: heroStats.winRate,
    games: heroStats.games,
    wins: heroStats.wins,
    losses: heroStats.losses,
    mapPerformance,
    playerContext: {
      totalGames: playerData.totalGames,
      overallWinRate: playerData.overallWinRate,
      topHeroes,
      roleStats: Object.fromEntries(
        Object.entries(playerData.roleStats).map(([role, stats]) => [
          role,
          { winRate: stats.winRate, games: stats.games }
        ])
      ),
      bestMaps: playerData.mapStats
        .sort((a, b) => b.winRate - a.winRate)
        .slice(0, 3)
        .map(m => ({ map: m.map, winRate: m.winRate })),
      knownSynergies
    }
  }
}

// Map Commentary Payload Types
export interface MapCommentaryPayload {
  map: string
  winRate: number
  games: number
  wins: number
  losses: number
  topHeroes: { hero: string; winRate: number; games: number }[]
  weakHeroes: { hero: string; winRate: number; games: number }[]
  highPotentialHeroes: { hero: string; winRate: number; games: number }[]
  playerContext: {
    overallWinRate: number
    totalGames: number
    allMapStats: { map: string; winRate: number; games: number }[]
    topHeroesOverall: { hero: string; winRate: number; games: number }[]
    rolePreferences: Record<string, { winRate: number; games: number }>
    mapRank: number
    totalMaps: number
  }
}

// Create map commentary payload with full player context
export function createMapPayload(
  mapStats: MapStats,
  playerData: PlayerData
): MapCommentaryPayload {
  // Calculate map rank
  const sortedMaps = [...playerData.mapStats].sort((a, b) => b.winRate - a.winRate)
  const mapRank = sortedMaps.findIndex(m => m.map === mapStats.map) + 1

  return {
    map: mapStats.map,
    winRate: mapStats.winRate,
    games: mapStats.games,
    wins: mapStats.wins,
    losses: mapStats.losses,
    topHeroes: mapStats.heroes
      ?.filter(h => h.games >= 5)
      .sort((a, b) => b.winRate - a.winRate)
      .slice(0, 3)
      .map(h => ({ hero: h.hero, winRate: h.winRate, games: h.games })) || [],
    weakHeroes: mapStats.heroes
      ?.filter(h => h.games >= 5 && h.winRate < mapStats.winRate - 10)
      .sort((a, b) => a.winRate - b.winRate)
      .slice(0, 3)
      .map(h => ({ hero: h.hero, winRate: h.winRate, games: h.games })) || [],
    highPotentialHeroes: mapStats.heroes
      ?.filter(h => h.games >= 2 && h.games < 5 && h.winRate >= 60)
      .sort((a, b) => b.winRate - a.winRate)
      .slice(0, 2)
      .map(h => ({ hero: h.hero, winRate: h.winRate, games: h.games })) || [],
    playerContext: {
      overallWinRate: playerData.overallWinRate,
      totalGames: playerData.totalGames,
      allMapStats: sortedMaps.map(m => ({
        map: m.map,
        winRate: m.winRate,
        games: m.games
      })),
      topHeroesOverall: playerData.heroStats
        .sort((a, b) => b.winRate - a.winRate)
        .slice(0, 5)
        .map(h => ({ hero: h.hero, winRate: h.winRate, games: h.games })),
      rolePreferences: Object.fromEntries(
        Object.entries(playerData.roleStats).map(([role, stats]) => [
          role,
          { winRate: stats.winRate, games: stats.games }
        ])
      ),
      mapRank,
      totalMaps: playerData.mapStats.length
    }
  }
}

// Power Pick Commentary Payload Types
export interface PowerPickCommentaryPayload {
  hero: string
  map: string
  winRate: number
  games: number
  wins: number
  losses: number
  playerContext: {
    heroOverallStats: { winRate: number; games: number; totalWins: number }
    mapOverallStats: { winRate: number; games: number }
    overallWinRate: number
    otherPowerPicks: { hero: string; map: string; winRate: number }[]
    heroRole: string
    rolePerformance: { winRate: number; games: number }
  }
}

// Create power pick payload with comparative context
export function createPowerPickPayload(
  powerPick: PowerPick,
  playerData: PlayerData
): PowerPickCommentaryPayload {
  // Get hero's overall performance
  const heroOverall = playerData.heroStats.find(h => h.hero === powerPick.hero)

  // Get map's overall performance
  const mapOverall = playerData.mapStats.find(m => m.map === powerPick.map)

  // Get other power picks for pattern analysis
  const otherPowerPicks = generatePowerPicks(playerData)
    .filter(pp => pp.hero !== powerPick.hero || pp.map !== powerPick.map)
    .slice(0, 3)

  return {
    hero: powerPick.hero,
    map: powerPick.map,
    winRate: powerPick.winRate,
    games: powerPick.games,
    wins: Math.round((powerPick.winRate / 100) * powerPick.games),
    losses: powerPick.games - Math.round((powerPick.winRate / 100) * powerPick.games),
    playerContext: {
      heroOverallStats: {
        winRate: heroOverall?.winRate || 0,
        games: heroOverall?.games || 0,
        totalWins: heroOverall?.wins || 0
      },
      mapOverallStats: {
        winRate: mapOverall?.winRate || 0,
        games: mapOverall?.games || 0
      },
      overallWinRate: playerData.overallWinRate,
      otherPowerPicks,
      heroRole: heroOverall?.role || 'Unknown',
      rolePerformance: playerData.roleStats[heroOverall?.role || ''] || { winRate: 0, games: 0 }
    }
  }
}

// Team Synergy Commentary Payload Types
export interface TeamSynergyCommentaryPayload {
  hero1: string
  hero2: string
  winRate: number
  games: number
  wins: number
  losses: number
  playerContext: {
    hero1Stats: { winRate: number; games: number; role: string; bestMaps: string[] }
    hero2Stats: { winRate: number; games: number; role: string; bestMaps: string[] }
    overallWinRate: number
    otherSynergies: { heroes: string; winRate: number; games: number }[]
    roleComfort: Record<string, { winRate: number; games: number }>
    averageDuoWinRate: number
  }
}

// Create team synergy payload with individual hero context
export function createTeamSynergyPayload(
  synergy: DuoStats,
  playerData: PlayerData
): TeamSynergyCommentaryPayload {
  const [hero1Name, hero2Name] = synergy.heroes.split(' + ')

  // Get individual hero stats
  const hero1Stats = playerData.heroStats.find(h => h.hero === hero1Name)
  const hero2Stats = playerData.heroStats.find(h => h.hero === hero2Name)

  // Get hero map performance
  const getHeroBestMaps = (heroName: string) => {
    const mapPerf: { map: string; winRate: number }[] = []
    playerData.mapStats.forEach(mapStat => {
      const heroOnMap = mapStat.heroes?.find(h => h.hero === heroName)
      if (heroOnMap && heroOnMap.games >= 3) {
        mapPerf.push({ map: mapStat.map, winRate: heroOnMap.winRate })
      }
    })
    return mapPerf.sort((a, b) => b.winRate - a.winRate).slice(0, 3).map(m => m.map)
  }

  // Calculate average duo win rate
  const allDuos = TEAM_COMPOSITIONS
  const averageDuoWinRate = allDuos.reduce((sum, d) => sum + d.winRate, 0) / allDuos.length

  return {
    hero1: hero1Name,
    hero2: hero2Name,
    winRate: synergy.winRate,
    games: synergy.games,
    wins: synergy.wins,
    losses: synergy.losses,
    playerContext: {
      hero1Stats: {
        winRate: hero1Stats?.winRate || 0,
        games: hero1Stats?.games || 0,
        role: hero1Stats?.role || 'Unknown',
        bestMaps: getHeroBestMaps(hero1Name)
      },
      hero2Stats: {
        winRate: hero2Stats?.winRate || 0,
        games: hero2Stats?.games || 0,
        role: hero2Stats?.role || 'Unknown',
        bestMaps: getHeroBestMaps(hero2Name)
      },
      overallWinRate: playerData.overallWinRate,
      otherSynergies: allDuos
        .filter(d => d.heroes !== synergy.heroes && d.games >= 2)
        .sort((a, b) => b.winRate - a.winRate)
        .slice(0, 5)
        .map(d => ({ heroes: d.heroes, winRate: d.winRate, games: d.games })),
      roleComfort: Object.fromEntries(
        Object.entries(playerData.roleStats).map(([role, stats]) => [
          role,
          { winRate: stats.winRate, games: stats.games }
        ])
      ),
      averageDuoWinRate
    }
  }
}

// Insight Commentary Payload Types
export interface InsightCommentaryPayload {
  insightType: 'success' | 'warning' | 'info' | 'tip'
  title: string
  description: string
  playerData: {
    totalGames: number
    overallWinRate: number
    heroStats: HeroStats[]
    mapStats: MapStats[]
    roleStats: Record<string, { wins: number; games: number; winRate: number }>
    powerPicks: PowerPick[]
  }
  insightData: any
}

// Create insight payload with complete player profile
export function createInsightPayload(
  insight: Insight,
  playerData: PlayerData,
  insightData?: any
): InsightCommentaryPayload {
  return {
    insightType: insight.type,
    title: insight.title,
    description: insight.description,
    playerData: {
      totalGames: playerData.totalGames,
      overallWinRate: playerData.overallWinRate,
      heroStats: playerData.heroStats,
      mapStats: playerData.mapStats,
      roleStats: playerData.roleStats,
      powerPicks: generatePowerPicks(playerData)
    },
    insightData: insightData || {}
  }
}
