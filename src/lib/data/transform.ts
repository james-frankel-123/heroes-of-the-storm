import { PlayerData, HeroStats, MapStats, PowerPick, Insight, ReplayData, PartyStats, PartyGroup } from '@/types'

// Role classification (same as Python script)
const TANKS = ["Anub'arak", "Arthas", "Blaze", "Cho", "Diablo", "E.T.C.", "Garrosh",
  "Johanna", "Mal'Ganis", "Muradin", "Stitches", "Tyrael", "Chen"]
const BRUISERS = ["Artanis", "D.Va", "Dehaka", "Imperius", "Leoric", "Malthael",
  "Ragnaros", "Sonya", "Thrall", "Varian", "Xul", "Yrel", "Rexxar"]
const HEALERS = ["Alexstrasza", "Ana", "Anduin", "Auriel", "Brightwing", "Deckard",
  "Kharazim", "Li Li", "Lt. Morales", "Lúcio", "Malfurion", "Rehgar",
  "Stukov", "Uther", "Whitemane", "Tyrande"]
const RANGED = ["Azmodan", "Cassia", "Chromie", "Falstad", "Fenix", "Gall", "Greymane",
  "Gul'dan", "Hanzo", "Jaina", "Junkrat", "Kael'thas", "Kel'Thuzad",
  "Li-Ming", "Lunara", "Mephisto", "Nazeebo", "Nova", "Orphea", "Probius",
  "Raynor", "Sgt. Hammer", "Sylvanas", "Tassadar", "Tracer", "Tychus",
  "Valla", "Zagara", "Zul'jin"]
const MELEE = ["Alarak", "Gazlowe", "Illidan", "Kerrigan", "Maiev", "Murky", "Qhira",
  "Samuro", "The Butcher", "Valeera", "Zeratul"]
const SUPPORT = ["Abathur", "Medivh", "The Lost Vikings", "Zarya"]

export function classifyHeroRole(heroName: string): string {
  if (TANKS.includes(heroName)) return 'Tank'
  if (BRUISERS.includes(heroName)) return 'Bruiser'
  if (HEALERS.includes(heroName)) return 'Healer'
  if (RANGED.includes(heroName)) return 'Ranged Assassin'
  if (MELEE.includes(heroName)) return 'Melee Assassin'
  if (SUPPORT.includes(heroName)) return 'Support'
  return 'Unknown'
}

// Transform from hero-grouped API format (hero → stats)
export function transformHeroGroupedData(rawData: any, battletag: string): PlayerData {
  const stormLeagueData = rawData['Storm League'] || rawData

  const heroStatsMap = new Map<string, {
    wins: number
    losses: number
    games: number
  }>()

  // Process hero-grouped data (no map breakdown available from this API format)
  Object.entries(stormLeagueData).forEach(([heroName, stats]: [string, any]) => {
    if (typeof stats !== 'object') {
      return
    }

    const wins = parseInt(stats.wins) || 0
    const losses = parseInt(stats.losses) || 0
    const games = parseInt(stats.games_played) || 0

    if (games > 0) {
      heroStatsMap.set(heroName, { wins, losses, games })
    }
  })

  // Convert to HeroStats array
  const heroStats: HeroStats[] = Array.from(heroStatsMap.entries())
    .filter(([_, stats]) => stats.games >= 10) // Only heroes with 10+ games
    .map(([hero, stats]) => ({
      hero,
      role: classifyHeroRole(hero),
      wins: stats.wins,
      losses: stats.losses,
      games: stats.games,
      winRate: stats.games > 0 ? (stats.wins / stats.games) * 100 : 0,
    }))
    .sort((a, b) => b.winRate - a.winRate)

  // Calculate role stats
  const roleStats: Record<string, { wins: number; games: number; winRate: number }> = {}
  heroStats.forEach((hero) => {
    if (!roleStats[hero.role]) {
      roleStats[hero.role] = { wins: 0, games: 0, winRate: 0 }
    }
    roleStats[hero.role].wins += hero.wins
    roleStats[hero.role].games += hero.games
  })

  Object.keys(roleStats).forEach((role) => {
    const stats = roleStats[role]
    stats.winRate = stats.games > 0 ? (stats.wins / stats.games) * 100 : 0
  })

  // Calculate totals
  const totalGames = heroStats.reduce((sum, hero) => sum + hero.games, 0)
  const totalWins = heroStats.reduce((sum, hero) => sum + hero.wins, 0)
  const totalLosses = heroStats.reduce((sum, hero) => sum + hero.losses, 0)
  const overallWinRate = totalGames > 0 ? (totalWins / totalGames) * 100 : 0

  return {
    playerName: battletag,
    totalGames,
    totalWins,
    totalLosses,
    overallWinRate,
    heroStats,
    mapStats: [], // No map breakdown available from hero-grouped API
    roleStats,
  }
}

// Transform from map-grouped format (map → hero → stats)
export function transformHeroStatsData(rawData: any, battletag: string): PlayerData {
  const stormLeagueData = rawData['Storm League'] || rawData

  // Aggregate hero stats across all maps
  const heroStatsMap = new Map<string, {
    wins: number
    losses: number
    games: number
    maps: Map<string, { wins: number; losses: number; games: number }>
  }>()

  // Aggregate map stats
  const mapStatsMap = new Map<string, {
    wins: number
    losses: number
    games: number
    heroes: Map<string, { wins: number; losses: number; games: number; winRate: number }>
  }>()

  // Process each map
  Object.entries(stormLeagueData).forEach(([mapName, heroes]: [string, any]) => {
    // Initialize map stats
    if (!mapStatsMap.has(mapName)) {
      mapStatsMap.set(mapName, { wins: 0, losses: 0, games: 0, heroes: new Map() })
    }
    const mapStats = mapStatsMap.get(mapName)!

    // Process each hero on this map
    Object.entries(heroes).forEach(([heroName, stats]: [string, any]) => {
      const { wins, losses, games_played, win_rate } = stats

      // Update map stats
      mapStats.wins += wins
      mapStats.losses += losses
      mapStats.games += games_played
      mapStats.heroes.set(heroName, { wins, losses, games: games_played, winRate: win_rate })

      // Update hero stats
      if (!heroStatsMap.has(heroName)) {
        heroStatsMap.set(heroName, { wins: 0, losses: 0, games: 0, maps: new Map() })
      }
      const heroStats = heroStatsMap.get(heroName)!
      heroStats.wins += wins
      heroStats.losses += losses
      heroStats.games += games_played
      heroStats.maps.set(mapName, { wins, losses, games: games_played })
    })
  })

  // Convert to HeroStats array
  const heroStats: HeroStats[] = Array.from(heroStatsMap.entries())
    .filter(([_, stats]) => stats.games >= 10) // Only heroes with 10+ games
    .map(([hero, stats]) => ({
      hero,
      role: classifyHeroRole(hero),
      wins: stats.wins,
      losses: stats.losses,
      games: stats.games,
      winRate: stats.games > 0 ? (stats.wins / stats.games) * 100 : 0,
    }))
    .sort((a, b) => b.winRate - a.winRate)

  // Convert to MapStats array
  const mapStats: MapStats[] = Array.from(mapStatsMap.entries())
    .map(([map, stats]) => ({
      map,
      wins: stats.wins,
      losses: stats.losses,
      games: stats.games,
      winRate: stats.games > 0 ? (stats.wins / stats.games) * 100 : 0,
      heroes: Array.from(stats.heroes.entries()).map(([hero, hStats]) => ({
        hero,
        role: classifyHeroRole(hero),
        wins: hStats.wins,
        losses: hStats.losses,
        games: hStats.games,
        winRate: hStats.winRate,
      })),
    }))
    .sort((a, b) => b.winRate - a.winRate)

  // Calculate role stats
  const roleStats: Record<string, { wins: number; games: number; winRate: number }> = {}
  heroStats.forEach((hero) => {
    if (!roleStats[hero.role]) {
      roleStats[hero.role] = { wins: 0, games: 0, winRate: 0 }
    }
    roleStats[hero.role].wins += hero.wins
    roleStats[hero.role].games += hero.games
  })

  Object.keys(roleStats).forEach((role) => {
    const stats = roleStats[role]
    stats.winRate = stats.games > 0 ? (stats.wins / stats.games) * 100 : 0
  })

  // Calculate totals
  const totalGames = heroStats.reduce((sum, hero) => sum + hero.games, 0)
  const totalWins = heroStats.reduce((sum, hero) => sum + hero.wins, 0)
  const totalLosses = heroStats.reduce((sum, hero) => sum + hero.losses, 0)
  const overallWinRate = totalGames > 0 ? (totalWins / totalGames) * 100 : 0

  return {
    playerName: battletag,
    totalGames,
    totalWins,
    totalLosses,
    overallWinRate,
    heroStats,
    mapStats,
    roleStats,
  }
}

// Generate power picks (65%+ win rate with 5+ games)
export function generatePowerPicks(playerData: PlayerData): PowerPick[] {
  const powerPicks: PowerPick[] = []

  playerData.mapStats.forEach((mapStat) => {
    mapStat.heroes.forEach((hero) => {
      if (hero.winRate >= 65 && hero.games >= 5) {
        powerPicks.push({
          hero: hero.hero,
          role: hero.role,
          map: mapStat.map,
          winRate: hero.winRate,
          games: hero.games,
          wins: hero.wins,
          losses: hero.losses,
        })
      }
    })
  })

  return powerPicks.sort((a, b) => b.winRate - a.winRate)
}

// Generate insights
export function generateInsights(playerData: PlayerData): Insight[] {
  const insights: Insight[] = []

  // Best map
  if (playerData.mapStats.length > 0) {
    const bestMap = playerData.mapStats[0]
    insights.push({
      type: 'success',
      title: `Dominant on ${bestMap.map}`,
      description: `You have a ${bestMap.winRate.toFixed(1)}% win rate on this map with ${bestMap.games} games played.`,
      icon: '🗺️',
    })
  }

  // Best hero
  if (playerData.heroStats.length > 0) {
    const bestHero = playerData.heroStats[0]
    if (bestHero.games >= 20) {
      insights.push({
        type: 'tip',
        title: `${bestHero.hero} Mastery`,
        description: `${bestHero.games} games played with ${bestHero.winRate.toFixed(1)}% win rate - Your signature hero!`,
        icon: '🏆',
      })
    }
  }

  // Weak roles
  const roleEntries = Object.entries(playerData.roleStats)
    .filter(([_, stats]) => stats.games >= 50)
    .sort((a, b) => a[1].winRate - b[1].winRate)

  if (roleEntries.length > 0 && roleEntries[0][1].winRate < 45) {
    const [role, stats] = roleEntries[0]
    insights.push({
      type: 'warning',
      title: `Struggle with ${role}`,
      description: `Only ${stats.winRate.toFixed(1)}% win rate across ${stats.games} games.`,
      icon: '⚠️',
    })
  }

  // Overall consistency
  if (playerData.overallWinRate >= 50) {
    insights.push({
      type: 'info',
      title: 'Consistent Performance',
      description: `Maintained ${playerData.overallWinRate.toFixed(1)}% overall win rate across ${playerData.totalGames.toLocaleString()} games.`,
      icon: '✨',
    })
  }

  return insights
}

// Helper: Create unique membership key for party group
export function createMembershipKey(members: string[]): string {
  return [...members].sort().join('|')
}

// Aggregate party statistics from replay data
export function aggregatePartyStats(replays: ReplayData[]): PartyStats {
  // Maps for each party size
  const duoMap = new Map<string, PartyGroup>()
  const trioMap = new Map<string, PartyGroup>()
  const quadMap = new Map<string, PartyGroup>()
  const quintMap = new Map<string, PartyGroup>()

  // Iterate through replays
  replays.forEach((replay) => {
    // Skip solo games
    if (replay.partySize === 1) return

    // Create membership key
    const membershipKey = createMembershipKey(replay.partyMembers)

    // Select appropriate map based on party size
    let partyMap: Map<string, PartyGroup>
    switch (replay.partySize) {
      case 2:
        partyMap = duoMap
        break
      case 3:
        partyMap = trioMap
        break
      case 4:
        partyMap = quadMap
        break
      case 5:
        partyMap = quintMap
        break
      default:
        return // Invalid party size
    }

    // Get or create party group
    if (!partyMap.has(membershipKey)) {
      partyMap.set(membershipKey, {
        members: [...replay.partyMembers].sort(),
        displayNames: replay.partyMembers
          .map(tag => tag.split('#')[0])
          .sort((a, b) => {
            // Sort by name, but keep order consistent with members
            const aIdx = replay.partyMembers.findIndex(m => m.split('#')[0] === a)
            const bIdx = replay.partyMembers.findIndex(m => m.split('#')[0] === b)
            return replay.partyMembers[aIdx].localeCompare(replay.partyMembers[bIdx])
          }),
        partySize: replay.partySize,
        totalGames: 0,
        totalWins: 0,
        totalLosses: 0,
        winRate: 0,
        commonHeroes: [],
        bestMaps: [],
        membershipKey,
        memberHeroes: {},
        compositions: [],
      })
    }

    const group = partyMap.get(membershipKey)!

    // Update stats
    group.totalGames++
    if (replay.result === 'win') {
      group.totalWins++
    } else {
      group.totalLosses++
    }
  })

  // Calculate derived metrics and filter for each party size
  const processPartyGroups = (partyMap: Map<string, PartyGroup>): PartyGroup[] => {
    const groups = Array.from(partyMap.values())
      .filter(group => group.totalGames >= 2) // Minimum 2 games threshold
      .map(group => {
        // Calculate win rate
        group.winRate = group.totalGames > 0
          ? Math.round((group.totalWins / group.totalGames) * 1000) / 10
          : 0

        // Aggregate heroes played in this party
        const heroMap = new Map<string, { games: number; wins: number }>()
        replays.forEach(replay => {
          if (createMembershipKey(replay.partyMembers) === group.membershipKey) {
            if (!heroMap.has(replay.hero)) {
              heroMap.set(replay.hero, { games: 0, wins: 0 })
            }
            const heroStats = heroMap.get(replay.hero)!
            heroStats.games++
            if (replay.result === 'win') {
              heroStats.wins++
            }
          }
        })

        // Convert to common heroes array
        group.commonHeroes = Array.from(heroMap.entries())
          .map(([hero, stats]) => ({
            hero,
            games: stats.games,
            winRate: stats.games > 0
              ? Math.round((stats.wins / stats.games) * 1000) / 10
              : 0,
          }))
          .sort((a, b) => b.games - a.games)

        // Aggregate heroes for EACH party member
        const memberHeroMaps = new Map<string, Map<string, { games: number; wins: number }>>()
        replays.forEach(replay => {
          if (createMembershipKey(replay.partyMembers) === group.membershipKey && replay.partyMemberHeroes) {
            Object.entries(replay.partyMemberHeroes).forEach(([battletag, hero]) => {
              if (!memberHeroMaps.has(battletag)) {
                memberHeroMaps.set(battletag, new Map())
              }
              const memberHeroMap = memberHeroMaps.get(battletag)!
              if (!memberHeroMap.has(hero)) {
                memberHeroMap.set(hero, { games: 0, wins: 0 })
              }
              const stats = memberHeroMap.get(hero)!
              stats.games++
              if (replay.result === 'win') {
                stats.wins++
              }
            })
          }
        })

        // Convert to memberHeroes object
        group.memberHeroes = {}
        memberHeroMaps.forEach((heroMap, battletag) => {
          group.memberHeroes[battletag] = Array.from(heroMap.entries())
            .map(([hero, stats]) => ({
              hero,
              games: stats.games,
              winRate: stats.games > 0
                ? Math.round((stats.wins / stats.games) * 1000) / 10
                : 0,
            }))
            .sort((a, b) => {
              // Sort by win rate (best to worst), then by games played
              if (b.winRate !== a.winRate) return b.winRate - a.winRate
              return b.games - a.games
            })
            .slice(0, 5) // Top 5 heroes per member
        })

        // Aggregate team compositions
        const compositionMap = new Map<string, { games: number; wins: number; losses: number }>()
        replays.forEach(replay => {
          if (createMembershipKey(replay.partyMembers) === group.membershipKey && replay.partyMemberHeroes) {
            // Get all heroes in the party for this game
            const heroes = Object.values(replay.partyMemberHeroes).sort()
            const compositionKey = heroes.join(' + ')

            if (!compositionMap.has(compositionKey)) {
              compositionMap.set(compositionKey, { games: 0, wins: 0, losses: 0 })
            }
            const compStats = compositionMap.get(compositionKey)!
            compStats.games++
            if (replay.result === 'win') {
              compStats.wins++
            } else {
              compStats.losses++
            }
          }
        })

        // Convert to compositions array
        group.compositions = Array.from(compositionMap.entries())
          .map(([composition, stats]) => ({
            composition,
            games: stats.games,
            wins: stats.wins,
            losses: stats.losses,
            winRate: stats.games > 0
              ? Math.round((stats.wins / stats.games) * 1000) / 10
              : 0,
          }))
          .sort((a, b) => {
            // Sort by win rate (best to worst), then by games played
            if (b.winRate !== a.winRate) return b.winRate - a.winRate
            return b.games - a.games
          })

        // Aggregate maps played in this party
        const mapMap = new Map<string, { games: number; wins: number }>()
        replays.forEach(replay => {
          if (createMembershipKey(replay.partyMembers) === group.membershipKey) {
            if (!mapMap.has(replay.map)) {
              mapMap.set(replay.map, { games: 0, wins: 0 })
            }
            const mapStats = mapMap.get(replay.map)!
            mapStats.games++
            if (replay.result === 'win') {
              mapStats.wins++
            }
          }
        })

        // Convert to best maps array (show all, sorted by games then win rate)
        group.bestMaps = Array.from(mapMap.entries())
          .map(([map, stats]) => ({
            map,
            games: stats.games,
            winRate: stats.games > 0
              ? Math.round((stats.wins / stats.games) * 1000) / 10
              : 0,
          }))
          .sort((a, b) => {
            // Sort by games first, then by win rate
            if (b.games !== a.games) return b.games - a.games
            return b.winRate - a.winRate
          })

        return group
      })
      .sort((a, b) => b.totalGames - a.totalGames) // Sort by most games played

    return groups
  }

  return {
    duos: processPartyGroups(duoMap),
    trios: processPartyGroups(trioMap),
    quadruples: processPartyGroups(quadMap),
    quintuples: processPartyGroups(quintMap),
  }
}
