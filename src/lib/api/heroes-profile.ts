// Heroes Profile API integration
const API_BASE = 'https://api.heroesprofile.com'
const API_KEY = 'SgyMDSbIpkC5ytO4BKlgOLPgDP8hcrhnXip2u3xOyWwHZXI2PcxRiYbomagi'

export interface PlayerStats {
  battletag: string
  heroStats: Record<string, {
    hero: string
    wins: number
    losses: number
    games: number
    winRate: number
  }>
  loading?: boolean
  error?: string
}

export async function fetchPlayerHeroStats(battletag: string): Promise<PlayerStats> {
  try {
    // Build URL with correct parameters (matching working Python code)
    // URLSearchParams will handle encoding the battletag
    const params = new URLSearchParams({
      mode: 'json',
      battletag: battletag, // Don't double-encode - URLSearchParams handles it
      region: '1',
      game_type: 'Storm League',
      api_token: API_KEY,
      group_by_map: 'True'
    })

    const url = `${API_BASE}/api/Player/Hero/All?${params.toString()}`

    console.log('Fetching player stats for:', battletag)
    console.log('API URL:', url)

    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
      },
    })

    console.log('API Response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('API error response:', errorText)
      throw new Error(`API error: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    console.log('API data received:', data)

    // Transform API response to our format
    // The API returns data structured as: { "Storm League": { "heroName": { wins, losses, games_played, win_rate } } }
    const heroStats: Record<string, any> = {}

    if (data && data['Storm League']) {
      console.log('Storm League heroes found:', Object.keys(data['Storm League']).length)

      // Parse heroes directly (not grouped by map despite the parameter)
      Object.entries(data['Storm League']).forEach(([heroName, stats]: [string, any]) => {
        // Skip if this doesn't look like hero stats
        if (typeof stats !== 'object' || !stats.wins) {
          console.warn(`Skipping invalid entry: ${heroName}`, stats)
          return
        }

        const games = stats.games_played || 0
        const wins = stats.wins || 0
        const winRate = games > 0 ? (wins / games) * 100 : 0

        heroStats[heroName] = {
          hero: heroName,
          wins: wins,
          losses: stats.losses || 0,
          games: games,
          winRate: Math.round(winRate * 10) / 10
        }
      })

      console.log('Sample heroes loaded:', Object.keys(heroStats).slice(0, 5))
      if (Object.keys(heroStats).length > 0) {
        const firstHero = Object.keys(heroStats)[0]
        console.log(`${firstHero} stats:`, heroStats[firstHero])
      }
    } else {
      console.error('No Storm League data found in response:', data)
    }

    console.log(`Loaded ${Object.keys(heroStats).length} heroes for ${battletag}`)

    return {
      battletag,
      heroStats,
    }
  } catch (error) {
    console.error('Failed to fetch player stats:', error)
    return {
      battletag,
      heroStats: {},
      error: error instanceof Error ? error.message : 'Failed to fetch stats',
    }
  }
}
