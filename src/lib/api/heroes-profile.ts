// Heroes Profile API integration via server-side proxy
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
    console.log('Fetching player stats for:', battletag)

    // Call our server-side API route instead of Heroes Profile directly
    const response = await fetch(`/api/heroes-profile/${encodeURIComponent(battletag)}`)

    console.log('API Response status:', response.status)

    if (!response.ok) {
      const errorData = await response.json()
      console.error('API error response:', errorData)
      throw new Error(`API error: ${response.status} - ${errorData.error || 'Unknown error'}`)
    }

    const data = await response.json()
    console.log(`Loaded ${Object.keys(data.heroStats).length} heroes for ${battletag}`)

    return data
  } catch (error) {
    console.error('Failed to fetch player stats:', error)
    return {
      battletag,
      heroStats: {},
      error: error instanceof Error ? error.message : 'Failed to fetch stats',
    }
  }
}
