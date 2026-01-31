import { NextResponse } from 'next/server'

const API_BASE = 'https://api.heroesprofile.com'
const API_KEY = process.env.HEROES_PROFILE_API_KEY

export async function GET(
  request: Request,
  { params }: { params: { battletag: string } }
) {
  try {
    const { battletag } = params

    if (!battletag) {
      return NextResponse.json(
        { error: 'Battletag is required' },
        { status: 400 }
      )
    }

    // Build URL with correct parameters
    const apiParams = new URLSearchParams()
    apiParams.append('mode', 'json')
    apiParams.append('battletag', battletag)
    apiParams.append('region', '1')
    apiParams.append('game_type', 'Storm League')
    if (API_KEY) {
      apiParams.append('api_token', API_KEY)
    }
    apiParams.append('group_by_map', 'True')

    const url = `${API_BASE}/api/Player/Hero/All?${apiParams.toString()}`

    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
      },
    })

    if (!response.ok) {
      const errorText = await response.text()
      return NextResponse.json(
        { error: `Heroes Profile API error: ${response.status}`, details: errorText },
        { status: response.status }
      )
    }

    const data = await response.json()

    // Transform API response to our format
    const heroStats: Record<string, any> = {}

    if (data && data['Storm League']) {
      Object.entries(data['Storm League']).forEach(([heroName, stats]: [string, any]) => {
        if (typeof stats !== 'object' || !stats.wins) {
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
    }

    return NextResponse.json({
      battletag,
      heroStats,
    })
  } catch (error) {
    console.error('Failed to fetch player stats:', error)
    return NextResponse.json(
      { error: 'Failed to fetch player stats', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}
