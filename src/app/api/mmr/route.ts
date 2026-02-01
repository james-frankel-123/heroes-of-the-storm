import { NextResponse } from 'next/server'

const API_BASE = 'https://api.heroesprofile.com'
const API_KEY = process.env.HEROES_PROFILE_API_KEY

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const battletag = searchParams.get('battletag')

    if (!battletag) {
      return NextResponse.json(
        { error: 'Battletag is required' },
        { status: 400 }
      )
    }

    const apiParams = new URLSearchParams()
    apiParams.append('mode', 'json')
    apiParams.append('battletag', battletag)
    apiParams.append('region', '1')
    apiParams.append('game_type', 'Storm League')
    if (API_KEY) {
      apiParams.append('api_token', API_KEY)
    }

    const url = `${API_BASE}/api/Player/MMR?${apiParams.toString()}`

    const response = await fetch(url, {
      headers: { 'Accept': 'application/json' },
    })

    if (!response.ok) {
      console.error(`MMR API error: ${response.status}`)
      return NextResponse.json(
        { mmr: null, league_tier: null },
        { status: 200 }
      )
    }

    const data = await response.json()

    // Extract MMR data - try case-insensitive battletag matching
    let mmrData = data[battletag]?.['Storm League']

    if (!mmrData) {
      // Try case-insensitive match
      const lowerBattletag = battletag.toLowerCase()
      const matchedKey = Object.keys(data).find(
        key => key.toLowerCase() === lowerBattletag
      )
      if (matchedKey) {
        mmrData = data[matchedKey]?.['Storm League']
      }
    }

    return NextResponse.json({
      mmr: mmrData?.mmr || null,
      league_tier: mmrData?.league_tier || null,
      games_played: mmrData?.games_played || 0,
    })
  } catch (error) {
    console.error('Error fetching MMR data:', error)
    return NextResponse.json(
      { mmr: null, league_tier: null },
      { status: 200 }
    )
  }
}
