import { NextResponse } from 'next/server'
import { transformHeroStatsData, transformHeroGroupedData } from '@/lib/data/transform'

const API_BASE = 'https://api.heroesprofile.com'
const API_KEY = process.env.HEROES_PROFILE_API_KEY

export async function GET(request: Request) {
  try {
    // Get battletag from query params
    const { searchParams } = new URL(request.url)
    const battletag = searchParams.get('battletag')

    if (!battletag) {
      return NextResponse.json(
        { error: 'Battletag is required' },
        { status: 400 }
      )
    }

    console.log(`\n========== Fetching data for ${battletag} ==========`)

    // Try with group_by_map=True first
    const apiParams = new URLSearchParams()
    apiParams.append('mode', 'json')
    apiParams.append('battletag', battletag)
    apiParams.append('region', '1')
    apiParams.append('game_type', 'Storm League')
    apiParams.append('group_by_map', 'true')
    if (API_KEY) {
      apiParams.append('api_token', API_KEY)
    }

    const url = `${API_BASE}/api/Player/Hero/All?${apiParams.toString()}`
    console.log('API URL:', url.replace(API_KEY || '', 'API_KEY_HIDDEN'))

    const response = await fetch(url, {
      headers: { 'Accept': 'application/json' },
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    const data = await response.json()

    // Log the structure
    console.log('\n=== API Response Structure ===')
    console.log('Root keys:', Object.keys(data))

    if (data['Storm League']) {
      const slData = data['Storm League']
      const topLevelKeys = Object.keys(slData).slice(0, 5)
      console.log('Storm League top level keys (first 5):', topLevelKeys)

      // Check the structure of the first entry
      const firstKey = Object.keys(slData)[0]
      const firstValue = slData[firstKey]
      console.log(`\nFirst entry: "${firstKey}"`)
      console.log('First entry type:', typeof firstValue)

      if (typeof firstValue === 'object' && firstValue !== null) {
        const innerKeys = Object.keys(firstValue)
        console.log('First entry has these keys:', innerKeys.slice(0, 10))

        // Check if it's map-grouped (innerKeys should be hero names)
        // or hero-grouped (innerKeys should be stats like 'wins', 'losses')
        const hasStatsFields = innerKeys.some(k => ['wins', 'losses', 'games_played', 'win_rate'].includes(k))
        const secondLevelKeys = !hasStatsFields && innerKeys.length > 0 ? Object.keys(firstValue[innerKeys[0]]) : []

        const isHeroGrouped = hasStatsFields
        console.log('Structure type:', isHeroGrouped ? 'HERO-GROUPED (hero → stats)' : 'MAP-GROUPED (map → hero → stats)')
        if (!hasStatsFields && secondLevelKeys.length > 0) {
          console.log('Second level keys (should be stats):', secondLevelKeys.slice(0, 5))
        }

        // Transform the data using the appropriate function
        const playerData = isHeroGrouped
          ? transformHeroGroupedData(data, battletag)
          : transformHeroStatsData(data, battletag)

        console.log('\n=== Transform Result ===')
        console.log('Total games:', playerData.totalGames)
        console.log('Total heroes:', playerData.heroStats.length)
        console.log('Total maps:', playerData.mapStats.length)
        console.log('========================================\n')

        return NextResponse.json({
          [battletag]: playerData,
        })
      }
    }

    // Fallback: assume hero-grouped if we couldn't determine structure
    const playerData = transformHeroGroupedData(data, battletag)

    console.log('\n=== Transform Result (Fallback) ===')
    console.log('Total games:', playerData.totalGames)
    console.log('Total heroes:', playerData.heroStats.length)
    console.log('========================================\n')

    return NextResponse.json({
      [battletag]: playerData,
    })

  } catch (error) {
    console.error('Error fetching player data:', error)
    return NextResponse.json(
      {
        error: 'Failed to fetch player data',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
}
