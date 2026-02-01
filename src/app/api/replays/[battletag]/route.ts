import { NextResponse } from 'next/server'
import { ReplayData, ReplayResponse } from '@/types'
import { aggregatePartyStats } from '@/lib/data/transform'

const API_BASE = 'https://api.heroesprofile.com'
const API_KEY = process.env.HEROES_PROFILE_API_KEY

// Disable caching for this route
export const dynamic = 'force-dynamic'

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

    const url = `${API_BASE}/api/Player/Replays?${apiParams.toString()}`

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

    console.log('=== REPLAY API RESPONSE - Fetching detailed party data ===')
    console.log('Keys:', Object.keys(data))
    if (data['Storm League']) {
      const stormLeagueKeys = Object.keys(data['Storm League'])
      console.log('Total Storm League replays:', stormLeagueKeys.length)
      const firstReplayId = stormLeagueKeys[0]
      if (firstReplayId) {
        console.log('First replay sample:', JSON.stringify(data['Storm League'][firstReplayId], null, 2).substring(0, 500))
      }
    }

    // Transform replay data (now fetches detailed party information)
    const replays = await transformReplayData(data, battletag)

    console.log('=== PARTY DATA SUMMARY ===')
    console.log(`Total replays processed: ${replays.length}`)
    const partySizeBreakdown = replays.reduce((acc, r) => {
      acc[r.partySize] = (acc[r.partySize] || 0) + 1
      return acc
    }, {} as Record<number, number>)
    console.log('Party size breakdown:', partySizeBreakdown)

    // Show sample party data
    const sampleParty = replays.find(r => r.partySize > 1)
    if (sampleParty) {
      console.log('Sample party game:', {
        replayId: sampleParty.replayId,
        partySize: sampleParty.partySize,
        partyMembers: sampleParty.partyMembers,
        result: sampleParty.result,
      })
    }

    // Count solo vs party games
    const soloGames = replays.filter(r => r.partySize === 1).length
    const partyGames = replays.filter(r => r.partySize > 1).length

    // Aggregate party statistics
    const partyStats = aggregatePartyStats(replays)

    const result: ReplayResponse = {
      battletag,
      totalReplays: replays.length,
      soloGames,
      partyGames,
      partyStats,
      replays,
    }

    return NextResponse.json(result)
  } catch (error) {
    console.error('Failed to fetch replays:', error)
    return NextResponse.json(
      { error: 'Failed to fetch replays', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}

async function transformReplayData(rawData: any, playerBattletag: string): Promise<ReplayData[]> {
  const replays: ReplayData[] = []

  // Navigate through the nested structure
  // Expected format: { "Storm League": { replayId: { ...replayData } } }
  if (!rawData || typeof rawData !== 'object') {
    return replays
  }

  // Get Storm League data
  const stormLeagueData = rawData['Storm League']
  if (!stormLeagueData || typeof stormLeagueData !== 'object') {
    return replays
  }

  // Get replay IDs, filter for party games only, and sort by date (most recent first)
  const partyReplayEntries = Object.entries(stormLeagueData)
    .filter(([_, replayData]) => {
      // Only include replays where player was in a party (party field != 0)
      const partyId = parseInt(String((replayData as any).party)) || 0
      return partyId !== 0
    })
    .sort((a, b) => {
      const dateA = new Date((a[1] as any).game_date || 0).getTime()
      const dateB = new Date((b[1] as any).game_date || 0).getTime()
      return dateB - dateA // Most recent first
    })

  // Limit to 200 most recent party games
  const recentPartyReplays = partyReplayEntries.slice(0, 200)

  console.log(`Processing ${recentPartyReplays.length} party games out of ${Object.keys(stormLeagueData).length} total for player: ${playerBattletag}`)

  for (const [replayId, replayData] of recentPartyReplays) {
    if (typeof replayData !== 'object') continue

    // Type assertion for raw API data
    const rawReplay = replayData as any

    try {
      // Fetch full replay data to get all players
      const detailParams = new URLSearchParams()
      detailParams.append('replayID', replayId)
      detailParams.append('mode', 'json')
      if (API_KEY) {
        detailParams.append('api_token', API_KEY)
      }

      const detailUrl = `${API_BASE}/api/Replay/Data?${detailParams.toString()}`
      const detailResponse = await fetch(detailUrl, {
        headers: { 'Accept': 'application/json' },
      })

      if (!detailResponse.ok) {
        const errorText = await detailResponse.text()
        console.warn(`Failed to fetch detail for replay ${replayId}: ${detailResponse.status} - ${errorText.substring(0, 200)}`)
        continue
      }

      const detailData = await detailResponse.json()

      // The response is { replayId: { ...metadata, battletag: { playerData }, ... } }
      const replayFullData = detailData[replayId]
      if (!replayFullData) continue

      // Filter out metadata fields - player battletags contain '#' character
      const playerKeys = Object.keys(replayFullData).filter(key => key.includes('#'))
      const replayPlayers: Record<string, any> = {}
      playerKeys.forEach(key => {
        replayPlayers[key] = replayFullData[key]
      })

      // Find the player's party ID (try exact match first)
      let playerData = replayPlayers[playerBattletag]
      let actualBattletag = playerBattletag

      // If exact match not found, try case-insensitive search
      if (!playerData) {
        const lowerBattletag = playerBattletag.toLowerCase()
        const foundKey = Object.keys(replayPlayers).find(
          key => key.toLowerCase() === lowerBattletag
        )
        if (foundKey) {
          console.log(`Found case mismatch: requested "${playerBattletag}", found "${foundKey}"`)
          playerData = replayPlayers[foundKey]
          actualBattletag = foundKey
        }
      }

      if (!playerData) {
        // Log first replay to see what players are available
        if (replayId === recentReplayIds[0]) {
          console.log(`Player ${playerBattletag} not found in first replay ${replayId}. Available players:`, Object.keys(replayPlayers).slice(0, 5))
        }
        continue
      }

      const playerPartyId = playerData.party || 0

      // Extract party members (players with same party ID, excluding 0 for solo)
      let partyMembers: string[] = [actualBattletag]
      let partySize = 1

      if (playerPartyId !== 0) {
        // Find all players with the same party ID (use actual battletags from API)
        partyMembers = Object.keys(replayPlayers).filter(
          battletag => replayPlayers[battletag].party === playerPartyId
        )
        partySize = partyMembers.length
      }

      // Build a map of party members to their heroes
      const partyMemberHeroes: { [battletag: string]: string } = {}
      partyMembers.forEach(member => {
        if (replayPlayers[member]?.hero) {
          partyMemberHeroes[member] = replayPlayers[member].hero
        }
      })

      // Determine result from winner field (boolean)
      const result: 'win' | 'loss' = playerData.winner === true ? 'win' : 'loss'

      // Get map name from detailed data (game_map field)
      const mapName = replayFullData.game_map || rawReplay.map || 'Unknown'

      replays.push({
        replayId,
        gameType: 'Storm League',
        hero: rawReplay.hero || playerData.hero || 'Unknown',
        map: mapName,
        result,
        date: rawReplay.game_date || new Date().toISOString(),
        duration: parseInt(rawReplay.game_length) || 0,
        partyMembers: partyMembers.sort(), // Sort for consistent membership keys
        partySize,
        kills: parseInt(rawReplay.kills) || parseInt(playerData.kills) || 0,
        deaths: parseInt(rawReplay.deaths) || parseInt(playerData.deaths) || 0,
        assists: parseInt(rawReplay.assists) || parseInt(playerData.assists) || 0,
        partyMemberHeroes,
      })
    } catch (err) {
      console.error('Error parsing replay:', replayId, err)
    }
  }

  // Sort by date descending (most recent first)
  replays.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())

  return replays
}
