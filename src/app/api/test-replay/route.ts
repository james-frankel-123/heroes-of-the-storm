import { NextResponse } from 'next/server'

const API_BASE = 'https://api.heroesprofile.com'
const API_KEY = process.env.HEROES_PROFILE_API_KEY

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const replayId = '18871040'

    const detailParams = new URLSearchParams()
    detailParams.append('replayID', replayId)
    detailParams.append('mode', 'json')
    if (API_KEY) {
      detailParams.append('api_token', API_KEY)
    }

    const detailUrl = `${API_BASE}/api/Replay/Data?${detailParams.toString()}`

    console.log('Fetching:', detailUrl.replace(API_KEY || '', 'API_KEY'))

    const detailResponse = await fetch(detailUrl, {
      headers: { 'Accept': 'application/json' },
    })

    console.log('Status:', detailResponse.status)
    console.log('OK:', detailResponse.ok)

    if (!detailResponse.ok) {
      const errorText = await detailResponse.text()
      return NextResponse.json({
        error: `Failed: ${detailResponse.status}`,
        details: errorText.substring(0, 500)
      }, { status: 500 })
    }

    const data = await detailResponse.json()
    const replayPlayers = data[replayId]

    if (!replayPlayers) {
      return NextResponse.json({ error: 'No players found' }, { status: 500 })
    }

    // Extract party information
    const playerBattletag = 'IAmTheJames#1590'
    const playerData = replayPlayers[playerBattletag]
    const playerPartyId = playerData?.party || 0

    const partyMembers = Object.keys(replayPlayers).filter(
      battletag => replayPlayers[battletag].party === playerPartyId && playerPartyId !== 0
    )

    return NextResponse.json({
      success: true,
      replayId,
      playerPartyId,
      partySize: partyMembers.length,
      partyMembers,
      allPlayers: Object.keys(replayPlayers),
    })
  } catch (error) {
    console.error('Test error:', error)
    return NextResponse.json({
      error: 'Exception',
      details: error instanceof Error ? error.message : 'Unknown'
    }, { status: 500 })
  }
}
