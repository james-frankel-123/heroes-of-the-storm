import { ReplayResponse } from '@/types'

export async function fetchPlayerReplays(battletag: string, includeAllGames: boolean = false): Promise<ReplayResponse> {
  try {
    const url = `/api/replays/${encodeURIComponent(battletag)}${includeAllGames ? '?includeAllGames=true' : ''}`
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`Failed to fetch replays: ${response.status}`)
    }
    return await response.json()
  } catch (error) {
    console.error('Error fetching replays:', error)
    return {
      battletag,
      totalReplays: 0,
      soloGames: 0,
      partyGames: 0,
      partyStats: {
        duos: [],
        trios: [],
        quadruples: [],
        quintuples: [],
      },
      replays: [],
    }
  }
}
