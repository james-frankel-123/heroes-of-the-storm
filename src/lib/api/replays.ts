import { ReplayResponse } from '@/types'

export async function fetchPlayerReplays(battletag: string): Promise<ReplayResponse> {
  try {
    const response = await fetch(`/api/replays/${encodeURIComponent(battletag)}`)
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
