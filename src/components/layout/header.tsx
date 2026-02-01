'use client'

import * as React from 'react'
import useSWR from 'swr'
import { usePlayer } from '@/contexts/player-context'
import { formatLeagueTier } from '@/lib/utils/rank'
import { PlayerMMR } from '@/types'

const fetcher = async (url: string) => {
  const response = await fetch(url)
  if (!response.ok) {
    return { mmr: null, league_tier: null, games_played: 0 }
  }
  return response.json()
}

export function Header() {
  const { battletag } = usePlayer()

  // Fetch MMR data
  const { data: mmrData } = useSWR<PlayerMMR>(
    battletag ? `/api/mmr?battletag=${encodeURIComponent(battletag)}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    }
  )

  // Extract name without discriminator for display
  const displayName = battletag?.split('#')[0] || 'Player'

  // Get formatted rank from league tier
  const rank = formatLeagueTier(mmrData?.league_tier || null)

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center justify-end px-6">
        {/* Player Profile */}
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-3 py-2">
          <div className="h-8 w-8 rounded-full bg-gradient-to-br from-primary-600 to-primary-500"></div>
          <div className="text-sm">
            <p className="font-medium">{displayName}</p>
            <p className="text-xs text-muted-foreground">{rank}</p>
          </div>
        </div>
      </div>
    </header>
  )
}
