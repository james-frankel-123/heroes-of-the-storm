'use client'

import useSWR from 'swr'
import { PlayerData } from '@/types'
import { usePlayer } from '@/contexts/player-context'

const fetcher = async (url: string) => {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error('Failed to fetch data')
  }
  return response.json()
}

export function usePlayerData(playerName?: string) {
  const { battletag } = usePlayer()

  // Use provided playerName or context battletag
  const targetBattletag = playerName || battletag

  const { data, error, isLoading } = useSWR<Record<string, PlayerData>>(
    targetBattletag ? `/api/data?battletag=${encodeURIComponent(targetBattletag)}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    }
  )

  const players = data ? Object.keys(data) : []
  const currentPlayer = targetBattletag || players[0]
  const playerData = currentPlayer ? data?.[currentPlayer] : null

  return {
    data: playerData,
    allPlayers: players,
    isLoading,
    error,
  }
}
