'use client'

import useSWR from 'swr'
import { PlayerData } from '@/types'

const fetcher = async (url: string) => {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error('Failed to fetch data')
  }
  return response.json()
}

export function usePlayerData(playerName?: string) {
  const { data, error, isLoading } = useSWR<Record<string, PlayerData>>(
    '/api/data',
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    }
  )

  const players = data ? Object.keys(data) : []
  const currentPlayer = playerName || players[0]
  const playerData = currentPlayer ? data?.[currentPlayer] : null

  return {
    data: playerData,
    allPlayers: players,
    isLoading,
    error,
  }
}
