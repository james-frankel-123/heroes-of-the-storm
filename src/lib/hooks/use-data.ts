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

  // Find player data with case-insensitive matching
  let playerData: PlayerData | null = null
  if (data && targetBattletag) {
    // Try exact match first
    playerData = data[targetBattletag] || null

    // If not found, try case-insensitive match
    if (!playerData) {
      const lowerTarget = targetBattletag.toLowerCase()
      const matchedKey = Object.keys(data).find(
        key => key.toLowerCase() === lowerTarget
      )
      if (matchedKey) {
        playerData = data[matchedKey]
        console.log(`Case mismatch: requested "${targetBattletag}", found "${matchedKey}"`)
      }
    }
  }

  return {
    data: playerData,
    allPlayers: players,
    isLoading,
    error,
  }
}
