'use client'

import { useMemo } from 'react'
import { usePlayerData } from './use-data'
import { useReplays } from './use-replays'
import { generateStatistics, StatisticsSummary } from '@/lib/data/statistics'

export interface UseStatisticsDataReturn {
  statistics: StatisticsSummary | null
  isLoading: boolean
  error: Error | null
}

/**
 * Hook to fetch and compute comprehensive statistics
 */
export function useStatisticsData(): UseStatisticsDataReturn {
  // Fetch player data
  const { data: playerData, isLoading: isLoadingPlayer, error: playerError } = usePlayerData()

  // Fetch replay data (including all games, not just party games)
  const { data: replayResponse, isLoading: isLoadingReplays, error: replayError } = useReplays(
    playerData?.playerName,
    true // Include all games for comprehensive statistics
  )

  // Generate statistics from the data
  const statistics = useMemo(() => {
    if (!playerData || !replayResponse) {
      return null
    }

    // Generate comprehensive statistics
    return generateStatistics(replayResponse.replays, playerData)
  }, [playerData, replayResponse])

  // Combine loading and error states
  const isLoading = isLoadingPlayer || isLoadingReplays
  const error = playerError || replayError

  return {
    statistics,
    isLoading,
    error,
  }
}
