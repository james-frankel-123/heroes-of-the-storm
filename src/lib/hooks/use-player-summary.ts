import { useEffect } from 'react'
import { useStreamingCommentary } from './use-streaming-commentary'
import { PlayerData } from '@/types'

interface UsePlayerSummaryOptions {
  autoFetch?: boolean
  onComplete?: (summary: string) => void
  onError?: (error: Error) => void
}

export function usePlayerSummary(
  playerData: PlayerData | null | undefined,
  options: UsePlayerSummaryOptions = {}
) {
  const { autoFetch = false } = options

  const {
    commentary: summary,
    isStreaming,
    error,
    fetchCommentary,
    reset,
  } = useStreamingCommentary({
    onComplete: options.onComplete,
    onError: options.onError,
  })

  useEffect(() => {
    if (autoFetch && playerData) {
      fetchCommentary('/api/commentary/player-summary', { playerData })
    }
  }, [autoFetch, playerData?.playerName]) // Only depend on player name to avoid infinite loops

  return {
    summary,
    isStreaming,
    error,
    fetchSummary: (data: PlayerData) =>
      fetchCommentary('/api/commentary/player-summary', { playerData: data }),
    reset,
  }
}
