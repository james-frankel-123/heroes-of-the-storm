import { useCallback, useEffect, useRef } from 'react'
import { useStreamingCommentary } from './use-streaming-commentary'
import { PlayerData } from '@/types'

interface UseMapCommentaryOptions {
  autoFetch?: boolean
  cacheTime?: number // Cache duration in milliseconds (default: 1 hour)
  onComplete?: (commentary: string) => void
  onError?: (error: Error) => void
}

interface UseMapCommentaryReturn {
  commentary: string
  isStreaming: boolean
  error: string | null
  fetchMapCommentary: (mapName: string, playerData: PlayerData) => Promise<void>
  reset: () => void
  isCached: boolean
}

// Simple in-memory cache
const commentaryCache = new Map<string, { commentary: string; timestamp: number }>()

export function useMapCommentary(
  mapName?: string,
  playerData?: PlayerData,
  options: UseMapCommentaryOptions = {}
): UseMapCommentaryReturn {
  const { cacheTime = 3600000, autoFetch = false, ...streamOptions } = options // Default 1 hour cache
  const cacheKeyRef = useRef<string>('')
  const isCachedRef = useRef(false)

  const streaming = useStreamingCommentary({
    ...streamOptions,
    onComplete: (commentary) => {
      // Cache the result
      if (cacheKeyRef.current) {
        commentaryCache.set(cacheKeyRef.current, {
          commentary,
          timestamp: Date.now(),
        })
      }
      streamOptions.onComplete?.(commentary)
    },
  })

  const fetchMapCommentary = useCallback(
    async (map: string, data: PlayerData) => {
      const cacheKey = `map:${map}:${data.playerName}`
      cacheKeyRef.current = cacheKey

      // Check cache
      const cached = commentaryCache.get(cacheKey)
      if (cached && Date.now() - cached.timestamp < cacheTime) {
        // Use cached commentary
        isCachedRef.current = true
        streaming.reset()
        return
      }

      isCachedRef.current = false

      // Fetch fresh commentary
      await streaming.fetchCommentary('/api/commentary/map', {
        mapName: map,
        playerData: data,
      })
    },
    [cacheTime, streaming]
  )

  // Auto-fetch if enabled and dependencies are provided
  useEffect(() => {
    if (autoFetch && mapName && playerData) {
      fetchMapCommentary(mapName, playerData)
    }
  }, [autoFetch, mapName, playerData, fetchMapCommentary])

  return {
    ...streaming,
    fetchMapCommentary,
    isCached: isCachedRef.current,
  }
}
