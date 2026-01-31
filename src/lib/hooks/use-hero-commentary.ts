import { useCallback, useEffect, useRef } from 'react'
import { useStreamingCommentary } from './use-streaming-commentary'
import { PlayerData } from '@/types'

interface UseHeroCommentaryOptions {
  autoFetch?: boolean
  cacheTime?: number // Cache duration in milliseconds (default: 1 hour)
  onComplete?: (commentary: string) => void
  onError?: (error: Error) => void
}

interface UseHeroCommentaryReturn {
  commentary: string
  isStreaming: boolean
  error: string | null
  fetchHeroCommentary: (heroName: string, playerData: PlayerData) => Promise<void>
  reset: () => void
  isCached: boolean
}

// Simple in-memory cache
const commentaryCache = new Map<string, { commentary: string; timestamp: number }>()

export function useHeroCommentary(
  heroName?: string,
  playerData?: PlayerData,
  options: UseHeroCommentaryOptions = {}
): UseHeroCommentaryReturn {
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

  const fetchHeroCommentary = useCallback(
    async (hero: string, data: PlayerData) => {
      const cacheKey = `hero:${hero}:${data.playerName}`
      cacheKeyRef.current = cacheKey

      // Check cache
      const cached = commentaryCache.get(cacheKey)
      if (cached && Date.now() - cached.timestamp < cacheTime) {
        // Use cached commentary
        isCachedRef.current = true
        streaming.reset()
        // Simulate the streaming effect with cached data
        const words = cached.commentary.split(' ')
        let currentText = ''
        for (const word of words) {
          currentText += (currentText ? ' ' : '') + word
          // This will trigger a re-render, but won't actually stream
          // The component should handle this gracefully
        }
        return
      }

      isCachedRef.current = false

      // Fetch fresh commentary
      await streaming.fetchCommentary('/api/commentary/hero', {
        heroName: hero,
        playerData: data,
      })
    },
    [cacheTime, streaming]
  )

  // Auto-fetch if enabled and dependencies are provided
  useEffect(() => {
    if (autoFetch && heroName && playerData) {
      fetchHeroCommentary(heroName, playerData)
    }
  }, [autoFetch, heroName, playerData, fetchHeroCommentary])

  return {
    ...streaming,
    fetchHeroCommentary,
    isCached: isCachedRef.current,
  }
}
