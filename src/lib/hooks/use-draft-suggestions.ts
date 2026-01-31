import { useCallback, useEffect, useRef } from 'react'
import { useStreamingCommentary } from './use-streaming-commentary'
import { PlayerData } from '@/types'

interface DraftState {
  yourTeam: string[]
  enemyTeam: string[]
  bannedHeroes: string[]
  selectedMap: string
  activeSlot?: number
}

interface UseDraftSuggestionsOptions {
  autoFetch?: boolean
  debounceMs?: number // Debounce time before fetching (default: 1000ms)
  onComplete?: (suggestions: string) => void
  onError?: (error: Error) => void
}

interface UseDraftSuggestionsReturn {
  suggestions: string
  isStreaming: boolean
  error: string | null
  fetchSuggestions: (draftState: DraftState, playerData: PlayerData) => Promise<void>
  reset: () => void
}

export function useDraftSuggestions(
  draftState?: DraftState,
  playerData?: PlayerData,
  options: UseDraftSuggestionsOptions = {}
): UseDraftSuggestionsReturn {
  const { debounceMs = 1000, autoFetch = false, ...streamOptions } = options
  const debounceTimerRef = useRef<NodeJS.Timeout>()
  const lastFetchRef = useRef<string>('')

  const streaming = useStreamingCommentary(streamOptions)

  const fetchSuggestions = useCallback(
    async (state: DraftState, data: PlayerData) => {
      // Create a cache key from the draft state
      const cacheKey = JSON.stringify({
        yourTeam: state.yourTeam.filter(h => h && h !== 'Flexible'),
        enemyTeam: state.enemyTeam.filter(h => h),
        bannedHeroes: state.bannedHeroes.filter(h => h),
        map: state.selectedMap,
      })

      // Skip if same as last fetch (avoid duplicate requests)
      if (cacheKey === lastFetchRef.current) {
        return
      }

      lastFetchRef.current = cacheKey

      // Fetch suggestions
      await streaming.fetchCommentary('/api/draft/suggest', {
        yourTeam: state.yourTeam,
        enemyTeam: state.enemyTeam,
        bannedHeroes: state.bannedHeroes,
        selectedMap: state.selectedMap,
        activeSlot: state.activeSlot,
        playerData: data,
      })
    },
    [streaming]
  )

  const debouncedFetchSuggestions = useCallback(
    (state: DraftState, data: PlayerData) => {
      // Clear existing timer
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }

      // Set new timer
      debounceTimerRef.current = setTimeout(() => {
        fetchSuggestions(state, data)
      }, debounceMs)
    },
    [fetchSuggestions, debounceMs]
  )

  // Auto-fetch if enabled and dependencies change
  useEffect(() => {
    if (autoFetch && draftState && playerData) {
      // Only fetch if we have actual draft changes
      const hasData =
        draftState.yourTeam.some(h => h && h !== 'Flexible') ||
        draftState.enemyTeam.some(h => h) ||
        draftState.bannedHeroes.some(h => h) ||
        draftState.selectedMap

      if (hasData) {
        debouncedFetchSuggestions(draftState, playerData)
      }
    }

    // Cleanup on unmount
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    autoFetch,
    // Use stable primitive values instead of object references
    draftState?.yourTeam.join(','),
    draftState?.enemyTeam.join(','),
    draftState?.bannedHeroes.join(','),
    draftState?.selectedMap,
    playerData?.playerName,
  ])

  return {
    suggestions: streaming.commentary,
    isStreaming: streaming.isStreaming,
    error: streaming.error,
    fetchSuggestions,
    reset: streaming.reset,
  }
}
