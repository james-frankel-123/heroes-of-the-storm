import { useState, useEffect, useRef } from 'react'
import { DraftTurn, DraftTeam } from '@/lib/draft/draft-sequence'
import { PartyMember } from '@/components/draft/draft-config-modal'

export interface DraftState {
  selectedMap: string
  yourTeam: DraftTeam
  currentTurn: DraftTurn
  bluePicks: (string | null)[]
  redPicks: (string | null)[]
  blueBans: string[]
  redBans: string[]
}

export interface HeroRecommendation {
  hero: string
  player: string
  slot: number
  winRate: number
  games: number
  reason: string
}

export interface QuickAnalysis {
  yourComp: string
  enemyComp: string
  roleNeed: string
}

export interface CoachCommentaryResult {
  commentary: string
  isStreaming: boolean
  error: string | null
  recommendations: HeroRecommendation[]
  quickAnalysis: QuickAnalysis | null
}

export function useCoachCommentary(
  draftState: DraftState,
  partyRoster: PartyMember[],
  draftHistory: Array<{ turn: DraftTurn; hero: string; timestamp: number; battletag?: string }>,
  autoFetch: boolean = true
): CoachCommentaryResult {
  const [commentary, setCommentary] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [recommendations, setRecommendations] = useState<HeroRecommendation[]>([])
  const [quickAnalysis, setQuickAnalysis] = useState<QuickAnalysis | null>(null)

  const abortControllerRef = useRef<AbortController | null>(null)
  const lastFetchRef = useRef<string>('')

  useEffect(() => {
    if (!autoFetch || draftHistory.length === 0) {
      return
    }

    // Generate a key for this draft state to avoid duplicate fetches
    const stateKey = JSON.stringify({
      historyLength: draftHistory.length,
      currentTurnIndex: draftState.currentTurn.turnIndex
    })

    if (stateKey === lastFetchRef.current) {
      return
    }

    lastFetchRef.current = stateKey

    // Fetch commentary
    fetchCommentary()

    return () => {
      // Cleanup: abort ongoing fetch
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [draftHistory.length, draftState.currentTurn.turnIndex])

  const fetchCommentary = async () => {
    // Abort previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    abortControllerRef.current = new AbortController()

    setIsStreaming(true)
    setError(null)
    setCommentary('')
    setRecommendations([])
    setQuickAnalysis(null)

    try {
      const response = await fetch('/api/draft/coach', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          draftState,
          partyRoster,
          draftHistory,
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }

      const decoder = new TextDecoder()
      let accumulated = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)

            if (data === '[DONE]') {
              setIsStreaming(false)
              break
            }

            // Check for special JSON blocks
            try {
              const parsed = JSON.parse(data)
              if (parsed.type === 'recommendations') {
                setRecommendations(parsed.data)
              } else if (parsed.type === 'quickAnalysis') {
                setQuickAnalysis(parsed.data)
              }
            } catch {
              // Not JSON, it's commentary text
              accumulated += data
              setCommentary(accumulated)
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        // Ignore abort errors
        return
      }
      console.error('Coach commentary error:', err)
      setError(err.message || 'Failed to fetch commentary')
    } finally {
      setIsStreaming(false)
    }
  }

  return {
    commentary,
    isStreaming,
    error,
    recommendations,
    quickAnalysis,
  }
}
