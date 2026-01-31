import { useState, useEffect } from 'react'
import { PlayerData } from '@/types'

interface Insight {
  type: 'success' | 'warning' | 'info'
  title: string
  description: string
}

interface UseAIInsightsOptions {
  autoFetch?: boolean
  onComplete?: (insights: Insight[]) => void
  onError?: (error: Error) => void
}

export function useAIInsights(
  playerData: PlayerData | null | undefined,
  options: UseAIInsightsOptions = {}
) {
  const { autoFetch = false } = options
  const [insights, setInsights] = useState<Insight[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (autoFetch && playerData) {
      fetchInsights(playerData)
    }
  }, [autoFetch, playerData?.playerName]) // Only depend on player name to avoid infinite loops

  const fetchInsights = async (data: PlayerData) => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/commentary/insights', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ playerData: data }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`API error: ${response.status} - ${errorText}`)
      }

      const result = await response.json()
      setInsights(result.insights || [])
      options.onComplete?.(result.insights || [])
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch insights'
      setError(errorMessage)
      options.onError?.(err instanceof Error ? err : new Error(errorMessage))
    } finally {
      setIsLoading(false)
    }
  }

  return {
    insights,
    isLoading,
    error,
    fetchInsights,
  }
}
