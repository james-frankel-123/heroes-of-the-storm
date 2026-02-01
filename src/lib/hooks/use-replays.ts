import { useState, useEffect } from 'react'
import { fetchPlayerReplays } from '@/lib/api/replays'
import { ReplayResponse } from '@/types'

export function useReplays(battletag?: string, includeAllGames: boolean = false) {
  const [data, setData] = useState<ReplayResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!battletag) {
      setIsLoading(false)
      return
    }

    setIsLoading(true)
    fetchPlayerReplays(battletag, includeAllGames)
      .then(setData)
      .catch(err => setError(err.message))
      .finally(() => setIsLoading(false))
  }, [battletag, includeAllGames])

  return { data, isLoading, error }
}
