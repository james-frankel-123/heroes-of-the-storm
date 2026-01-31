import { useState, useCallback, useRef } from 'react'
import { parseStreamingResponse } from '@/lib/utils/commentary'

interface UseStreamingCommentaryOptions {
  onComplete?: (commentary: string) => void
  onError?: (error: Error) => void
}

interface UseStreamingCommentaryReturn {
  commentary: string
  isStreaming: boolean
  error: string | null
  fetchCommentary: (endpoint: string, payload: any) => Promise<void>
  reset: () => void
}

export function useStreamingCommentary(
  options: UseStreamingCommentaryOptions = {}
): UseStreamingCommentaryReturn {
  const [commentary, setCommentary] = useState<string>('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    setCommentary('')
    setError(null)
    setIsStreaming(false)
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
  }, [])

  const fetchCommentary = useCallback(
    async (endpoint: string, payload: any) => {
      // Reset state
      setCommentary('')
      setError(null)
      setIsStreaming(true)

      // Create new abort controller
      abortControllerRef.current = new AbortController()

      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
          signal: abortControllerRef.current.signal,
        })

        if (!response.ok) {
          const errorText = await response.text()
          throw new Error(`API error: ${response.status} - ${errorText}`)
        }

        let fullCommentary = ''

        // Stream the response
        for await (const chunk of parseStreamingResponse(response)) {
          fullCommentary += chunk
          setCommentary(fullCommentary)
        }

        // Streaming complete
        setIsStreaming(false)
        options.onComplete?.(fullCommentary)
      } catch (err) {
        if (err instanceof Error) {
          if (err.name === 'AbortError') {
            // Request was cancelled
            setError('Request cancelled')
          } else {
            setError(err.message)
            options.onError?.(err)
          }
        } else {
          setError('An unknown error occurred')
        }
        setIsStreaming(false)
      }
    },
    [options]
  )

  return {
    commentary,
    isStreaming,
    error,
    fetchCommentary,
    reset,
  }
}
