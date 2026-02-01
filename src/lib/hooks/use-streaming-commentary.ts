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
  const requestIdRef = useRef(0)

  const reset = useCallback(() => {
    console.log('🔄 Resetting commentary hook')
    setCommentary('')
    setError(null)
    setIsStreaming(false)
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    // Increment request ID to invalidate any in-flight requests
    requestIdRef.current += 1
  }, [])

  const fetchCommentary = useCallback(
    async (endpoint: string, payload: any) => {
      // Abort any existing request
      if (abortControllerRef.current) {
        console.log('🛑 Aborting previous request')
        abortControllerRef.current.abort()
      }

      // Increment request ID for this new request
      requestIdRef.current += 1
      const currentRequestId = requestIdRef.current

      console.log('🚀 Starting new request, ID:', currentRequestId)

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
        let chunkCount = 0

        // Stream the response (already formatted from server)
        for await (const chunk of parseStreamingResponse(response)) {
          // Check if this request has been superseded
          if (currentRequestId !== requestIdRef.current) {
            console.log('⚠️ Request', currentRequestId, 'superseded by', requestIdRef.current, '- ignoring chunk')
            return // Stop processing this outdated request
          }

          chunkCount++
          const previousLength = fullCommentary.length
          fullCommentary += chunk

          console.log('=== Chunk Received ===')
          console.log('Request ID:', currentRequestId)
          console.log('Chunk #:', chunkCount)
          console.log('Chunk length:', chunk.length)
          console.log('Chunk content:', JSON.stringify(chunk))
          console.log('Previous total length:', previousLength)
          console.log('New total length:', fullCommentary.length)
          console.log('Length change:', fullCommentary.length - previousLength)

          setCommentary(fullCommentary)
        }

        // Check one final time before marking complete
        if (currentRequestId !== requestIdRef.current) {
          console.log('⚠️ Request', currentRequestId, 'superseded - not marking complete')
          return
        }

        // Streaming complete
        console.log('✅ Request', currentRequestId, 'completed successfully')
        setIsStreaming(false)
        options.onComplete?.(fullCommentary)
      } catch (err) {
        // Only handle errors for the current request
        if (currentRequestId !== requestIdRef.current) {
          console.log('⚠️ Request', currentRequestId, 'superseded - ignoring error')
          return
        }

        if (err instanceof Error) {
          if (err.name === 'AbortError') {
            // Request was cancelled
            console.log('❌ Request', currentRequestId, 'was aborted')
            setError('Request cancelled')
          } else {
            console.log('❌ Request', currentRequestId, 'failed:', err.message)
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
