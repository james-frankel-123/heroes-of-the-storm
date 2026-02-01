'use client'

import { useState, useCallback } from 'react'
import { PlayerData } from '@/types'
import { ContextData } from '@/components/stats/clickable-metric'

interface Message {
  question: string
  contexts: (ContextData & { id: number })[]
  answer: string
}

export function useStatisticsAnalysis() {
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentResponse, setCurrentResponse] = useState('')
  const [conversationHistory, setConversationHistory] = useState<Message[]>([])
  const [contextCards, setContextCards] = useState<(ContextData & { id: number })[]>([])

  // Add context card when user clicks an element
  const addContextCard = useCallback((context: ContextData) => {
    setContextCards((prev) => [...prev, { id: Date.now(), ...context }])
  }, [])

  // Remove a context card
  const removeContextCard = useCallback((id: number) => {
    setContextCards((prev) => prev.filter((card) => card.id !== id))
  }, [])

  // Send question with current context cards
  const sendQuestion = useCallback(
    async (userQuestion: string, playerData: PlayerData | null) => {
      if (!userQuestion.trim() || contextCards.length === 0 || !playerData) {
        return
      }

      setIsStreaming(true)
      setCurrentResponse('')

      try {
        const response = await fetch('/api/commentary/statistics/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            userQuestion,
            contexts: contextCards,
            playerData,
            conversationHistory,
          }),
        })

        if (!response.ok) {
          throw new Error('Failed to get AI response')
        }

        // Parse SSE stream
        const reader = response.body?.getReader()
        const decoder = new TextDecoder()
        let fullResponse = ''

        if (reader) {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            const chunk = decoder.decode(value)
            const lines = chunk.split('\n')

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6)
                if (data === '[DONE]') {
                  setIsStreaming(false)
                  // Add to conversation history
                  setConversationHistory((prev) => [
                    ...prev,
                    {
                      question: userQuestion,
                      contexts: [...contextCards],
                      answer: fullResponse,
                    },
                  ])
                  // Clear context cards after sending
                  setContextCards([])
                } else {
                  try {
                    const content = JSON.parse(data)
                    fullResponse += content
                    setCurrentResponse((prev) => prev + content)
                  } catch (e) {
                    // Ignore parse errors
                  }
                }
              }
            }
          }
        }
      } catch (error) {
        console.error('Error sending question:', error)
        setIsStreaming(false)
        setCurrentResponse('Sorry, I encountered an error analyzing your statistics. Please try again.')
      }
    },
    [contextCards, conversationHistory]
  )

  const clearAll = useCallback(() => {
    setConversationHistory([])
    setContextCards([])
    setCurrentResponse('')
  }, [])

  return {
    // Context management
    contextCards,
    addContextCard,
    removeContextCard,

    // Question sending
    sendQuestion,
    isStreaming,
    currentResponse,

    // Conversation management
    conversationHistory,
    clearAll,
  }
}
