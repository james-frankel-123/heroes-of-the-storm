'use client'

import * as React from 'react'
import { Send, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ContextCard } from './context-card'
import { ContextData } from './clickable-metric'
import { StreamingText } from '@/components/commentary/streaming-text'

interface Message {
  question: string
  contexts: (ContextData & { id: number })[]
  answer: string
}

interface AIAnalysisPanelProps {
  contextCards: (ContextData & { id: number })[]
  onRemoveCard: (id: number) => void
  onSendQuestion: (question: string) => void
  conversationHistory: Message[]
  isStreaming: boolean
  currentResponse: string
  onClearAll: () => void
}

export function AIAnalysisPanel({
  contextCards,
  onRemoveCard,
  onSendQuestion,
  conversationHistory,
  isStreaming,
  currentResponse,
  onClearAll,
}: AIAnalysisPanelProps) {
  const [userQuestion, setUserQuestion] = React.useState('')
  const conversationRef = React.useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new messages arrive
  React.useEffect(() => {
    if (conversationRef.current) {
      conversationRef.current.scrollTop = conversationRef.current.scrollHeight
    }
  }, [conversationHistory, currentResponse])

  const handleSend = () => {
    if (userQuestion.trim() && contextCards.length > 0 && !isStreaming) {
      onSendQuestion(userQuestion)
      setUserQuestion('')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const suggestedQuestions = contextCards.length > 0 ? [
    'Why did this happen?',
    'How can I improve?',
    'What should I focus on?',
  ] : []

  return (
    <div className="w-96 border-l border-border bg-card/30 backdrop-blur-sm flex flex-col h-screen">
      {/* Header */}
      <div className="p-4 border-b border-border flex-shrink-0">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <span>🤖</span>
            <span>AI Assistant</span>
          </h3>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClearAll}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            Clear
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Click stats to add context, then ask questions
        </p>
      </div>

      {/* Context cards area */}
      <div className="p-4 border-b border-border flex-shrink-0 max-h-60 overflow-y-auto">
        {contextCards.length > 0 ? (
          <div className="space-y-2">
            {contextCards.map((card) => (
              <ContextCard
                key={card.id}
                data={card}
                onRemove={() => onRemoveCard(card.id)}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-4">
            Click a statistic to start analyzing
          </p>
        )}
      </div>

      {/* Chat input */}
      <div className="p-4 border-b border-border flex-shrink-0">
        <textarea
          value={userQuestion}
          onChange={(e) => setUserQuestion(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about the stats above..."
          className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm resize-none"
          rows={3}
          disabled={isStreaming || contextCards.length === 0}
        />
        <Button
          onClick={handleSend}
          disabled={isStreaming || !userQuestion.trim() || contextCards.length === 0}
          className="mt-2 w-full"
        >
          {isStreaming ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Send className="h-4 w-4 mr-2" />
              Send
            </>
          )}
        </Button>
      </div>

      {/* Suggested questions */}
      {suggestedQuestions.length > 0 && !isStreaming && (
        <div className="p-4 border-b border-border flex-shrink-0">
          <p className="text-xs font-medium text-muted-foreground mb-2">Suggested:</p>
          <div className="space-y-1">
            {suggestedQuestions.map((question, idx) => (
              <button
                key={idx}
                onClick={() => setUserQuestion(question)}
                className="w-full text-left text-xs px-2 py-1 rounded hover:bg-muted transition-colors text-muted-foreground"
              >
                • {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Conversation history */}
      <div ref={conversationRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {conversationHistory.length === 0 && !currentResponse ? (
          <p className="text-sm text-muted-foreground text-center">
            Ask a question to start the conversation
          </p>
        ) : (
          <>
            {conversationHistory.map((message, idx) => (
              <div key={idx} className="space-y-2">
                {/* User question */}
                <div className="bg-primary-500/10 rounded-lg p-3">
                  <p className="text-sm font-medium mb-1">👤 You:</p>
                  <p className="text-sm">{message.question}</p>
                  {message.contexts.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {message.contexts.map((ctx) => (
                        <span
                          key={ctx.id}
                          className="text-xs px-2 py-0.5 rounded bg-background/50"
                        >
                          {ctx.label}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* AI answer */}
                <div className="bg-muted/30 rounded-lg p-3">
                  <p className="text-sm font-medium mb-1">🤖 AI:</p>
                  <div className="text-sm prose prose-sm dark:prose-invert max-w-none">
                    <StreamingText text={message.answer} isStreaming={false} />
                  </div>
                </div>
              </div>
            ))}

            {/* Current streaming response */}
            {currentResponse && (
              <div className="space-y-2">
                <div className="bg-muted/30 rounded-lg p-3">
                  <p className="text-sm font-medium mb-1">🤖 AI:</p>
                  <div className="text-sm prose prose-sm dark:prose-invert max-w-none">
                    <StreamingText text={currentResponse} isStreaming={isStreaming} />
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
