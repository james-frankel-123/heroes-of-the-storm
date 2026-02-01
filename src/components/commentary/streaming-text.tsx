'use client'

import React from 'react'
import { Loader2, Lock } from 'lucide-react'
import { cn } from '@/lib/utils'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface StreamingTextProps {
  text: string
  isStreaming: boolean
  className?: string
  showCursor?: boolean
}

export function StreamingText({
  text,
  isStreaming,
  className,
  showCursor = true,
}: StreamingTextProps) {
  const [isAuthenticated, setIsAuthenticated] = React.useState(false)
  const [password, setPassword] = React.useState('')
  const [error, setError] = React.useState('')
  const [isClient, setIsClient] = React.useState(false)

  const renderCountRef = React.useRef(0)
  const lastTextLengthRef = React.useRef(0)
  const lastNewlineCountRef = React.useRef(0)

  // Check authentication on mount
  React.useEffect(() => {
    setIsClient(true)
    const stored = sessionStorage.getItem('protected_pages_auth')
    if (stored === 'ronpaul2012') {
      setIsAuthenticated(true)
    }
  }, [])

  const handleUnlock = (e: React.FormEvent) => {
    e.preventDefault()
    if (password === 'ronpaul2012') {
      sessionStorage.setItem('protected_pages_auth', password)
      setIsAuthenticated(true)
      setError('')
    } else {
      setError('Incorrect password')
      setPassword('')
    }
  }

  React.useEffect(() => {
    if (isStreaming) {
      renderCountRef.current += 1
      const newLength = text.length
      const lengthDiff = newLength - lastTextLengthRef.current
      const newlineCount = (text.match(/\n/g) || []).length
      const newlineDiff = newlineCount - lastNewlineCountRef.current

      console.log('=== StreamingText Render ===')
      console.log('Render #:', renderCountRef.current)
      console.log('Text length:', newLength)
      console.log('Length diff:', lengthDiff, lengthDiff < 0 ? '⚠️ SHRINKING!' : '✓')
      console.log('Newline count:', newlineCount)
      console.log('Newline diff:', newlineDiff)
      console.log('Last 80 chars:', JSON.stringify(text.substring(Math.max(0, newLength - 80), newLength)))

      if (lengthDiff < 0) {
        console.error('🚨 TEXT SHRUNK! Previous length:', lastTextLengthRef.current, '→ New length:', newLength)
      }

      lastTextLengthRef.current = newLength
      lastNewlineCountRef.current = newlineCount
    }
  }, [text, isStreaming])

  // Wait for client-side hydration
  if (!isClient) {
    return null
  }

  // Show password prompt if not authenticated
  if (!isAuthenticated) {
    return (
      <div className="rounded-lg border border-border bg-muted/30 p-6">
        <div className="flex flex-col items-center space-y-4 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary-500/10">
            <Lock className="h-6 w-6 text-primary-500" />
          </div>
          <div>
            <h3 className="font-semibold text-lg">AI Commentary Locked</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Enter password to view AI-generated insights
            </p>
          </div>
          <form onSubmit={handleUnlock} className="w-full max-w-sm space-y-3">
            <Input
              type="password"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className={error ? 'border-red-500' : ''}
            />
            {error && (
              <p className="text-sm text-red-500">{error}</p>
            )}
            <Button type="submit" className="w-full">
              Unlock Commentary
            </Button>
          </form>
        </div>
      </div>
    )
  }

  if (!text && !isStreaming) {
    return null
  }

  // Show spinner while loading
  if (isStreaming && !text) {
    return (
      <div className="flex items-center gap-2 text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span className="text-xs">Analyzing...</span>
      </div>
    )
  }

  return (
    <div
      className={cn('relative prose prose-sm dark:prose-invert max-w-none', className)}
      style={{
        // Prevent layout shifts during streaming
        minHeight: isStreaming ? '100px' : 'auto',
      }}
    >
      <div className={isStreaming ? 'opacity-90' : ''}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h2: ({node, ...props}) => <h2 className="text-xl font-bold mt-6 mb-4 first:mt-0" {...props} />,
            h3: ({node, ...props}) => <h3 className="text-lg font-semibold mt-4 mb-3" {...props} />,
            ul: ({node, ...props}) => <ul className="list-disc pl-6 my-4 space-y-2" {...props} />,
            li: ({node, ...props}) => <li className="text-sm" {...props} />,
            p: ({node, ...props}) => <p className="my-2" {...props} />,
            strong: ({node, ...props}) => <strong className="font-bold text-foreground" {...props} />,
          }}
        >
          {text}
        </ReactMarkdown>
      </div>
      {isStreaming && showCursor && (
        <span className="inline-block w-2 h-4 bg-primary-500 animate-pulse ml-1" />
      )}
    </div>
  )
}

// Loading skeleton for when commentary is being fetched
export function StreamingTextSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn('space-y-2 animate-pulse', className)}>
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-full" />
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6" />
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-4/6" />
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-full" />
      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4" />
    </div>
  )
}
