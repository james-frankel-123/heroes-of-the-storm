'use client'

import React from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'

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
  const renderCountRef = React.useRef(0)
  const lastTextLengthRef = React.useRef(0)
  const lastNewlineCountRef = React.useRef(0)

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
