'use client'

import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

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
  const containerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom as text streams in
  useEffect(() => {
    if (containerRef.current && isStreaming) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [text, isStreaming])

  if (!text && !isStreaming) {
    return null
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        'prose prose-sm max-w-none dark:prose-invert',
        'whitespace-pre-wrap break-words',
        className
      )}
    >
      {text}
      {isStreaming && showCursor && (
        <span className="inline-block w-2 h-4 ml-1 bg-blue-500 animate-pulse" />
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
