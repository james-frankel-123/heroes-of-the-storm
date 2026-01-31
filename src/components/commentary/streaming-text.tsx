'use client'

import { Loader2 } from 'lucide-react'
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
      className={cn(
        'prose prose-sm max-w-none dark:prose-invert',
        'whitespace-pre-wrap break-words',
        className
      )}
    >
      {text}
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
