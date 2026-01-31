'use client'

import { ReactNode } from 'react'
import { Sparkles, RefreshCw, X, AlertCircle } from 'lucide-react'
import { StreamingText, StreamingTextSkeleton } from './streaming-text'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface CommentaryCardProps {
  title: string
  description?: string
  commentary: string
  isStreaming: boolean
  error: string | null
  onRefresh?: () => void
  onClose?: () => void
  className?: string
  children?: ReactNode
}

export function CommentaryCard({
  title,
  description,
  commentary,
  isStreaming,
  error,
  onRefresh,
  onClose,
  className,
  children,
}: CommentaryCardProps) {
  return (
    <Card className={cn('w-full', className)}>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-2">
            <Sparkles className="w-5 h-5 text-blue-500 mt-0.5" />
            <div>
              <CardTitle className="text-lg">{title}</CardTitle>
              {description && (
                <CardDescription className="mt-1">{description}</CardDescription>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {onRefresh && !isStreaming && commentary && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onRefresh}
                className="h-8 w-8 p-0"
              >
                <RefreshCw className="w-4 h-4" />
              </Button>
            )}
            {onClose && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="h-8 w-8 p-0"
              >
                <X className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {error ? (
          <div className="flex items-start gap-2 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-red-900 dark:text-red-100">
                Error generating commentary
              </p>
              <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
              {onRefresh && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onRefresh}
                  className="mt-2"
                >
                  Try again
                </Button>
              )}
            </div>
          </div>
        ) : isStreaming && !commentary ? (
          <StreamingTextSkeleton />
        ) : (
          <>
            <StreamingText
              text={commentary}
              isStreaming={isStreaming}
              showCursor={isStreaming}
            />
            {children}
          </>
        )}
      </CardContent>
    </Card>
  )
}
