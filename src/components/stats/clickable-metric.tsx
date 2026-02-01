'use client'

import * as React from 'react'
import { Sparkles } from 'lucide-react'

export interface ContextData {
  type: 'metric' | 'chart-point' | 'hero-row' | 'map-metric' | 'streak' | 'time-period' | 'comparison'
  label: string
  value: number | string
  trend?: 'increasing' | 'decreasing' | 'stable'
  change?: string
  timeRange?: {
    start: string
    end: string
    granularity?: 'daily' | 'weekly' | 'monthly'
  }
  relatedMetrics?: Record<string, any>
  chartData?: any[]
  hero?: string
  map?: string
  compareWith?: any
}

interface ClickableMetricProps {
  context: ContextData
  onAddToChat: (context: ContextData) => void
  children: React.ReactNode
  className?: string
}

export function ClickableMetric({
  context,
  onAddToChat,
  children,
  className = '',
}: ClickableMetricProps) {
  const [isHovered, setIsHovered] = React.useState(false)

  const handleClick = () => {
    onAddToChat(context)
  }

  return (
    <div
      className={`relative cursor-pointer transition-all ${
        isHovered ? 'scale-[1.02]' : 'scale-100'
      } ${className}`}
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      role="button"
      tabIndex={0}
      onKeyPress={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          handleClick()
        }
      }}
      aria-label={`Click to analyze ${context.label}`}
    >
      {children}

      {/* Sparkles indicator on hover */}
      {isHovered && (
        <div className="absolute top-2 right-2 animate-pulse">
          <Sparkles className="h-4 w-4 text-primary-500" />
        </div>
      )}

      {/* Hover overlay */}
      {isHovered && (
        <div className="absolute inset-0 bg-primary-500/5 rounded-lg pointer-events-none" />
      )}
    </div>
  )
}
