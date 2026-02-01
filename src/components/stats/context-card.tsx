'use client'

import { X } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { ContextData } from './clickable-metric'

interface ContextCardProps {
  data: ContextData & { id: number }
  onRemove: () => void
}

export function ContextCard({ data, onRemove }: ContextCardProps) {
  const getTypeColor = () => {
    switch (data.type) {
      case 'metric':
        return 'bg-blue-500/20 text-blue-500 border-blue-500/30'
      case 'chart-point':
        return 'bg-purple-500/20 text-purple-500 border-purple-500/30'
      case 'hero-row':
        return 'bg-green-500/20 text-green-500 border-green-500/30'
      case 'streak':
        return 'bg-yellow-500/20 text-yellow-500 border-yellow-500/30'
      default:
        return 'bg-primary-500/20 text-primary-500 border-primary-500/30'
    }
  }

  return (
    <div className={`relative p-3 rounded-md border ${getTypeColor()} group`}>
      {/* Remove button */}
      <button
        onClick={onRemove}
        className="absolute top-1 right-1 p-1 rounded-full hover:bg-background/20 transition-colors opacity-0 group-hover:opacity-100"
        aria-label="Remove context"
      >
        <X className="h-3 w-3" />
      </button>

      {/* Content */}
      <div className="pr-6">
        <div className="flex items-center gap-2 mb-1">
          <Badge variant="outline" className="text-xs px-1.5 py-0 h-5">
            {data.type}
          </Badge>
          <span className="text-xs font-medium">{data.label}</span>
        </div>

        <div className="text-sm font-bold">
          {typeof data.value === 'number' && data.value % 1 !== 0
            ? data.value.toFixed(1)
            : data.value}
          {data.type === 'metric' && typeof data.value === 'number' && data.label.toLowerCase().includes('rate') && '%'}
        </div>

        {data.trend && (
          <div className="text-xs text-muted-foreground mt-1">
            {data.trend === 'increasing' && '↗ Trending up'}
            {data.trend === 'decreasing' && '↘ Trending down'}
            {data.trend === 'stable' && '→ Stable'}
            {data.change && ` (${data.change})`}
          </div>
        )}

        {data.timeRange && (
          <div className="text-xs text-muted-foreground mt-1">
            {new Date(data.timeRange.start).toLocaleDateString()} - {new Date(data.timeRange.end).toLocaleDateString()}
          </div>
        )}
      </div>
    </div>
  )
}
