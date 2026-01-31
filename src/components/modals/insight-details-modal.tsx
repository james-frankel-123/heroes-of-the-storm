'use client'

import * as React from 'react'
import { PlayerData, Insight } from '@/types'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { StreamingText } from '@/components/commentary/streaming-text'
import { useStreamingCommentary } from '@/lib/hooks/use-streaming-commentary'
import { TrendingUp, AlertTriangle, Lightbulb, Info, Sparkles } from 'lucide-react'

interface InsightDetailsModalProps {
  insight: Insight
  playerData: PlayerData
  open: boolean
  onOpenChange: (open: boolean) => void
}

const iconMap = {
  success: TrendingUp,
  warning: AlertTriangle,
  info: Info,
  tip: Lightbulb,
}

const colorMap = {
  success: 'text-gaming-success border-gaming-success/30',
  warning: 'text-gaming-warning border-gaming-warning/30',
  info: 'text-primary-500 border-primary-500/30',
  tip: 'text-accent-cyan border-accent-cyan/30',
}

export function InsightDetailsModal({
  insight,
  playerData,
  open,
  onOpenChange,
}: InsightDetailsModalProps) {
  const { commentary, isStreaming, error, fetchCommentary } = useStreamingCommentary()

  const Icon = iconMap[insight.type]

  React.useEffect(() => {
    if (open && insight && playerData) {
      fetchCommentary('/api/commentary/insight-detail', {
        insight,
        playerData,
      })
    }
  }, [open, insight.title, playerData?.playerName])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-2xl">
            <Icon className={`h-6 w-6 ${colorMap[insight.type]}`} />
            <span>{insight.title}</span>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Original Insight */}
          <div className={`glass border rounded-lg p-4 ${colorMap[insight.type]} bg-${insight.type}/5`}>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {insight.description}
            </p>
          </div>

          {/* Expanded AI Analysis */}
          <div className="glass border border-accent-cyan/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="h-4 w-4 text-accent-cyan" />
              <h3 className="font-semibold">Detailed Analysis & Recommendations</h3>
            </div>
            {error ? (
              <p className="text-sm text-gaming-danger">{error}</p>
            ) : (
              <StreamingText
                text={commentary}
                isStreaming={isStreaming}
                className="text-sm text-muted-foreground leading-relaxed"
              />
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
