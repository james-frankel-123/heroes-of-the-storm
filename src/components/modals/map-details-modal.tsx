'use client'

import * as React from 'react'
import { MapStats, PlayerData } from '@/types'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { StreamingText } from '@/components/commentary/streaming-text'
import { useMapCommentary } from '@/lib/hooks/use-map-commentary'
import { Trophy, Target, TrendingUp, Sparkles } from 'lucide-react'
import { getWinRateColor } from '@/lib/utils'

interface MapDetailsModalProps {
  map: MapStats
  playerData: PlayerData
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function MapDetailsModal({
  map,
  playerData,
  open,
  onOpenChange,
}: MapDetailsModalProps) {
  const { commentary, isStreaming, error } = useMapCommentary(
    map.map,
    playerData,
    { autoFetch: open } // Only fetch when modal is open
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl">{map.map}</DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="glass border border-primary-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-1">
                <Trophy className="h-4 w-4" />
                <span className="text-xs font-medium">Games</span>
              </div>
              <p className="text-2xl font-bold">{map.games}</p>
            </div>

            <div className="glass border border-primary-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-1">
                <Target className="h-4 w-4" />
                <span className="text-xs font-medium">Win Rate</span>
              </div>
              <p className={`text-2xl font-bold ${getWinRateColor(map.winRate)}`}>
                {map.winRate.toFixed(1)}%
              </p>
            </div>

            <div className="glass border border-primary-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-1">
                <TrendingUp className="h-4 w-4" />
                <span className="text-xs font-medium">Wins</span>
              </div>
              <p className="text-2xl font-bold text-gaming-success">{map.wins}</p>
            </div>

            <div className="glass border border-primary-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-1">
                <TrendingUp className="h-4 w-4 rotate-180" />
                <span className="text-xs font-medium">Losses</span>
              </div>
              <p className="text-2xl font-bold text-gaming-danger">{map.losses}</p>
            </div>
          </div>

          {/* AI Commentary */}
          <div className="glass border border-accent-cyan/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="h-4 w-4 text-accent-cyan" />
              <h3 className="font-semibold">AI Analysis</h3>
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
