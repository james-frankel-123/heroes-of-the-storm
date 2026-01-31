'use client'

import * as React from 'react'
import { PlayerData, PowerPick } from '@/types'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { StreamingText } from '@/components/commentary/streaming-text'
import { useStreamingCommentary } from '@/lib/hooks/use-streaming-commentary'
import { Trophy, Target, Map as MapIcon, Sparkles } from 'lucide-react'
import { getWinRateColor } from '@/lib/utils'

interface PowerPickModalProps {
  powerPick: PowerPick
  playerData: PlayerData
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function PowerPickModal({
  powerPick,
  playerData,
  open,
  onOpenChange,
}: PowerPickModalProps) {
  const { commentary, isStreaming, error, fetchCommentary } = useStreamingCommentary()

  React.useEffect(() => {
    if (open && powerPick && playerData) {
      fetchCommentary('/api/commentary/power-pick', {
        powerPick,
        playerData,
      })
    }
  }, [open, powerPick.hero, powerPick.map, playerData?.playerName])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-2xl">
            <span>{powerPick.hero}</span>
            <Badge variant="outline" className="text-xs">
              {powerPick.role}
            </Badge>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Map Badge */}
          <div className="flex items-center gap-2">
            <MapIcon className="h-4 w-4 text-primary-500" />
            <span className="text-sm text-muted-foreground">Best on</span>
            <Badge className="bg-primary-500/20 text-primary-500 border-primary-500/30">
              {powerPick.map}
            </Badge>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="glass border border-primary-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-1">
                <Trophy className="h-4 w-4" />
                <span className="text-xs font-medium">Games</span>
              </div>
              <p className="text-2xl font-bold">{powerPick.games}</p>
            </div>

            <div className="glass border border-primary-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-1">
                <Target className="h-4 w-4" />
                <span className="text-xs font-medium">Win Rate</span>
              </div>
              <p className={`text-2xl font-bold ${getWinRateColor(powerPick.winRate)}`}>
                {powerPick.winRate.toFixed(1)}%
              </p>
            </div>

            <div className="glass border border-gaming-success/30 rounded-lg p-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-1">
                <Trophy className="h-4 w-4" />
                <span className="text-xs font-medium">Wins</span>
              </div>
              <p className="text-2xl font-bold text-gaming-success">{powerPick.wins}</p>
            </div>

            <div className="glass border border-gaming-danger/30 rounded-lg p-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-1">
                <Trophy className="h-4 w-4" />
                <span className="text-xs font-medium">Losses</span>
              </div>
              <p className="text-2xl font-bold text-gaming-danger">{powerPick.losses}</p>
            </div>
          </div>

          {/* AI Commentary */}
          <div className="glass border border-accent-cyan/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="h-4 w-4 text-accent-cyan" />
              <h3 className="font-semibold">Why This is a Power Pick</h3>
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
