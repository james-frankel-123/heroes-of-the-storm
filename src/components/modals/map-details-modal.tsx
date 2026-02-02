'use client'

import * as React from 'react'
import { MapStats, PlayerData } from '@/types'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Trophy, Target, TrendingUp } from 'lucide-react'
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
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[75vw] max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="text-2xl">{map.map}</DialogTitle>
        </DialogHeader>

        <div className="max-h-[calc(90vh-8rem)] overflow-hidden">
          {/* Stats and Details */}
          <div className="space-y-6 overflow-y-auto pr-2">
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
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
