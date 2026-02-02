'use client'

import * as React from 'react'
import { HeroStats, PlayerData } from '@/types'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Trophy, Target, TrendingUp, Map } from 'lucide-react'
import { getWinRateColor } from '@/lib/utils'

interface HeroDetailsModalProps {
  hero: HeroStats
  playerData: PlayerData
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function HeroDetailsModal({
  hero,
  playerData,
  open,
  onOpenChange,
}: HeroDetailsModalProps) {
  // Get map performance for this hero from playerData
  const mapPerformance = React.useMemo(() => {
    const performance: Array<{
      map: string
      wins: number
      losses: number
      games: number
      winRate: number
    }> = []

    // Find this hero's performance across all maps
    playerData.mapStats.forEach((mapStat) => {
      const heroOnMap = mapStat.heroes.find(h => h.hero === hero.hero)
      if (heroOnMap && heroOnMap.games >= 3) {
        performance.push({
          map: mapStat.map,
          wins: heroOnMap.wins,
          losses: heroOnMap.losses,
          games: heroOnMap.games,
          winRate: heroOnMap.winRate,
        })
      }
    })

    return performance.sort((a, b) => b.winRate - a.winRate)
  }, [hero.hero, playerData.mapStats])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[75vw] max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-2xl">
            <span>{hero.hero}</span>
            <Badge variant="outline" className="text-xs">
              {hero.role}
            </Badge>
          </DialogTitle>
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
                <p className="text-2xl font-bold">{hero.games}</p>
              </div>

              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Target className="h-4 w-4" />
                  <span className="text-xs font-medium">Win Rate</span>
                </div>
                <p className={`text-2xl font-bold ${getWinRateColor(hero.winRate)}`}>
                  {hero.winRate.toFixed(1)}%
                </p>
              </div>

              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <TrendingUp className="h-4 w-4" />
                  <span className="text-xs font-medium">Wins</span>
                </div>
                <p className="text-2xl font-bold text-gaming-success">{hero.wins}</p>
              </div>

              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <TrendingUp className="h-4 w-4 rotate-180" />
                  <span className="text-xs font-medium">Losses</span>
                </div>
                <p className="text-2xl font-bold text-gaming-danger">{hero.losses}</p>
              </div>
            </div>

            {/* Map Performance */}
            {mapPerformance.length > 0 && (
              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Map className="h-4 w-4 text-primary-500" />
                  <h3 className="font-semibold">Map Performance</h3>
                </div>
                <div className="space-y-2">
                  {mapPerformance.slice(0, 5).map((mapStat) => (
                    <div
                      key={mapStat.map}
                      className="flex items-center justify-between text-sm"
                    >
                      <span className="text-muted-foreground">{mapStat.map}</span>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-muted-foreground">
                          {mapStat.wins}W - {mapStat.losses}L
                        </span>
                        <span className={`font-semibold ${getWinRateColor(mapStat.winRate)}`}>
                          {mapStat.winRate.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
