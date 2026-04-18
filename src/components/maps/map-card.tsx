'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RoleBadge } from '@/components/shared/role-badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import {
  cn,
  formatPercent,
  formatNumber,
  getWinRateColor,
  confidenceAdjustedWinRate,
} from '@/lib/utils'
import type { MapStats, HeroMapStats } from '@/lib/types'

interface MapCardProps {
  mapStat: MapStats
  personalStats: {
    battletag: string
    games: number
    wins: number
    winRate: number
  }[]
  topHeroes: HeroMapStats[]
  bottomHeroes: HeroMapStats[]
  onClick: () => void
}

export function MapCard({
  mapStat,
  personalStats,
  topHeroes,
  bottomHeroes,
  onClick,
}: MapCardProps) {
  return (
    <Card
      className="cursor-pointer hover:border-primary/40 transition-colors"
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center justify-between">
          <span>{mapStat.map}</span>
          <span className="text-xs font-normal text-muted-foreground">
            {formatNumber(mapStat.games)} games
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Top heroes on this map */}
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
            Strongest Heroes
          </p>
          <div className="space-y-1">
            {topHeroes.map((h) => {
              const role = getHeroRole(h.hero)
              const adjWR = confidenceAdjustedWinRate(h.wins, h.games, 50)
              return (
                <div
                  key={h.hero}
                  className="flex items-center justify-between text-sm"
                >
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{h.hero}</span>
                    {role && (
                      <RoleBadge role={role!} className="text-[8px] px-1 py-0" />
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      {formatNumber(h.games)}g
                    </span>
                    <span
                      className={cn(
                        'font-semibold text-xs w-12 text-right',
                        getWinRateColor(adjWR)
                      )}
                    >
                      {formatPercent(adjWR)}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Weakest heroes preview */}
        {bottomHeroes.length > 0 && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
              Weakest Heroes
            </p>
            <div className="space-y-1">
              {bottomHeroes.map((h) => {
                const adjWR = confidenceAdjustedWinRate(h.wins, h.games, 50)
                return (
                  <div
                    key={h.hero}
                    className="flex items-center justify-between text-sm"
                  >
                    <span className="text-muted-foreground">{h.hero}</span>
                    <span
                      className={cn(
                        'font-semibold text-xs w-12 text-right',
                        getWinRateColor(adjWR)
                      )}
                    >
                      {formatPercent(adjWR)}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Personal stats */}
        {personalStats.length > 0 && (
          <div className="border-t pt-2">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
              Your Stats
            </p>
            <div className="space-y-1">
              {personalStats.map((ps) => (
                <div
                  key={ps.battletag}
                  className="flex items-center justify-between text-sm"
                >
                  <span className="text-xs truncate max-w-[140px]">
                    {ps.battletag}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      {ps.games}g
                    </span>
                    <span
                      className={cn(
                        'font-semibold text-xs w-12 text-right',
                        getWinRateColor(ps.winRate)
                      )}
                    >
                      {formatPercent(ps.winRate)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
