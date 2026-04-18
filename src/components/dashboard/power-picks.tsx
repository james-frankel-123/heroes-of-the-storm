'use client'

import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { heroImageSrc } from '@/lib/data/hero-images'
import { mapImageSrc } from '@/lib/data/map-images'
import { CURRENT_MAP_ROTATION, isInRotation } from '@/lib/data/map-rotation'
import { formatPercent, getWinRateColor, formatNumber } from '@/lib/utils'
import type { HeroMapStats } from '@/lib/types'

interface PowerPicksProps {
  picks: HeroMapStats[]
}

export function PowerPicks({ picks }: PowerPicksProps) {
  const filtered = useMemo(
    () => picks.filter((p) => isInRotation(p.map)),
    [picks]
  )

  if (filtered.length === 0) {
    return null
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          Power Picks
          <span className="text-xs font-normal text-muted-foreground">
            Hero+map combos with 55%+ win rate (100+ games)
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Current Map Rotation */}
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
            Current Map Rotation
          </p>
          <div className="flex flex-wrap gap-2">
            {CURRENT_MAP_ROTATION.map((map) => {
              const img = mapImageSrc(map)
              return (
                <div key={map} className="flex flex-col items-center gap-0.5 w-16">
                  {img && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={img}
                      alt={map}
                      className="w-full h-8 rounded object-cover border border-border"
                    />
                  )}
                  <span className="text-[9px] text-muted-foreground truncate w-full text-center">
                    {map}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Power picks list */}
        <div className="space-y-2">
          {filtered.map((pick, i) => {
            const role = getHeroRole(pick.hero)
            return (
              <div
                key={`${pick.hero}-${pick.map}`}
                className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-accent/50 transition-colors"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <span className="text-xs text-muted-foreground w-5 text-right">
                    {i + 1}
                  </span>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={heroImageSrc(pick.hero)} alt="" loading="lazy" className="w-7 h-7 rounded object-cover shrink-0" />
                  <span className="font-medium text-sm">{pick.hero}</span>
                  <span className="text-muted-foreground text-sm">on</span>
                  <span className="text-sm">{pick.map}</span>
                </div>
                <div className="flex items-center gap-4 text-sm shrink-0">
                  <span className="text-muted-foreground">
                    {formatNumber(pick.games)} games
                  </span>
                  <span className={`font-semibold w-14 text-right ${getWinRateColor(pick.winRate)}`}>
                    {formatPercent(pick.winRate)}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
