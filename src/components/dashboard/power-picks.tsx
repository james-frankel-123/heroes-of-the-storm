'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { formatPercent, getWinRateColor, formatNumber } from '@/lib/utils'
import type { HeroMapStats } from '@/lib/types'

interface PowerPicksProps {
  picks: HeroMapStats[]
}

export function PowerPicks({ picks }: PowerPicksProps) {
  if (picks.length === 0) {
    return null
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          Power Picks
          <span className="text-xs font-normal text-muted-foreground">
            Hero+map combos with 60%+ win rate
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {picks.map((pick, i) => {
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
