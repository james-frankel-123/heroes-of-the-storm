'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  formatPercent,
  getWinRateColor,
  cn,
} from '@/lib/utils'
import type { PlayerHeroStats, TrackedBattletag } from '@/lib/types'

interface PersonalInsightsProps {
  battletag: TrackedBattletag
  heroStats: PlayerHeroStats[]
}

export function PersonalInsights({ battletag, heroStats }: PersonalInsightsProps) {
  const sorted = [...heroStats].sort((a, b) => b.winRate - a.winRate)

  const overperforming = sorted
    .filter((h) => h.winRate >= 52 && h.games >= 5)
    .slice(0, 8)

  const underperforming = sorted
    .filter((h) => h.winRate < 48 && h.games >= 10)
    .reverse()
    .slice(0, 5)

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-3">
          {battletag.battletag}
          {battletag.lastSynced && (
            <span className="text-xs font-normal text-muted-foreground">
              Last synced: {battletag.lastSynced.toLocaleDateString()}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Strong heroes */}
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-3">
              Strongest Heroes
            </h4>
            {overperforming.length === 0 ? (
              <p className="text-sm text-muted-foreground">Not enough data yet</p>
            ) : (
              <HeroTable heroes={overperforming} />
            )}
          </div>

          {/* Weak heroes */}
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-3">
              Needs Improvement
            </h4>
            {underperforming.length === 0 ? (
              <p className="text-sm text-muted-foreground">No underperforming heroes</p>
            ) : (
              <HeroTable heroes={underperforming} />
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function HeroTable({ heroes }: { heroes: PlayerHeroStats[] }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-muted-foreground">
          <th className="text-left font-medium pb-1.5">Hero</th>
          <th className="text-right font-medium pb-1.5 w-16">Games</th>
          <th className="text-right font-medium pb-1.5 w-16">Win %</th>
          <th className="text-right font-medium pb-1.5 w-16">Trend</th>
        </tr>
      </thead>
      <tbody>
        {heroes.map((h) => {
          const trend = h.trend ?? 0
          return (
            <tr
              key={h.hero}
              className="hover:bg-accent/50 transition-colors"
            >
              <td className="py-1.5 pr-2 font-medium">{h.hero}</td>
              <td className="py-1.5 text-right text-muted-foreground">
                {h.games}
              </td>
              <td
                className={cn(
                  'py-1.5 text-right font-semibold',
                  getWinRateColor(h.winRate)
                )}
              >
                {formatPercent(h.winRate)}
              </td>
              <td
                className={cn(
                  'py-1.5 text-right',
                  trend > 0
                    ? 'text-gaming-success'
                    : trend < 0
                      ? 'text-gaming-danger'
                      : 'text-muted-foreground'
                )}
              >
                {h.games >= 20 && trend !== 0
                  ? `${trend > 0 ? '+' : ''}${formatPercent(trend)}`
                  : '-'}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
