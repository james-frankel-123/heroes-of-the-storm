'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import {
  formatPercent,
  getWinRateColor,
  confidenceAdjustedMawp,
  confidenceLabel,
  cn,
} from '@/lib/utils'
import type { PlayerHeroStats, TrackedBattletag } from '@/lib/types'

interface PersonalInsightsProps {
  battletag: TrackedBattletag
  heroStats: PlayerHeroStats[]
}

export function PersonalInsights({ battletag, heroStats }: PersonalInsightsProps) {
  // Sort by confidence-adjusted MAWP descending
  const sorted = [...heroStats].sort((a, b) => {
    const aMawp = confidenceAdjustedMawp(a.mawp ?? a.winRate, a.games, 30)
    const bMawp = confidenceAdjustedMawp(b.mawp ?? b.winRate, b.games, 30)
    return bMawp - aMawp
  })

  const overperforming = sorted.filter((h) => {
    const adj = confidenceAdjustedMawp(h.mawp ?? h.winRate, h.games, 30)
    return adj >= 52
  }).slice(0, 8)

  const underperforming = sorted.filter((h) => {
    const adj = confidenceAdjustedMawp(h.mawp ?? h.winRate, h.games, 30)
    return adj < 48 && h.games >= 10
  }).reverse().slice(0, 5)

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
            <div className="space-y-2">
              {overperforming.map((hero) => (
                <HeroRow key={hero.hero} stats={hero} />
              ))}
              {overperforming.length === 0 && (
                <p className="text-sm text-muted-foreground">Not enough data yet</p>
              )}
            </div>
          </div>

          {/* Weak heroes */}
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-3">
              Needs Improvement
            </h4>
            <div className="space-y-2">
              {underperforming.map((hero) => (
                <HeroRow key={hero.hero} stats={hero} />
              ))}
              {underperforming.length === 0 && (
                <p className="text-sm text-muted-foreground">No underperforming heroes</p>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function HeroRow({ stats }: { stats: PlayerHeroStats }) {
  const rawMawp = stats.mawp ?? stats.winRate
  const adjMawp = confidenceAdjustedMawp(rawMawp, stats.games, 30)
  const confidence = confidenceLabel(stats.games, 30)
  const role = getHeroRole(stats.hero)
  const trend = stats.trend ?? 0

  return (
    <div className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-accent/50 transition-colors">
      <div className="flex items-center gap-2 min-w-0">
        <span className="font-medium text-sm truncate">{stats.hero}</span>
        {confidence !== 'high' && (
          <Badge
            variant="outline"
            className={cn(
              'text-[10px] px-1 py-0',
              confidence === 'low' ? 'border-gaming-danger/50 text-gaming-danger' : 'border-gaming-warning/50 text-gaming-warning'
            )}
          >
            {confidence === 'low' ? 'Low data' : 'Limited'}
          </Badge>
        )}
      </div>
      <div className="flex items-center gap-3 text-sm shrink-0">
        <span className="text-muted-foreground text-xs">{stats.games}g</span>
        <div className="text-right w-20">
          <span className={`font-semibold ${getWinRateColor(adjMawp)}`}>
            {formatPercent(adjMawp)}
          </span>
          {confidence === 'high' && rawMawp !== stats.winRate && (
            <span className="text-[10px] text-muted-foreground ml-1">
              MAWP
            </span>
          )}
        </div>
        {trend !== 0 && stats.games >= 20 && (
          <span
            className={cn(
              'text-xs w-12 text-right',
              trend > 0 ? 'text-gaming-success' : 'text-gaming-danger'
            )}
          >
            {trend > 0 ? '+' : ''}
            {formatPercent(trend)}
          </span>
        )}
      </div>
    </div>
  )
}
