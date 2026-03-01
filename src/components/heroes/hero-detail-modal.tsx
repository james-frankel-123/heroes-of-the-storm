'use client'

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { getHeroRole } from '@/lib/data/hero-roles'
import {
  cn,
  formatPercent,
  formatNumber,
  getWinRateColor,
  confidenceAdjustedMawp,
  confidenceLabel,
} from '@/lib/utils'
import type {
  HeroStats,
  HeroMapStats,
  HeroTalentStats,
  HeroPairwiseStats,
  PlayerHeroStats,
  PlayerMatch,
  SkillTier,
} from '@/lib/types'

interface HeroDetailModalProps {
  open: boolean
  onClose: () => void
  heroName: string
  statsByTier: HeroStats[]
  mapStats: HeroMapStats[]
  talents: HeroTalentStats[]
  synergies: HeroPairwiseStats[]
  counters: HeroPairwiseStats[]
  personalStats: { battletag: string; stats: PlayerHeroStats | null }[]
  recentMatches: { battletag: string; matches: PlayerMatch[] }[]
  currentTier: SkillTier
}

export function HeroDetailModal({
  open,
  onClose,
  heroName,
  statsByTier,
  mapStats,
  talents,
  synergies,
  counters,
  personalStats,
  recentMatches,
  currentTier,
}: HeroDetailModalProps) {
  const role = getHeroRole(heroName)

  // Group talents by tier
  const talentsByTier: Record<number, HeroTalentStats[]> = {}
  for (const t of talents) {
    if (!talentsByTier[t.talentTier]) talentsByTier[t.talentTier] = []
    talentsByTier[t.talentTier].push(t)
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <span className="text-xl">{heroName}</span>
            {role && (
              <Badge variant="outline" className="text-xs">
                {role}
              </Badge>
            )}
          </DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="overview" className="mt-2">
          <TabsList className="w-full justify-start">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="maps">Maps</TabsTrigger>
            <TabsTrigger value="talents">Talents</TabsTrigger>
            <TabsTrigger value="matchups">Matchups</TabsTrigger>
            <TabsTrigger value="personal">Personal</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4 mt-4">
            {/* Stats across tiers */}
            <div className="grid grid-cols-3 gap-3">
              {(['low', 'mid', 'high'] as SkillTier[]).map((tier) => {
                const s = statsByTier.find((st) => st.skillTier === tier)
                if (!s) return null
                const tierLabel =
                  tier === 'low'
                    ? 'Bronze+Silver'
                    : tier === 'mid'
                      ? 'Gold+Plat'
                      : 'Diamond+Master'
                return (
                  <Card
                    key={tier}
                    className={cn(
                      tier === currentTier && 'border-primary/50'
                    )}
                  >
                    <CardHeader className="pb-2 pt-4 px-4">
                      <CardTitle className="text-xs font-medium text-muted-foreground">
                        {tierLabel}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="px-4 pb-4 space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Win Rate</span>
                        <span
                          className={cn(
                            'font-semibold',
                            getWinRateColor(s.winRate)
                          )}
                        >
                          {formatPercent(s.winRate)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Games</span>
                        <span>{formatNumber(s.games)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Pick</span>
                        <span>{formatPercent(s.pickRate)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Ban</span>
                        <span>{formatPercent(s.banRate)}</span>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>

          </TabsContent>

          {/* Maps Tab */}
          <TabsContent value="maps" className="mt-4">
            <div className="rounded-lg border overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                      Map
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">
                      Win %
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">
                      Games
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {mapStats
                    .sort((a, b) => b.winRate - a.winRate)
                    .map((ms) => (
                      <tr
                        key={ms.map}
                        className="border-b last:border-0 hover:bg-accent/30"
                      >
                        <td className="px-3 py-2">{ms.map}</td>
                        <td
                          className={cn(
                            'px-3 py-2 text-right font-semibold',
                            getWinRateColor(ms.winRate)
                          )}
                        >
                          {formatPercent(ms.winRate)}
                        </td>
                        <td className="px-3 py-2 text-right text-muted-foreground">
                          {formatNumber(ms.games)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </TabsContent>

          {/* Talents Tab */}
          <TabsContent value="talents" className="mt-4 space-y-4">
            {talents.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No talent data available for this hero.
              </p>
            ) : (
              Object.entries(talentsByTier)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([tier, tierTalents]) => (
                  <div key={tier}>
                    <h4 className="text-xs font-medium text-muted-foreground mb-2">
                      Level {tier}
                    </h4>
                    <div className="space-y-1">
                      {tierTalents
                        .sort((a, b) => b.pickRate - a.pickRate)
                        .map((t) => (
                          <div
                            key={t.talentName}
                            className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-accent/30 text-sm"
                          >
                            <span className="font-medium">{t.talentName}</span>
                            <div className="flex items-center gap-4">
                              <span className="text-muted-foreground text-xs">
                                {formatPercent(t.pickRate)} pick
                              </span>
                              <span
                                className={cn(
                                  'font-semibold w-14 text-right',
                                  getWinRateColor(t.winRate)
                                )}
                              >
                                {formatPercent(t.winRate)}
                              </span>
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                ))
            )}
          </TabsContent>

          {/* Matchups Tab */}
          <TabsContent value="matchups" className="mt-4 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-2">
                  Best With (Synergies)
                </h4>
                {synergies.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No data</p>
                ) : (
                  <div className="space-y-1">
                    {synergies.map((p) => {
                      const partner =
                        p.heroA === heroName ? p.heroB : p.heroA
                      return (
                        <div
                          key={partner}
                          className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-accent/30 text-sm"
                        >
                          <span>{partner}</span>
                          <div className="flex items-center gap-3">
                            <span className="text-muted-foreground text-xs">
                              {formatNumber(p.games)}g
                            </span>
                            <span
                              className={cn(
                                'font-semibold w-14 text-right',
                                getWinRateColor(p.winRate)
                              )}
                            >
                              {formatPercent(p.winRate)}
                            </span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-2">
                  Strong Against (Counters)
                </h4>
                {counters.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No data</p>
                ) : (
                  <div className="space-y-1">
                    {counters.map((p) => {
                      const opponent =
                        p.heroA === heroName ? p.heroB : p.heroA
                      return (
                        <div
                          key={opponent}
                          className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-accent/30 text-sm"
                        >
                          <span>{opponent}</span>
                          <div className="flex items-center gap-3">
                            <span className="text-muted-foreground text-xs">
                              {formatNumber(p.games)}g
                            </span>
                            <span
                              className={cn(
                                'font-semibold w-14 text-right',
                                getWinRateColor(p.winRate)
                              )}
                            >
                              {formatPercent(p.winRate)}
                            </span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          {/* Personal Tab */}
          <TabsContent value="personal" className="mt-4 space-y-4">
            {personalStats.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No tracked battletags.
              </p>
            ) : (
              personalStats.map(({ battletag, stats }) => (
                <Card key={battletag}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">{battletag}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {stats ? (
                      <div className="space-y-3">
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                          <StatBlock
                            label="Games"
                            value={String(stats.games)}
                          />
                          <StatBlock
                            label="Win Rate"
                            value={formatPercent(stats.winRate)}
                            color={getWinRateColor(stats.winRate)}
                          />
                          <StatBlock
                            label="Momentum WR"
                            value={
                              stats.mawp != null
                                ? formatPercent(
                                    confidenceAdjustedMawp(
                                      stats.mawp,
                                      stats.games,
                                      30
                                    )
                                  )
                                : '-'
                            }
                            color={
                              stats.mawp != null
                                ? getWinRateColor(
                                    confidenceAdjustedMawp(
                                      stats.mawp,
                                      stats.games,
                                      30
                                    )
                                  )
                                : undefined
                            }
                            badge={
                              confidenceLabel(stats.games, 30) !== 'high'
                                ? confidenceLabel(stats.games, 30) === 'low'
                                  ? 'Low data'
                                  : 'Limited'
                                : undefined
                            }
                          />
                          <StatBlock
                            label="KDA"
                            value={`${stats.avgKills}/${stats.avgDeaths}/${stats.avgAssists}`}
                          />
                        </div>

                        {/* Recent matches for this hero */}
                        {(() => {
                          const btMatches = recentMatches.find(
                            (m) => m.battletag === battletag
                          )
                          const heroMatches =
                            btMatches?.matches.filter(
                              (m) => m.hero === heroName
                            ) ?? []
                          if (heroMatches.length === 0) return null
                          return (
                            <div>
                              <h5 className="text-xs font-medium text-muted-foreground mb-2">
                                Recent Games
                              </h5>
                              <div className="space-y-1">
                                {heroMatches.slice(0, 10).map((m) => (
                                  <div
                                    key={m.replayId}
                                    className="flex items-center justify-between text-xs py-1 px-2 rounded hover:bg-accent/30"
                                  >
                                    <div className="flex items-center gap-2">
                                      <span
                                        className={cn(
                                          'font-semibold w-6',
                                          m.win
                                            ? 'text-gaming-success'
                                            : 'text-gaming-danger'
                                        )}
                                      >
                                        {m.win ? 'W' : 'L'}
                                      </span>
                                      <span className="text-muted-foreground">
                                        {m.map}
                                      </span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                      <span>
                                        {m.kills}/{m.deaths}/{m.assists}
                                      </span>
                                      <span className="text-muted-foreground">
                                        {Math.floor(m.gameLength / 60)}m
                                      </span>
                                      <span className="text-muted-foreground">
                                        {m.gameDate.toLocaleDateString()}
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )
                        })()}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        No games on this hero.
                      </p>
                    )}
                  </CardContent>
                </Card>
              ))
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}

function StatBlock({
  label,
  value,
  color,
  badge,
}: {
  label: string
  value: string
  color?: string
  badge?: string
}) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <div className="flex items-center gap-1.5">
        <p className={cn('text-lg font-semibold', color)}>{value}</p>
        {badge && (
          <Badge
            variant="outline"
            className="text-[9px] px-1 py-0 border-gaming-warning/50 text-gaming-warning"
          >
            {badge}
          </Badge>
        )}
      </div>
    </div>
  )
}
