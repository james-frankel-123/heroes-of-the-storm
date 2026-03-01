'use client'

import { useState, useMemo } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { getHeroRole, type HeroRole } from '@/lib/data/hero-roles'
import {
  cn,
  formatPercent,
  formatNumber,
  getWinRateColor,
} from '@/lib/utils'
import type { HeroMapStats, PlayerMatch, SkillTier } from '@/lib/types'

function roleBadgeVariant(role: string | null) {
  switch (role) {
    case 'Tank': return 'tank' as const
    case 'Bruiser': return 'bruiser' as const
    case 'Healer': return 'healer' as const
    case 'Ranged Assassin': return 'ranged' as const
    case 'Melee Assassin': return 'melee' as const
    case 'Support': return 'support' as const
    default: return 'secondary' as const
  }
}

type SortField = 'hero' | 'winRate' | 'games'

const ROLES: (HeroRole | 'All')[] = [
  'All',
  'Tank',
  'Bruiser',
  'Melee Assassin',
  'Ranged Assassin',
  'Healer',
  'Support',
]

interface MapDetailModalProps {
  open: boolean
  onClose: () => void
  mapName: string
  heroStats: HeroMapStats[]
  personalStats: {
    battletag: string
    games: number
    wins: number
    winRate: number
  }[]
  personalMatches: {
    battletag: string
    matches: PlayerMatch[]
  }[]
  currentTier: SkillTier
}

export function MapDetailModal({
  open,
  onClose,
  mapName,
  heroStats,
  personalStats,
  personalMatches,
  currentTier,
}: MapDetailModalProps) {
  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<HeroRole | 'All'>('All')
  const [sortField, setSortField] = useState<SortField>('winRate')
  const [sortAsc, setSortAsc] = useState(false)

  const tierLabel =
    currentTier === 'low'
      ? 'Bronze+Silver'
      : currentTier === 'mid'
        ? 'Gold+Plat'
        : 'Diamond+Master'

  const filteredHeroes = useMemo(() => {
    let result = heroStats

    if (search) {
      const q = search.toLowerCase()
      result = result.filter((h) => h.hero.toLowerCase().includes(q))
    }

    if (roleFilter !== 'All') {
      result = result.filter((h) => getHeroRole(h.hero) === roleFilter)
    }

    result.sort((a, b) => {
      if (sortField === 'hero') {
        return sortAsc
          ? a.hero.localeCompare(b.hero)
          : b.hero.localeCompare(a.hero)
      }
      const aVal = a[sortField] as number
      const bVal = b[sortField] as number
      return sortAsc ? aVal - bVal : bVal - aVal
    })

    return result
  }, [heroStats, search, roleFilter, sortField, sortAsc])

  function handleSort(field: SortField) {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(field === 'hero')
    }
  }

  const SortHeader = ({
    field,
    children,
    className,
  }: {
    field: SortField
    children: React.ReactNode
    className?: string
  }) => (
    <th
      className={cn(
        'px-3 py-2 text-xs font-medium text-muted-foreground cursor-pointer hover:text-foreground select-none',
        className
      )}
      onClick={() => handleSort(field)}
    >
      <span className="inline-flex items-center gap-1">
        {children}
        {sortField === field && (
          <span className="text-primary">
            {sortAsc ? '\u25B2' : '\u25BC'}
          </span>
        )}
      </span>
    </th>
  )

  // Aggregate personal match data per hero per battletag on this map
  const personalHeroBreakdown = useMemo(() => {
    const result: {
      battletag: string
      heroes: { hero: string; games: number; wins: number; winRate: number }[]
    }[] = []

    for (const pm of personalMatches) {
      const byHero: Record<string, { games: number; wins: number }> = {}
      for (const m of pm.matches) {
        if (!byHero[m.hero]) byHero[m.hero] = { games: 0, wins: 0 }
        byHero[m.hero].games++
        if (m.win) byHero[m.hero].wins++
      }
      const heroes = Object.entries(byHero)
        .map(([hero, { games, wins }]) => ({
          hero,
          games,
          wins,
          winRate: Math.round((wins / games) * 1000) / 10,
        }))
        .sort((a, b) => b.games - a.games)

      if (heroes.length > 0) {
        result.push({ battletag: pm.battletag, heroes })
      }
    }

    return result
  }, [personalMatches])

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl">{mapName}</DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="heroes" className="mt-2">
          <TabsList className="w-full justify-start">
            <TabsTrigger value="heroes">Hero Win Rates</TabsTrigger>
            <TabsTrigger value="personal">Personal</TabsTrigger>
          </TabsList>

          {/* Heroes on this map */}
          <TabsContent value="heroes" className="mt-4 space-y-3">
            <p className="text-sm text-muted-foreground">
              Hero performance on {mapName} &mdash; {tierLabel}
            </p>

            {/* Filters */}
            <div className="flex flex-col sm:flex-row gap-3">
              <Input
                placeholder="Search heroes..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="max-w-xs"
              />
              <div className="flex flex-wrap gap-1">
                {ROLES.map((role) => (
                  <button
                    key={role}
                    onClick={() => setRoleFilter(role)}
                    className={cn(
                      'px-2 py-1 rounded-md text-xs font-medium transition-colors',
                      roleFilter === role
                        ? 'bg-primary/20 text-primary'
                        : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                    )}
                  >
                    {role}
                  </button>
                ))}
              </div>
            </div>

            {/* Hero table for this map */}
            <div className="rounded-lg border overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <SortHeader field="hero" className="text-left">
                      Hero
                    </SortHeader>
                    <th className="px-3 py-2 text-xs font-medium text-muted-foreground text-left">
                      Role
                    </th>
                    <SortHeader field="winRate" className="text-right">
                      Win %
                    </SortHeader>
                    <SortHeader field="games" className="text-right">
                      Games
                    </SortHeader>
                  </tr>
                </thead>
                <tbody>
                  {filteredHeroes.map((h) => {
                    const role = getHeroRole(h.hero)
                    return (
                      <tr
                        key={h.hero}
                        className="border-b last:border-0 hover:bg-accent/30"
                      >
                        <td className="px-3 py-2 font-medium">{h.hero}</td>
                        <td className="px-3 py-2">
                          {role && (
                            <Badge
                              variant={roleBadgeVariant(role)}
                              className="text-[10px] px-1.5 py-0"
                            >
                              {role}
                            </Badge>
                          )}
                        </td>
                        <td
                          className={cn(
                            'px-3 py-2 text-right font-semibold',
                            getWinRateColor(h.winRate)
                          )}
                        >
                          {formatPercent(h.winRate)}
                        </td>
                        <td className="px-3 py-2 text-right text-muted-foreground">
                          {formatNumber(h.games)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <p className="text-xs text-muted-foreground">
              Showing {filteredHeroes.length} of {heroStats.length} heroes
            </p>
          </TabsContent>

          {/* Personal stats on this map */}
          <TabsContent value="personal" className="mt-4 space-y-4">
            {personalStats.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No tracked battletags have games on this map.
              </p>
            ) : (
              <>
                {/* Overall map win rates */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {personalStats.map((ps) => (
                    <Card key={ps.battletag}>
                      <CardHeader className="pb-2 pt-4 px-4">
                        <CardTitle className="text-sm truncate">
                          {ps.battletag}
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="px-4 pb-4">
                        <div className="flex items-baseline gap-2">
                          <span
                            className={cn(
                              'text-2xl font-bold',
                              getWinRateColor(ps.winRate)
                            )}
                          >
                            {formatPercent(ps.winRate)}
                          </span>
                          <span className="text-sm text-muted-foreground">
                            ({ps.wins}W {ps.games - ps.wins}L)
                          </span>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                {/* Per-hero breakdown on this map */}
                {personalHeroBreakdown.map(({ battletag, heroes }) => (
                  <div key={battletag}>
                    <h4 className="text-sm font-medium text-muted-foreground mb-2">
                      {battletag} &mdash; Heroes on {mapName}
                    </h4>
                    <div className="rounded-lg border overflow-hidden">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b bg-muted/50">
                            <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                              Hero
                            </th>
                            <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">
                              Win %
                            </th>
                            <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">
                              W-L
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {heroes.map((h) => (
                            <tr
                              key={h.hero}
                              className="border-b last:border-0 hover:bg-accent/30"
                            >
                              <td className="px-3 py-1.5 font-medium">
                                {h.hero}
                              </td>
                              <td
                                className={cn(
                                  'px-3 py-1.5 text-right font-semibold',
                                  getWinRateColor(h.winRate)
                                )}
                              >
                                {formatPercent(h.winRate)}
                              </td>
                              <td className="px-3 py-1.5 text-right text-muted-foreground">
                                {h.wins}-{h.games - h.wins}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}

                {/* Recent matches on this map */}
                {personalMatches
                  .filter((pm) => pm.matches.length > 0)
                  .map(({ battletag, matches }) => (
                    <div key={battletag}>
                      <h4 className="text-sm font-medium text-muted-foreground mb-2">
                        {battletag} &mdash; Recent Games
                      </h4>
                      <div className="space-y-1">
                        {matches.slice(0, 15).map((m) => (
                          <div
                            key={m.replayId}
                            className="flex items-center justify-between text-xs py-1.5 px-2 rounded hover:bg-accent/30"
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
                              <span className="font-medium">{m.hero}</span>
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
                  ))}
              </>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
