'use client'

import * as React from 'react'
import { ArrowUpDown, Sparkles } from 'lucide-react'
import { HeroKDA } from '@/lib/data/statistics'
import { ContextData } from './clickable-metric'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface HeroAnalyticsProps {
  kdaByHero: HeroKDA[]
  onRowClick: (context: ContextData) => void
}

type SortField = 'hero' | 'games' | 'winRate' | 'kda' | 'kills' | 'deaths' | 'assists'
type SortDirection = 'asc' | 'desc'

export function HeroAnalytics({ kdaByHero, onRowClick }: HeroAnalyticsProps) {
  const [sortField, setSortField] = React.useState<SortField>('games')
  const [sortDirection, setSortDirection] = React.useState<SortDirection>('desc')

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('desc')
    }
  }

  const sortedData = React.useMemo(() => {
    return [...kdaByHero].sort((a, b) => {
      let aValue: number | string = 0
      let bValue: number | string = 0

      switch (sortField) {
        case 'hero':
          aValue = a.hero
          bValue = b.hero
          break
        case 'games':
          aValue = a.games
          bValue = b.games
          break
        case 'winRate':
          aValue = a.winRate
          bValue = b.winRate
          break
        case 'kda':
          aValue = a.kda
          bValue = b.kda
          break
        case 'kills':
          aValue = a.avgKills
          bValue = b.avgKills
          break
        case 'deaths':
          aValue = a.avgDeaths
          bValue = b.avgDeaths
          break
        case 'assists':
          aValue = a.avgAssists
          bValue = b.avgAssists
          break
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortDirection === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue)
      }

      return sortDirection === 'asc' ? (aValue as number) - (bValue as number) : (bValue as number) - (aValue as number)
    })
  }, [kdaByHero, sortField, sortDirection])

  const handleRowClick = (hero: HeroKDA) => {
    onRowClick({
      type: 'hero-row',
      label: `${hero.hero} Performance`,
      value: hero.winRate,
      relatedMetrics: {
        games: hero.games,
        winRate: hero.winRate,
        wins: hero.wins,
        losses: hero.losses,
        kda: hero.kda,
        avgKills: hero.avgKills,
        avgDeaths: hero.avgDeaths,
        avgAssists: hero.avgAssists,
      },
      hero: hero.hero,
    })
  }

  const SortButton = ({ field, label }: { field: SortField; label: string }) => (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => handleSort(field)}
      className="h-auto p-0 hover:bg-transparent"
    >
      <span className="flex items-center gap-1">
        {label}
        <ArrowUpDown className="h-3 w-3" />
      </span>
    </Button>
  )

  return (
    <Card className="p-6 glass-card">
      <div className="mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          Hero Performance Analytics
          <Sparkles className="h-4 w-4 text-primary-500/50" />
        </h3>
        <p className="text-sm text-muted-foreground">
          Click any row to analyze hero-specific performance
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-3 px-2 font-medium text-muted-foreground">
                <SortButton field="hero" label="Hero" />
              </th>
              <th className="text-center py-3 px-2 font-medium text-muted-foreground">
                <SortButton field="games" label="Games" />
              </th>
              <th className="text-center py-3 px-2 font-medium text-muted-foreground">
                <SortButton field="winRate" label="Win Rate" />
              </th>
              <th className="text-center py-3 px-2 font-medium text-muted-foreground">
                <SortButton field="kda" label="KDA" />
              </th>
              <th className="text-center py-3 px-2 font-medium text-muted-foreground">
                <SortButton field="kills" label="K" />
              </th>
              <th className="text-center py-3 px-2 font-medium text-muted-foreground">
                <SortButton field="deaths" label="D" />
              </th>
              <th className="text-center py-3 px-2 font-medium text-muted-foreground">
                <SortButton field="assists" label="A" />
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedData.map((hero, index) => (
              <tr
                key={hero.hero}
                onClick={() => handleRowClick(hero)}
                className="border-b border-border/50 hover:bg-primary-500/5 cursor-pointer transition-colors group"
              >
                <td className="py-3 px-2 font-medium">
                  <div className="flex items-center gap-2">
                    {hero.hero}
                    <Sparkles className="h-3 w-3 text-primary-500/0 group-hover:text-primary-500/70 transition-colors" />
                  </div>
                </td>
                <td className="text-center py-3 px-2 text-muted-foreground">{hero.games}</td>
                <td className="text-center py-3 px-2">
                  <span
                    className={
                      hero.winRate >= 55
                        ? 'text-green-500 font-semibold'
                        : hero.winRate <= 45
                        ? 'text-red-500'
                        : 'text-muted-foreground'
                    }
                  >
                    {hero.winRate.toFixed(1)}%
                  </span>
                </td>
                <td className="text-center py-3 px-2">
                  <span
                    className={
                      hero.kda >= 3
                        ? 'text-green-500 font-semibold'
                        : hero.kda <= 2
                        ? 'text-red-500'
                        : 'text-muted-foreground'
                    }
                  >
                    {hero.kda.toFixed(2)}
                  </span>
                </td>
                <td className="text-center py-3 px-2 text-muted-foreground">{hero.avgKills.toFixed(1)}</td>
                <td className="text-center py-3 px-2 text-muted-foreground">{hero.avgDeaths.toFixed(1)}</td>
                <td className="text-center py-3 px-2 text-muted-foreground">{hero.avgAssists.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {sortedData.length === 0 && (
          <div className="text-center py-12 text-muted-foreground">
            No hero data available
          </div>
        )}
      </div>
    </Card>
  )
}
