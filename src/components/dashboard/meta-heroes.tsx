'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { formatPercent, getWinRateColor, formatNumber } from '@/lib/utils'
import type { HeroStats } from '@/lib/types'

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

interface MetaHeroesProps {
  topHeroes: HeroStats[]
  bottomHeroes: HeroStats[]
}

export function MetaHeroes({ topHeroes, bottomHeroes }: MetaHeroesProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <span className="text-gaming-success">&#9650;</span>
            Overperforming Heroes
          </CardTitle>
        </CardHeader>
        <CardContent>
          <HeroList heroes={topHeroes} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <span className="text-gaming-danger">&#9660;</span>
            Underperforming Heroes
          </CardTitle>
        </CardHeader>
        <CardContent>
          <HeroList heroes={bottomHeroes} />
        </CardContent>
      </Card>
    </div>
  )
}

function HeroList({ heroes }: { heroes: HeroStats[] }) {
  return (
    <div className="space-y-2">
      {heroes.map((hero, i) => {
        const role = getHeroRole(hero.hero)
        return (
          <div
            key={hero.hero}
            className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-accent/50 transition-colors"
          >
            <div className="flex items-center gap-3 min-w-0">
              <span className="text-xs text-muted-foreground w-5 text-right">
                {i + 1}
              </span>
              <span className="font-medium text-sm truncate">{hero.hero}</span>
              {role && (
                <Badge variant={roleBadgeVariant(role)} className="text-xs px-1.5 py-0">
                  {role}
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-4 text-sm shrink-0">
              <span className="text-muted-foreground">
                {formatNumber(hero.games)} games
              </span>
              <span className={`font-semibold w-14 text-right ${getWinRateColor(hero.winRate)}`}>
                {formatPercent(hero.winRate)}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
