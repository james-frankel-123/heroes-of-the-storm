'use client'

import * as React from 'react'
import { Badge } from '@/components/ui/badge'
import { formatPercent, getWinRateColor, getRoleColor } from '@/lib/utils'

interface Hero {
  hero: string
  role: string
  games: number
  wins: number
  losses: number
  winRate: number
}

interface HeroTableProps {
  heroes: Hero[]
}

const medalEmojis = ['🥇', '🥈', '🥉']

export function HeroTable({ heroes }: HeroTableProps) {
  return (
    <div className="overflow-hidden rounded-lg border border-border/50">
      <table className="w-full">
        <thead>
          <tr className="border-b border-border/50 bg-muted/50">
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Rank
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Hero
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Role
            </th>
            <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Games
            </th>
            <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Record
            </th>
            <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Win Rate
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/50">
          {heroes.map((hero, index) => (
            <tr
              key={hero.hero}
              className="group transition-colors hover:bg-primary-500/5"
            >
              <td className="px-4 py-3">
                <span className="text-lg">
                  {index < 3 ? medalEmojis[index] : `${index + 1}.`}
                </span>
              </td>
              <td className="px-4 py-3">
                <span className="font-semibold">{hero.hero}</span>
              </td>
              <td className="px-4 py-3">
                <Badge
                  variant={hero.role.toLowerCase().replace(' ', '') as any}
                  className="text-xs"
                >
                  {hero.role}
                </Badge>
              </td>
              <td className="px-4 py-3 text-right text-sm text-muted-foreground">
                {hero.games}
              </td>
              <td className="px-4 py-3 text-right text-sm">
                {hero.wins}-{hero.losses}
              </td>
              <td className="px-4 py-3 text-right">
                <span
                  className={`font-bold ${getWinRateColor(hero.winRate)}`}
                >
                  {formatPercent(hero.winRate, 2)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
