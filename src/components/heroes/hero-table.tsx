'use client'

import { useState, useMemo } from 'react'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { getHeroRole, type HeroRole } from '@/lib/data/hero-roles'
import {
  cn,
  formatPercent,
  formatNumber,
  getWinRateColor,
} from '@/lib/utils'
import type { HeroStats } from '@/lib/types'

type SortField =
  | 'hero'
  | 'winRate'
  | 'games'
  | 'pickRate'
  | 'banRate'

const ROLES: (HeroRole | 'All')[] = [
  'All',
  'Tank',
  'Bruiser',
  'Melee Assassin',
  'Ranged Assassin',
  'Healer',
  'Support',
]

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

interface HeroTableProps {
  heroes: HeroStats[]
  onHeroClick: (hero: string) => void
}

export function HeroTable({ heroes, onHeroClick }: HeroTableProps) {
  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<HeroRole | 'All'>('All')
  const [sortField, setSortField] = useState<SortField>('winRate')
  const [sortAsc, setSortAsc] = useState(false)
  const filtered = useMemo(() => {
    let result = heroes

    if (search) {
      const q = search.toLowerCase()
      result = result.filter((h) => h.hero.toLowerCase().includes(q))
    }

    if (roleFilter !== 'All') {
      result = result.filter((h) => getHeroRole(h.hero) === roleFilter)
    }

    result.sort((a, b) => {
      const aVal = a[sortField as keyof HeroStats]
      const bVal = b[sortField as keyof HeroStats]
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortAsc
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal)
      }
      const aNum = aVal as number
      const bNum = bVal as number
      return sortAsc ? aNum - bNum : bNum - aNum
    })

    return result
  }, [heroes, search, roleFilter, sortField, sortAsc])

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
        'px-3 py-2 text-xs font-medium text-muted-foreground cursor-pointer hover:text-foreground select-none whitespace-nowrap',
        className
      )}
      onClick={() => handleSort(field)}
    >
      <span className="inline-flex items-center gap-1">
        {children}
        {sortField === field && (
          <span className="text-primary">{sortAsc ? '\u25B2' : '\u25BC'}</span>
        )}
      </span>
    </th>
  )

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
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
                'px-2.5 py-1 rounded-md text-xs font-medium transition-colors',
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

      {/* Table */}
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
              <SortHeader field="pickRate" className="text-right">
                Pick %
              </SortHeader>
              <SortHeader field="banRate" className="text-right">
                Ban %
              </SortHeader>
            </tr>
          </thead>
          <tbody>
            {filtered.map((hero) => {
              const role = getHeroRole(hero.hero)
              return (
                <tr
                  key={hero.hero}
                  className="border-b last:border-0 hover:bg-accent/30 cursor-pointer transition-colors"
                  onClick={() => onHeroClick(hero.hero)}
                >
                  <td className="px-3 py-2.5 font-medium">{hero.hero}</td>
                  <td className="px-3 py-2.5">
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
                      'px-3 py-2.5 text-right font-semibold',
                      getWinRateColor(hero.winRate)
                    )}
                  >
                    {formatPercent(hero.winRate)}
                  </td>
                  <td className="px-3 py-2.5 text-right text-muted-foreground">
                    {formatNumber(hero.games)}
                  </td>
                  <td className="px-3 py-2.5 text-right text-muted-foreground">
                    {formatPercent(hero.pickRate)}
                  </td>
                  <td className="px-3 py-2.5 text-right text-muted-foreground">
                    {formatPercent(hero.banRate)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-muted-foreground">
        Showing {filtered.length} of {heroes.length} heroes. Click a row for
        details.
      </p>
    </div>
  )
}
