'use client'

import { useState, useMemo, useRef, useEffect } from 'react'
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
  | 'avgKills'
  | 'avgDeaths'
  | 'avgAssists'
  | 'avgHeroDamage'
  | 'avgDamageSoaked'
  | 'avgHealing'
  | 'avgSiegeDamage'
  | 'avgExperience'
  | 'avgMercCaptures'
  | 'avgSelfHealing'
  | 'avgTimeDead'

/** Columns that can be toggled on/off via the dropdown */
type OptionalColumn =
  | 'avgSiegeDamage'
  | 'avgExperience'
  | 'avgMercCaptures'
  | 'avgSelfHealing'
  | 'avgTimeDead'

const OPTIONAL_COLUMNS: { key: OptionalColumn; label: string }[] = [
  { key: 'avgSiegeDamage', label: 'Siege Damage' },
  { key: 'avgExperience', label: 'Experience' },
  { key: 'avgMercCaptures', label: 'Merc Captures' },
  { key: 'avgSelfHealing', label: 'Self Healing' },
  { key: 'avgTimeDead', label: 'Time Dead' },
]

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

function formatColumnValue(field: OptionalColumn | SortField, value: number): string {
  switch (field) {
    case 'avgSiegeDamage':
    case 'avgHeroDamage':
    case 'avgDamageSoaked':
    case 'avgSelfHealing':
    case 'avgExperience':
    case 'avgHealing':
      return formatNumber(value)
    case 'avgTimeDead':
      return `${value}s`
    case 'avgMercCaptures':
      return String(value)
    case 'winRate':
    case 'pickRate':
    case 'banRate':
      return formatPercent(value)
    case 'games':
      return formatNumber(value)
    default:
      return String(value)
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
  const [visibleOptional, setVisibleOptional] = useState<Set<OptionalColumn>>(new Set())
  const [columnsOpen, setColumnsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setColumnsOpen(false)
      }
    }
    if (columnsOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [columnsOpen])

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

  function toggleOptionalColumn(col: OptionalColumn) {
    setVisibleOptional((prev) => {
      const next = new Set(prev)
      if (next.has(col)) {
        next.delete(col)
      } else {
        next.add(col)
      }
      return next
    })
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

        {/* Column toggle dropdown */}
        <div className="relative ml-auto" ref={dropdownRef}>
          <button
            onClick={() => setColumnsOpen(!columnsOpen)}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors border',
              columnsOpen
                ? 'border-primary/50 text-primary bg-primary/10'
                : 'border-border text-muted-foreground hover:text-foreground hover:bg-accent'
            )}
          >
            <svg
              className="w-3.5 h-3.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7"
              />
            </svg>
            Columns
            {visibleOptional.size > 0 && (
              <span className="ml-1 px-1.5 py-0.5 rounded-full bg-primary/20 text-primary text-[10px] leading-none">
                +{visibleOptional.size}
              </span>
            )}
          </button>
          {columnsOpen && (
            <div className="absolute right-0 mt-1 w-48 bg-popover border rounded-lg shadow-lg z-50 py-1">
              <p className="px-3 py-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                Optional Columns
              </p>
              {OPTIONAL_COLUMNS.map((col) => (
                <label
                  key={col.key}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm hover:bg-accent/50 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={visibleOptional.has(col.key)}
                    onChange={() => toggleOptionalColumn(col.key)}
                    className="rounded border-border"
                  />
                  <span>{col.label}</span>
                </label>
              ))}
            </div>
          )}
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
              <SortHeader field="avgKills" className="text-right">
                K
              </SortHeader>
              <SortHeader field="avgDeaths" className="text-right">
                D
              </SortHeader>
              <SortHeader field="avgAssists" className="text-right">
                A
              </SortHeader>
              <SortHeader field="avgHeroDamage" className="text-right">
                Hero Dmg
              </SortHeader>
              <SortHeader field="avgDamageSoaked" className="text-right">
                Soak
              </SortHeader>
              <SortHeader field="avgHealing" className="text-right">
                Healing
              </SortHeader>
              {/* Optional columns */}
              {OPTIONAL_COLUMNS.filter((c) => visibleOptional.has(c.key)).map(
                (col) => (
                  <SortHeader
                    key={col.key}
                    field={col.key}
                    className="text-right"
                  >
                    {col.label}
                  </SortHeader>
                )
              )}
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
                  <td className="px-3 py-2.5 text-right">{hero.avgKills}</td>
                  <td className="px-3 py-2.5 text-right">{hero.avgDeaths}</td>
                  <td className="px-3 py-2.5 text-right">{hero.avgAssists}</td>
                  <td className="px-3 py-2.5 text-right text-muted-foreground">
                    {formatNumber(hero.avgHeroDamage)}
                  </td>
                  <td className="px-3 py-2.5 text-right text-muted-foreground">
                    {formatNumber(hero.avgDamageSoaked)}
                  </td>
                  <td className="px-3 py-2.5 text-right text-muted-foreground">
                    {formatNumber(hero.avgHealing)}
                  </td>
                  {/* Optional columns */}
                  {OPTIONAL_COLUMNS.filter((c) =>
                    visibleOptional.has(c.key)
                  ).map((col) => (
                    <td
                      key={col.key}
                      className="px-3 py-2.5 text-right text-muted-foreground"
                    >
                      {formatColumnValue(
                        col.key,
                        hero[col.key as keyof HeroStats] as number
                      )}
                    </td>
                  ))}
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
