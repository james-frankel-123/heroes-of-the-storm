'use client'

import { useState, useMemo } from 'react'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { HERO_ROLES, getHeroRole, type HeroRole } from '@/lib/data/hero-roles'
import { cn } from '@/lib/utils'

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

const ALL_HEROES = Object.keys(HERO_ROLES)

const ROLES: (HeroRole | 'All')[] = [
  'All',
  'Tank',
  'Bruiser',
  'Melee Assassin',
  'Ranged Assassin',
  'Healer',
  'Support',
]

interface HeroPickerProps {
  unavailable: Set<string>
  onSelect: (hero: string) => void
  currentStepType: 'ban' | 'pick'
  isOurTurn: boolean
}

export function HeroPicker({
  unavailable,
  onSelect,
  currentStepType,
  isOurTurn,
}: HeroPickerProps) {
  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<HeroRole | 'All'>('All')

  const heroes = useMemo(() => {
    let result = ALL_HEROES

    if (search) {
      const q = search.toLowerCase()
      result = result.filter((h) => h.toLowerCase().includes(q))
    }

    if (roleFilter !== 'All') {
      result = result.filter((h) => getHeroRole(h) === roleFilter)
    }

    return result.sort((a, b) => a.localeCompare(b))
  }, [search, roleFilter])

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">
          {currentStepType === 'ban'
            ? isOurTurn
              ? 'Select a hero to ban'
              : 'Select enemy ban'
            : isOurTurn
              ? 'Pick a hero for your team'
              : 'Select enemy pick'}
        </h3>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-2">
        <Input
          placeholder="Search heroes..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-xs h-8 text-sm"
          autoFocus
        />
        <div className="flex flex-wrap gap-1">
          {ROLES.map((role) => (
            <button
              key={role}
              onClick={() => setRoleFilter(role)}
              className={cn(
                'px-2 py-0.5 rounded text-[11px] font-medium transition-colors',
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

      {/* Hero grid */}
      <div className="grid grid-cols-4 sm:grid-cols-5 md:grid-cols-6 lg:grid-cols-7 gap-1.5 max-h-[400px] overflow-y-auto pr-1">
        {heroes.map((hero) => {
          const role = getHeroRole(hero)
          const isUnavailable = unavailable.has(hero)

          return (
            <button
              key={hero}
              onClick={() => !isUnavailable && onSelect(hero)}
              disabled={isUnavailable}
              className={cn(
                'flex flex-col items-center gap-0.5 p-2 rounded-md border text-xs transition-colors',
                isUnavailable
                  ? 'opacity-25 cursor-not-allowed border-border/50 bg-muted/20'
                  : currentStepType === 'ban'
                    ? 'border-border hover:border-gaming-danger/60 hover:bg-gaming-danger/10 cursor-pointer'
                    : 'border-border hover:border-primary/60 hover:bg-primary/10 cursor-pointer'
              )}
            >
              <span
                className={cn(
                  'font-medium truncate w-full text-center text-[11px] leading-tight',
                  isUnavailable && 'line-through'
                )}
              >
                {hero}
              </span>
              {role && (
                <Badge
                  variant={roleBadgeVariant(role)}
                  className="text-[7px] px-1 py-0"
                >
                  {role.split(' ')[0]}
                </Badge>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
