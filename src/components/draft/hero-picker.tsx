'use client'

import { useState, useMemo } from 'react'
import { Input } from '@/components/ui/input'
import { RoleBadge } from '@/components/shared/role-badge'
import { HERO_ROLES, getHeroRole, type HeroRole } from '@/lib/data/hero-roles'
import { heroImageSrc } from '@/lib/data/hero-images'
import { cn } from '@/lib/utils'
import { HEX_CLIP, METALLIC_FRAME } from './hex/constants'

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
    <div
      className="space-y-3 rounded-sm p-3 border border-[#3a4050]"
      style={{ background: 'rgba(15, 20, 48, 0.6)' }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm tracking-[0.2em] text-[#d6dbe0] font-light">
          {currentStepType === 'ban'
            ? isOurTurn
              ? 'SELECT A HERO TO BAN'
              : 'SELECT ENEMY BAN'
            : isOurTurn
              ? 'PICK A HERO FOR YOUR TEAM'
              : 'SELECT ENEMY PICK'}
        </h3>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-2">
        <Input
          placeholder="Search heroes..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-xs h-8 text-sm bg-[#0a0d1f]/80 border-[#3a4050] text-[#d6dbe0] placeholder:text-[#6b7078]"
          autoFocus
        />
        <div className="flex flex-wrap gap-1">
          {ROLES.map((role) => (
            <button
              key={role}
              onClick={() => setRoleFilter(role)}
              className={cn(
                'px-2 py-0.5 rounded text-[11px] font-medium transition-colors border',
                roleFilter === role
                  ? 'bg-[#d6dbe0]/10 text-[#e8ecef] border-[#d6dbe0]/40'
                  : 'text-[#8b9bc8] border-transparent hover:text-[#d6dbe0] hover:bg-[#3a4050]/40'
              )}
            >
              {role}
            </button>
          ))}
        </div>
      </div>

      {/* Hero grid — hex thumbs */}
      <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-9 lg:grid-cols-10 gap-2 max-h-[400px] overflow-y-auto pr-1">
        {heroes.map((hero) => {
          const role = getHeroRole(hero)
          const isUnavailable = unavailable.has(hero)

          return (
            <button
              key={hero}
              onClick={() => {
                if (isUnavailable) return
                setSearch('')
                onSelect(hero)
              }}
              disabled={isUnavailable}
              className={cn(
                'flex flex-col items-center gap-0.5 p-1 text-xs transition-all group',
                isUnavailable
                  ? 'opacity-30 cursor-not-allowed'
                  : 'cursor-pointer hover:scale-[1.05]'
              )}
            >
              <div className="relative w-11 h-11">
                {/* Steel frame */}
                <div
                  className="absolute inset-0"
                  style={{ clipPath: HEX_CLIP, background: METALLIC_FRAME }}
                />
                {/* Portrait */}
                <div
                  className="absolute inset-[2px] bg-[#0a0d1f] overflow-hidden"
                  style={{ clipPath: HEX_CLIP }}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={heroImageSrc(hero)}
                    alt=""
                    loading="lazy"
                    className={cn(
                      'w-full h-full object-cover',
                      isUnavailable && 'grayscale',
                      !isUnavailable && 'group-hover:brightness-125'
                    )}
                  />
                </div>
              </div>
              <span
                className={cn(
                  'font-medium truncate w-full text-center text-[10px] leading-tight',
                  isUnavailable ? 'line-through text-[#6b7078]' : 'text-[#d6dbe0]'
                )}
              >
                {hero}
              </span>
              {role && (
                <RoleBadge role={role!} className="text-[7px] px-1 py-0 opacity-80" short />
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
