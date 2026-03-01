'use client'

import { cn } from '@/lib/utils'
import type { SkillTier } from '@/lib/types'

const TIER_CONFIG: { value: SkillTier; label: string; ranks: string }[] = [
  { value: 'low', label: 'Low', ranks: 'Bronze + Silver' },
  { value: 'mid', label: 'Mid', ranks: 'Gold + Plat' },
  { value: 'high', label: 'High', ranks: 'Diamond + Master' },
]

interface TierSelectorProps {
  value: SkillTier
  onChange: (tier: SkillTier) => void
  className?: string
}

export function TierSelector({ value, onChange, className }: TierSelectorProps) {
  return (
    <div className={cn('flex items-center gap-1 p-1 rounded-lg bg-muted', className)}>
      {TIER_CONFIG.map((tier) => (
        <button
          key={tier.value}
          onClick={() => onChange(tier.value)}
          className={cn(
            'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
            value === tier.value
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          )}
          title={tier.ranks}
        >
          {tier.label}
        </button>
      ))}
    </div>
  )
}

export function getTierLabel(tier: SkillTier): string {
  return TIER_CONFIG.find((t) => t.value === tier)?.ranks ?? tier
}
