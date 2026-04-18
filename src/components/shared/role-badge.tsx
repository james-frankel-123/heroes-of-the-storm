'use client'

import { Badge } from '@/components/ui/badge'
import { RoleIcon } from './role-icon'
import type { HeroRole } from '@/lib/data/hero-roles'

function roleBadgeVariant(role: string) {
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

interface RoleBadgeProps {
  role: HeroRole
  className?: string
  /** Show abbreviated text (first word only) */
  short?: boolean
}

export function RoleBadge({ role, className, short }: RoleBadgeProps) {
  return (
    <Badge variant={roleBadgeVariant(role)} className={className}>
      <RoleIcon role={role} size={10} className="shrink-0 opacity-80" />
      <span className="ml-0.5">{short ? role.split(' ')[0] : role}</span>
    </Badge>
  )
}
