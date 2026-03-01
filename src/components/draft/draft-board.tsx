'use client'

import { DRAFT_SEQUENCE } from '@/lib/draft/types'
import type { DraftState } from '@/lib/draft/types'
import { getHeroRole } from '@/lib/data/hero-roles'
import { Badge } from '@/components/ui/badge'
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

interface DraftBoardProps {
  state: DraftState
  currentStep: number
}

export function DraftBoard({ state, currentStep }: DraftBoardProps) {
  // Separate into Team A and Team B rows
  const teamABans: { stepIdx: number; hero: string | null }[] = []
  const teamBBans: { stepIdx: number; hero: string | null }[] = []
  const teamAPicks: { stepIdx: number; hero: string | null }[] = []
  const teamBPicks: { stepIdx: number; hero: string | null }[] = []

  DRAFT_SEQUENCE.forEach((step, idx) => {
    const hero = state.selections[idx] ?? null
    const entry = { stepIdx: idx, hero }
    if (step.type === 'ban') {
      if (step.team === 'A') teamABans.push(entry)
      else teamBBans.push(entry)
    } else {
      if (step.team === 'A') teamAPicks.push(entry)
      else teamBPicks.push(entry)
    }
  })

  const isOurTeam = (team: 'A' | 'B') => team === state.ourTeam

  return (
    <div className="rounded-lg border p-4 space-y-3">
      {/* Team A row */}
      <TeamRow
        teamLabel={`Team A${isOurTeam('A') ? ' (You)' : ''}`}
        isOurs={isOurTeam('A')}
        bans={teamABans}
        picks={teamAPicks}
        currentStep={currentStep}
      />

      {/* Divider */}
      <div className="border-t border-dashed" />

      {/* Team B row */}
      <TeamRow
        teamLabel={`Team B${isOurTeam('B') ? ' (You)' : ''}`}
        isOurs={isOurTeam('B')}
        bans={teamBBans}
        picks={teamBPicks}
        currentStep={currentStep}
      />
    </div>
  )
}

function TeamRow({
  teamLabel,
  isOurs,
  bans,
  picks,
  currentStep,
}: {
  teamLabel: string
  isOurs: boolean
  bans: { stepIdx: number; hero: string | null }[]
  picks: { stepIdx: number; hero: string | null }[]
  currentStep: number
}) {
  return (
    <div className="space-y-2">
      <p
        className={cn(
          'text-xs font-semibold uppercase tracking-wider',
          isOurs ? 'text-primary' : 'text-muted-foreground'
        )}
      >
        {teamLabel}
      </p>
      <div className="flex items-center gap-4">
        {/* Bans */}
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-muted-foreground uppercase mr-1">
            Bans
          </span>
          {bans.map(({ stepIdx, hero }) => (
            <Slot
              key={stepIdx}
              hero={hero}
              isCurrent={stepIdx === currentStep}
              isBan
            />
          ))}
        </div>

        {/* Separator */}
        <div className="w-px h-8 bg-border" />

        {/* Picks */}
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-muted-foreground uppercase mr-1">
            Picks
          </span>
          {picks.map(({ stepIdx, hero }) => (
            <Slot
              key={stepIdx}
              hero={hero}
              isCurrent={stepIdx === currentStep}
              isBan={false}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function Slot({
  hero,
  isCurrent,
  isBan,
}: {
  hero: string | null
  isCurrent: boolean
  isBan: boolean
}) {
  const role = hero ? getHeroRole(hero) : null

  return (
    <div
      className={cn(
        'flex items-center justify-center rounded-md border text-xs font-medium transition-all',
        isBan ? 'min-w-[80px] h-8' : 'min-w-[100px] h-10',
        hero
          ? isBan
            ? 'border-gaming-danger/40 bg-gaming-danger/10 text-gaming-danger line-through'
            : 'border-border bg-accent/30 text-foreground'
          : isCurrent
            ? 'border-primary border-dashed bg-primary/5 animate-pulse'
            : 'border-border/50 bg-muted/30 text-muted-foreground/50'
      )}
    >
      {hero ? (
        <div className="flex items-center gap-1 px-2 truncate">
          <span className="truncate">{hero}</span>
          {role && !isBan && (
            <Badge
              variant={roleBadgeVariant(role)}
              className="text-[7px] px-1 py-0 shrink-0"
            >
              {role.split(' ')[0]}
            </Badge>
          )}
        </div>
      ) : isCurrent ? (
        <span className="text-primary text-[10px]">Select...</span>
      ) : (
        <span className="text-[10px]">&mdash;</span>
      )}
    </div>
  )
}
