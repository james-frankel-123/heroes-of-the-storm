'use client'

import { DRAFT_SEQUENCE } from '@/lib/draft/types'
import type { DraftState } from '@/lib/draft/types'
import { getHeroRole } from '@/lib/data/hero-roles'
import { heroImageSrc } from '@/lib/data/hero-images'
import { RoleBadge } from '@/components/shared/role-badge'
import { cn } from '@/lib/utils'
import { compactLabel } from '@/lib/draft/assignments'

interface DraftBoardProps {
  state: DraftState
  currentStep: number
  /** Who? options per team: party tags and matched teammates; matched enemies */
  assignOptions: { ours: { value: string; label: string }[]; theirs: { value: string; label: string }[] }
  /** Display names for assignment values that are not BattleTags (matched players) */
  assignLabels: Record<string, string>
  /** Assignments to players no longer matched in the lobby (not used by the scorer) */
  staleAssignLabels: Record<string, string>
  /** All battletags assigned so far: stepIndex → battletag */
  playerAssignments: Record<number, string>
  onAssignPlayer: (stepIndex: number, battletag: string) => void
  /** Running win % for Team A (null if no picks yet) */
  teamAWinPct: number | null
  /** Running win % for Team B (null if no picks yet) */
  teamBWinPct: number | null
}

export function DraftBoard({
  state,
  currentStep,
  assignOptions,
  assignLabels,
  staleAssignLabels,
  playerAssignments,
  onAssignPlayer,
  teamAWinPct,
  teamBWinPct,
}: DraftBoardProps) {
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
        options={isOurTeam('A') ? assignOptions.ours : assignOptions.theirs}
        heroAt={state.selections}
        bans={teamABans}
        picks={teamAPicks}
        currentStep={currentStep}
        assignLabels={assignLabels}
        staleAssignLabels={staleAssignLabels}
        playerAssignments={playerAssignments}
        onAssignPlayer={onAssignPlayer}
        winPct={teamAWinPct}
      />

      {/* Divider */}
      <div className="border-t border-dashed" />

      {/* Team B row */}
      <TeamRow
        teamLabel={`Team B${isOurTeam('B') ? ' (You)' : ''}`}
        isOurs={isOurTeam('B')}
        options={isOurTeam('B') ? assignOptions.ours : assignOptions.theirs}
        heroAt={state.selections}
        bans={teamBBans}
        picks={teamBPicks}
        currentStep={currentStep}
        assignLabels={assignLabels}
        staleAssignLabels={staleAssignLabels}
        playerAssignments={playerAssignments}
        onAssignPlayer={onAssignPlayer}
        winPct={teamBWinPct}
      />
    </div>
  )
}

function winPctColor(pct: number): string {
  if (pct > 53) return 'text-gaming-success bg-gaming-success/10 border-gaming-success/30'
  if (pct < 47) return 'text-gaming-danger bg-gaming-danger/10 border-gaming-danger/30'
  return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30'
}

function TeamRow({
  teamLabel,
  isOurs,
  options,
  heroAt,
  bans,
  picks,
  currentStep,
  assignLabels,
  staleAssignLabels,
  playerAssignments,
  onAssignPlayer,
  winPct,
}: {
  teamLabel: string
  isOurs: boolean
  options: { value: string; label: string }[]
  heroAt: Record<number, string>
  bans: { stepIdx: number; hero: string | null }[]
  picks: { stepIdx: number; hero: string | null }[]
  currentStep: number
  assignLabels: Record<string, string>
  staleAssignLabels: Record<string, string>
  playerAssignments: Record<number, string>
  onAssignPlayer: (stepIndex: number, battletag: string) => void
  winPct: number | null
}) {
  // where each player is assigned now, so an option can say "(on Valla)"; choosing it moves them
  const placedOn = new Map<string, number>()
  for (const [k, v] of Object.entries(playerAssignments)) placedOn.set(v.toLowerCase(), Number(k))
  const labelOf = (v: string) => staleAssignLabels[v] ?? assignLabels[v] ?? v.replace(/^\d+:/, '').split('#')[0]
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <p
          className={cn(
            'text-xs font-semibold uppercase tracking-wider',
            isOurs ? 'text-primary' : 'text-muted-foreground'
          )}
        >
          {teamLabel}
        </p>
        {winPct !== null && (
          <span
            className={cn(
              'text-[10px] font-bold px-1.5 py-0.5 rounded border tabular-nums',
              winPctColor(winPct)
            )}
          >
            {winPct.toFixed(1)}%
          </span>
        )}
      </div>
      <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:gap-4">
        {/* Bans */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] text-muted-foreground uppercase mr-1">
            Bans
          </span>
          {bans.map(({ stepIdx, hero }) => (
            <Slot
              key={stepIdx}
              hero={hero}
              isCurrent={stepIdx === currentStep}
              isBan
              isSkipped={!hero && stepIdx < currentStep}
            />
          ))}
        </div>

        {/* Separator */}
        <div className="hidden sm:block w-px h-8 bg-border" />

        {/* Picks */}
        <div className="flex flex-wrap items-start gap-1.5">
          <span className="text-[10px] text-muted-foreground uppercase mr-1 self-center">
            Picks
          </span>
          {picks.map(({ stepIdx, hero }) => (
            <div key={stepIdx} className="flex flex-col items-center gap-0.5">
              <Slot
                hero={hero}
                isCurrent={stepIdx === currentStep}
                isBan={false}
              />
              {/* Player assignment: our picks always; enemy picks once enemies are matched */}
              {hero && (isOurs || options.length > 0 || playerAssignments[stepIdx]) && (
                <select
                  aria-label={`Player on ${hero}`}
                  title={playerAssignments[stepIdx] ? labelOf(playerAssignments[stepIdx]) : undefined}
                  value={playerAssignments[stepIdx] ?? ''}
                  onChange={(e) => {
                    onAssignPlayer(stepIdx, e.target.value || '')
                  }}
                  className={cn(
                    'w-full max-w-[100px] h-5 px-1 text-[9px] rounded border bg-background truncate',
                    playerAssignments[stepIdx]
                      ? playerAssignments[stepIdx] in staleAssignLabels
                        ? 'border-yellow-500/40 text-yellow-400'
                        : 'border-purple-500/40 text-purple-400'
                      : 'border-border/50 text-muted-foreground'
                  )}
                >
                  <option value="">Who?</option>
                  {/* Show currently assigned battletag + available ones */}
                  {playerAssignments[stepIdx] && (
                    <option value={playerAssignments[stepIdx]}>
                      {/* the closed control shows this: compact so region and games stay readable */}
                      {compactLabel(labelOf(playerAssignments[stepIdx]))}
                    </option>
                  )}
                  {options
                    .filter((o) => o.value.toLowerCase() !== playerAssignments[stepIdx]?.toLowerCase())
                    .map((o) => {
                      const on = placedOn.get(o.value.toLowerCase())
                      return (
                        <option key={o.value} value={o.value}>
                          {o.label}
                          {on !== undefined && heroAt[on] ? ` (on ${heroAt[on]})` : ''}
                        </option>
                      )
                    })}
                </select>
              )}
            </div>
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
  isSkipped = false,
}: {
  hero: string | null
  isCurrent: boolean
  isBan: boolean
  isSkipped?: boolean
}) {
  const role = hero ? getHeroRole(hero) : null

  return (
    <div
      className={cn(
        'flex items-center justify-center rounded-md border text-xs font-medium transition-all',
        isBan ? 'min-w-[72px] sm:min-w-[80px] h-8' : 'min-w-[88px] sm:min-w-[100px] h-10',
        hero
          ? isBan
            ? 'border-gaming-danger/40 bg-gaming-danger/10 text-gaming-danger line-through'
            : 'border-border bg-accent/30 text-foreground'
          : isSkipped
            ? 'border-yellow-500/30 bg-yellow-500/5 text-yellow-500/60'
            : isCurrent
              ? 'border-primary border-dashed bg-primary/5 animate-pulse'
              : 'border-border/50 bg-muted/30 text-muted-foreground/50'
      )}
    >
      {hero ? (
        <div className="flex items-center gap-1 px-2 truncate">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={heroImageSrc(hero)}
            alt=""
            loading="lazy"
            className={cn('rounded shrink-0 object-cover', isBan ? 'w-5 h-5 opacity-60' : 'w-6 h-6')}
          />
          <span className="truncate">{hero}</span>
          {role && !isBan && (
            <RoleBadge role={role!} className="text-[7px] px-1 py-0 shrink-0" short />
          )}
        </div>
      ) : isSkipped ? (
        <span className="text-[10px] italic">Missed</span>
      ) : isCurrent ? (
        <span className="text-primary text-[10px]">Select...</span>
      ) : (
        <span className="text-[10px]">&mdash;</span>
      )}
    </div>
  )
}
