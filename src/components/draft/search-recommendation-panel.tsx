'use client'

import { RoleBadge } from '@/components/shared/role-badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { heroImageSrc } from '@/lib/data/hero-images'
import { cn } from '@/lib/utils'
import { HEX_CLIP, METALLIC_FRAME } from './hex/constants'
import type { DraftData } from '@/lib/draft/types'
import { scorePlayerStrength } from '@/lib/draft/engine'

/**
 * A row shown on our turns.
 *
 * winPct comes from the partial-draft win-probability model, the same
 * neutral judge family that scores the finished draft, so row numbers and
 * the final evaluation share one scale and converge by construction
 * (2026-08-07: replaced the MCTS child-Q display, whose level measured as
 * a state-insensitive constant). Pick rows project the state after WE take
 * the hero. Ban rows project the state if THE ENEMY takes the hero (the
 * threat a ban denies), so for bans LOWER is a stronger ban and the list
 * sorts ascending. MCTS still selects the shortlist, contributes the
 * personalization nudges, and flags its top choice.
 */
export interface OurTurnRow {
  hero: string
  /** Projected final win % under the follow-the-AI prior (see above) */
  winPct: number
  /** Delta vs the current estimate, in percentage points */
  deltaPp: number
  /** True when the row comes from greedy padding rather than the search */
  isGreedyPad: boolean
  /** True for the MCTS search's own top choice (shown as a subtle badge) */
  isAiTop?: boolean
}

/** A row shown on opponent turns: how likely they take the hero + what it does to us. */
export interface OpponentPredictionRow {
  hero: string
  /** P(opponent takes this hero next), 0-100, from the opponent model. Null for stats-based padding rows. */
  probabilityPct: number | null
  /** Expected change to OUR displayed win % (percentage points) if they take it. Null when not computable (bans). */
  impactPp: number | null
}

interface SearchRecommendationPanelProps {
  /** Our-turn rows (MCTS results + greedy padding), precomputed by the caller */
  ourRows?: OurTurnRow[]
  /** Opponent-turn rows (used when !isOurTurn) */
  opponentRows?: OpponentPredictionRow[]
  /** Short info label shown when the search is done, e.g. "142 sims" */
  searchInfo?: string | null
  searching: boolean
  statusText?: string
  isBanPhase: boolean
  isOurTurn: boolean
  onSelect: (hero: string) => void
  unavailable: Set<string>
  /** Optional: enables the "which player should draft this" byline */
  draftData?: DraftData
  availableBattletags?: string[]
  /** Current map — enables map-specific (≥25-game) override in byline */
  map?: string | null
}

export function SearchRecommendationPanel({
  ourRows,
  opponentRows,
  searchInfo,
  searching,
  statusText,
  isBanPhase,
  isOurTurn,
  onSelect,
  unavailable,
  draftData,
  availableBattletags,
  map,
}: SearchRecommendationPanelProps) {
  const canShowPlayerByline =
    isOurTurn && !isBanPhase && !!draftData && !!availableBattletags && availableBattletags.length > 0
  const title = isBanPhase
    ? isOurTurn ? 'Ban Suggestions' : 'Likely Enemy Bans'
    : isOurTurn ? 'Search Recommendations' : 'Likely Enemy Picks'
  const subtitle = isOurTurn
    ? isBanPhase
      ? 'Your projected win chance if the enemy takes each hero (lower = more urgent ban); same model that scores the final draft'
      : 'Projected win chance after each pick, judged by the same model that scores the final draft'
    : isBanPhase
      ? 'How likely the enemy is to ban each hero'
      : 'How likely the enemy is to pick each hero, and what it would do to your win chance'

  const ourRowsShown = isOurTurn
    ? (ourRows ?? []).filter(r => !unavailable.has(r.hero)).slice(0, 10)
    : []
  const oppRowsShown = !isOurTurn
    ? (opponentRows ?? []).filter(r => !unavailable.has(r.hero)).slice(0, 10)
    : []
  const isEmpty = isOurTurn ? ourRowsShown.length === 0 : oppRowsShown.length === 0

  const rowButtonClass = (isGreedyPad: boolean) => cn(
    'w-full text-left px-3 py-2 rounded border transition-all',
    'hover:scale-[1.01] active:scale-[0.99]',
    isGreedyPad && 'opacity-60',
    isBanPhase
      ? 'border-[#3a2222] bg-[#0a0d1f]/40 hover:border-[#d46b6b]/60 hover:bg-[#d46b6b]/10'
      : 'border-[#3a4050] bg-[#0a0d1f]/40 hover:border-[#6b8dd4]/60 hover:bg-[#6b8dd4]/10'
  )

  const heroCell = (hero: string) => {
    const role = getHeroRole(hero)
    return (
      <div className="flex items-center gap-2 min-w-0">
        <div className="relative w-9 h-9 shrink-0">
          <div className="absolute inset-0" style={{ clipPath: HEX_CLIP, background: METALLIC_FRAME }} />
          <div className="absolute inset-[1.5px] bg-[#0a0d1f] overflow-hidden" style={{ clipPath: HEX_CLIP }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={heroImageSrc(hero)} alt="" loading="lazy" className="w-full h-full object-cover" />
          </div>
        </div>
        <span className="text-sm font-medium text-[#e8ecef] truncate">{hero}</span>
        {role && (
          <RoleBadge role={role!} className="text-[10px] px-1 py-0 shrink-0" />
        )}
      </div>
    )
  }

  const playerByline = (hero: string) => {
    const playerInfo = canShowPlayerByline
      ? scorePlayerStrength(hero, availableBattletags!, draftData!, map ?? null)
      : null
    if (!playerInfo?.reason) return null
    return (
      <p className="mt-1 text-[10px] text-purple-400">
        {playerInfo.reason.label}
        {playerInfo.player && (
          <span className="text-muted-foreground"> · {playerInfo.player.split('#')[0]} should play this</span>
        )}
      </p>
    )
  }

  return (
    <div
      className="space-y-2 rounded-sm p-3 border border-[#3a4050]"
      style={{ background: 'rgba(15, 20, 48, 0.6)' }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm tracking-[0.2em] text-[#d6dbe0] font-light">
          {title.toUpperCase()}
        </h3>
        {searching ? (
          <span className="text-xs text-[#6b8dd4] animate-pulse">
            {statusText || 'Searching...'}
          </span>
        ) : searchInfo ? (
          <span className="text-xs text-[#8b9bc8]">
            {searchInfo}
          </span>
        ) : null}
      </div>
      <p className="text-[10px] text-[#8b9bc8]">{subtitle}</p>

      {isEmpty ? (
        <div className="space-y-2">
          {searching && (
            <div className="w-full bg-muted/30 rounded-full h-1.5 overflow-hidden">
              <div className="bg-blue-500/60 h-full rounded-full animate-pulse" style={{ width: '60%' }} />
            </div>
          )}
          <p className="text-xs text-muted-foreground">
            {statusText || (searching ? 'Computing recommendations...' : 'Waiting for draft action...')}
          </p>
        </div>
      ) : (
        <div className="space-y-1 max-h-[450px] overflow-y-auto pr-1">
          {ourRowsShown.map((rec) => (
            <button
              key={rec.hero}
              className={rowButtonClass(rec.isGreedyPad)}
              onClick={() => onSelect(rec.hero)}
              data-testid="rec-row"
              data-hero={rec.hero}
              data-winpct={rec.winPct.toFixed(1)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 min-w-0">
                  {heroCell(rec.hero)}
                  {rec.isAiTop && (
                    <span
                      className="shrink-0 rounded border border-[#6b8dd4]/50 px-1 py-px text-[9px] font-medium tracking-wide text-[#6b8dd4]"
                      title="The AI search's top choice for this turn"
                      data-testid="ai-pick-badge"
                    >
                      AI pick
                    </span>
                  )}
                </div>
                <div className="text-right shrink-0">
                  <div
                    title={isBanPhase
                      ? `Your projected win chance if the enemy takes ${rec.hero}. Lower means the hero is a bigger threat and a stronger ban. Judged by the same model that evaluates the finished draft.`
                      : `Projected win chance after picking ${rec.hero}, judged by the same model that evaluates the finished draft, so this number and the final score converge as the draft completes.`}
                  >
                    <span className={cn(
                      'text-sm font-bold tabular-nums',
                      rec.deltaPp >= 3
                        ? 'text-[#6fd46f]'
                        : rec.deltaPp >= 0
                          ? 'text-[#d4b85a]'
                          : 'text-[#d46b6b]'
                    )}>
                      {rec.winPct.toFixed(1)}%
                    </span>
                    <span className="ml-1 text-[9px] text-[#8b9bc8]">{isBanPhase ? 'if enemy takes' : 'proj. final'}</span>
                  </div>
                  {!isBanPhase && (
                    <div
                      className="text-[10px] tabular-nums text-[#8b9bc8]"
                      title="Change vs. your current win estimate, in percentage points"
                    >
                      {rec.deltaPp >= 0 ? '+' : ''}{rec.deltaPp.toFixed(1)}pp vs now
                    </div>
                  )}
                </div>
              </div>
              {playerByline(rec.hero)}
            </button>
          ))}

          {oppRowsShown.map((rec) => (
            <button
              key={rec.hero}
              className={rowButtonClass(rec.probabilityPct == null)}
              onClick={() => onSelect(rec.hero)}
            >
              <div className="flex items-center justify-between">
                {heroCell(rec.hero)}
                <div className="text-right shrink-0">
                  {rec.probabilityPct != null ? (
                    <div title="How likely the enemy is to take this hero next (opponent model trained on real drafts)">
                      <span className="text-sm font-bold tabular-nums text-[#d6dbe0]">
                        {Math.round(rec.probabilityPct)}%
                      </span>
                      <span className="ml-1 text-[9px] text-[#8b9bc8]">likely</span>
                    </div>
                  ) : (
                    <div className="text-[10px] text-[#8b9bc8]" title="Statistically strong option for the enemy (not from the opponent model)">
                      likely (stats)
                    </div>
                  )}
                  {rec.impactPp != null && (
                    <div
                      className={cn(
                        'text-[10px] tabular-nums',
                        rec.impactPp <= -1 ? 'text-[#d46b6b]' : rec.impactPp >= 1 ? 'text-[#6fd46f]' : 'text-[#8b9bc8]'
                      )}
                      title="Expected change to YOUR win chance if the enemy takes this hero, in percentage points"
                    >
                      {rec.impactPp >= 0 ? '+' : ''}{rec.impactPp.toFixed(1)}pp for us
                    </div>
                  )}
                </div>
              </div>
              {playerByline(rec.hero)}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
