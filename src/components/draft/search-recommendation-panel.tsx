'use client'

import { RoleBadge } from '@/components/shared/role-badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { heroImageSrc } from '@/lib/data/hero-images'
import { cn } from '@/lib/utils'
import { HEX_CLIP, METALLIC_FRAME } from './hex/constants'
import type { ExpectimaxResult } from '@/lib/draft/expectimax/types'
import type { DraftData } from '@/lib/draft/types'
import { scorePlayerStrength } from '@/lib/draft/engine'

/** A row shown on opponent turns: how likely they take the hero + what it does to us. */
export interface OpponentPredictionRow {
  hero: string
  /** P(opponent takes this hero next), 0-100, from the opponent model. Null for stats-based padding rows. */
  probabilityPct: number | null
  /** Expected change to OUR displayed win % (percentage points) if they take it. Null when not computable (bans). */
  impactPp: number | null
}

interface SearchRecommendationPanelProps {
  /** Our-turn candidates from the expectimax search */
  results: ExpectimaxResult[]
  /** Greedy recommendations to fill remaining slots (our turn) */
  greedyFallback?: { hero: string; netDelta: number }[]
  /** Current expected win % for our team (un-normalized, same evaluator as search scores) */
  baselineWinPct?: number | null
  /** Opponent-turn rows (used instead of results/greedyFallback when !isOurTurn) */
  opponentRows?: OpponentPredictionRow[]
  searchDepth: number | null
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

const clampPct = (v: number) => Math.max(1, Math.min(99, v))

export function SearchRecommendationPanel({
  results,
  searchDepth,
  greedyFallback,
  baselineWinPct,
  opponentRows,
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
    ? `Your expected win chance after this ${isBanPhase ? 'ban' : 'pick'}`
    : isBanPhase
      ? 'How likely the enemy is to ban each hero'
      : 'How likely the enemy is to pick each hero, and what it would do to your win chance'

  // Baseline for "how much better than staying put" deltas (our turn)
  const baseline = baselineWinPct ?? 50

  interface Row {
    hero: string
    isGreedyPad: boolean
    /** Our turn: absolute expected win % after selecting */
    winPct?: number
    /** Our turn: delta vs current estimate, in percentage points */
    deltaPp?: number
    /** Opponent turn */
    probabilityPct?: number | null
    impactPp?: number | null
  }

  let rows: Row[]
  if (isOurTurn) {
    // Merge search results with greedy fallback to fill to 10.
    // Search scores are "expected win % − 50" at the searched leaf;
    // greedy netDelta is the marginal pp contribution of adding the hero.
    const searchHeroes = new Set(results.map(r => r.hero))
    rows = [
      ...results
        .filter(r => !unavailable.has(r.hero))
        .map(r => ({
          hero: r.hero,
          isGreedyPad: false,
          winPct: clampPct(50 + r.score),
          deltaPp: 50 + r.score - baseline,
        })),
      ...(greedyFallback ?? [])
        .filter(r => !searchHeroes.has(r.hero) && !unavailable.has(r.hero))
        .map(r => ({
          hero: r.hero,
          isGreedyPad: true,
          winPct: clampPct(baseline + r.netDelta),
          deltaPp: r.netDelta,
        })),
    ].slice(0, 10)
  } else {
    rows = (opponentRows ?? [])
      .filter(r => !unavailable.has(r.hero))
      .slice(0, 10)
      .map(r => ({
        hero: r.hero,
        isGreedyPad: r.probabilityPct == null,
        probabilityPct: r.probabilityPct,
        impactPp: r.impactPp,
      }))
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
            {statusText || `Searching depth ${(searchDepth ?? 0) + 2}...`}
          </span>
        ) : searchDepth ? (
          <span className="text-xs text-[#8b9bc8]">
            Depth {searchDepth}
          </span>
        ) : null}
      </div>
      <p className="text-[10px] text-[#8b9bc8]">{subtitle}</p>

      {rows.length === 0 ? (
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
          {rows.map((rec) => {
            const role = getHeroRole(rec.hero)

            const playerInfo = canShowPlayerByline
              ? scorePlayerStrength(rec.hero, availableBattletags!, draftData!, map ?? null)
              : null

            return (
              <button
                key={rec.hero}
                className={cn(
                  'w-full text-left px-3 py-2 rounded border transition-all',
                  'hover:scale-[1.01] active:scale-[0.99]',
                  rec.isGreedyPad && 'opacity-60',
                  isBanPhase
                    ? 'border-[#3a2222] bg-[#0a0d1f]/40 hover:border-[#d46b6b]/60 hover:bg-[#d46b6b]/10'
                    : 'border-[#3a4050] bg-[#0a0d1f]/40 hover:border-[#6b8dd4]/60 hover:bg-[#6b8dd4]/10'
                )}
                onClick={() => onSelect(rec.hero)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 min-w-0">
                    <div className="relative w-9 h-9 shrink-0">
                      <div className="absolute inset-0" style={{ clipPath: HEX_CLIP, background: METALLIC_FRAME }} />
                      <div className="absolute inset-[1.5px] bg-[#0a0d1f] overflow-hidden" style={{ clipPath: HEX_CLIP }}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={heroImageSrc(rec.hero)} alt="" loading="lazy" className="w-full h-full object-cover" />
                      </div>
                    </div>
                    <span className="text-sm font-medium text-[#e8ecef] truncate">{rec.hero}</span>
                    {role && (
                      <RoleBadge role={role!} className="text-[10px] px-1 py-0 shrink-0" />
                    )}
                  </div>

                  {isOurTurn ? (
                    <div className="text-right shrink-0">
                      <div
                        title={`Estimated chance your team wins if you ${isBanPhase ? 'ban' : 'pick'} ${rec.hero} (statistical draft model)`}
                      >
                        <span className={cn(
                          'text-sm font-bold tabular-nums',
                          (rec.deltaPp ?? 0) >= 3
                            ? 'text-[#6fd46f]'
                            : (rec.deltaPp ?? 0) >= 0
                              ? 'text-[#d4b85a]'
                              : 'text-[#d46b6b]'
                        )}>
                          {rec.winPct!.toFixed(1)}%
                        </span>
                        <span className="ml-1 text-[9px] text-[#8b9bc8]">win chance</span>
                      </div>
                      <div
                        className="text-[10px] tabular-nums text-[#8b9bc8]"
                        title="Change vs. your current win estimate, in percentage points"
                      >
                        {(rec.deltaPp ?? 0) >= 0 ? '+' : ''}{rec.deltaPp!.toFixed(1)}pp vs now
                      </div>
                    </div>
                  ) : (
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
                  )}
                </div>
                {playerInfo?.reason && (
                  <p className="mt-1 text-[10px] text-purple-400">
                    {playerInfo.reason.label}
                    {playerInfo.player && (
                      <span className="text-muted-foreground"> · {playerInfo.player.split('#')[0]} should play this</span>
                    )}
                  </p>
                )}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
