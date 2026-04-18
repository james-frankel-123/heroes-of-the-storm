'use client'

import { RoleBadge } from '@/components/shared/role-badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { heroImageSrc } from '@/lib/data/hero-images'
import { cn } from '@/lib/utils'
import { HEX_CLIP, METALLIC_FRAME } from './hex/constants'
import type { ExpectimaxResult } from '@/lib/draft/expectimax/types'
import type { DraftData } from '@/lib/draft/types'
import { scorePlayerStrength } from '@/lib/draft/engine'

interface SearchRecommendationPanelProps {
  results: ExpectimaxResult[]
  /** Greedy recommendations to fill remaining slots */
  greedyFallback?: { hero: string; netDelta: number }[]
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

export function SearchRecommendationPanel({
  results,
  searchDepth,
  greedyFallback,
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

  // Merge search results with greedy fallback to fill to 10
  const searchHeroes = new Set(results.map(r => r.hero))
  const greedyPadding: ExpectimaxResult[] = (greedyFallback ?? [])
    .filter(r => !searchHeroes.has(r.hero) && !unavailable.has(r.hero))
    .map(r => ({ hero: r.hero, score: r.netDelta, depth: -1, nodesVisited: 0 }))
  const merged = [
    ...results.filter(r => !unavailable.has(r.hero)),
    ...greedyPadding,
  ].slice(0, 10)
  const filtered = merged

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

      {filtered.length === 0 ? (
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
          {filtered.map((rec) => {
            const role = getHeroRole(rec.hero)
            const isGreedyPad = rec.depth === -1
            const deltaColor = rec.score >= 3
              ? 'text-[#6fd46f]'
              : rec.score >= 0
                ? 'text-[#d4b85a]'
                : 'text-[#d46b6b]'

            const playerInfo = canShowPlayerByline
              ? scorePlayerStrength(rec.hero, availableBattletags!, draftData!, map ?? null)
              : null

            return (
              <button
                key={rec.hero}
                className={cn(
                  'w-full text-left px-3 py-2 rounded border transition-all',
                  'hover:scale-[1.01] active:scale-[0.99]',
                  isGreedyPad && 'opacity-60',
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
                  <span className={cn('text-sm font-bold tabular-nums', deltaColor)}>
                    {rec.score >= 0 ? '+' : ''}{rec.score.toFixed(1)}
                  </span>
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
