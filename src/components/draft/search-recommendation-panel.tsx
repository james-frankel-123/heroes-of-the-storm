'use client'

import { Badge } from '@/components/ui/badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { cn } from '@/lib/utils'
import type { ExpectimaxResult } from '@/lib/draft/expectimax/types'
import type { DraftData } from '@/lib/draft/types'
import { scorePlayerStrength } from '@/lib/draft/engine'

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
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        {searching ? (
          <span className="text-xs text-blue-400 animate-pulse">
            {statusText || `Searching depth ${(searchDepth ?? 0) + 2}...`}
          </span>
        ) : searchDepth ? (
          <span className="text-xs text-muted-foreground">
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
              ? 'text-gaming-success'
              : rec.score >= 0
                ? 'text-gaming-warning'
                : 'text-gaming-danger'

            const playerInfo = canShowPlayerByline
              ? scorePlayerStrength(rec.hero, availableBattletags!, draftData!, map ?? null)
              : null

            return (
              <button
                key={rec.hero}
                className={cn(
                  'w-full text-left px-3 py-2 rounded-md border transition-all',
                  'hover:scale-[1.01] active:scale-[0.99]',
                  isGreedyPad && 'opacity-60',
                  isBanPhase
                    ? 'border-red-900/40 hover:border-red-700/60 hover:bg-red-950/30'
                    : 'border-border/40 hover:border-violet-700/60 hover:bg-violet-950/20'
                )}
                onClick={() => onSelect(rec.hero)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-white">{rec.hero}</span>
                    {role && (
                      <Badge variant={roleBadgeVariant(role)} className="text-[10px] px-1 py-0">
                        {role}
                      </Badge>
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
