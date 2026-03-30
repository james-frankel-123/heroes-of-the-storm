'use client'

import { Badge } from '@/components/ui/badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { cn } from '@/lib/utils'
import type { ExpectimaxResult } from '@/lib/draft/expectimax/types'

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
  searchDepth: number | null
  searching: boolean
  isBanPhase: boolean
  isOurTurn: boolean
  onSelect: (hero: string) => void
  unavailable: Set<string>
}

export function SearchRecommendationPanel({
  results,
  searchDepth,
  searching,
  isBanPhase,
  isOurTurn,
  onSelect,
  unavailable,
}: SearchRecommendationPanelProps) {
  const title = isBanPhase
    ? isOurTurn ? 'Ban Suggestions' : 'Likely Enemy Bans'
    : isOurTurn ? 'Search Recommendations' : 'Likely Enemy Picks'

  const filtered = results.filter(r => !unavailable.has(r.hero))

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        {searching ? (
          <span className="text-xs text-violet-400 animate-pulse">
            Searching depth {(searchDepth ?? 0) + 2}...
          </span>
        ) : searchDepth ? (
          <span className="text-xs text-muted-foreground">
            Depth {searchDepth}
          </span>
        ) : null}
      </div>

      {filtered.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          {searching ? 'Computing...' : 'Select a map to see recommendations'}
        </p>
      ) : (
        <div className="space-y-1 max-h-[450px] overflow-y-auto pr-1">
          {filtered.slice(0, 12).map((rec) => {
            const role = getHeroRole(rec.hero)
            const deltaColor = rec.score >= 3
              ? 'text-gaming-success'
              : rec.score >= 0
                ? 'text-gaming-warning'
                : 'text-gaming-danger'

            return (
              <button
                key={rec.hero}
                className={cn(
                  'w-full text-left px-3 py-2 rounded-md border transition-all',
                  'hover:scale-[1.01] active:scale-[0.99]',
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
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
