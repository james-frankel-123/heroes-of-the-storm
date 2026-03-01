'use client'

import { Badge } from '@/components/ui/badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { cn } from '@/lib/utils'
import type { DraftRecommendation, RecommendationReason } from '@/lib/draft/types'

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

function reasonBadgeColor(type: RecommendationReason['type']): string {
  switch (type) {
    case 'map_strong': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
    case 'counter': return 'bg-red-500/20 text-red-400 border-red-500/30'
    case 'synergy': return 'bg-green-500/20 text-green-400 border-green-500/30'
    case 'role_need': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
    case 'player_strong': return 'bg-purple-500/20 text-purple-400 border-purple-500/30'
    case 'meta_strong': return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30'
    case 'ban_worthy': return 'bg-red-500/20 text-red-400 border-red-500/30'
  }
}

interface RecommendationPanelProps {
  recommendations: DraftRecommendation[]
  isBanPhase: boolean
  isOurTurn: boolean
  onSelect: (hero: string) => void
  unavailable: Set<string>
}

export function RecommendationPanel({
  recommendations,
  isBanPhase,
  isOurTurn,
  onSelect,
  unavailable,
}: RecommendationPanelProps) {
  const title = isBanPhase
    ? 'Ban Suggestions'
    : isOurTurn
      ? 'Pick Recommendations'
      : 'Likely Enemy Picks'

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-white">{title}</h3>

      {recommendations.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          Select a map to see recommendations
        </p>
      ) : (
        <div className="space-y-1 max-h-[450px] overflow-y-auto pr-1">
          {recommendations
            .filter((r) => !unavailable.has(r.hero))
            .slice(0, 12)
            .map((rec, idx) => {
              const role = getHeroRole(rec.hero)

              return (
                <button
                  key={rec.hero}
                  onClick={() => onSelect(rec.hero)}
                  className={cn(
                    'w-full flex items-start gap-2 p-2 rounded-md border text-left transition-colors',
                    isBanPhase
                      ? 'border-border hover:border-gaming-danger/60 hover:bg-gaming-danger/10'
                      : 'border-border hover:border-primary/60 hover:bg-primary/10'
                  )}
                >
                  {/* Rank number */}
                  <span className="text-[10px] text-muted-foreground font-mono w-4 pt-0.5 shrink-0">
                    {idx + 1}
                  </span>

                  <div className="flex-1 min-w-0 space-y-1">
                    {/* Hero name + role + score */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-sm font-medium truncate">
                        {rec.hero}
                      </span>
                      {role && (
                        <Badge
                          variant={roleBadgeVariant(role)}
                          className="text-[7px] px-1 py-0 shrink-0"
                        >
                          {role.split(' ')[0]}
                        </Badge>
                      )}
                      <span className="ml-auto text-[10px] text-muted-foreground shrink-0">
                        {rec.score}
                      </span>
                    </div>

                    {/* Reason tags */}
                    {rec.reasons.length > 0 && (
                      <div className="flex flex-wrap gap-0.5">
                        {rec.reasons.slice(0, 3).map((reason, ri) => (
                          <span
                            key={ri}
                            className={cn(
                              'inline-block px-1.5 py-0 rounded text-[9px] border',
                              reasonBadgeColor(reason.type)
                            )}
                          >
                            {reason.label}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Suggested player */}
                    {rec.suggestedPlayer && (
                      <p className="text-[10px] text-purple-400">
                        {rec.suggestedPlayer.split('#')[0]} should play this
                      </p>
                    )}
                  </div>
                </button>
              )
            })}
        </div>
      )}
    </div>
  )
}
