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

function reasonColor(type: RecommendationReason['type']): string {
  switch (type) {
    case 'hero_wr': return 'text-cyan-400'
    case 'counter': return 'text-red-400'
    case 'synergy': return 'text-green-400'
    case 'role_need': return 'text-yellow-400'
    case 'role_penalty': return 'text-orange-400'
    case 'player_strong': return 'text-purple-400'
    case 'ban_worthy': return 'text-red-400'
  }
}

function formatDelta(d: number): string {
  return `${d >= 0 ? '+' : ''}${d.toFixed(1)}%`
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
            .map((rec) => {
              const role = getHeroRole(rec.hero)
              const deltaColor = rec.netDelta >= 3
                ? 'text-gaming-success'
                : rec.netDelta >= 0
                  ? 'text-gaming-warning'
                  : 'text-gaming-danger'

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
                  <div className="flex-1 min-w-0 space-y-1">
                    {/* Hero name + role + net delta */}
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
                      <span className={cn('ml-auto text-xs font-semibold shrink-0', deltaColor)}>
                        {formatDelta(rec.netDelta)}
                      </span>
                    </div>

                    {/* Reason breakdown — show each delta */}
                    {rec.reasons.length > 0 && (
                      <div className="flex flex-wrap gap-x-2 gap-y-0.5">
                        {rec.reasons
                          .filter((r) => Math.abs(r.delta) >= 0.5)
                          .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
                          .slice(0, 4)
                          .map((reason, ri) => (
                            <span
                              key={ri}
                              className={cn('text-[10px]', reasonColor(reason.type))}
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
