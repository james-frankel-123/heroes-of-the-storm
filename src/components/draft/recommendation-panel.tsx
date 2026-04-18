'use client'

import { RoleBadge } from '@/components/shared/role-badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { heroImageSrc } from '@/lib/data/hero-images'
import { cn } from '@/lib/utils'
import type { DraftRecommendation, RecommendationReason } from '@/lib/draft/types'
import { HEX_CLIP, METALLIC_FRAME } from './hex/constants'

function reasonColor(type: RecommendationReason['type']): string {
  switch (type) {
    case 'hero_wr': return 'text-cyan-400'
    case 'counter': return 'text-red-400'
    case 'synergy': return 'text-green-400'
    case 'player_strong': return 'text-purple-400'
    case 'ban_worthy': return 'text-red-400'
    case 'comp_wr': return 'text-blue-400'
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
    ? isOurTurn
      ? 'Your Ban Suggestions'
      : 'Likely Enemy Bans'
    : isOurTurn
      ? 'Pick Recommendations'
      : 'Likely Enemy Picks'

  return (
    <div
      className="space-y-2 rounded-sm p-3 border border-[#3a4050]"
      style={{ background: 'rgba(15, 20, 48, 0.6)' }}
    >
      <h3 className="text-sm tracking-[0.2em] text-[#d6dbe0] font-light">{title.toUpperCase()}</h3>

      {recommendations.length === 0 ? (
        <p className="text-xs text-[#8b9bc8]">
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
                ? 'text-[#6fd46f]'
                : rec.netDelta >= 0
                  ? 'text-[#d4b85a]'
                  : 'text-[#d46b6b]'

              return (
                <button
                  key={rec.hero}
                  onClick={() => onSelect(rec.hero)}
                  className={cn(
                    'w-full flex items-start gap-2 p-2 rounded border text-left transition-colors',
                    isBanPhase
                      ? 'border-[#3a2222] bg-[#0a0d1f]/40 hover:border-[#d46b6b]/60 hover:bg-[#d46b6b]/10'
                      : 'border-[#3a4050] bg-[#0a0d1f]/40 hover:border-[#6b8dd4]/60 hover:bg-[#6b8dd4]/10'
                  )}
                >
                  <div className="relative w-9 h-9 shrink-0">
                    <div className="absolute inset-0" style={{ clipPath: HEX_CLIP, background: METALLIC_FRAME }} />
                    <div className="absolute inset-[1.5px] bg-[#0a0d1f] overflow-hidden" style={{ clipPath: HEX_CLIP }}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={heroImageSrc(rec.hero)} alt="" loading="lazy" className="w-full h-full object-cover" />
                    </div>
                  </div>
                  <div className="flex-1 min-w-0 space-y-1">
                    {/* Hero name + role + net delta */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-sm font-medium truncate text-[#e8ecef]">
                        {rec.hero}
                      </span>
                      {role && (
                        <RoleBadge role={role!} className="text-[7px] px-1 py-0 shrink-0" short />
                      )}
                      <span className={cn('ml-auto text-xs font-semibold shrink-0', deltaColor)}>
                        {formatDelta(rec.netDelta)}
                      </span>
                    </div>

                    {/* Reason breakdown */}
                    {rec.reasons.length > 0 && (
                      <div className="flex flex-wrap gap-x-2 gap-y-0.5">
                        {rec.reasons
                          .filter((r) => {
                            // Composition/role reasons always shown (ranking-only, no delta threshold)
                            if (r.type === 'comp_wr') return true
                            // Data-backed reasons need meaningful delta
                            return Math.abs(r.delta) >= 0.5
                          })
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
