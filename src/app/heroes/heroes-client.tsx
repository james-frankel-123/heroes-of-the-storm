'use client'

import { useState, useMemo } from 'react'
import { TierSelector, getTierLabel } from '@/components/shared/tier-selector'
import { HeroTable } from '@/components/heroes/hero-table'
import type { HeroRow } from '@/components/heroes/hero-table'
import { HeroDetailModal } from '@/components/heroes/hero-detail-modal'
import { cn } from '@/lib/utils'
import type {
  SkillTier,
  HeroStats,
  HeroTalentStats,
  HeroPairwiseStats,
  PlayerHeroStats,
  PlayerMatch,
} from '@/lib/types'

/** 'global' for aggregate data, or a battletag string for personal */
type ViewMode = 'global' | string

interface HeroesClientProps {
  heroStatsByTier: Record<SkillTier, HeroStats[]>
  heroStatsByName: Record<string, HeroStats[]>
  talentsByTier: Record<SkillTier, Record<string, HeroTalentStats[]>>
  pairwiseByTier: Record<
    SkillTier,
    Record<string, { synergies: HeroPairwiseStats[]; counters: HeroPairwiseStats[] }>
  >
  personalData: {
    battletag: string
    heroStats: PlayerHeroStats[]
    matches: PlayerMatch[]
  }[]
}

/** Convert PlayerHeroStats to HeroRow so the same table component works */
function toHeroRow(p: PlayerHeroStats): HeroRow {
  return {
    hero: p.hero,
    skillTier: 'mid' as SkillTier, // not tier-bucketed
    games: p.games,
    wins: p.wins,
    winRate: p.winRate,
    banRate: 0,
    pickRate: 0,
    avgKills: p.avgKills,
    avgDeaths: p.avgDeaths,
    avgAssists: p.avgAssists,
    avgHeroDamage: 0,
    avgSiegeDamage: 0,
    avgHealing: 0,
    avgExperience: 0,
    avgDamageSoaked: 0,
    avgMercCaptures: 0,
    avgSelfHealing: 0,
    avgTimeDead: 0,
    patchTag: null,
    mawp: p.mawp,
  }
}

export function HeroesClient({
  heroStatsByTier,
  heroStatsByName,
  talentsByTier,
  pairwiseByTier,
  personalData,
}: HeroesClientProps) {
  const [tier, setTier] = useState<SkillTier>('mid')
  const [viewMode, setViewMode] = useState<ViewMode>('global')
  const [selectedHero, setSelectedHero] = useState<string | null>(null)

  const isPersonal = viewMode !== 'global'

  // Build rows for current view
  const heroes: HeroRow[] = useMemo(() => {
    if (!isPersonal) return heroStatsByTier[tier]
    const pd = personalData.find((p) => p.battletag === viewMode)
    if (!pd) return []
    return pd.heroStats.map(toHeroRow)
  }, [isPersonal, viewMode, tier, heroStatsByTier, personalData])

  // Subtitle text
  const subtitle = isPersonal
    ? `Personal stats for ${viewMode}`
    : `Aggregate hero stats \u2014 ${getTierLabel(tier)}`

  // Build detail data for selected hero
  const detailStatsByTier = selectedHero
    ? heroStatsByName[selectedHero] ?? []
    : []

  const detailTalents = selectedHero
    ? talentsByTier[tier][selectedHero] ?? []
    : []

  const detailPairwise = selectedHero
    ? pairwiseByTier[tier][selectedHero] ?? { synergies: [], counters: [] }
    : { synergies: [], counters: [] }

  const detailPersonal = selectedHero
    ? personalData.map((p) => ({
        battletag: p.battletag,
        stats:
          p.heroStats.find((h) => h.hero === selectedHero) ?? null,
      }))
    : []

  const detailMatches = selectedHero
    ? personalData.map((p) => ({
        battletag: p.battletag,
        matches: p.matches.filter((m) => m.hero === selectedHero),
      }))
    : []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Heroes</h1>
          <p className="text-muted-foreground mt-1">{subtitle}</p>
        </div>
        {/* Only show tier selector in global mode */}
        {!isPersonal && <TierSelector value={tier} onChange={setTier} />}
      </div>

      {/* View mode selector */}
      {personalData.length > 0 && (
        <div className="flex items-center gap-1 p-1 rounded-lg bg-muted w-fit">
          <button
            onClick={() => setViewMode('global')}
            className={cn(
              'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
              viewMode === 'global'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            Global
          </button>
          {personalData.map((p) => (
            <button
              key={p.battletag}
              onClick={() => setViewMode(p.battletag)}
              className={cn(
                'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                viewMode === p.battletag
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {p.battletag.split('#')[0]}
            </button>
          ))}
        </div>
      )}

      <HeroTable
        heroes={heroes}
        onHeroClick={setSelectedHero}
        personal={isPersonal}
      />

      {selectedHero && (
        <HeroDetailModal
          open={!!selectedHero}
          onClose={() => setSelectedHero(null)}
          heroName={selectedHero}
          statsByTier={detailStatsByTier}
          mapStats={[]}
          talents={detailTalents}
          synergies={detailPairwise.synergies}
          counters={detailPairwise.counters}
          personalStats={detailPersonal}
          recentMatches={detailMatches}
          currentTier={tier}
        />
      )}
    </div>
  )
}
