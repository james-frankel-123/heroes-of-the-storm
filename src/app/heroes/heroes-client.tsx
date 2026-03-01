'use client'

import { useState } from 'react'
import { TierSelector, getTierLabel } from '@/components/shared/tier-selector'
import { HeroTable } from '@/components/heroes/hero-table'
import { HeroDetailModal } from '@/components/heroes/hero-detail-modal'
import type {
  SkillTier,
  HeroStats,
  HeroTalentStats,
  HeroPairwiseStats,
  PlayerHeroStats,
  PlayerMatch,
} from '@/lib/types'

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

export function HeroesClient({
  heroStatsByTier,
  heroStatsByName,
  talentsByTier,
  pairwiseByTier,
  personalData,
}: HeroesClientProps) {
  const [tier, setTier] = useState<SkillTier>('mid')
  const [selectedHero, setSelectedHero] = useState<string | null>(null)

  const heroes = heroStatsByTier[tier]

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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Heroes</h1>
          <p className="text-muted-foreground mt-1">
            Aggregate hero stats &mdash; {getTierLabel(tier)}
          </p>
        </div>
        <TierSelector value={tier} onChange={setTier} />
      </div>

      <HeroTable heroes={heroes} onHeroClick={setSelectedHero} />

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
