'use client'

import { useState } from 'react'
import { TierSelector, getTierLabel } from '@/components/shared/tier-selector'
import { MetaHeroes } from '@/components/dashboard/meta-heroes'
import { MetaPairwise } from '@/components/dashboard/meta-pairwise'
import { PowerPicks } from '@/components/dashboard/power-picks'
import { PersonalInsights } from '@/components/dashboard/personal-insights'
import type {
  SkillTier,
  HeroStats,
  HeroPairwiseStats,
  HeroMapStats,
  PlayerHeroStats,
  TrackedBattletag,
} from '@/lib/types'

interface MetaData {
  topHeroes: HeroStats[]
  bottomHeroes: HeroStats[]
  synergies: HeroPairwiseStats[]
  counters: HeroPairwiseStats[]
  powerPicks: HeroMapStats[]
}

interface DashboardClientProps {
  metaByTier: Record<SkillTier, MetaData>
  personalData: { battletag: TrackedBattletag; heroStats: PlayerHeroStats[] }[]
}

export function DashboardClient({ metaByTier, personalData }: DashboardClientProps) {
  const [tier, setTier] = useState<SkillTier>('mid')
  const meta = metaByTier[tier]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Meta trends and personal insights &mdash; {getTierLabel(tier)}
          </p>
        </div>
        <TierSelector value={tier} onChange={setTier} />
      </div>

      {/* Meta Madness */}
      <section>
        <h2 className="text-xl font-semibold text-white mb-4">Meta Madness</h2>
        <div className="space-y-4">
          <MetaHeroes topHeroes={meta.topHeroes} bottomHeroes={meta.bottomHeroes} />
          <MetaPairwise synergies={meta.synergies} counters={meta.counters} />
          <PowerPicks picks={meta.powerPicks} />
        </div>
      </section>

      {/* Personal Insights */}
      {personalData.length > 0 && (
        <section>
          <h2 className="text-xl font-semibold text-white mb-4">
            Personal Insights
          </h2>
          <div className="space-y-4">
            {personalData.map(({ battletag, heroStats }) => (
              <PersonalInsights
                key={battletag.battletag}
                battletag={battletag}
                heroStats={heroStats}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
