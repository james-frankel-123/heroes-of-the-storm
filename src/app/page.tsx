export const dynamic = 'force-dynamic'

import { DashboardClient } from './dashboard-client'
import {
  getTopHeroes,
  getBottomHeroes,
  getTopSynergies,
  getTopCounters,
  getPowerPicks,
  getTrackedBattletags,
  getPlayerHeroStats,
} from '@/lib/data/queries'
import type { SkillTier, PlayerHeroStats, TrackedBattletag } from '@/lib/types'

// Pre-fetch data for all tiers so tier switching is instant (no round-trip)
async function fetchMetaData(tier: SkillTier) {
  const [topHeroes, bottomHeroes, synergies, counters, powerPicks] =
    await Promise.all([
      getTopHeroes(tier, 10),
      getBottomHeroes(tier, 10),
      getTopSynergies(tier, 10),
      getTopCounters(tier, 10),
      getPowerPicks(tier, 60, 10),
    ])
  return { topHeroes, bottomHeroes, synergies, counters, powerPicks }
}

export default async function DashboardPage() {
  const [lowMeta, midMeta, highMeta, trackedBattletags] = await Promise.all([
    fetchMetaData('low'),
    fetchMetaData('mid'),
    fetchMetaData('high'),
    getTrackedBattletags(),
  ])

  // Fetch personal stats for each tracked battletag
  const personalData: { battletag: TrackedBattletag; heroStats: PlayerHeroStats[] }[] =
    await Promise.all(
      trackedBattletags.map(async (bt) => ({
        battletag: bt,
        heroStats: await getPlayerHeroStats(bt.battletag),
      }))
    )

  return (
    <DashboardClient
      metaByTier={{
        low: lowMeta,
        mid: midMeta,
        high: highMeta,
      }}
      personalData={personalData}
    />
  )
}
