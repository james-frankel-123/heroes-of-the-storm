// Stats only change when the nightly sync runs — serve a cached page and
// re-render at most hourly instead of ~50 DB queries per page view.
export const revalidate = 3600

import { DashboardClient } from './dashboard-client'
import {
  getTopHeroes,
  getBottomHeroes,
  getTopSynergies,
  getTopCounters,
  getPowerPicks,
  getTrackedBattletags,
  getPlayerHeroStats,
  getMapRotation,
} from '@/lib/data/queries'
import { filterToRotation } from '@/lib/data/map-rotation'
import type { SkillTier, PlayerHeroStats, TrackedBattletag } from '@/lib/types'

// Pre-fetch data for all tiers so tier switching is instant (no round-trip)
async function fetchMetaData(tier: SkillTier, rotation: string[]) {
  const [topHeroes, bottomHeroes, synergies, counters, powerPicks] =
    await Promise.all([
      getTopHeroes(tier, 10),
      getBottomHeroes(tier, 10),
      getTopSynergies(tier, 100),
      getTopCounters(tier, 100),
      getPowerPicks(tier, 55, 500),
    ])
  return {
    topHeroes,
    bottomHeroes,
    synergies,
    counters,
    powerPicks: filterToRotation(powerPicks, rotation),
  }
}

export default async function DashboardPage() {
  const [rotation, trackedBattletags] = await Promise.all([
    getMapRotation(),
    getTrackedBattletags(),
  ])
  const [lowMeta, midMeta, highMeta] = await Promise.all([
    fetchMetaData('low', rotation),
    fetchMetaData('mid', rotation),
    fetchMetaData('high', rotation),
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
      mapRotation={rotation}
    />
  )
}
