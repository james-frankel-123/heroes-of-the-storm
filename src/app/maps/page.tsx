export const dynamic = 'force-dynamic'

import { MapsClient } from './maps-client'
import {
  getMapStats,
  getHeroMapStats,
  getTrackedBattletags,
  getPlayerMapStats,
  getPlayerMatchHistory,
  getAllMaps,
} from '@/lib/data/queries'
import type { SkillTier, HeroMapStats } from '@/lib/types'

const TIERS: SkillTier[] = ['low', 'mid', 'high']

export default async function MapsPage() {
  // Pre-fetch map stats for all tiers
  const [mapStatsLow, mapStatsMid, mapStatsHigh, trackedBattletags] =
    await Promise.all([
      getMapStats('low'),
      getMapStats('mid'),
      getMapStats('high'),
      getTrackedBattletags(),
    ])

  const mapStatsByTier = {
    low: mapStatsLow,
    mid: mapStatsMid,
    high: mapStatsHigh,
  }

  // Pre-fetch hero performance per map per tier
  const heroMapByTier: Record<SkillTier, Record<string, HeroMapStats[]>> = {
    low: {},
    mid: {},
    high: {},
  }

  const allHeroMapData = await Promise.all(
    TIERS.map(async (tier) => {
      const data = await getHeroMapStats(tier)
      return { tier, data }
    })
  )

  for (const { tier, data } of allHeroMapData) {
    const byMap: Record<string, HeroMapStats[]> = {}
    for (const d of data) {
      if (!byMap[d.map]) byMap[d.map] = []
      byMap[d.map].push(d)
    }
    heroMapByTier[tier] = byMap
  }

  // Personal map stats per battletag
  const personalData = await Promise.all(
    trackedBattletags.map(async (bt) => ({
      battletag: bt.battletag,
      mapStats: await getPlayerMapStats(bt.battletag),
      matches: await getPlayerMatchHistory(bt.battletag, 200),
    }))
  )

  return (
    <MapsClient
      mapStatsByTier={mapStatsByTier}
      heroMapByTier={heroMapByTier}
      personalData={personalData}
    />
  )
}
