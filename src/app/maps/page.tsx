// Stats only change when the nightly sync runs — cache, re-render hourly.
export const revalidate = 3600

import { MapsClient } from './maps-client'
import {
  getMapStats,
  getHeroMapStats,
  getTrackedBattletags,
  getPlayerMapStats,
  getPlayerMatchHistory,
  getMapRotation,
} from '@/lib/data/queries'
import { filterToRotation } from '@/lib/data/map-rotation'
import type { SkillTier, HeroMapStats } from '@/lib/types'

const TIERS: SkillTier[] = ['low', 'mid', 'high']

export default async function MapsPage() {
  // Pre-fetch map stats for all tiers
  const [mapStatsLow, mapStatsMid, mapStatsHigh, trackedBattletags, rotation] =
    await Promise.all([
      getMapStats('low'),
      getMapStats('mid'),
      getMapStats('high'),
      getTrackedBattletags(),
      getMapRotation(),
    ])

  const mapStatsByTier = {
    low: filterToRotation(mapStatsLow, rotation),
    mid: filterToRotation(mapStatsMid, rotation),
    high: filterToRotation(mapStatsHigh, rotation),
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
    for (const d of filterToRotation(data, rotation)) {
      if (!byMap[d.map]) byMap[d.map] = []
      byMap[d.map].push(d)
    }
    heroMapByTier[tier] = byMap
  }

  // Personal map stats per battletag
  const personalData = await Promise.all(
    trackedBattletags.map(async (bt) => ({
      battletag: bt.battletag,
      mapStats: await getPlayerMapStats(bt.battletag, rotation),
      // Only shown per selected map, so off-rotation games are never displayed
      matches: filterToRotation(await getPlayerMatchHistory(bt.battletag, 200), rotation),
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
