export const dynamic = 'force-dynamic'

import { HeroesClient } from './heroes-client'
import {
  getHeroStats,
  getAllTalentStats,
  getAllPairwiseStats,
  getAllHeroMapStats,
  getTrackedBattletags,
  getPlayerHeroStats,
  getPlayerMatchHistory,
  getPlayerMapStats,
} from '@/lib/data/queries'
import type { SkillTier, HeroStats, HeroMapStats } from '@/lib/types'

const TIERS: SkillTier[] = ['low', 'mid', 'high']

/**
 * Pre-fetch hero data using bulk queries.
 *
 * Previously fetched talents and pairwise stats per-hero per-tier
 * (90 heroes x 3 tiers x 3 query types = ~810 queries).
 * Now uses 3 bulk queries per tier (9 total) + hero stats (3) + personal (~6).
 * Total: ~18 queries instead of ~820.
 */
export default async function HeroesPage() {
  // Parallel: hero stats for all tiers + tracked battletags
  const [heroStatsByTier, trackedBattletags] = await Promise.all([
    (async () => {
      const [low, mid, high] = await Promise.all(
        TIERS.map((tier) => getHeroStats(tier))
      )
      return { low, mid, high } as Record<SkillTier, HeroStats[]>
    })(),
    getTrackedBattletags(),
  ])

  // Parallel: bulk talent + pairwise + hero-map for all tiers + personal data
  const [talentsByTier, pairwiseByTier, heroMapByTier, personalData] = await Promise.all([
    // Talents: 1 query per tier = 3 total
    (async () => {
      const results = await Promise.all(
        TIERS.map(async (tier) => ({
          tier,
          data: await getAllTalentStats(tier),
        }))
      )
      const out: Record<SkillTier, Record<string, import('@/lib/types').HeroTalentStats[]>> = {
        low: {},
        mid: {},
        high: {},
      }
      for (const { tier, data } of results) {
        out[tier] = data
      }
      return out
    })(),

    // Pairwise: 1 query per tier = 3 total
    (async () => {
      const results = await Promise.all(
        TIERS.map(async (tier) => ({
          tier,
          data: await getAllPairwiseStats(tier),
        }))
      )
      const out: Record<
        SkillTier,
        Record<string, { synergies: import('@/lib/types').HeroPairwiseStats[]; counters: import('@/lib/types').HeroPairwiseStats[] }>
      > = { low: {}, mid: {}, high: {} }

      for (const { tier, data } of results) {
        // Merge synergies and counters into per-hero records
        const allHeroes = new Set([
          ...Object.keys(data.synergies),
          ...Object.keys(data.counters),
        ])
        for (const hero of allHeroes) {
          out[tier][hero] = {
            synergies: data.synergies[hero] ?? [],
            counters: data.counters[hero] ?? [],
          }
        }
      }
      return out
    })(),

    // Hero-map stats: 1 query per tier = 3 total
    (async () => {
      const results = await Promise.all(
        TIERS.map(async (tier) => ({
          tier,
          data: await getAllHeroMapStats(tier),
        }))
      )
      const out: Record<SkillTier, Record<string, HeroMapStats[]>> = {
        low: {},
        mid: {},
        high: {},
      }
      for (const { tier, data } of results) {
        out[tier] = data
      }
      return out
    })(),

    // Personal stats: 2 queries per battletag
    Promise.all(
      trackedBattletags.map(async (bt) => ({
        battletag: bt.battletag,
        heroStats: await getPlayerHeroStats(bt.battletag),
        matches: await getPlayerMatchHistory(bt.battletag, 100),
        mapStats: await getPlayerMapStats(bt.battletag),
      }))
    ),
  ])

  // Build heroStatsByName from the already-fetched tier data (no extra queries)
  const heroStatsByName: Record<string, HeroStats[]> = {}
  for (const tier of TIERS) {
    for (const h of heroStatsByTier[tier]) {
      if (!heroStatsByName[h.hero]) heroStatsByName[h.hero] = []
      heroStatsByName[h.hero].push(h)
    }
  }

  return (
    <HeroesClient
      heroStatsByTier={heroStatsByTier}
      heroStatsByName={heroStatsByName}
      talentsByTier={talentsByTier}
      pairwiseByTier={pairwiseByTier}
      heroMapByTier={heroMapByTier}
      personalData={personalData}
    />
  )
}
