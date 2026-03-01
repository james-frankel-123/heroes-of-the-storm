export const dynamic = 'force-dynamic'

import { HeroesClient } from './heroes-client'
import {
  getHeroStats,
  getTalentStats,
  getPairwiseStats,
  getTrackedBattletags,
  getPlayerHeroStats,
  getPlayerMatchHistory,
} from '@/lib/data/queries'
import { HERO_ROLES } from '@/lib/data/hero-roles'
import type { SkillTier, HeroStats } from '@/lib/types'

// Pre-fetch hero list for all tiers
async function fetchAllTiers() {
  const [low, mid, high] = await Promise.all([
    getHeroStats('low'),
    getHeroStats('mid'),
    getHeroStats('high'),
  ])
  return { low, mid, high }
}

export default async function HeroesPage() {
  const [heroStatsByTier, trackedBattletags] = await Promise.all([
    fetchAllTiers(),
    getTrackedBattletags(),
  ])

  // Pre-fetch detail data for all heroes that have talent/pairwise data
  // For the detail modal, we load on-demand via the client, but
  // we pre-load all hero stats by name, map stats, and personal data
  const allHeroes = Object.keys(HERO_ROLES)

  // Pre-fetch talent and pairwise data for all tiers
  const allTalentData = await Promise.all(
    (['low', 'mid', 'high'] as SkillTier[]).map(async (tier) => {
      const talents = await Promise.all(
        allHeroes.map(async (hero) => ({
          hero,
          talents: await getTalentStats(hero, tier),
        }))
      )
      return { tier, talents }
    })
  )

  const allPairwiseData = await Promise.all(
    (['low', 'mid', 'high'] as SkillTier[]).map(async (tier) => {
      const pairs = await Promise.all(
        allHeroes.map(async (hero) => ({
          hero,
          synergies: await getPairwiseStats(hero, 'with', tier),
          counters: await getPairwiseStats(hero, 'against', tier),
        }))
      )
      return { tier, pairs }
    })
  )

  // Personal stats per battletag
  const personalData = await Promise.all(
    trackedBattletags.map(async (bt) => ({
      battletag: bt.battletag,
      heroStats: await getPlayerHeroStats(bt.battletag),
      matches: await getPlayerMatchHistory(bt.battletag, 100),
    }))
  )

  // All hero stats by name (for detail modal tier comparison)
  const heroStatsByName: Record<string, HeroStats[]> = {}
  for (const hero of allHeroes) {
    heroStatsByName[hero] = [
      heroStatsByTier.low.find((h) => h.hero === hero),
      heroStatsByTier.mid.find((h) => h.hero === hero),
      heroStatsByTier.high.find((h) => h.hero === hero),
    ].filter(Boolean) as HeroStats[]
  }

  // Flatten talent data
  const talentsByTier: Record<SkillTier, Record<string, typeof allTalentData[0]['talents'][0]['talents']>> = {
    low: {},
    mid: {},
    high: {},
  }
  for (const { tier, talents } of allTalentData) {
    for (const { hero, talents: t } of talents) {
      talentsByTier[tier][hero] = t
    }
  }

  // Flatten pairwise data
  const pairwiseByTier: Record<SkillTier, Record<string, { synergies: typeof allPairwiseData[0]['pairs'][0]['synergies']; counters: typeof allPairwiseData[0]['pairs'][0]['counters'] }>> = {
    low: {},
    mid: {},
    high: {},
  }
  for (const { tier, pairs } of allPairwiseData) {
    for (const { hero, synergies, counters } of pairs) {
      pairwiseByTier[tier][hero] = { synergies, counters }
    }
  }

  return (
    <HeroesClient
      heroStatsByTier={heroStatsByTier}
      heroStatsByName={heroStatsByName}
      talentsByTier={talentsByTier}
      pairwiseByTier={pairwiseByTier}
      personalData={personalData}
    />
  )
}
