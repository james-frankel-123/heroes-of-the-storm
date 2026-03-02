export const dynamic = 'force-dynamic'

import { DraftClient } from './draft-client'
import { getDraftData, getTrackedBattletags, getAllMaps } from '@/lib/data/queries'
import type { SkillTier } from '@/lib/types'
import type { DraftData } from '@/lib/draft/types'

const TIERS: SkillTier[] = ['low', 'mid', 'high']

/**
 * Pre-fetch draft data once per tier (not per tier × map).
 *
 * Hero stats, synergies, counters, and player stats don't vary by map.
 * Map-specific hero data isn't used by the engine (API doesn't provide it).
 * This reduces DB queries from ~420 to ~30.
 */
export default async function DraftPage() {
  const [trackedBattletags, maps] = await Promise.all([
    getTrackedBattletags(),
    getAllMaps(),
  ])
  const battletags = trackedBattletags.map((bt) => bt.battletag)

  // Fetch once per tier — map dimension is unused by the engine
  const dataByTier: Record<SkillTier, DraftData> = {} as Record<SkillTier, DraftData>

  const fetches = TIERS.map(async (tier) => {
    const data = await getDraftData(tier, maps[0] ?? '', battletags)
    return { tier, data }
  })

  const results = await Promise.all(fetches)
  for (const { tier, data } of results) {
    dataByTier[tier] = data
  }

  return (
    <DraftClient
      dataByTier={dataByTier}
      maps={maps}
      registeredBattletags={battletags}
    />
  )
}
