export const dynamic = 'force-dynamic'

import { DraftClient } from './draft-client'
import { getDraftData, getTrackedBattletags, getAllMaps } from '@/lib/data/queries'
import type { SkillTier } from '@/lib/types'
import type { DraftData } from '@/lib/draft/types'

const TIERS: SkillTier[] = ['low', 'mid', 'high']

/**
 * Pre-fetch draft data once per tier with all maps included.
 *
 * Hero map win rates are loaded for all maps in a single query per tier.
 * The client selects the appropriate map's data based on user selection.
 */
export default async function DraftPage() {
  const [trackedBattletags, maps] = await Promise.all([
    getTrackedBattletags(),
    getAllMaps(),
  ])
  const battletags = trackedBattletags.map((bt) => bt.battletag)

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
