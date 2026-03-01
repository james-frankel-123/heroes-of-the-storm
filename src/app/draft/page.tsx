import { DraftClient } from './draft-client'
import { getDraftData, getTrackedBattletags } from '@/lib/data/queries'
import { MAPS } from '@/lib/mock/data'
import type { SkillTier } from '@/lib/types'
import type { DraftData } from '@/lib/draft/types'

const TIERS: SkillTier[] = ['low', 'mid', 'high']

/**
 * Pre-fetch draft data for ALL tier × map combinations.
 * This ensures zero API calls during the timed draft.
 */
export default async function DraftPage() {
  const trackedBattletags = await getTrackedBattletags()
  const battletags = trackedBattletags.map((bt) => bt.battletag)

  // Pre-fetch every tier × map combo
  const dataByTierMap: Record<SkillTier, Record<string, DraftData>> = {
    low: {},
    mid: {},
    high: {},
  }

  const fetches = TIERS.flatMap((tier) =>
    MAPS.map(async (map) => {
      const data = await getDraftData(tier, map, battletags)
      return { tier, map, data }
    })
  )

  const results = await Promise.all(fetches)
  for (const { tier, map, data } of results) {
    dataByTierMap[tier][map] = data
  }

  return (
    <DraftClient
      dataByTierMap={dataByTierMap}
      maps={MAPS}
      registeredBattletags={battletags}
    />
  )
}
