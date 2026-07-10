import { NextResponse } from 'next/server'
import { eq, inArray, sql as dsql } from 'drizzle-orm'
import { db } from '@/lib/db'
import { draftRatings, ratingItems } from '@/lib/db/schema'
import {
  assignedItemIds,
  calibrationOrder,
  extendedOrder,
  isTestRater,
  normalizeRater,
  raterSlot,
  screenerOrder,
  sideSwapped,
} from '@/lib/rating/assignment'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

/**
 * GET /api/ratings/items?rater=NAME[&slot=N]
 *
 * Returns the rater's blinded items in two arms:
 *   - `items`: the 8 shared screener items FIRST, then the 40 shared
 *     calibration items, then the 45 core latin-square items (93 total;
 *     2 + 2 + 3 = 7 for test raters, unless the name starts with "testfull"
 *     which gets the full real assignment).
 *   - `extendedItems`: the uncapped extended pool in the rater's serving order
 *     (stable per-rater order, under-coverage tiebreak), with items this rater
 *     has already rated removed — so the arm resumes cleanly on any device.
 * Both are blinded: no provenance; A/B display side randomized per rater.
 * Also returns rated counts so the client can resume mid-screener,
 * mid-calibration, mid-core, or deep in the extended arm after a hard refresh.
 */
export async function GET(req: Request) {
  const url = new URL(req.url)
  const rater = (url.searchParams.get('rater') ?? '').trim()
  if (!rater || rater.length > 100) {
    return NextResponse.json({ error: 'rater query param required (1-100 chars)' }, { status: 400 })
  }
  const slotParam = url.searchParams.get('slot')
  let slotOverride: number | undefined
  if (slotParam !== null && slotParam !== '') {
    const n = Number(slotParam)
    if (!Number.isInteger(n) || n < 0 || n > 9) {
      return NextResponse.json({ error: 'slot must be an integer 0-9' }, { status: 400 })
    }
    slotOverride = n
  }

  const rows = await db
    .select({
      id: ratingItems.id,
      block: ratingItems.block,
      teams: ratingItems.teams,
      map: ratingItems.map,
      tier: ratingItems.tier,
      // provenance intentionally NOT selected — raters are blind to it
    })
    .from(ratingItems)
  if (rows.length === 0) {
    return NextResponse.json({ error: 'no rating items seeded' }, { status: 500 })
  }

  const byId = new Map(rows.map((r) => [r.id, r]))
  const screenerIds = rows.filter((r) => r.block === 'screener').map((r) => r.id)
  const calibrationIds = rows.filter((r) => r.block === 'calibration').map((r) => r.id)
  const coreIds = rows.filter((r) => r.block === 'core').map((r) => r.id)
  const extendedIds = rows.filter((r) => r.block === 'extended').map((r) => r.id)

  const slot = raterSlot(rater, slotOverride)
  // Screener comes FIRST for every rater, then the shared calibration block,
  // then the latin-square core.
  const assignedCoreIds = [
    ...screenerOrder(screenerIds, rater),
    ...calibrationOrder(calibrationIds, rater),
    ...assignedItemIds(coreIds, rater, slot),
  ]

  // This rater's already-rated ids (for cross-device resume).
  const rated = await db
    .select({ itemId: draftRatings.itemId })
    .from(draftRatings)
    .where(eq(draftRatings.rater, normalizeRater(rater)))
  const ratedIds = rated.map((r) => r.itemId)
  const ratedSet = new Set(ratedIds)

  // Global coverage of the extended pool: how many ratings each extended item
  // already has, so the serving order prioritizes globally under-covered items.
  const coverage = new Map<number, number>()
  if (extendedIds.length > 0) {
    const covRows = await db
      .select({ itemId: draftRatings.itemId, n: dsql<number>`count(*)::int` })
      .from(draftRatings)
      .where(inArray(draftRatings.itemId, extendedIds))
      .groupBy(draftRatings.itemId)
    for (const r of covRows) coverage.set(r.itemId, Number(r.n))
  }
  const orderedExtendedIds = extendedOrder(extendedIds, rater, coverage, ratedSet)

  const blind = (id: number) => {
    const row = byId.get(id)!
    const teams = row.teams as { team0: string[]; team1: string[] }
    const swapped = sideSwapped(rater, id)
    return {
      id,
      map: row.map,
      tier: row.tier,
      teamA: swapped ? teams.team1 : teams.team0,
      teamB: swapped ? teams.team0 : teams.team1,
    }
  }

  const items = assignedCoreIds.map(blind)
  const extendedItems = orderedExtendedIds.map(blind)

  const ratedCoreCount = assignedCoreIds.filter((id) => ratedSet.has(id)).length
  const extendedIdSet = new Set(extendedIds)
  const ratedExtendedCount = ratedIds.filter((id) => extendedIdSet.has(id)).length

  return NextResponse.json({
    rater,
    slot,
    isTest: isTestRater(rater),
    items,
    extendedItems,
    ratedItemIds: ratedIds,
    ratedCoreCount,
    ratedExtendedCount,
  })
}
