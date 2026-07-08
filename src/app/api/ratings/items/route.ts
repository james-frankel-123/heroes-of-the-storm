import { NextResponse } from 'next/server'
import { eq } from 'drizzle-orm'
import { db } from '@/lib/db'
import { draftRatings, ratingItems } from '@/lib/db/schema'
import {
  assignedItemIds,
  isTestRater,
  normalizeRater,
  raterSlot,
  sideSwapped,
} from '@/lib/rating/assignment'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

/**
 * GET /api/ratings/items?rater=NAME[&slot=N]
 *
 * Returns the rater's assigned items, blinded: no provenance, A/B display
 * side randomized deterministically per rater, personalized order.
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
      teams: ratingItems.teams,
      map: ratingItems.map,
      tier: ratingItems.tier,
      // provenance intentionally NOT selected — raters are blind to it
    })
    .from(ratingItems)
  if (rows.length === 0) {
    return NextResponse.json({ error: 'no rating items seeded' }, { status: 500 })
  }

  const slot = raterSlot(rater, slotOverride)
  const ids = assignedItemIds(rows.map((r) => r.id), rater, slot)
  const byId = new Map(rows.map((r) => [r.id, r]))

  const rated = await db
    .select({ itemId: draftRatings.itemId })
    .from(draftRatings)
    .where(eq(draftRatings.rater, normalizeRater(rater)))
  const ratedIds = rated.map((r) => r.itemId)

  const items = ids.map((id) => {
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
  })

  return NextResponse.json({
    rater,
    slot,
    isTest: isTestRater(rater),
    items,
    ratedItemIds: ratedIds,
  })
}
