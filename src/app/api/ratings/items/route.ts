import { NextResponse } from 'next/server'
import { eq } from 'drizzle-orm'
import { db } from '@/lib/db'
import { draftRatings, ratingItems } from '@/lib/db/schema'
import {
  fullSequence,
  isTestRater,
  normalizeRater,
  NUM_SLOTS,
  raterSlot,
  sideSwapped,
  type PoolIds,
} from '@/lib/rating/assignment'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

/**
 * GET /api/ratings/items?rater=NAME[&slot=N]
 *
 * Returns the rater's complete blinded 240-item sequence (v4 fixed design:
 * interleaved screener+calibration 48, then 60 pairs + 129 anchors shuffled
 * per rater with 3 catch items at fixed positions; 7-item smoke flow for
 * test raters unless the name starts with "testfull", which gets the full
 * real sequence).
 *
 * Blinded: no provenance, no block labels; A/B display side randomized per
 * rater. Also returns already-rated ids so the client can resume anywhere in
 * the sequence on any device after a hard refresh.
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
    if (!Number.isInteger(n) || n < 0 || n >= NUM_SLOTS) {
      return NextResponse.json(
        { error: `slot must be an integer 0-${NUM_SLOTS - 1}` },
        { status: 400 }
      )
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
  const anchorsByTier = new Map<string, number[]>()
  for (const r of rows) {
    if (r.block !== 'anchors') continue
    if (!anchorsByTier.has(r.tier)) anchorsByTier.set(r.tier, [])
    anchorsByTier.get(r.tier)!.push(r.id)
  }
  const pool: PoolIds = {
    screener: rows.filter((r) => r.block === 'screener').map((r) => r.id),
    calibration: rows.filter((r) => r.block === 'calibration').map((r) => r.id),
    pairs: rows.filter((r) => r.block === 'pairs').map((r) => r.id),
    anchorsByTier,
    catch: rows.filter((r) => r.block === 'catch').map((r) => r.id),
  }

  const slot = raterSlot(rater, slotOverride)
  const sequenceIds = fullSequence(pool, rater, slot)

  // This rater's already-rated ids (for cross-device resume).
  const rated = await db
    .select({ itemId: draftRatings.itemId })
    .from(draftRatings)
    .where(eq(draftRatings.rater, normalizeRater(rater)))
  const ratedIds = rated.map((r) => r.itemId)
  const ratedSet = new Set(ratedIds)

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

  const items = sequenceIds.map(blind)
  const ratedCount = sequenceIds.filter((id) => ratedSet.has(id)).length

  return NextResponse.json({
    rater,
    slot,
    isTest: isTestRater(rater),
    items,
    ratedItemIds: ratedIds,
    ratedCount,
  })
}
