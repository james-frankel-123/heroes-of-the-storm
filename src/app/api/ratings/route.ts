import { NextResponse } from 'next/server'
import { sql } from 'drizzle-orm'
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

interface RatingPayload {
  rater?: unknown
  slot?: unknown
  itemId?: unknown
  pTeamA?: unknown
  betterTeam?: unknown
  confidence?: unknown
  msTaken?: unknown
}

/**
 * POST /api/ratings
 * Body: { rater, slot?, itemId, pTeamA (0-100), betterTeam ('A'|'B'), confidence (1-5), msTaken? }
 * Upserts one rating per (rater, item).
 */
export async function POST(req: Request) {
  let body: RatingPayload
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: 'invalid JSON body' }, { status: 400 })
  }

  const rater = typeof body.rater === 'string' ? body.rater.trim() : ''
  if (!rater || rater.length > 100) {
    return NextResponse.json({ error: 'rater required (1-100 chars)' }, { status: 400 })
  }

  let slotOverride: number | undefined
  if (body.slot !== undefined && body.slot !== null) {
    if (
      !Number.isInteger(body.slot) ||
      (body.slot as number) < 0 ||
      (body.slot as number) >= NUM_SLOTS
    ) {
      return NextResponse.json(
        { error: `slot must be an integer 0-${NUM_SLOTS - 1}` },
        { status: 400 }
      )
    }
    slotOverride = body.slot as number
  }

  const itemId = body.itemId
  if (!Number.isInteger(itemId)) {
    return NextResponse.json({ error: 'itemId must be an integer' }, { status: 400 })
  }

  const pTeamA = body.pTeamA
  if (!Number.isInteger(pTeamA) || (pTeamA as number) < 0 || (pTeamA as number) > 100) {
    return NextResponse.json({ error: 'pTeamA must be an integer 0-100' }, { status: 400 })
  }

  const betterTeam = body.betterTeam
  if (betterTeam !== 'A' && betterTeam !== 'B') {
    return NextResponse.json({ error: "betterTeam must be 'A' or 'B'" }, { status: 400 })
  }

  const confidence = body.confidence
  if (!Number.isInteger(confidence) || (confidence as number) < 1 || (confidence as number) > 5) {
    return NextResponse.json({ error: 'confidence must be an integer 1-5' }, { status: 400 })
  }

  let msTaken: number | null = null
  if (body.msTaken !== undefined && body.msTaken !== null) {
    if (!Number.isInteger(body.msTaken) || (body.msTaken as number) < 0) {
      return NextResponse.json({ error: 'msTaken must be a non-negative integer' }, { status: 400 })
    }
    msTaken = Math.min(body.msTaken as number, 2_000_000_000)
  }

  // Look up the item's block and verify it is in this rater's fixed 240-item
  // sequence. Screener, calibration, and catch items are shared (every rater
  // rates all of them); pairs and anchors must fall in the rater's slot
  // assignment. The v4 design has no open pool: everything is fixed.
  const items = await db
    .select({ id: ratingItems.id, block: ratingItems.block, tier: ratingItems.tier })
    .from(ratingItems)
  const target = items.find((r) => r.id === (itemId as number))
  if (!target) {
    return NextResponse.json({ error: 'unknown itemId' }, { status: 400 })
  }
  const anchorsByTier = new Map<string, number[]>()
  for (const r of items) {
    if (r.block !== 'anchors') continue
    if (!anchorsByTier.has(r.tier)) anchorsByTier.set(r.tier, [])
    anchorsByTier.get(r.tier)!.push(r.id)
  }
  const pool: PoolIds = {
    screener: items.filter((r) => r.block === 'screener').map((r) => r.id),
    calibration: items.filter((r) => r.block === 'calibration').map((r) => r.id),
    pairs: items.filter((r) => r.block === 'pairs').map((r) => r.id),
    anchorsByTier,
    catch: items.filter((r) => r.block === 'catch').map((r) => r.id),
  }
  const slot = raterSlot(rater, slotOverride)
  const sequence = fullSequence(pool, rater, slot)
  if (!sequence.includes(itemId as number)) {
    return NextResponse.json({ error: "item is not in this rater's assignment" }, { status: 400 })
  }

  const normalized = normalizeRater(rater)
  const values = {
    rater: normalized,
    itemId: itemId as number,
    pTeamA: pTeamA as number,
    betterTeam,
    confidence: confidence as number,
    msTaken,
    teamAIsTeam0: !sideSwapped(rater, itemId as number),
    isTest: isTestRater(rater),
    block: target.block,
  }

  await db
    .insert(draftRatings)
    .values(values)
    .onConflictDoUpdate({
      target: [draftRatings.rater, draftRatings.itemId],
      set: {
        pTeamA: values.pTeamA,
        betterTeam: values.betterTeam,
        confidence: values.confidence,
        msTaken: values.msTaken,
        teamAIsTeam0: values.teamAIsTeam0,
        isTest: values.isTest,
        block: values.block,
        createdAt: sql`now()`,
      },
    })

  return NextResponse.json({ ok: true })
}
