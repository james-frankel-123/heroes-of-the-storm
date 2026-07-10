/**
 * Deterministic assignment + blinding logic for the expert draft-rating study
 * (v4: paid raters, fixed 240-item assignment, 14 slots).
 *
 * Every rater's sequence is exactly 240 items:
 *   - Positions 1-48: the 8 shared SCREENER items INTERLEAVED among the 40
 *     shared CALIBRATION items in a deterministic per-rater shuffle (no
 *     consecutive screener run — dilutes "weird = fake" priming; the >= 7/8
 *     screener gate is computed from the same 8 items regardless of order).
 *   - Positions 49-240: the rater's 60 assigned PAIRS + 129 assigned ANCHORS
 *     in a deterministic per-rater shuffle, with the 3 shared CATCH items at
 *     fixed positions 121 / 181 / 231 (late-session attention checks,
 *     >= 2/3 gate).
 *
 * PAIRS (280): sorted ids split into 14 blocks of 20; slot s is assigned
 * blocks {s, s+1, s+2} (mod 14) → 60 pairs each; every pair is covered by
 * exactly 3 of the 14 slots.
 *
 * ANCHORS (570 = 190/tier): per tier, a GLOBAL seeded order (not per-rater —
 * coverage must be a function of slot only) is dealt cyclically: slot s takes
 * positions [43s, 43s+43) of the sequence index i % 190 → 43 anchors per
 * tier per rater (129 total); each anchor is covered by exactly 3 slots
 * except the first 32 per tier in the dealt order, which get 4
 * (14 × 43 = 602 = 3 × 190 + 32). A slot's 43-wide window can never contain
 * the same anchor twice (windows are narrower than the 190-item cycle).
 *
 * Item order and A/B display side are randomized per rater with a
 * deterministic seeded PRNG so a rater always sees the same thing, but the
 * canonical team0/team1 (and provenance, which lives server-side only) never
 * leaks through consistent positioning. Item ids are globally shuffled at
 * generation, so neither id nor position reveals an item's block.
 *
 * Test raters (name starting with "test") get a 7-item smoke flow (2
 * screener + 2 calibration + 3 assigned) and their ratings are flagged
 * is_test. Names starting with "testfull" get the complete real 240-item
 * sequence, still flagged is_test — used for end-to-end verification.
 */

export const NUM_SLOTS = 14
export const PAIR_BLOCKS_PER_RATER = 3
export const PAIRS_PER_RATER = 60 // 280 / 14 * 3
export const ANCHORS_PER_TIER_PER_RATER = 43 // × 3 tiers = 129
export const SEQUENCE_LENGTH = 240 // 48 + 60 + 129 + 3
/** 1-indexed positions of the shared catch items in the 240-item sequence. */
export const CATCH_POSITIONS = [121, 181, 231]
export const TEST_SCREENER_COUNT = 2
export const TEST_CALIBRATION_COUNT = 2
export const TEST_ASSIGNED_COUNT = 3
export const TEST_ITEM_COUNT =
  TEST_SCREENER_COUNT + TEST_CALIBRATION_COUNT + TEST_ASSIGNED_COUNT

/** FNV-1a 32-bit hash of a string. */
export function fnv1a(str: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/** Deterministic PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Seeded Fisher-Yates shuffle (returns a new array). */
export function seededShuffle<T>(arr: T[], seed: number): T[] {
  const out = arr.slice()
  const rand = mulberry32(seed)
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

export function normalizeRater(rater: string): string {
  return rater.trim().toLowerCase()
}

export function isTestRater(rater: string): boolean {
  return normalizeRater(rater).startsWith('test')
}

/**
 * Test rater that receives the FULL real 240-item assignment while still
 * being flagged is_test. Lets us verify the complete flow end-to-end without
 * sending a real invite.
 */
export function isFullTestRater(rater: string): boolean {
  return normalizeRater(rater).startsWith('testfull')
}

/**
 * Rater slot 0-13. An explicit slot (from the invite link) takes precedence
 * so the 14 real raters can be assigned distinct slots; otherwise fall back
 * to a name hash.
 */
export function raterSlot(rater: string, slotOverride?: number | null): number {
  if (
    slotOverride !== undefined &&
    slotOverride !== null &&
    Number.isInteger(slotOverride) &&
    slotOverride >= 0 &&
    slotOverride < NUM_SLOTS
  ) {
    return slotOverride
  }
  return fnv1a(normalizeRater(rater)) % NUM_SLOTS
}

/**
 * The PAIR item ids assigned to a slot (unordered — ordering happens in
 * fullSequence). `pairItemIds` is the full block='pairs' id set (any order;
 * sorted internally into 14 blocks of 20).
 */
export function assignedPairIds(pairItemIds: number[], slot: number): number[] {
  const sorted = pairItemIds.slice().sort((a, b) => a - b)
  const blockSize = Math.ceil(sorted.length / NUM_SLOTS)
  const picked: number[] = []
  for (let b = 0; b < PAIR_BLOCKS_PER_RATER; b++) {
    const block = (slot + b) % NUM_SLOTS
    picked.push(...sorted.slice(block * blockSize, (block + 1) * blockSize))
  }
  return picked
}

/**
 * The ANCHOR item ids assigned to a slot (unordered): per tier, a cyclic
 * deal over a GLOBAL seeded order — slot s takes positions [43s, 43s+43) of
 * the sequence index i % tierCount. Coverage is a pure function of slot, so
 * each anchor is rated by exactly 3 slots (4 for the first
 * 14×43−3×190 = 32 per tier in the dealt order).
 */
export function assignedAnchorIds(anchorIdsByTier: Map<string, number[]>, slot: number): number[] {
  const picked: number[] = []
  for (const tier of [...anchorIdsByTier.keys()].sort()) {
    const ids = anchorIdsByTier.get(tier)!
    const order = seededShuffle(
      ids.slice().sort((a, b) => a - b),
      fnv1a('anchordeal|' + tier)
    )
    const start = slot * ANCHORS_PER_TIER_PER_RATER
    for (let i = start; i < start + ANCHORS_PER_TIER_PER_RATER; i++) {
      picked.push(order[i % order.length])
    }
  }
  return picked
}

export interface PoolIds {
  screener: number[]
  calibration: number[]
  pairs: number[]
  anchorsByTier: Map<string, number[]>
  catch: number[]
}

/**
 * The rater's complete 240-item sequence, in serving order:
 * interleaved screener+calibration (48), then shuffled pairs+anchors (189)
 * with the catch items inserted at fixed positions 121/181/231.
 * Test raters ("test…" but not "testfull…") get the 7-item smoke flow.
 */
export function fullSequence(pool: PoolIds, rater: string, slot: number): number[] {
  const name = normalizeRater(rater)
  const first48 = seededShuffle(
    [...pool.screener, ...pool.calibration].sort((a, b) => a - b),
    fnv1a('first48|' + name)
  )
  const assigned = seededShuffle(
    [...assignedPairIds(pool.pairs, slot), ...assignedAnchorIds(pool.anchorsByTier, slot)],
    fnv1a('order|' + name)
  )
  if (isTestRater(rater) && !isFullTestRater(rater)) {
    const screenerSet = new Set(pool.screener)
    const smoke = [
      ...first48.filter((id) => screenerSet.has(id)).slice(0, TEST_SCREENER_COUNT),
      ...first48.filter((id) => !screenerSet.has(id)).slice(0, TEST_CALIBRATION_COUNT),
      ...assigned.slice(0, TEST_ASSIGNED_COUNT),
    ]
    return smoke
  }
  const seq = [...first48, ...assigned]
  // Insert catch items at their fixed 1-indexed positions. Insertions are in
  // ascending position order, so each splice lands exactly at CATCH_POSITIONS.
  const catchOrdered = seededShuffle(
    pool.catch.slice().sort((a, b) => a - b),
    fnv1a('catch|' + name)
  )
  CATCH_POSITIONS.forEach((pos, i) => {
    if (catchOrdered[i] !== undefined) seq.splice(pos - 1, 0, catchOrdered[i])
  })
  return seq
}

/**
 * Whether the displayed A/B sides are swapped relative to canonical
 * team0/team1 for this rater+item. Deterministic so the server can recompute
 * the mapping at scoring time.
 */
export function sideSwapped(rater: string, itemId: number): boolean {
  return (fnv1a('side|' + normalizeRater(rater) + '|' + itemId) & 1) === 1
}
