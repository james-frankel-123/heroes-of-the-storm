/**
 * Deterministic assignment + blinding logic for the expert draft-rating study.
 *
 * Design:
 * - 100 items, 10 rater slots. Items (sorted by id) are split into 10 blocks
 *   of 10; slot s is assigned blocks {s, s+1, s+2} (mod 10) → 30 items each,
 *   and every item is covered by exactly 3 of the 10 slots (latin-square
 *   style balanced coverage).
 * - Item order and A/B display side are randomized per rater with a
 *   deterministic seeded PRNG so a rater always sees the same thing, but the
 *   canonical team0/team1 (and strategy provenance, which lives server-side
 *   only) never leaks through consistent positioning.
 * - Test raters (name starting with "test") get 5 items only and their
 *   ratings are flagged is_test.
 */

export const NUM_SLOTS = 10
export const BLOCKS_PER_RATER = 3
export const TEST_ITEM_COUNT = 5

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
 * Rater slot 0-9. An explicit slot (from the invite link) takes precedence so
 * the 10 real raters can be assigned distinct slots; otherwise fall back to a
 * name hash.
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
 * The set of item ids assigned to a slot, in the rater's personalized order.
 * `allItemIds` is the full item set (any order; sorted internally).
 */
export function assignedItemIds(allItemIds: number[], rater: string, slot: number): number[] {
  const sorted = allItemIds.slice().sort((a, b) => a - b)
  const blockSize = Math.ceil(sorted.length / NUM_SLOTS)
  const picked: number[] = []
  for (let b = 0; b < BLOCKS_PER_RATER; b++) {
    const block = (slot + b) % NUM_SLOTS
    picked.push(...sorted.slice(block * blockSize, (block + 1) * blockSize))
  }
  // Personalized presentation order (deterministic per rater name).
  const ordered = seededShuffle(picked, fnv1a('order|' + normalizeRater(rater)))
  return isTestRater(rater) ? ordered.slice(0, TEST_ITEM_COUNT) : ordered
}

/**
 * Whether the displayed A/B sides are swapped relative to canonical
 * team0/team1 for this rater+item. Deterministic so the server can recompute
 * the mapping at scoring time.
 */
export function sideSwapped(rater: string, itemId: number): boolean {
  return (fnv1a('side|' + normalizeRater(rater) + '|' + itemId) & 1) === 1
}
