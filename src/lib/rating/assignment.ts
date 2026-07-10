/**
 * Deterministic assignment + blinding logic for the expert draft-rating study.
 *
 * Design:
 * - CALIBRATION: 20 shared known-outcome anchor items rated by EVERY rater
 *   FIRST, before their latin-square core 30 (progress reads "1 / 50"). Same
 *   blinding as everything else; presentation order is a deterministic
 *   per-rater shuffle. Ratings carry block='calibration'.
 * - CORE: 100 items, 10 rater slots. Core items (sorted by id) are split into
 *   10 blocks of 10; slot s is assigned blocks {s, s+1, s+2} (mod 10) → 30
 *   items each, and every core item is covered by exactly 3 of the 10 slots
 *   (latin-square style balanced coverage). This is the pre-registered design;
 *   its statistics (30 items/rater, 3 ratings/item) are unchanged.
 * - EXTENDED: 700 items served AFTER a rater finishes their calibration 20 +
 *   core 30, uncapped. Ordered per rater by a deterministic per-rater shuffle
 *   (PRIMARY) with global under-coverage as a STABLE tiebreak, so a returning
 *   rater resumes at a stable position and never re-sees a rated item even as
 *   global coverage shifts underneath them.
 * - Item order and A/B display side are randomized per rater with a
 *   deterministic seeded PRNG so a rater always sees the same thing, but the
 *   canonical team0/team1 (and strategy provenance, which lives server-side
 *   only) never leaks through consistent positioning.
 * - Test raters (name starting with "test") get a 5-item smoke flow (2
 *   calibration + 3 core) and their ratings are flagged is_test; they may
 *   still enter the extended arm. Names starting with "testfull" get the
 *   complete real assignment (20 calibration + 30 core), still flagged
 *   is_test — used for end-to-end verification without real invites.
 */

export const NUM_SLOTS = 10
export const BLOCKS_PER_RATER = 3
export const TEST_CALIBRATION_COUNT = 2
export const TEST_CORE_COUNT = 3
export const TEST_ITEM_COUNT = TEST_CALIBRATION_COUNT + TEST_CORE_COUNT

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
 * Test rater that receives the FULL real assignment (20 calibration + 30
 * core) while still being flagged is_test. Lets us verify the complete
 * 50-item flow end-to-end without sending a real invite.
 */
export function isFullTestRater(rater: string): boolean {
  return normalizeRater(rater).startsWith('testfull')
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
 * The set of CORE item ids assigned to a slot, in the rater's personalized
 * order. `coreItemIds` is the full core item set (any order; sorted
 * internally). Callers must pass only block='core' ids.
 */
export function assignedItemIds(coreItemIds: number[], rater: string, slot: number): number[] {
  const sorted = coreItemIds.slice().sort((a, b) => a - b)
  const blockSize = Math.ceil(sorted.length / NUM_SLOTS)
  const picked: number[] = []
  for (let b = 0; b < BLOCKS_PER_RATER; b++) {
    const block = (slot + b) % NUM_SLOTS
    picked.push(...sorted.slice(block * blockSize, (block + 1) * blockSize))
  }
  // Personalized presentation order (deterministic per rater name).
  const ordered = seededShuffle(picked, fnv1a('order|' + normalizeRater(rater)))
  return isTestRater(rater) && !isFullTestRater(rater)
    ? ordered.slice(0, TEST_CORE_COUNT)
    : ordered
}

/**
 * The shared CALIBRATION block for one rater: ALL calibration item ids, in a
 * deterministic per-rater presentation order. Every rater rates every
 * calibration item, and always BEFORE their core assignment. Callers must
 * pass only block='calibration' ids.
 */
export function calibrationOrder(calibrationItemIds: number[], rater: string): number[] {
  const ordered = seededShuffle(
    calibrationItemIds.slice().sort((a, b) => a - b),
    fnv1a('calib|' + normalizeRater(rater))
  )
  return isTestRater(rater) && !isFullTestRater(rater)
    ? ordered.slice(0, TEST_CALIBRATION_COUNT)
    : ordered
}

/**
 * Whether the displayed A/B sides are swapped relative to canonical
 * team0/team1 for this rater+item. Deterministic so the server can recompute
 * the mapping at scoring time.
 */
export function sideSwapped(rater: string, itemId: number): boolean {
  return (fnv1a('side|' + normalizeRater(rater) + '|' + itemId) & 1) === 1
}

/**
 * Serving order for the EXTENDED pool for one rater.
 *
 * `extendedItemIds` — all block='extended' ids. `coverage` — current global
 * rating count per extended item id (0 if absent). `alreadyRated` — item ids
 * this rater has already rated (excluded from the result).
 *
 * Ordering is a deterministic per-rater shuffle of the pool (PRIMARY key), with
 * global under-coverage as a STABLE TIEBREAK. Making the per-rater shuffle the
 * primary key is what guarantees resume stability: a returning rater's ordering
 * of the items they have NOT yet rated is a pure function of (rater, itemId), so
 * it does not shift when *other* raters rate items and change global coverage —
 * the exact next item is reproducible across reloads and devices, and rated
 * items are always filtered out, so resume never repeats or skips work.
 *
 * Coverage is still honored two ways: (1) the per-rater shuffles are mutually
 * independent, so across the ~10 raters the pool is walked in different orders
 * and global coverage spreads out evenly on its own; (2) where two items would
 * otherwise tie, the less-covered one is served first. (Per Max: stability of a
 * returning rater's queue takes precedence over live coverage re-ranking, so
 * prioritization is applied as a stable tiebreak rather than a volatile primary
 * sort.)
 */
export function extendedOrder(
  extendedItemIds: number[],
  rater: string,
  coverage: Map<number, number>,
  alreadyRated: Iterable<number> = []
): number[] {
  const ratedSet = new Set(alreadyRated)
  const seed = fnv1a('ext|' + normalizeRater(rater))
  // Stable per-rater rank: position in a deterministic shuffle of ALL extended
  // ids (independent of coverage). This is the PRIMARY sort key.
  const rank = new Map<number, number>()
  seededShuffle(
    extendedItemIds.slice().sort((a, b) => a - b),
    seed
  ).forEach((id, i) => rank.set(id, i))
  return extendedItemIds
    .filter((id) => !ratedSet.has(id))
    .sort((a, b) => {
      const ra = rank.get(a) ?? 0
      const rb = rank.get(b) ?? 0
      if (ra !== rb) return ra - rb
      // Tiebreak (ranks are unique in practice): under-covered first.
      return (coverage.get(a) ?? 0) - (coverage.get(b) ?? 0)
    })
}
