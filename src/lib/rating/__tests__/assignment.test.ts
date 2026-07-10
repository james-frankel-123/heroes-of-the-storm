import { describe, expect, it } from 'vitest'
import {
  ANCHORS_PER_TIER_PER_RATER,
  CATCH_POSITIONS,
  NUM_SLOTS,
  PAIRS_PER_RATER,
  SEQUENCE_LENGTH,
  TEST_ITEM_COUNT,
  assignedAnchorIds,
  assignedPairIds,
  fullSequence,
  isFullTestRater,
  isTestRater,
  raterSlot,
  sideSwapped,
  type PoolIds,
} from '../assignment'

// v4 pool shape: ids are globally shuffled at generation, so the tests use
// deliberately non-contiguous id ranges to prove nothing relies on ranges.
const SCREENER_IDS = Array.from({ length: 8 }, (_, i) => i * 7 + 3)
const CALIB_IDS = Array.from({ length: 40 }, (_, i) => i * 3 + 100)
const PAIR_IDS = Array.from({ length: 280 }, (_, i) => i * 2 + 301)
const ANCHOR_IDS_BY_TIER = new Map<string, number[]>([
  ['low', Array.from({ length: 190 }, (_, i) => i + 1000)],
  ['mid', Array.from({ length: 190 }, (_, i) => i + 2000)],
  ['high', Array.from({ length: 190 }, (_, i) => i + 3000)],
])
const CATCH_IDS = [4001, 4002, 4003]
const POOL: PoolIds = {
  screener: SCREENER_IDS,
  calibration: CALIB_IDS,
  pairs: PAIR_IDS,
  anchorsByTier: ANCHOR_IDS_BY_TIER,
  catch: CATCH_IDS,
}

describe('assignedPairIds', () => {
  it('gives every slot exactly 60 pairs', () => {
    for (let slot = 0; slot < NUM_SLOTS; slot++) {
      expect(assignedPairIds(PAIR_IDS, slot)).toHaveLength(PAIRS_PER_RATER)
    }
  })

  it('covers every pair by exactly 3 of the 14 slots (latin-square balance)', () => {
    const coverage = new Map<number, number>()
    for (let slot = 0; slot < NUM_SLOTS; slot++) {
      for (const id of assignedPairIds(PAIR_IDS, slot)) {
        coverage.set(id, (coverage.get(id) ?? 0) + 1)
      }
    }
    expect(coverage.size).toBe(280)
    for (const id of PAIR_IDS) expect(coverage.get(id)).toBe(3)
  })

  it('does not depend on input id order', () => {
    expect(assignedPairIds([...PAIR_IDS].reverse(), 4)).toEqual(assignedPairIds(PAIR_IDS, 4))
  })
})

describe('assignedAnchorIds', () => {
  it('gives every slot exactly 43 anchors per tier (129 total), no duplicates', () => {
    for (let slot = 0; slot < NUM_SLOTS; slot++) {
      const ids = assignedAnchorIds(ANCHOR_IDS_BY_TIER, slot)
      expect(ids).toHaveLength(3 * ANCHORS_PER_TIER_PER_RATER)
      expect(new Set(ids).size).toBe(ids.length)
      for (const [tier, tierIds] of ANCHOR_IDS_BY_TIER) {
        const tierSet = new Set(tierIds)
        expect(ids.filter((id) => tierSet.has(id))).toHaveLength(ANCHORS_PER_TIER_PER_RATER)
        void tier
      }
    }
  })

  it('covers every anchor by 3 or 4 slots, totalling 602 judgments per tier', () => {
    const coverage = new Map<number, number>()
    for (let slot = 0; slot < NUM_SLOTS; slot++) {
      for (const id of assignedAnchorIds(ANCHOR_IDS_BY_TIER, slot)) {
        coverage.set(id, (coverage.get(id) ?? 0) + 1)
      }
    }
    for (const [, tierIds] of ANCHOR_IDS_BY_TIER) {
      let total = 0
      let fours = 0
      for (const id of tierIds) {
        const n = coverage.get(id) ?? 0
        expect(n === 3 || n === 4).toBe(true)
        total += n
        if (n === 4) fours++
      }
      expect(total).toBe(NUM_SLOTS * ANCHORS_PER_TIER_PER_RATER) // 602
      expect(fours).toBe(NUM_SLOTS * ANCHORS_PER_TIER_PER_RATER - 3 * 190) // 32
    }
  })

  it('is a pure function of slot (not rater), independent of input order', () => {
    const reversed = new Map(
      [...ANCHOR_IDS_BY_TIER].map(([t, ids]) => [t, [...ids].reverse()])
    )
    expect(assignedAnchorIds(reversed, 6)).toEqual(assignedAnchorIds(ANCHOR_IDS_BY_TIER, 6))
  })
})

describe('fullSequence', () => {
  it('serves exactly 240 items with no duplicates', () => {
    for (const rater of ['Alice', 'Bob', 'testfull-claude']) {
      const seq = fullSequence(POOL, rater, 2)
      expect(seq).toHaveLength(SEQUENCE_LENGTH)
      expect(new Set(seq).size).toBe(SEQUENCE_LENGTH)
    }
  })

  it('interleaves all 8 screener + 40 calibration items in positions 1-48', () => {
    const seq = fullSequence(POOL, 'Alice', 0)
    const first48 = new Set(seq.slice(0, 48))
    for (const id of [...SCREENER_IDS, ...CALIB_IDS]) expect(first48.has(id)).toBe(true)
  })

  it('places the 3 catch items at exactly positions 121/181/231', () => {
    for (const rater of ['Alice', 'Bob', 'Carol']) {
      const seq = fullSequence(POOL, rater, 5)
      const catchSet = new Set(CATCH_IDS)
      const positions = seq
        .map((id, i) => (catchSet.has(id) ? i + 1 : null))
        .filter((x): x is number => x !== null)
      expect(positions).toEqual(CATCH_POSITIONS)
    }
  })

  it("contains exactly the slot's assigned pairs and anchors", () => {
    const seq = fullSequence(POOL, 'Alice', 7)
    const pairSet = new Set(PAIR_IDS)
    const served = seq.filter((id) => pairSet.has(id)).sort((a, b) => a - b)
    const assigned = assignedPairIds(PAIR_IDS, 7).sort((a, b) => a - b)
    expect(served).toEqual(assigned)
  })

  it('is deterministic per rater; same slot = same sets, different order', () => {
    expect(fullSequence(POOL, 'Alice', 2)).toEqual(fullSequence(POOL, 'alice ', 2))
    const a = fullSequence(POOL, 'Alice', 2)
    const b = fullSequence(POOL, 'Bob', 2)
    expect([...a].sort((x, y) => x - y)).toEqual([...b].sort((x, y) => x - y))
    expect(a).not.toEqual(b)
  })

  it('gives abbreviated test raters the 7-item smoke flow', () => {
    const seq = fullSequence(POOL, 'test-claude', 0)
    expect(seq).toHaveLength(TEST_ITEM_COUNT)
    const screenerSet = new Set(SCREENER_IDS)
    const calibSet = new Set(CALIB_IDS)
    expect(seq.filter((id) => screenerSet.has(id))).toHaveLength(2)
    expect(seq.filter((id) => calibSet.has(id))).toHaveLength(2)
  })

  it('gives testfull raters the complete 240-item sequence', () => {
    expect(fullSequence(POOL, 'testfull-claude', 0)).toHaveLength(SEQUENCE_LENGTH)
  })
})

describe('raterSlot', () => {
  it('uses the explicit slot when valid (0-13)', () => {
    expect(raterSlot('anyone', 13)).toBe(13)
    expect(raterSlot('anyone', 0)).toBe(0)
  })

  it('falls back to a stable hash slot otherwise', () => {
    const s = raterSlot('Alice')
    expect(s).toBeGreaterThanOrEqual(0)
    expect(s).toBeLessThan(NUM_SLOTS)
    expect(raterSlot('Alice')).toBe(s)
    expect(raterSlot('Alice', 14)).toBe(s)
    expect(raterSlot('Alice', null)).toBe(s)
  })
})

describe('isTestRater / isFullTestRater', () => {
  it('flags names starting with "test" (case/whitespace-insensitive)', () => {
    expect(isTestRater('test-claude')).toBe(true)
    expect(isTestRater(' TEST person')).toBe(true)
    expect(isTestRater('Ernest')).toBe(false)
    expect(isTestRater('contest')).toBe(false)
  })

  it('flags only names starting with "testfull"; both stay is_test', () => {
    expect(isFullTestRater('testfull-claude')).toBe(true)
    expect(isTestRater('testfull-claude')).toBe(true)
    expect(isFullTestRater('test-claude')).toBe(false)
    expect(isFullTestRater('Ernest')).toBe(false)
  })
})

describe('sideSwapped', () => {
  it('is deterministic and varies across items', () => {
    const swaps = PAIR_IDS.map((id) => sideSwapped('Alice', id))
    expect(PAIR_IDS.map((id) => sideSwapped('Alice', id))).toEqual(swaps)
    // Not all the same value — the blinding actually flips sides sometimes.
    expect(new Set(swaps).size).toBe(2)
    // Roughly balanced (loose bound).
    const flips = swaps.filter(Boolean).length
    expect(flips).toBeGreaterThan(80)
    expect(flips).toBeLessThan(200)
  })
})
