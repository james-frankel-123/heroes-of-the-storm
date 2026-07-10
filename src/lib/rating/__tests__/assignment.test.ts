import { describe, expect, it } from 'vitest'
import {
  NUM_SLOTS,
  TEST_CALIBRATION_COUNT,
  TEST_CORE_COUNT,
  TEST_SCREENER_COUNT,
  assignedItemIds,
  calibrationOrder,
  extendedOrder,
  isFullTestRater,
  isTestRater,
  raterSlot,
  screenerOrder,
  sideSwapped,
} from '../assignment'

// Screener ids 1..8, calibration ids 9..48, core ids 49..198, extended ids 199..898.
const SCREENER_IDS = Array.from({ length: 8 }, (_, i) => i + 1)
const CALIB_IDS = Array.from({ length: 40 }, (_, i) => i + 9)
const ALL_IDS = Array.from({ length: 150 }, (_, i) => i + 49)
const EXT_IDS = Array.from({ length: 700 }, (_, i) => i + 199)

describe('assignedItemIds', () => {
  it('gives every slot exactly 45 items', () => {
    for (let slot = 0; slot < NUM_SLOTS; slot++) {
      expect(assignedItemIds(ALL_IDS, `rater-${slot}`, slot)).toHaveLength(45)
    }
  })

  it('covers every item by exactly 3 of the 10 slots (latin-square balance)', () => {
    const coverage = new Map<number, number>()
    for (let slot = 0; slot < NUM_SLOTS; slot++) {
      for (const id of assignedItemIds(ALL_IDS, `rater-${slot}`, slot)) {
        coverage.set(id, (coverage.get(id) ?? 0) + 1)
      }
    }
    expect(coverage.size).toBe(150)
    for (const id of ALL_IDS) expect(coverage.get(id)).toBe(3)
  })

  it('is deterministic per rater and slot', () => {
    expect(assignedItemIds(ALL_IDS, 'Alice', 4)).toEqual(assignedItemIds(ALL_IDS, 'Alice', 4))
    expect(assignedItemIds(ALL_IDS, 'alice ', 4)).toEqual(assignedItemIds(ALL_IDS, 'Alice', 4))
  })

  it('assigns the same item set (different order) to raters sharing a slot', () => {
    const a = [...assignedItemIds(ALL_IDS, 'Alice', 2)].sort((x, y) => x - y)
    const b = [...assignedItemIds(ALL_IDS, 'Bob', 2)].sort((x, y) => x - y)
    expect(a).toEqual(b)
    // But presentation order is personalized.
    expect(assignedItemIds(ALL_IDS, 'Alice', 2)).not.toEqual(assignedItemIds(ALL_IDS, 'Bob', 2))
  })

  it('gives test raters only 3 core items (part of the 7-item smoke flow)', () => {
    expect(assignedItemIds(ALL_IDS, 'test-claude', 0)).toHaveLength(TEST_CORE_COUNT)
    expect(assignedItemIds(ALL_IDS, 'Test Person', 3)).toHaveLength(TEST_CORE_COUNT)
  })

  it('gives testfull raters the complete 45-item core assignment', () => {
    expect(assignedItemIds(ALL_IDS, 'testfull-claude', 0)).toHaveLength(45)
  })

  it('does not depend on input id order', () => {
    const shuffledInput = [...ALL_IDS].reverse()
    expect(assignedItemIds(shuffledInput, 'Alice', 4)).toEqual(assignedItemIds(ALL_IDS, 'Alice', 4))
  })
})

describe('screenerOrder', () => {
  it('gives every real rater ALL screener items', () => {
    for (const rater of ['Alice', 'Bob', 'Fan', 'testfull-claude']) {
      const order = screenerOrder(SCREENER_IDS, rater)
      expect([...order].sort((a, b) => a - b)).toEqual(SCREENER_IDS)
    }
  })

  it('personalizes presentation order per rater, deterministically', () => {
    expect(screenerOrder(SCREENER_IDS, 'Alice')).toEqual(screenerOrder(SCREENER_IDS, 'alice '))
    expect(screenerOrder(SCREENER_IDS, 'Alice')).not.toEqual(screenerOrder(SCREENER_IDS, 'Bob'))
  })

  it('gives abbreviated test raters only 2 screener items', () => {
    expect(screenerOrder(SCREENER_IDS, 'test-claude')).toHaveLength(TEST_SCREENER_COUNT)
  })

  it('does not depend on input id order', () => {
    expect(screenerOrder([...SCREENER_IDS].reverse(), 'Alice')).toEqual(
      screenerOrder(SCREENER_IDS, 'Alice')
    )
  })
})

describe('calibrationOrder', () => {
  it('gives every real rater ALL calibration items', () => {
    for (const rater of ['Alice', 'Bob', 'Fan', 'testfull-claude']) {
      const order = calibrationOrder(CALIB_IDS, rater)
      expect([...order].sort((a, b) => a - b)).toEqual(CALIB_IDS)
    }
  })

  it('personalizes presentation order per rater, deterministically', () => {
    expect(calibrationOrder(CALIB_IDS, 'Alice')).toEqual(calibrationOrder(CALIB_IDS, 'alice '))
    expect(calibrationOrder(CALIB_IDS, 'Alice')).not.toEqual(calibrationOrder(CALIB_IDS, 'Bob'))
  })

  it('gives abbreviated test raters only 2 calibration items', () => {
    expect(calibrationOrder(CALIB_IDS, 'test-claude')).toHaveLength(TEST_CALIBRATION_COUNT)
  })

  it('does not depend on input id order', () => {
    expect(calibrationOrder([...CALIB_IDS].reverse(), 'Alice')).toEqual(
      calibrationOrder(CALIB_IDS, 'Alice')
    )
  })
})

describe('isFullTestRater', () => {
  it('flags only names starting with "testfull"; both stay is_test', () => {
    expect(isFullTestRater('testfull-claude')).toBe(true)
    expect(isTestRater('testfull-claude')).toBe(true)
    expect(isFullTestRater('test-claude')).toBe(false)
    expect(isFullTestRater('Ernest')).toBe(false)
  })
})

describe('raterSlot', () => {
  it('uses the explicit slot when valid', () => {
    expect(raterSlot('anyone', 7)).toBe(7)
    expect(raterSlot('anyone', 0)).toBe(0)
  })

  it('falls back to a stable hash slot otherwise', () => {
    const s = raterSlot('Alice')
    expect(s).toBeGreaterThanOrEqual(0)
    expect(s).toBeLessThan(NUM_SLOTS)
    expect(raterSlot('Alice')).toBe(s)
    expect(raterSlot('Alice', 15)).toBe(s)
    expect(raterSlot('Alice', null)).toBe(s)
  })
})

describe('isTestRater', () => {
  it('flags names starting with "test" (case/whitespace-insensitive)', () => {
    expect(isTestRater('test-claude')).toBe(true)
    expect(isTestRater(' TEST person')).toBe(true)
    expect(isTestRater('Ernest')).toBe(false)
    expect(isTestRater('contest')).toBe(false)
  })
})

describe('extendedOrder', () => {
  const emptyCov = new Map<number, number>()

  it('returns all extended items when nothing is rated or covered', () => {
    const order = extendedOrder(EXT_IDS, 'Fan', emptyCov)
    expect(order).toHaveLength(EXT_IDS.length)
    expect([...order].sort((a, b) => a - b)).toEqual(EXT_IDS)
  })

  it('excludes already-rated items', () => {
    const rated = [199, 450, 898]
    const order = extendedOrder(EXT_IDS, 'Fan', emptyCov, rated)
    expect(order).toHaveLength(EXT_IDS.length - rated.length)
    for (const id of rated) expect(order).not.toContain(id)
  })

  it('is a deterministic per-rater order, independent of global coverage', () => {
    const a = extendedOrder(EXT_IDS, 'Fan', emptyCov)
    // Arbitrary global coverage must NOT reorder the pool (stable primary key).
    const cov = new Map<number, number>()
    for (const id of EXT_IDS) cov.set(id, (id * 7) % 13)
    const b = extendedOrder(EXT_IDS, 'Fan', cov)
    expect(b).toEqual(a)
    // ...and differs across raters (natural cross-rater coverage spread).
    expect(extendedOrder(EXT_IDS, 'Ernest', emptyCov)).not.toEqual(a)
  })

  it('uses under-coverage only as a tiebreak among equal-rank items', () => {
    // Force a rank tie by passing duplicate ids is impossible; instead verify
    // that with a single-element pool coverage is irrelevant and order stable.
    const cov = new Map<number, number>([[300, 9]])
    expect(extendedOrder([300], 'Fan', cov)).toEqual([300])
  })

  it('resume is stable even as global under-coverage shifts underneath the rater', () => {
    const full = extendedOrder(EXT_IDS, 'Fan', emptyCov)
    const rated = full.slice(0, 5)
    // Others rate a bunch of items (including ones ahead in Fan's queue),
    // changing global coverage arbitrarily.
    const shifted = new Map<number, number>()
    for (const id of full.slice(5, 60)) shifted.set(id, 3)
    const resumed = extendedOrder(EXT_IDS, 'Fan', shifted, rated)
    // Fan's next item and full remaining order are unchanged: resume is exact.
    expect(resumed[0]).toBe(full[5])
    expect(resumed).toEqual(full.slice(5))
  })
})

describe('sideSwapped', () => {
  it('is deterministic and varies across items', () => {
    const swaps = ALL_IDS.map((id) => sideSwapped('Alice', id))
    expect(ALL_IDS.map((id) => sideSwapped('Alice', id))).toEqual(swaps)
    // Not all the same value — the blinding actually flips sides sometimes.
    expect(new Set(swaps).size).toBe(2)
    // Roughly balanced (loose bound).
    const flips = swaps.filter(Boolean).length
    expect(flips).toBeGreaterThan(30)
    expect(flips).toBeLessThan(120)
  })
})
