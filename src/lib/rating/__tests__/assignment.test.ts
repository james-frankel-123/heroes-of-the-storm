import { describe, expect, it } from 'vitest'
import {
  NUM_SLOTS,
  TEST_ITEM_COUNT,
  assignedItemIds,
  isTestRater,
  raterSlot,
  sideSwapped,
} from '../assignment'

const ALL_IDS = Array.from({ length: 100 }, (_, i) => i + 1)

describe('assignedItemIds', () => {
  it('gives every slot exactly 30 items', () => {
    for (let slot = 0; slot < NUM_SLOTS; slot++) {
      expect(assignedItemIds(ALL_IDS, `rater-${slot}`, slot)).toHaveLength(30)
    }
  })

  it('covers every item by exactly 3 of the 10 slots (latin-square balance)', () => {
    const coverage = new Map<number, number>()
    for (let slot = 0; slot < NUM_SLOTS; slot++) {
      for (const id of assignedItemIds(ALL_IDS, `rater-${slot}`, slot)) {
        coverage.set(id, (coverage.get(id) ?? 0) + 1)
      }
    }
    expect(coverage.size).toBe(100)
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

  it('gives test raters only 5 items', () => {
    expect(assignedItemIds(ALL_IDS, 'test-claude', 0)).toHaveLength(TEST_ITEM_COUNT)
    expect(assignedItemIds(ALL_IDS, 'Test Person', 3)).toHaveLength(TEST_ITEM_COUNT)
  })

  it('does not depend on input id order', () => {
    const shuffledInput = [...ALL_IDS].reverse()
    expect(assignedItemIds(shuffledInput, 'Alice', 4)).toEqual(assignedItemIds(ALL_IDS, 'Alice', 4))
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

describe('sideSwapped', () => {
  it('is deterministic and varies across items', () => {
    const swaps = ALL_IDS.map((id) => sideSwapped('Alice', id))
    expect(ALL_IDS.map((id) => sideSwapped('Alice', id))).toEqual(swaps)
    // Not all the same value — the blinding actually flips sides sometimes.
    expect(new Set(swaps).size).toBe(2)
    // Roughly balanced (loose bound).
    const flips = swaps.filter(Boolean).length
    expect(flips).toBeGreaterThan(20)
    expect(flips).toBeLessThan(80)
  })
})
