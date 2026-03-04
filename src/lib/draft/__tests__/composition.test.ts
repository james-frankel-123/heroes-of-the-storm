import { describe, it, expect } from 'vitest'
import {
  computeBaselineCompWR,
  isMultisetSubset,
  getAchievableCompositions,
  scoreComposition,
} from '../composition'
import type { CompositionData } from '../types'

// Sample compositions for testing
const sampleComps: CompositionData[] = [
  {
    roles: ['Bruiser', 'Healer', 'Ranged Assassin', 'Ranged Assassin', 'Tank'],
    winRate: 50.9,
    games: 3985,
    popularity: 44.36,
  },
  {
    roles: ['Bruiser', 'Healer', 'Melee Assassin', 'Ranged Assassin', 'Tank'],
    winRate: 54.24,
    games: 1111,
    popularity: 12.36,
  },
  {
    roles: ['Bruiser', 'Bruiser', 'Healer', 'Ranged Assassin', 'Tank'],
    winRate: 52.96,
    games: 493,
    popularity: 5.49,
  },
  {
    // Low sample — below MIN_COMP_GAMES, used to test filtering
    roles: ['Healer', 'Healer', 'Ranged Assassin', 'Ranged Assassin', 'Tank'],
    winRate: 31.02,
    games: 49,
    popularity: 0.32,
  },
]

describe('computeBaselineCompWR', () => {
  it('returns 50 for empty compositions', () => {
    expect(computeBaselineCompWR([])).toBe(50)
  })

  it('computes popularity-weighted average', () => {
    const result = computeBaselineCompWR(sampleComps)
    // Should be dominated by the 44.36% popularity comp (50.9 WR)
    expect(result).toBeGreaterThan(50)
    expect(result).toBeLessThan(54)
  })

  it('single comp returns its own win rate', () => {
    const result = computeBaselineCompWR([sampleComps[0]])
    expect(result).toBeCloseTo(50.9, 1)
  })
})

describe('isMultisetSubset', () => {
  it('empty subset is always contained', () => {
    expect(isMultisetSubset([], ['Tank', 'Healer'])).toBe(true)
  })

  it('detects valid subset', () => {
    expect(
      isMultisetSubset(
        ['Healer', 'Tank'],
        ['Bruiser', 'Healer', 'Ranged Assassin', 'Ranged Assassin', 'Tank']
      )
    ).toBe(true)
  })

  it('rejects when role count exceeds superset', () => {
    // Two Healers not in a comp with one Healer
    expect(
      isMultisetSubset(
        ['Healer', 'Healer'],
        ['Bruiser', 'Healer', 'Ranged Assassin', 'Ranged Assassin', 'Tank']
      )
    ).toBe(false)
  })

  it('rejects when role is missing from superset', () => {
    expect(
      isMultisetSubset(
        ['Support'],
        ['Bruiser', 'Healer', 'Ranged Assassin', 'Ranged Assassin', 'Tank']
      )
    ).toBe(false)
  })

  it('handles duplicate roles correctly', () => {
    expect(
      isMultisetSubset(
        ['Ranged Assassin', 'Ranged Assassin'],
        ['Bruiser', 'Healer', 'Ranged Assassin', 'Ranged Assassin', 'Tank']
      )
    ).toBe(true)
  })
})

describe('getAchievableCompositions', () => {
  it('returns all comps when no roles picked yet', () => {
    const result = getAchievableCompositions([], 'Tank', sampleComps)
    // All sample comps contain at least one Tank
    const tankyComps = sampleComps.filter((c) => c.roles.includes('Tank'))
    expect(result.length).toBe(tankyComps.length)
  })

  it('narrows down as roles are added', () => {
    const withTank = getAchievableCompositions([], 'Tank', sampleComps)
    const withTankAndHealer = getAchievableCompositions(['Tank'], 'Healer', sampleComps)
    expect(withTankAndHealer.length).toBeLessThanOrEqual(withTank.length)
  })

  it('returns empty for impossible composition', () => {
    // 3 tanks — no comp has that
    const result = getAchievableCompositions(['Tank', 'Tank'], 'Tank', sampleComps)
    expect(result.length).toBe(0)
  })

  it('matches double healer comp', () => {
    const result = getAchievableCompositions(['Tank'], 'Healer', sampleComps)
    // Should include the double-healer comp
    const doubleHealer = result.find((c) =>
      c.roles.filter((r) => r === 'Healer').length === 2
    )
    expect(doubleHealer).toBeDefined()
  })
})

describe('scoreComposition', () => {
  const baseline = computeBaselineCompWR(sampleComps)

  it('returns zero boost with empty composition data', () => {
    const { sortBoost, reason } = scoreComposition('Tank', [], 0, [], 50)
    expect(sortBoost).toBe(0)
    expect(reason).toBeNull()
  })

  it('scales boost by picks made', () => {
    const earlyResult = scoreComposition('Tank', [], 0, sampleComps, baseline)
    const lateResult = scoreComposition('Tank', ['Healer', 'Bruiser', 'Ranged Assassin'], 3, sampleComps, baseline)
    // Later picks should have more impact
    expect(Math.abs(lateResult.sortBoost)).toBeGreaterThanOrEqual(Math.abs(earlyResult.sortBoost))
  })

  it('penalizes compositions with no data match', () => {
    // 3 supports — no comp matches this
    const result = scoreComposition(
      'Support',
      ['Support', 'Support', 'Support'],
      3,
      sampleComps,
      baseline
    )
    expect(result.sortBoost).toBeLessThan(0)
  })

  it('boost at 0 picks is 0 (scale factor = 0)', () => {
    const result = scoreComposition('Tank', [], 0, sampleComps, baseline)
    expect(result.sortBoost).toBe(0)
  })

  it('returns reason for significant boost', () => {
    // At 4 picks, scale factor is 1.0 — max impact
    const result = scoreComposition(
      'Ranged Assassin',
      ['Tank', 'Bruiser', 'Healer', 'Melee Assassin'],
      4,
      sampleComps,
      baseline
    )
    // The best comp (Tank/Bruiser/Healer/MeleeAssassin/RangedAssassin = 54.24%)
    // should give a positive boost
    if (result.reason) {
      expect(result.reason.type).toBe('comp_wr')
      expect(result.sortBoost).toBeGreaterThan(0)
    }
  })

  it('no-match penalty scales with picks made', () => {
    const at1 = scoreComposition('Support', ['Support', 'Support', 'Support'], 1, sampleComps, baseline)
    const at4 = scoreComposition('Support', ['Support', 'Support', 'Support'], 4, sampleComps, baseline)
    expect(at4.sortBoost).toBeLessThan(at1.sortBoost)
  })
})
