import { describe, it, expect } from 'vitest'
import { computeTeamWinEstimate } from '../win-estimate'
import type { DraftData } from '../types'

function makeData(overrides: Partial<DraftData> = {}): DraftData {
  return {
    heroStats: {},
    heroMapWinRates: {},
    synergies: {},
    counters: {},
    playerStats: {},
    playerMapStats: {},
    compositions: [],
    baselineCompWR: 50,
    ...overrides,
  }
}

describe('computeTeamWinEstimate', () => {
  it('returns 50% for empty picks', () => {
    const result = computeTeamWinEstimate([], [], makeData())
    expect(result.winPct).toBe(50)
    expect(result.breakdown.heroWR).toBe(0)
    expect(result.breakdown.synergies).toBe(0)
    expect(result.breakdown.counters).toBe(0)
    expect(result.breakdown.playerAdj).toBe(0)
  })

  it('adds hero base WR deltas', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 54, pickRate: 10, banRate: 5, games: 1000 },
        Raynor: { winRate: 48, pickRate: 8, banRate: 2, games: 1000 },
      },
    })
    const result = computeTeamWinEstimate(['Jaina', 'Raynor'], [], data)
    // heroWR delta: (54-50) + (48-50) = 4 + (-2) = 2
    expect(result.breakdown.heroWR).toBe(2)
    expect(result.winPct).toBe(52)
  })

  it('adds synergy deltas for ally pairs (counted once per pair)', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 },
        Arthas: { winRate: 50, pickRate: 8, banRate: 2, games: 1000 },
      },
      synergies: {
        Jaina: { Arthas: { winRate: 55, games: 100 } },
        Arthas: { Jaina: { winRate: 55, games: 100 } },
      },
    })
    const result = computeTeamWinEstimate(['Jaina', 'Arthas'], [], data)
    // Only one pair (Jaina, Arthas) → delta = 55-50 = 5, counted once
    expect(result.breakdown.synergies).toBe(5)
    expect(result.winPct).toBe(55)
  })

  it('adds counter deltas for each (ourHero, enemyHero) pair', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 },
      },
      counters: {
        Jaina: { Illidan: { winRate: 56, games: 200 } },
      },
    })
    const result = computeTeamWinEstimate(['Jaina'], ['Illidan'], data)
    expect(result.breakdown.counters).toBe(6)
    expect(result.winPct).toBe(56)
  })

  it('skips synergies/counters with too few games', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 },
        Arthas: { winRate: 50, pickRate: 8, banRate: 2, games: 1000 },
      },
      synergies: {
        Jaina: { Arthas: { winRate: 60, games: 10 } }, // too few games
      },
      counters: {
        Jaina: { Illidan: { winRate: 60, games: 5 } }, // too few games
      },
    })
    const result = computeTeamWinEstimate(['Jaina', 'Arthas'], ['Illidan'], data)
    expect(result.breakdown.synergies).toBe(0)
    expect(result.breakdown.counters).toBe(0)
  })

  it('applies player adjustment replacing hero base WR', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 52, pickRate: 10, banRate: 5, games: 1000 },
      },
      playerStats: {
        'player#123': {
          Jaina: { games: 50, wins: 30, winRate: 60, mawp: 62 },
        },
      },
    })
    const result = computeTeamWinEstimate(
      ['Jaina'], [], data,
      { 0: 'player#123' },
    )
    // heroWR: 52-50 = 2
    // playerAdj: (62-50) - (52-50) = 12 - 2 = 10
    // total: 50 + 2 + 10 = 62
    expect(result.breakdown.heroWR).toBe(2)
    expect(result.breakdown.playerAdj).toBe(10)
    expect(result.winPct).toBe(62)
  })

  it('uses confidence-adjusted MAWP for low-game players', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 },
      },
      playerStats: {
        'player#123': {
          Jaina: { games: 15, wins: 12, winRate: 80, mawp: 70 },
        },
      },
    })
    const result = computeTeamWinEstimate(
      ['Jaina'], [], data,
      { 0: 'player#123' },
    )
    // games=15, threshold=30 → weight=0.5
    // adjusted MAWP = 70*0.5 + 50*0.5 = 60
    // heroWR: 50-50 = 0
    // playerAdj: (60-50) - (50-50) = 10 - 0 = 10
    expect(result.breakdown.playerAdj).toBe(10)
    expect(result.winPct).toBe(60)
  })

  it('clamps result to [1, 99]', () => {
    const data = makeData({
      heroStats: {
        A: { winRate: 90, pickRate: 10, banRate: 5, games: 1000 },
        B: { winRate: 90, pickRate: 10, banRate: 5, games: 1000 },
        C: { winRate: 90, pickRate: 10, banRate: 5, games: 1000 },
      },
    })
    const high = computeTeamWinEstimate(['A', 'B', 'C'], [], data)
    expect(high.winPct).toBeLessThanOrEqual(99)

    const lowData = makeData({
      heroStats: {
        X: { winRate: 10, pickRate: 10, banRate: 5, games: 1000 },
        Y: { winRate: 10, pickRate: 10, banRate: 5, games: 1000 },
      },
    })
    const low = computeTeamWinEstimate(['X', 'Y'], [], lowData)
    expect(low.winPct).toBeGreaterThanOrEqual(1)
  })

  it('combines all factors', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 53, pickRate: 10, banRate: 5, games: 1000 },
        Arthas: { winRate: 51, pickRate: 8, banRate: 2, games: 1000 },
      },
      synergies: {
        Jaina: { Arthas: { winRate: 52, games: 100 } },
      },
      counters: {
        Jaina: { Illidan: { winRate: 54, games: 100 } },
        Arthas: { Illidan: { winRate: 48, games: 100 } },
      },
    })
    const result = computeTeamWinEstimate(
      ['Jaina', 'Arthas'], ['Illidan'], data
    )
    // heroWR: (53-50) + (51-50) = 4
    // synergies: (52-50) = 2
    // counters: (54-50) + (48-50) = 4 + (-2) = 2
    // total: 50 + 4 + 2 + 2 = 58
    expect(result.breakdown.heroWR).toBe(4)
    expect(result.breakdown.synergies).toBe(2)
    expect(result.breakdown.counters).toBe(2)
    expect(result.winPct).toBe(58)
  })

  it('prefers map-specific win rate over overall', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 52, pickRate: 10, banRate: 5, games: 1000 },
      },
      heroMapWinRates: {
        Jaina: { winRate: 58, games: 200 },
      },
    })
    const result = computeTeamWinEstimate(['Jaina'], [], data)
    // Should use map WR (58) not overall (52)
    expect(result.breakdown.heroWR).toBe(8)
    expect(result.winPct).toBe(58)
  })

  it('falls back to overall WR when map data has too few games', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 52, pickRate: 10, banRate: 5, games: 1000 },
      },
      heroMapWinRates: {
        Jaina: { winRate: 70, games: 20 }, // below 50-game threshold
      },
    })
    const result = computeTeamWinEstimate(['Jaina'], [], data)
    // Should use overall WR (52) since map has <50 games
    expect(result.breakdown.heroWR).toBe(2)
    expect(result.winPct).toBe(52)
  })

  it('player adjustment uses map WR as baseline when available', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 52, pickRate: 10, banRate: 5, games: 1000 },
      },
      heroMapWinRates: {
        Jaina: { winRate: 55, games: 200 },
      },
      playerStats: {
        'player#123': {
          Jaina: { games: 50, wins: 30, winRate: 60, mawp: 62 },
        },
      },
    })
    const result = computeTeamWinEstimate(
      ['Jaina'], [], data,
      { 0: 'player#123' },
    )
    // heroWR uses map: 55-50 = 5
    // playerAdj: (62-50) - (55-50) = 12 - 5 = 7
    // total: 50 + 5 + 7 = 62
    expect(result.breakdown.heroWR).toBe(5)
    expect(result.breakdown.playerAdj).toBe(7)
    expect(result.winPct).toBe(62)
  })

  it('ignores player assignment with too few games', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 52, pickRate: 10, banRate: 5, games: 1000 },
      },
      playerStats: {
        'player#123': {
          Jaina: { games: 5, wins: 5, winRate: 100, mawp: 90 },
        },
      },
    })
    const result = computeTeamWinEstimate(
      ['Jaina'], [], data,
      { 0: 'player#123' },
    )
    // Player has <10 games → no player adjustment
    expect(result.breakdown.playerAdj).toBe(0)
    expect(result.winPct).toBe(52)
  })
})
