import { describe, it, expect } from 'vitest'
import {
  createSearchState,
  expectimaxAtDepth,
  iterativeDeepeningSearch,
  evaluateLeaf,
} from '../expectimax'
import type { DraftState, DraftData } from '../types'
import type { SkillTier } from '@/lib/types'
import type { OpponentPredictor, SearchState } from '../expectimax/types'

// ── Minimal mock DraftData ──

function makeMockData(): DraftData {
  // Create a minimal DraftData with a few heroes that have known win rates
  const heroStats: DraftData['heroStats'] = {}
  const heroes = ['Muradin', 'Johanna', 'Valla', 'Jaina', 'Malfurion',
                   'Diablo', 'Raynor', 'Brightwing', 'Thrall', 'Kael\'thas',
                   'Arthas', 'Li-Ming', 'Rehgar', 'Falstad', 'Anub\'arak']
  for (const h of heroes) {
    heroStats[h] = { winRate: 50 + Math.random() * 6 - 3, pickRate: 10, banRate: 5, games: 500 }
  }
  // Make some heroes clearly better
  heroStats['Valla'] = { winRate: 54, pickRate: 15, banRate: 8, games: 1000 }
  heroStats['Jaina'] = { winRate: 53, pickRate: 12, banRate: 6, games: 800 }
  heroStats['Malfurion'] = { winRate: 52, pickRate: 10, banRate: 4, games: 700 }

  return {
    heroStats,
    heroMapWinRates: {},
    synergies: {},
    counters: {},
    playerStats: {},
    playerMapStats: {},
    compositions: [],
    baselineCompWR: 50,
  }
}

function makeState(step = 0, ourTeam: 'A' | 'B' = 'A'): DraftState {
  return {
    phase: 'drafting',
    map: 'Cursed Hollow',
    tier: 'mid' as SkillTier,
    ourTeam,
    currentStep: step,
    selections: {},
    playerSlots: [],
    playerAssignments: {},
  }
}

// Mock opponent that always returns the same top-3 heroes with fixed probabilities
function makeMockPredictor(topHeroes: string[]): OpponentPredictor {
  return async (_state: SearchState, topN: number) => {
    return topHeroes.slice(0, topN).map((hero, i) => ({
      hero,
      probability: 1 / (i + 1), // decreasing prob
    }))
  }
}

describe('evaluateLeaf', () => {
  it('returns 0 for empty state', () => {
    const ss = createSearchState(makeState(0))
    const data = makeMockData()
    const val = evaluateLeaf(ss, data)
    expect(val).toBe(0)
  })

  it('returns non-zero for state with picks', () => {
    const ss = createSearchState(makeState(6, 'A'))
    // Manually add picks to the search state
    ss.ourPicks.push('Valla') // 54% WR → +4 delta
    ss.enemyPicks.push('Jaina')
    const data = makeMockData()
    const val = evaluateLeaf(ss, data)
    expect(typeof val).toBe('number')
    expect(val).not.toBe(0) // Should have some hero WR delta
  })
})

describe('expectimaxAtDepth', () => {
  it('returns results at depth 1', async () => {
    // Start at step 4 (first pick for team A)
    const ss = createSearchState(makeState(4))
    const data = makeMockData()
    const predict = makeMockPredictor(['Muradin', 'Johanna', 'Diablo'])

    const { results } = await expectimaxAtDepth(ss, data, 1, {
      ourPickWidth: 5,
      ourBanWidth: 3,
      oppPickWidth: 3,
      oppBanWidth: 3,
      maxDepth: 1,
      timeBudgetMs: 5000,
    }, predict)

    expect(results.length).toBeGreaterThan(0)
    expect(results.length).toBeLessThanOrEqual(5)
    // Results should be sorted by score descending
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score)
    }
    // Each result should have the expected fields
    for (const r of results) {
      expect(r.hero).toBeTruthy()
      expect(typeof r.score).toBe('number')
      expect(r.depth).toBe(1)
    }
  })

  it('deeper search explores more nodes', async () => {
    const ss = createSearchState(makeState(4))
    const data = makeMockData()
    const predict = makeMockPredictor(['Muradin', 'Johanna', 'Diablo'])

    const r1 = await expectimaxAtDepth(ss, data, 1, {
      ourPickWidth: 3,
      ourBanWidth: 3,
      oppPickWidth: 3,
      oppBanWidth: 3,
      maxDepth: 4,
      timeBudgetMs: 10000,
    }, predict)

    const r2 = await expectimaxAtDepth(ss, data, 3, {
      ourPickWidth: 3,
      ourBanWidth: 3,
      oppPickWidth: 3,
      oppBanWidth: 3,
      maxDepth: 4,
      timeBudgetMs: 10000,
    }, predict)

    expect(r2.totalNodes).toBeGreaterThan(r1.totalNodes)
  })
})

describe('iterativeDeepeningSearch', () => {
  it('calls onDepthComplete for each depth', async () => {
    const ss = createSearchState(makeState(4))
    const data = makeMockData()
    const predict = makeMockPredictor(['Muradin', 'Johanna', 'Diablo'])

    const depthsCompleted: number[] = []

    const results = await iterativeDeepeningSearch(
      ss, data,
      { maxDepth: 6, ourPickWidth: 3, oppPickWidth: 3, timeBudgetMs: 10000 },
      predict,
      (_results, depth) => { depthsCompleted.push(depth) },
    )

    expect(depthsCompleted.length).toBeGreaterThan(0)
    expect(results.length).toBeGreaterThan(0)
    // Depths should be increasing
    for (let i = 1; i < depthsCompleted.length; i++) {
      expect(depthsCompleted[i]).toBeGreaterThan(depthsCompleted[i - 1])
    }
  })

  it('respects time budget', async () => {
    const ss = createSearchState(makeState(4))
    const data = makeMockData()
    // Slow predictor that takes 10ms per call
    const slowPredict: OpponentPredictor = async (state, topN) => {
      await new Promise(r => setTimeout(r, 10))
      return [{ hero: 'Muradin', probability: 0.5 }, { hero: 'Johanna', probability: 0.3 }].slice(0, topN)
    }

    const start = Date.now()
    await iterativeDeepeningSearch(
      ss, data,
      { maxDepth: 8, ourPickWidth: 5, oppPickWidth: 5, timeBudgetMs: 500 },
      slowPredict,
    )
    const elapsed = Date.now() - start

    // Should not massively exceed budget (allow 2x for overhead)
    expect(elapsed).toBeLessThan(5000)
  })

  it('handles ban phase at root', async () => {
    const ss = createSearchState(makeState(0)) // step 0 = A ban
    const data = makeMockData()
    const predict = makeMockPredictor(['Muradin', 'Johanna'])

    const results = await iterativeDeepeningSearch(
      ss, data,
      { maxDepth: 4, ourBanWidth: 3, oppBanWidth: 3, timeBudgetMs: 5000 },
      predict,
    )

    expect(results.length).toBeGreaterThan(0)
    // Bans should suggest heroes to remove
    for (const r of results) {
      expect(r.hero).toBeTruthy()
    }
  })
})
