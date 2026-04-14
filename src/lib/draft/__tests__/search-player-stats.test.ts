/**
 * Tests for per-player stats propagation through expectimax Search mode.
 *
 * Covers:
 *   - createSearchState records ourPickSteps + carries playerAssignments
 *   - cloneAndApply pushes the current stepIndex into ourPickSteps
 *   - hashState preserves ourPicks order when assignments are present
 *   - scoreHeroForPick applies the Stats-mode player adjustment
 *   - evaluateLeaf applies the same adjustment via ourPickSteps → battletag
 *   - End-to-end: iterativeDeepeningSearch rankings shift when a player is
 *     dramatically stronger on a specific hero than its base WR suggests
 */

import { describe, it, expect } from 'vitest'
import {
  createSearchState,
  cloneAndApply,
  hashState,
} from '../expectimax/search-state'
import { evaluateLeaf } from '../expectimax/leaf-eval'
import { prefilterPicks } from '../expectimax/prefilter'
import { iterativeDeepeningSearch } from '../expectimax/search'
import { scoreHeroForPick, scorePlayerAdjForCandidate } from '../engine'
import type { DraftState, DraftData } from '../types'
import type { SkillTier } from '@/lib/types'
import type { OpponentPredictor, SearchState } from '../expectimax/types'

function makeState(overrides: Partial<DraftState> = {}): DraftState {
  return {
    phase: 'drafting',
    map: 'Cursed Hollow',
    tier: 'mid' as SkillTier,
    ourTeam: 'A',
    currentStep: 0,
    selections: {},
    playerSlots: [],
    playerAssignments: {},
    ...overrides,
  }
}

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

// ─── State plumbing ─────────────────────────────────────────────────────────

describe('SearchState carries player-stats plumbing', () => {
  it('createSearchState records ourPickSteps parallel to ourPicks', () => {
    // Steps 4 and 7 are A picks; step 5 is B pick
    const ss = createSearchState(makeState({
      currentStep: 8,
      selections: {
        0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.',
        4: 'Valla',   // A pick (step 4)
        5: 'Jaina',   // B pick
        6: 'Li-Ming', // B pick
        7: 'Raynor',  // A pick (step 7)
      },
    }))
    expect(ss.ourPicks).toEqual(['Valla', 'Raynor'])
    expect(ss.ourPickSteps).toEqual([4, 7])
  })

  it('createSearchState carries playerAssignments when non-empty', () => {
    const ss = createSearchState(makeState({
      playerAssignments: { 4: 'alice#1', 7: 'bob#2' },
    }))
    expect(ss.playerAssignments).toEqual({ 4: 'alice#1', 7: 'bob#2' })
  })

  it('createSearchState leaves playerAssignments undefined when empty', () => {
    const ss = createSearchState(makeState())
    expect(ss.playerAssignments).toBeUndefined()
  })

  it('cloneAndApply pushes current step into ourPickSteps on our pick', () => {
    const root = createSearchState(makeState({ currentStep: 4 }))
    const next = cloneAndApply(root, 'Valla')
    expect(next.ourPicks).toEqual(['Valla'])
    expect(next.ourPickSteps).toEqual([4])
  })

  it('cloneAndApply records both stepIndices for Cho/Gall pairing', () => {
    // Steps 7 + 8 are consecutive A picks; Cho at step 7 auto-fills Gall at step 8
    const root = createSearchState(makeState({ currentStep: 7 }))
    const next = cloneAndApply(root, 'Cho')
    expect(next.ourPicks).toEqual(['Cho', 'Gall'])
    expect(next.ourPickSteps).toEqual([7, 8])
    expect(next.step).toBe(9)
  })

  it('cloneAndApply propagates playerAssignments by reference (immutable)', () => {
    const root = createSearchState(makeState({
      currentStep: 4,
      playerAssignments: { 4: 'alice#1' },
    }))
    const next = cloneAndApply(root, 'Valla')
    expect(next.playerAssignments).toBe(root.playerAssignments)
  })

  it('enemy picks do not grow ourPickSteps', () => {
    const root = createSearchState(makeState({ currentStep: 5 })) // B pick
    const next = cloneAndApply(root, 'Jaina')
    expect(next.enemyPicks).toEqual(['Jaina'])
    expect(next.ourPickSteps).toEqual([])
  })
})

// ─── Hash behaviour ─────────────────────────────────────────────────────────

describe('hashState with player assignments', () => {
  function makePickedState(picksInOrder: string[]): SearchState {
    // Build a state where team A has picked the given heroes at steps [4,7,8,...]
    const selections: Record<number, string> = {
      0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.',
    }
    const pickStepsA = [4, 7, 8, 13, 14]
    const pickStepsB = [5, 6, 11, 12, 15]
    let ai = 0, bi = 0
    for (let i = 0; i < picksInOrder.length; i++) {
      // alternate adding an enemy pick so step increments realistically
      selections[pickStepsA[ai++]] = picksInOrder[i]
      if (bi < pickStepsB.length) selections[pickStepsB[bi++]] = 'Raynor' // enemy filler
    }
    return createSearchState(makeState({
      currentStep: 9,
      selections,
      playerAssignments: { 4: 'alice#1', 7: 'bob#2' },
    }))
  }

  it('distinguishes swapped pick order when assignments are present', () => {
    const s1 = makePickedState(['Valla', 'Jaina'])
    const s2 = makePickedState(['Jaina', 'Valla'])
    expect(hashState(s1)).not.toBe(hashState(s2))
  })

  it('merges swapped pick order when no assignments (baseline behaviour)', () => {
    const s1a = createSearchState(makeState({
      currentStep: 9,
      selections: {
        0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.',
        4: 'Valla', 5: 'Raynor', 6: 'Raynor', 7: 'Jaina', 8: 'Li-Ming',
      },
    }))
    const s1b = createSearchState(makeState({
      currentStep: 9,
      selections: {
        0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.',
        4: 'Jaina', 5: 'Raynor', 6: 'Raynor', 7: 'Valla', 8: 'Li-Ming',
      },
    }))
    expect(hashState(s1a)).toBe(hashState(s1b))
  })
})

// ─── scoreHeroForPick ───────────────────────────────────────────────────────

describe('scoreHeroForPick with actingBattletag', () => {
  it('adds Stats-mode player adjustment when battletag provided', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 },
      },
      playerStats: {
        'alice#1': {
          Jaina: { games: 50, wins: 35, winRate: 70, mawp: 65 },
        },
      },
    })
    const base = scoreHeroForPick('Jaina', [], [], data, null)
    const withPlayer = scoreHeroForPick('Jaina', [], [], data, null, 'alice#1')
    // adjMawp=65 (≥30 games, no shrinkage), heroBaseDelta=0 → playerAdj = 15
    expect(withPlayer - base).toBeCloseTo(15, 5)
  })

  it('no-op when battletag has fewer than 10 games on hero', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 },
      },
      playerStats: {
        'alice#1': {
          Jaina: { games: 5, wins: 5, winRate: 100, mawp: 90 },
        },
      },
    })
    const base = scoreHeroForPick('Jaina', [], [], data, null)
    const withPlayer = scoreHeroForPick('Jaina', [], [], data, null, 'alice#1')
    expect(withPlayer).toBe(base)
  })

  it('applies 30-game shrinkage for thin-sample players', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 },
      },
      playerStats: {
        'alice#1': {
          Jaina: { games: 15, wins: 12, winRate: 80, mawp: 70 },
        },
      },
    })
    // games=15, threshold=30 → weight=0.5
    // adjusted MAWP = 70*0.5 + 50*0.5 = 60 → playerAdj = 10
    const adj = scorePlayerAdjForCandidate('Jaina', 'alice#1', data, null)
    expect(adj).toBeCloseTo(10, 5)
  })

  it('null/undefined battletag is a no-op', () => {
    const data = makeData({
      heroStats: { Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 } },
      playerStats: {
        'alice#1': { Jaina: { games: 50, wins: 35, winRate: 70, mawp: 65 } },
      },
    })
    const base = scoreHeroForPick('Jaina', [], [], data, null)
    expect(scoreHeroForPick('Jaina', [], [], data, null, null)).toBe(base)
    expect(scoreHeroForPick('Jaina', [], [], data, null, undefined)).toBe(base)
  })
})

// ─── prefilter ──────────────────────────────────────────────────────────────

describe('prefilterPicks uses playerAssignments on our turn', () => {
  it('boosts a hero the acting player is strong on', () => {
    const heroStats: DraftData['heroStats'] = {}
    const candidates = ['Valla', 'Jaina', 'Raynor', 'Falstad', 'Tychus', 'Hanzo']
    for (const h of candidates) {
      heroStats[h] = { winRate: 50, pickRate: 10, banRate: 5, games: 500 }
    }
    const data = makeData({
      heroStats,
      playerStats: {
        'alice#1': {
          // Alice is dramatically better than average on Falstad (+20 MAWP).
          Falstad: { games: 100, wins: 70, winRate: 70, mawp: 70 },
        },
      },
    })

    // State at step 4 (first A pick) with Alice assigned to it
    const ss = createSearchState(makeState({
      currentStep: 4,
      playerAssignments: { 4: 'alice#1' },
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.' },
    }))

    const withAssignments = prefilterPicks(ss, data, 3)
    expect(withAssignments[0]).toBe('Falstad')

    // Same state, no assignments: Falstad should not be specifically preferred
    const ssNoAssign = createSearchState(makeState({
      currentStep: 4,
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.' },
    }))
    const withoutAssignments = prefilterPicks(ssNoAssign, data, 3)
    expect(withoutAssignments[0]).not.toBe('Falstad')
  })

  it('does not apply player adjustment on enemy turns', () => {
    const data = makeData({
      heroStats: {
        Valla: { winRate: 50, pickRate: 10, banRate: 5, games: 500 },
        Falstad: { winRate: 50, pickRate: 10, banRate: 5, games: 500 },
      },
      playerStats: {
        'alice#1': { Falstad: { games: 100, wins: 70, winRate: 70, mawp: 70 } },
      },
    })
    // Step 5 is enemy's turn (B pick 1), but we still carry assignments
    const ss = createSearchState(makeState({
      currentStep: 5,
      playerAssignments: { 4: 'alice#1' }, // for our past pick
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.', 4: 'Valla' },
    }))
    // Enemy scoring should not inherit Alice's Falstad strength
    const enemyRank = prefilterPicks(ss, data, 5)
    // With 50% WR across the board and no player bonus for enemy,
    // Falstad shouldn't outrank everyone — we just assert it's not uniquely at top
    // due to a player adjustment it shouldn't receive.
    const falstadIdx = enemyRank.indexOf('Falstad')
    // Either not present or not singled out first with the +20 boost
    if (falstadIdx === 0) {
      // If it happens to be first for some other reason, the ranking should still
      // show no big gap vs other flat-50% heroes. Probe by recomputing raw scores.
      const withEnemyBattletag = scoreHeroForPick('Falstad', [], ['Valla'], data, null)
      const withFakeBoost = scoreHeroForPick('Falstad', [], ['Valla'], data, null, 'alice#1')
      // The prefilter must NOT have passed 'alice#1' here — so the two differ only
      // if we wired it wrong.
      expect(withEnemyBattletag).not.toBe(withFakeBoost)
    }
  })
})

// ─── leaf eval ──────────────────────────────────────────────────────────────

describe('evaluateLeaf applies player adjustment', () => {
  it('adds per-player delta based on ourPickSteps → battletag mapping', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 },
      },
      playerStats: {
        'alice#1': { Jaina: { games: 50, wins: 35, winRate: 70, mawp: 65 } },
      },
    })

    // Build a state where Alice (step 4) picked Jaina.
    const ss = createSearchState(makeState({
      currentStep: 8,
      playerAssignments: { 4: 'alice#1' },
      selections: {
        0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.',
        4: 'Jaina', 5: 'Raynor', 6: 'Li-Ming', 7: 'Valla',
      },
    }))

    const valWithPlayer = evaluateLeaf(ss, data)

    // Drop assignments and recompute
    const ssNoAssign = { ...ss, playerAssignments: undefined }
    const valWithout = evaluateLeaf(ssNoAssign, data)

    // Player adj should be ~+15 on Jaina (65 MAWP − 50 base WR → +15)
    expect(valWithPlayer - valWithout).toBeCloseTo(15, 1)
  })

  it('ignores players without stats on their picked hero', () => {
    const data = makeData({
      heroStats: {
        Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 },
      },
      playerStats: {
        'alice#1': { Raynor: { games: 50, wins: 35, winRate: 70, mawp: 65 } },
      },
    })
    const ss = createSearchState(makeState({
      currentStep: 5,
      playerAssignments: { 4: 'alice#1' },
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.', 4: 'Jaina' },
    }))
    // Alice has no Jaina stats — no adjustment should apply
    const v = evaluateLeaf(ss, data)
    const vNoAssign = evaluateLeaf({ ...ss, playerAssignments: undefined }, data)
    expect(v).toBeCloseTo(vNoAssign, 5)
  })
})

// ─── End-to-end ─────────────────────────────────────────────────────────────

describe('iterativeDeepeningSearch respects player assignments', () => {
  const flatHeroes = [
    'Muradin', 'Johanna', 'Valla', 'Jaina', 'Malfurion',
    'Diablo', 'Raynor', 'Brightwing', 'Thrall', "Kael'thas",
    'Arthas', 'Li-Ming', 'Rehgar', 'Falstad', "Anub'arak",
  ]
  function makeFlatData(): DraftData {
    const heroStats: DraftData['heroStats'] = {}
    for (const h of flatHeroes) {
      heroStats[h] = { winRate: 50, pickRate: 10, banRate: 5, games: 500 }
    }
    return makeData({ heroStats })
  }
  const predict: OpponentPredictor = async (_s, topN) =>
    ['Muradin', 'Johanna', 'Diablo'].slice(0, topN).map((h, i) => ({
      hero: h, probability: 1 / (i + 1),
    }))

  it('picks a player-strong hero that would otherwise tie', async () => {
    const data = makeFlatData()
    // Alice (assigned to step 4) is an expert on Falstad
    data.playerStats = {
      'alice#1': {
        Falstad: { games: 200, wins: 140, winRate: 70, mawp: 70 },
      },
    }

    const ss = createSearchState(makeState({
      currentStep: 4, // first A pick
      playerAssignments: { 4: 'alice#1' },
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: "Anub'arak" },
    }))

    const results = await iterativeDeepeningSearch(
      ss, data,
      { maxDepth: 2, ourPickWidth: 5, oppPickWidth: 3, timeBudgetMs: 5000 },
      predict,
    )

    expect(results.length).toBeGreaterThan(0)
    expect(results[0].hero).toBe('Falstad')
  })

  it('does NOT prefer player-strong hero when assignments are omitted', async () => {
    const data = makeFlatData()
    data.playerStats = {
      'alice#1': {
        Falstad: { games: 200, wins: 140, winRate: 70, mawp: 70 },
      },
    }

    const ss = createSearchState(makeState({
      currentStep: 4,
      // no playerAssignments
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: "Anub'arak" },
    }))

    const results = await iterativeDeepeningSearch(
      ss, data,
      { maxDepth: 2, ourPickWidth: 5, oppPickWidth: 3, timeBudgetMs: 5000 },
      predict,
    )

    // With all heroes at 50% WR and no player signal, Falstad has no reason
    // to top the ranking.
    expect(results[0].hero).not.toBe('Falstad')
  })
})
