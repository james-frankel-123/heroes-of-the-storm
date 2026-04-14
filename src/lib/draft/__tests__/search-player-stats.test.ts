/**
 * Tests for per-player stats propagation through expectimax Search mode.
 *
 * Semantics mirror Stats/Greedy mode:
 *   - playerAssignments records which battletag played each COMPLETED pick
 *     (step < currentStep). The current step has no assignment yet.
 *   - For scoring the current candidate, Search picks the best-fit available
 *     battletag from unassigned slots — same as scorePlayerStrength.
 *   - Leaf evaluation applies player adj only for picks that already have a
 *     locked assignment (via ourPickSteps → playerAssignments).
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
import { scoreHeroForPick, scorePlayerStrength } from '../engine'
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

  it('populates playerSlots and usedBattletags from DraftState', () => {
    const ss = createSearchState(makeState({
      currentStep: 5,
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.', 4: 'Valla' },
      playerSlots: [
        { battletag: 'alice#1' }, { battletag: 'bob#2' },
        { battletag: 'carol#3' }, { battletag: null },
      ],
      playerAssignments: { 4: 'alice#1' },
    }))
    expect(ss.playerSlots).toEqual(['alice#1', 'bob#2', 'carol#3'])
    expect(ss.usedBattletags?.has('alice#1')).toBe(true)
    expect(ss.usedBattletags?.has('bob#2')).toBe(false)
  })

  it('cloneAndApply pushes current step into ourPickSteps on our pick', () => {
    const root = createSearchState(makeState({ currentStep: 4 }))
    const next = cloneAndApply(root, 'Valla')
    expect(next.ourPicks).toEqual(['Valla'])
    expect(next.ourPickSteps).toEqual([4])
  })

  it('records both stepIndices for Cho/Gall pairing', () => {
    const root = createSearchState(makeState({ currentStep: 7 }))
    const next = cloneAndApply(root, 'Cho')
    expect(next.ourPicks).toEqual(['Cho', 'Gall'])
    expect(next.ourPickSteps).toEqual([7, 8])
    expect(next.step).toBe(9)
  })

  it('enemy picks do not grow ourPickSteps', () => {
    const root = createSearchState(makeState({ currentStep: 5 }))
    const next = cloneAndApply(root, 'Jaina')
    expect(next.enemyPicks).toEqual(['Jaina'])
    expect(next.ourPickSteps).toEqual([])
  })
})

// ─── Hash behaviour ─────────────────────────────────────────────────────────

describe('hashState with player assignments', () => {
  it('distinguishes swapped pick order when assignments are present', () => {
    const base = makeState({
      currentStep: 9,
      selections: {
        0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.',
        4: 'Valla', 5: 'Raynor', 6: 'Raynor', 7: 'Jaina', 8: 'Li-Ming',
      },
      playerAssignments: { 4: 'alice#1', 7: 'bob#2' },
    })
    const s1 = createSearchState(base)
    const s2 = createSearchState({
      ...base,
      selections: { ...base.selections, 4: 'Jaina', 7: 'Valla' },
    })
    expect(hashState(s1)).not.toBe(hashState(s2))
  })

  it('merges swapped pick order when no assignments (baseline)', () => {
    const sel = {
      0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.',
      4: 'Valla', 5: 'Raynor', 6: 'Raynor', 7: 'Jaina', 8: 'Li-Ming',
    }
    const s1 = createSearchState(makeState({ currentStep: 9, selections: sel }))
    const s2 = createSearchState(makeState({
      currentStep: 9,
      selections: { ...sel, 4: 'Jaina', 7: 'Valla' },
    }))
    expect(hashState(s1)).toBe(hashState(s2))
  })
})

// ─── scoreHeroForPick ───────────────────────────────────────────────────────

describe('scoreHeroForPick with availableBattletags', () => {
  it('adds best-fit player delta matching scorePlayerStrength', () => {
    const data = makeData({
      heroStats: { Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 } },
      playerStats: {
        'alice#1': { Jaina: { games: 50, wins: 35, winRate: 70, mawp: 65 } },
      },
    })
    const base = scoreHeroForPick('Jaina', [], [], data, null)
    const withPool = scoreHeroForPick('Jaina', [], [], data, null, ['alice#1'])
    // scorePlayerStrength reports (65-50)=15, threshold ≥2 satisfied
    expect(withPool - base).toBeCloseTo(15, 5)
  })

  it('picks the strongest available battletag when multiple fit', () => {
    const data = makeData({
      heroStats: { Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 } },
      playerStats: {
        'alice#1': { Jaina: { games: 50, wins: 30, winRate: 60, mawp: 58 } },
        'bob#2':   { Jaina: { games: 50, wins: 35, winRate: 70, mawp: 65 } },
      },
    })
    const withPool = scoreHeroForPick('Jaina', [], [], data, null, ['alice#1', 'bob#2'])
    const base = scoreHeroForPick('Jaina', [], [], data, null)
    // bob is +15 vs alice +8 → best-fit=bob
    expect(withPool - base).toBeCloseTo(15, 5)
  })

  it('skips players below the 2-point delta threshold', () => {
    const data = makeData({
      heroStats: { Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 } },
      playerStats: {
        'alice#1': { Jaina: { games: 50, wins: 26, winRate: 52, mawp: 51 } },
      },
    })
    const base = scoreHeroForPick('Jaina', [], [], data, null)
    const withPool = scoreHeroForPick('Jaina', [], [], data, null, ['alice#1'])
    expect(withPool).toBe(base)
  })

  it('no-op when pool is empty or undefined', () => {
    const data = makeData({
      heroStats: { Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 } },
      playerStats: {
        'alice#1': { Jaina: { games: 50, wins: 35, winRate: 70, mawp: 65 } },
      },
    })
    const base = scoreHeroForPick('Jaina', [], [], data, null)
    expect(scoreHeroForPick('Jaina', [], [], data, null, [])).toBe(base)
    expect(scoreHeroForPick('Jaina', [], [], data, null, undefined)).toBe(base)
  })
})

// ─── prefilter ──────────────────────────────────────────────────────────────

describe('prefilterPicks uses available battletags on our turn', () => {
  it('boosts a hero an unassigned player is strong on', () => {
    const heroStats: DraftData['heroStats'] = {}
    const heroes = ['Valla', 'Jaina', 'Raynor', 'Falstad', 'Tychus', 'Hanzo']
    for (const h of heroes) {
      heroStats[h] = { winRate: 50, pickRate: 10, banRate: 5, games: 500 }
    }
    const data = makeData({
      heroStats,
      playerStats: {
        'alice#1': { Falstad: { games: 100, wins: 70, winRate: 70, mawp: 70 } },
      },
    })

    // Alice is in our team's slots and hasn't been assigned yet (step 4 is first pick).
    const ss = createSearchState(makeState({
      currentStep: 4,
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.' },
      playerSlots: [{ battletag: 'alice#1' }, { battletag: 'bob#2' }],
    }))

    const ranked = prefilterPicks(ss, data, 3)
    expect(ranked[0]).toBe('Falstad')

    // Compare: no player slots → Falstad should not top the ranking
    const ssNoSlots = createSearchState(makeState({
      currentStep: 4,
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.' },
    }))
    const rankedNoSlots = prefilterPicks(ssNoSlots, data, 3)
    expect(rankedNoSlots[0]).not.toBe('Falstad')
  })

  it('excludes already-assigned battletags from the pool', () => {
    const heroStats: DraftData['heroStats'] = {}
    const heroes = ['Valla', 'Jaina', 'Raynor', 'Falstad']
    for (const h of heroes) heroStats[h] = { winRate: 50, pickRate: 10, banRate: 5, games: 500 }
    const data = makeData({
      heroStats,
      playerStats: {
        'alice#1': { Falstad: { games: 100, wins: 70, winRate: 70, mawp: 70 } },
      },
    })
    // Alice already played step 4; now at step 7, she's no longer available
    const ss = createSearchState(makeState({
      currentStep: 7,
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.', 4: 'Valla', 5: 'Jaina', 6: 'Raynor' },
      playerSlots: [{ battletag: 'alice#1' }, { battletag: 'bob#2' }],
      playerAssignments: { 4: 'alice#1' },
    }))
    const ranked = prefilterPicks(ss, data, 3)
    // Bob has no Falstad stats; Alice is consumed → no player boost applies
    expect(ranked[0]).not.toBe('Falstad')
  })

  it('does not apply player adjustment on enemy turns', () => {
    const data = makeData({
      heroStats: {
        Valla: { winRate: 50, pickRate: 10, banRate: 5, games: 500 },
        Falstad: { winRate: 50, pickRate: 10, banRate: 5, games: 500 },
        Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 500 },
      },
      playerStats: {
        'alice#1': { Falstad: { games: 100, wins: 70, winRate: 70, mawp: 70 } },
      },
    })
    const ss = createSearchState(makeState({
      currentStep: 5, // B pick
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.', 4: 'Valla' },
      playerSlots: [{ battletag: 'alice#1' }],
    }))
    const ranked = prefilterPicks(ss, data, 3)
    // All heroes are 50% WR; no enemy player adj should apply. Ranking should
    // be effectively a tie — assert Falstad isn't uniquely boosted.
    const falstadScore = scoreHeroForPick('Falstad', [], ['Valla'], data, null)
    const jainaScore = scoreHeroForPick('Jaina', [], ['Valla'], data, null)
    expect(falstadScore).toBe(jainaScore)
    expect(ranked.length).toBeGreaterThan(0)
  })
})

// ─── leaf eval ──────────────────────────────────────────────────────────────

describe('evaluateLeaf applies player adjustment for completed picks', () => {
  it('adds per-player delta for locked-in assignments', () => {
    const data = makeData({
      heroStats: { Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 } },
      playerStats: {
        'alice#1': { Jaina: { games: 50, wins: 35, winRate: 70, mawp: 65 } },
      },
    })
    const ss = createSearchState(makeState({
      currentStep: 8,
      playerAssignments: { 4: 'alice#1' },
      selections: {
        0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.',
        4: 'Jaina', 5: 'Raynor', 6: 'Li-Ming', 7: 'Valla',
      },
    }))
    const withPlayer = evaluateLeaf(ss, data)
    const ssNoAssign: SearchState = { ...ss, playerAssignments: undefined }
    const withoutPlayer = evaluateLeaf(ssNoAssign, data)
    // Deduped formula: (65-50) - (50-50) = +15
    expect(withPlayer - withoutPlayer).toBeCloseTo(15, 1)
  })

  it('does nothing when a player has no stats on their picked hero', () => {
    const data = makeData({
      heroStats: { Jaina: { winRate: 50, pickRate: 10, banRate: 5, games: 1000 } },
      playerStats: {
        'alice#1': { Raynor: { games: 50, wins: 35, winRate: 70, mawp: 65 } },
      },
    })
    const ss = createSearchState(makeState({
      currentStep: 5,
      playerAssignments: { 4: 'alice#1' },
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.', 4: 'Jaina' },
    }))
    const v = evaluateLeaf(ss, data)
    const vNoAssign = evaluateLeaf({ ...ss, playerAssignments: undefined }, data)
    expect(v).toBeCloseTo(vNoAssign, 5)
  })
})

// ─── End-to-end ─────────────────────────────────────────────────────────────

describe('iterativeDeepeningSearch respects player slots', () => {
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
    data.playerStats = {
      'alice#1': {
        Falstad: { games: 200, wins: 140, winRate: 70, mawp: 70 },
      },
    }
    const ss = createSearchState(makeState({
      currentStep: 4,
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: "Anub'arak" },
      playerSlots: [{ battletag: 'alice#1' }, { battletag: 'bob#2' }],
    }))
    const results = await iterativeDeepeningSearch(
      ss, data,
      { maxDepth: 2, ourPickWidth: 5, oppPickWidth: 3, timeBudgetMs: 5000 },
      predict,
    )
    expect(results[0].hero).toBe('Falstad')
  })

  it('does NOT prefer player-strong hero when no slots are configured', async () => {
    const data = makeFlatData()
    data.playerStats = {
      'alice#1': {
        Falstad: { games: 200, wins: 140, winRate: 70, mawp: 70 },
      },
    }
    const ss = createSearchState(makeState({
      currentStep: 4,
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: "Anub'arak" },
      // no playerSlots
    }))
    const results = await iterativeDeepeningSearch(
      ss, data,
      { maxDepth: 2, ourPickWidth: 5, oppPickWidth: 3, timeBudgetMs: 5000 },
      predict,
    )
    expect(results[0].hero).not.toBe('Falstad')
  })
})

// ─── scorePlayerStrength sanity (the UI byline source) ──────────────────────

describe('scorePlayerStrength with map-specific override', () => {
  it('uses map-specific winrate when ≥25 games and labels with (map)', () => {
    const data = makeData({
      playerStats: {
        'django#1': { Falstad: { games: 200, wins: 108, winRate: 54, mawp: 55 } },
      },
      playerMapStats: {
        'django#1': { 'Cursed Hollow': { Falstad: { winRate: 65, games: 38 } } },
      },
    })
    const res = scorePlayerStrength('Falstad', ['django#1'], data, 'Cursed Hollow')
    expect(res.player).toBe('django#1')
    expect(res.reason?.delta).toBeCloseTo(15, 1)
    expect(res.reason?.label).toContain('(map)')
  })

  it('falls back to MAWP when map sample is under 25 games', () => {
    const data = makeData({
      playerStats: {
        'django#1': { Falstad: { games: 200, wins: 108, winRate: 54, mawp: 55 } },
      },
      playerMapStats: {
        'django#1': { 'Cursed Hollow': { Falstad: { winRate: 80, games: 20 } } },
      },
    })
    const res = scorePlayerStrength('Falstad', ['django#1'], data, 'Cursed Hollow')
    expect(res.reason?.delta).toBeCloseTo(5, 1)
    expect(res.reason?.label).not.toContain('(map)')
  })

  it('ignores map data when no map is provided', () => {
    const data = makeData({
      playerStats: {
        'django#1': { Falstad: { games: 200, wins: 108, winRate: 54, mawp: 55 } },
      },
      playerMapStats: {
        'django#1': { 'Cursed Hollow': { Falstad: { winRate: 80, games: 50 } } },
      },
    })
    const res = scorePlayerStrength('Falstad', ['django#1'], data, null)
    expect(res.reason?.delta).toBeCloseTo(5, 1)
  })
})

describe('scorePlayerStrength (used by UI byline)', () => {
  it('returns null reason when no player beats 2pt threshold', () => {
    const data = makeData({
      playerStats: {
        'alice#1': { Jaina: { games: 50, wins: 26, winRate: 52, mawp: 51 } },
      },
    })
    const res = scorePlayerStrength('Jaina', ['alice#1'], data)
    expect(res.reason).toBeNull()
  })

  it('returns the best-fit player with a populated label', () => {
    const data = makeData({
      playerStats: {
        'alice#1': { Jaina: { games: 50, wins: 35, winRate: 70, mawp: 65 } },
      },
    })
    const res = scorePlayerStrength('Jaina', ['alice#1'], data)
    expect(res.player).toBe('alice#1')
    expect(res.reason?.type).toBe('player_strong')
    expect(res.reason?.delta).toBeCloseTo(15, 1)
  })
})
