import { describe, it, expect } from 'vitest'
import {
  createSearchState,
  cloneAndApply,
  hashState,
  getValidHeroes,
  isTerminal,
  isOurTurn,
  isBanPhase,
} from '../expectimax/search-state'
import type { DraftState } from '../types'
import type { SkillTier } from '@/lib/types'

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

describe('createSearchState', () => {
  it('creates empty state from setup', () => {
    const ss = createSearchState(makeState())
    expect(ss.ourPicks).toEqual([])
    expect(ss.enemyPicks).toEqual([])
    expect(ss.bans).toEqual([])
    expect(ss.taken.size).toBe(0)
    expect(ss.step).toBe(0)
    expect(ss.map).toBe('Cursed Hollow')
    expect(ss.tier).toBe('mid')
    expect(ss.ourTeam).toBe('A')
  })

  it('extracts picks and bans from selections', () => {
    // Steps 0-3 are bans (A,B,A,B), step 4 is A pick, step 5 is B pick
    const ss = createSearchState(makeState({
      currentStep: 6,
      selections: {
        0: 'Muradin',    // A ban
        1: 'Johanna',    // B ban
        2: 'Diablo',     // A ban
        3: 'E.T.C.',     // B ban
        4: 'Valla',      // A pick (ours since ourTeam=A)
        5: 'Jaina',      // B pick (enemy)
      },
    }))
    expect(ss.bans).toEqual(['Muradin', 'Johanna', 'Diablo', 'E.T.C.'])
    expect(ss.ourPicks).toEqual(['Valla'])
    expect(ss.enemyPicks).toEqual(['Jaina'])
    expect(ss.taken.size).toBe(6)
    expect(ss.step).toBe(6)
  })

  it('handles team B as our team', () => {
    const ss = createSearchState(makeState({
      ourTeam: 'B',
      currentStep: 5,
      selections: {
        0: 'Muradin',  // A ban (opponent ban)
        1: 'Johanna',  // B ban (our ban)
        2: 'Diablo',   // A ban
        3: 'E.T.C.',   // B ban
        4: 'Valla',    // A pick (enemy since ourTeam=B)
      },
    }))
    expect(ss.ourPicks).toEqual([])
    expect(ss.enemyPicks).toEqual(['Valla'])
    expect(ss.ourTeam).toBe('B')
  })
})

describe('cloneAndApply', () => {
  it('applies a ban action', () => {
    const root = createSearchState(makeState({ currentStep: 0 }))
    const next = cloneAndApply(root, 'Muradin')
    expect(next.bans).toEqual(['Muradin'])
    expect(next.taken.has('Muradin')).toBe(true)
    expect(next.step).toBe(1)
    // Original unchanged
    expect(root.bans).toEqual([])
    expect(root.step).toBe(0)
  })

  it('applies a pick for our team', () => {
    const root = createSearchState(makeState({ currentStep: 4 })) // step 4 = A pick
    const next = cloneAndApply(root, 'Valla')
    expect(next.ourPicks).toEqual(['Valla'])
    expect(next.enemyPicks).toEqual([])
    expect(next.step).toBe(5)
  })

  it('applies a pick for enemy team', () => {
    const root = createSearchState(makeState({ currentStep: 5 })) // step 5 = B pick
    const next = cloneAndApply(root, 'Jaina')
    expect(next.ourPicks).toEqual([])
    expect(next.enemyPicks).toEqual(['Jaina'])
    expect(next.step).toBe(6)
  })

  it('does not mutate original state', () => {
    const root = createSearchState(makeState({ currentStep: 4 }))
    const next = cloneAndApply(root, 'Valla')
    expect(root.ourPicks.length).toBe(0)
    expect(root.taken.size).toBe(0)
    expect(next.ourPicks.length).toBe(1)
  })

  it('handles Cho/Gall auto-pairing', () => {
    // Step 4 = A pick 1, step 5 = B pick 1 — but let's test step 7+8 (A pick 2 + A pick 3)
    // Actually steps 7-8 are both A picks, so Cho picked at step 7 auto-fills Gall at step 8
    const root = createSearchState(makeState({
      currentStep: 7, // A pick 2 (consecutive: steps 7,8 are both A picks)
      ourTeam: 'A',
    }))
    const next = cloneAndApply(root, 'Cho')
    expect(next.ourPicks).toContain('Cho')
    expect(next.ourPicks).toContain('Gall')
    expect(next.taken.has('Cho')).toBe(true)
    expect(next.taken.has('Gall')).toBe(true)
    expect(next.step).toBe(9) // consumed 2 slots
  })
})

describe('hashState', () => {
  it('produces same hash for same state', () => {
    const s1 = createSearchState(makeState({
      currentStep: 6,
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.', 4: 'Valla', 5: 'Jaina' },
    }))
    const s2 = createSearchState(makeState({
      currentStep: 6,
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.', 4: 'Valla', 5: 'Jaina' },
    }))
    expect(hashState(s1)).toBe(hashState(s2))
  })

  it('produces different hash for different steps', () => {
    const s1 = createSearchState(makeState({ currentStep: 4 }))
    const s2 = createSearchState(makeState({ currentStep: 5 }))
    expect(hashState(s1)).not.toBe(hashState(s2))
  })
})

describe('getValidHeroes', () => {
  it('returns all heroes when nothing taken', () => {
    const s = createSearchState(makeState({ currentStep: 0 }))
    const valid = getValidHeroes(s)
    expect(valid.length).toBeGreaterThan(85)
  })

  it('excludes taken heroes', () => {
    const s = createSearchState(makeState({
      currentStep: 4,
      selections: { 0: 'Muradin', 1: 'Johanna', 2: 'Diablo', 3: 'E.T.C.' },
    }))
    const valid = getValidHeroes(s)
    expect(valid).not.toContain('Muradin')
    expect(valid).not.toContain('Johanna')
    expect(valid).not.toContain('Diablo')
    expect(valid).not.toContain('E.T.C.')
  })
})

describe('state queries', () => {
  it('isTerminal at step 16', () => {
    const s = createSearchState(makeState({ currentStep: 0 }))
    expect(isTerminal(s)).toBe(false)
    const s2 = { ...s, step: 16 }
    expect(isTerminal(s2)).toBe(true)
  })

  it('isOurTurn correct for team A', () => {
    // Step 0 = A ban (our turn), step 1 = B ban (not ours)
    const s0 = createSearchState(makeState({ currentStep: 0, ourTeam: 'A' }))
    expect(isOurTurn(s0)).toBe(true)
    const s1 = { ...s0, step: 1 }
    expect(isOurTurn(s1)).toBe(false)
  })

  it('isBanPhase correct', () => {
    const s0 = createSearchState(makeState({ currentStep: 0 }))
    expect(isBanPhase(s0)).toBe(true) // step 0 = ban
    const s4 = { ...s0, step: 4 }
    expect(isBanPhase(s4)).toBe(false) // step 4 = pick
  })
})
