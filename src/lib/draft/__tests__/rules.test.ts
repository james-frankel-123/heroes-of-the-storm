import { describe, expect, it } from 'vitest'
import { applySelection, isLegal, lineups, positionProblems, unavailableAt } from '../rules'

// Steps: 0-3 bans (A,B,A,B); 4 A pick; 5,6 B picks; 7,8 A picks; 9,10 bans (B,A);
// 11,12 B picks; 13,14 A picks; 15 B pick.
const bansDone = { 0: 'Ana', 1: 'Li Li', 2: 'Uther', 3: 'Rehgar' }

describe("draft rules: Cho'gall", () => {
  it('a Cho pick brings Gall into the next slot of the same team and skips it', () => {
    const r = applySelection(bansDone, 5, 'Cho')
    expect(r.selections[6]).toBe('Gall')
    expect(r.step).toBe(7)
    expect(lineups(r.selections).B).toEqual(['Cho', 'Gall'])
    expect(positionProblems(r.selections)).toEqual([])
  })

  it('a Gall pick brings Cho the same way', () => {
    const r = applySelection({ ...bansDone, 4: 'Muradin', 5: 'Valla', 6: 'Jaina' }, 7, 'Gall')
    expect(r.selections[8]).toBe('Cho')
    expect(r.step).toBe(9)
    expect(lineups(r.selections).A).toEqual(['Muradin', 'Gall', 'Cho'])
  })

  it('blocks both halves on a single-pick turn and after either is drafted or banned', () => {
    expect(unavailableAt(bansDone, 4).has('Cho')).toBe(true) // A pick 1 is a single slot
    expect(unavailableAt(bansDone, 4).has('Gall')).toBe(true)
    expect(isLegal(bansDone, 5, 'Cho')).toBe(true) // B has picks 5 and 6
    const banned = { 0: 'Gall' }
    expect(unavailableAt(banned, 1).has('Cho')).toBe(true)
    const picked = applySelection(bansDone, 5, 'Cho').selections
    expect(unavailableAt(picked, 7).has('Gall')).toBe(true)
  })

  it('treats the last slot of a two-pick turn as a single slot', () => {
    const sel = { ...bansDone, 4: 'Muradin', 5: 'Valla', 6: 'Jaina', 7: 'Diablo', 8: 'Malfurion', 9: 'Sonya', 10: 'Zeratul', 11: 'Thrall' }
    expect(unavailableAt(sel, 12).has('Cho')).toBe(true) // B's second of two, already one used
    expect(unavailableAt({ ...sel, 12: 'Azmodan' }, 13).has('Cho')).toBe(false) // A has 13 and 14
  })

  it('flags impossible positions', () => {
    expect(positionProblems({ 4: 'Rehgar', 5: 'Rehgar' })).toEqual(['Rehgar at steps 4 and 5'])
    expect(positionProblems({ 5: 'Cho' })).toContain('unpaired Cho')
    expect(positionProblems({ 4: 'Muradin' }, true)[0]).toMatch(/complete draft/)
  })
})
