import { describe, expect, it } from 'vitest'
import type { MatchResult } from '../accounts'
import { accountKey, resolveLobby, type Candidate, type MatchSnapshot, type PartyAccount } from '../lobby'
import { lobbyRequestError, splitList } from '../match'
import { bestAssignment, personalValue } from '../draft-personal'

const draft = (s: Record<string, number> = {}) => ({ model: 'chosen' as const, s, offRole: {}, b2: 4, b3: -0.05 })

const cand = (name: string, discriminator: number, region = 1, s: Record<string, number> = { Ana: 0.03 }): Candidate => ({
  name,
  discriminator,
  region,
  regionName: 'Americas',
  mmr: 2500,
  lastPlayed: '2026-10-01',
  games: 100,
  recentGames: 10,
  topHeroes: [],
  draft: draft(s),
})

const matched = (query: string, c: Candidate): MatchResult => ({ query, status: 'matched', candidates: [c] })
const choose = (query: string, cs: Candidate[]): MatchResult => ({ query, status: 'choose', candidates: cs })

const snap = (results: MatchResult[], inputs: Record<string, string>, party: PartyAccount[] = [], region = ''): MatchSnapshot => ({
  results,
  party,
  coefs: { b0: 0, b1: 1 },
  region,
  inputs,
})

const empty4 = ['', '', '', '']
const empty5 = ['', '', '', '', '']

describe('in-game lobby resolution', () => {
  const alpha = cand('Alpha', 1111)

  it('counts a clear match and hands the same player to the scorer', () => {
    const l = resolveLobby(snap([matched('Alpha', alpha)], { a0: 'Alpha' }), ['Alpha', '', '', ''], empty5, '', {})
    expect(l.count).toBe(1)
    expect(l.ours.map(p => p.id)).toEqual([accountKey(alpha)])
    expect(l.slots.a0.kind).toBe('selected')
  })

  it('stops counting a slot as soon as its name is edited (R1-02)', () => {
    const s = snap([matched('Alpha', alpha)], { a0: 'Alpha' })
    const l = resolveLobby(s, ['QaZz92761', '', '', ''], empty5, '', {})
    expect(l.count).toBe(0)
    expect(l.ours).toEqual([])
    expect(l.slots.a0.kind).toBe('pending')
    // an explicit pick on the old result does not survive the edit either
    expect(resolveLobby(s, ['QaZz92761', '', '', ''], empty5, '', { a0: 0 }).count).toBe(0)
  })

  it('stops counting every name when the region changes, keeps the party (R1-02)', () => {
    const party: PartyAccount = { ...cand('Party', 2222), tag: 'Party#2222' }
    const s = snap([matched('Alpha', alpha)], { a0: 'Alpha' }, [party], '1')
    const l = resolveLobby(s, ['Alpha', '', '', ''], empty5, '2', {})
    expect(l.slots.a0.kind).toBe('pending')
    expect(l.count).toBe(1)
    expect(l.ours[0].tag).toBe('Party#2222')
  })

  it('treats nothing as matched without a snapshot (reset or failed rematch)', () => {
    const l = resolveLobby(null, ['Alpha', '', '', ''], ['Beta', '', '', '', ''], '', {})
    expect(l.count).toBe(0)
  })

  it('counts one account once across slots and teams (R1-03)', () => {
    const s = snap([matched('Alpha', alpha)], { a0: 'Alpha', a1: 'Alpha', e0: 'Alpha' })
    const l = resolveLobby(s, ['Alpha', 'Alpha', '', ''], ['Alpha', '', '', '', ''], '', {}, k => k)
    expect(l.count).toBe(1)
    expect(l.ours).toHaveLength(1)
    expect(l.theirs).toHaveLength(0)
    expect(l.slots.a1).toMatchObject({ kind: 'duplicate', usedBy: 'a0' })
    expect(l.slots.e0).toMatchObject({ kind: 'duplicate', usedBy: 'a0' })
  })

  it('rejects a typed account that is already in the registered party', () => {
    const party: PartyAccount = { ...alpha, tag: 'Alpha#1111' }
    const s = snap([matched('Alpha#1111', alpha)], { a0: 'Alpha#1111' }, [party])
    const l = resolveLobby(s, ['Alpha#1111', '', '', ''], empty5, '', {})
    expect(l.count).toBe(1)
    expect(l.slots.a0).toMatchObject({ kind: 'duplicate', usedBy: 'your party' })
  })

  it('keeps different accounts with the same visible name apart (R1-03)', () => {
    const a1 = cand('Same', 1001, 1, { Ana: 0.05 })
    const a2 = cand('Same', 1002, 1, { Ana: 0.01 })
    const s = snap([choose('Same', [a1, a2])], { a0: 'Same', e0: 'Same' })
    const l = resolveLobby(s, ['Same', '', '', ''], ['Same', '', '', '', ''], '', { a0: 0, e0: 1 })
    expect(l.count).toBe(2)
    expect(l.ours[0].id).not.toBe(l.theirs[0].id)
    expect(l.ours[0].label).toBe(l.theirs[0].label)
  })

  it('needs a pick for an uncertain candidate, and Not them drops a clear match', () => {
    const s = snap([choose('Alpha', [alpha]), matched('Beta', cand('Beta', 3333))], { a0: 'Alpha', e0: 'Beta' })
    const l = resolveLobby(s, ['Alpha', '', '', ''], ['Beta', '', '', '', ''], '', { e0: null })
    expect(l.slots.a0.kind).toBe('unselected')
    expect(l.slots.e0.kind).toBe('unselected')
    expect(l.count).toBe(0)
    expect(resolveLobby(s, ['Alpha', '', '', ''], ['Beta', '', '', '', ''], '', { a0: 0 }).count).toBe(2)
  })

  it('has room for five allies and five enemies (R1-04)', () => {
    const names = ['A1', 'A2', 'A3', 'A4', 'A5', 'E1', 'E2', 'E3', 'E4', 'E5']
    const results = names.map((n, i) => matched(n, cand(n, 4000 + i)))
    const inputs: Record<string, string> = {}
    names.forEach((n, i) => (inputs[i < 5 ? `a${i}` : `e${i - 5}`] = n))
    const l = resolveLobby(snap(results, inputs), names.slice(0, 5), names.slice(5), '', {})
    expect(l.count).toBe(10)
    expect(l.ours).toHaveLength(5)
    expect(l.theirs).toHaveLength(5)
  })

  it('ignores blank slots', () => {
    expect(resolveLobby(snap([], {}), empty4, empty5, '', {}).count).toBe(0)
  })
})

describe('match request validation (R1-04)', () => {
  it('accepts a full ten-name lobby and rejects more instead of truncating', () => {
    const ten = Array.from({ length: 10 }, (_, i) => `Name${i}`)
    expect(lobbyRequestError(ten, [])).toBeNull()
    expect(lobbyRequestError([...ten, 'Extra'], [])).toMatch(/at most 10/)
    expect(lobbyRequestError(ten.slice(0, 6), ['P#1', 'Q#2', 'R#3', 'S#4', 'T#5'])).toMatch(/at most 10/)
    expect(lobbyRequestError([], ['P#1', 'Q#2', 'R#3', 'S#4', 'T#5', 'U#6'])).toMatch(/at most 5/)
    expect(lobbyRequestError([], [])).toMatch(/at least one/)
  })

  it('splits without dropping entries', () => {
    expect(splitList(' a, b ,,c ')).toEqual(['a', 'b', 'c'])
    expect(splitList(null)).toEqual([])
    expect(splitList(Array.from({ length: 12 }, (_, i) => `n${i}`).join(','))).toHaveLength(12)
  })
})

describe('assignment identity (R1-03, R1-07)', () => {
  const c = { b0: 0, b1: 1 }
  const p = (id: string, label: string, ana: number) => ({ id, label, s: { Ana: ana }, offRole: {}, b2: 4, b3: -0.05 })

  it('pins a pick by id even when two players share a visible name', () => {
    const players = [p('1:same#1', 'Same', 0.05), p('1:same#2', 'Same', -0.02)]
    expect(bestAssignment(['Ana'], players, c).who).toEqual({ Ana: '1:same#1' })
    const pinned = bestAssignment(['Ana'], players, c, { Ana: '1:same#2' })
    expect(pinned.who).toEqual({ Ana: '1:same#2' })
    expect(pinned.value).toBeCloseTo(4 * -0.02, 12)
  })

  it('an observed assignment changes the estimate', () => {
    const players = [p('x', 'X', 0.05), p('y', 'Y', -0.03)]
    const best = personalValue(0.5, ['Ana'], [], players, [], c).value
    const actual = personalValue(0.5, ['Ana'], [], players, [], c, { Ana: 'y' }).value
    expect(actual).toBeLessThan(best)
  })
})

describe('heroes without a personal estimate (R2-10)', () => {
  it("lists heroes in the game but not in the model, such as Xal'atath", async () => {
    const { NO_ESTIMATE_HEROES, HERO_NAMES } = await import('../card')
    expect(NO_ESTIMATE_HEROES).toContain("Xal'atath")
    for (const h of NO_ESTIMATE_HEROES) expect(HERO_NAMES).not.toContain(h)
  })
})

describe('Unicode names (R3-05)', () => {
  const nfcName = 'Éclair'
  const nfdName = nfcName.normalize('NFD')

  it('treats composed and decomposed spellings as one account', () => {
    expect(nfdName).not.toBe(nfcName)
    expect(accountKey({ region: 2, name: nfdName, discriminator: 1234 })).toBe(accountKey({ region: 2, name: nfcName, discriminator: 1234 }))
  })

  it('parses a decomposed BattleTag to the stored NFC name', async () => {
    const { parseTag } = await import('../match')
    expect(parseTag(`${nfdName}#1234`)).toEqual({ name: nfcName, discriminator: 1234 })
  })
})
