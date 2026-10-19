import { describe, expect, it } from 'vitest'
import fixture from './fixtures/site_cards.json'
import { decodeCard, draftProfile, HERO_NAMES, H } from '../card'
import { matchScore, rankCandidates, MATCH_PARAMS } from '../match'
import { bestAssignment, banGain, personalValue, type PersonalPlayer } from '../draft-personal'

const F = fixture as unknown as {
  cards: {
    card_hex: string
    model: 'main' | 'imp2'
    games: number
    played: Record<string, { games: number; draft: number; mean: number }>
    draft_s: number[]
  }[]
  match: {
    ctx: { region: number; mmr: number; asOf: string }
    candidates: { region: number; mmr: number | null; lastPlayed: string }[]
    scores: number[]
    params: Record<string, number>
  }
}

const bytes = (hex: string) => Uint8Array.from(Buffer.from(hex, 'hex'))

describe('published cards (encoded by p3_site_publish.py)', () => {
  for (const [i, c] of F.cards.entries()) {
    it(`card ${i}: decodes and expands the draft term for all heroes`, () => {
      const card = decodeCard(bytes(c.card_hex))
      expect(card.model).toBe(c.model)
      expect(card.games).toBe(c.games)
      expect(card.played.length).toBe(Object.keys(c.played).length)
      for (const p of card.played) {
        expect(p.games).toBe(c.played[p.hero].games)
        expect(p.draft).toBeCloseTo(c.played[p.hero].draft, 10)
        expect(p.mean).toBeCloseTo(c.played[p.hero].mean, 10)
      }
      // played heroes are sorted by games
      for (let k = 1; k < card.played.length; k++) expect(card.played[k - 1].games).toBeGreaterThanOrEqual(card.played[k].games)
      const d = draftProfile(card)
      expect(d.s.length).toBe(H)
      for (let h = 0; h < H; h++) expect(Math.abs(d.s[h] - c.draft_s[h])).toBeLessThan(1e-9)
    })
  }

  it('flags off-role heroes only with 50+ games', () => {
    const many = F.cards.map(c => decodeCard(bytes(c.card_hex))).find(c => c.played.reduce((a, p) => a + p.games, 0) >= 50)!
    const d = draftProfile(many)
    const playedSet = new Set(many.played.map(p => p.hero))
    // a hero the player plays a lot is never off-role
    expect(d.offRole[HERO_NAMES.indexOf(many.played[0].hero)]).toBe(false)
    expect(playedSet.size).toBeGreaterThan(0)
    const few = decodeCard(bytes(F.cards.find(c => c.games > 0 && c.games < 50)?.card_hex ?? F.cards[1].card_hex))
    if (few.played.reduce((a, p) => a + p.games, 0) < 50) expect(draftProfile(few).offRole.every(x => !x)).toBe(true)
  })

  it('rejects a truncated card', () => {
    const b = bytes(F.cards[2].card_hex)
    expect(() => decodeCard(b.slice(0, b.length - 1))).toThrow()
  })
})

describe('name matching (same rule as the simulation)', () => {
  it('matches the Python scores', () => {
    const py = F.match.params
    expect([py.region, py.mmr_scale, py.mmr_unknown, py.recency, py.recency_days, py.margin, py.min_score]).toEqual([
      MATCH_PARAMS.region, MATCH_PARAMS.mmrScale, MATCH_PARAMS.mmrUnknown, MATCH_PARAMS.recency,
      MATCH_PARAMS.recencyDays, MATCH_PARAMS.margin, MATCH_PARAMS.minScore,
    ])
    F.match.candidates.forEach((c, i) => {
      expect(matchScore(c, F.match.ctx)).toBeCloseTo(F.match.scores[i], 10)
    })
  })

  it('selects only a clear winner', () => {
    const ctx = { region: 1, mmr: 2500, asOf: '2026-09-27' }
    const one = rankCandidates([{ region: 1, mmr: 2550, lastPlayed: '2026-09-20' }], ctx)
    expect(one.clear).toBe(true)
    const otherRegion = rankCandidates([{ region: 2, mmr: 2500, lastPlayed: '2026-09-27' }], ctx)
    expect(otherRegion.clear).toBe(false)
    const twoClose = rankCandidates(
      [
        { region: 1, mmr: 2500, lastPlayed: '2026-09-27' },
        { region: 1, mmr: 2550, lastPlayed: '2026-09-25' },
      ],
      ctx,
    )
    expect(twoClose.clear).toBe(false)
    expect(twoClose.ranked[0].item.mmr).toBe(2500)
  })
})

describe('personal draft terms', () => {
  const c = { b0: 0, b1: 1 }
  const p = (label: string, s: Record<string, number>, off: Record<string, boolean> = {}): PersonalPlayer => ({
    label,
    s,
    offRole: off,
    b2: 4,
    b3: -0.055,
  })

  it('assigns heroes to the players best at them', () => {
    const a = bestAssignment(['Ana', 'Muradin'], [p('x', { Ana: 0.05, Muradin: 0.0 }), p('y', { Ana: 0.04, Muradin: 0.03 })], c)
    expect(a.who).toEqual({ Ana: 'x', Muradin: 'y' })
    expect(a.value).toBeCloseTo(4 * 0.08, 12)
  })

  it('lets unmatched players take heroes at 0', () => {
    const a = bestAssignment(['Ana', 'Muradin'], [p('x', { Ana: -0.05, Muradin: -0.02 })], c)
    expect(a.value).toBeCloseTo(0, 12)
  })

  it('honours a pinned pick and counts off-role', () => {
    const a = bestAssignment(['Ana'], [p('x', { Ana: 0.02 }, { Ana: true }), p('y', { Ana: 0.01 })], c, { Ana: 'x' })
    expect(a.who).toEqual({ Ana: 'x' })
    expect(a.value).toBeCloseTo(4 * 0.02 - 0.055, 12)
  })

  it('equals the population value with nobody matched', () => {
    const v = personalValue(0.53, ['Ana'], ['Muradin'], [], [], { b0: 0, b1: 1 })
    expect(v.value).toBeCloseTo(0.53, 12)
  })

  it('values a ban by the drop in an enemy best option', () => {
    const e = [p('e', { Ana: 0.06, Lucio: 0.02 })]
    const g = banGain('Ana', ['Ana', 'Lúcio', 'Lucio'], e, new Set(), c)
    expect(g.who).toBe('e')
    expect(g.gain).toBeCloseTo(4 * 0.04, 12)
    expect(banGain('Lucio', ['Ana', 'Lucio'], e, new Set(), c).gain).toBe(0)
  })
})
