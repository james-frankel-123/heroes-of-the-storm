import { describe, expect, it } from 'vitest'
import { enrichDraftContext } from '../ama/context'
import type { DraftData, DraftRecommendation, DraftState } from '../draft/types'

const state = {
  phase: 'drafting', map: 'Haunted Mines', tier: 'mid', ourTeam: 'A', currentStep: 4, selections: {},
  playerSlots: [], playerAssignments: {},
} as unknown as DraftState
const data = { heroStats: {}, heroMapWinRates: {}, counters: {}, synergies: {}, playerStats: {}, playerMapStats: {} } as unknown as DraftData
const statsRecs = [{ hero: "Xal'atath", netDelta: 17.5, reasons: [], suggestedPlayer: null }] as unknown as DraftRecommendation[]

describe('coach context (R3-08)', () => {
  it('lists the displayed searched rows, not the stats ranking, as the top picks', () => {
    const ctx = enrichDraftContext(state, statsRecs, data, null, 5, {
      rows: [
        { hero: 'Alarak', winPct: 67.3, deltaPp: 1.2, searched: true },
        { hero: "Xal'atath", winPct: 63.0, deltaPp: 0, searched: false },
      ],
      nowPct: 52.4,
    })
    expect(ctx.card.topRecsSource).toBe('displayed')
    expect(ctx.card.topRecs.map(r => r.hero)).toEqual(['Alarak'])
    expect(ctx.card.winPct).toBe(52.4)
    expect(ctx.textBlock).toContain("Xal'atath: not searched")
    expect(ctx.textBlock).toContain('STATISTICAL ALTERNATIVES')
  })

  it('labels the stats ranking as such when nothing is displayed', () => {
    const ctx = enrichDraftContext(state, statsRecs, data, null, 5, { rows: [], nowPct: null })
    expect(ctx.card.topRecsSource).toBe('stats')
    expect(ctx.textBlock).toContain('+17.5 pp')
  })
})
