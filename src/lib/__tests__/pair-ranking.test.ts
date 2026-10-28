import { describe, expect, it } from 'vitest'
import { rankPairs, pairScore } from '../data/pair-ranking'
import type { HeroPairwiseStats } from '../types'

const row = (heroA: string, heroB: string, games: number, winRate: number, wins = Math.round((games * winRate) / 100)): HeroPairwiseStats => ({
  heroA, heroB, relationship: 'against', skillTier: 'mid', games, wins, winRate,
})

describe('pair ranking', () => {
  it('ranks on the displayed win rate, not the stored count', () => {
    // an inverted count (179 of 241 for a 25.7% row) must not lift the row
    const ranked = rankPairs([row('Diablo', "Xal'atath", 241, 25.73, 179), row('Ana', 'Li', 200, 55)], 10)
    expect(ranked[0].heroA).toBe('Ana')
  })

  it('keeps the stronger direction of a counter whatever the alphabetical order', () => {
    const ranked = rankPairs([row('Diablo', "Xal'atath", 241, 25.73), row("Xal'atath", 'Diablo', 277, 69.68)], 10)
    expect(ranked).toHaveLength(1)
    expect(ranked[0]).toMatchObject({ heroA: "Xal'atath", heroB: 'Diablo' })
  })

  it('needs 50 games and shrinks small samples toward 50%', () => {
    expect(rankPairs([row('A', 'B', 10, 100)], 10)).toEqual([])
    expect(pairScore({ winRate: 70, games: 60 })).toBeLessThan(pairScore({ winRate: 65, games: 600 }))
  })

  it("merges Cho and Gall into one Cho'gall row", () => {
    const ranked = rankPairs([row('Cho', 'Fenix', 100, 60), row('Fenix', 'Gall', 100, 40), row('Cho', 'Gall', 100, 50)], 10)
    expect(ranked).toHaveLength(1)
    expect(ranked[0]).toMatchObject({ heroA: "Cho'gall", heroB: 'Fenix', winRate: 60 })
  })
})
