import { describe, expect, it } from 'vitest'
import { parseMatchupData } from '../matchup-rows'

describe('matchup rows', () => {
  it("keeps a counter row's wins and win rate on hero A's side (asymmetric matchup)", () => {
    // v1 enemy row as delivered: win_rate is the opponent's (74.27%), but the counts are
    // A's own (62 of 241); the old code inverted the counts into 179 wins
    const rows = parseMatchupData('Diablo', { "Xal'atath": { enemy: { wins: 62, losses: 179, win_rate: 74.27 } } })
    expect(rows).toEqual([{ heroA: 'Diablo', heroB: "Xal'atath", relationship: 'against', games: 241, wins: 62, winRate: 25.73 }])
  })

  it('every counter row satisfies wins = round(games * winRate / 100)', () => {
    const rows = parseMatchupData('Ana', {
      Ana: {
        Muradin: { enemy: { wins_against: 10, losses_against: 30, win_rate_against: 40 } },
        Li: { ally: { wins_with: 55, losses_with: 45, win_rate_as_ally: 55 }, enemy: { wins: 70, losses: 30, win_rate: 31 } },
      },
    })
    for (const r of rows) expect(r.wins).toBe(Math.round((r.games * r.winRate) / 100))
    expect(rows.find(r => r.heroB === 'Li' && r.relationship === 'with')).toMatchObject({ games: 100, wins: 55, winRate: 55 })
    expect(rows.find(r => r.heroB === 'Muradin')).toMatchObject({ games: 40, winRate: 60, wins: 24 })
  })
})
