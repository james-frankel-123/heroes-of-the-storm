import { describe, it, expect } from 'vitest'
import { aggregateByHero, aggregateByHeroMap } from '@/lib/player-stats/aggregate'
import { computeMAWP } from '@/lib/mawp'

function match(hero: string, map: string, win: boolean, daysAgo: number, kda = [0, 0, 0]) {
  const d = new Date()
  d.setDate(d.getDate() - daysAgo)
  return { hero, map, win, gameDate: d, kills: kda[0], deaths: kda[1], assists: kda[2] }
}

describe('aggregateByHero', () => {
  it('groups per hero with games/wins/winRate', () => {
    const rows = [
      match('Nazeebo', 'Sky Temple', true, 1),
      match('Nazeebo', 'Cursed Hollow', false, 2),
      match('Thrall', 'Sky Temple', true, 3),
    ]
    const agg = aggregateByHero(rows).sort((a, b) => a.hero.localeCompare(b.hero))
    expect(agg.map((a) => a.hero)).toEqual(['Nazeebo', 'Thrall'])
    const naz = agg.find((a) => a.hero === 'Nazeebo')!
    expect(naz.games).toBe(2)
    expect(naz.wins).toBe(1)
    expect(naz.winRate).toBe(50)
  })

  it('MAWP matches the reference impl × 100', () => {
    const rows = [
      match('Valla', 'Sky Temple', true, 1),
      match('Valla', 'Sky Temple', true, 2),
      match('Valla', 'Sky Temple', false, 3),
    ]
    const agg = aggregateByHero(rows)
    const expected = computeMAWP(rows.map((r) => ({ win: r.win, gameDate: r.gameDate }))) * 100
    expect(agg[0].mawp).toBeCloseTo(expected, 6)
  })

  it('recentWinRate is null under 5 games, trend follows it', () => {
    const rows = [match('Muradin', 'Sky Temple', true, 1)]
    const agg = aggregateByHero(rows)
    expect(agg[0].recentWinRate).toBeNull()
    expect(agg[0].trend).toBeNull()
  })
})

describe('aggregateByHeroMap', () => {
  it('groups per (hero, map)', () => {
    const rows = [
      { hero: 'Nazeebo', map: 'Sky Temple', win: true },
      { hero: 'Nazeebo', map: 'Sky Temple', win: false },
      { hero: 'Nazeebo', map: 'Cursed Hollow', win: true },
    ]
    const agg = aggregateByHeroMap(rows)
    const skyTemple = agg.find((a) => a.map === 'Sky Temple')!
    expect(skyTemple.games).toBe(2)
    expect(skyTemple.wins).toBe(1)
    expect(skyTemple.winRate).toBe(50)
    expect(agg).toHaveLength(2)
  })
})
