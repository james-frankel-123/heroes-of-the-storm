import { describe, expect, it } from 'vitest'
import {
  FALLBACK_MAP_ROTATION,
  filterToRotation,
  resolveRotation,
} from '@/lib/data/map-rotation'

describe('resolveRotation', () => {
  it('returns the data-driven rotation when present', () => {
    expect(resolveRotation(['Cursed Hollow', 'Sky Temple'])).toEqual([
      'Cursed Hollow',
      'Sky Temple',
    ])
  })

  it('falls back to the static list when the query returns nothing', () => {
    expect(resolveRotation([])).toEqual([...FALLBACK_MAP_ROTATION])
  })
})

describe('filterToRotation', () => {
  const rows = [
    { map: 'Cursed Hollow', games: 10 },
    { map: 'Hanamura Temple', games: 5 },
    { map: 'Warhead Junction', games: 3 },
    { map: 'Sky Temple', games: 7 },
  ]

  it('drops maps outside the rotation and keeps order', () => {
    expect(filterToRotation(rows, ['Sky Temple', 'Cursed Hollow']).map((r) => r.map)).toEqual([
      'Cursed Hollow',
      'Sky Temple',
    ])
  })

  it('returns nothing for an empty rotation or empty rows', () => {
    expect(filterToRotation(rows, [])).toEqual([])
    expect(filterToRotation([], ['Sky Temple'])).toEqual([])
  })

  it('fallback excludes the retired maps', () => {
    for (const m of ['Hanamura Temple', 'Warhead Junction', "Blackheart's Bay"]) {
      expect(FALLBACK_MAP_ROTATION).not.toContain(m)
    }
  })
})
