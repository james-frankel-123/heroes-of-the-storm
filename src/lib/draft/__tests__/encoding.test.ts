import { describe, it, expect } from 'vitest'
import { HEROES, MAPS, NUM_HEROES, NUM_MAPS, STATE_DIM, WP_BASE_DIM, WP_INPUT_DIM } from '../encoding'
import { HERO_ROLES } from '@/lib/data/hero-roles'
import { mapImageSrc } from '@/lib/data/map-images'
import { _testFineRoleMap } from '../ai-inference'

describe('model encoding (training/shared.py HOTS_HERO_SET=v2)', () => {
  it('keeps the original 90 heroes and 14 maps in order, with additions appended', () => {
    const v1 = HEROES.slice(0, 90)
    expect([...v1].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))).toEqual(v1)
    expect(HEROES[90]).toBe("Xal'atath")
    expect(MAPS[14]).toBe('Haunted Mines')
    expect(new Set(HEROES).size).toBe(HEROES.length)
    expect(new Set(MAPS).size).toBe(MAPS.length)
  })

  it('has model input sizes matching the exported ONNX models', () => {
    expect(NUM_HEROES).toBe(91)
    expect(NUM_MAPS).toBe(15)
    expect(STATE_DIM).toBe(294)
    expect(WP_BASE_DIM).toBe(200)
    expect(WP_INPUT_DIM).toBe(286)
  })

  it('gives every encoded hero a Blizzard role and a fine role', () => {
    for (const h of HEROES) {
      expect(HERO_ROLES[h], h).toBeDefined()
      expect(_testFineRoleMap[h], h).toBeDefined()
    }
    expect(HERO_ROLES["Xal'atath"]).toBe('Ranged Assassin')
  })

  it('has an image for every encoded map', () => {
    for (const m of MAPS) expect(mapImageSrc(m), m).not.toBeNull()
  })
})
