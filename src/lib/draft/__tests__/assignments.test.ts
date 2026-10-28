import { describe, expect, it } from 'vitest'
import { assignOptionsFor, assignPlayer, pinsFor } from '../assignments'
import type { PersonalPlayer } from '@/lib/personal/draft-personal'

const player = (id: string, label: string, tag?: string): PersonalPlayer => ({ id, label, tag, s: {}, offRole: {}, b2: 4, b3: 0 })

describe('hero assignments (user report 1)', () => {
  const ally = player('1:ally#111', 'Ally')
  const party = player('1:me#222', 'Me', 'Me#222')
  const enemy = player('1:foe#333', 'Foe')

  it('offers matched teammates on every pick, including ones already assigned elsewhere', () => {
    const opts = assignOptionsFor('ours', ['Me#222'], [party, ally])
    expect(opts.map(o => o.label)).toEqual(['Me', 'Ally'])
  })

  it('offers matched enemies for enemy picks', () => {
    expect(assignOptionsFor('theirs', ['Me#222'], [enemy])).toEqual([{ value: '1:foe#333', label: 'Foe' }])
  })

  it('moves a player to the newly chosen hero instead of refusing', () => {
    let a = assignPlayer({}, 4, ally.id)
    a = assignPlayer(a, 7, 'Me#222')
    a = assignPlayer(a, 8, ally.id)
    expect(a).toEqual({ 7: 'Me#222', 8: ally.id })
    expect(assignPlayer(a, 8, '')).toEqual({ 7: 'Me#222' })
  })

  it('pins heroes to the assigned players on both teams', () => {
    const a = { 4: ally.id, 7: 'me#222', 5: enemy.id }
    expect(pinsFor([{ hero: 'Abathur', stepIndex: 4 }, { hero: 'Valla', stepIndex: 7 }], a, [party, ally])).toEqual({ Abathur: ally.id, Valla: party.id })
    expect(pinsFor([{ hero: 'Johanna', stepIndex: 5 }], a, [enemy])).toEqual({ Johanna: enemy.id })
  })
})

describe('enemy pins reach the estimate', () => {
  it('an assigned enemy changes the personal value', async () => {
    const { personalValue } = await import('@/lib/personal/draft-personal')
    const c = { b0: 0, b1: 1 }
    const strong = { ...player('e1', 'Strong'), s: { Johanna: 0.06 } }
    const weak = { ...player('e2', 'Weak'), s: { Johanna: -0.04 } }
    const free = personalValue(0.5, [], ['Johanna'], [], [strong, weak], c).value
    const pinned = personalValue(0.5, [], ['Johanna'], [], [strong, weak], c, {}, { Johanna: 'e2' }).value
    expect(pinned).toBeGreaterThan(free)
  })
})
