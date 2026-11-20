import { describe, expect, it } from 'vitest'
import { runMCTSSearch } from '../mcts-search'
import { HEROES, HERO_TO_IDX } from '../encoding'
import { DRAFT_SEQUENCE } from '../types'
import { searchHooks, type PersonalPlayer } from '../../personal/draft-personal'

// Policy that puts almost all mass on Muradin and next to none on Malthael; GD uniform.
function sessions() {
  const ort = { Tensor: class { constructor(public type: string, public data: Float32Array, public dims: number[]) {} } }
  const logits = () => {
    const x = new Float32Array(HEROES.length).fill(0)
    x[HERO_TO_IDX['Muradin']] = 30
    x[HERO_TO_IDX['Malthael']] = -30
    return x
  }
  const policy = { run: async () => ({ policy_logits: { data: logits() }, value: { data: new Float32Array([0.5]) } }) }
  const gd = { run: async () => ({ hero_logits: { data: new Float32Array(HEROES.length) } }) }
  return { ort, policy, gd }
}

const ROOT = { team0Picks: [], team1Picks: [], bans: ['Ana', 'Li Li', 'Uther', 'Rehgar'], map: 'Cursed Hollow', tier: 'mid', step: 4, ourTeam: 0, stepType: 'pick' as const }
const flat = async () => 0.5

function player(id: string, s: Record<string, number>): PersonalPlayer {
  return { id, label: id, s, offRole: {}, b2: 3.7, b3: 0 }
}

describe('search breadth and player-aware proposals', () => {
  it('evaluates many heroes at the root even with a near-deterministic policy', async () => {
    const { ort, policy, gd } = sessions()
    const res = await runMCTSSearch(ort, policy, gd, ROOT, new Set(ROOT.bans), undefined, flat, flat,
      { rng: () => 0.37, maxSims: 120, minSims: 120, timeBudgetMs: 1e9 })
    expect(res.recommendations.length).toBeGreaterThanOrEqual(12)
  })

  it('finds a matched player\'s strong hero the policy ignores, and ranks it first', async () => {
    const { ort, policy, gd } = sessions()
    const watson = player('watson', { Malthael: 0.12 })
    const hooks = searchHooks([watson], [], { b0: 0, b1: 1 }, true, HEROES, st => DRAFT_SEQUENCE[st]?.type === 'ban')
    const res = await runMCTSSearch(ort, policy, gd, ROOT, new Set(ROOT.bans), undefined, flat, flat, {
      rng: () => 0.37, maxSims: 200, minSims: 200, timeBudgetMs: 1e9,
      adjustLeaf: hooks.adjustLeaf, proposal: hooks.proposal,
      rootCandidates: hooks.topHeroes([], [], ROOT.step, h => !ROOT.bans.includes(h)),
    })
    const heroes = res.recommendations.map(r => r.hero)
    expect(heroes).toContain('Malthael')
    const byQ = [...res.recommendations].sort((a, b) => b.q - a.q)
    expect(byQ[0].hero).toBe('Malthael')
  })

  it('without players the search is unchanged in kind: no proposal, population leaf', async () => {
    const hooks = searchHooks([], [], { b0: 0, b1: 1 }, true, HEROES, () => false)
    expect(hooks.proposal([], [], 4, new Float32Array(HEROES.length).fill(1))).toBeNull()
    expect(hooks.adjustLeaf([], [], 0.6)).toBeCloseTo(0.6, 10)
  })
})
