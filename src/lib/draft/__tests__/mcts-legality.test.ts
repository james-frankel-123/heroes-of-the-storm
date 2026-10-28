import { describe, expect, it } from 'vitest'
import { runMCTSSearch } from '../mcts-search'
import { HEROES, HERO_TO_IDX } from '../encoding'
import { lineups } from '../rules'

function mulberry32(a: number) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Synthetic sessions: random logits, with Cho and Gall made attractive so the search tries
// them often (the R4-02 path); value 0.5.
function fakeOrt(seed: number) {
  const rnd = mulberry32(seed)
  const logits = () => {
    const x = new Float32Array(HEROES.length)
    for (let i = 0; i < x.length; i++) x[i] = rnd() * 4
    x[HERO_TO_IDX['Cho']] += 3
    x[HERO_TO_IDX['Gall']] += 3
    return x
  }
  const ort = { Tensor: class { constructor(public type: string, public data: Float32Array, public dims: number[]) {} } }
  const policy = { run: async () => ({ policy_logits: { data: logits() }, value: { data: new Float32Array([0.5]) } }) }
  const gd = { run: async () => ({ hero_logits: { data: logits() } }) }
  return { ort, policy, gd }
}

const ROOTS = [
  // our first pick as Team A after bans
  { team0Picks: [], team1Picks: [], bans: ['Ana', 'Li Li', 'Uther', 'Rehgar'], step: 4, ourTeam: 0, stepType: 'pick' as const },
  // Team B's double pick
  { team0Picks: ['Abathur'], team1Picks: [], bans: ['Ana', 'Li Li', 'Uther', 'Rehgar'], step: 5, ourTeam: 1, stepType: 'pick' as const },
  // our bans, with opponent picks to come
  { team0Picks: ['Abathur', 'Valla', 'Muradin'], team1Picks: ['Johanna', 'Anduin'], bans: ['Ana', 'Li Li', 'Uther', 'Rehgar', 'Diablo'], step: 10, ourTeam: 0, stepType: 'ban' as const },
  // our last double pick as A, last enemy pick to come
  { team0Picks: ['Abathur', 'Valla', 'Muradin'], team1Picks: ['Johanna', 'Anduin', 'Jaina', 'Sonya'], bans: ['Ana', 'Li Li', 'Uther', 'Rehgar', 'Diablo', 'Thrall'], step: 13, ourTeam: 0, stepType: 'pick' as const },
]

describe('search legality (R4-01, R4-02)', () => {
  it('evaluates no impossible position under varied opponent samples', async () => {
    let evaluated = 0
    const problems: string[] = []
    for (const [ri, root] of ROOTS.entries()) {
      for (let seed = 1; seed <= 6; seed++) {
        const { ort, policy, gd } = fakeOrt(seed * 97 + ri)
        const taken = new Set([...root.team0Picks, ...root.team1Picks, ...root.bans])
        const res = await runMCTSSearch(
          ort, policy, gd, { ...root, map: 'Haunted Mines', tier: 'mid' }, taken, undefined,
          async () => 0.5,
          async () => 0.5,
          {
            rng: mulberry32(seed), maxSims: 120, minSims: 120, timeBudgetMs: 1e9,
            onEvaluate: (_sel, p) => { evaluated++; problems.push(...p) },
          },
        )
        expect(res.illegal).toBe(0)
        expect(res.sims).toBe(120)
        // recommendations are legal first actions
        for (const r of res.recommendations) expect(taken.has(r.hero)).toBe(false)
      }
    }
    expect(evaluated).toBeGreaterThan(2000)
    expect(problems).toEqual([])
  })

  it("explores Cho'gall only as a pair", async () => {
    let paired = 0
    const { ort, policy, gd } = fakeOrt(7)
    await runMCTSSearch(ort, policy, gd, { ...ROOTS[1], map: 'Haunted Mines', tier: 'mid' }, new Set(ROOTS[1].bans.concat(ROOTS[1].team0Picks)), undefined,
      async () => 0.5, async () => 0.5,
      { rng: mulberry32(3), maxSims: 150, minSims: 150, timeBudgetMs: 1e9, onEvaluate: sel => {
        for (const team of [lineups(sel).A, lineups(sel).B]) {
          if (team.includes('Cho') || team.includes('Gall')) { paired++; expect(team.includes('Cho') && team.includes('Gall')).toBe(true) }
        }
      } })
    expect(paired).toBeGreaterThan(0)
  })
})
