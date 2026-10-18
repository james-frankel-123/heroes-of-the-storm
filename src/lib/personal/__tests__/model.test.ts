import { describe, expect, it } from 'vitest'
import fixture from './fixtures/reference.json'
import {
  ARTIFACT,
  H,
  computeDetail,
  computeProfile,
  expOffset,
  proxyOf,
  quarterOfDay,
  type HistoryRow,
  type MetricEvidence,
} from '../model'

interface Case {
  label: string
  rows: { hero: number[]; y: number[]; day: number[] }
  metric?: MetricEvidence & { n_rows: number }
  as_of: number
  expected: {
    N: number[]
    offset: number[]
    imp2: { mean: number[]; var: number[] }
    main?: {
      gauss_mean: number[]
      gauss_var: number[]
      m0: number[]
      v0: number[]
      vol_mean: number[]
      vol_var: number[]
      rho: number
      nu: number
      shift: number[]
      u: (number | null)[]
    }
  }
}

const cases = (fixture as unknown as { cases: Case[] }).cases

function rowsOf(c: Case): HistoryRow[] {
  return c.rows.hero.map((hero, i) => ({ hero, win: c.rows.y[i] === 1, day: c.rows.day[i] }))
}

// Skill values are win-probability deviations; 1e-7 is 0.00001 percentage points.
const TOL = 1e-7

function close(a: number[], b: number[], tol = TOL) {
  expect(a.length).toBe(b.length)
  let worst = 0
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]))
  expect(worst).toBeLessThanOrEqual(tol)
}

describe('personal skill model vs the Python reference', () => {
  it('has the reference cases', () => {
    expect(cases.length).toBeGreaterThanOrEqual(8)
    expect(cases.filter(c => c.label.startsWith('real_')).length).toBe(3)
  })

  for (const c of cases) {
    it(`${c.label}: counts, offsets and imp2 posterior`, () => {
      const d = computeDetail(rowsOf(c), c.as_of, c.metric ?? null)
      close(d.N, c.expected.N, 0)
      close(d.offset, c.expected.offset, 1e-12)
      close(d.imp2.mean, c.expected.imp2.mean)
      close(d.imp2.var, c.expected.imp2.var, 1e-9)
    })
    if (!c.expected.main) continue
    it(`${c.label}: main posterior, volume card and temporal layer`, () => {
      const d = computeDetail(rowsOf(c), c.as_of, c.metric!)
      const e = c.expected.main!
      const m = d.main!
      close(m.gaussMean, e.gauss_mean)
      close(m.gaussVar, e.gauss_var, 1e-9)
      close(m.m0, e.m0)
      close(m.v0, e.v0, 1e-9)
      close(m.volMean, e.vol_mean)
      close(m.volVar, e.vol_var, 1e-9)
      expect(Math.abs(m.rho - e.rho)).toBeLessThan(TOL)
      expect(Math.abs(m.nu - e.nu)).toBeLessThan(TOL)
      close(m.shift, e.shift)
      const uRef = e.u.map(x => (x === null ? NaN : x))
      const sorted = rowsOf(c)
      expect(m.u.length).toBe(sorted.length)
      close(m.u.filter(Number.isFinite), uRef.filter(Number.isFinite))
    })
  }
})

describe('personal skill model: profile and edge cases', () => {
  it('empty history gives no hero cards and a full never-played ranking', () => {
    const p = computeProfile([], 20_000, null)
    expect(p.model).toBe('imp2')
    expect(p.heroes).toHaveLength(0)
    expect(p.suggestions).toHaveLength(5)
    for (const s of p.suggestions) expect(s.hi).toBeGreaterThan(s.lo)
  })

  it('uses the main model only with metric evidence', () => {
    const c = cases.find(x => x.label === 'real_mid')!
    const withM = computeProfile(rowsOf(c), c.as_of, c.metric!)
    const without = computeProfile(rowsOf(c), c.as_of, null)
    expect(withM.model).toBe('main')
    expect(without.model).toBe('imp2')
    expect(withM.heroes.length).toBe(without.heroes.length)
    const games = withM.heroes.reduce((a, h) => a + h.games, 0)
    expect(games).toBe(c.rows.hero.length)
  })

  it('band is the 80% interval of the display scale', () => {
    const c = cases.find(x => x.label === 'real_heavy')!
    const p = computeProfile(rowsOf(c), c.as_of, c.metric!)
    const top = p.heroes[0]
    expect(top.hi - top.mean).toBeCloseTo(1.2815515655446004 * top.sd, 12)
    expect(top.sd).toBeGreaterThan(ARTIFACT.main.scale.k0)
  })

  it('ignores games on the request day and unknown heroes', () => {
    const rows: HistoryRow[] = [
      { hero: 3, win: true, day: 19_000 },
      { hero: 3, win: false, day: 19_001 },
      { hero: -1, win: true, day: 19_000 },
    ]
    const p = computeProfile(rows, 19_001, null)
    expect(p.games).toBe(1)
  })

  it('offsets and proxies follow the bins', () => {
    expect(expOffset(0, 0)).toBe(ARTIFACT.expTable[0])
    expect(expOffset(10_000, 10_000)).toBe(ARTIFACT.expTable[ARTIFACT.expTable.length - 1])
    // quarters since 1970: 2026-10-18 is 2026 Q4
    expect(quarterOfDay(Math.floor(Date.UTC(2026, 9, 18) / 86_400_000))).toBe(56 * 4 + 3)
    for (let h = 0; h < H; h++) {
      const p = proxyOf(h, 20_700)
      expect(p).toBeGreaterThanOrEqual(0.2)
      expect(p).toBeLessThanOrEqual(0.8)
    }
  })
})
