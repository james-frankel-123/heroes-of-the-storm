/**
 * Per player x hero skill estimates for the site (docs/personalization-site-design.md).
 *
 * TypeScript port of the paper-3 rich-CF posterior (training/personalization/p3_rcf.py
 * `posterior`), the play-volume prior card (p3_rcf_tails.py `post`, card "vol") and the
 * per-player temporal layer, with constants from model-artifact.json
 * (training/personalization/p3_site_export.py artifact). The reference outputs in
 * __tests__/fixtures/reference.json come from the Python functions themselves.
 *
 * Units: every skill value is a win-probability deviation (0.03 = +3 percentage points)
 * against the population win rate on that hero.
 */
import artifactJson from './model-artifact.json'

export interface ModelArtifact {
  version: string
  heroNames: string[]
  npEdges: number[]
  nphEdges: number[]
  expTable: number[]
  proxy: { q0: number; table: number[][]; base: number[]; clip: [number, number] | number[] }
  main: {
    U: number[][]
    s2: number[]
    scale: { c: number; k0: number }
    layer: LayerConsts
    gaussLayer: LayerConsts
    volParams: number[]
    volEdges: number[]
    volSpecialist: number
    combiner: number[]
  }
  imp2: {
    U: number[][]
    s2: number[]
    W: number[][]
    zmean: number[]
    zsd: number[]
    scale: { c: number; k0: number }
  }
  grid: { lo: number; hi: number; n: number }
  validation: Validation
}

/** Held-out results the UI quotes (results/rcf_full: p3_c_tests_main, cover JSONs). */
export interface Validation {
  accuracy: { v2: { a2s: number; imp2: number; main: number }; oot: { a2s: number; imp2: number; main: number } }
  cover80: { main: number; imp2: number }
}

export interface LayerConsts {
  hl_d: number
  kd: number
  ke: number
  drift_scale: number
  learn_scale: number
}

/** One Storm League game of the player. hero is the model index; day = days since 1970-01-01 (UTC). */
export interface HistoryRow {
  hero: number
  win: boolean
  day: number
}

/** Corpus scoreboard evidence reduced to the factor space (p3_site_export.evidence). */
export interface MetricEvidence {
  Lam: number[][]
  eta: number[]
}

export const ARTIFACT = artifactJson as unknown as ModelArtifact
export const H = ARTIFACT.heroNames.length
export const Z80 = 1.2815515655446004

const HERO_INDEX = new Map(ARTIFACT.heroNames.map((n, i) => [n, i]))

/** Model index of a hero name as Heroes Profile spells it, or -1. */
export function heroIndex(name: string): number {
  return HERO_INDEX.get(name) ?? -1
}

// ---------------------------------------------------------------- small linear algebra

/** Solve A x = b for a small symmetric positive definite A (Cholesky). Returns x and L. */
function cholesky(A: number[][]): number[][] {
  const n = A.length
  const L = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j]
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]
      if (i === j) {
        if (s <= 0) throw new Error('matrix not positive definite')
        L[i][i] = Math.sqrt(s)
      } else {
        L[i][j] = s / L[j][j]
      }
    }
  }
  return L
}

function forward(L: number[][], b: number[]): number[] {
  const n = L.length
  const y = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k]
    y[i] = s / L[i][i]
  }
  return y
}

function backward(L: number[][], y: number[]): number[] {
  const n = L.length
  const x = new Array<number>(n)
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]
    x[i] = s / L[i][i]
  }
  return x
}

// ---------------------------------------------------------------- inputs

/** Quarters since 1970 (p3_c_merged.quarter): month index // 3. */
export function quarterOfDay(day: number): number {
  const d = new Date(day * 86_400_000)
  const months = (d.getUTCFullYear() - 1970) * 12 + d.getUTCMonth()
  return Math.floor(months / 3)
}

/** Win-rate proxy for a game without a draft (p3_c_merged.wp_proxy, "hero_era"). */
export function proxyOf(hero: number, day: number, A: ModelArtifact = ARTIFACT): number {
  const { q0, table, base, clip } = A.proxy
  const nq = table[0].length
  const q = quarterOfDay(day)
  const p = q < q0 ? base[hero] : table[hero][Math.min(Math.max(q - q0, 0), nq - 1)]
  return Math.min(Math.max(p, clip[0]), clip[1])
}

/** numpy searchsorted(edges, x, side="right") - 1 */
function binOf(edges: number[], x: number): number {
  let i = 0
  while (i < edges.length && edges[i] <= x) i++
  return i - 1
}

/** Experience offset by (games overall, games on the hero) before the game (p3_hs_core.exp_bins). */
export function expOffset(nP: number, nPh: number, A: ModelArtifact = ARTIFACT): number {
  return A.expTable[binOf(A.npEdges, nP) * A.nphEdges.length + binOf(A.nphEdges, nPh)]
}

interface Prepared {
  rows: HistoryRow[]
  nP: number[]
  nPh: number[]
  r: number[]
  v: number[]
  rAdj: number[]
}

/** Rows sorted by day with lag-1 counts (games on EARLIER days), residuals and offsets. */
export function prepare(rowsIn: HistoryRow[], A: ModelArtifact = ARTIFACT): Prepared {
  const rows = rowsIn
    .filter(x => x.hero >= 0 && x.hero < H && Number.isFinite(x.day))
    .map((x, i) => ({ ...x, i }))
    .sort((a, b) => a.day - b.day || a.i - b.i)
  const n = rows.length
  const nP = new Array<number>(n)
  const nPh = new Array<number>(n)
  const r = new Array<number>(n)
  const v = new Array<number>(n)
  const rAdj = new Array<number>(n)
  const cntH = new Array<number>(H).fill(0)
  let cnt = 0
  let i = 0
  while (i < n) {
    let j = i
    while (j < n && rows[j].day === rows[i].day) j++
    for (let k = i; k < j; k++) {
      nP[k] = cnt
      nPh[k] = cntH[rows[k].hero]
    }
    for (let k = i; k < j; k++) {
      cnt++
      cntH[rows[k].hero]++
    }
    i = j
  }
  for (let k = 0; k < n; k++) {
    const p = proxyOf(rows[k].hero, rows[k].day, A)
    r[k] = (rows[k].win ? 1 : 0) - p
    v[k] = p * (1 - p)
    rAdj[k] = r[k] - expOffset(nP[k], nPh[k], A)
  }
  return { rows: rows.map(({ hero, win, day }) => ({ hero, win, day })), nP, nPh, r, v, rAdj }
}

interface State {
  S: number[]
  P: number[]
  N: number[]
}

function stateBefore(pr: Prepared, asOf: number): State {
  const S = new Array<number>(H).fill(0)
  const P = new Array<number>(H).fill(0)
  const N = new Array<number>(H).fill(0)
  for (let k = 0; k < pr.rows.length; k++) {
    const row = pr.rows[k]
    if (row.day >= asOf) continue
    S[row.hero] += pr.rAdj[k] / pr.v[k]
    P[row.hero] += 1 / pr.v[k]
    N[row.hero] += 1
  }
  return { S, P, N }
}

// ---------------------------------------------------------------- posterior

interface FactorModel {
  U: number[][]
  s2: number[]
  W?: number[][]
  zmean?: number[]
  zsd?: number[]
}

/** Play preference z (p3_rcf.z_features), rounded to float32 as the reference does. */
export function zFeatures(N: number[], m: FactorModel): number[] {
  const L = N.map(x => Math.log1p(x))
  const tot = L.reduce((a, b) => a + b, 0)
  const F = [...L.map(x => (tot > 0 ? x / Math.max(tot, 1e-9) : 0)), Math.log1p(N.reduce((a, b) => a + b, 0))]
  return F.map((f, j) => (tot > 0 ? Math.fround((f - m.zmean![j]) / m.zsd![j]) : 0))
}

interface Factor {
  L: number[][]
  ahat: number[]
}

function factorPosterior(m: FactorModel, S: number[], P: number[], N: number[] | null, metric: MetricEvidence | null): Factor {
  const K = m.U[0].length
  const Lam = Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, j) => (i === j ? 1 : 0)))
  const eta = new Array<number>(K).fill(0)
  for (let h = 0; h < H; h++) {
    const den = 1 + m.s2[h] * P[h]
    const w = P[h] / den
    const wy = S[h] / den
    const u = m.U[h]
    for (let i = 0; i < K; i++) {
      eta[i] += wy * u[i]
      if (w === 0) continue
      for (let j = 0; j < K; j++) Lam[i][j] += w * u[i] * u[j]
    }
  }
  if (m.W && N) {
    const z = zFeatures(N, m)
    for (let i = 0; i < K; i++) {
      let s = 0
      for (let j = 0; j < z.length; j++) s += z[j] * m.W[i][j]
      eta[i] += s
    }
  }
  if (metric) {
    for (let i = 0; i < K; i++) {
      eta[i] += metric.eta[i]
      for (let j = 0; j < K; j++) Lam[i][j] += metric.Lam[i][j]
    }
  }
  const L = cholesky(Lam)
  const ahat = backward(L, forward(L, eta))
  return { L, ahat }
}

function atHero(m: FactorModel, f: Factor, S: number[], P: number[], h: number): { mean: number; var: number } {
  const den = 1 + m.s2[h] * P[h]
  const w = P[h] / den
  const wy = S[h] / den
  const a = 1 - m.s2[h] * w
  const u = m.U[h]
  let ua = 0
  for (let i = 0; i < u.length; i++) ua += u[i] * f.ahat[i]
  const q = forward(f.L, u)
  let qq = 0
  for (const x of q) qq += x * x
  return { mean: a * ua + m.s2[h] * wy, var: a * a * qq + m.s2[h] * a }
}

/** Posterior mean and variance of skill at every hero (p3_rcf.posterior, want_all). */
export function posteriorAll(
  m: FactorModel,
  S: number[],
  P: number[],
  opts: { N?: number[]; metric?: MetricEvidence | null } = {},
): { mean: number[]; var: number[] } {
  const f = factorPosterior(m, S, P, opts.N ?? null, opts.metric ?? null)
  const mean = new Array<number>(H)
  const v = new Array<number>(H)
  for (let h = 0; h < H; h++) {
    const o = atHero(m, f, S, P, h)
    mean[h] = o.mean
    v[h] = o.var
  }
  return { mean, var: v }
}

/** Prior at hero h from everything except h's own win record (the tail card's m0, v0). */
export function leaveOwnHeroOut(
  m: FactorModel,
  S: number[],
  P: number[],
  h: number,
  metric: MetricEvidence | null,
): { mean: number; var: number } {
  const S0 = S.slice()
  const P0 = P.slice()
  S0[h] = 0
  P0[h] = 0
  const f = factorPosterior(m, S0, P0, null, metric)
  return atHero(m, f, S0, P0, h)
}

// ---------------------------------------------------------------- play-volume card

let GRID: number[] | null = null
function grid(A: ModelArtifact): number[] {
  if (!GRID) {
    const { lo, hi, n } = A.grid
    const step = (hi - lo) / (n - 1)
    GRID = Array.from({ length: n }, (_, i) => (i === n - 1 ? hi : lo + i * step))
  }
  return GRID
}

/** Play-volume cell: games on the hero (0, 1-9, 10-29, 30-99, 100+) x specialist (share >= 20%). */
export function volBin(nh: number, np: number, A: ModelArtifact = ARTIFACT): number {
  const b = binOf(A.main.volEdges, nh)
  const spec = nh / Math.max(np, 1) >= A.main.volSpecialist ? 1 : 0
  return b * 2 + spec
}

/** Posterior of skill under the play-volume prior (p3_rcf_tails.post, card "vol"). */
export function volPosterior(
  m0: number,
  v0: number,
  Sh: number,
  Ph: number,
  bin: number,
  A: ModelArtifact = ARTIFACT,
): { mean: number; var: number } {
  const x = grid(A)
  const p = A.main.volParams
  const mu = p[bin]
  const s = Math.exp(p[10 + bin])
  const s0 = Math.sqrt(Math.max(v0, 1e-10))
  const yb = Ph > 0 ? Sh / Math.max(Ph, 1e-12) : 0
  const lw = new Array<number>(x.length)
  let mx = -Infinity
  for (let i = 0; i < x.length; i++) {
    const z = (x[i] - mu) / s
    const th = m0 + s0 * x[i]
    const ll = Ph > 0 ? -0.5 * Ph * (yb - th) ** 2 : 0
    lw[i] = -0.5 * z * z - Math.log(s) + ll
    if (lw[i] > mx) mx = lw[i]
  }
  let sw = 0
  for (let i = 0; i < x.length; i++) {
    lw[i] = Math.exp(lw[i] - mx)
    sw += lw[i]
  }
  let mean = 0
  let m2 = 0
  for (let i = 0; i < x.length; i++) {
    const w = lw[i] / sw
    const th = m0 + s0 * x[i]
    mean += w * th
    m2 += w * th * th
  }
  return { mean, var: m2 - mean * mean }
}

// ---------------------------------------------------------------- temporal layer

/**
 * Per-player drift and new-hero learning terms (p3_rcf.temporal_layer, volume arm).
 * Walks the player's games by day: u = r - (offset + volume-card mean), state through the
 * previous day; the metric evidence is the current summary throughout.
 */
export function layerTerms(
  pr: Prepared,
  asOf: number,
  metric: MetricEvidence,
  A: ModelArtifact = ARTIFACT,
): { rho: number; nu: number; u: number[] } {
  const lay = A.main.layer
  const m = A.main
  const n = pr.rows.length
  const u = new Array<number>(n).fill(NaN)
  const early = new Array<boolean>(n).fill(false)
  const S = new Array<number>(H).fill(0)
  const P = new Array<number>(H).fill(0)
  const N = new Array<number>(H).fill(0)
  let total = 0
  let i = 0
  while (i < n && pr.rows[i].day < asOf) {
    let j = i
    while (j < n && pr.rows[j].day === pr.rows[i].day) j++
    for (let k = i; k < j; k++) {
      const h = pr.rows[k].hero
      const o = leaveOwnHeroOut(m, S, P, h, metric)
      const t = volPosterior(o.mean, o.var, S[h], P[h], volBin(N[h], total, A), A)
      u[k] = pr.r[k] - (expOffset(pr.nP[k], pr.nPh[k], A) + t.mean)
      early[k] = N[h] < 5
    }
    for (let k = i; k < j; k++) {
      const h = pr.rows[k].hero
      S[h] += pr.rAdj[k] / pr.v[k]
      P[h] += 1 / pr.v[k]
      N[h] += 1
      total += 1
    }
    i = j
  }
  const lam = Math.log(2) / lay.hl_d
  let sw = 0
  let swu = 0
  let se = 0
  let ne = 0
  for (let k = 0; k < n; k++) {
    if (!(pr.rows[k].day < asOf) || !Number.isFinite(u[k])) continue
    const w = Math.exp(-lam * (asOf - pr.rows[k].day))
    sw += w
    swu += w * u[k]
    if (early[k]) {
      se += u[k]
      ne += 1
    }
  }
  return { rho: swu / (sw + lay.kd), nu: se / (ne + lay.ke), u }
}

// ---------------------------------------------------------------- profile

export interface HeroEstimate {
  hero: string
  games: number
  wins: number
  lastDay: number
  /** Expected win-probability deviation in the player's next game on this hero. */
  mean: number
  /** Display sd: the V1-fit scale c var + k0^2 (80% band = mean +- 1.2816 sd). */
  sd: number
  lo: number
  hi: number
}

export interface Suggestion {
  hero: string
  /** Hero-specific skill once learned (no first-games offset). */
  mean: number
  sd: number
  lo: number
  hi: number
}

export interface Profile {
  model: 'main' | 'imp2'
  modelVersion: string
  asOfDay: number
  games: number
  heroes: HeroEstimate[]
  suggestions: Suggestion[]
}

export interface ProfileDetail {
  N: number[]
  offset: number[]
  imp2: { mean: number[]; var: number[] }
  main?: {
    gaussMean: number[]
    gaussVar: number[]
    m0: number[]
    v0: number[]
    volMean: number[]
    volVar: number[]
    rho: number
    nu: number
    shift: number[]
    u: number[]
  }
}

/** Every intermediate quantity the reference fixture checks. */
export function computeDetail(
  rows: HistoryRow[],
  asOf: number,
  metric: MetricEvidence | null,
  A: ModelArtifact = ARTIFACT,
): ProfileDetail {
  const pr = prepare(rows, A)
  const { S, P, N } = stateBefore(pr, asOf)
  const nTot = N.reduce((a, b) => a + b, 0)
  const offset = N.map(nh => expOffset(nTot, nh, A))
  const imp2 = posteriorAll(A.imp2, S, P, { N })
  const out: ProfileDetail = { N, offset, imp2 }
  if (!metric) return out
  const g = posteriorAll(A.main, S, P, { metric })
  const m0 = new Array<number>(H)
  const v0 = new Array<number>(H)
  const volMean = new Array<number>(H)
  const volVar = new Array<number>(H)
  for (let h = 0; h < H; h++) {
    const o = leaveOwnHeroOut(A.main, S, P, h, metric)
    m0[h] = o.mean
    v0[h] = o.var
    const t = volPosterior(o.mean, o.var, S[h], P[h], volBin(N[h], nTot, A), A)
    volMean[h] = t.mean
    volVar[h] = t.var
  }
  const { rho, nu, u } = layerTerms(pr, asOf, metric, A)
  const lay = A.main.layer
  const shift = N.map(nh => lay.drift_scale * rho + lay.learn_scale * nu * (nh < 5 ? 1 : 0))
  out.main = { gaussMean: g.mean, gaussVar: g.var, m0, v0, volMean, volVar, rho, nu, shift, u }
  return out
}

export function dayOf(date: Date | string): number {
  const t = typeof date === 'string' ? Date.parse(date.length <= 10 ? `${date}T00:00:00Z` : date) : date.getTime()
  return Math.floor(t / 86_400_000)
}

/**
 * The cards: played heroes (main model through the volume card when corpus scoreboard
 * evidence exists, else imp2) and the never-played ranking (imp2).
 */
export function computeProfile(
  rows: HistoryRow[],
  asOf: number,
  metric: MetricEvidence | null,
  opts: { suggestions?: number } = {},
  A: ModelArtifact = ARTIFACT,
): Profile {
  const d = computeDetail(rows, asOf, metric, A)
  const wins = new Array<number>(H).fill(0)
  const last = new Array<number>(H).fill(-1)
  for (const r of rows) {
    if (r.hero < 0 || r.hero >= H || r.day >= asOf) continue
    if (r.win) wins[r.hero]++
    if (r.day > last[r.hero]) last[r.hero] = r.day
  }
  const band = (mean: number, v: number, sc: { c: number; k0: number }) => {
    const sd = Math.sqrt(sc.c * Math.max(v, 0) + sc.k0 * sc.k0)
    return { mean, sd, lo: mean - Z80 * sd, hi: mean + Z80 * sd }
  }
  const heroes: HeroEstimate[] = []
  for (let h = 0; h < H; h++) {
    if (d.N[h] <= 0) continue
    const b = d.main
      ? band(d.offset[h] + d.main.volMean[h] + d.main.shift[h], d.main.gaussVar[h], A.main.scale)
      : band(d.offset[h] + d.imp2.mean[h], d.imp2.var[h], A.imp2.scale)
    heroes.push({ hero: A.heroNames[h], games: d.N[h], wins: wins[h], lastDay: last[h], ...b })
  }
  heroes.sort((a, b) => b.games - a.games || b.mean - a.mean)
  const never: Suggestion[] = []
  for (let h = 0; h < H; h++) {
    if (d.N[h] > 0) continue
    never.push({ hero: A.heroNames[h], ...band(d.imp2.mean[h], d.imp2.var[h], A.imp2.scale) })
  }
  never.sort((a, b) => b.mean - a.mean)
  return {
    model: d.main ? 'main' : 'imp2',
    modelVersion: A.version,
    asOfDay: asOf,
    games: d.N.reduce((a, b) => a + b, 0),
    heroes,
    suggestions: never.slice(0, opts.suggestions ?? 5),
  }
}
