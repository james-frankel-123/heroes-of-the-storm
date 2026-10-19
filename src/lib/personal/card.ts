/**
 * Player cards published by training/personalization/p3_site_publish.py
 * (docs/personalization-site-design.md). Server only: decodes the bytea card and expands
 * the draft term s(p, h) for all 90 heroes.
 *
 * Units: skill values are win-probability deviations (0.031 = +3.1 percentage points)
 * against the population win rate on the hero.
 */
import artifactJson from './model-artifact.json'

interface Artifact {
  version: string
  heroNames: string[]
  npEdges: number[]
  nphEdges: number[]
  expTable: number[]
  main: { U: number[][] }
  imp2: { U: number[][] }
  draft: { main: number[]; imp2: number[]; offRole: number; offRoleImp2: number; offRoleShare: number; offRoleMinGames: number }
  heroFineRole: number[]
  validation: {
    accuracy: { v2: Record<string, number>; oot: Record<string, number> }
    cover80: Record<string, number>
  }
}

export const ARTIFACT = artifactJson as unknown as Artifact
export const HERO_NAMES = ARTIFACT.heroNames
export const H = HERO_NAMES.length
const EPOCH14 = Date.UTC(2014, 0, 1) / 86_400_000
export const CARD_VERSION = 1

export interface PlayedHero {
  hero: string
  games: number
  lastPlayed: string
  /** Displayed estimate: next game on this hero vs an average player (play-volume card). */
  mean: number
  /** Display sd; the 80% range is mean +- 1.2816 sd. */
  sd: number
  /** The draft term s(p, h) of the model's own combiner (Gaussian posterior). */
  draft: number
}

export interface SuggestedHero {
  hero: string
  mean: number
  sd: number
}

export interface Card {
  model: 'main' | 'imp2'
  games: number
  ahat: [number, number, number]
  played: PlayedHero[]
  suggestions: SuggestedHero[]
}

function dayToIso(day: number): string {
  return new Date(day * 86_400_000).toISOString().slice(0, 10)
}

export function decodeCard(buf: Uint8Array): Card {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  let o = 0
  const version = dv.getUint8(o++)
  if (version !== CARD_VERSION) throw new Error(`card version ${version} not supported`)
  const model = dv.getUint8(o++) === 1 ? 'main' : 'imp2'
  const games = dv.getUint32(o, true)
  o += 4
  const ahat: [number, number, number] = [0, 0, 0]
  for (let i = 0; i < 3; i++, o += 4) ahat[i] = dv.getFloat32(o, true)
  const k = dv.getUint8(o++)
  const played: PlayedHero[] = []
  for (let i = 0; i < k; i++) {
    const h = dv.getUint8(o)
    const n = dv.getUint16(o + 1, true)
    const last = dv.getUint16(o + 3, true)
    const mean = dv.getInt16(o + 5, true) / 1e4
    const sd = dv.getUint8(o + 7) / 1e3
    const draft = dv.getInt16(o + 8, true) / 1e4
    o += 10
    played.push({ hero: HERO_NAMES[h], games: n, lastPlayed: dayToIso(EPOCH14 + last), mean, sd, draft })
  }
  const j = dv.getUint8(o++)
  const suggestions: SuggestedHero[] = []
  for (let i = 0; i < j; i++) {
    suggestions.push({ hero: HERO_NAMES[dv.getUint8(o)], mean: dv.getInt16(o + 1, true) / 1e4, sd: dv.getUint8(o + 3) / 1e3 })
    o += 4
  }
  if (o !== buf.byteLength) throw new Error(`card length ${buf.byteLength}, decoded ${o}`)
  return { model, games, ahat, played, suggestions }
}

/** numpy searchsorted(edges, x, side="right") - 1 */
function binOf(edges: number[], x: number): number {
  let i = 0
  while (i < edges.length && edges[i] <= x) i++
  return i - 1
}

export function expOffset(nP: number, nPh: number): number {
  const A = ARTIFACT
  return A.expTable[binOf(A.npEdges, nP) * A.nphEdges.length + binOf(A.nphEdges, nPh)]
}

export interface DraftProfile {
  model: 'main' | 'imp2'
  /** s(p, h) for every hero, model order (HERO_NAMES). */
  s: number[]
  /** Off-role flag per hero (P3_DRAFTER: fine role under 10% of 50+ games). */
  offRole: boolean[]
  /** This player's combiner weight on S (b2 of its model). */
  b2: number
  /** This player's off-role weight (b3 of its model). */
  b3: number
}

/**
 * The draft term for every hero. Played heroes carry the published value; a never-played
 * hero's posterior mean is U_h . a_hat (no own evidence), plus the offset at 0 games on it.
 */
export function draftProfile(card: Card): DraftProfile {
  const A = ARTIFACT
  const U = card.model === 'main' ? A.main.U : A.imp2.U
  const s = new Array<number>(H)
  for (let h = 0; h < H; h++) {
    s[h] = expOffset(card.games, 0) + U[h][0] * card.ahat[0] + U[h][1] * card.ahat[1] + U[h][2] * card.ahat[2]
  }
  const roleGames = new Array<number>(A.heroFineRole.reduce((a, b) => Math.max(a, b), 0) + 1).fill(0)
  for (const p of card.played) {
    const h = HERO_NAMES.indexOf(p.hero)
    s[h] = p.draft
    roleGames[A.heroFineRole[h]] += p.games
  }
  const total = card.played.reduce((a, p) => a + p.games, 0)
  const offRole = HERO_NAMES.map((_, h) =>
    total >= A.draft.offRoleMinGames && roleGames[A.heroFineRole[h]] < A.draft.offRoleShare * total,
  )
  const b2 = (card.model === 'main' ? A.draft.main : A.draft.imp2)[2]
  const b3 = card.model === 'main' ? A.draft.offRole : A.draft.offRoleImp2
  return { model: card.model, s, offRole, b2, b3 }
}

/** The draft term keyed by hero name (what the page's draft code consumes). */
export function draftByName(card: Card): {
  model: Card['model']
  s: Record<string, number>
  offRole: Record<string, boolean>
  b2: number
  b3: number
} {
  const d = draftProfile(card)
  const s: Record<string, number> = {}
  const offRole: Record<string, boolean> = {}
  HERO_NAMES.forEach((n, h) => {
    s[n] = Math.round(d.s[h] * 1e5) / 1e5
    if (d.offRole[h]) offRole[n] = true
  })
  return { model: d.model, s, offRole, b2: d.b2, b3: d.b3 }
}

/** b0, b1 of the main model's combiner (b2 and b3 travel with each player). */
export function draftCoefs(): { b0: number; b1: number } {
  return { b0: ARTIFACT.draft.main[0], b1: ARTIFACT.draft.main[1] }
}
