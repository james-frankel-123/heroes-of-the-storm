/**
 * Player cards published by training/personalization/p3_site_publish.py
 * (docs/personalization-site-design.md). Server only: decodes the bytea card.
 *
 * Model: the chosen shrinkage model of the definitive build (model-artifact.json `version`).
 * The card carries its state writer's estimate s(p, h) for all 90 heroes and the display band.
 *
 * Units: skill values are win-probability deviations (0.031 = +3.1 percentage points)
 * against the population win rate on the hero.
 */
import artifactJson from './model-artifact.json'

interface Artifact {
  version: string
  model: string
  estimatesAsOf: string
  heroNames: string[]
  heroFineRole: number[]
  draft: { b0: number; b1: number; b2: number; b3: number; offRoleShare: number; offRoleMinGames: number }
  validation: {
    accuracy: { v2: number; oot: number }
    cover80: { v2: number; oot: number }
  }
}

export const ARTIFACT = artifactJson as unknown as Artifact
export const HERO_NAMES = ARTIFACT.heroNames
export const H = HERO_NAMES.length
const EPOCH14 = Date.UTC(2014, 0, 1) / 86_400_000
export const CARD_VERSION = 2
const MODEL_CHOSEN = 3

export interface PlayedHero {
  hero: string
  games: number
  lastPlayed: string
  /** Estimate for the next game on this hero vs an average player (the model's s). */
  mean: number
  /** Display band sd; the 80% range is mean +- 1.2816 sd. */
  sd: number
  /** The draft term s(p, h) (same value as mean). */
  draft: number
}

export interface SuggestedHero {
  hero: string
  mean: number
  sd: number
}

export interface Card {
  model: 'chosen'
  games: number
  /** s(p, h) for every hero, HERO_NAMES order. */
  s: number[]
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
  const model = dv.getUint8(o++)
  if (model !== MODEL_CHOSEN) throw new Error(`card model ${model} not supported`)
  const games = dv.getUint32(o, true)
  o += 4
  const s = new Array<number>(H)
  for (let h = 0; h < H; h++, o += 2) s[h] = dv.getInt16(o, true) / 1e4
  const k = dv.getUint8(o++)
  const played: PlayedHero[] = []
  for (let i = 0; i < k; i++) {
    const h = dv.getUint8(o)
    const n = dv.getUint16(o + 1, true)
    const last = dv.getUint16(o + 3, true)
    const sd = dv.getUint8(o + 5) / 1e3
    o += 6
    played.push({ hero: HERO_NAMES[h], games: n, lastPlayed: dayToIso(EPOCH14 + last), mean: s[h], sd, draft: s[h] })
  }
  const j = dv.getUint8(o++)
  const suggestions: SuggestedHero[] = []
  for (let i = 0; i < j; i++) {
    const h = dv.getUint8(o)
    suggestions.push({ hero: HERO_NAMES[h], mean: s[h], sd: dv.getUint8(o + 1) / 1e3 })
    o += 2
  }
  if (o !== buf.byteLength) throw new Error(`card length ${buf.byteLength}, decoded ${o}`)
  return { model: 'chosen', games, s, played, suggestions }
}

export interface DraftProfile {
  model: Card['model']
  /** s(p, h) for every hero, model order (HERO_NAMES). */
  s: number[]
  /** Off-role flag per hero (P3_DRAFTER: fine role under 10% of 50+ games). */
  offRole: boolean[]
  /** Combiner weight on S (b2). */
  b2: number
  /** Off-role weight (b3, refit by p3_site_offrole.py). */
  b3: number
}

/** The draft term for every hero (the card's s) and the off-role flags. */
export function draftProfile(card: Card): DraftProfile {
  const A = ARTIFACT
  const roleGames = new Array<number>(A.heroFineRole.reduce((a, b) => Math.max(a, b), 0) + 1).fill(0)
  for (const p of card.played) roleGames[A.heroFineRole[HERO_NAMES.indexOf(p.hero)]] += p.games
  const total = card.played.reduce((a, p) => a + p.games, 0)
  const offRole = HERO_NAMES.map((_, h) =>
    total >= A.draft.offRoleMinGames && roleGames[A.heroFineRole[h]] < A.draft.offRoleShare * total,
  )
  return { model: card.model, s: card.s.slice(), offRole, b2: A.draft.b2, b3: A.draft.b3 }
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

/** b0, b1 of the combiner (b2 and b3 travel with each player). */
export function draftCoefs(): { b0: number; b1: number } {
  return { b0: ARTIFACT.draft.b0, b1: ARTIFACT.draft.b1 }
}
