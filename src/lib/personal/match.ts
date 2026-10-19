/**
 * In-game name matching (docs/personalization-site-design.md, "Matching").
 * Pure scoring, shared by the API and the tests; the same rule as
 * training/personalization/p3_site_publish.py `match_score` (PARAMS_MATCH there),
 * which measured it on held-out lobbies.
 */

export interface MatchCandidate {
  region: number
  mmr: number | null
  /** ISO date */
  lastPlayed: string
}

export interface MatchContext {
  region: number | null
  mmr: number | null
  /** ISO date of "now" for recency (the data's end date, not the wall clock) */
  asOf: string
}

export const MATCH_PARAMS = {
  region: 10,
  mmrScale: 300,
  mmrUnknown: 1,
  recency: 1,
  recencyDays: 30,
  margin: 1.5,
  minScore: 9.5,
} as const

const dayOf = (iso: string) => Math.floor(Date.parse(`${iso.slice(0, 10)}T00:00:00Z`) / 86_400_000)

export function matchScore(c: MatchCandidate, ctx: MatchContext, p = MATCH_PARAMS): number {
  let s = ctx.region !== null && c.region === ctx.region ? p.region : 0
  s -= c.mmr !== null && ctx.mmr !== null ? Math.abs(c.mmr - ctx.mmr) / p.mmrScale : p.mmrUnknown
  s += p.recency * Math.exp(-Math.max(dayOf(ctx.asOf) - dayOf(c.lastPlayed), 0) / p.recencyDays)
  return s
}

export interface Ranked<T> {
  ranked: { item: T; score: number }[]
  /** true when the top candidate clearly wins and can be selected without asking */
  clear: boolean
}

/** Rank candidates; "clear" = top score >= minScore and ahead of the next by the margin. */
export function rankCandidates<T extends MatchCandidate>(cands: T[], ctx: MatchContext, p = MATCH_PARAMS): Ranked<T> {
  const ranked = cands
    .map(item => ({ item, score: matchScore(item, ctx, p) }))
    .sort((a, b) => b.score - a.score)
  const clear =
    ranked.length > 0 &&
    ranked[0].score >= p.minScore &&
    (ranked.length === 1 || ranked[0].score - ranked[1].score >= p.margin)
  return { ranked, clear }
}
