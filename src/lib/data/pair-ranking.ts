/**
 * Ranking for the dashboard's synergy and counter lists. Pure; tested in
 * src/lib/__tests__/pair-ranking.test.ts.
 *
 * Pairs need evidence: at least MIN_PAIR_GAMES games, ordered by the displayed win rate
 * pulled toward 50% by PAIR_PRIOR_GAMES phantom games. The score uses winRate (the field
 * shown), never the stored win count, so ranking and display cannot disagree.
 */
import type { HeroPairwiseStats } from '@/lib/types'

export const MIN_PAIR_GAMES = 50
export const PAIR_PRIOR_GAMES = 50

export function pairScore(p: { winRate: number; games: number }): number {
  return ((p.winRate / 100) * p.games + PAIR_PRIOR_GAMES / 2) / (p.games + PAIR_PRIOR_GAMES)
}

/**
 * Best pairs first, one row per unordered pair: for directed counter rows the stronger
 * direction represents the pair, whatever the heroes' alphabetical order.
 */
export function rankPairs(rows: HeroPairwiseStats[], limit: number): HeroPairwiseStats[] {
  const eligible = rows.filter((r) => r.games >= MIN_PAIR_GAMES)
  return mergeChoGallPairwise([...eligible].sort((a, b) => pairScore(b) - pairScore(a))).slice(0, limit)
}

/**
 * Merge Cho/Gall references in pairwise stats into "Cho'gall".
 * The DB has separate rows for Cho+X and Gall+X with identical stats.
 * Rename both to Cho'gall, drop the Cho+Gall self-pair, then dedup
 * by keeping the first occurrence of each (heroA, heroB) pair.
 */
export function mergeChoGallPairwise(pairs: HeroPairwiseStats[]): HeroPairwiseStats[] {
  const renamed = pairs.map((p) => ({
    ...p,
    heroA: p.heroA === 'Cho' || p.heroA === 'Gall' ? "Cho'gall" : p.heroA,
    heroB: p.heroB === 'Cho' || p.heroB === 'Gall' ? "Cho'gall" : p.heroB,
  }))

  // Drop self-pairs (Cho'gall + Cho'gall)
  const filtered = renamed.filter((p) => p.heroA !== p.heroB)

  // Dedup: keep first occurrence of each pair (order-independent key,
  // since Cho→Fenix and Fenix→Gall both become Cho'gall↔Fenix after rename)
  const seen = new Set<string>()
  return filtered.filter((p) => {
    const [a, b] = [p.heroA, p.heroB].sort()
    const key = `${a}|${b}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

