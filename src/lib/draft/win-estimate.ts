/**
 * Running team win % estimate.
 *
 * Starting from a 50% baseline, accumulates data-backed deltas:
 *   1. Hero base WR:     sum of (heroWR - 50) for each picked hero
 *   2. Intra-team synergies: average of (pairwiseWithWR - 50) across all ally pairs
 *   3. Cross-team counters:  average of (pairwiseVsWR - 50) across all matchups
 *   4. Player strength:  if a player is assigned, (adjustedMAWP - 50) replaces hero base WR
 *
 * Result: 50 + totalDelta, clamped to [1, 99].
 */

import type { DraftData } from './types'
import { getHeroWinRate } from './engine'
import { confidenceAdjustedMawp } from '@/lib/utils'
import { scoreCurrentComposition } from './composition'

export interface WinEstimateBreakdown {
  heroWR: number
  synergies: number
  counters: number
  playerAdj: number
  compWR: number
}

export interface WinEstimateResult {
  winPct: number
  breakdown: WinEstimateBreakdown
}

/**
 * Compute expected win % for a team given their picks and the enemy's picks.
 *
 * @param picks         - heroes picked by this team
 * @param enemyPicks    - heroes picked by the opponent
 * @param data          - precomputed draft data (hero stats, synergies, counters, player stats)
 * @param playerAssignments - optional map of pick index → battletag (index into `picks` array)
 */
export function computeTeamWinEstimate(
  picks: string[],
  enemyPicks: string[],
  data: DraftData,
  map: string | null,
  playerAssignments?: Record<number, string>,
): WinEstimateResult {
  if (picks.length === 0) {
    return { winPct: 50, breakdown: { heroWR: 0, synergies: 0, counters: 0, playerAdj: 0, compWR: 0 } }
  }

  let heroWRDelta = 0
  let synergyDelta = 0
  let counterDelta = 0
  let playerAdj = 0

  // Track which pick indices have player overrides so we can skip their hero base WR
  const playerOverrideIndices = new Set<number>()

  // 4. Player strength — compute first so we know which indices to skip for hero WR
  if (playerAssignments) {
    for (let i = 0; i < picks.length; i++) {
      const bt = playerAssignments[i]
      if (!bt) continue
      const stats = data.playerStats[bt]?.[picks[i]]
      if (!stats || stats.games < 10) continue

      const adjMawp = stats.mawp != null
        ? confidenceAdjustedMawp(stats.mawp, stats.games, 30)
        : (stats.wins / stats.games) * 100

      const mawpDelta = adjMawp - 50
      const resolved = getHeroWinRate(picks[i], data, map)
      const heroBaseDelta = resolved ? resolved.winRate - 50 : 0

      // Player adjustment is the difference between MAWP and the hero base WR
      // that would otherwise be counted
      playerAdj += mawpDelta - heroBaseDelta
      playerOverrideIndices.add(i)
    }
  }

  // 1. Hero WR — prefer map-specific, fall back to overall
  for (const hero of picks) {
    const resolved = getHeroWinRate(hero, data, map)
    if (!resolved) continue
    heroWRDelta += resolved.winRate - 50
  }

  // 2. Intra-team synergies — average across all pairs
  // Normalized: subtract expected pair WR based on both heroes' base rates
  // to isolate the synergy-specific effect beyond individual hero strength
  {
    let sum = 0
    let count = 0
    for (let i = 0; i < picks.length; i++) {
      for (let j = i + 1; j < picks.length; j++) {
        const d = data.synergies[picks[i]]?.[picks[j]]
        if (d && d.games >= 30) {
          const wr1 = getHeroWinRate(picks[i], data, map)?.winRate ?? 50
          const wr2 = getHeroWinRate(picks[j], data, map)?.winRate ?? 50
          const expectedWR = 50 + (wr1 - 50) + (wr2 - 50)
          sum += d.winRate - expectedWR
          count++
        }
      }
    }
    if (count > 0) synergyDelta = sum / count
  }

  // 3. Cross-team counters — average across all matchups
  // Normalized: subtract expected matchup WR based on both heroes' base rates
  // to isolate the matchup-specific effect beyond individual hero strength
  {
    let sum = 0
    let count = 0
    for (const ourHero of picks) {
      const ourWR = getHeroWinRate(ourHero, data, map)?.winRate ?? 50
      for (const enemyHero of enemyPicks) {
        const d = data.counters[ourHero]?.[enemyHero]
        if (d && d.games >= 30) {
          const enemyWR = getHeroWinRate(enemyHero, data, map)?.winRate ?? 50
          const expectedWR = ourWR + (100 - enemyWR) - 50
          sum += d.winRate - expectedWR
          count++
        }
      }
    }
    if (count > 0) counterDelta = sum / count
  }

  // 5. Composition WR — data-driven boost/penalty based on team roles
  const compDelta = scoreCurrentComposition(picks, data.compositions, data.baselineCompWR)

  const totalDelta = heroWRDelta + synergyDelta + counterDelta + playerAdj + compDelta
  const raw = 50 + totalDelta
  const winPct = Math.round(Math.max(1, Math.min(99, raw)) * 10) / 10

  return {
    winPct,
    breakdown: {
      heroWR: Math.round(heroWRDelta * 10) / 10,
      synergies: Math.round(synergyDelta * 10) / 10,
      counters: Math.round(counterDelta * 10) / 10,
      playerAdj: Math.round(playerAdj * 10) / 10,
      compWR: compDelta,
    },
  }
}
