/**
 * Running team win % estimate.
 *
 * Starting from a 50% baseline, accumulates data-backed deltas:
 *   1. Hero base WR:     sum of (heroWR - 50) for each picked hero
 *   2. Intra-team synergies: for each pair (A, B), (pairwiseWithWR - 50), counted once
 *   3. Cross-team counters:  for each (ourHero, enemyHero), (pairwiseVsWR - 50)
 *   4. Player strength:  if a player is assigned, (adjustedMAWP - 50) replaces hero base WR
 *
 * Result: 50 + totalDelta, clamped to [1, 99].
 */

import type { DraftData } from './types'
import { getHeroWinRate } from './engine'
import { confidenceAdjustedMawp } from '@/lib/utils'

export interface WinEstimateBreakdown {
  heroWR: number
  synergies: number
  counters: number
  playerAdj: number
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
  playerAssignments?: Record<number, string>,
): WinEstimateResult {
  if (picks.length === 0) {
    return { winPct: 50, breakdown: { heroWR: 0, synergies: 0, counters: 0, playerAdj: 0 } }
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
      const resolved = getHeroWinRate(picks[i], data)
      const heroBaseDelta = resolved ? resolved.winRate - 50 : 0

      // Player adjustment is the difference between MAWP and the hero base WR
      // that would otherwise be counted
      playerAdj += mawpDelta - heroBaseDelta
      playerOverrideIndices.add(i)
    }
  }

  // 1. Hero WR — prefer map-specific, fall back to overall
  for (const hero of picks) {
    const resolved = getHeroWinRate(hero, data)
    if (!resolved) continue
    heroWRDelta += resolved.winRate - 50
  }

  // 2. Intra-team synergies — each pair counted once
  for (let i = 0; i < picks.length; i++) {
    for (let j = i + 1; j < picks.length; j++) {
      const d = data.synergies[picks[i]]?.[picks[j]]
      if (d && d.games >= 30) {
        synergyDelta += d.winRate - 50
      }
    }
  }

  // 3. Cross-team counters
  for (const ourHero of picks) {
    for (const enemyHero of enemyPicks) {
      const d = data.counters[ourHero]?.[enemyHero]
      if (d && d.games >= 30) {
        counterDelta += d.winRate - 50
      }
    }
  }

  const totalDelta = heroWRDelta + synergyDelta + counterDelta + playerAdj
  const raw = 50 + totalDelta
  const winPct = Math.round(Math.max(1, Math.min(99, raw)) * 10) / 10

  return {
    winPct,
    breakdown: {
      heroWR: Math.round(heroWRDelta * 10) / 10,
      synergies: Math.round(synergyDelta * 10) / 10,
      counters: Math.round(counterDelta * 10) / 10,
      playerAdj: Math.round(playerAdj * 10) / 10,
    },
  }
}
