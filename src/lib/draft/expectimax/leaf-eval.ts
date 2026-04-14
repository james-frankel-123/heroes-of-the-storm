/**
 * Leaf evaluation for expectimax search.
 *
 * Evaluates a complete or partial hypothetical draft state using the
 * existing computeTeamWinEstimate function, which includes all 5 scoring
 * factors: hero WR, counters, synergies, player strength, composition WR.
 */

import type { DraftData } from '../types'
import { computeTeamWinEstimate } from '../win-estimate'
import type { SearchState } from './types'

/**
 * Evaluate a search leaf state.
 *
 * Returns the expected win delta from 50% baseline for our team.
 * Positive = good for us, negative = bad.
 *
 * If the search state carries root-level player assignments, maps each of our
 * picks to its acting battletag via the parallel ourPickSteps array so
 * computeTeamWinEstimate's Stats-mode playerAdj term applies here too.
 */
export function evaluateLeaf(state: SearchState, data: DraftData): number {
  if (state.ourPicks.length === 0) return 0

  let pickAssignments: Record<number, string> | undefined
  if (state.playerAssignments) {
    const map: Record<number, string> = {}
    for (let i = 0; i < state.ourPicks.length; i++) {
      const stepIdx = state.ourPickSteps[i]
      const bt = stepIdx !== undefined ? state.playerAssignments[stepIdx] : undefined
      if (bt) map[i] = bt
    }
    if (Object.keys(map).length > 0) pickAssignments = map
  }

  const result = computeTeamWinEstimate(
    state.ourPicks,
    state.enemyPicks,
    data,
    state.map || null,
    pickAssignments,
  )

  return result.winPct - 50
}
