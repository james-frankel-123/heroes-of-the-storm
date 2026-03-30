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
 * Uses computeTeamWinEstimate which naturally scores the COMPLETE team
 * composition — a hero that looks mediocre alone but completes a strong
 * comp gets credit here.
 *
 * No player assignments in speculative states (the search doesn't know
 * which player gets which hero). Player strength only affects the root
 * prefilter where actual assignments are known.
 */
export function evaluateLeaf(state: SearchState, data: DraftData): number {
  if (state.ourPicks.length === 0) return 0

  const result = computeTeamWinEstimate(
    state.ourPicks,
    state.enemyPicks,
    data,
    state.map || null,
    // No player assignments in search — pass undefined
  )

  // Return delta from 50% baseline
  return result.winPct - 50
}
