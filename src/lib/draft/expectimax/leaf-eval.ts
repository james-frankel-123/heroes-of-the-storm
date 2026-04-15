/**
 * Leaf evaluation for expectimax search.
 *
 * Evaluates a complete or partial hypothetical draft state using the
 * existing computeTeamWinEstimate function, which includes all 5 scoring
 * factors: hero WR, counters, synergies, player strength, composition WR.
 */

import type { DraftData } from '../types'
import { computeTeamWinEstimate } from '../win-estimate'
import { resolvePlayerHeroWRForLeafEval } from '../engine'
import type { SearchState } from './types'

/**
 * Evaluate a search leaf state.
 *
 * Returns the expected win delta from 50% baseline for our team.
 * Positive = good for us, negative = bad.
 *
 * Builds a per-pick battletag map from two sources:
 *   1. Locked assignments (playerAssignments[stepIndex]) for picks the user
 *      has already made before search started.
 *   2. Greedy best-fit attribution for speculative picks made inside the
 *      search tree — for each unassigned pick, find the available battletag
 *      with the strongest signal on this (hero, map) and consume it. This
 *      matches Stats mode's "who should play this" logic, so the +X player
 *      boost shown in the byline actually counts toward the search score.
 */
export function evaluateLeaf(state: SearchState, data: DraftData): number {
  if (state.ourPicks.length === 0) return 0

  const map: Record<number, string> = {}
  const consumed = new Set<string>()

  // 1. Locked assignments for picks that pre-existed the search.
  if (state.playerAssignments) {
    for (let i = 0; i < state.ourPicks.length; i++) {
      const stepIdx = state.ourPickSteps[i]
      const bt = stepIdx !== undefined ? state.playerAssignments[stepIdx] : undefined
      if (bt) {
        map[i] = bt
        consumed.add(bt)
      }
    }
  }

  // 2. Greedy best-fit for speculative picks. Iterate in pick order so earlier
  // picks claim their best player first.
  if (state.playerSlots && state.playerSlots.length > 0) {
    for (let i = 0; i < state.ourPicks.length; i++) {
      if (map[i]) continue
      const hero = state.ourPicks[i]
      let bestBt: string | null = null
      let bestVal = -Infinity
      for (const bt of state.playerSlots) {
        if (consumed.has(bt)) continue
        const r = resolvePlayerHeroWRForLeafEval(bt, hero, data, state.map || null)
        if (r && r.value > bestVal) {
          bestVal = r.value
          bestBt = bt
        }
      }
      if (bestBt) {
        map[i] = bestBt
        consumed.add(bestBt)
      }
    }
  }

  const pickAssignments = Object.keys(map).length > 0 ? map : undefined

  const result = computeTeamWinEstimate(
    state.ourPicks,
    state.enemyPicks,
    data,
    state.map || null,
    pickAssignments,
  )

  return result.winPct - 50
}
