/**
 * Prefilter candidates for expectimax search.
 *
 * Uses the greedy Draft Insights scoring to rank all valid heroes,
 * then returns the top-K. This bounds the search tree width while
 * ensuring we don't miss obviously good candidates.
 */

import type { DraftData } from '../types'
import { scoreHeroForPick, scoreHeroForBan } from '../engine'
import { getValidHeroes, isOurTurn } from './search-state'
import type { SearchState } from './types'

/**
 * Get top-K pick candidates for the current team, ranked by greedy score.
 */
export function prefilterPicks(
  state: SearchState,
  data: DraftData,
  width: number,
): string[] {
  const valid = getValidHeroes(state)
  const isOurs = isOurTurn(state)

  const ourPicks = isOurs ? state.ourPicks : state.enemyPicks
  const enemyPicks = isOurs ? state.enemyPicks : state.ourPicks

  const scored = valid.map(hero => ({
    hero,
    score: scoreHeroForPick(hero, ourPicks, enemyPicks, data, state.map || null),
  }))

  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, width).map(s => s.hero)
}

/**
 * Get top-K ban candidates, ranked by greedy ban score.
 */
export function prefilterBans(
  state: SearchState,
  data: DraftData,
  width: number,
): string[] {
  const valid = getValidHeroes(state)
  const isOurs = isOurTurn(state)

  // Bans protect our team (our ban) or enemy team (enemy ban)
  const picksToProtect = isOurs ? state.ourPicks : state.enemyPicks
  const opponentPicks = isOurs ? state.enemyPicks : state.ourPicks

  const scored = valid.map(hero => ({
    hero,
    score: scoreHeroForBan(hero, picksToProtect, opponentPicks, data, state.map || null),
  }))

  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, width).map(s => s.hero)
}
