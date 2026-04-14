/**
 * Types for the expectimax tree search.
 */

import type { SkillTier } from '@/lib/types'

export type Team = 'A' | 'B'

export interface ExpectimaxConfig {
  /** Width for our pick steps (top-K candidates from greedy prefilter) */
  ourPickWidth: number
  /** Width for our ban steps */
  ourBanWidth: number
  /** Width for opponent pick steps (top-N from GD distribution) */
  oppPickWidth: number
  /** Width for opponent ban steps */
  oppBanWidth: number
  /** Maximum search depth in plies */
  maxDepth: number
  /** Time budget in milliseconds (iterative deepening aborts if exceeded) */
  timeBudgetMs: number
}

export const DEFAULT_CONFIG: ExpectimaxConfig = {
  ourPickWidth: 8,
  ourBanWidth: 4,
  oppPickWidth: 6,
  oppBanWidth: 3,
  maxDepth: 8,
  timeBudgetMs: 3000,
}

export interface ExpectimaxResult {
  /** Hero name */
  hero: string
  /** Expected win delta from 50% baseline (positive = good for us) */
  score: number
  /** Search depth that produced this result */
  depth: number
  /** Total nodes visited for this candidate */
  nodesVisited: number
}

/**
 * Injected dependency: returns top-N opponent predictions with probabilities.
 * Used at CHANCE nodes to model opponent behavior.
 * The returned array should be sorted by probability (descending).
 */
export type OpponentPredictor = (
  state: SearchState,
  topN: number,
) => Promise<{ hero: string; probability: number }[]>

/** Compact draft state for tree search — optimized for clone + hash */
export interface SearchState {
  /** Our team's picks in draft order */
  ourPicks: string[]
  /** stepIndex (into DRAFT_SEQUENCE) of each ourPicks entry — parallel array */
  ourPickSteps: number[]
  /** Enemy team's picks in draft order */
  enemyPicks: string[]
  /** All bans (both teams) */
  bans: string[]
  /** All unavailable heroes (picks + bans + Cho/Gall expansion) */
  taken: Set<string>
  /** Current draft step index (0-15 into DRAFT_SEQUENCE) */
  step: number
  /** Map name */
  map: string
  /** Skill tier */
  tier: SkillTier
  /** Which team we are ('A' bans first, 'B' bans second) */
  ourTeam: Team
  /** Root-level player assignments (stepIndex → battletag) for already-completed picks. Immutable. */
  playerAssignments?: Record<number, string>
  /** All non-null battletags in our team's player slots. Immutable. */
  playerSlots?: string[]
  /** Battletags already locked into past picks (derived from playerAssignments). Immutable. */
  usedBattletags?: Set<string>
}
