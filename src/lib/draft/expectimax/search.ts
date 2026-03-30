/**
 * Core expectimax tree search with transposition table.
 *
 * MAX nodes (our turn): pick the candidate that maximizes expected value.
 * CHANCE nodes (opponent turn): expected value over GD probability distribution.
 * Leaf nodes: full-state evaluation using computeTeamWinEstimate.
 *
 * Supports iterative deepening with time budget and progress callbacks.
 */

import type { DraftData } from '../types'
import type {
  SearchState,
  ExpectimaxConfig,
  ExpectimaxResult,
  OpponentPredictor,
} from './types'
import { DEFAULT_CONFIG } from './types'
import {
  cloneAndApply,
  hashState,
  isTerminal,
  isOurTurn,
  isBanPhase,
} from './search-state'
import { evaluateLeaf } from './leaf-eval'
import { prefilterPicks, prefilterBans } from './prefilter'

/**
 * Run expectimax search at a fixed depth from the given root state.
 *
 * Returns scored results for each root candidate (the heroes we could pick/ban).
 */
export async function expectimaxAtDepth(
  root: SearchState,
  data: DraftData,
  depth: number,
  config: ExpectimaxConfig,
  opponentPredict: OpponentPredictor,
): Promise<{ results: ExpectimaxResult[]; totalNodes: number }> {
  const transpositionTable = new Map<string, number>()
  let totalNodes = 0

  // Get root candidates
  const rootIsOurs = isOurTurn(root)
  const rootIsBan = isBanPhase(root)
  let candidates: string[]

  if (rootIsBan) {
    candidates = prefilterBans(root, data, rootIsOurs ? config.ourBanWidth : config.oppBanWidth)
  } else {
    candidates = prefilterPicks(root, data, rootIsOurs ? config.ourPickWidth : config.oppPickWidth)
  }

  // Evaluate each root candidate
  const results: ExpectimaxResult[] = []

  for (const hero of candidates) {
    const childState = cloneAndApply(root, hero)
    const value = await expectimax(
      childState, depth - 1, data, config, opponentPredict, transpositionTable,
    )
    totalNodes += 1
    results.push({
      hero,
      score: Math.round(value * 100) / 100,
      depth,
      nodesVisited: 0, // filled in later
    })
  }

  // Sort by score descending
  results.sort((a, b) => b.score - a.score)

  // Distribute total node count
  for (const r of results) r.nodesVisited = Math.round(transpositionTable.size / results.length)

  return { results, totalNodes: transpositionTable.size }
}

/**
 * Internal recursive expectimax function.
 */
async function expectimax(
  state: SearchState,
  depth: number,
  data: DraftData,
  config: ExpectimaxConfig,
  opponentPredict: OpponentPredictor,
  tt: Map<string, number>,
): Promise<number> {
  // Terminal or depth exhausted → leaf evaluation
  if (isTerminal(state) || depth <= 0) {
    return evaluateLeaf(state, data)
  }

  // Transposition table lookup
  // Include depth in key to avoid using shallow evaluations for deep lookups
  const key = hashState(state) + ':' + depth
  const cached = tt.get(key)
  if (cached !== undefined) return cached

  const ours = isOurTurn(state)
  const ban = isBanPhase(state)
  let value: number

  if (ours) {
    // MAX node — pick the hero that maximizes value
    const width = ban ? config.ourBanWidth : config.ourPickWidth
    const candidates = ban
      ? prefilterBans(state, data, width)
      : prefilterPicks(state, data, width)

    value = -Infinity
    for (const hero of candidates) {
      const child = cloneAndApply(state, hero)
      const v = await expectimax(child, depth - 1, data, config, opponentPredict, tt)
      if (v > value) value = v
    }
    if (value === -Infinity) value = evaluateLeaf(state, data)
  } else {
    // CHANCE node — expected value over opponent's likely actions
    const width = ban ? config.oppBanWidth : config.oppPickWidth

    let predictions: { hero: string; probability: number }[]
    try {
      predictions = await opponentPredict(state, width)
    } catch {
      // Fallback: use greedy prefilter with uniform distribution
      const fallbackCandidates = ban
        ? prefilterBans(state, data, width)
        : prefilterPicks(state, data, width)
      predictions = fallbackCandidates.map(h => ({
        hero: h,
        probability: 1 / fallbackCandidates.length,
      }))
    }

    // Filter to valid heroes and renormalize
    const validPredictions = predictions.filter(p => !state.taken.has(p.hero))
    const totalProb = validPredictions.reduce((s, p) => s + p.probability, 0)

    if (totalProb === 0 || validPredictions.length === 0) {
      value = evaluateLeaf(state, data)
    } else {
      value = 0
      for (const { hero, probability } of validPredictions) {
        const child = cloneAndApply(state, hero)
        const v = await expectimax(child, depth - 1, data, config, opponentPredict, tt)
        value += (probability / totalProb) * v
      }
    }
  }

  tt.set(key, value)
  return value
}

/**
 * Iterative deepening search.
 *
 * Runs expectimax at increasing depths (4, 6, 8, ...) up to config.maxDepth.
 * Calls onDepthComplete after each depth finishes.
 * Respects config.timeBudgetMs — aborts if time exceeded.
 * For draft steps near the end (step ≥ 12), searches to completion automatically.
 */
export async function iterativeDeepeningSearch(
  root: SearchState,
  data: DraftData,
  config: Partial<ExpectimaxConfig> = {},
  opponentPredict: OpponentPredictor,
  onDepthComplete?: (results: ExpectimaxResult[], depth: number) => void,
): Promise<ExpectimaxResult[]> {
  const cfg = { ...DEFAULT_CONFIG, ...config }
  const startTime = Date.now()

  // For late-draft positions, search to completion
  const stepsRemaining = 16 - root.step
  const effectiveMaxDepth = Math.min(cfg.maxDepth, stepsRemaining)

  // Iterative deepening: 4, 6, 8, ... up to maxDepth
  const depths: number[] = []
  for (let d = 4; d <= effectiveMaxDepth; d += 2) {
    depths.push(d)
  }
  // Ensure we at least try depth 2 for very shallow searches
  if (depths.length === 0) {
    depths.push(Math.min(2, effectiveMaxDepth))
  }
  // Always include the effective max if not already there
  if (depths[depths.length - 1] !== effectiveMaxDepth && effectiveMaxDepth > 0) {
    depths.push(effectiveMaxDepth)
  }

  let bestResults: ExpectimaxResult[] = []

  for (const depth of depths) {
    // Check time budget
    if (Date.now() - startTime > cfg.timeBudgetMs && bestResults.length > 0) {
      break
    }

    const { results } = await expectimaxAtDepth(
      root, data, depth, cfg, opponentPredict,
    )

    bestResults = results
    onDepthComplete?.(results, depth)
  }

  return bestResults
}
