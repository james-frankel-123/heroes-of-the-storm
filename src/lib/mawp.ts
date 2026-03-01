/**
 * Momentum-Adjusted Win Percentage (MAWP)
 *
 * Pure implementation of the MAWP formula from HOTS_FEVER_SPEC.md.
 * Duplicated from sync/compute-derived.ts so we can:
 *   1. Unit-test the formula without DB dependencies
 *   2. Recompute client-side if needed (e.g. from match history)
 *
 * The formula estimates a player's current win probability for a hero,
 * combining three mechanisms:
 *
 * 1. Game-count weighting — recent games matter more:
 *    w_games(i) = 1.0                         if rank <= 30
 *    w_games(i) = exp(-lambda_g * (rank - 30))  otherwise
 *    lambda_g = ln(2) / 30   (half-life of 30 additional games)
 *
 * 2. Time-decay blending — old game outcomes blend toward 50%:
 *    w_time(i) = 1.0                          if days <= 180
 *    w_time(i) = exp(-lambda_t * (days - 180))  otherwise
 *    lambda_t = ln(2) / 90   (half-life of 90 days past the cliff)
 *
 *    effectiveOutcome(i) = outcome(i) * w_time(i) + 0.5 * (1 - w_time(i))
 *
 *    This ensures old games contribute ~50% rather than vanishing entirely.
 *
 * 3. Bayesian padding — low game counts shrink toward 50%:
 *    If games < 30, pad with (30 - games) phantom 50% observations at
 *    full weight. This prevents extreme MAWP with few games.
 *
 * Final calculation:
 *   MAWP = (SUM(w_games(i) * effectiveOutcome(i)) + phantomPadding)
 *        / (SUM(w_games(i)) + phantomCount)
 *
 * Returns a value in [0, 1]. Multiply by 100 for percentage display.
 * Only considers storm league games. Not split by skill level groupings.
 */

export const LAMBDA_G = Math.LN2 / 30
export const LAMBDA_T = Math.LN2 / 90

/** Confidence threshold: pad up to this many games */
export const CONFIDENCE_THRESHOLD = 30

export interface MatchInput {
  win: boolean
  gameDate: Date
}

/**
 * Compute the game-count weight for a match at the given rank.
 * Rank is 1-based (1 = most recent game).
 */
export function gameCountWeight(rank: number): number {
  if (rank <= 30) return 1.0
  return Math.exp(-LAMBDA_G * (rank - 30))
}

/**
 * Compute the time-decay factor for a match played `daysDiff` days ago.
 * Used to blend the outcome toward 50%, not as a weight multiplier.
 */
export function timeWeight(daysDiff: number): number {
  if (daysDiff <= 180) return 1.0
  return Math.exp(-LAMBDA_T * (daysDiff - 180))
}

/**
 * Compute MAWP from a list of matches.
 *
 * @param matches  Array of { win, gameDate } records
 * @param now      Reference date for time decay (defaults to Date.now())
 * @returns MAWP as a fraction in [0, 1]. Returns 0.5 for empty input.
 */
export function computeMAWP(matches: MatchInput[], now?: Date): number {
  if (matches.length === 0) return 0.5

  const refTime = (now ?? new Date()).getTime()

  // Sort newest first
  const sorted = [...matches].sort(
    (a, b) => b.gameDate.getTime() - a.gameDate.getTime()
  )

  let weightedSum = 0
  let weightSum = 0

  for (let i = 0; i < sorted.length; i++) {
    const rank = i + 1 // 1-based
    const match = sorted[i]

    const wGames = gameCountWeight(rank)

    const daysDiff =
      (refTime - match.gameDate.getTime()) / (1000 * 60 * 60 * 24)
    const wTime = timeWeight(daysDiff)

    // Blend outcome toward 50% based on time decay
    const outcome = match.win ? 1 : 0
    const effectiveOutcome = outcome * wTime + 0.5 * (1 - wTime)

    weightedSum += wGames * effectiveOutcome
    weightSum += wGames
  }

  // Bayesian padding: add phantom 50% games to reach confidence threshold
  if (sorted.length < CONFIDENCE_THRESHOLD) {
    const phantomCount = CONFIDENCE_THRESHOLD - sorted.length
    weightedSum += phantomCount * 0.5
    weightSum += phantomCount
  }

  return weightSum > 0 ? weightedSum / weightSum : 0.5
}

/**
 * Compute MAWP and return as a percentage (0-100).
 */
export function computeMAWPPercent(matches: MatchInput[], now?: Date): number {
  return computeMAWP(matches, now) * 100
}
