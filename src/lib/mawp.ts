/**
 * Momentum-Adjusted Win Percentage (MAWP)
 *
 * Pure implementation of the MAWP formula from HOTS_FEVER_SPEC.md.
 * Duplicated from sync/compute-derived.ts so we can:
 *   1. Unit-test the formula without DB dependencies
 *   2. Recompute client-side if needed (e.g. from match history)
 *
 * Weight for game i:  w(i) = w_games(i) * w_time(i)
 *
 * Game count factor — full weight for last 30 games, then exponential decay:
 *   w_games(i) = 1.0                         if rank <= 30
 *   w_games(i) = exp(-lambda_g * (rank - 30))  otherwise
 *   lambda_g = ln(2) / 30   (half-life of 30 additional games, so game #60 has weight 0.5)
 *
 * Time factor — full weight for last 180 days, then exponential decay:
 *   w_time(i) = 1.0                          if days <= 180
 *   w_time(i) = exp(-lambda_t * (days - 180))  otherwise
 *   lambda_t = ln(2) / 90   (half-life of 90 days past the cliff, so 9 months ago ~= 0.5)
 *
 * Final calculation:
 *   MAWP = SUM(w(i) * outcome(i)) / SUM(w(i))
 *   where outcome(i) = 1 for win, 0 for loss
 *
 * Returns a value in [0, 1]. Multiply by 100 for percentage display.
 * Only considers storm league games. Not split by skill level groupings (it's personal data).
 */

export const LAMBDA_G = Math.LN2 / 30
export const LAMBDA_T = Math.LN2 / 90

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
 * Compute the time-decay weight for a match played `daysDiff` days ago.
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
 * @returns MAWP as a fraction in [0, 1]. Returns 0 for empty input.
 */
export function computeMAWP(matches: MatchInput[], now?: Date): number {
  if (matches.length === 0) return 0

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

    const weight = wGames * wTime
    weightedSum += weight * (match.win ? 1 : 0)
    weightSum += weight
  }

  return weightSum > 0 ? weightedSum / weightSum : 0
}

/**
 * Compute MAWP and return as a percentage (0-100).
 */
export function computeMAWPPercent(matches: MatchInput[], now?: Date): number {
  return computeMAWP(matches, now) * 100
}
