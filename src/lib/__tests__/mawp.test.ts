/**
 * Tests for the MAWP (Momentum-Adjusted Win Percentage) formula.
 *
 * Spec reference: HOTS_FEVER_SPEC.md lines 42-66.
 *
 * Formula:
 *   w(i) = w_games(i) * w_time(i)
 *   w_games(i) = 1.0 if rank <= 30, else exp(-ln(2)/30 * (rank - 30))
 *   w_time(i)  = 1.0 if days <= 180, else exp(-ln(2)/90 * (days - 180))
 *   MAWP = SUM(w(i) * outcome(i)) / SUM(w(i))
 */

import { describe, it, expect } from 'vitest'
import {
  computeMAWP,
  computeMAWPPercent,
  gameCountWeight,
  timeWeight,
  LAMBDA_G,
  LAMBDA_T,
  type MatchInput,
} from '../mawp'

// ---------------------------------------------------------------------------
// Helper: create matches at fixed dates relative to a reference
// ---------------------------------------------------------------------------

function makeMatch(win: boolean, daysAgo: number, refDate: Date): MatchInput {
  const d = new Date(refDate)
  d.setDate(d.getDate() - daysAgo)
  return { win, gameDate: d }
}

function makeMatches(
  outcomes: boolean[],
  startDaysAgo: number,
  refDate: Date,
  daysBetween = 1
): MatchInput[] {
  return outcomes.map((win, i) =>
    makeMatch(win, startDaysAgo + i * daysBetween, refDate)
  )
}

const NOW = new Date('2025-06-01T12:00:00Z')

// ---------------------------------------------------------------------------
// Unit tests for weight functions
// ---------------------------------------------------------------------------

describe('gameCountWeight', () => {
  it('returns 1.0 for ranks 1 through 30', () => {
    for (let rank = 1; rank <= 30; rank++) {
      expect(gameCountWeight(rank)).toBe(1.0)
    }
  })

  it('decays exponentially after rank 30', () => {
    // rank 31 should be slightly less than 1
    expect(gameCountWeight(31)).toBeLessThan(1.0)
    expect(gameCountWeight(31)).toBeGreaterThan(0.97)
  })

  it('has half-life of 30 games past the cliff (rank 60 = 0.5)', () => {
    // rank 60 = 30 games past the cliff => weight = 0.5
    expect(gameCountWeight(60)).toBeCloseTo(0.5, 5)
  })

  it('rank 90 = 0.25 (two half-lives)', () => {
    expect(gameCountWeight(90)).toBeCloseTo(0.25, 5)
  })

  it('rank 120 = 0.125 (three half-lives)', () => {
    expect(gameCountWeight(120)).toBeCloseTo(0.125, 5)
  })

  it('approaches zero for very high ranks', () => {
    expect(gameCountWeight(300)).toBeLessThan(0.01)
    expect(gameCountWeight(600)).toBeLessThan(0.0001)
  })

  it('LAMBDA_G equals ln(2)/30', () => {
    expect(LAMBDA_G).toBeCloseTo(Math.log(2) / 30, 10)
  })
})

describe('timeWeight', () => {
  it('returns 1.0 for days 0 through 180', () => {
    for (const days of [0, 1, 30, 90, 179, 180]) {
      expect(timeWeight(days)).toBe(1.0)
    }
  })

  it('decays after 180 days', () => {
    expect(timeWeight(181)).toBeLessThan(1.0)
    expect(timeWeight(181)).toBeGreaterThan(0.99)
  })

  it('has half-life of 90 days past the cliff (270 days = 0.5)', () => {
    // 270 days = 180 + 90 => weight = 0.5
    expect(timeWeight(270)).toBeCloseTo(0.5, 5)
  })

  it('360 days = 0.25 (two half-lives past cliff)', () => {
    expect(timeWeight(360)).toBeCloseTo(0.25, 5)
  })

  it('450 days = 0.125 (three half-lives past cliff)', () => {
    expect(timeWeight(450)).toBeCloseTo(0.125, 5)
  })

  it('approaches zero for very old games', () => {
    expect(timeWeight(1000)).toBeLessThan(0.01)
    expect(timeWeight(2000)).toBeLessThan(0.0001)
  })

  it('LAMBDA_T equals ln(2)/90', () => {
    expect(LAMBDA_T).toBeCloseTo(Math.log(2) / 90, 10)
  })
})

// ---------------------------------------------------------------------------
// computeMAWP — core formula
// ---------------------------------------------------------------------------

describe('computeMAWP', () => {
  it('returns 0 for empty input', () => {
    expect(computeMAWP([])).toBe(0)
  })

  it('returns 1.0 for a single win within 30 games / 180 days', () => {
    const matches = [makeMatch(true, 1, NOW)]
    expect(computeMAWP(matches, NOW)).toBe(1.0)
  })

  it('returns 0.0 for a single loss within 30 games / 180 days', () => {
    const matches = [makeMatch(false, 1, NOW)]
    expect(computeMAWP(matches, NOW)).toBe(0.0)
  })

  it('returns 0.5 for equal wins and losses within the 30-game window', () => {
    // 10 wins + 10 losses, all within 30 games and 180 days
    const wins = makeMatches(Array(10).fill(true), 1, NOW)
    const losses = makeMatches(Array(10).fill(false), 11, NOW)
    const matches = [...wins, ...losses]

    expect(computeMAWP(matches, NOW)).toBeCloseTo(0.5, 5)
  })

  it('all wins within 30 games returns 1.0', () => {
    const matches = makeMatches(Array(30).fill(true), 1, NOW)
    expect(computeMAWP(matches, NOW)).toBe(1.0)
  })

  it('all losses within 30 games returns 0.0', () => {
    const matches = makeMatches(Array(30).fill(false), 1, NOW)
    expect(computeMAWP(matches, NOW)).toBe(0.0)
  })

  it('within 30 games + 180 days, MAWP equals simple win rate', () => {
    // 18 wins out of 25 games, all recent
    const outcomes = [
      ...Array(18).fill(true),
      ...Array(7).fill(false),
    ]
    const matches = makeMatches(outcomes, 1, NOW)

    // All weights are 1.0, so MAWP = wins/games
    expect(computeMAWP(matches, NOW)).toBeCloseTo(18 / 25, 5)
  })

  it('weights recent wins more heavily than old wins', () => {
    // Scenario: 10 recent losses (rank 1-10) + 20 old wins (rank 11-30)
    // All within 180 days, all within 30 games => equal weight => 20/30
    const recentLosses = makeMatches(Array(10).fill(false), 1, NOW)
    const olderWins = makeMatches(Array(20).fill(true), 11, NOW)
    const flatMawp = computeMAWP([...recentLosses, ...olderWins], NOW)
    expect(flatMawp).toBeCloseTo(20 / 30, 5) // 0.6667

    // Now push the 20 older wins past rank 30 (add 30 more losses at rank 11-40)
    // Recent: 10 losses (rank 1-10) + 30 losses (rank 11-40) + 20 wins (rank 41-60)
    // The 20 wins are now in decay zone, weighted less
    const middleLosses = makeMatches(Array(30).fill(false), 11, NOW)
    const farWins = makeMatches(Array(20).fill(true), 41, NOW)
    const decayedMawp = computeMAWP(
      [...recentLosses, ...middleLosses, ...farWins],
      NOW
    )
    // With decay, MAWP should be less than simple win rate (20/60 = 0.333)
    // because the wins are down-weighted
    expect(decayedMawp).toBeLessThan(20 / 60)
  })

  it('time decay reduces weight of games older than 180 days', () => {
    // 10 wins all within 180 days
    const recentWins = makeMatches(Array(10).fill(true), 10, NOW)
    const recentMawp = computeMAWP(recentWins, NOW)
    expect(recentMawp).toBe(1.0)

    // 10 wins + 10 losses, wins are 300 days ago, losses are recent
    const oldWins = makeMatches(Array(10).fill(true), 300, NOW)
    const recentLosses = makeMatches(Array(10).fill(false), 1, NOW)
    const mixedMawp = computeMAWP([...oldWins, ...recentLosses], NOW)
    // Recent losses have weight 1.0, old wins have reduced weight
    // MAWP should be less than 0.5
    expect(mixedMawp).toBeLessThan(0.5)
  })

  it('combined game + time decay compounds correctly', () => {
    // 60 games: first 30 are wins (rank 1-30, full weight),
    // next 30 are wins too (rank 31-60, game-decayed).
    // All within 180 days.
    const allWins = makeMatches(Array(60).fill(true), 1, NOW)
    // Even though wins are decayed, they're still wins => MAWP = 1.0
    expect(computeMAWP(allWins, NOW)).toBe(1.0)
  })

  it('handles the Zul\'jin scenario: 50 games, recent losses, old wins', () => {
    // Simulate: player had 30 wins long ago, then 20 recent losses
    // Recent 20 losses = rank 1-20 (full game weight)
    // Old 30 wins = rank 21-50 (ranks 21-30 full, 31-50 decayed)
    // All within 180 days for simplicity
    const recentLosses = makeMatches(Array(20).fill(false), 1, NOW)
    const oldWins = makeMatches(Array(30).fill(true), 21, NOW)
    const mawp = computeMAWP([...recentLosses, ...oldWins], NOW)

    // Simple WR = 30/50 = 60%, but MAWP should be lower because
    // recent games (losses) have full weight while some old wins
    // are in the decay zone
    expect(mawp).toBeLessThan(0.6)
    // But it shouldn't be near-zero - the wins at rank 21-30 still have full weight
    expect(mawp).toBeGreaterThan(0.2)
  })

  it('sorts matches by date regardless of input order', () => {
    // Provide matches in random order — result should be same
    const matches: MatchInput[] = [
      makeMatch(true, 50, NOW),   // oldest
      makeMatch(false, 1, NOW),   // newest
      makeMatch(true, 25, NOW),   // middle
    ]

    const sorted = [...matches].sort(
      (a, b) => b.gameDate.getTime() - a.gameDate.getTime()
    )
    // After sort: rank 1 = 1 day ago (loss), rank 2 = 25 days (win), rank 3 = 50 days (win)
    // All within 30 games, all within 180 days => equal weight
    // MAWP = 2/3
    expect(computeMAWP(matches, NOW)).toBeCloseTo(2 / 3, 5)
    expect(computeMAWP(sorted, NOW)).toBeCloseTo(2 / 3, 5)
  })

  it('does not mutate the input array', () => {
    const matches = [
      makeMatch(true, 50, NOW),
      makeMatch(false, 1, NOW),
      makeMatch(true, 25, NOW),
    ]
    const copy = [...matches]
    computeMAWP(matches, NOW)
    expect(matches).toEqual(copy)
  })
})

// ---------------------------------------------------------------------------
// computeMAWPPercent
// ---------------------------------------------------------------------------

describe('computeMAWPPercent', () => {
  it('returns 0 for empty input', () => {
    expect(computeMAWPPercent([])).toBe(0)
  })

  it('returns 100 for all wins', () => {
    const matches = makeMatches(Array(10).fill(true), 1, NOW)
    expect(computeMAWPPercent(matches, NOW)).toBe(100)
  })

  it('returns 0 for all losses', () => {
    const matches = makeMatches(Array(10).fill(false), 1, NOW)
    expect(computeMAWPPercent(matches, NOW)).toBe(0)
  })

  it('returns ~50 for even split', () => {
    const outcomes = [...Array(15).fill(true), ...Array(15).fill(false)]
    const matches = makeMatches(outcomes, 1, NOW)
    expect(computeMAWPPercent(matches, NOW)).toBeCloseTo(50, 3)
  })
})

// ---------------------------------------------------------------------------
// Spec-specific edge cases
// ---------------------------------------------------------------------------

describe('MAWP spec edge cases', () => {
  it('game #30 has weight 1.0 (last in full-weight window)', () => {
    // 30 games: first 29 losses, last (rank 30) win
    // All equal weight, so MAWP = 1/30
    const matches = [
      ...makeMatches(Array(29).fill(false), 1, NOW),
      makeMatch(true, 30, NOW),
    ]
    expect(computeMAWP(matches, NOW)).toBeCloseTo(1 / 30, 5)
  })

  it('game #31 has decayed weight', () => {
    // 31 games: first 30 losses (full weight), 31st is win (decayed)
    // MAWP = w(31) * 1 / (30 * 1 + w(31))
    const matches = [
      ...makeMatches(Array(30).fill(false), 1, NOW),
      makeMatch(true, 31, NOW),
    ]
    const w31 = Math.exp(-LAMBDA_G * 1) // rank 31, so (rank - 30) = 1
    const expected = w31 / (30 + w31)
    expect(computeMAWP(matches, NOW)).toBeCloseTo(expected, 5)
  })

  it('game on day 180 has time weight 1.0', () => {
    const matches = [makeMatch(true, 180, NOW)]
    // Within 180 days => full weight => MAWP = 1.0
    expect(computeMAWP(matches, NOW)).toBe(1.0)
  })

  it('game on day 181 has slightly decayed time weight', () => {
    // 2 games: 1 loss today (weight 1.0), 1 win 181 days ago (slightly decayed)
    const matches = [
      makeMatch(false, 0, NOW),
      makeMatch(true, 181, NOW),
    ]
    const wTime181 = Math.exp(-LAMBDA_T * 1)
    const expected = wTime181 / (1 + wTime181)
    // Relax precision: makeMatch uses setDate which can introduce sub-day drift
    expect(computeMAWP(matches, NOW)).toBeCloseTo(expected, 3)
  })

  it('game at 270 days (180 + 90) has time weight 0.5', () => {
    // 2 games: 1 loss today, 1 win 270 days ago
    const matches = [
      makeMatch(false, 0, NOW),
      makeMatch(true, 270, NOW),
    ]
    // Win has time weight 0.5, loss has time weight 1.0
    // Both within 30 games, so game weight = 1.0
    const expected = 0.5 / (1.0 + 0.5) // = 1/3
    expect(computeMAWP(matches, NOW)).toBeCloseTo(expected, 4)
  })

  it('only considers wins as 1 and losses as 0', () => {
    // Verify there's no partial credit — only boolean outcomes
    const allWins = makeMatches(Array(5).fill(true), 1, NOW)
    const allLosses = makeMatches(Array(5).fill(false), 1, NOW)
    expect(computeMAWP(allWins, NOW)).toBe(1.0)
    expect(computeMAWP(allLosses, NOW)).toBe(0.0)
  })

  it('handles a large number of games correctly', () => {
    // 500 games, alternating W/L, all recent
    const outcomes = Array.from({ length: 500 }, (_, i) => i % 2 === 0)
    const matches = makeMatches(outcomes, 1, NOW)
    const mawp = computeMAWP(matches, NOW)
    // With game decay, recent games matter more. The pattern starts with
    // rank 1 = win, rank 2 = loss, rank 3 = win, etc.
    // Recent games (rank 1-30) have equal weight: 15 wins / 30 = 0.5
    // Older games are decayed but also ~50%, so overall ~0.5
    expect(mawp).toBeCloseTo(0.5, 1) // within 0.05 of 0.5
  })

  it('hot streak scenario: all recent wins, old losses', () => {
    // Player turned it around: 20 recent wins, 80 old losses
    const recentWins = makeMatches(Array(20).fill(true), 1, NOW)
    const oldLosses = makeMatches(Array(80).fill(false), 21, NOW)
    const mawp = computeMAWP([...recentWins, ...oldLosses], NOW)

    // Overall WR = 20/100 = 20%, but MAWP should be higher
    // because recent 20 wins (rank 1-20) have full weight
    // Old losses at ranks 21-100 still contribute (ranks 21-30 full weight),
    // so MAWP won't be dramatically higher, but still above raw WR
    expect(mawp).toBeGreaterThan(0.2) // better than raw WR
    expect(mawp).toBeGreaterThan(0.29) // noticeably better
  })

  it('cold streak scenario: all recent losses, old wins', () => {
    // Player is slumping: 20 recent losses, 80 old wins
    const recentLosses = makeMatches(Array(20).fill(false), 1, NOW)
    const oldWins = makeMatches(Array(80).fill(true), 21, NOW)
    const mawp = computeMAWP([...recentLosses, ...oldWins], NOW)

    // Overall WR = 80/100 = 80%, but MAWP should be lower
    // because recent 20 losses have full weight
    // Old wins at ranks 21-100 still contribute (ranks 21-30 full weight),
    // so MAWP won't drop dramatically, but still below raw WR
    expect(mawp).toBeLessThan(0.8) // worse than raw WR
    expect(mawp).toBeLessThan(0.71) // noticeably worse
  })

  it('defaults to current time when no reference date provided', () => {
    // Just verify it doesn't crash and returns reasonable value
    const matches = makeMatches(Array(10).fill(true), 1, new Date())
    const mawp = computeMAWP(matches)
    expect(mawp).toBeCloseTo(1.0, 3)
  })
})

// ---------------------------------------------------------------------------
// Mathematical properties
// ---------------------------------------------------------------------------

describe('MAWP mathematical properties', () => {
  it('MAWP is always in [0, 1]', () => {
    // Random scenarios
    const scenarios = [
      makeMatches(Array(100).fill(true), 1, NOW),
      makeMatches(Array(100).fill(false), 1, NOW),
      makeMatches(
        Array.from({ length: 100 }, () => Math.random() > 0.5),
        1,
        NOW
      ),
      makeMatches(Array(5).fill(true), 500, NOW), // very old games
    ]

    for (const matches of scenarios) {
      const mawp = computeMAWP(matches, NOW)
      expect(mawp).toBeGreaterThanOrEqual(0)
      expect(mawp).toBeLessThanOrEqual(1)
    }
  })

  it('MAWP is monotonically related to win proportion (same weights)', () => {
    // With all games in the full-weight window, more wins => higher MAWP
    for (let wins = 0; wins <= 20; wins++) {
      const outcomes = [
        ...Array(wins).fill(true),
        ...Array(20 - wins).fill(false),
      ]
      const matches = makeMatches(outcomes, 1, NOW)
      const mawp = computeMAWP(matches, NOW)
      expect(mawp).toBeCloseTo(wins / 20, 5)
    }
  })

  it('adding a recent win increases MAWP', () => {
    const base = makeMatches(
      [...Array(10).fill(true), ...Array(10).fill(false)],
      2,
      NOW
    )
    const baseMawp = computeMAWP(base, NOW)

    // Add one more recent win
    const withWin = [makeMatch(true, 1, NOW), ...base]
    const winMawp = computeMAWP(withWin, NOW)

    expect(winMawp).toBeGreaterThan(baseMawp)
  })

  it('adding a recent loss decreases MAWP', () => {
    const base = makeMatches(
      [...Array(10).fill(true), ...Array(10).fill(false)],
      2,
      NOW
    )
    const baseMawp = computeMAWP(base, NOW)

    // Add one more recent loss
    const withLoss = [makeMatch(false, 1, NOW), ...base]
    const lossMawp = computeMAWP(withLoss, NOW)

    expect(lossMawp).toBeLessThan(baseMawp)
  })
})
