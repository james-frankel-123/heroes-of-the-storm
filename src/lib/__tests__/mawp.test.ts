/**
 * Tests for the MAWP (Momentum-Adjusted Win Percentage) formula.
 *
 * Spec reference: HOTS_FEVER_SPEC.md §Momentum-Adjusted Win %
 *
 * Three mechanisms:
 * 1. Game-count weighting: full weight for last 30, then exp decay
 * 2. Time-decay blending: old game outcomes blend toward 50%
 * 3. Bayesian padding: pad with phantom 50% games up to 30
 */

import { describe, it, expect } from 'vitest'
import {
  computeMAWP,
  computeMAWPPercent,
  gameCountWeight,
  timeWeight,
  LAMBDA_G,
  LAMBDA_T,
  CONFIDENCE_THRESHOLD,
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
    expect(gameCountWeight(31)).toBeLessThan(1.0)
    expect(gameCountWeight(31)).toBeGreaterThan(0.97)
  })

  it('has half-life of 30 games past the cliff (rank 60 = 0.5)', () => {
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
  it('returns 0.5 for empty input (Bayesian prior)', () => {
    expect(computeMAWP([])).toBe(0.5)
  })

  it('CONFIDENCE_THRESHOLD is 30', () => {
    expect(CONFIDENCE_THRESHOLD).toBe(30)
  })

  // --- Bayesian padding tests ---

  it('single recent win is pulled toward 50% by padding', () => {
    // 1 game + 29 phantom 50% games
    // MAWP = (1 + 29*0.5) / 30 = 15.5/30 = 0.5167
    const matches = [makeMatch(true, 1, NOW)]
    expect(computeMAWP(matches, NOW)).toBeCloseTo(15.5 / 30, 4)
  })

  it('single recent loss is pulled toward 50% by padding', () => {
    // 1 loss + 29 phantom 50% games
    // MAWP = (0 + 29*0.5) / 30 = 14.5/30 = 0.4833
    const matches = [makeMatch(false, 1, NOW)]
    expect(computeMAWP(matches, NOW)).toBeCloseTo(14.5 / 30, 4)
  })

  it('8 recent games at 62.5% WR gives ~53% MAWP (E.T.C. scenario)', () => {
    // 5 wins + 3 losses, all within 180 days
    // 8 real games + 22 phantom = 30 total
    // MAWP = (5 + 22*0.5) / 30 = (5 + 11) / 30 = 16/30 ≈ 53.3%
    const matches = [
      ...makeMatches(Array(5).fill(true), 1, NOW),
      ...makeMatches(Array(3).fill(false), 6, NOW),
    ]
    const mawp = computeMAWP(matches, NOW)
    expect(mawp).toBeCloseTo(16 / 30, 3)
    expect(mawp * 100).toBeGreaterThan(50)
    expect(mawp * 100).toBeLessThan(55)
  })

  it('27 recent games at 59.3% WR gives ~58% MAWP (Auriel scenario)', () => {
    // 16 wins + 11 losses, all within 180 days
    // 27 real games + 3 phantom = 30 total
    // MAWP = (16 + 3*0.5) / 30 = 17.5/30 ≈ 58.3%
    const matches = [
      ...makeMatches(Array(16).fill(true), 1, NOW),
      ...makeMatches(Array(11).fill(false), 17, NOW),
    ]
    const mawp = computeMAWP(matches, NOW)
    expect(mawp).toBeCloseTo(17.5 / 30, 3)
    expect(mawp * 100).toBeGreaterThan(55)
    expect(mawp * 100).toBeLessThan(60)
  })

  it('30+ games get no padding', () => {
    // 30 games, all recent wins => MAWP = 30/30 = 1.0
    const matches = makeMatches(Array(30).fill(true), 1, NOW)
    expect(computeMAWP(matches, NOW)).toBe(1.0)
  })

  it('exactly 30 games: no padding applied', () => {
    // 15W/15L all recent => (15/30) = 0.5
    const matches = [
      ...makeMatches(Array(15).fill(true), 1, NOW),
      ...makeMatches(Array(15).fill(false), 16, NOW),
    ]
    expect(computeMAWP(matches, NOW)).toBeCloseTo(0.5, 5)
  })

  // --- Time-decay blending tests ---

  it('recent games within 180 days have full outcome (no blending)', () => {
    // 30 games all within 180 days, all wins => MAWP = 1.0
    const matches = makeMatches(Array(30).fill(true), 1, NOW)
    expect(computeMAWP(matches, NOW)).toBe(1.0)
  })

  it('old losses blend toward 50% instead of vanishing', () => {
    // 30 games: 15 recent wins + 15 old losses (400 days ago)
    // Old formula: old losses would nearly vanish, MAWP ≈ 100%
    // New formula: old losses contribute ~0.39 each, MAWP well below 100%
    const recentWins = makeMatches(Array(15).fill(true), 1, NOW)
    const oldLosses = makeMatches(Array(15).fill(false), 400, NOW, 1)
    const mawp = computeMAWP([...recentWins, ...oldLosses], NOW)
    // Recent wins contribute 1.0, old losses contribute ~0.39
    // MAWP = (15 + 15*0.39) / 30 ≈ 0.695
    expect(mawp).toBeGreaterThan(0.6)
    expect(mawp).toBeLessThan(0.8)
    // Critical: it must NOT be near 100% (which was the old bug)
    expect(mawp).toBeLessThan(0.85)
  })

  it('very old game outcome approaches 50%', () => {
    // A game from 3 years ago: wTime ≈ 0, effectiveOutcome ≈ 0.5
    // 1 very old win + 29 phantom at 50%
    const matches = [makeMatch(true, 1200, NOW)]
    const mawp = computeMAWP(matches, NOW)
    // effectiveOutcome ≈ 1.0 * ~0 + 0.5 * ~1 ≈ 0.5
    // (0.5 + 29*0.5) / 30 = 15/30 = 0.5
    expect(mawp).toBeCloseTo(0.5, 1)
  })

  it('recent win at 270 days still has signal (wTime ≈ 0.5)', () => {
    // 1 win at 270 days: effectiveOutcome = 1*0.5 + 0.5*0.5 = 0.75
    // + 29 phantom at 0.5
    // MAWP = (0.75 + 14.5) / 30 = 15.25 / 30 ≈ 0.508
    const matches = [makeMatch(true, 270, NOW)]
    const mawp = computeMAWP(matches, NOW)
    expect(mawp).toBeCloseTo(15.25 / 30, 2)
  })

  // --- Combined behavior tests ---

  it('E.T.C. with old losses: ~57% not 99% (the key bug fix)', () => {
    // 5 recent wins + 3 losses from 400 days ago + 22 phantom
    const recentWins = makeMatches(Array(5).fill(true), 1, NOW)
    const oldLosses = makeMatches(Array(3).fill(false), 400, NOW, 1)
    const mawp = computeMAWP([...recentWins, ...oldLosses], NOW)

    // Time weight at 400 days: exp(-ln2/90 * 220) ≈ 0.22
    // Old loss effective outcome: 0 * 0.22 + 0.5 * 0.78 = 0.39
    // MAWP = (5*1.0 + 3*0.39 + 22*0.5) / 30 = 17.17/30 ≈ 0.572
    expect(mawp * 100).toBeGreaterThan(53)
    expect(mawp * 100).toBeLessThan(62)
    // Not anywhere near 99%
    expect(mawp * 100).toBeLessThan(65)
  })

  it('hot streak: recent wins, old losses — higher than raw WR but reasonable', () => {
    // 20 recent wins, 80 old losses
    const recentWins = makeMatches(Array(20).fill(true), 1, NOW)
    const oldLosses = makeMatches(Array(80).fill(false), 21, NOW)
    const mawp = computeMAWP([...recentWins, ...oldLosses], NOW)

    // Raw WR = 20/100 = 20%, MAWP should be higher (recent wins matter more)
    // but old losses blend toward 50%, not vanish, keeping it reasonable
    expect(mawp).toBeGreaterThan(0.2)
    expect(mawp).toBeLessThan(0.55)
  })

  it('cold streak: recent losses, old wins — lower than raw WR but reasonable', () => {
    // 20 recent losses, 80 old wins
    const recentLosses = makeMatches(Array(20).fill(false), 1, NOW)
    const oldWins = makeMatches(Array(80).fill(true), 21, NOW)
    const mawp = computeMAWP([...recentLosses, ...oldWins], NOW)

    // Raw WR = 80/100 = 80%, MAWP should be lower
    expect(mawp).toBeLessThan(0.8)
    expect(mawp).toBeGreaterThan(0.45)
  })

  // --- Input handling ---

  it('sorts matches by date regardless of input order', () => {
    const matches: MatchInput[] = [
      makeMatch(true, 50, NOW),
      makeMatch(false, 1, NOW),
      makeMatch(true, 25, NOW),
    ]
    const sorted = [...matches].sort(
      (a, b) => b.gameDate.getTime() - a.gameDate.getTime()
    )
    expect(computeMAWP(matches, NOW)).toBeCloseTo(
      computeMAWP(sorted, NOW),
      5
    )
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

  it('defaults to current time when no reference date provided', () => {
    const matches = makeMatches(Array(30).fill(true), 1, new Date())
    const mawp = computeMAWP(matches)
    expect(mawp).toBeCloseTo(1.0, 2)
  })
})

// ---------------------------------------------------------------------------
// computeMAWPPercent
// ---------------------------------------------------------------------------

describe('computeMAWPPercent', () => {
  it('returns 50 for empty input', () => {
    expect(computeMAWPPercent([])).toBe(50)
  })

  it('returns 100 for 30+ all wins', () => {
    const matches = makeMatches(Array(30).fill(true), 1, NOW)
    expect(computeMAWPPercent(matches, NOW)).toBe(100)
  })

  it('returns 0 for 30+ all losses', () => {
    const matches = makeMatches(Array(30).fill(false), 1, NOW)
    expect(computeMAWPPercent(matches, NOW)).toBe(0)
  })

  it('returns ~50 for even split of 30+ games', () => {
    const outcomes = [...Array(15).fill(true), ...Array(15).fill(false)]
    const matches = makeMatches(outcomes, 1, NOW)
    expect(computeMAWPPercent(matches, NOW)).toBeCloseTo(50, 3)
  })
})

// ---------------------------------------------------------------------------
// Spec edge cases
// ---------------------------------------------------------------------------

describe('MAWP spec edge cases', () => {
  it('game #30 has weight 1.0 (last in full-weight window)', () => {
    // 30 games: 29 losses + 1 win (oldest, rank 30), all recent
    // All weight 1.0, + no padding
    // MAWP = 1/30
    const matches = [
      ...makeMatches(Array(29).fill(false), 1, NOW),
      makeMatch(true, 30, NOW),
    ]
    expect(computeMAWP(matches, NOW)).toBeCloseTo(1 / 30, 5)
  })

  it('game #31 has decayed game-count weight', () => {
    // 31 games: 30 losses + 1 win at rank 31 (decayed)
    const matches = [
      ...makeMatches(Array(30).fill(false), 1, NOW),
      makeMatch(true, 31, NOW),
    ]
    const w31 = Math.exp(-LAMBDA_G * 1)
    // effectiveOutcome still 1.0 (within 180 days)
    // MAWP = w31 / (30 + w31)
    const expected = w31 / (30 + w31)
    expect(computeMAWP(matches, NOW)).toBeCloseTo(expected, 5)
  })

  it('game on day 180 has full outcome (no blending)', () => {
    // 30 wins all on day 180 => MAWP = 1.0
    const matches = makeMatches(Array(30).fill(true), 180, NOW, 0)
    expect(computeMAWP(matches, NOW)).toBe(1.0)
  })

  it('handles a large number of games correctly', () => {
    // 500 games, alternating W/L, all recent
    const outcomes = Array.from({ length: 500 }, (_, i) => i % 2 === 0)
    const matches = makeMatches(outcomes, 1, NOW)
    const mawp = computeMAWP(matches, NOW)
    // ~50% overall, no padding (>30 games)
    expect(mawp).toBeCloseTo(0.5, 1)
  })
})

// ---------------------------------------------------------------------------
// Mathematical properties
// ---------------------------------------------------------------------------

describe('MAWP mathematical properties', () => {
  it('MAWP is always in [0, 1]', () => {
    const scenarios = [
      makeMatches(Array(100).fill(true), 1, NOW),
      makeMatches(Array(100).fill(false), 1, NOW),
      makeMatches(
        Array.from({ length: 100 }, () => Math.random() > 0.5),
        1,
        NOW
      ),
      makeMatches(Array(5).fill(true), 500, NOW),
      [],
    ]

    for (const matches of scenarios) {
      const mawp = computeMAWP(matches, NOW)
      expect(mawp).toBeGreaterThanOrEqual(0)
      expect(mawp).toBeLessThanOrEqual(1)
    }
  })

  it('with 30+ recent games, MAWP equals simple win rate', () => {
    // No padding, no time decay => MAWP = wins/games
    for (let wins = 0; wins <= 30; wins++) {
      const outcomes = [
        ...Array(wins).fill(true),
        ...Array(30 - wins).fill(false),
      ]
      const matches = makeMatches(outcomes, 1, NOW)
      const mawp = computeMAWP(matches, NOW)
      expect(mawp).toBeCloseTo(wins / 30, 5)
    }
  })

  it('adding a recent win increases MAWP', () => {
    const base = makeMatches(
      [...Array(10).fill(true), ...Array(10).fill(false)],
      2,
      NOW
    )
    const baseMawp = computeMAWP(base, NOW)

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

    const withLoss = [makeMatch(false, 1, NOW), ...base]
    const lossMawp = computeMAWP(withLoss, NOW)
    expect(lossMawp).toBeLessThan(baseMawp)
  })

  it('fewer games → closer to 50% (padding effect)', () => {
    // All wins, but fewer games = more padding toward 50%
    const mawp1 = computeMAWP(makeMatches([true], 1, NOW), NOW)
    const mawp5 = computeMAWP(makeMatches(Array(5).fill(true), 1, NOW), NOW)
    const mawp15 = computeMAWP(makeMatches(Array(15).fill(true), 1, NOW), NOW)
    const mawp30 = computeMAWP(makeMatches(Array(30).fill(true), 1, NOW), NOW)

    // More games = further from 50%
    expect(mawp1).toBeLessThan(mawp5)
    expect(mawp5).toBeLessThan(mawp15)
    expect(mawp15).toBeLessThan(mawp30)
    // And 30 games with all wins = 1.0
    expect(mawp30).toBe(1.0)
  })

  it('old games have less influence than recent games', () => {
    // Same 15W/15L, but wins are recent vs wins are old
    const recentWins = [
      ...makeMatches(Array(15).fill(true), 1, NOW),
      ...makeMatches(Array(15).fill(false), 400, NOW, 1),
    ]
    const recentLosses = [
      ...makeMatches(Array(15).fill(false), 1, NOW),
      ...makeMatches(Array(15).fill(true), 400, NOW, 1),
    ]

    const mawpRecentWins = computeMAWP(recentWins, NOW)
    const mawpRecentLosses = computeMAWP(recentLosses, NOW)

    // Recent wins → MAWP > 50%
    expect(mawpRecentWins).toBeGreaterThan(0.5)
    // Recent losses → MAWP < 50%
    expect(mawpRecentLosses).toBeLessThan(0.5)
  })
})
