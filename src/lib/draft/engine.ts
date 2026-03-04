/**
 * Draft recommendation engine.
 *
 * Scores each hero as a net win-rate delta from a 50% baseline.
 * Every factor is expressed in percentage points so the displayed
 * score reads as "picking this hero shifts our win probability by +X%".
 *
 * All factors contribute to the displayed netDelta:
 *   1. Hero base WR:    (heroWR - 50), preferring map-specific data
 *   2. Counter-picks:   sum of (pairwise vs enemy - 50) for each enemy
 *   3. Synergies:       sum of (pairwise with ally - 50) for each ally
 *   4. Player strength: best available battletag's (MAWP - 50) on this hero
 *   5. Composition WR:  data-driven boost/penalty based on achievable team compositions
 *                       from Heroes Profile. Scaled by picks made (0 at start → full at last pick).
 */

import {
  type DraftState,
  type DraftData,
  type DraftRecommendation,
  type RecommendationReason,
  DRAFT_SEQUENCE,
} from './types'
import {
  HERO_ROLES,
} from '@/lib/data/hero-roles'
import { confidenceAdjustedMawp } from '@/lib/utils'
import { scoreCompositionForHero } from './composition'

// ---------------------------------------------------------------------------
// Cho'gall pairing — they must be picked/banned together
// ---------------------------------------------------------------------------

/** If Cho or Gall is in the set, add the other */
export function expandChoGall(heroes: Set<string>): Set<string> {
  if (heroes.has('Cho') || heroes.has('Gall')) {
    heroes.add('Cho')
    heroes.add('Gall')
  }
  return heroes
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getUnavailableHeroes(state: DraftState): Set<string> {
  return expandChoGall(new Set(Object.values(state.selections)))
}

function getOurPicks(state: DraftState): string[] {
  const heroes: string[] = []
  for (let i = 0; i < state.currentStep; i++) {
    const step = DRAFT_SEQUENCE[i]
    if (step.type === 'pick' && step.team === state.ourTeam && state.selections[i]) {
      heroes.push(state.selections[i])
    }
  }
  return heroes
}

function getEnemyPicks(state: DraftState): string[] {
  const enemyTeam = state.ourTeam === 'A' ? 'B' : 'A'
  const heroes: string[] = []
  for (let i = 0; i < state.currentStep; i++) {
    const step = DRAFT_SEQUENCE[i]
    if (step.type === 'pick' && step.team === enemyTeam && state.selections[i]) {
      heroes.push(state.selections[i])
    }
  }
  return heroes
}

/** Get battletags that haven't been assigned to a pick yet */
function getAvailableBattletags(state: DraftState): string[] {
  const assignedBattletags = new Set(Object.values(state.playerAssignments))
  return state.playerSlots
    .map((s) => s.battletag)
    .filter((bt): bt is string => bt !== null && !assignedBattletags.has(bt))
}

/**
 * Count how many consecutive unfilled pick slots a team has starting
 * from (and including) a given step index. Stops at the first ban,
 * the first pick for the other team, or an already-filled slot.
 * This represents the picks available in the "current turn".
 */
export function consecutivePicksRemaining(
  stepIndex: number,
  team: 'A' | 'B',
  selections: Record<number, string>
): number {
  let count = 0
  for (let i = stepIndex; i < DRAFT_SEQUENCE.length; i++) {
    const s = DRAFT_SEQUENCE[i]
    if (s.type !== 'pick' || s.team !== team) break
    if (selections[i]) break // already filled (e.g. Cho'gall companion)
    count++
  }
  return count
}

/** Format a delta as +X.X or -X.X */
function fmtDelta(d: number): string {
  return `${d >= 0 ? '+' : ''}${d.toFixed(1)}%`
}

// ---------------------------------------------------------------------------
// Scoring — each returns reasons with deltas in percentage points
// ---------------------------------------------------------------------------

/** Resolve hero win rate: prefer map-specific data, fall back to overall. */
export function getHeroWinRate(
  hero: string,
  data: DraftData
): { winRate: number; games: number; isMapSpecific: boolean } | null {
  const mapStats = data.heroMapWinRates[hero]
  if (mapStats && mapStats.games >= 50) {
    return { winRate: mapStats.winRate, games: mapStats.games, isMapSpecific: true }
  }
  const stats = data.heroStats[hero]
  if (!stats) return null
  return { winRate: stats.winRate, games: stats.games, isMapSpecific: false }
}

function scoreHeroWR(
  hero: string,
  data: DraftData
): { delta: number; reason: RecommendationReason | null } {
  const resolved = getHeroWinRate(hero, data)
  if (!resolved || resolved.games < 100) return { delta: 0, reason: null }

  const delta = Math.round((resolved.winRate - 50) * 10) / 10

  // Always include delta in netDelta; only show reason for notable deltas
  const reason = Math.abs(delta) < 0.5 ? null : {
    type: 'hero_wr' as const,
    label: `${hero} ${fmtDelta(delta)}${resolved.isMapSpecific ? ' map' : ''} WR`,
    delta,
  }

  return { delta, reason }
}

function scoreCounters(
  hero: string,
  enemyPicks: string[],
  data: DraftData
): { totalDelta: number; reasons: RecommendationReason[] } {
  const reasons: RecommendationReason[] = []
  let totalDelta = 0
  for (const enemy of enemyPicks) {
    const d = data.counters[hero]?.[enemy]
    if (!d || d.games < 30) continue
    const delta = Math.round((d.winRate - 50) * 10) / 10
    totalDelta += delta
    // Only show as a reason if the delta is notable
    if (Math.abs(delta) >= 1) {
      reasons.push({
        type: 'counter',
        label: `${fmtDelta(delta)} vs ${enemy}`,
        delta,
      })
    }
  }
  return { totalDelta: Math.round(totalDelta * 10) / 10, reasons }
}

function scoreSynergies(
  hero: string,
  ourPicks: string[],
  data: DraftData
): { totalDelta: number; reasons: RecommendationReason[] } {
  const reasons: RecommendationReason[] = []
  let totalDelta = 0
  for (const ally of ourPicks) {
    const d = data.synergies[hero]?.[ally]
    if (!d || d.games < 30) continue
    const delta = Math.round((d.winRate - 50) * 10) / 10
    totalDelta += delta
    // Only show as a reason if the delta is notable
    if (Math.abs(delta) >= 1) {
      reasons.push({
        type: 'synergy',
        label: `${fmtDelta(delta)} with ${ally}`,
        delta,
      })
    }
  }
  return { totalDelta: Math.round(totalDelta * 10) / 10, reasons }
}

function scorePlayerStrength(
  hero: string,
  availableBattletags: string[],
  data: DraftData
): { reason: RecommendationReason | null; player: string | null } {
  if (availableBattletags.length === 0) return { reason: null, player: null }

  let bestDelta = 0
  let bestPlayer: string | null = null

  for (const bt of availableBattletags) {
    const stats = data.playerStats[bt]?.[hero]
    if (!stats || stats.games < 10) continue

    const adjMawp = stats.mawp != null
      ? confidenceAdjustedMawp(stats.mawp, stats.games, 30)
      : (stats.wins / stats.games) * 100

    const delta = Math.round((adjMawp - 50) * 10) / 10

    if (delta > bestDelta) {
      bestDelta = delta
      bestPlayer = bt
    }
  }

  if (bestPlayer && bestDelta >= 2) {
    return {
      reason: {
        type: 'player_strong',
        label: `${bestPlayer.split('#')[0]} ${fmtDelta(bestDelta)} on ${hero}`,
        delta: bestDelta,
      },
      player: bestPlayer,
    }
  }

  return { reason: null, player: bestPlayer }
}

// ---------------------------------------------------------------------------
// Ban scoring
// ---------------------------------------------------------------------------

/**
 * Score a ban candidate from the banning team's perspective.
 *
 * @param picksToProtect  The banning team's own picks (ban heroes strong against these)
 * @param opponentPicks   The opponent's picks (used for role-aware downranking)
 */
function scoreBanCandidate(
  hero: string,
  data: DraftData,
  picksToProtect: string[],
  opponentPicks: string[]
): { netDelta: number; sortBoost: number; reasons: RecommendationReason[] } {
  const reasons: RecommendationReason[] = []
  let netDelta = 0
  let sortBoost = 0

  const stats = data.heroStats[hero]
  if (!stats) return { netDelta: 0, sortBoost: 0, reasons: [] }

  // High WR = good ban target (denying a strong hero)
  const wrDelta = Math.round((stats.winRate - 50) * 10) / 10
  if (wrDelta > 1) {
    reasons.push({
      type: 'ban_worthy',
      label: `${hero} ${fmtDelta(wrDelta)} WR`,
      delta: wrDelta,
    })
    netDelta += wrDelta
  }

  // Strong against the banning team's own picks
  for (const ally of picksToProtect) {
    const d = data.counters[hero]?.[ally]
    if (d && d.games >= 30 && d.winRate >= 53) {
      const delta = Math.round((d.winRate - 50) * 10) / 10
      reasons.push({
        type: 'counter',
        label: `Strong vs ${ally} (${fmtDelta(delta)})`,
        delta,
      })
      netDelta += delta
    }
  }

  return { netDelta: Math.round(netDelta * 10) / 10, sortBoost, reasons }
}

// ---------------------------------------------------------------------------
// Main engine
// ---------------------------------------------------------------------------

export function generateRecommendations(
  state: DraftState,
  data: DraftData
): DraftRecommendation[] {
  if (state.currentStep >= DRAFT_SEQUENCE.length) return []

  const currentDraftStep = DRAFT_SEQUENCE[state.currentStep]
  const unavailable = getUnavailableHeroes(state)
  const ourPicks = getOurPicks(state)
  const enemyPicks = getEnemyPicks(state)
  const availableBattletags = getAvailableBattletags(state)
  const isBanPhase = currentDraftStep.type === 'ban'
  const isOurTurn = currentDraftStep.team === state.ourTeam

  const allHeroes = Object.keys(HERO_ROLES)
  let available = allHeroes.filter((h) => !unavailable.has(h))

  // Cho'gall requires 2 pick slots in the current turn — exclude them
  // if the current team has <2 consecutive picks remaining right now.
  // This applies to BOTH teams (game rule, not team-specific).
  if (!isBanPhase) {
    const turnsLeft = consecutivePicksRemaining(
      state.currentStep, currentDraftStep.team, state.selections
    )
    if (turnsLeft < 2) {
      available = available.filter((h) => h !== 'Cho' && h !== 'Gall')
    }
  }

  if (isBanPhase) {
    // Score from the banning team's perspective:
    // - Our ban: protect our picks, opponent = enemy
    // - Enemy ban: protect enemy picks, opponent = us
    const picksToProtect = isOurTurn ? ourPicks : enemyPicks
    const opponentPicks = isOurTurn ? enemyPicks : ourPicks
    const scored = available.map((hero) => {
      const { netDelta, sortBoost, reasons } = scoreBanCandidate(hero, data, picksToProtect, opponentPicks)
      return { hero, netDelta, sortBoost, reasons, suggestedPlayer: null }
    })
    return scored.sort((a, b) => (b.netDelta + b.sortBoost) - (a.netDelta + a.sortBoost)).slice(0, 15)
  }

  if (!isOurTurn) {
    // Enemy pick — show what they might pick (high WR heroes that counter our team)
    const scored = available.map((hero) => {
      const reasons: RecommendationReason[] = []
      let netDelta = 0

      const { delta: wrDelta, reason: wrReason } = scoreHeroWR(hero, data)
      netDelta += wrDelta
      if (wrReason) { reasons.push(wrReason) }

      // Enemy counters to OUR picks (from enemy's perspective)
      for (const ally of ourPicks) {
        const d = data.counters[hero]?.[ally]
        if (!d || d.games < 30) continue
        const delta = Math.round((d.winRate - 50) * 10) / 10
        netDelta += delta
        if (Math.abs(delta) >= 1) {
          reasons.push({
            type: 'counter',
            label: `${fmtDelta(delta)} vs ${ally}`,
            delta,
          })
        }
      }

      return {
        hero,
        netDelta: Math.round(netDelta * 10) / 10,
        sortBoost: 0,
        reasons,
        suggestedPlayer: null,
      }
    })
    return scored.sort((a, b) => (b.netDelta + b.sortBoost) - (a.netDelta + a.sortBoost)).slice(0, 15)
  }

  // Our pick — full scoring
  const scored = available.map((hero) => {
    const reasons: RecommendationReason[] = []
    let netDelta = 0

    // 1. Hero base WR — data-backed
    const { delta: wrDelta, reason: wrReason } = scoreHeroWR(hero, data)
    netDelta += wrDelta
    if (wrReason) { reasons.push(wrReason) }

    // 2. Counter-picks vs enemy — all pairwise deltas
    const { totalDelta: counterDelta, reasons: counterReasons } = scoreCounters(hero, enemyPicks, data)
    netDelta += counterDelta
    for (const r of counterReasons) { reasons.push(r) }

    // 3. Synergies with allies — all pairwise deltas
    const { totalDelta: synergyDelta, reasons: synergyReasons } = scoreSynergies(hero, ourPicks, data)
    netDelta += synergyDelta
    for (const r of synergyReasons) { reasons.push(r) }

    // 4. Player strength — data-backed
    const { reason: playerReason, player } = scorePlayerStrength(
      hero, availableBattletags, data
    )
    if (playerReason) { reasons.push(playerReason); netDelta += playerReason.delta }

    // 5. Composition win rate — data-driven role/comp scoring
    const { sortBoost: compDelta, reason: compReason } = scoreCompositionForHero(
      hero, ourPicks, data.compositions, data.baselineCompWR
    )
    if (compReason) { reasons.push(compReason) }
    netDelta += compDelta

    return {
      hero,
      netDelta: Math.round(netDelta * 10) / 10,
      sortBoost: 0,
      reasons,
      suggestedPlayer: player,
    }
  })

  return scored.sort((a, b) => b.netDelta - a.netDelta).slice(0, 15)
}
