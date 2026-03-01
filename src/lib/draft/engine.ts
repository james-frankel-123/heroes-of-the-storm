/**
 * Draft recommendation engine.
 *
 * Pure function: takes draft state + precomputed data, returns scored
 * recommendations. Runs entirely client-side for <200ms latency.
 *
 * Scoring weights (tuned for HotS draft priorities):
 *   - Map performance:      25%
 *   - Counter-pick value:   20%
 *   - Synergy with team:    15%
 *   - Role need:            20%
 *   - Player strength:      15%
 *   - Meta / overall WR:     5%
 */

import {
  type DraftState,
  type DraftData,
  type DraftRecommendation,
  type RecommendationReason,
  DRAFT_SEQUENCE,
} from './types'
import {
  getHeroRole,
  calculateRoleBalance,
  analyzeRoleNeeds,
  HERO_ROLES,
} from '@/lib/data/hero-roles'
import { confidenceAdjustedWinRate } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Get all heroes that are already banned or picked */
function getUnavailableHeroes(state: DraftState): Set<string> {
  return new Set(Object.values(state.selections))
}

/** Get heroes picked by our team */
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

/** Get heroes picked by the enemy team */
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

/** Get assigned battletags for our team */
function getOurBattletags(state: DraftState): string[] {
  return state.playerSlots
    .map((s) => s.battletag)
    .filter((bt): bt is string => bt !== null)
}

/** Normalize a score to 0-100 range */
function normalize(value: number, min: number, max: number): number {
  if (max === min) return 50
  return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100))
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

function scoreMapPerformance(
  hero: string,
  data: DraftData
): { score: number; reason: RecommendationReason | null } {
  const mapStats = data.heroMapWinRates[hero]
  if (!mapStats || mapStats.games < 20) {
    return { score: 50, reason: null }
  }

  const score = normalize(mapStats.winRate, 42, 58)

  if (mapStats.winRate >= 53) {
    return {
      score,
      reason: {
        type: 'map_strong',
        label: `${mapStats.winRate.toFixed(1)}% WR on this map`,
        weight: score / 100,
      },
    }
  }

  return { score, reason: null }
}

function scoreCounterPick(
  hero: string,
  enemyPicks: string[],
  data: DraftData
): { score: number; reasons: RecommendationReason[] } {
  if (enemyPicks.length === 0) return { score: 50, reasons: [] }

  const reasons: RecommendationReason[] = []
  let totalScore = 0
  let count = 0

  for (const enemy of enemyPicks) {
    const counterData = data.counters[hero]?.[enemy]
    if (counterData && counterData.games >= 30) {
      const s = normalize(counterData.winRate, 45, 60)
      totalScore += s
      count++

      if (counterData.winRate >= 53) {
        reasons.push({
          type: 'counter',
          label: `${counterData.winRate.toFixed(1)}% vs ${enemy}`,
          weight: s / 100,
        })
      }
    } else {
      totalScore += 50
      count++
    }
  }

  return {
    score: count > 0 ? totalScore / count : 50,
    reasons,
  }
}

function scoreSynergy(
  hero: string,
  ourPicks: string[],
  data: DraftData
): { score: number; reasons: RecommendationReason[] } {
  if (ourPicks.length === 0) return { score: 50, reasons: [] }

  const reasons: RecommendationReason[] = []
  let totalScore = 0
  let count = 0

  for (const ally of ourPicks) {
    const synergyData = data.synergies[hero]?.[ally]
    if (synergyData && synergyData.games >= 30) {
      const s = normalize(synergyData.winRate, 45, 60)
      totalScore += s
      count++

      if (synergyData.winRate >= 53) {
        reasons.push({
          type: 'synergy',
          label: `${synergyData.winRate.toFixed(1)}% with ${ally}`,
          weight: s / 100,
        })
      }
    } else {
      totalScore += 50
      count++
    }
  }

  return {
    score: count > 0 ? totalScore / count : 50,
    reasons,
  }
}

function scoreRoleNeed(
  hero: string,
  ourPicks: string[]
): { score: number; reason: RecommendationReason | null } {
  const role = getHeroRole(hero)
  if (!role) return { score: 50, reason: null }

  const balance = calculateRoleBalance(ourPicks)
  const needs = analyzeRoleNeeds(balance)

  for (const need of needs) {
    const matches =
      need.role === 'Damage'
        ? role === 'Ranged Assassin' || role === 'Melee Assassin'
        : role === need.role

    if (matches) {
      const score =
        need.priority === 'critical' ? 95 : need.priority === 'important' ? 75 : 60

      return {
        score,
        reason: {
          type: 'role_need',
          label: `Fills ${need.priority} ${need.role} need`,
          weight: score / 100,
        },
      }
    }
  }

  // If no specific need, slightly favor roles that aren't over-represented
  return { score: 50, reason: null }
}

function scorePlayerStrength(
  hero: string,
  battletags: string[],
  data: DraftData
): { score: number; reason: RecommendationReason | null; player: string | null } {
  if (battletags.length === 0) return { score: 50, reason: null, player: null }

  let bestScore = 0
  let bestPlayer: string | null = null
  let bestWinRate = 0

  for (const bt of battletags) {
    const stats = data.playerStats[bt]?.[hero]
    if (!stats || stats.games < 10) continue

    const adjWr = confidenceAdjustedWinRate(stats.wins, stats.games, 30)

    const s = normalize(adjWr, 45, 65)

    // Also check map-specific performance
    const mapStats = data.playerMapStats[bt]?.[hero]
    const mapBonus = mapStats && mapStats.games >= 5 && mapStats.winRate >= 55 ? 10 : 0

    const finalScore = s + mapBonus

    if (finalScore > bestScore) {
      bestScore = finalScore
      bestPlayer = bt
      bestWinRate = adjWr
    }
  }

  if (bestPlayer && bestWinRate >= 53) {
    return {
      score: Math.min(100, bestScore),
      reason: {
        type: 'player_strong',
        label: `${bestPlayer.split('#')[0]} ${bestWinRate.toFixed(1)}% WR`,
        weight: bestScore / 100,
      },
      player: bestPlayer,
    }
  }

  return { score: Math.max(50, bestScore), reason: null, player: bestPlayer }
}

function scoreMetaStrength(
  hero: string,
  data: DraftData
): { score: number; reason: RecommendationReason | null } {
  const stats = data.heroStats[hero]
  if (!stats || stats.games < 100) {
    return { score: 50, reason: null }
  }

  const score = normalize(stats.winRate, 44, 56)

  if (stats.winRate >= 54) {
    return {
      score,
      reason: {
        type: 'meta_strong',
        label: `${stats.winRate.toFixed(1)}% overall WR`,
        weight: score / 100,
      },
    }
  }

  return { score, reason: null }
}

// ---------------------------------------------------------------------------
// Ban recommendations
// ---------------------------------------------------------------------------

function scoreBanCandidate(
  hero: string,
  data: DraftData,
  enemyPicks: string[],
  ourPicks: string[]
): { score: number; reasons: RecommendationReason[] } {
  const reasons: RecommendationReason[] = []
  let score = 0

  const stats = data.heroStats[hero]
  if (!stats) return { score: 0, reasons: [] }

  // High win rate heroes are ban-worthy
  const wrScore = normalize(stats.winRate, 48, 58) * 0.4
  score += wrScore

  // High ban rate = community agrees it's ban-worthy
  const banScore = normalize(stats.banRate, 5, 30) * 0.3
  score += banScore

  // Map-specific dominance
  const mapStats = data.heroMapWinRates[hero]
  if (mapStats && mapStats.winRate >= 55) {
    const mapScore = normalize(mapStats.winRate, 50, 62) * 0.3
    score += mapScore
    reasons.push({
      type: 'ban_worthy',
      label: `${mapStats.winRate.toFixed(1)}% on this map`,
      weight: mapScore / 100,
    })
  } else {
    score += 15 // neutral map contribution
  }

  if (stats.winRate >= 53 || stats.banRate >= 15) {
    reasons.push({
      type: 'ban_worthy',
      label: `${stats.winRate.toFixed(1)}% WR, ${stats.banRate.toFixed(0)}% ban rate`,
      weight: wrScore / 100,
    })
  }

  return { score, reasons }
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
  const ourBattletags = getOurBattletags(state)
  const isBanPhase = currentDraftStep.type === 'ban'
  const isOurTurn = currentDraftStep.team === state.ourTeam

  const allHeroes = Object.keys(HERO_ROLES)
  const available = allHeroes.filter((h) => !unavailable.has(h))

  if (isBanPhase) {
    // Ban recommendations
    const scored = available.map((hero) => {
      const { score, reasons } = scoreBanCandidate(
        hero,
        data,
        enemyPicks,
        ourPicks
      )
      return {
        hero,
        score: Math.round(score),
        reasons,
        suggestedPlayer: null,
      }
    })

    return scored.sort((a, b) => b.score - a.score).slice(0, 15)
  }

  if (!isOurTurn) {
    // Enemy pick phase — show what they might pick so we can plan
    // Just show top heroes by meta strength + map
    const scored = available.map((hero) => {
      const map = scoreMapPerformance(hero, data)
      const meta = scoreMetaStrength(hero, data)
      const counter = scoreCounterPick(hero, ourPicks, data)

      const score = map.score * 0.4 + meta.score * 0.3 + counter.score * 0.3
      const reasons: RecommendationReason[] = []
      if (map.reason) reasons.push(map.reason)
      if (meta.reason) reasons.push(meta.reason)
      counter.reasons.forEach((r) => reasons.push(r))

      return {
        hero,
        score: Math.round(score),
        reasons,
        suggestedPlayer: null,
      }
    })

    return scored.sort((a, b) => b.score - a.score).slice(0, 15)
  }

  // Our pick phase — full recommendation engine
  const scored = available.map((hero) => {
    const map = scoreMapPerformance(hero, data)
    const counter = scoreCounterPick(hero, enemyPicks, data)
    const synergy = scoreSynergy(hero, ourPicks, data)
    const role = scoreRoleNeed(hero, ourPicks)
    const player = scorePlayerStrength(hero, ourBattletags, data)
    const meta = scoreMetaStrength(hero, data)

    // Weighted composite
    const score =
      map.score * 0.25 +
      counter.score * 0.20 +
      synergy.score * 0.15 +
      role.score * 0.20 +
      player.score * 0.15 +
      meta.score * 0.05

    const reasons: RecommendationReason[] = []
    if (role.reason) reasons.push(role.reason)
    if (player.reason) reasons.push(player.reason)
    if (map.reason) reasons.push(map.reason)
    counter.reasons.forEach((r) => reasons.push(r))
    synergy.reasons.forEach((r) => reasons.push(r))
    if (meta.reason) reasons.push(meta.reason)

    return {
      hero,
      score: Math.round(score),
      reasons,
      suggestedPlayer: player.player,
    }
  })

  return scored.sort((a, b) => b.score - a.score).slice(0, 15)
}
