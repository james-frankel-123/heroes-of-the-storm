/**
 * Draft recommendation engine.
 *
 * Scores each hero as a net win-rate delta from a 50% baseline.
 * Every factor is expressed in percentage points so the final score
 * reads as "we estimate picking this hero shifts our win probability
 * by +X%".
 *
 * Factors:
 *   1. Hero base WR:    (heroWR - 50)
 *   2. Counter-picks:   sum of (pairwise vs enemy - 50) for each enemy
 *   3. Synergies:       sum of (pairwise with ally - 50) for each ally
 *   4. Player strength: best available battletag's (MAWP - 50) on this hero
 *   5. Role need:       +3 for filling critical, +1.5 for important
 *   6. Role penalty:    -15 for 2nd healer/tank, -8 for bad comp
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
  HERO_ROLES,
} from '@/lib/data/hero-roles'
import { confidenceAdjustedMawp } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getUnavailableHeroes(state: DraftState): Set<string> {
  return new Set(Object.values(state.selections))
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

/** Format a delta as +X.X or -X.X */
function fmtDelta(d: number): string {
  return `${d >= 0 ? '+' : ''}${d.toFixed(1)}%`
}

// ---------------------------------------------------------------------------
// Scoring — each returns reasons with deltas in percentage points
// ---------------------------------------------------------------------------

function scoreHeroWR(
  hero: string,
  data: DraftData
): RecommendationReason | null {
  const stats = data.heroStats[hero]
  if (!stats || stats.games < 100) return null

  const delta = Math.round((stats.winRate - 50) * 10) / 10
  if (Math.abs(delta) < 0.5) return null

  return {
    type: 'hero_wr',
    label: `${hero} ${fmtDelta(delta)} base WR`,
    delta,
  }
}

function scoreCounters(
  hero: string,
  enemyPicks: string[],
  data: DraftData
): RecommendationReason[] {
  const reasons: RecommendationReason[] = []
  for (const enemy of enemyPicks) {
    const d = data.counters[hero]?.[enemy]
    if (!d || d.games < 30) continue
    const delta = Math.round((d.winRate - 50) * 10) / 10
    if (Math.abs(delta) < 1) continue
    reasons.push({
      type: 'counter',
      label: `${fmtDelta(delta)} vs ${enemy}`,
      delta,
    })
  }
  return reasons
}

function scoreSynergies(
  hero: string,
  ourPicks: string[],
  data: DraftData
): RecommendationReason[] {
  const reasons: RecommendationReason[] = []
  for (const ally of ourPicks) {
    const d = data.synergies[hero]?.[ally]
    if (!d || d.games < 30) continue
    const delta = Math.round((d.winRate - 50) * 10) / 10
    if (Math.abs(delta) < 1) continue
    reasons.push({
      type: 'synergy',
      label: `${fmtDelta(delta)} with ${ally}`,
      delta,
    })
  }
  return reasons
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

function scoreRoleNeed(
  hero: string,
  ourPicks: string[]
): RecommendationReason | null {
  const role = getHeroRole(hero)
  if (!role) return null

  const balance = calculateRoleBalance(ourPicks)

  // Critical: no tank yet
  if (role === 'Tank' && balance.tank === 0) {
    return { type: 'role_need', label: 'Fills Tank', delta: 3 }
  }
  // Critical: no healer yet
  if (role === 'Healer' && balance.healer === 0) {
    return { type: 'role_need', label: 'Fills Healer', delta: 3 }
  }
  // Critical: no damage yet
  if ((role === 'Ranged Assassin' || role === 'Melee Assassin') &&
      balance.rangedAssassin + balance.meleeAssassin === 0) {
    return { type: 'role_need', label: 'Fills Damage', delta: 3 }
  }
  // Important: no bruiser/melee and we have a tank
  if ((role === 'Bruiser' || role === 'Melee Assassin') &&
      balance.bruiser === 0 && balance.meleeAssassin === 0 && balance.tank >= 1) {
    return { type: 'role_need', label: 'Fills Bruiser/Melee', delta: 1.5 }
  }

  return null
}

function scoreRolePenalty(
  hero: string,
  ourPicks: string[]
): RecommendationReason | null {
  const role = getHeroRole(hero)
  if (!role) return null

  const balance = calculateRoleBalance(ourPicks)

  // Penalize 2nd healer
  if (role === 'Healer' && balance.healer >= 1) {
    return { type: 'role_penalty', label: 'Already have a healer', delta: -15 }
  }
  // Penalize 2nd tank
  if (role === 'Tank' && balance.tank >= 1) {
    return { type: 'role_penalty', label: 'Already have a tank', delta: -15 }
  }
  // Penalize 3rd+ support/bruiser
  if (role === 'Support' && balance.support >= 2) {
    return { type: 'role_penalty', label: 'Too many supports', delta: -8 }
  }

  return null
}

// ---------------------------------------------------------------------------
// Ban scoring
// ---------------------------------------------------------------------------

function scoreBanCandidate(
  hero: string,
  data: DraftData,
  enemyPicks: string[],
  ourPicks: string[]
): { netDelta: number; reasons: RecommendationReason[] } {
  const reasons: RecommendationReason[] = []
  let netDelta = 0

  const stats = data.heroStats[hero]
  if (!stats) return { netDelta: 0, reasons: [] }

  // High WR = good ban target (denying this hero from the enemy)
  const wrDelta = Math.round((stats.winRate - 50) * 10) / 10
  if (wrDelta > 1) {
    reasons.push({
      type: 'ban_worthy',
      label: `${hero} ${fmtDelta(wrDelta)} WR`,
      delta: wrDelta,
    })
    netDelta += wrDelta
  }

  // High ban rate = community agrees
  if (stats.banRate >= 15) {
    const banDelta = Math.round(stats.banRate * 0.1 * 10) / 10
    reasons.push({
      type: 'ban_worthy',
      label: `${stats.banRate.toFixed(0)}% ban rate`,
      delta: banDelta,
    })
    netDelta += banDelta
  }

  // Would counter our existing picks
  for (const ally of ourPicks) {
    const d = data.counters[hero]?.[ally]
    if (d && d.games >= 30 && d.winRate >= 53) {
      const delta = Math.round((d.winRate - 50) * 10) / 10
      reasons.push({
        type: 'counter',
        label: `Threatens ${ally} (${fmtDelta(delta)})`,
        delta,
      })
      netDelta += delta
    }
  }

  // Role-aware banning: don't suggest banning a healer/tank if enemy has one
  const heroRole = getHeroRole(hero)
  const enemyBalance = calculateRoleBalance(enemyPicks)
  if (heroRole === 'Healer' && enemyBalance.healer >= 1) {
    netDelta -= 8
    reasons.push({
      type: 'role_penalty',
      label: 'Enemy already has healer',
      delta: -8,
    })
  }
  if (heroRole === 'Tank' && enemyBalance.tank >= 1) {
    netDelta -= 8
    reasons.push({
      type: 'role_penalty',
      label: 'Enemy already has tank',
      delta: -8,
    })
  }

  return { netDelta: Math.round(netDelta * 10) / 10, reasons }
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
  const available = allHeroes.filter((h) => !unavailable.has(h))

  if (isBanPhase) {
    const scored = available.map((hero) => {
      const { netDelta, reasons } = scoreBanCandidate(hero, data, enemyPicks, ourPicks)
      return { hero, netDelta, reasons, suggestedPlayer: null }
    })
    return scored.sort((a, b) => b.netDelta - a.netDelta).slice(0, 15)
  }

  if (!isOurTurn) {
    // Enemy pick — show what they might pick (high WR heroes that counter our team)
    const scored = available.map((hero) => {
      const reasons: RecommendationReason[] = []
      let netDelta = 0

      const heroWR = scoreHeroWR(hero, data)
      if (heroWR) { reasons.push(heroWR); netDelta += heroWR.delta }

      // Enemy counters to OUR picks (from enemy's perspective)
      for (const ally of ourPicks) {
        const d = data.counters[hero]?.[ally]
        if (d && d.games >= 30) {
          const delta = Math.round((d.winRate - 50) * 10) / 10
          if (Math.abs(delta) >= 1) {
            reasons.push({
              type: 'counter',
              label: `${fmtDelta(delta)} vs your ${ally}`,
              delta,
            })
            netDelta += delta
          }
        }
      }

      return {
        hero,
        netDelta: Math.round(netDelta * 10) / 10,
        reasons,
        suggestedPlayer: null,
      }
    })
    return scored.sort((a, b) => b.netDelta - a.netDelta).slice(0, 15)
  }

  // Our pick — full scoring
  const scored = available.map((hero) => {
    const reasons: RecommendationReason[] = []
    let netDelta = 0

    // 1. Hero base WR
    const heroWR = scoreHeroWR(hero, data)
    if (heroWR) { reasons.push(heroWR); netDelta += heroWR.delta }

    // 2. Counter-picks vs enemy
    const counterReasons = scoreCounters(hero, enemyPicks, data)
    for (const r of counterReasons) { reasons.push(r); netDelta += r.delta }

    // 3. Synergies with allies
    const synergyReasons = scoreSynergies(hero, ourPicks, data)
    for (const r of synergyReasons) { reasons.push(r); netDelta += r.delta }

    // 4. Player strength (only from unassigned battletags)
    const { reason: playerReason, player } = scorePlayerStrength(
      hero, availableBattletags, data
    )
    if (playerReason) { reasons.push(playerReason); netDelta += playerReason.delta }

    // 5. Role need bonus
    const roleNeed = scoreRoleNeed(hero, ourPicks)
    if (roleNeed) { reasons.push(roleNeed); netDelta += roleNeed.delta }

    // 6. Role penalty
    const rolePenalty = scoreRolePenalty(hero, ourPicks)
    if (rolePenalty) { reasons.push(rolePenalty); netDelta += rolePenalty.delta }

    return {
      hero,
      netDelta: Math.round(netDelta * 10) / 10,
      reasons,
      suggestedPlayer: player,
    }
  })

  return scored.sort((a, b) => b.netDelta - a.netDelta).slice(0, 15)
}
