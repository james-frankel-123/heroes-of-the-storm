/**
 * Draft recommendation engine.
 *
 * Scores each hero as a net win-rate delta from a 50% baseline.
 * Every data-backed factor is expressed in percentage points so the
 * displayed score reads as "picking this hero shifts our win probability
 * by +X%".
 *
 * Data-backed factors (shown as netDelta):
 *   1. Hero base WR:    (heroWR - 50)
 *   2. Counter-picks:   sum of (pairwise vs enemy - 50) for each enemy
 *   3. Synergies:       sum of (pairwise with ally - 50) for each ally
 *   4. Player strength: best available battletag's (MAWP - 50) on this hero
 *
 * Ranking-only factors (sortBoost — affects order, not displayed %):
 *   5. Role need:       +5 for filling critical (tank/healer), scaled by urgency
 *   6. Role penalty:    -25 for 2nd healer/tank, -10 for dup support/bruiser
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
  ourPicks: string[],
  totalOurPickSlots: number
): RecommendationReason | null {
  const role = getHeroRole(hero)
  if (!role) return null

  const balance = calculateRoleBalance(ourPicks)
  const picksRemaining = totalOurPickSlots - ourPicks.length

  // Scale urgency: the fewer picks remaining, the more critical unfilled roles are.
  // At 5 picks remaining (start): base bonus. At 1 pick remaining (last): 3x bonus.
  const urgency = picksRemaining <= 1 ? 3.0
    : picksRemaining <= 2 ? 2.0
    : picksRemaining <= 3 ? 1.5
    : 1.0

  // Critical: no tank yet
  if (role === 'Tank' && balance.tank === 0) {
    const delta = Math.round(5 * urgency * 10) / 10
    return { type: 'role_need', label: 'Fills Tank', delta }
  }
  // Critical: no healer yet
  if (role === 'Healer' && balance.healer === 0) {
    const delta = Math.round(5 * urgency * 10) / 10
    return { type: 'role_need', label: 'Fills Healer', delta }
  }
  // Important: no damage yet
  if ((role === 'Ranged Assassin' || role === 'Melee Assassin') &&
      balance.rangedAssassin + balance.meleeAssassin === 0) {
    const delta = Math.round(3 * urgency * 10) / 10
    return { type: 'role_need', label: 'Fills Damage', delta }
  }
  // Useful: no bruiser/melee and we have a tank
  if ((role === 'Bruiser' || role === 'Melee Assassin') &&
      balance.bruiser === 0 && balance.meleeAssassin === 0 && balance.tank >= 1) {
    const delta = Math.round(2 * urgency * 10) / 10
    return { type: 'role_need', label: 'Fills Bruiser/Melee', delta }
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

  // Penalize 2nd healer — very heavy, should never appear near top
  if (role === 'Healer' && balance.healer >= 1) {
    return { type: 'role_penalty', label: 'Already have a healer', delta: -25 }
  }
  // Penalize 2nd tank — very heavy, should never appear near top
  if (role === 'Tank' && balance.tank >= 1) {
    return { type: 'role_penalty', label: 'Already have a tank', delta: -25 }
  }
  // Penalize 2nd+ support
  if (role === 'Support' && balance.support >= 1) {
    return { type: 'role_penalty', label: 'Already have a support', delta: -10 }
  }
  // Penalize 3rd+ bruiser
  if (role === 'Bruiser' && balance.bruiser >= 2) {
    return { type: 'role_penalty', label: 'Too many bruisers', delta: -10 }
  }

  return null
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

  // High pick rate = likely to be picked if not banned (ranking boost)
  if (stats.pickRate >= 10) {
    const pickBoost = Math.round(stats.pickRate * 0.3 * 10) / 10
    sortBoost += pickBoost
    reasons.push({
      type: 'ban_worthy',
      label: `${stats.pickRate.toFixed(0)}% pick rate`,
      delta: pickBoost,
    })
  }

  // Role-aware: don't ban a healer/tank if opponent already has one
  // This is a ranking heuristic, not data-backed WR — goes into sortBoost
  const heroRole = getHeroRole(hero)
  const opponentBalance = calculateRoleBalance(opponentPicks)
  if (heroRole === 'Healer' && opponentBalance.healer >= 1) {
    sortBoost -= 8
    reasons.push({
      type: 'role_penalty',
      label: 'Opponent already has healer',
      delta: -8,
    })
  }
  if (heroRole === 'Tank' && opponentBalance.tank >= 1) {
    sortBoost -= 8
    reasons.push({
      type: 'role_penalty',
      label: 'Opponent already has tank',
      delta: -8,
    })
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
  // if the team has <2 consecutive picks remaining right now.
  if (!isBanPhase && isOurTurn) {
    const turnsLeft = consecutivePicksRemaining(
      state.currentStep, state.ourTeam, state.selections
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
        sortBoost: 0,
        reasons,
        suggestedPlayer: null,
      }
    })
    return scored.sort((a, b) => (b.netDelta + b.sortBoost) - (a.netDelta + a.sortBoost)).slice(0, 15)
  }

  // Our pick — full scoring
  const totalOurPickSlots = DRAFT_SEQUENCE.filter(
    (s) => s.type === 'pick' && s.team === state.ourTeam
  ).length

  const scored = available.map((hero) => {
    const reasons: RecommendationReason[] = []
    let netDelta = 0
    let sortBoost = 0

    // 1. Hero base WR — data-backed, goes into netDelta
    const heroWR = scoreHeroWR(hero, data)
    if (heroWR) { reasons.push(heroWR); netDelta += heroWR.delta }

    // 2. Counter-picks vs enemy — data-backed
    const counterReasons = scoreCounters(hero, enemyPicks, data)
    for (const r of counterReasons) { reasons.push(r); netDelta += r.delta }

    // 3. Synergies with allies — data-backed
    const synergyReasons = scoreSynergies(hero, ourPicks, data)
    for (const r of synergyReasons) { reasons.push(r); netDelta += r.delta }

    // 4. Player strength — data-backed
    const { reason: playerReason, player } = scorePlayerStrength(
      hero, availableBattletags, data
    )
    if (playerReason) { reasons.push(playerReason); netDelta += playerReason.delta }

    // 5. Role need — ranking boost only (not shown in netDelta)
    const roleNeed = scoreRoleNeed(hero, ourPicks, totalOurPickSlots)
    if (roleNeed) { reasons.push(roleNeed); sortBoost += roleNeed.delta }

    // 6. Role penalty — ranking penalty only (not shown in netDelta)
    const rolePenalty = scoreRolePenalty(hero, ourPicks)
    if (rolePenalty) { reasons.push(rolePenalty); sortBoost += rolePenalty.delta }

    return {
      hero,
      netDelta: Math.round(netDelta * 10) / 10,
      sortBoost: Math.round(sortBoost * 10) / 10,
      reasons,
      suggestedPlayer: player,
    }
  })

  return scored.sort((a, b) => (b.netDelta + b.sortBoost) - (a.netDelta + a.sortBoost)).slice(0, 15)
}
