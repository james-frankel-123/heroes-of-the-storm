/**
 * Composition scoring for the draft engine.
 *
 * Replaces the old hard-coded scoreRoleNeed + scoreRolePenalty with
 * data-driven composition win rates from Heroes Profile.
 *
 * Key idea: given the roles already picked and a candidate hero's role,
 * find the best achievable final composition (i.e. a known 5-role comp
 * that contains the current roles + candidate as a subset) and use its
 * win rate delta from the baseline as a sortBoost.
 */

import type { CompositionData, RecommendationReason } from './types'
import { getHeroRole } from '@/lib/data/hero-roles'

/**
 * Compute the popularity-weighted average win rate across all compositions.
 * This serves as the "baseline" — a comp at exactly this WR contributes 0 boost.
 */
export function computeBaselineCompWR(comps: CompositionData[]): number {
  if (comps.length === 0) return 50

  let totalWeight = 0
  let weightedSum = 0
  for (const c of comps) {
    weightedSum += c.winRate * c.popularity
    totalWeight += c.popularity
  }

  return totalWeight > 0 ? weightedSum / totalWeight : 50
}

/**
 * Convert a list of role strings into a sorted multiset key.
 * e.g. ["Tank", "Healer", "Bruiser"] → "Bruiser,Healer,Tank"
 */
function rolesToMultisetKey(roles: string[]): string {
  return [...roles].sort().join(',')
}

/**
 * Check if `subset` (as a multiset) is contained within `superset`.
 * Both are sorted arrays of role strings.
 *
 * Example: ["Healer","Tank"] is a subset of ["Bruiser","Healer","Ranged Assassin","Ranged Assassin","Tank"]
 */
export function isMultisetSubset(subset: string[], superset: string[]): boolean {
  // Build count map of superset
  const counts = new Map<string, number>()
  for (const r of superset) {
    counts.set(r, (counts.get(r) ?? 0) + 1)
  }

  // Check each element of subset
  for (const r of subset) {
    const count = counts.get(r) ?? 0
    if (count === 0) return false
    counts.set(r, count - 1)
  }

  return true
}

/**
 * Find all compositions that are "achievable" given the current team's roles
 * plus a candidate hero's role.
 *
 * A composition is achievable if the combined roles (current + candidate)
 * form a multiset subset of the composition's 5 roles.
 */
export function getAchievableCompositions(
  currentRoles: string[],
  candidateRole: string,
  allComps: CompositionData[]
): CompositionData[] {
  const combined = [...currentRoles, candidateRole].sort()
  return allComps.filter((comp) => isMultisetSubset(combined, comp.roles))
}

/**
 * Score a candidate hero based on composition win rate data.
 *
 * Returns a sortBoost and a reason for the recommendation panel.
 *
 * @param candidateRole - The role of the hero being evaluated
 * @param currentRoles  - Roles of heroes already picked by our team
 * @param picksMade     - Number of picks our team has already made (0-4)
 * @param comps         - Composition data for the current tier
 * @param baselineWR    - Popularity-weighted average comp WR
 */
export function scoreComposition(
  candidateRole: string,
  currentRoles: string[],
  picksMade: number,
  comps: CompositionData[],
  baselineWR: number
): { sortBoost: number; reason: RecommendationReason | null } {
  if (comps.length === 0) {
    return { sortBoost: 0, reason: null }
  }

  const achievable = getAchievableCompositions(currentRoles, candidateRole, comps)

  // Scale factor: ramps from 0 at start to 1 at last pick
  // This prevents composition scoring from dominating early picks
  const scaleFactor = Math.min(picksMade / 4, 1)

  if (achievable.length === 0) {
    // No known composition matches — penalize
    const penalty = -15 * scaleFactor
    if (Math.abs(penalty) < 0.5) {
      return { sortBoost: 0, reason: null }
    }
    return {
      sortBoost: Math.round(penalty * 10) / 10,
      reason: {
        type: 'comp_wr',
        label: 'No comp data',
        delta: Math.round(penalty * 10) / 10,
      },
    }
  }

  // Find the best achievable composition by win rate
  let best = achievable[0]
  for (let i = 1; i < achievable.length; i++) {
    if (achievable[i].winRate > best.winRate) {
      best = achievable[i]
    }
  }

  const wrDelta = best.winRate - baselineWR
  const sortBoost = Math.round(wrDelta * scaleFactor * 10) / 10

  if (Math.abs(sortBoost) < 0.5) {
    return { sortBoost, reason: null }
  }

  const label = wrDelta >= 0
    ? `Comp ${best.roles.map((r) => r.split(' ')[0]).join('/')} +${wrDelta.toFixed(1)}%`
    : `Comp ${best.roles.map((r) => r.split(' ')[0]).join('/')} ${wrDelta.toFixed(1)}%`

  return {
    sortBoost,
    reason: {
      type: 'comp_wr',
      label,
      delta: sortBoost,
    },
  }
}

/**
 * Convenience: score a hero by name using composition data.
 * Used by the draft engine to replace scoreRoleNeed + scoreRolePenalty.
 */
export function scoreCompositionForHero(
  hero: string,
  ourPicks: string[],
  comps: CompositionData[],
  baselineWR: number
): { sortBoost: number; reason: RecommendationReason | null } {
  const candidateRole = getHeroRole(hero)
  if (!candidateRole) {
    return { sortBoost: 0, reason: null }
  }

  const currentRoles: string[] = ourPicks
    .map(getHeroRole)
    .filter((r): r is NonNullable<typeof r> => r !== null)

  const picksMade = ourPicks.length

  return scoreComposition(candidateRole, currentRoles, picksMade, comps, baselineWR)
}
