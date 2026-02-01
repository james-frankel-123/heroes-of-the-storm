/**
 * Hero Recommendation Matcher
 *
 * Matches recommended heroes to capable players based on competency scores
 */

import { PlayerCompetency, HeroCompetency, findBestPlayerForHero, isCompetentWith } from './competency-score'

export interface HeroRecommendation {
  hero: string
  player: {
    battletag: string
    slot: number
  } | null
  competency: HeroCompetency | null
  reasoning: string
  priority: 'critical' | 'important' | 'nice-to-have'
  noOneCompetent: boolean  // True if no one can play this hero well
}

export interface RecommendationResult {
  recommendations: HeroRecommendation[]
  warnings: string[]  // E.g., "No one can play tanks"
}

/**
 * Match a list of hero candidates to capable players
 */
export function matchHeroesToPlayers(
  heroCandidates: string[],
  playerCompetencies: PlayerCompetency[],
  roleNeeds: {
    role: string
    priority: 'critical' | 'important' | 'nice-to-have'
    heroes: string[]
  }[]
): RecommendationResult {
  const recommendations: HeroRecommendation[] = []
  const warnings: string[] = []

  // For each hero candidate, find the best player
  for (const hero of heroCandidates) {
    const { player, competency } = findBestPlayerForHero(hero, playerCompetencies)

    // Determine which role need this hero fulfills
    let priority: 'critical' | 'important' | 'nice-to-have' = 'nice-to-have'
    let reasoning = 'Good overall pick'

    for (const roleNeed of roleNeeds) {
      if (roleNeed.heroes.includes(hero)) {
        priority = roleNeed.priority
        reasoning = `Fills ${roleNeed.role} role`
        break
      }
    }

    const noOneCompetent = !competency || competency.games < 5

    recommendations.push({
      hero,
      player: player ? {
        battletag: player.battletag,
        slot: player.slot
      } : null,
      competency,
      reasoning,
      priority,
      noOneCompetent
    })
  }

  // Sort recommendations by priority, then competency score
  recommendations.sort((a, b) => {
    // Priority order
    const priorityOrder = { critical: 0, important: 1, 'nice-to-have': 2 }
    if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
      return priorityOrder[a.priority] - priorityOrder[b.priority]
    }

    // Then by competency score
    const aScore = a.competency?.competencyScore || 0
    const bScore = b.competency?.competencyScore || 0
    return bScore - aScore
  })

  // Generate warnings for critical roles no one can play
  const criticalNeeds = roleNeeds.filter(need => need.priority === 'critical')
  for (const need of criticalNeeds) {
    const anyoneCompetent = need.heroes.some(hero => {
      const { player, competency } = findBestPlayerForHero(hero, playerCompetencies)
      return competency && competency.games >= 5
    })

    if (!anyoneCompetent) {
      warnings.push(`⚠️ No one has experience with ${need.role} heroes`)
    }
  }

  return {
    recommendations: recommendations.slice(0, 10),  // Top 10
    warnings
  }
}

/**
 * Get top 3 recommendations for current turn
 */
export function getTop3Recommendations(
  result: RecommendationResult
): HeroRecommendation[] {
  return result.recommendations.slice(0, 3)
}

/**
 * Format recommendation for AI prompt
 */
export function formatRecommendationForPrompt(rec: HeroRecommendation): string {
  if (rec.noOneCompetent || !rec.player || !rec.competency) {
    return `${rec.hero} - Would be ideal but NO ONE has experience (consider finding player who can)`
  }

  const mapIndicator = rec.competency.mapBonus ? ' ⭐ (strong on this map)' : ''

  return `${rec.hero} - FOR ${rec.player.battletag} (Slot ${rec.player.slot + 1})
   • Stats: ${rec.competency.winRate}% WR, ${rec.competency.games} games${mapIndicator}
   • Reason: ${rec.reasoning}`
}

/**
 * Format recommendation for display in UI
 */
export function formatRecommendationForUI(rec: HeroRecommendation): {
  heroName: string
  playerName: string
  stats: string
  reason: string
  warning: string | null
} {
  if (rec.noOneCompetent || !rec.player || !rec.competency) {
    return {
      heroName: rec.hero,
      playerName: 'No one available',
      stats: 'No experience',
      reason: rec.reasoning,
      warning: 'No one on your team has played this hero'
    }
  }

  const stats = `${rec.competency.winRate}% WR, ${rec.competency.games}g`
  const mapBonus = rec.competency.mapBonus ? ' (Strong on map)' : ''

  return {
    heroName: rec.hero,
    playerName: rec.player.battletag.split('#')[0],
    stats: stats + mapBonus,
    reason: rec.reasoning,
    warning: null
  }
}

/**
 * Check if any player in roster can play a specific role
 */
export function canAnyonePlayRole(
  role: string,
  roleHeroes: string[],
  playerCompetencies: PlayerCompetency[]
): boolean {
  for (const hero of roleHeroes) {
    for (const player of playerCompetencies) {
      if (isCompetentWith(hero, player)) {
        return true
      }
    }
  }
  return false
}

/**
 * Get heroes that a specific player can play from a list
 */
export function getPlayerCapableHeroes(
  availableHeroes: string[],
  playerCompetency: PlayerCompetency,
  minGames: number = 5
): HeroCompetency[] {
  return playerCompetency.topHeroes.filter(
    h => availableHeroes.includes(h.hero) && h.games >= minGames
  )
}

/**
 * Build role analysis for current draft state
 */
export interface RoleAnalysis {
  role: string
  current: number  // How many in current comp
  needed: boolean
  priority: 'critical' | 'important' | 'nice-to-have'
  availableHeroes: string[]
  capablePlayers: {
    battletag: string
    heroes: HeroCompetency[]
  }[]
}

export function analyzeRoleNeeds(
  currentPicks: (string | null)[],
  roleDefinitions: { [hero: string]: string },
  availableHeroes: string[],
  playerCompetencies: PlayerCompetency[]
): RoleAnalysis[] {
  // Count current roles
  const roleCounts: { [role: string]: number } = {}
  const definedRoles = ['Tank', 'Healer', 'Bruiser', 'Ranged Assassin', 'Melee Assassin', 'Support']

  for (const pick of currentPicks) {
    if (pick && roleDefinitions[pick]) {
      const role = roleDefinitions[pick]
      roleCounts[role] = (roleCounts[role] || 0) + 1
    }
  }

  // Analyze each role
  const analyses: RoleAnalysis[] = []

  for (const role of definedRoles) {
    const current = roleCounts[role] || 0
    let needed = false
    let priority: 'critical' | 'important' | 'nice-to-have' = 'nice-to-have'

    // Determine if role is needed (simple heuristics)
    if (role === 'Tank' && current === 0) {
      needed = true
      priority = 'critical'
    } else if (role === 'Healer' && current === 0) {
      needed = true
      priority = 'critical'
    } else if ((role === 'Ranged Assassin' || role === 'Melee Assassin') && current < 2) {
      needed = true
      priority = 'important'
    }

    // Find available heroes for this role
    const roleHeroes = availableHeroes.filter(h => roleDefinitions[h] === role)

    // Find capable players
    const capablePlayers = playerCompetencies
      .map(player => ({
        battletag: player.battletag,
        heroes: getPlayerCapableHeroes(roleHeroes, player)
      }))
      .filter(p => p.heroes.length > 0)

    analyses.push({
      role,
      current,
      needed,
      priority,
      availableHeroes: roleHeroes,
      capablePlayers
    })
  }

  return analyses.filter(a => a.needed)
}
