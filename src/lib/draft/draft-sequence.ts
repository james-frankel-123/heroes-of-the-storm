/**
 * Draft Sequence State Machine for Storm League Draft
 *
 * Defines the official Heroes of the Storm Storm League draft order
 * with 16 total turns across 3 phases:
 * - Phase 1: Opening bans + first picks (5 turns)
 * - Phase 2: Middle bans + flex picks (5 turns)
 * - Phase 3: Final bans + last picks (6 turns)
 */

export type DraftTeam = 'blue' | 'red'
export type DraftAction = 'ban' | 'pick'
export type DraftPhase = 1 | 2 | 3

export interface DraftTurn {
  team: DraftTeam
  action: DraftAction
  phase: DraftPhase
  number: number  // Ban/pick number within its type (e.g., ban #1, pick #2)
  pickSlot?: number  // Which of the 5 pick slots (0-4) - only for picks
  turnIndex: number  // Overall turn index (0-15)
}

/**
 * Official Storm League draft sequence
 * Total: 6 bans (3 per team), 5 picks per team = 16 turns
 */
export const DRAFT_SEQUENCE: DraftTurn[] = [
  // Phase 1: Opening Bans + First Picks
  { team: 'red', action: 'ban', phase: 1, number: 1, turnIndex: 0 },
  { team: 'blue', action: 'ban', phase: 1, number: 1, turnIndex: 1 },
  { team: 'blue', action: 'pick', phase: 1, number: 1, pickSlot: 0, turnIndex: 2 },
  { team: 'red', action: 'pick', phase: 1, number: 1, pickSlot: 0, turnIndex: 3 },
  { team: 'red', action: 'pick', phase: 1, number: 2, pickSlot: 1, turnIndex: 4 },

  // Phase 2: Middle Bans + Flex Picks
  { team: 'blue', action: 'ban', phase: 2, number: 2, turnIndex: 5 },
  { team: 'red', action: 'ban', phase: 2, number: 2, turnIndex: 6 },
  { team: 'red', action: 'pick', phase: 2, number: 3, pickSlot: 2, turnIndex: 7 },
  { team: 'blue', action: 'pick', phase: 2, number: 2, pickSlot: 1, turnIndex: 8 },
  { team: 'blue', action: 'pick', phase: 2, number: 3, pickSlot: 2, turnIndex: 9 },

  // Phase 3: Final Bans + Last Picks
  { team: 'red', action: 'ban', phase: 3, number: 3, turnIndex: 10 },
  { team: 'blue', action: 'ban', phase: 3, number: 3, turnIndex: 11 },
  { team: 'blue', action: 'pick', phase: 3, number: 4, pickSlot: 3, turnIndex: 12 },
  { team: 'red', action: 'pick', phase: 3, number: 4, pickSlot: 3, turnIndex: 13 },
  { team: 'red', action: 'pick', phase: 3, number: 5, pickSlot: 4, turnIndex: 14 },
  { team: 'blue', action: 'pick', phase: 3, number: 5, pickSlot: 4, turnIndex: 15 },
]

/**
 * Get the current turn by index
 */
export function getCurrentTurn(turnIndex: number): DraftTurn | null {
  if (turnIndex < 0 || turnIndex >= DRAFT_SEQUENCE.length) {
    return null
  }
  return DRAFT_SEQUENCE[turnIndex]
}

/**
 * Get the next turn
 */
export function getNextTurn(currentTurnIndex: number): DraftTurn | null {
  const nextIndex = currentTurnIndex + 1
  if (nextIndex >= DRAFT_SEQUENCE.length) {
    return null
  }
  return DRAFT_SEQUENCE[nextIndex]
}

/**
 * Check if it's the specified team's turn
 */
export function isYourTurn(currentTurn: DraftTurn, yourTeam: DraftTeam): boolean {
  return currentTurn.team === yourTeam
}

/**
 * Get the draft phase name
 */
export function getDraftPhaseName(phase: DraftPhase): string {
  switch (phase) {
    case 1:
      return 'Opening'
    case 2:
      return 'Middle'
    case 3:
      return 'Final'
  }
}

/**
 * Get a human-readable turn description
 */
export function getTurnDescription(turn: DraftTurn, yourTeam: DraftTeam): string {
  const teamLabel = turn.team === yourTeam ? 'YOUR' : 'OPPONENT'
  const actionLabel = turn.action.toUpperCase()

  if (turn.action === 'ban') {
    return `${teamLabel} ${actionLabel} #${turn.number}`
  } else {
    return `${teamLabel} PICK #${turn.number}`
  }
}

/**
 * Check if draft is complete
 */
export function isDraftComplete(turnIndex: number): boolean {
  return turnIndex >= DRAFT_SEQUENCE.length
}

/**
 * Get all bans for a specific team up to a turn index
 */
export function getBansForTeam(
  team: DraftTeam,
  turnIndex: number,
  bans: { [key in DraftTeam]: string[] }
): string[] {
  return bans[team] || []
}

/**
 * Get all picks for a specific team up to a turn index
 */
export function getPicksForTeam(
  team: DraftTeam,
  turnIndex: number,
  picks: { [key in DraftTeam]: (string | null)[] }
): (string | null)[] {
  return picks[team] || [null, null, null, null, null]
}

/**
 * Get all banned heroes (both teams)
 */
export function getAllBannedHeroes(bans: { [key in DraftTeam]: string[] }): string[] {
  return [...bans.blue, ...bans.red]
}

/**
 * Get all picked heroes (both teams)
 */
export function getAllPickedHeroes(picks: { [key in DraftTeam]: (string | null)[] }): string[] {
  return [...picks.blue, ...picks.red].filter((hero): hero is string => hero !== null)
}

/**
 * Get available heroes (not banned or picked)
 */
export function getAvailableHeroes(
  allHeroes: string[],
  bans: { [key in DraftTeam]: string[] },
  picks: { [key in DraftTeam]: (string | null)[] }
): string[] {
  const unavailable = new Set([
    ...getAllBannedHeroes(bans),
    ...getAllPickedHeroes(picks)
  ])

  return allHeroes.filter(hero => !unavailable.has(hero))
}

/**
 * Format turn for display in turn indicator
 */
export function formatTurnIndicator(
  turn: DraftTurn,
  yourTeam: DraftTeam
): {
  isYourTurn: boolean
  teamLabel: string
  actionLabel: string
  phaseLabel: string
  description: string
} {
  const isYourTurnFlag = isYourTurn(turn, yourTeam)
  const teamLabel = turn.team === yourTeam ? 'YOUR TEAM' : 'OPPONENT'
  const teamColor = turn.team === 'blue' ? 'BLUE' : 'RED'

  let actionLabel = ''
  if (turn.action === 'ban') {
    actionLabel = `BAN #${turn.number}`
  } else {
    actionLabel = `PICK #${turn.number}`
  }

  const phaseLabel = `Phase ${turn.phase}: ${getDraftPhaseName(turn.phase)}`

  const description = isYourTurnFlag
    ? `Select a hero to ${turn.action}`
    : `Waiting for ${teamColor} team to ${turn.action}...`

  return {
    isYourTurn: isYourTurnFlag,
    teamLabel,
    actionLabel,
    phaseLabel,
    description
  }
}
