/**
 * Draft assistant types.
 *
 * Draft sequence (from spec): 16 steps total (6 bans + 10 picks)
 *   Phase 1 — 4 bans:  A ban, B ban, A ban, B ban
 *   Phase 2 — 5 picks: A pick 1, B pick 2, A pick 2
 *   Phase 3 — 2 bans:  B ban, A ban
 *   Phase 4 — 5 picks: B pick 2, A pick 2, B pick 1
 */

import type { SkillTier } from '@/lib/types'

export type Team = 'A' | 'B'
export type StepType = 'ban' | 'pick'

export interface DraftStep {
  team: Team
  type: StepType
  /** How many heroes this step selects (always 1 for bans, 1 or 2 for picks) */
  count: number
  /** Label for display */
  label: string
}

/**
 * The full 16-step draft sequence.
 * "count" means how many heroes fill this phase — the UI handles them as
 * individual selections but they belong to the same phase.
 */
export const DRAFT_SEQUENCE: DraftStep[] = [
  // Phase 1: Opening bans (4)
  { team: 'A', type: 'ban', count: 1, label: 'Ban 1' },
  { team: 'B', type: 'ban', count: 1, label: 'Ban 1' },
  { team: 'A', type: 'ban', count: 1, label: 'Ban 2' },
  { team: 'B', type: 'ban', count: 1, label: 'Ban 2' },
  // Phase 2: First picks (5)
  { team: 'A', type: 'pick', count: 1, label: 'Pick 1' },
  { team: 'B', type: 'pick', count: 1, label: 'Pick 1' },
  { team: 'B', type: 'pick', count: 1, label: 'Pick 2' },
  { team: 'A', type: 'pick', count: 1, label: 'Pick 2' },
  { team: 'A', type: 'pick', count: 1, label: 'Pick 3' },
  // Phase 3: Mid bans (2)
  { team: 'B', type: 'ban', count: 1, label: 'Ban 3' },
  { team: 'A', type: 'ban', count: 1, label: 'Ban 3' },
  // Phase 4: Final picks (5)
  { team: 'B', type: 'pick', count: 1, label: 'Pick 3' },
  { team: 'B', type: 'pick', count: 1, label: 'Pick 4' },
  { team: 'A', type: 'pick', count: 1, label: 'Pick 4' },
  { team: 'A', type: 'pick', count: 1, label: 'Pick 5' },
  { team: 'B', type: 'pick', count: 1, label: 'Pick 5' },
]

export interface PlayerSlot {
  /** null = no battletag assigned (use generic stats) */
  battletag: string | null
}

export type DraftPhase = 'setup' | 'drafting' | 'complete'

export interface DraftState {
  phase: DraftPhase
  map: string | null
  tier: SkillTier
  /** Which team is "our" team */
  ourTeam: Team
  /** Current step index into DRAFT_SEQUENCE */
  currentStep: number
  /** All selections made so far: stepIndex → hero name */
  selections: Record<number, string>
  /** Player slots for our team (up to 5) */
  playerSlots: PlayerSlot[]
  /** Which battletag drafted each of our picks: stepIndex → battletag */
  playerAssignments: Record<number, string>
}

export interface RecommendationReason {
  type:
    | 'hero_wr'         // Hero base win rate delta from 50%
    | 'counter'         // Counter-pick delta vs enemy hero
    | 'synergy'         // Synergy delta with ally hero
    | 'role_need'       // Fills a needed role (bonus)
    | 'role_penalty'    // Bad composition (duplicate healer/tank, etc.)
    | 'player_strong'   // A player on the team is strong with this hero
    | 'ban_worthy'      // High ban/win rate (for ban suggestions)
  label: string
  /** Win rate delta in percentage points (e.g. +3.2 or -2.0) */
  delta: number
}

export interface DraftRecommendation {
  hero: string
  /** Net expected win rate delta from 50% baseline (sum of all deltas) */
  netDelta: number
  reasons: RecommendationReason[]
  /** If a specific player should play this hero */
  suggestedPlayer: string | null
}

/**
 * All precomputed data needed for the draft engine.
 * Loaded once by the server component, passed to client.
 */
export interface DraftData {
  /** hero → { winRate, pickRate, banRate, games } per tier */
  heroStats: Record<string, {
    winRate: number
    pickRate: number
    banRate: number
    games: number
  }>
  /** hero → winRate on selected map at selected tier */
  heroMapWinRates: Record<string, { winRate: number; games: number }>
  /** heroA → heroB → { winRate, games } for 'with' relationship */
  synergies: Record<string, Record<string, { winRate: number; games: number }>>
  /** heroA → heroB → { winRate, games } for 'against' relationship */
  counters: Record<string, Record<string, { winRate: number; games: number }>>
  /** battletag → hero → player stats */
  playerStats: Record<string, Record<string, {
    games: number
    wins: number
    winRate: number
    mawp: number | null
  }>>
  /** battletag → hero → { winRate, games } on the selected map */
  playerMapStats: Record<string, Record<string, {
    winRate: number
    games: number
  }>>
}
