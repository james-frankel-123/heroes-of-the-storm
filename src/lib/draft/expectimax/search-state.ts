/**
 * Lightweight draft state for expectimax tree search.
 *
 * Optimized for fast cloning, hashing, and hero availability checks.
 * Converts from the UI DraftState at the search boundary.
 */

import { DRAFT_SEQUENCE } from '../types'
import type { DraftState } from '../types'
import { expandChoGall, consecutivePicksRemaining } from '../engine'
import { HERO_ROLES } from '@/lib/data/hero-roles'
import type { SearchState, Team } from './types'

/**
 * Convert the UI DraftState into a compact SearchState for tree search.
 */
export function createSearchState(state: DraftState): SearchState {
  const ourPicks: string[] = []
  const enemyPicks: string[] = []
  const bans: string[] = []
  const taken = new Set<string>()

  for (let i = 0; i < state.currentStep; i++) {
    const hero = state.selections[i]
    if (!hero) continue
    const step = DRAFT_SEQUENCE[i]
    taken.add(hero)

    if (step.type === 'ban') {
      bans.push(hero)
    } else if (step.team === state.ourTeam) {
      ourPicks.push(hero)
    } else {
      enemyPicks.push(hero)
    }
  }

  // Cho/Gall expansion
  if (taken.has('Cho') || taken.has('Gall')) {
    taken.add('Cho')
    taken.add('Gall')
  }

  return {
    ourPicks,
    enemyPicks,
    bans,
    taken,
    step: state.currentStep,
    map: state.map ?? '',
    tier: state.tier,
    ourTeam: state.ourTeam,
  }
}

/**
 * Clone a SearchState and apply a hero action at the current step.
 * Returns a new SearchState with the hero applied and step advanced.
 * Handles Cho/Gall auto-pairing (consumes 2 pick slots).
 */
export function cloneAndApply(state: SearchState, hero: string): SearchState {
  const step = DRAFT_SEQUENCE[state.step]
  if (!step) throw new Error(`Invalid step ${state.step}`)

  const newState: SearchState = {
    ourPicks: [...state.ourPicks],
    enemyPicks: [...state.enemyPicks],
    bans: [...state.bans],
    taken: new Set(state.taken),
    step: state.step + 1,
    map: state.map,
    tier: state.tier,
    ourTeam: state.ourTeam,
  }

  newState.taken.add(hero)

  if (step.type === 'ban') {
    newState.bans.push(hero)
  } else if (step.team === state.ourTeam) {
    newState.ourPicks.push(hero)
  } else {
    newState.enemyPicks.push(hero)
  }

  // Cho/Gall auto-pairing: if Cho or Gall is picked, auto-fill the partner
  if (step.type === 'pick' && (hero === 'Cho' || hero === 'Gall')) {
    const partner = hero === 'Cho' ? 'Gall' : 'Cho'
    newState.taken.add(partner)

    // The partner fills the next pick slot for the same team
    const nextStep = DRAFT_SEQUENCE[newState.step]
    if (nextStep && nextStep.type === 'pick' && nextStep.team === step.team) {
      if (step.team === state.ourTeam) {
        newState.ourPicks.push(partner)
      } else {
        newState.enemyPicks.push(partner)
      }
      newState.step++ // consume the extra slot
    }
  }

  return newState
}

/**
 * Compute a hash key for the transposition table.
 * Two states with the same picks, bans, and step produce the same hash
 * regardless of the order heroes were picked within a phase.
 */
export function hashState(state: SearchState): string {
  const op = [...state.ourPicks].sort().join(',')
  const ep = [...state.enemyPicks].sort().join(',')
  const b = [...state.bans].sort().join(',')
  return `${op}|${ep}|${b}|${state.step}`
}

/**
 * Get all valid heroes that can be picked/banned at the current step.
 * Excludes taken heroes and handles Cho/Gall consecutive-pick requirement.
 */
export function getValidHeroes(state: SearchState): string[] {
  const allHeroes = Object.keys(HERO_ROLES)
  let valid = allHeroes.filter(h => !state.taken.has(h))

  // Cho/Gall requires 2 consecutive pick slots for the current team
  const currentStep = DRAFT_SEQUENCE[state.step]
  if (currentStep && currentStep.type === 'pick') {
    // Build a minimal selections record for consecutivePicksRemaining
    const selections: Record<number, string> = {}
    // Mark all steps before current as filled
    for (let i = 0; i < state.step; i++) {
      selections[i] = '_filled_' // exact hero doesn't matter, just that it's filled
    }
    const remaining = consecutivePicksRemaining(
      state.step, currentStep.team, selections
    )
    if (remaining < 2) {
      valid = valid.filter(h => h !== 'Cho' && h !== 'Gall')
    }
  }

  return valid
}

/** Is the draft complete? */
export function isTerminal(state: SearchState): boolean {
  return state.step >= DRAFT_SEQUENCE.length
}

/** Is it our team's turn at the current step? */
export function isOurTurn(state: SearchState): boolean {
  const step = DRAFT_SEQUENCE[state.step]
  return step ? step.team === state.ourTeam : false
}

/** Is the current step a ban phase? */
export function isBanPhase(state: SearchState): boolean {
  const step = DRAFT_SEQUENCE[state.step]
  return step ? step.type === 'ban' : false
}

/** Get the current team at this step ('A' or 'B') */
export function currentTeam(state: SearchState): Team {
  const step = DRAFT_SEQUENCE[state.step]
  return step ? step.team as Team : 'A'
}
