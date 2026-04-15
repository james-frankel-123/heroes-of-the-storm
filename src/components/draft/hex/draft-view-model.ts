/**
 * Turn a DraftState into a team-keyed view model for the HotS-style layout.
 *
 * Each team has 5 pick slots and 3 ban slots, in the order they occur in
 * DRAFT_SEQUENCE. The view model answers, for each slot: {hero, stepIndex,
 * isCurrent, isSkipped}. `isSkipped` covers the Cho'gall auto-pair case where
 * a pick slot is consumed by the partner hero without a user action.
 */

import { DRAFT_SEQUENCE, type DraftState, type Team } from '@/lib/draft/types'

export interface SlotView {
  hero: string | null
  stepIndex: number
  isCurrent: boolean
  isSkipped: boolean
}

export interface DraftView {
  bansA: SlotView[]
  bansB: SlotView[]
  picksA: SlotView[]
  picksB: SlotView[]
}

export function buildDraftView(state: DraftState): DraftView {
  const bansA: SlotView[] = []
  const bansB: SlotView[] = []
  const picksA: SlotView[] = []
  const picksB: SlotView[] = []

  for (let i = 0; i < DRAFT_SEQUENCE.length; i++) {
    const step = DRAFT_SEQUENCE[i]
    const hero = state.selections[i] ?? null
    const slot: SlotView = {
      hero,
      stepIndex: i,
      isCurrent: i === state.currentStep,
      isSkipped: false,
    }
    const arr = step.type === 'ban'
      ? (step.team === 'A' ? bansA : bansB)
      : (step.team === 'A' ? picksA : picksB)
    arr.push(slot)
  }

  return { bansA, bansB, picksA, picksB }
}

export function teamFor(ourTeam: Team, which: 'ours' | 'enemy'): Team {
  if (which === 'ours') return ourTeam
  return ourTeam === 'A' ? 'B' : 'A'
}
