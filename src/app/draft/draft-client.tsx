'use client'

import { useReducer, useMemo, useCallback } from 'react'
import { TierSelector } from '@/components/shared/tier-selector'
import { DraftBoard } from '@/components/draft/draft-board'
import { HeroPicker } from '@/components/draft/hero-picker'
import { RecommendationPanel } from '@/components/draft/recommendation-panel'
import { PlayerSlots } from '@/components/draft/player-slots'
import { generateRecommendations, expandChoGall } from '@/lib/draft/engine'
import { DRAFT_SEQUENCE } from '@/lib/draft/types'
import type { DraftState, DraftPhase, DraftData, Team } from '@/lib/draft/types'
import type { SkillTier } from '@/lib/types'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// State management
// ---------------------------------------------------------------------------

type DraftAction =
  | { type: 'SET_MAP'; map: string }
  | { type: 'SET_TIER'; tier: SkillTier }
  | { type: 'SET_TEAM'; team: Team }
  | { type: 'SET_PLAYER'; slotIndex: number; battletag: string | null }
  | { type: 'START_DRAFT' }
  | { type: 'SELECT_HERO'; hero: string }
  | { type: 'ASSIGN_PLAYER'; stepIndex: number; battletag: string }
  | { type: 'UNDO' }
  | { type: 'RESET' }

function createInitialState(): DraftState {
  return {
    phase: 'setup',
    map: null,
    tier: 'mid',
    ourTeam: 'A',
    currentStep: 0,
    selections: {},
    playerSlots: [
      { battletag: null },
      { battletag: null },
      { battletag: null },
      { battletag: null },
      { battletag: null },
    ],
    playerAssignments: {},
  }
}

function draftReducer(state: DraftState, action: DraftAction): DraftState {
  switch (action.type) {
    case 'SET_MAP':
      return { ...state, map: action.map }
    case 'SET_TIER':
      return { ...state, tier: action.tier }
    case 'SET_TEAM':
      return { ...state, ourTeam: action.team }
    case 'SET_PLAYER': {
      const slots = [...state.playerSlots]
      slots[action.slotIndex] = { battletag: action.battletag }
      return { ...state, playerSlots: slots }
    }
    case 'START_DRAFT':
      if (!state.map) return state
      return { ...state, phase: 'drafting' }
    case 'SELECT_HERO': {
      if (state.currentStep >= DRAFT_SEQUENCE.length) return state
      const currentDraftStep = DRAFT_SEQUENCE[state.currentStep]
      const newSelections = { ...state.selections, [state.currentStep]: action.hero }

      // Cho'gall auto-pair: picking Cho pre-fills Gall (and vice versa)
      // into the next pick slot for the same team. That slot will be
      // auto-skipped when the draft reaches it.
      if (currentDraftStep.type === 'pick' && (action.hero === 'Cho' || action.hero === 'Gall')) {
        const companion = action.hero === 'Cho' ? 'Gall' : 'Cho'
        for (let i = state.currentStep + 1; i < DRAFT_SEQUENCE.length; i++) {
          const s = DRAFT_SEQUENCE[i]
          if (s.type === 'pick' && s.team === currentDraftStep.team) {
            newSelections[i] = companion
            break
          }
        }
      }

      // Advance past any steps that are already pre-filled (Cho'gall companion)
      let nextStep = state.currentStep + 1
      while (nextStep < DRAFT_SEQUENCE.length && newSelections[nextStep]) {
        nextStep++
      }

      const phase: DraftPhase =
        nextStep >= DRAFT_SEQUENCE.length ? 'complete' : 'drafting'
      return {
        ...state,
        selections: newSelections,
        currentStep: nextStep,
        phase,
      }
    }
    case 'ASSIGN_PLAYER': {
      const newAssignments = { ...state.playerAssignments, [action.stepIndex]: action.battletag }
      return { ...state, playerAssignments: newAssignments }
    }
    case 'UNDO': {
      if (state.currentStep === 0) return state
      const newSelections = { ...state.selections }
      const newAssignments = { ...state.playerAssignments }
      let prevStep = state.currentStep - 1

      // Clear the step we're undoing
      const undoneHero = newSelections[prevStep]
      delete newSelections[prevStep]
      delete newAssignments[prevStep]

      // Cho'gall undo: if the undone hero is Cho or Gall, also clear
      // the auto-filled companion (which could be ahead or behind).
      if (undoneHero === 'Cho' || undoneHero === 'Gall') {
        const companion = undoneHero === 'Cho' ? 'Gall' : 'Cho'
        const team = DRAFT_SEQUENCE[prevStep]?.team
        // Check ahead for auto-filled companion
        for (let i = prevStep + 1; i < DRAFT_SEQUENCE.length; i++) {
          if (newSelections[i] === companion && DRAFT_SEQUENCE[i]?.team === team
              && DRAFT_SEQUENCE[i]?.type === 'pick') {
            delete newSelections[i]
            delete newAssignments[i]
            break
          }
        }
        // Check behind — maybe we're at the auto-filled step, undo source too
        for (let i = prevStep - 1; i >= 0; i--) {
          if (newSelections[i] === companion && DRAFT_SEQUENCE[i]?.team === team
              && DRAFT_SEQUENCE[i]?.type === 'pick') {
            delete newSelections[i]
            delete newAssignments[i]
            prevStep = i
            break
          }
        }
      }

      return {
        ...state,
        selections: newSelections,
        playerAssignments: newAssignments,
        currentStep: prevStep,
        phase: 'drafting',
      }
    }
    case 'RESET':
      return createInitialState()
    default:
      return state
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface DraftClientProps {
  dataByTier: Record<SkillTier, DraftData>
  maps: string[]
  registeredBattletags: string[]
}

export function DraftClient({
  dataByTier,
  maps,
  registeredBattletags,
}: DraftClientProps) {
  const [state, dispatch] = useReducer(draftReducer, undefined, createInitialState)

  // Resolve draft data for current tier
  const draftData = useMemo(() => {
    if (!state.map) return null
    return dataByTier[state.tier] ?? null
  }, [dataByTier, state.tier, state.map])

  // Generate recommendations
  const recommendations = useMemo(() => {
    if (!draftData || state.phase !== 'drafting') return []
    return generateRecommendations(state, draftData)
  }, [state, draftData])

  // Heroes that are already selected (banned or picked)
  // Cho'gall: if either Cho or Gall is selected, both are unavailable
  // Also block Cho/Gall from being picked when our team has <2 picks remaining
  const unavailableHeroes = useMemo(() => {
    const set = expandChoGall(new Set(Object.values(state.selections)))

    // If it's our pick turn with <2 picks remaining, Cho'gall is invalid
    if (state.phase === 'drafting' && state.currentStep < DRAFT_SEQUENCE.length) {
      const step = DRAFT_SEQUENCE[state.currentStep]
      if (step.type === 'pick' && step.team === state.ourTeam) {
        const ourPickCount = Object.entries(state.selections).filter(([idx]) => {
          const s = DRAFT_SEQUENCE[Number(idx)]
          return s.type === 'pick' && s.team === state.ourTeam
        }).length
        const totalOurPicks = DRAFT_SEQUENCE.filter(
          (s) => s.type === 'pick' && s.team === state.ourTeam
        ).length
        if (totalOurPicks - ourPickCount < 2) {
          set.add('Cho')
          set.add('Gall')
        }
      }
    }

    return set
  }, [state.selections, state.phase, state.currentStep, state.ourTeam])

  const handleSelectHero = useCallback(
    (hero: string) => dispatch({ type: 'SELECT_HERO', hero }),
    []
  )

  // Battletags that haven't been assigned to a pick yet
  const availableBattletags = useMemo(() => {
    const assigned = new Set(Object.values(state.playerAssignments))
    return state.playerSlots
      .map((s) => s.battletag)
      .filter((bt): bt is string => bt !== null && !assigned.has(bt))
  }, [state.playerSlots, state.playerAssignments])

  const currentStep =
    state.currentStep < DRAFT_SEQUENCE.length
      ? DRAFT_SEQUENCE[state.currentStep]
      : null

  // ---------------------------------------------------------------------------
  // Setup phase
  // ---------------------------------------------------------------------------
  if (state.phase === 'setup') {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-white">Draft Assistant</h1>
          <p className="text-muted-foreground mt-1">
            Configure your draft, then start picking
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl">
          {/* Map selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-muted-foreground">
              Map
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              {maps.map((map) => (
                <button
                  key={map}
                  onClick={() => dispatch({ type: 'SET_MAP', map })}
                  className={cn(
                    'px-3 py-2 rounded-md text-sm text-left transition-colors border',
                    state.map === map
                      ? 'border-primary bg-primary/10 text-primary font-medium'
                      : 'border-border text-muted-foreground hover:text-foreground hover:bg-accent'
                  )}
                >
                  {map}
                </button>
              ))}
            </div>
          </div>

          {/* Right column */}
          <div className="space-y-6">
            {/* Tier */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-muted-foreground">
                Skill Tier
              </label>
              <TierSelector
                value={state.tier}
                onChange={(tier) => dispatch({ type: 'SET_TIER', tier })}
              />
            </div>

            {/* Team side */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-muted-foreground">
                Your Team (banning first?)
              </label>
              <div className="flex gap-2">
                {(['A', 'B'] as Team[]).map((team) => (
                  <button
                    key={team}
                    onClick={() => dispatch({ type: 'SET_TEAM', team })}
                    className={cn(
                      'flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors border',
                      state.ourTeam === team
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-border text-muted-foreground hover:text-foreground hover:bg-accent'
                    )}
                  >
                    Team {team} {team === 'A' ? '(Bans first)' : '(Bans second)'}
                  </button>
                ))}
              </div>
            </div>

            {/* Player slots */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-muted-foreground">
                Team Players (optional)
              </label>
              <p className="text-xs text-muted-foreground">
                Assign registered battletags for personalized recommendations
              </p>
              <PlayerSlots
                slots={state.playerSlots}
                registeredBattletags={registeredBattletags}
                onSetPlayer={(idx, bt) =>
                  dispatch({ type: 'SET_PLAYER', slotIndex: idx, battletag: bt })
                }
              />
            </div>
          </div>
        </div>

        {/* Start button */}
        <button
          onClick={() => dispatch({ type: 'START_DRAFT' })}
          disabled={!state.map}
          className={cn(
            'px-6 py-3 rounded-lg text-sm font-semibold transition-colors',
            state.map
              ? 'bg-primary text-primary-foreground hover:bg-primary/90'
              : 'bg-muted text-muted-foreground cursor-not-allowed'
          )}
        >
          Start Draft
        </button>
      </div>
    )
  }

  // ---------------------------------------------------------------------------
  // Drafting / Complete phase
  // ---------------------------------------------------------------------------
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">
            Draft &mdash; {state.map}
          </h1>
          <p className="text-muted-foreground text-sm">
            {state.phase === 'complete'
              ? 'Draft complete'
              : currentStep
                ? `${currentStep.team === state.ourTeam ? 'Your' : 'Enemy'} ${currentStep.type === 'ban' ? 'Ban' : 'Pick'}`
                : ''}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => dispatch({ type: 'UNDO' })}
            disabled={state.currentStep === 0}
            className={cn(
              'px-3 py-1.5 rounded-md text-xs font-medium border transition-colors',
              state.currentStep > 0
                ? 'border-border text-muted-foreground hover:text-foreground hover:bg-accent'
                : 'border-border/50 text-muted-foreground/50 cursor-not-allowed'
            )}
          >
            Undo
          </button>
          <button
            onClick={() => dispatch({ type: 'RESET' })}
            className="px-3 py-1.5 rounded-md text-xs font-medium border border-gaming-danger/50 text-gaming-danger hover:bg-gaming-danger/10 transition-colors"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Draft board */}
      <DraftBoard
        state={state}
        currentStep={state.currentStep}
        availableBattletags={availableBattletags}
        playerAssignments={state.playerAssignments}
        onAssignPlayer={(stepIdx, bt) =>
          dispatch({ type: 'ASSIGN_PLAYER', stepIndex: stepIdx, battletag: bt })
        }
      />

      {/* Main drafting area */}
      {state.phase !== 'complete' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Hero picker — 2/3 */}
          <div className="lg:col-span-2">
            <HeroPicker
              unavailable={unavailableHeroes}
              onSelect={handleSelectHero}
              currentStepType={currentStep?.type ?? 'pick'}
              isOurTurn={currentStep?.team === state.ourTeam}
            />
          </div>

          {/* Recommendations — 1/3 */}
          <div>
            <RecommendationPanel
              recommendations={recommendations}
              isBanPhase={currentStep?.type === 'ban'}
              isOurTurn={currentStep?.team === state.ourTeam}
              onSelect={handleSelectHero}
              unavailable={unavailableHeroes}
            />
          </div>
        </div>
      )}

      {/* Complete summary */}
      {state.phase === 'complete' && (
        <div className="rounded-lg border p-6 text-center space-y-3">
          <p className="text-lg font-semibold text-white">
            Draft Complete
          </p>
          <p className="text-sm text-muted-foreground">
            Good luck in your game!
          </p>
          <button
            onClick={() => dispatch({ type: 'RESET' })}
            className="px-6 py-2 rounded-md text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            New Draft
          </button>
        </div>
      )}
    </div>
  )
}
