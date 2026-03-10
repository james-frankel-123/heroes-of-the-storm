'use client'

import { useState, useEffect, useRef } from 'react'
import { Badge } from '@/components/ui/badge'
import { getHeroRole } from '@/lib/data/hero-roles'
import { cn } from '@/lib/utils'
import type { DraftState, DraftStep } from '@/lib/draft/types'
import { DRAFT_SEQUENCE } from '@/lib/draft/types'
import {
  loadAIModels,
  isAILoaded,
  getAIRecommendations,
  type AIDraftState,
  type AIRecommendation,
} from '@/lib/draft/ai-inference'

function roleBadgeVariant(role: string | null) {
  switch (role) {
    case 'Tank': return 'tank' as const
    case 'Bruiser': return 'bruiser' as const
    case 'Healer': return 'healer' as const
    case 'Ranged Assassin': return 'ranged' as const
    case 'Melee Assassin': return 'melee' as const
    case 'Support': return 'support' as const
    default: return 'secondary' as const
  }
}

interface AIRecommendationPanelProps {
  state: DraftState
  unavailable: Set<string>
  onSelect: (hero: string) => void
  currentStep: DraftStep | null
}

export function AIRecommendationPanel({
  state,
  unavailable,
  onSelect,
  currentStep,
}: AIRecommendationPanelProps) {
  const [loading, setLoading] = useState(!isAILoaded())
  const [error, setError] = useState<string | null>(null)
  const [recommendations, setRecommendations] = useState<AIRecommendation[]>([])
  const [valueEstimate, setValueEstimate] = useState<number | null>(null)
  const prevStateRef = useRef<string>('')

  // Load models on mount
  useEffect(() => {
    if (isAILoaded()) {
      setLoading(false)
      return
    }
    loadAIModels()
      .then(() => setLoading(false))
      .catch((err) => {
        setError(`Failed to load AI models: ${err.message}`)
        setLoading(false)
      })
  }, [])

  // Run inference when state changes
  useEffect(() => {
    if (loading || error || !isAILoaded()) return
    if (state.currentStep >= DRAFT_SEQUENCE.length) return
    if (!state.map) return

    // Deduplicate
    const stateKey = `${state.currentStep}-${JSON.stringify(state.selections)}`
    if (stateKey === prevStateRef.current) return
    prevStateRef.current = stateKey

    const run = async () => {
      try {
        const aiState = convertToAIState(state)
        const taken = new Set(Object.values(state.selections))
        const { recommendations: recs, valueEstimate: ve } = await getAIRecommendations(
          aiState, taken
        )
        setRecommendations(recs)
        setValueEstimate(ve)
      } catch (err: any) {
        console.error('[AI] Inference error:', err)
      }
    }
    run()
  }, [state.currentStep, state.selections, state.map, state.tier, loading, error])

  if (loading) {
    return (
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-violet-300">
          AI Recommendations
        </h3>
        <div className="flex items-center gap-2 p-4 border border-violet-500/30 rounded-md bg-violet-500/5">
          <div className="animate-spin h-4 w-4 border-2 border-violet-400 border-t-transparent rounded-full" />
          <span className="text-xs text-violet-300">Loading AI models...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-violet-300">
          AI Recommendations
        </h3>
        <p className="text-xs text-gaming-danger p-3 border border-gaming-danger/30 rounded-md">
          {error}
        </p>
      </div>
    )
  }

  const isBanPhase = currentStep?.type === 'ban'
  const isOurTurn = currentStep?.team === state.ourTeam

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-semibold text-violet-300">
          AI {isBanPhase ? 'Ban' : 'Pick'} Suggestions
        </h3>
        {valueEstimate !== null && (
          <span className={cn(
            'text-[10px] font-bold px-1.5 py-0.5 rounded border tabular-nums',
            valueEstimate > 0.53
              ? 'text-gaming-success bg-gaming-success/10 border-gaming-success/30'
              : valueEstimate < 0.47
                ? 'text-gaming-danger bg-gaming-danger/10 border-gaming-danger/30'
                : 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30'
          )}>
            WP: {(valueEstimate * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <p className="text-[10px] text-muted-foreground">
        Preliminary model — training with more data
      </p>

      {recommendations.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          Computing recommendations...
        </p>
      ) : (
        <div className="space-y-1 max-h-[450px] overflow-y-auto pr-1">
          {recommendations
            .filter((r) => !unavailable.has(r.hero))
            .slice(0, 12)
            .map((rec, i) => {
              const role = getHeroRole(rec.hero)
              return (
                <button
                  key={rec.hero}
                  onClick={() => onSelect(rec.hero)}
                  className={cn(
                    'w-full flex items-center gap-2 p-2 rounded-md border text-left transition-colors',
                    isBanPhase
                      ? 'border-border hover:border-gaming-danger/60 hover:bg-gaming-danger/10'
                      : 'border-border hover:border-violet-500/60 hover:bg-violet-500/10'
                  )}
                >
                  <span className="text-[10px] text-muted-foreground w-4 shrink-0">
                    {i + 1}
                  </span>
                  <div className="flex-1 min-w-0 flex items-center gap-1.5">
                    <span className="text-sm font-medium truncate">
                      {rec.hero}
                    </span>
                    {role && (
                      <Badge
                        variant={roleBadgeVariant(role)}
                        className="text-[7px] px-1 py-0 shrink-0"
                      >
                        {role.split(' ')[0]}
                      </Badge>
                    )}
                  </div>
                  <span className="text-xs font-mono text-violet-300 shrink-0">
                    {(rec.prior * 100).toFixed(1)}%
                  </span>
                </button>
              )
            })}
        </div>
      )}
    </div>
  )
}

/**
 * Convert the UI DraftState to the AI's expected format.
 * Maps Team A/B to team 0/1 based on which team the user selected as "ours".
 */
function convertToAIState(state: DraftState): AIDraftState {
  // Team A = team that bans first. In the AI model, team 0 bans first.
  // So Team A → team 0, Team B → team 1 (regardless of which is "ours").
  const team0Picks: string[] = []
  const team1Picks: string[] = []
  const bans: string[] = []

  for (let i = 0; i < state.currentStep; i++) {
    const step = DRAFT_SEQUENCE[i]
    const hero = state.selections[i]
    if (!hero) continue

    if (step.type === 'ban') {
      bans.push(hero)
    } else if (step.team === 'A') {
      team0Picks.push(hero)
    } else {
      team1Picks.push(hero)
    }
  }

  const currentDraftStep = state.currentStep < DRAFT_SEQUENCE.length
    ? DRAFT_SEQUENCE[state.currentStep]
    : null

  return {
    team0Picks,
    team1Picks,
    bans,
    map: state.map || 'Cursed Hollow',
    tier: state.tier,
    step: state.currentStep,
    stepType: currentDraftStep?.type || 'pick',
  }
}
