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
  getGenericDraftPredictions,
  getValueEstimate,
  getWinProbability,
  type AIDraftState,
  type AIRecommendation,
  type PlayerMAWPData,
} from '@/lib/draft/ai-inference'
import type { DraftData } from '@/lib/draft/types'

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

/** Generic Draft prediction for opponent turns */
interface GDPrediction {
  hero: string
  probability: number
}

interface AIRecommendationPanelProps {
  state: DraftState
  unavailable: Set<string>
  onSelect: (hero: string) => void
  currentStep: DraftStep | null
  draftData: DraftData | null
  availableBattletags: string[]
  /** Callback to report AI win probability to parent (for draft board display) */
  onValueEstimate?: (wp: number | null) => void
}

export function AIRecommendationPanel({
  state,
  unavailable,
  onSelect,
  currentStep,
  draftData,
  availableBattletags,
  onValueEstimate,
}: AIRecommendationPanelProps) {
  const [loading, setLoading] = useState(!isAILoaded())
  const [error, setError] = useState<string | null>(null)
  const [recommendations, setRecommendations] = useState<AIRecommendation[]>([])
  const [gdPredictions, setGdPredictions] = useState<GDPrediction[]>([])
  const [valueEstimate, setValueEstimate] = useState<number | null>(null)
  const [isOpponentTurn, setIsOpponentTurn] = useState(false)
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

  // Report value estimate to parent
  useEffect(() => {
    onValueEstimate?.(valueEstimate)
  }, [valueEstimate, onValueEstimate])

  // Run inference when state changes
  useEffect(() => {
    if (loading || error || !isAILoaded()) return
    if (state.currentStep >= DRAFT_SEQUENCE.length) return
    if (!state.map) return

    // Deduplicate — include map/tier/team so perspective changes recompute
    const stateKey = [
      state.currentStep,
      state.map,
      state.tier,
      state.ourTeam,
      JSON.stringify(state.selections),
      availableBattletags.join(','),
    ].join('|')
    const opponentTurn = currentStep?.team !== state.ourTeam
    setIsOpponentTurn(opponentTurn)
    if (stateKey === prevStateRef.current) return
    prevStateRef.current = stateKey

    const run = async () => {
      try {
        const aiState = convertToAIState(state)
        const taken = new Set(Object.values(state.selections))

        // Build player data for MAWP adjustments
        let playerData: PlayerMAWPData | undefined
        if (draftData?.playerStats && availableBattletags.length > 0) {
          playerData = {
            playerStats: draftData.playerStats,
            availableBattletags,
          }
        }

        if (opponentTurn) {
          // Opponent turn: use Generic Draft model to predict what they'll pick/ban
          const preds = await getGenericDraftPredictions(aiState, taken)
          setGdPredictions(preds)
          setRecommendations([])
        } else {
          // Our turn: use Policy model for recommendations
          const { recommendations: recs } = await getAIRecommendations(
            aiState, taken, currentStep?.team ?? 'A', playerData
          )
          setRecommendations(recs)
          setGdPredictions([])
        }

        // Always update WP using the enriched WP model (properly symmetrized)
        // instead of the policy value head (which has training artifacts)
        const wp = await getWinProbability(
          aiState.team0Picks, aiState.team1Picks,
          aiState.map, aiState.tier, draftData ?? undefined
        )
        // WP model returns P(team0 wins); convert to our team's perspective
        const ourWp = aiState.ourTeam === 0 ? wp : 1 - wp
        setValueEstimate(ourWp)
      } catch (err: any) {
        console.error('[AI] Inference error:', err)
      }
    }
    run()
  }, [state.currentStep, state.selections, state.map, state.tier, state.ourTeam, loading, error, draftData, availableBattletags, currentStep])

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

  // Determine which list to show
  const showOpponentPredictions = isOpponentTurn && gdPredictions.length > 0
  const showOurRecommendations = !isOpponentTurn && recommendations.length > 0

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-semibold text-violet-300">
          {isOpponentTurn
            ? `Enemy ${isBanPhase ? 'Ban' : 'Pick'} Predictions`
            : `AI ${isBanPhase ? 'Ban' : 'Pick'} Suggestions`}
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
        {isOpponentTurn
          ? 'Generic Draft model — predicted opponent choices'
          : 'HotsZero policy model'}
      </p>

      {!showOpponentPredictions && !showOurRecommendations ? (
        <p className="text-xs text-muted-foreground">
          Computing...
        </p>
      ) : showOpponentPredictions ? (
        <div className="space-y-1 max-h-[450px] overflow-y-auto pr-1">
          {gdPredictions
            .filter((p) => !unavailable.has(p.hero))
            .slice(0, 12)
            .map((pred, i) => {
              const role = getHeroRole(pred.hero)
              return (
                <button
                  key={pred.hero}
                  onClick={() => onSelect(pred.hero)}
                  className="w-full flex items-center gap-2 p-2 rounded-md border border-border text-left transition-colors hover:border-orange-500/60 hover:bg-orange-500/10"
                >
                  <span className="text-[10px] text-muted-foreground w-4 shrink-0">
                    {i + 1}
                  </span>
                  <div className="flex-1 min-w-0 flex items-center gap-1.5">
                    <span className="text-sm font-medium truncate">
                      {pred.hero}
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
                  <span className="text-xs font-mono text-orange-300 shrink-0">
                    {(pred.probability * 100).toFixed(1)}%
                  </span>
                </button>
              )
            })}
        </div>
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
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
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
                    {rec.suggestedPlayer && (
                      <span className="text-[9px] text-emerald-400 truncate block">
                        {rec.suggestedPlayer.split('#')[0]}
                        {rec.mawpAdj > 0 && (
                          <span className="text-emerald-500 ml-1">
                            +{(rec.mawpAdj * 100).toFixed(1)}%
                          </span>
                        )}
                      </span>
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
    ourTeam: state.ourTeam === 'A' ? 0 : 1,
  }
}
