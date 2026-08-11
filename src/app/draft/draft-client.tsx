'use client'

import { useReducer, useMemo, useCallback, useState, useEffect } from 'react'
import { computeTeamWinEstimate } from '@/lib/draft/win-estimate'
import { TierSelector } from '@/components/shared/tier-selector'
import { DraftBoard } from '@/components/draft/draft-board'
import { BanBar } from '@/components/draft/hex/BanBar'
import { TeamColumn } from '@/components/draft/hex/TeamColumn'
import { buildDraftView } from '@/components/draft/hex/draft-view-model'
import { HeroPicker } from '@/components/draft/hero-picker'
import { SearchRecommendationPanel, type OpponentPredictionRow, type OurTurnRow } from '@/components/draft/search-recommendation-panel'
import { loadAIModels, getGenericDraftPredictions, getAIRecommendations, getPartialProjection } from '@/lib/draft/ai-inference'
import { PlayerSlots } from '@/components/draft/player-slots'
import { generateRecommendations, expandChoGall, consecutivePicksRemaining } from '@/lib/draft/engine'
import { DRAFT_SEQUENCE } from '@/lib/draft/types'
import type { DraftState, DraftPhase, DraftData, Team } from '@/lib/draft/types'
import type { SkillTier } from '@/lib/types'
import { cn } from '@/lib/utils'
import { mapImageSrc } from '@/lib/data/map-images'
import { AMADrawer } from '@/components/draft/ama-drawer'
import { enrichDraftContext } from '@/lib/ama/context'

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
  | { type: 'SKIP_BAN' }
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
    case 'SKIP_BAN': {
      if (state.currentStep >= DRAFT_SEQUENCE.length) return state
      const step = DRAFT_SEQUENCE[state.currentStep]
      if (step.type !== 'ban') return state // only allow skipping bans
      return {
        ...state,
        currentStep: state.currentStep + 1,
      }
    }
    case 'ASSIGN_PLAYER': {
      const newAssignments = { ...state.playerAssignments }
      if (action.battletag) {
        newAssignments[action.stepIndex] = action.battletag
      } else {
        delete newAssignments[action.stepIndex]
      }
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

/**
 * Our team's win % as displayed in the banner: the two teams' raw estimates
 * normalized to sum to 100 (falling back to a single-sided estimate when only
 * one team has picks). Used to quantify the impact of hypothetical picks.
 */
function displayedOurWinPct(
  ourPicks: string[],
  enemyPicks: string[],
  data: DraftData,
  map: string | null,
  ourPlayerMap: Record<number, string>,
): number {
  const ourRaw = ourPicks.length > 0
    ? computeTeamWinEstimate(ourPicks, enemyPicks, data, map, ourPlayerMap).winPct
    : null
  const enemyRaw = enemyPicks.length > 0
    ? computeTeamWinEstimate(enemyPicks, ourPicks, data, map).winPct
    : null
  if (ourRaw !== null && enemyRaw !== null) return ourRaw / (ourRaw + enemyRaw) * 100
  if (ourRaw !== null) return ourRaw
  if (enemyRaw !== null) return 100 - enemyRaw
  return 50
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

  // AMA drawer state
  const [amaOpen, setAmaOpen] = useState(false)

  // MCTS search state (runs on main thread)
  const [searchResults, setSearchResults] = useState<OurTurnRow[]>([])
  const [searchInfo, setSearchInfo] = useState<string | null>(null)
  const [searching, setSearching] = useState(false)
  const [searchStatus, setSearchStatus] = useState<string>('')
  const [gdOpponentPreds, setGdOpponentPreds] = useState<OpponentPredictionRow[]>([])
  const [gdLoading, setGdLoading] = useState(false)

  // Build AI state helper
  const buildAIState = useCallback(() => {
    const step = state.currentStep < 16 ? DRAFT_SEQUENCE[state.currentStep] : null
    const aiState: import('@/lib/draft/ai-inference').AIDraftState = {
      team0Picks: [], team1Picks: [], bans: [],
      map: state.map ?? '', tier: state.tier,
      step: state.currentStep,
      stepType: (step?.type ?? 'pick') as 'ban' | 'pick',
      ourTeam: state.ourTeam === 'A' ? 0 : 1,
    }
    for (let i = 0; i < state.currentStep; i++) {
      const s = DRAFT_SEQUENCE[i]
      const hero = state.selections[i]
      if (!hero) continue
      if (s.type === 'ban') aiState.bans.push(hero)
      else if (s.team === 'A') aiState.team0Picks.push(hero)
      else aiState.team1Picks.push(hero)
    }
    return aiState
  }, [state.currentStep, state.selections, state.map, state.tier, state.ourTeam])

  // Run expectimax on our turns, GD predictions on opponent turns
  useEffect(() => {
    if (!draftData || state.phase !== 'drafting') return
    const step = state.currentStep < 16 ? DRAFT_SEQUENCE[state.currentStep] : null
    if (!step) return

    let cancelled = false
    const isOurs = step.team === state.ourTeam

    if (!isOurs) {
      // Opponent turn: show GD predictions + their expected impact on our win %
      setSearchResults([])
      setSearching(false)
      setGdLoading(true)
      setSearchStatus('Loading predictions...')
      ;(async () => {
        try {
          await loadAIModels()
          if (cancelled) return
          const preds = await getGenericDraftPredictions(buildAIState(), unavailableHeroes, 12)
          if (!cancelled) {
            const { ourPicks, enemyPicks, ourPlayerMap } = pickArrays
            const isPick = step.type === 'pick'
            const before = isPick
              ? displayedOurWinPct(ourPicks, enemyPicks, draftData, state.map, ourPlayerMap)
              : 0
            // Pad the model's predictions with statistically strong options
            const gdHeroes = new Set(preds.map(p => p.hero))
            const candidates: { hero: string; probability: number | null }[] = [
              ...preds.map(p => ({ hero: p.hero, probability: p.probability as number | null })),
              ...recommendations
                .filter(r => !gdHeroes.has(r.hero) && !unavailableHeroes.has(r.hero))
                .map(r => ({ hero: r.hero, probability: null })),
            ].slice(0, 10)
            setGdOpponentPreds(candidates.map(({ hero, probability }) => ({
              hero,
              probabilityPct: probability != null ? probability * 100 : null,
              impactPp: isPick
                ? Math.round((displayedOurWinPct(ourPicks, [...enemyPicks, hero], draftData, state.map, ourPlayerMap) - before) * 10) / 10
                : null,
            })))
            setGdLoading(false)
            setSearchStatus('')
          }
        } catch {
          if (!cancelled) { setGdLoading(false); setSearchStatus('Prediction failed') }
        }
      })()
      return () => { cancelled = true }
    }

    // Our turn: run MCTS (policy priors/values + GD opponent sampling + WP
    // terminal evaluation) on the main thread.
    setSearchResults([])
    setSearchInfo(null)
    setSearching(true)
    setGdOpponentPreds([])
    setSearchStatus('Loading AI models...')

    ;(async () => {
      try {
        await loadAIModels()
        if (cancelled) return

        setSearchStatus('Searching (MCTS)...')
        const playerData = draftData.playerStats && availableBattletags.length > 0
          ? { playerStats: draftData.playerStats, availableBattletags }
          : undefined
        const { recommendations: mctsRecs, sims } = await getAIRecommendations(
          buildAIState(), unavailableHeroes, step.team, playerData, 10, draftData,
        )
        if (cancelled) return

        // Projections come from the partial-draft WP model (the neutral
        // judge, same scale and feature family as the final evaluation), NOT
        // from search Q: instrumentation on 2,000 real drafts (2026-08-07)
        // showed the policy value head's level is a state-insensitive
        // constant, which made Q-based projections sit near 80% all draft
        // and cliff at the end. MCTS still chooses WHICH candidates make the
        // shortlist and carries the personalization (MAWP) nudges, applied
        // as a delta (winProb - q) on top of the neutral projection. Pick
        // rows project the state after WE take the hero; ban rows project
        // the state if THE ENEMY gets the hero (a ban's greedy value is the
        // threat it denies), so for bans LOWER means "ban this first" and
        // the list sorts ascending.
        const aiState = buildAIState()
        const ourIs0 = aiState.ourTeam === 0
        const [our, enemy] = ourIs0
          ? [aiState.team0Picks, aiState.team1Picks]
          : [aiState.team1Picks, aiState.team0Picks]
        const projFor = async (t0: string[], t1: string[], stepIdx: number) => {
          const p = await getPartialProjection(t0, t1, aiState.map, aiState.tier, stepIdx, draftData)
          return (ourIs0 ? p : 1 - p) * 100
        }
        // Training convention (train_partial_wp): a state's step index is the
        // pick_number of the action just INCLUDED in it. A candidate state
        // includes the action at aiState.step; the current state's last
        // included action is aiState.step - 1.
        const nextStep = Math.min(aiState.step, 15)
        const projNow = await projFor(aiState.team0Picks, aiState.team1Picks, Math.max(aiState.step - 1, 0))
        const isBan = aiState.stepType === 'ban'
        const mctsAdj = new Map(mctsRecs.map(r => [r.hero, r.mawpAdj * 100]))
        // Candidate state in ABSOLUTE team order: on a pick the hero joins
        // OUR team; on a ban the projection is the threat state where the
        // ENEMY has taken the hero.
        const candState = (hero: string): [string[], string[]] => {
          const heroToTeam0 = isBan ? !ourIs0 : ourIs0
          return heroToTeam0
            ? [[...aiState.team0Picks, hero], aiState.team1Picks]
            : [aiState.team0Picks, [...aiState.team1Picks, hero]]
        }
        const toRow = async (hero: string, isGreedyPad: boolean): Promise<OurTurnRow> => {
          const [t0, t1] = candState(hero)
          const ourPct = (await projFor(t0, t1, nextStep)) + (mctsAdj.get(hero) ?? 0)
          if (isBan) {
            // Display the THREAT from the enemy's side: their win chance if
            // they get the hero. Biggest number = biggest threat = ban first,
            // so the list sorts descending like every other list on the page.
            const enemyPct = 100 - ourPct
            return { hero, isGreedyPad, winPct: enemyPct, deltaPp: enemyPct - (100 - projNow) }
          }
          return { hero, isGreedyPad, winPct: ourPct, deltaPp: ourPct - projNow }
        }

        const mctsHeroes = new Set(mctsRecs.map(r => r.hero))
        const aiTopHero = mctsRecs[0]?.hero ?? null
        const candidates = [
          ...mctsRecs.map(r => ({ hero: r.hero, pad: false })),
          ...recommendations
            .filter(r => !mctsHeroes.has(r.hero) && !unavailableHeroes.has(r.hero))
            .map(r => ({ hero: r.hero, pad: true })),
        ].slice(0, 10)
        const rows: OurTurnRow[] = (
          await Promise.all(candidates.map(c => toRow(c.hero, c.pad)))
        )
          .sort((a, b) => b.winPct - a.winPct)
          .map(r => (r.hero === aiTopHero ? { ...r, isAiTop: true } : r))

        if (!cancelled) {
          setSearchResults(rows)
          setSearchInfo(`${sims ?? 0} sims`)
          setSearching(false)
          setSearchStatus('')
        }
      } catch (err) {
        console.error('Search failed:', err)
        if (!cancelled) {
          setSearching(false)
          setSearchStatus(`Search failed: ${err instanceof Error ? err.message : 'unknown error'}`)
        }
      }
    })()
    return () => { cancelled = true }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftData, state.currentStep, state.selections, state.phase, buildAIState])

  // Pick arrays + our-team player assignments (pick index → battletag)
  const pickArrays = useMemo(() => {
    const ourPicks: string[] = []
    const enemyPicks: string[] = []
    const ourPlayerMap: Record<number, string> = {}
    for (let i = 0; i < DRAFT_SEQUENCE.length; i++) {
      const step = DRAFT_SEQUENCE[i]
      const hero = state.selections[i]
      if (!hero || step.type !== 'pick') continue
      if (step.team === state.ourTeam) {
        const pickIdx = ourPicks.length
        ourPicks.push(hero)
        if (state.playerAssignments[i]) {
          ourPlayerMap[pickIdx] = state.playerAssignments[i]
        }
      } else {
        enemyPicks.push(hero)
      }
    }
    return { ourPicks, enemyPicks, ourPlayerMap }
  }, [state.selections, state.playerAssignments, state.ourTeam])

  // Compute running win % for both teams
  const { ourWinPct, enemyWinPct } = useMemo(() => {
    if (!draftData) return { ourWinPct: null, enemyWinPct: null }

    const { ourPicks, enemyPicks, ourPlayerMap } = pickArrays

    if (ourPicks.length === 0 && enemyPicks.length === 0) {
      return { ourWinPct: null, enemyWinPct: null }
    }

    const ourRaw = ourPicks.length > 0
      ? computeTeamWinEstimate(ourPicks, enemyPicks, draftData, state.map, ourPlayerMap)
      : null
    const enemyRaw = enemyPicks.length > 0
      ? computeTeamWinEstimate(enemyPicks, ourPicks, draftData, state.map)
      : null

    // Normalize so the two percentages sum to 100
    if (ourRaw && enemyRaw) {
      const sum = ourRaw.winPct + enemyRaw.winPct
      return {
        ourWinPct: Math.round(ourRaw.winPct / sum * 1000) / 10,
        enemyWinPct: Math.round(enemyRaw.winPct / sum * 1000) / 10,
      }
    }

    return {
      ourWinPct: ourRaw?.winPct ?? null,
      enemyWinPct: enemyRaw?.winPct ?? null,
    }
  }, [pickArrays, state.map, draftData])

  // Raw win estimate (with breakdown) for AMA context enrichment and as the
  // baseline for search-recommendation deltas
  const ourWinEstimate = useMemo(() => {
    if (!draftData || state.phase !== 'drafting') return null
    const { ourPicks, enemyPicks, ourPlayerMap } = pickArrays
    if (ourPicks.length === 0) return null
    return computeTeamWinEstimate(ourPicks, enemyPicks, draftData, state.map, ourPlayerMap)
  }, [pickArrays, state.phase, state.map, draftData])

  // AMA context (updates live with every pick/ban)
  const amaContext = useMemo(() => {
    if (!draftData || state.phase !== 'drafting') return null
    return enrichDraftContext(state, recommendations, draftData, ourWinEstimate)
  }, [state, recommendations, draftData, ourWinEstimate])

  // Sync AMA context to sessionStorage so the standalone /AMA page can read it
  useEffect(() => {
    if (amaContext) {
      try {
        sessionStorage.setItem('ama-draft-context', JSON.stringify(amaContext))
      } catch {
        // sessionStorage unavailable
      }
    }
  }, [amaContext])

  // Heroes that are already selected (banned or picked)
  // Cho'gall: if either Cho or Gall is selected, both are unavailable
  // Also block Cho/Gall from being picked when our team has <2 picks remaining
  const unavailableHeroes = useMemo(() => {
    const set = expandChoGall(new Set(Object.values(state.selections)))

    // Block Cho/Gall if the current team has <2 consecutive picks this turn
    // (applies to both our team and enemy team — it's a game rule, not team-specific)
    if (state.phase === 'drafting' && state.currentStep < DRAFT_SEQUENCE.length) {
      const step = DRAFT_SEQUENCE[state.currentStep]
      if (step.type === 'pick') {
        const turnsLeft = consecutivePicksRemaining(
          state.currentStep, step.team, state.selections
        )
        if (turnsLeft < 2) {
          set.add('Cho')
          set.add('Gall')
        }
      }
    }

    return set
  }, [state.selections, state.phase, state.currentStep])

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
            <div className="grid grid-cols-2 gap-2">
              {maps.map((map) => {
                const img = mapImageSrc(map)
                return (
                  <button
                    key={map}
                    onClick={() => dispatch({ type: 'SET_MAP', map })}
                    className={cn(
                      'rounded-lg overflow-hidden text-sm text-left transition-all border',
                      state.map === map
                        ? 'border-primary ring-1 ring-primary shadow-lg scale-[1.02]'
                        : 'border-border hover:border-foreground/40 hover:shadow-md'
                    )}
                  >
                    {img && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={img}
                        alt=""
                        loading="lazy"
                        className={cn(
                          'w-full h-20 object-cover transition-all',
                          state.map === map ? 'brightness-110' : 'brightness-75 hover:brightness-100'
                        )}
                      />
                    )}
                    <div className={cn(
                      'px-2 py-1.5',
                      state.map === map
                        ? 'text-primary font-medium'
                        : 'text-muted-foreground'
                    )}>
                      {map}
                    </div>
                  </button>
                )
              })}
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
  const draftView = buildDraftView(state)
  const ourPicksView = state.ourTeam === 'A' ? draftView.picksA : draftView.picksB
  const enemyPicksView = state.ourTeam === 'A' ? draftView.picksB : draftView.picksA
  return (
    <div className={cn('space-y-4 transition-all duration-300 ease-in-out', amaOpen && 'sm:mr-[420px]')}>
      {/* HotS-inspired arena wrapper — dark navy gradient background */}
      <div
        className="rounded-lg p-4 sm:p-6 space-y-4"
        style={{
          background: 'radial-gradient(ellipse at top, #1a1f3a 0%, #0a0d1f 70%)',
        }}
      >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">
            Draft &mdash; {state.map}
          </h1>
          <p className="text-[#8b9bc8] text-sm">
            {state.phase === 'complete'
              ? 'Draft complete'
              : currentStep
                ? `${currentStep.team === state.ourTeam ? 'Your' : 'Enemy'} ${currentStep.type === 'ban' ? 'Ban' : 'Pick'}`
                : ''}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {currentStep?.type === 'ban' && (
            <button
              onClick={() => dispatch({ type: 'SKIP_BAN' })}
              className="px-3 py-1.5 rounded text-xs font-medium border border-[#d4b85a]/50 text-[#d4b85a] hover:bg-[#d4b85a]/10 transition-colors"
            >
              No Ban
            </button>
          )}
          <button
            onClick={() => dispatch({ type: 'UNDO' })}
            disabled={state.currentStep === 0}
            className={cn(
              'px-3 py-1.5 rounded text-xs font-medium border transition-colors',
              state.currentStep > 0
                ? 'border-[#3a4050] text-[#8b9bc8] hover:text-[#d6dbe0] hover:bg-[#3a4050]/40'
                : 'border-[#3a4050]/50 text-[#8b9bc8]/50 cursor-not-allowed'
            )}
          >
            Undo
          </button>
          <button
            onClick={() => dispatch({ type: 'RESET' })}
            className="px-3 py-1.5 rounded text-xs font-medium border border-[#d46b6b]/50 text-[#d46b6b] hover:bg-[#d46b6b]/10 transition-colors"
          >
            Reset
          </button>
          <button
            onClick={() => setAmaOpen(true)}
            className="px-3 py-1.5 rounded text-xs font-medium border border-[#b48ad4]/50 text-[#b48ad4] hover:bg-[#b48ad4]/10 transition-colors"
          >
            Ask the Coach
          </button>
        </div>
      </div>

      {/* Ban bar */}
      <BanBar
        bansA={draftView.bansA}
        bansB={draftView.bansB}
        ourTeam={state.ourTeam}
      />

      {/* Win % banner */}
      <div className="flex items-center justify-center gap-6 text-sm">
        <WinPctBadge label="YOUR TEAM" pct={ourWinPct} accent="blue" />
        <WinPctBadge label="ENEMY" pct={enemyWinPct} accent="red" />
      </div>

      {/* Main drafting area — 3 column hex layout */}
      {state.phase !== 'complete' && (
        <div className="grid grid-cols-1 lg:grid-cols-[220px_1fr_220px] gap-6">
          {/* Our team column */}
          <TeamColumn picks={ourPicksView} accent="blue" label="YOUR TEAM" />

          {/* Center — recs on top, picker below */}
          <div className="space-y-4">
            <div>
            {currentStep?.team === state.ourTeam ? (
              // Our turn: MCTS recommendations, padded with greedy to 10
              <SearchRecommendationPanel
                ourRows={searchResults}
                searchInfo={searchInfo}
                searching={searching}
                statusText={searchStatus}
                isBanPhase={currentStep?.type === 'ban'}
                isOurTurn={true}
                onSelect={handleSelectHero}
                unavailable={unavailableHeroes}
                draftData={draftData ?? undefined}
                availableBattletags={availableBattletags}
                map={state.map}
              />
            ) : (
              // Opponent turn: show GD model predictions + impact on our win %
              <SearchRecommendationPanel
                opponentRows={gdOpponentPreds}
                searching={gdLoading}
                statusText={searchStatus}
                isBanPhase={currentStep?.type === 'ban'}
                isOurTurn={false}
                onSelect={handleSelectHero}
                unavailable={unavailableHeroes}
              />
            )}
            </div>
            <HeroPicker
              unavailable={unavailableHeroes}
              onSelect={handleSelectHero}
              currentStepType={currentStep?.type ?? 'pick'}
              isOurTurn={currentStep?.team === state.ourTeam}
            />
          </div>

          {/* Enemy team column */}
          <TeamColumn picks={enemyPicksView} accent="red" label="ENEMY" />
        </div>
      )}

      </div>
      {/* End HotS arena wrapper */}

      {/* Detailed draft board (keeps player-assignment controls) */}
      <DraftBoard
        state={state}
        currentStep={state.currentStep}
        availableBattletags={availableBattletags}
        playerAssignments={state.playerAssignments}
        onAssignPlayer={(stepIdx, bt) =>
          dispatch({ type: 'ASSIGN_PLAYER', stepIndex: stepIdx, battletag: bt })
        }
        teamAWinPct={state.ourTeam === 'A' ? ourWinPct : enemyWinPct}
        teamBWinPct={state.ourTeam === 'B' ? ourWinPct : enemyWinPct}
      />

      {/* Complete summary */}
      {state.phase === 'complete' && (
        <div className="rounded-lg border p-6 text-center space-y-3 overflow-hidden relative">
          {ourWinPct !== null && ourWinPct >= 60 && (
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/10 via-amber-500/20 to-yellow-500/10 animate-pulse" />
              <div className="absolute -inset-1 bg-gradient-to-r from-transparent via-yellow-400/5 to-transparent animate-[shimmer_2s_infinite]" />
            </div>
          )}
          {ourWinPct !== null && ourWinPct >= 60 ? (
            <>
              <div className="relative">
                <p className="text-4xl md:text-5xl font-black tracking-tighter bg-gradient-to-r from-yellow-300 via-amber-400 to-yellow-300 bg-clip-text text-transparent animate-pulse drop-shadow-lg">
                  DRAFT DOMINATION
                </p>
                <p className="text-6xl md:text-7xl font-black tabular-nums bg-gradient-to-b from-yellow-200 to-amber-500 bg-clip-text text-transparent mt-2">
                  {ourWinPct.toFixed(1)}%
                </p>
                <p className="text-sm text-yellow-400/80 mt-2 font-medium tracking-wide uppercase">
                  Expected Win Rate
                </p>
              </div>
              <div className="flex justify-center gap-1 text-2xl mt-1">
                <span className="animate-bounce" style={{ animationDelay: '0ms' }}>&#x1F451;</span>
                <span className="animate-bounce" style={{ animationDelay: '150ms' }}>&#x1F525;</span>
                <span className="animate-bounce" style={{ animationDelay: '300ms' }}>&#x1F451;</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Your draft is looking unstoppable. Go destroy them!
              </p>
            </>
          ) : (
            <>
              <p className="text-lg font-semibold text-white">
                Draft Complete
              </p>
              <p className="text-sm text-muted-foreground">
                Good luck in your game!
              </p>
            </>
          )}
          <button
            onClick={() => dispatch({ type: 'RESET' })}
            className="relative px-6 py-2 rounded-md text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            New Draft
          </button>
        </div>
      )}

      <AMADrawer
        open={amaOpen}
        onClose={() => setAmaOpen(false)}
        draftContext={amaContext}
      />
    </div>
  )
}

function WinPctBadge({
  label, pct, accent,
}: { label: string; pct: number | null; accent: 'blue' | 'red' }) {
  const tint = accent === 'blue' ? '#6b8dd4' : '#d46b6b'
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] tracking-[0.2em]" style={{ color: tint }}>{label}</span>
      <span
        className="text-lg font-bold tabular-nums"
        style={{ color: pct === null ? '#6b7078' : '#e0e4ea' }}
      >
        {pct === null ? '—' : `${pct.toFixed(1)}%`}
      </span>
    </div>
  )
}
