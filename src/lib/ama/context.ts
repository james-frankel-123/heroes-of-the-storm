/**
 * Draft context enricher for AMA.
 *
 * Takes the live draft state, current recommendations, and full DraftData,
 * and serializes everything into a compact, information-dense text block
 * suitable for injection into the LLM system prompt.
 *
 * Also returns a parsed card for UI display.
 */

import type { DraftState, DraftData, DraftRecommendation } from '@/lib/draft/types'
import type { WinEstimateResult, WinEstimateBreakdown } from '@/lib/draft/win-estimate'
import { DRAFT_SEQUENCE } from '@/lib/draft/types'

export interface DraftContextCard {
  map: string | null
  tier: string
  ourTeam: 'A' | 'B'
  ourPicks: string[]
  enemyPicks: string[]
  bans: string[]
  step: number
  totalSteps: number
  stepType: 'pick' | 'ban' | null
  isOurTurn: boolean
  /**
   * 'displayed': the page's own recommendation rows (searched rows; netDelta in pp vs the
   * search line). 'stats': the statistical ranking, a different metric (netDelta in pp).
   */
  topRecsSource: 'displayed' | 'stats'
  topRecs: Array<{
    rank: number
    hero: string
    netDelta: number
    reasons: Array<{ type: string; label: string; delta: number }>
    suggestedPlayer: string | null
  }>
  winPct: number | null
  winBreakdown: WinEstimateBreakdown | null
}

export interface EnrichedDraftContext {
  /** Full text block for LLM injection */
  textBlock: string
  /** Parsed structure for UI display */
  card: DraftContextCard
}

function fmt(n: number): string {
  return `${n >= 0 ? '+' : ''}${n.toFixed(1)} pp`
}

function fmtWR(wr: number, games: number): string {
  return `${wr.toFixed(1)}% WR / ${games} games`
}

/** What the draft page currently displays, so the coach describes the same ranking and numbers. */
export interface DisplayedDraft {
  rows: { hero: string; winPct: number; deltaPp: number; searched: boolean; note?: string }[]
  /** The banner's win estimate (our team, %), or null before any pick */
  nowPct: number | null
}

export function enrichDraftContext(
  state: DraftState,
  recommendations: DraftRecommendation[],
  draftData: DraftData,
  winEstimate: WinEstimateResult | null,
  topN = 5,
  displayed?: DisplayedDraft,
): EnrichedDraftContext {
  // Derive picks and bans from state
  const ourPicks: string[] = []
  const enemyPicks: string[] = []
  const bans: string[] = []
  const enemyTeam = state.ourTeam === 'A' ? 'B' : 'A'

  for (let i = 0; i < state.currentStep; i++) {
    const step = DRAFT_SEQUENCE[i]
    const hero = state.selections[i]
    if (!hero) continue
    if (step.type === 'ban') {
      bans.push(hero)
    } else if (step.team === state.ourTeam) {
      ourPicks.push(hero)
    } else {
      enemyPicks.push(hero)
    }
  }

  const currentStep = state.currentStep < DRAFT_SEQUENCE.length
    ? DRAFT_SEQUENCE[state.currentStep]
    : null
  const isOurTurn = currentStep?.team === state.ourTeam

  // Build card: the displayed (searched) rows when there are any, else the stats ranking
  const shownRows = (displayed?.rows ?? []).filter(r => r.searched)
  const topRecsSource: 'displayed' | 'stats' = shownRows.length > 0 ? 'displayed' : 'stats'
  const topRecs = topRecsSource === 'displayed'
    ? shownRows.slice(0, topN).map((r, i) => ({
        rank: i + 1,
        hero: r.hero,
        netDelta: r.deltaPp,
        reasons: [] as Array<{ type: string; label: string; delta: number }>,
        suggestedPlayer: null,
      }))
    : recommendations
        .slice(0, topN)
        .map((rec, i) => ({
          rank: i + 1,
          hero: rec.hero,
          netDelta: rec.netDelta,
          reasons: rec.reasons,
          suggestedPlayer: rec.suggestedPlayer,
        }))

  const card: DraftContextCard = {
    map: state.map,
    tier: state.tier,
    ourTeam: state.ourTeam,
    ourPicks,
    enemyPicks,
    bans,
    step: state.currentStep,
    totalSteps: DRAFT_SEQUENCE.length,
    stepType: currentStep?.type ?? null,
    isOurTurn,
    topRecsSource,
    topRecs,
    winPct: displayed ? displayed.nowPct : winEstimate?.winPct ?? null,
    winBreakdown: winEstimate?.breakdown ?? null,
  }

  // Build text block
  const lines: string[] = []

  lines.push('=== CURRENT DRAFT STATE ===')
  lines.push(
    `Map: ${state.map ?? 'Not selected'} | Tier: ${state.tier} | Our team: ${state.ourTeam} | Step: ${state.currentStep}/${DRAFT_SEQUENCE.length}${currentStep ? ` (${isOurTurn ? 'OUR' : 'ENEMY'} ${currentStep.type.toUpperCase()})` : ' (complete)'}`
  )
  lines.push('')

  if (ourPicks.length > 0) {
    lines.push(`Our picks: ${ourPicks.join(', ')}`)
  } else {
    lines.push('Our picks: (none yet)')
  }

  if (enemyPicks.length > 0) {
    lines.push(`Enemy picks: ${enemyPicks.join(', ')}`)
  } else {
    lines.push('Enemy picks: (none yet)')
  }

  if (bans.length > 0) {
    lines.push(`Bans: ${bans.join(', ')}`)
  } else {
    lines.push('Bans: (none yet)')
  }

  if (displayed?.nowPct != null) {
    lines.push(`Win estimate shown on the page (banner): ${displayed.nowPct.toFixed(1)}% for our team`)
  }
  if (displayed && displayed.rows.length > 0) {
    lines.push('')
    lines.push('=== RECOMMENDATIONS AS DISPLAYED (search: projected final win chance if we keep following the search; change in pp vs the search line) ===')
    displayed.rows.slice(0, 10).forEach((r, i) => {
      const value = r.searched
        ? `${r.winPct.toFixed(1)}% projected final, ${r.deltaPp >= 0 ? '+' : ''}${r.deltaPp.toFixed(1)} pp vs search line`
        : 'not searched (listed from statistics or a matched player\'s pool; no win chance computed)'
      lines.push(`#${i + 1} ${r.hero}: ${value}${r.note ? ` | ${r.note}` : ''}`)
    })
  }

  if (winEstimate && winEstimate.winPct !== 50) {
    const bd = winEstimate.breakdown
    lines.push(
      `Stats-model estimate (background, not the banner): ${winEstimate.winPct.toFixed(1)}% ` +
      `(heroWR: ${fmt(bd.heroWR)}, synergies: ${fmt(bd.synergies)}, counters: ${fmt(bd.counters)}, playerAdj: ${fmt(bd.playerAdj)})`
    )
  }

  lines.push('')
  lines.push(`=== TOP ${Math.min(topN, recommendations.length)} STATISTICAL ALTERNATIVES (stats mode: a different ranking and metric than the displayed rows; deltas in pp) ===`)

  if (recommendations.length === 0) {
    lines.push('(No recommendations available — draft may not be in progress or no map selected)')
  }

  for (let i = 0; i < Math.min(topN, recommendations.length); i++) {
    const rec = recommendations[i]
    lines.push(`#${i + 1} ${rec.hero} — netDelta: ${fmt(rec.netDelta)}`)

    // Enrich each reason with raw source data from draftData
    for (const reason of rec.reasons) {
      let detail = ''

      if (reason.type === 'hero_wr') {
        const overall = draftData.heroStats[rec.hero]
        const mapWR = state.map ? draftData.heroMapWinRates[state.map]?.[rec.hero] : null
        if (mapWR && mapWR.games >= 50) {
          detail = `(map: ${fmtWR(mapWR.winRate, mapWR.games)}; overall: ${fmtWR(overall?.winRate ?? 50, overall?.games ?? 0)})`
        } else if (overall) {
          detail = `(overall: ${fmtWR(overall.winRate, overall.games)})`
        }
      } else if (reason.type === 'counter') {
        // Extract enemy hero name from label (e.g. "vs Illidan")
        const match = reason.label.match(/vs (.+)$/)
        if (match) {
          const enemy = match[1]
          const pairData = draftData.counters[rec.hero]?.[enemy]
          if (pairData) {
            detail = `(pairwise vs ${enemy}: ${fmtWR(pairData.winRate, pairData.games)}, normalized)`
          }
        }
      } else if (reason.type === 'synergy') {
        // Extract ally hero name from label (e.g. "with Uther")
        const match = reason.label.match(/with (.+)$/)
        if (match) {
          const ally = match[1]
          const pairData = draftData.synergies[rec.hero]?.[ally]
          if (pairData) {
            detail = `(pairwise with ${ally}: ${fmtWR(pairData.winRate, pairData.games)}, normalized)`
          }
        }
      } else if (reason.type === 'player_strong') {
        // Extract battletag from label
        const match = reason.label.match(/^(.+?) [+\-]/)
        if (match) {
          const battletag = match[1]
          const mapStats = state.map
            ? draftData.playerMapStats[battletag]?.[state.map]?.[rec.hero]
            : null
          const overallStats = draftData.playerStats[battletag]?.[rec.hero]
          if (mapStats && mapStats.games >= 25) {
            detail = `(${battletag}: ${fmtWR(mapStats.winRate, mapStats.games)} on ${state.map})`
          } else if (overallStats) {
            detail = `(${battletag}: ${fmtWR(overallStats.winRate, overallStats.games)} overall, MAWP: ${overallStats.mawp?.toFixed(1) ?? 'n/a'})`
          }
        }
      }

      lines.push(`  • ${reason.type}: ${fmt(reason.delta)} — ${reason.label} ${detail}`)
    }

    if (rec.suggestedPlayer) {
      lines.push(`  → Suggested player: ${rec.suggestedPlayer}`)
    }
  }

  // Add any relevant raw matchup data for heroes on the board
  if ((ourPicks.length > 0 || enemyPicks.length > 0) && recommendations.length > 0) {
    lines.push('')
    lines.push('=== KEY MATCHUP DATA FOR CURRENT BOARD ===')

    const topHero = recommendations[0]?.hero
    if (topHero) {
      // Show counter data vs each enemy pick
      for (const enemy of enemyPicks) {
        const d = draftData.counters[topHero]?.[enemy]
        if (d && d.games >= 30) {
          lines.push(`${topHero} vs ${enemy}: ${fmtWR(d.winRate, d.games)}`)
        }
      }
      // Show synergy data with each ally pick
      for (const ally of ourPicks) {
        const d = draftData.synergies[topHero]?.[ally]
        if (d && d.games >= 30) {
          lines.push(`${topHero} with ${ally}: ${fmtWR(d.winRate, d.games)}`)
        }
      }
    }
  }

  return {
    textBlock: lines.join('\n'),
    card,
  }
}

/**
 * Starter questions for the coach. A "why is X top" prompt only when the page shows a
 * searched top pick; a finished draft gets questions about the composition and the game plan.
 */
export function coachStarterPrompts(card: DraftContextCard | null): string[] {
  if (card && card.step >= card.totalSteps) {
    return [
      'What is our team composition trying to accomplish?',
      'What are the biggest threats in the enemy draft?',
      `What should our game plan be on ${card.map ?? 'this map'}?`,
      'Which enemy hero should we watch most closely?',
    ]
  }
  const top = card?.topRecsSource === 'displayed' ? card.topRecs[0]?.hero : undefined
  return [
    top ? `Why is ${top} the top recommendation right now?` : 'What should we prioritize with this pick?',
    'What is our team composition trying to do?',
    'What are the biggest threats in the enemy draft?',
    'Is our ban strategy sound given the enemy team?',
  ]
}
