#!/usr/bin/env tsx
/**
 * Benchmark: Expectimax search vs greedy baseline on real replay data.
 *
 * For each replay:
 *   - Steps through the actual draft order
 *   - At "our" steps: runs both greedy and expectimax to choose a hero
 *   - At opponent steps: uses the actual replay pick (fair comparison)
 *   - Records draft quality metrics for both strategies
 *
 * Usage:
 *   set -a && source .env && set +a
 *   npx tsx benchmark/expectimax-bench.ts --depth 6 --drafts 500 --tier mid
 */

import { parseArgs } from 'node:util'
import { HERO_ROLES } from '@/lib/data/hero-roles'
import { DRAFT_SEQUENCE, type DraftData } from '@/lib/draft/types'
import { scoreHeroForPick, scoreHeroForBan, getHeroWinRate } from '@/lib/draft/engine'
import {
  createSearchState,
  cloneAndApply,
  getValidHeroes,
  isTerminal,
  isOurTurn,
  isBanPhase,
  iterativeDeepeningSearch,
} from '@/lib/draft/expectimax'
import type { SearchState, ExpectimaxConfig } from '@/lib/draft/expectimax/types'
import { loadDraftData } from './load-draft-data'
import { loadReplays, type ReplayDraft } from './load-replays'
import { createNodeGDPredictor } from './gd-adapter-node'
import type { SkillTier } from '@/lib/types'

// ── CLI args ──

const { values: args } = parseArgs({
  options: {
    depth: { type: 'string', default: '6' },
    'our-pick-width': { type: 'string', default: '8' },
    'our-ban-width': { type: 'string', default: '4' },
    'opp-pick-width': { type: 'string', default: '6' },
    'opp-ban-width': { type: 'string', default: '3' },
    drafts: { type: 'string', default: '500' },
    tier: { type: 'string', default: 'mid' },
    'time-budget': { type: 'string', default: '5000' },
  },
})

const DEPTH = parseInt(args.depth!)
const OUR_PICK_WIDTH = parseInt(args['our-pick-width']!)
const OUR_BAN_WIDTH = parseInt(args['our-ban-width']!)
const OPP_PICK_WIDTH = parseInt(args['opp-pick-width']!)
const OPP_BAN_WIDTH = parseInt(args['opp-ban-width']!)
const NUM_DRAFTS = parseInt(args.drafts!)
const TIER = args.tier! as SkillTier
const TIME_BUDGET = parseInt(args['time-budget']!)

// Fine-grained role mapping for degen detection
const HERO_ROLE_FINE: Record<string, string> = {}
const FINE_MAP: Record<string, string> = {
  'Tank': 'tank', 'Bruiser': 'bruiser', 'Healer': 'healer',
  'Ranged Assassin': 'ranged', 'Melee Assassin': 'melee', 'Support': 'support',
}
for (const [hero, role] of Object.entries(HERO_ROLES)) {
  HERO_ROLE_FINE[hero] = FINE_MAP[role] || 'ranged'
}

// ── Draft quality metrics ──

function counterDelta(ha: string, hb: string, data: DraftData, map: string | null): number | null {
  const d = data.counters[ha]?.[hb]
  if (!d || d.games < 30) return null
  const wrA = getHeroWinRate(ha, data, map)?.winRate ?? 50
  const wrB = getHeroWinRate(hb, data, map)?.winRate ?? 50
  return d.winRate - (wrA + (100 - wrB) - 50)
}

function synergyDelta(ha: string, hb: string, data: DraftData, map: string | null): number | null {
  const d = data.synergies[ha]?.[hb]
  if (!d || d.games < 30) return null
  const wrA = getHeroWinRate(ha, data, map)?.winRate ?? 50
  const wrB = getHeroWinRate(hb, data, map)?.winRate ?? 50
  return d.winRate - (50 + (wrA - 50) + (wrB - 50))
}

interface DraftMetrics {
  counter: number
  counterLate: number
  synergy: number
  resilGrad: number
  healer: boolean
  degen: boolean
  heroes: string[]
}

function computeMetrics(
  pickSteps: { hero: string; team: 'ours' | 'theirs'; step: number }[],
  data: DraftData,
  map: string,
): DraftMetrics {
  const our = pickSteps.filter(p => p.team === 'ours')
  const opp = pickSteps.filter(p => p.team === 'theirs')
  const ourHeroes = our.map(p => p.hero)

  // Resilience
  const exposures: number[] = []
  for (const { hero, step } of our) {
    const subs = opp.filter(o => o.step > step).map(o => o.hero)
    if (subs.length === 0) { exposures.push(0); continue }
    const ds = subs.map(s => counterDelta(s, hero, data, map)).filter((d): d is number => d !== null)
    exposures.push(ds.length > 0 ? ds.reduce((a, b) => a + b, 0) / ds.length : 0)
  }
  const resilGrad = exposures.length >= 4
    ? (exposures.slice(-2).reduce((a, b) => a + b, 0) / 2) - (exposures.slice(0, 2).reduce((a, b) => a + b, 0) / 2)
    : 0

  // Counter
  const ctrs: number[] = []
  for (const { hero, step } of our) {
    const priors = opp.filter(o => o.step < step).map(o => o.hero)
    if (priors.length === 0) { ctrs.push(0); continue }
    const ds = priors.map(p => counterDelta(hero, p, data, map)).filter((d): d is number => d !== null)
    ctrs.push(ds.length > 0 ? ds.reduce((a, b) => a + b, 0) / ds.length : 0)
  }
  const counter = ctrs.length > 0 ? ctrs.reduce((a, b) => a + b, 0) / ctrs.length : 0
  const counterLate = ctrs.length >= 2 ? ctrs.slice(-2).reduce((a, b) => a + b, 0) / 2 : 0

  // Synergy
  const syns: number[] = []
  for (let i = 0; i < ourHeroes.length; i++) {
    for (let j = i + 1; j < ourHeroes.length; j++) {
      const d = synergyDelta(ourHeroes[i], ourHeroes[j], data, map)
      if (d !== null) syns.push(d)
    }
  }
  const synergy = syns.length > 0 ? syns.reduce((a, b) => a + b, 0) / syns.length : 0

  // Composition
  const healers = new Set(Object.entries(HERO_ROLES).filter(([_, r]) => r === 'Healer').map(([h]) => h))
  const frontline = new Set(Object.entries(HERO_ROLES).filter(([_, r]) => r === 'Tank' || r === 'Bruiser').map(([h]) => h))
  const ranged = new Set(Object.entries(HERO_ROLES).filter(([_, r]) => r === 'Ranged Assassin').map(([h]) => h))
  const hasHealer = ourHeroes.some(h => healers.has(h))
  const hasFront = ourHeroes.some(h => frontline.has(h))
  const hasRanged = ourHeroes.some(h => ranged.has(h))
  const roles: Record<string, number> = {}
  for (const h of ourHeroes) {
    const r = HERO_ROLES[h] || 'Unknown'
    roles[r] = (roles[r] || 0) + 1
  }
  // 3+ of Tank, Melee Assassin, Support, or Healer is degenerate
  // 3 Ranged Assassins or 3 Bruisers are viable compositions
  const degenStackRoles = new Set(['Tank', 'Melee Assassin', 'Support', 'Healer'])
  const badStack = Object.entries(roles).some(([role, count]) => count >= 3 && degenStackRoles.has(role))
  const degen = !hasHealer || !hasFront || !hasRanged || badStack

  return { counter, counterLate, synergy, resilGrad, healer: hasHealer, degen, heroes: ourHeroes }
}

// ── Greedy strategy ──

function greedyPick(
  ourPicks: string[], enemyPicks: string[], validHeroes: string[],
  data: DraftData, map: string,
): string {
  let best = validHeroes[0]
  let bestScore = -Infinity
  for (const h of validHeroes) {
    const s = scoreHeroForPick(h, ourPicks, enemyPicks, data, map)
    if (s > bestScore) { bestScore = s; best = h }
  }
  return best
}

function greedyBan(
  picksToProtect: string[], opponentPicks: string[], validHeroes: string[],
  data: DraftData, map: string,
): string {
  let best = validHeroes[0]
  let bestScore = -Infinity
  for (const h of validHeroes) {
    const s = scoreHeroForBan(h, picksToProtect, opponentPicks, data, map)
    if (s > bestScore) { bestScore = s; best = h }
  }
  return best
}

// ── Simulate draft ──

async function simulateDraft(
  replay: ReplayDraft,
  strategy: 'greedy' | 'search',
  data: DraftData,
  config: ExpectimaxConfig,
  opponentPredict: ReturnType<typeof createNodeGDPredictor> extends Promise<infer T> ? T : never,
  ourTeam: 0 | 1,
): Promise<DraftMetrics> {
  const ourTeamLabel: 'A' | 'B' = ourTeam === 0 ? 'A' : 'B'
  const t0Set = new Set(replay.team0Heroes)
  const t1Set = new Set(replay.team1Heroes)

  // Walk through draft, building state
  const pickSteps: { hero: string; team: 'ours' | 'theirs'; step: number }[] = []
  const taken = new Set<string>()
  const ourPicks: string[] = []
  const enemyPicks: string[] = []

  // Build a SearchState-compatible structure for expectimax
  let searchState: SearchState = {
    ourPicks: [], enemyPicks: [], bans: [],
    taken: new Set(),
    step: 0,
    map: replay.gameMap,
    tier: replay.skillTier as SkillTier,
    ourTeam: ourTeamLabel,
  }

  for (let stepIdx = 0; stepIdx < 16; stepIdx++) {
    if (isTerminal(searchState)) break
    // searchState.step may differ from stepIdx if Cho/Gall consumed extra slots
    const currentStep = searchState.step
    if (currentStep >= DRAFT_SEQUENCE.length) break
    const draftStep = DRAFT_SEQUENCE[currentStep]
    const replayEntry = replay.draftOrder[stepIdx]
    const replayHero = replayEntry?.hero
    const isOurs = draftStep.team === ourTeamLabel
    const isPick = draftStep.type === 'pick'

    if (isOurs) {
      // Our turn — use strategy
      const valid = getValidHeroes(searchState)
      let chosenHero: string

      if (strategy === 'search' && isPick) {
        const results = await iterativeDeepeningSearch(
          searchState, data, config, opponentPredict,
        )
        chosenHero = results.length > 0 ? results[0].hero : valid[0]
      } else if (isPick) {
        chosenHero = greedyPick(ourPicks, enemyPicks, valid, data, replay.gameMap)
      } else {
        // Ban
        chosenHero = greedyBan(ourPicks, enemyPicks, valid, data, replay.gameMap)
      }

      searchState = cloneAndApply(searchState, chosenHero)
      taken.add(chosenHero)
      if (isPick) {
        ourPicks.push(chosenHero)
        pickSteps.push({ hero: chosenHero, team: 'ours', step: currentStep })
      }
    } else {
      // Opponent turn — use actual replay pick
      if (!replayHero) continue

      // Determine actual hero from replay
      let actualHero = replayHero
      if (taken.has(actualHero)) {
        // Hero already taken in our simulation — opponent must pick something else
        // This happens because our strategy diverged from the replay
        // Use GD model prediction as fallback
        const valid = getValidHeroes(searchState)
        if (valid.length === 0) continue
        const preds = await opponentPredict(searchState, 1)
        actualHero = preds.length > 0 ? preds[0].hero : valid[0]
      }

      searchState = cloneAndApply(searchState, actualHero)
      taken.add(actualHero)
      if (isPick) {
        enemyPicks.push(actualHero)
        pickSteps.push({ hero: actualHero, team: 'theirs', step: currentStep })
      }
    }
  }

  return computeMetrics(pickSteps, data, replay.gameMap)
}

// ── Main ──

async function main() {
  console.log('='.repeat(80))
  console.log('  Expectimax Benchmark')
  console.log(`  Depth: ${DEPTH}, Our width: ${OUR_PICK_WIDTH}/${OUR_BAN_WIDTH}, Opp width: ${OPP_PICK_WIDTH}/${OPP_BAN_WIDTH}`)
  console.log(`  Drafts: ${NUM_DRAFTS}, Tier: ${TIER}, Time budget: ${TIME_BUDGET}ms`)
  console.log('='.repeat(80))

  console.log('\nLoading data...')
  const [data, replays, opponentPredict] = await Promise.all([
    loadDraftData(TIER),
    loadReplays(NUM_DRAFTS, TIER),
    createNodeGDPredictor(),
  ])
  console.log(`  ${Object.keys(data.heroStats).length} heroes, ${replays.length} replays`)

  const config: ExpectimaxConfig = {
    ourPickWidth: OUR_PICK_WIDTH,
    ourBanWidth: OUR_BAN_WIDTH,
    oppPickWidth: OPP_PICK_WIDTH,
    oppBanWidth: OPP_BAN_WIDTH,
    maxDepth: DEPTH,
    timeBudgetMs: TIME_BUDGET,
  }

  // Run both strategies
  for (const strategy of ['greedy', 'search'] as const) {
    console.log(`\n--- ${strategy.toUpperCase()} ---`)
    const t0 = Date.now()
    const allMetrics: DraftMetrics[] = []
    const heroCounter: Record<string, number> = {}

    for (let i = 0; i < replays.length; i++) {
      const replay = replays[i]
      const ourTeam = (i % 2) as 0 | 1

      const metrics = await simulateDraft(
        replay, strategy, data, config, opponentPredict, ourTeam,
      )
      allMetrics.push(metrics)
      for (const h of metrics.heroes) heroCounter[h] = (heroCounter[h] || 0) + 1

      if ((i + 1) % 50 === 0) {
        const elapsed = Date.now() - t0
        const eta = (elapsed / (i + 1)) * (replays.length - i - 1)
        process.stdout.write(`  ${i + 1}/${replays.length} (${Math.round(eta / 1000)}s remaining)\r`)
      }
    }

    const elapsed = Date.now() - t0
    const n = allMetrics.length

    const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length
    const agg = {
      counter: avg(allMetrics.map(m => m.counter)),
      counterLate: avg(allMetrics.map(m => m.counterLate)),
      synergy: avg(allMetrics.map(m => m.synergy)),
      resilGrad: avg(allMetrics.map(m => m.resilGrad)),
      healer: avg(allMetrics.map(m => m.healer ? 1 : 0)) * 100,
      degen: avg(allMetrics.map(m => m.degen ? 1 : 0)) * 100,
      distinct: Object.keys(heroCounter).length,
    }

    console.log(`\n  ${strategy}: ctr=${agg.counter >= 0 ? '+' : ''}${agg.counter.toFixed(3)} ` +
      `ctrL=${agg.counterLate >= 0 ? '+' : ''}${agg.counterLate.toFixed(3)} ` +
      `syn=${agg.synergy.toFixed(3)} rG=${agg.resilGrad >= 0 ? '+' : ''}${agg.resilGrad.toFixed(3)} ` +
      `hlr=${agg.healer.toFixed(0)}% deg=${agg.degen.toFixed(0)}% div=${agg.distinct} ` +
      `(${Math.round(elapsed / 1000)}s)`)
  }

  console.log('\n' + '='.repeat(80))
  console.log('  Reference baselines:')
  console.log('  E MCTS:        ctr=-0.082  syn=0.503  rG=-0.578  hlr=86%  deg=26%')
  console.log('  Greedy enrich: ctr=+0.305  syn=1.171  rG=+0.119  hlr=74%  deg=55%')
  console.log('='.repeat(80))
}

main().catch(err => { console.error(err); process.exit(1) })
