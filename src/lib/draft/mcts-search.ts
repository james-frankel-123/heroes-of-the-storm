/**
 * Main-thread MCTS search using the already-loaded ONNX sessions.
 *
 * AlphaZero-style search per HOTS_FEVER_SPEC.md:
 * - policy network provides tree priors + value estimates
 * - GD model samples opponent moves during search
 * - WP model evaluates terminal (complete) drafts when an evaluator is provided
 *
 * Runs MIN_SIMS-MAX_SIMS simulations (50-200 per spec), stopping early once
 * the time budget is exhausted after the minimum simulation count.
 */

import {
  HEROES, NUM_HEROES, NUM_MAPS, NUM_TIERS, STATE_DIM, HERO_TO_IDX, MAP_TO_IDX, TIER_TO_IDX,
} from './encoding'
import { DRAFT_SEQUENCE } from './types'
import { applySelection, lineups, positionProblems, unavailableAt, CHO, GALL, type Selections } from './rules'

const C_PUCT = 2.0
// Tunables: at least MIN_SIMS simulations are always run; after that the
// search stops when TIME_BUDGET_MS is exceeded or MAX_SIMS is reached.
const MIN_SIMS = 50
const MAX_SIMS = 200
const TIME_BUDGET_MS = 1500

function softmaxMasked(logits: Float32Array, mask: Float32Array): Float32Array {
  const result = new Float32Array(logits.length)
  let maxVal = -Infinity
  for (let i = 0; i < logits.length; i++) {
    if (mask[i] > 0 && logits[i] > maxVal) maxVal = logits[i]
  }
  let sum = 0
  for (let i = 0; i < logits.length; i++) {
    if (mask[i] > 0) {
      result[i] = Math.exp(logits[i] - maxVal)
      sum += result[i]
    }
  }
  if (sum > 0) {
    for (let i = 0; i < result.length; i++) result[i] /= sum
  }
  return result
}

/**
 * Tree over legal positions (training/cuda_mcts/ref_search.py semantics):
 *  - decision nodes (our steps): PUCT over the legal actions of THEIR OWN position, priors
 *    from the policy, untried actions at Q = 0, children created on demand;
 *  - chance nodes (opponent steps): outcome sampled from the GD model over that position's
 *    legal actions, one child per distinct outcome, progressive widening
 *    limit(N) = max(1, ceil(PW_K * N^PW_ALPHA)); at the limit the outcome is redrawn from the
 *    distribution restricted to existing children;
 *  - every node holds its position, so a node's statistics and children always describe one
 *    state (the old tree reused a decision node across different sampled opponent moves and
 *    could pick a hero the opponent had just taken).
 * Transitions and legal actions come from ./rules (the board reducer's own rules: Cho'gall
 * fills two pick slots, either half blocks both, and needs two consecutive open slots).
 * The leaf is the first unexpanded decision node or a complete draft, as before.
 */
const PW_K = 1.0
const PW_ALPHA = 0.5

type NodeKind = 'decision' | 'chance' | 'terminal'

interface MCTSNode {
  kind: NodeKind
  sel: Selections
  step: number
  action: number
  parent: MCTSNode | null
  children: Map<number, MCTSNode>
  visitCount: number
  valueSum: number
  /** decision: tempered policy priors over legal actions; chance: GD probabilities over legal actions */
  probs: Float32Array | null
}

function kindAt(step: number, ourTeam: number): NodeKind {
  if (step >= DRAFT_SEQUENCE.length) return 'terminal'
  return teamIdx(DRAFT_SEQUENCE[step].team) === ourTeam ? 'decision' : 'chance'
}

const teamIdx = (t: 'A' | 'B') => (t === 'A' ? 0 : 1)

function makeNode(sel: Selections, step: number, action: number, parent: MCTSNode | null, ourTeam: number): MCTSNode {
  return { kind: kindAt(step, ourTeam), sel, step, action, parent, children: new Map(), visitCount: 0, valueSum: 0, probs: null }
}

/** Legal-action mask of a position, encoding order. */
function legalMask(sel: Selections, step: number, extraTaken: Set<string>): Float32Array {
  const mask = new Float32Array(NUM_HEROES)
  const blocked = unavailableAt(sel, step)
  for (let i = 0; i < NUM_HEROES; i++) mask[i] = blocked.has(HEROES[i]) || extraTaken.has(HEROES[i]) ? 0 : 1
  return mask
}

/** First index whose cumulative mass exceeds r (zero entries skipped; last positive on overrun). */
function sampleIndex(p: Float32Array, r: number, allowed?: (i: number) => boolean): number {
  let tot = 0
  for (let i = 0; i < p.length; i++) if (p[i] > 0 && (!allowed || allowed(i))) tot += p[i]
  const x0 = r * tot
  let cum = 0
  let last = -1
  for (let i = 0; i < p.length; i++) {
    if (p[i] <= 0 || (allowed && !allowed(i))) continue
    last = i
    cum += p[i]
    if (cum > x0) return i
  }
  return last
}

function heroesToMultiHot(names: string[]): Float32Array {
  const vec = new Float32Array(NUM_HEROES)
  for (const name of names) {
    const idx = HERO_TO_IDX[name]
    if (idx !== undefined) vec[idx] = 1
  }
  return vec
}

function encodeState(
  team0Picks: string[], team1Picks: string[], bans: string[],
  map: string, tier: string, step: number, stepType: 'ban' | 'pick', ourTeam: number,
): Float32Array {
  const input = new Float32Array(STATE_DIM)
  let offset = 0
  input.set(heroesToMultiHot(team0Picks), offset); offset += NUM_HEROES
  input.set(heroesToMultiHot(team1Picks), offset); offset += NUM_HEROES
  input.set(heroesToMultiHot(bans), offset); offset += NUM_HEROES
  const mapIdx = MAP_TO_IDX[map]
  if (mapIdx !== undefined) input[offset + mapIdx] = 1
  offset += NUM_MAPS
  const tierIdx = TIER_TO_IDX[tier]
  if (tierIdx !== undefined) input[offset + tierIdx] = 1
  offset += NUM_TIERS
  input[offset++] = step / 15.0
  input[offset++] = stepType === 'pick' ? 1.0 : 0.0
  input[offset++] = ourTeam
  return input
}

/**
 * Rebuild the board's selections from the search input (picks per team and bans, all at
 * steps before `step`). Which skipped ban slot is empty does not matter: those steps are past.
 */
export function selectionsFromLineups(team0: string[], team1: string[], bans: string[], step: number): Selections {
  const sel: Selections = {}
  const queues = { A: [...team0], B: [...team1], ban: [...bans] }
  for (let i = 0; i < Math.min(step, DRAFT_SEQUENCE.length); i++) {
    const s = DRAFT_SEQUENCE[i]
    const q = s.type === 'ban' ? queues.ban : queues[s.team]
    const h = q.shift()
    if (h) sel[i] = h
  }
  return sel
}

/**
 * Run MCTS search on the main thread using pre-loaded ONNX sessions.
 */
export interface MCTSRecommendation {
  hero: string
  /** Fraction of root visits (0-1) */
  visits: number
  /** Mean backed-up value: P(our team wins) if we take this hero */
  q: number
}

export async function runMCTSSearch(
  ort: any,
  policySession: any,
  gdSession: any,
  draftState: {
    team0Picks: string[], team1Picks: string[], bans: string[],
    map: string, tier: string, step: number, ourTeam: number,
    stepType: 'ban' | 'pick',
  },
  takenHeroes: Set<string>,
  withLock: <T>(fn: () => Promise<T>) => Promise<T> = (fn) => fn(),
  /** Optional WP leaf evaluator for complete drafts: returns P(team0 wins). */
  evaluateTerminal?: (team0: string[], team1: string[]) => Promise<number>,
  /**
   * Optional partial-draft leaf evaluator for INCOMPLETE states: returns
   * P(team0 wins) given current picks and the index of the last completed
   * draft action. When provided it replaces the policy value head at
   * non-terminal leaves (the head measured as state-insensitive,
   * 2026-08-07), making backed-up Q state-sensitive; ban values then emerge
   * from search dynamics (a banned hero is unavailable in every branch).
   */
  evaluatePartial?: (team0: string[], team1: string[], lastActionIdx: number) => Promise<number>,
  /**
   * Test-mode overrides. Defaults reproduce production behavior exactly:
   * rng = Math.random, and the standard MIN/MAX_SIMS + time budget. Used by the
   * parity oracle to drive a deterministic (shared-RNG, fixed-sim) search.
   */
  options?: {
    rng?: () => number; maxSims?: number; minSims?: number; timeBudgetMs?: number
    /** Test hook: called with every position about to be evaluated and its legality problems */
    onEvaluate?: (sel: Selections, problems: string[]) => void
  },
): Promise<{ recommendations: MCTSRecommendation[]; valueEstimate: number; sims: number; illegal: number }> {

  const ourTeam = draftState.ourTeam
  const rng = options?.rng ?? Math.random
  const maxSims = options?.maxSims ?? MAX_SIMS
  const minSims = options?.minSims ?? MIN_SIMS
  const timeBudgetMs = options?.timeBudgetMs ?? TIME_BUDGET_MS

  async function runPolicy(state: Float32Array, mask: Float32Array) {
    const stateTensor = new ort.Tensor('float32', state, [1, STATE_DIM])
    const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])
    const result: any = await withLock(() => policySession.run({ state: stateTensor, valid_mask: maskTensor }))
    const logits = result.policy_logits.data as Float32Array
    const value = (result.value.data as Float32Array)[0]
    // Apply temperature to policy logits before softmax.
    // The RL-trained policy is near-deterministic (99%+ on top hero).
    // Temperature softens priors so MCTS can explore alternatives.
    const PRIOR_TEMP = 3.0
    const tempered = new Float32Array(logits.length)
    for (let i = 0; i < logits.length; i++) tempered[i] = logits[i] / PRIOR_TEMP
    return { priors: softmaxMasked(tempered, mask), value }
  }

  /** GD opponent distribution over the position's legal actions. */
  async function runGD(state: Float32Array, mask: Float32Array): Promise<Float32Array> {
    const gdState = state.slice(0, STATE_DIM - 1)
    const stateTensor = new ort.Tensor('float32', gdState, [1, STATE_DIM - 1])
    const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])
    const result: any = await withLock(() => gdSession.run({ state: stateTensor, valid_mask: maskTensor }))
    const logits = result.hero_logits.data as Float32Array
    return softmaxMasked(logits, mask)
  }

  // Root position. Cho/Gall in takenHeroes may be a temporary root-only block (not enough
  // open pick slots this turn); the rules recompute that at every simulated step, so only
  // other extra exclusions are kept.
  const rootSel = selectionsFromLineups(draftState.team0Picks, draftState.team1Picks, draftState.bans, draftState.step)
  const extraTaken = new Set<string>()
  const drafted = new Set(Object.values(rootSel))
  for (const h of takenHeroes) if (!drafted.has(h) && h !== CHO && h !== GALL && HERO_TO_IDX[h] !== undefined) extraTaken.add(h)

  const encodeAt = (node: MCTSNode) => {
    const { A, B, bans } = lineups(node.sel)
    const stepType: 'ban' | 'pick' = node.step < DRAFT_SEQUENCE.length ? DRAFT_SEQUENCE[node.step].type : 'pick'
    return {
      state: encodeState(A, B, bans, draftState.map, draftState.tier, node.step, stepType, ourTeam),
      mask: legalMask(node.sel, node.step, extraTaken),
    }
  }

  let illegal = 0
  const assertLegal = (node: MCTSNode) => {
    const problems = positionProblems(node.sel, node.kind === 'terminal')
    options?.onEvaluate?.(node.sel, problems)
    if (problems.length) {
      illegal++
      throw new Error(`search reached an illegal position: ${problems.join('; ')}`)
    }
  }

  const evaluate = async (node: MCTSNode, policyValue: () => Promise<number>): Promise<number> => {
    assertLegal(node)
    const { A, B } = lineups(node.sel)
    if (node.kind === 'terminal') {
      if (evaluateTerminal) {
        const wp0 = await evaluateTerminal(A, B)
        return ourTeam === 0 ? wp0 : 1 - wp0
      }
      return policyValue()
    }
    if (evaluatePartial) {
      const p = await evaluatePartial(A, B, Math.max(0, Math.min(15, node.step - 1)))
      return ourTeam === 0 ? p : 1 - p
    }
    return policyValue()
  }

  const expandDecision = async (node: MCTSNode): Promise<number> => {
    const { state, mask } = encodeAt(node)
    const { priors, value } = await runPolicy(state, mask)
    node.probs = priors
    return value
  }

  const root = makeNode(rootSel, draftState.step, -1, null, ourTeam)
  if (root.kind !== 'decision') throw new Error('search root must be one of our decisions')
  const rootValue = await expandDecision(root)

  const childFor = (node: MCTSNode, a: number): MCTSNode => {
    let child = node.children.get(a)
    if (!child) {
      const next = applySelection(node.sel, node.step, HEROES[a])
      child = makeNode(next.selections, next.step, a, node, ourTeam)
      node.children.set(a, child)
    }
    return child
  }

  const startTime = performance.now()
  let simsRun = 0

  for (let sim = 0; sim < maxSims; sim++) {
    if (sim >= minSims && performance.now() - startTime > timeBudgetMs) break
    simsRun++

    let node = root
    const path: MCTSNode[] = [root]
    let value: number
    for (;;) {
      if (node.kind === 'terminal') {
        value = await evaluate(node, async () => (await runPolicy(encodeAt(node).state, encodeAt(node).mask)).value)
        break
      }
      if (node.kind === 'decision') {
        if (!node.probs) {
          const v = await expandDecision(node)
          value = await evaluate(node, async () => v)
          break
        }
        const sq = Math.sqrt(node.visitCount)
        let best = -Infinity
        let bestA = -1
        for (let a = 0; a < NUM_HEROES; a++) {
          const p = node.probs[a]
          if (!(p > 0)) continue
          const c = node.children.get(a)
          const n = c ? c.visitCount : 0
          const q = c && c.visitCount > 0 ? c.valueSum / c.visitCount : 0
          const score = q + C_PUCT * p * sq / (1 + n)
          if (score > best) { best = score; bestA = a }
        }
        if (bestA < 0) throw new Error('search found no legal action at a decision node')
        node = childFor(node, bestA)
      } else {
        if (!node.probs) {
          const { state, mask } = encodeAt(node)
          node.probs = await runGD(state, mask)
        }
        const r = 1 - rng()
        const limit = Math.max(1, Math.ceil(PW_K * Math.pow(node.visitCount, PW_ALPHA) - 1e-3))
        const a = node.children.size < limit
          ? sampleIndex(node.probs, r)
          : sampleIndex(node.probs, r, i => node.children.has(i))
        if (a < 0) throw new Error('search found no legal action at a chance node')
        node = childFor(node, a)
      }
      path.push(node)
    }

    for (const n of path) {
      n.visitCount++
      n.valueSum += value
    }
  }

  // Build recommendations (visit fraction + mean action value)
  const recommendations: MCTSRecommendation[] = []
  let visitSum = 0
  for (const [, child] of root.children) visitSum += child.visitCount
  for (const [action, child] of root.children) {
    if (child.visitCount > 0) {
      recommendations.push({
        hero: HEROES[action],
        visits: visitSum > 0 ? child.visitCount / visitSum : 0,
        q: child.valueSum / child.visitCount,
      })
    }
  }
  recommendations.sort((a, b) => b.visits - a.visits)

  // Search-refined estimate of the current position; falls back to the
  // network's direct value estimate when no simulations completed.
  const valueEstimate = root.visitCount > 0 ? root.valueSum / root.visitCount : rootValue

  return { recommendations, valueEstimate, sims: simsRun, illegal }
}
