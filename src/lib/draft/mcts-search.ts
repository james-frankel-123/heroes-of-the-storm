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

const NUM_HEROES = 90
const NUM_MAPS = 14
const NUM_TIERS = 3
const STATE_DIM = NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS + 2 + 1 // 290

const HEROES = [
  "Abathur","Alarak","Alexstrasza","Ana","Anduin","Anub'arak","Artanis",
  "Arthas","Auriel","Azmodan","Blaze","Brightwing","Cassia","Chen","Cho",
  "Chromie","D.Va","Deathwing","Deckard","Dehaka","Diablo","E.T.C.",
  "Falstad","Fenix","Gall","Garrosh","Gazlowe","Genji","Greymane",
  "Gul'dan","Hanzo","Hogger","Illidan","Imperius","Jaina","Johanna",
  "Junkrat","Kael'thas","Kel'Thuzad","Kerrigan","Kharazim","Leoric",
  "Li Li","Li-Ming","Lt. Morales","Lunara","Lúcio","Maiev","Mal'Ganis",
  "Malfurion","Malthael","Medivh","Mei","Mephisto","Muradin","Murky",
  "Nazeebo","Nova","Orphea","Probius","Qhira","Ragnaros","Raynor",
  "Rehgar","Rexxar","Samuro","Sgt. Hammer","Sonya","Stitches","Stukov",
  "Sylvanas","Tassadar","The Butcher","The Lost Vikings","Thrall","Tracer",
  "Tychus","Tyrael","Tyrande","Uther","Valeera","Valla","Varian",
  "Whitemane","Xul","Yrel","Zagara","Zarya","Zeratul","Zul'jin",
]
const HERO_TO_IDX: Record<string, number> = {}
HEROES.forEach((h, i) => { HERO_TO_IDX[h] = i })

const MAPS = [
  "Alterac Pass", "Battlefield of Eternity", "Blackheart's Bay",
  "Braxis Holdout", "Cursed Hollow", "Dragon Shire",
  "Garden of Terror", "Hanamura Temple", "Infernal Shrines",
  "Sky Temple", "Tomb of the Spider Queen", "Towers of Doom",
  "Volskaya Foundry", "Warhead Junction",
]
const MAP_TO_IDX: Record<string, number> = {}
MAPS.forEach((m, i) => { MAP_TO_IDX[m] = i })

const SKILL_TIERS = ["low", "mid", "high"]
const TIER_TO_IDX: Record<string, number> = {}
SKILL_TIERS.forEach((t, i) => { TIER_TO_IDX[t] = i })

const DRAFT_ORDER: [number, 'ban' | 'pick'][] = [
  [0, 'ban'], [1, 'ban'], [0, 'ban'], [1, 'ban'],
  [0, 'pick'], [1, 'pick'], [1, 'pick'], [0, 'pick'], [0, 'pick'],
  [1, 'ban'], [0, 'ban'],
  [1, 'pick'], [1, 'pick'], [0, 'pick'], [0, 'pick'], [1, 'pick'],
]

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

interface MCTSNode {
  action: number
  parent: MCTSNode | null
  children: Map<number, MCTSNode>
  visitCount: number
  valueSum: number
  prior: number
  isExpanded: boolean
}

function createNode(action: number, parent: MCTSNode | null, prior: number): MCTSNode {
  return { action, parent, children: new Map(), visitCount: 0, valueSum: 0, prior, isExpanded: false }
}

function ucbScore(node: MCTSNode): number {
  if (!node.parent) return 0
  const q = node.visitCount === 0 ? 0 : node.valueSum / node.visitCount
  return q + C_PUCT * node.prior * Math.sqrt(node.parent.visitCount) / (1 + node.visitCount)
}

interface DraftMCTSState {
  team0Picks: number[]
  team1Picks: number[]
  bans: number[]
  taken: Set<number>
  step: number
  map: string
  tier: string
  ourTeam: number
}

function cloneState(s: DraftMCTSState): DraftMCTSState {
  return {
    team0Picks: [...s.team0Picks], team1Picks: [...s.team1Picks],
    bans: [...s.bans], taken: new Set(s.taken),
    step: s.step, map: s.map, tier: s.tier, ourTeam: s.ourTeam,
  }
}

function applyAction(s: DraftMCTSState, heroIdx: number) {
  const [team, actionType] = DRAFT_ORDER[s.step]
  s.taken.add(heroIdx)
  if (actionType === 'ban') s.bans.push(heroIdx)
  else if (team === 0) s.team0Picks.push(heroIdx)
  else s.team1Picks.push(heroIdx)
  s.step++
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

function stateToTensors(s: DraftMCTSState) {
  const t0Names = s.team0Picks.map(i => HEROES[i])
  const t1Names = s.team1Picks.map(i => HEROES[i])
  const banNames = s.bans.map(i => HEROES[i])
  const stepType: 'ban' | 'pick' = s.step < 16 ? DRAFT_ORDER[s.step][1] : 'pick'
  const state = encodeState(t0Names, t1Names, banNames, s.map, s.tier, s.step, stepType, s.ourTeam)
  const mask = new Float32Array(NUM_HEROES).fill(1)
  for (const idx of s.taken) mask[idx] = 0
  return { state, mask }
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
): Promise<{ recommendations: MCTSRecommendation[]; valueEstimate: number; sims: number }> {

  const ourTeam = draftState.ourTeam

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

  async function runGD(state: Float32Array, mask: Float32Array): Promise<number> {
    const gdState = state.slice(0, STATE_DIM - 1)
    const stateTensor = new ort.Tensor('float32', gdState, [1, STATE_DIM - 1])
    const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])
    const result: any = await withLock(() => gdSession.run({ state: stateTensor, valid_mask: maskTensor }))
    const logits = result.hero_logits.data as Float32Array
    const probs = softmaxMasked(logits, mask)
    const r = Math.random()
    let cumSum = 0
    for (let i = 0; i < NUM_HEROES; i++) {
      cumSum += probs[i]
      if (r < cumSum) return i
    }
    for (let i = 0; i < NUM_HEROES; i++) { if (mask[i] > 0) return i }
    return 0
  }

  // Build initial state
  const t0Idx = draftState.team0Picks.map(h => HERO_TO_IDX[h]).filter(i => i !== undefined)
  const t1Idx = draftState.team1Picks.map(h => HERO_TO_IDX[h]).filter(i => i !== undefined)
  const banIdx = draftState.bans.map(h => HERO_TO_IDX[h]).filter(i => i !== undefined)
  const taken = new Set<number>()
  for (const h of takenHeroes) {
    const idx = HERO_TO_IDX[h]
    if (idx !== undefined) taken.add(idx)
  }

  const rootState: DraftMCTSState = {
    team0Picks: t0Idx, team1Picks: t1Idx, bans: banIdx,
    taken, step: draftState.step, map: draftState.map, tier: draftState.tier, ourTeam,
  }

  const root = createNode(-1, null, 0)
  const { state: rootEncoded, mask: rootMask } = stateToTensors(rootState)
  const { priors, value: rootValue } = await runPolicy(rootEncoded, rootMask)

  root.isExpanded = true
  for (let a = 0; a < NUM_HEROES; a++) {
    if (rootMask[a] > 0) root.children.set(a, createNode(a, root, priors[a]))
  }

  const startTime = performance.now()
  let simsRun = 0

  for (let sim = 0; sim < MAX_SIMS; sim++) {
    if (sim >= MIN_SIMS && performance.now() - startTime > TIME_BUDGET_MS) break
    simsRun++

    let node = root
    const scratch = cloneState(rootState)

    // Selection
    while (node.isExpanded && scratch.step < 16) {
      const currentTeam = DRAFT_ORDER[scratch.step][0]
      if (currentTeam === ourTeam) {
        let bestScore = -Infinity
        let bestChild: MCTSNode | null = null
        for (const child of node.children.values()) {
          const score = ucbScore(child)
          if (score > bestScore) { bestScore = score; bestChild = child }
        }
        if (!bestChild) break
        applyAction(scratch, bestChild.action)
        node = bestChild
      } else {
        const { state: s, mask: m } = stateToTensors(scratch)
        const oppAction = await runGD(s, m)
        applyAction(scratch, oppAction)
      }
    }

    // Roll the draft forward through any opponent steps with GD sampling so
    // the leaf we expand/evaluate sits at OUR next decision (or terminal).
    // Without this, a leaf whose next step belongs to the opponent could
    // never expand and the tree would stay one ply deep.
    while (scratch.step < 16 && DRAFT_ORDER[scratch.step][0] !== ourTeam) {
      const { state: s, mask: m } = stateToTensors(scratch)
      const oppAction = await runGD(s, m)
      applyAction(scratch, oppAction)
    }

    let value: number
    if (scratch.step >= 16) {
      // Terminal: full 10-hero draft. Prefer the dedicated WP model
      // (leaf evaluator per spec); fall back to the policy value head.
      if (evaluateTerminal) {
        const wpTeam0 = await evaluateTerminal(
          scratch.team0Picks.map(i => HEROES[i]),
          scratch.team1Picks.map(i => HEROES[i]),
        )
        value = ourTeam === 0 ? wpTeam0 : 1 - wpTeam0
      } else {
        const { state: s, mask: m } = stateToTensors(scratch)
        const { value: v } = await runPolicy(s, m)
        value = v
      }
    } else if (!node.isExpanded) {
      // Expand at our decision point: policy priors + value estimate
      const { state: s, mask: m } = stateToTensors(scratch)
      const { priors: leafPriors, value: v } = await runPolicy(s, m)
      value = v
      node.isExpanded = true
      for (let a = 0; a < NUM_HEROES; a++) {
        if (m[a] > 0) node.children.set(a, createNode(a, node, leafPriors[a]))
      }
    } else {
      const { state: s, mask: m } = stateToTensors(scratch)
      const { value: v } = await runPolicy(s, m)
      value = v
    }

    // Backpropagate
    let current: MCTSNode | null = node
    while (current) {
      current.visitCount++
      current.valueSum += value
      current = current.parent
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

  return { recommendations, valueEstimate, sims: simsRun }
}
