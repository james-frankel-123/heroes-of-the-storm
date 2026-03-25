/* eslint-disable no-restricted-globals */
declare function importScripts(...urls: string[]): void

/**
 * Web Worker for browser-side MCTS inference.
 *
 * Loads quantized ONNX models via onnxruntime-web (WASM backend),
 * runs MCTS search with adaptive simulation count and time budgeting.
 *
 * Communication protocol:
 *   Main → Worker: { type: 'init' } | { type: 'search', state, takenHeroes, ourTeam, stepType }
 *   Worker → Main: { type: 'ready' } | { type: 'result', recommendations, valueEstimate }
 *                | { type: 'error', message }
 */

// Draft order matching DRAFT_SEQUENCE in types.ts
// (team 0 = A = bans first)
const DRAFT_ORDER: [number, 'ban' | 'pick'][] = [
  [0, 'ban'], [1, 'ban'], [0, 'ban'], [1, 'ban'],
  [0, 'pick'], [1, 'pick'], [1, 'pick'], [0, 'pick'], [0, 'pick'],
  [1, 'ban'], [0, 'ban'],
  [1, 'pick'], [1, 'pick'], [0, 'pick'], [0, 'pick'], [1, 'pick'],
]

const NUM_HEROES = 90
const NUM_MAPS = 14
const NUM_TIERS = 3
const STATE_DIM = NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS + 2 + 1 // 290 (last = our_team)

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

const C_PUCT = 2.0
const TIME_BUDGET_MS = 400

// ── ORT globals ──
let ort: any = null
let policySession: any = null
let gdSession: any = null

// ── Encoding helpers ──

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
  map: string, tier: string, step: number, stepType: 'ban' | 'pick',
  ourTeam: number,
): Float32Array {
  const input = new Float32Array(STATE_DIM)
  const t0 = heroesToMultiHot(team0Picks)
  const t1 = heroesToMultiHot(team1Picks)
  const b = heroesToMultiHot(bans)
  let offset = 0
  input.set(t0, offset); offset += NUM_HEROES
  input.set(t1, offset); offset += NUM_HEROES
  input.set(b, offset); offset += NUM_HEROES
  // Map one-hot
  const mapIdx = MAP_TO_IDX[map]
  if (mapIdx !== undefined) input[offset + mapIdx] = 1
  offset += NUM_MAPS
  // Tier one-hot
  const tierIdx = TIER_TO_IDX[tier]
  if (tierIdx !== undefined) input[offset + tierIdx] = 1
  offset += NUM_TIERS
  input[offset++] = step / 15.0
  input[offset++] = stepType === 'pick' ? 1.0 : 0.0
  input[offset++] = ourTeam
  return input
}

function buildValidMask(taken: Set<string>): Float32Array {
  const mask = new Float32Array(NUM_HEROES).fill(1)
  for (const hero of taken) {
    const idx = HERO_TO_IDX[hero]
    if (idx !== undefined) mask[idx] = 0
  }
  return mask
}

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

// ── MCTS Node ──

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
  return {
    action, parent,
    children: new Map(),
    visitCount: 0, valueSum: 0, prior,
    isExpanded: false,
  }
}

function qValue(node: MCTSNode): number {
  return node.visitCount === 0 ? 0 : node.valueSum / node.visitCount
}

function ucbScore(node: MCTSNode): number {
  if (!node.parent) return 0
  const exploration = C_PUCT * node.prior * Math.sqrt(node.parent.visitCount) / (1 + node.visitCount)
  return qValue(node) + exploration
}

// ── Draft state for MCTS simulation ──

interface DraftMCTSState {
  team0Picks: number[]  // hero indices
  team1Picks: number[]
  bans: number[]
  taken: Set<number>
  step: number
  map: string
  tier: string
  ourTeam: number  // 0 or 1
}

function cloneState(s: DraftMCTSState): DraftMCTSState {
  return {
    team0Picks: [...s.team0Picks],
    team1Picks: [...s.team1Picks],
    bans: [...s.bans],
    taken: new Set(s.taken),
    step: s.step,
    map: s.map,
    tier: s.tier,
    ourTeam: s.ourTeam,
  }
}

function applyAction(s: DraftMCTSState, heroIdx: number) {
  const [team, actionType] = DRAFT_ORDER[s.step]
  s.taken.add(heroIdx)
  if (actionType === 'ban') {
    s.bans.push(heroIdx)
  } else if (team === 0) {
    s.team0Picks.push(heroIdx)
  } else {
    s.team1Picks.push(heroIdx)
  }
  s.step++
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

// ── Inference helpers ──

async function runPolicy(state: Float32Array, mask: Float32Array): Promise<{ priors: Float32Array; value: number }> {
  const stateTensor = new ort.Tensor('float32', state, [1, STATE_DIM])
  const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])
  const result = await policySession.run({ state: stateTensor, valid_mask: maskTensor })
  const logits = result.policy_logits.data as Float32Array
  const value = (result.value.data as Float32Array)[0]
  const priors = softmaxMasked(logits, mask)
  return { priors, value }
}

async function runGD(state: Float32Array, mask: Float32Array): Promise<number> {
  // GD model expects 289 dims (no ourTeam indicator)
  const gdState = state.slice(0, STATE_DIM - 1)
  const stateTensor = new ort.Tensor('float32', gdState, [1, STATE_DIM - 1])
  const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])
  const result = await gdSession.run({ state: stateTensor, valid_mask: maskTensor })
  const logits = result.hero_logits.data as Float32Array
  const probs = softmaxMasked(logits, mask)
  // Weighted random sample
  const r = Math.random()
  let cumSum = 0
  for (let i = 0; i < NUM_HEROES; i++) {
    cumSum += probs[i]
    if (r < cumSum) return i
  }
  // Fallback: first valid
  for (let i = 0; i < NUM_HEROES; i++) {
    if (mask[i] > 0) return i
  }
  return 0
}

// ── MCTS Search ──

async function mctsSearch(
  rootState: DraftMCTSState,
  ourTeam: number,
  maxSimulations: number,
  timeBudgetMs: number,
): Promise<{ visits: Float32Array; value: number }> {
  const root = createNode(-1, null, 0)
  const { state: rootEncoded, mask: rootMask } = stateToTensors(rootState)
  const { priors, value: rootValue } = await runPolicy(rootEncoded, rootMask)

  // Expand root
  root.isExpanded = true
  for (let a = 0; a < NUM_HEROES; a++) {
    if (rootMask[a] > 0) {
      root.children.set(a, createNode(a, root, priors[a]))
    }
  }

  const startTime = performance.now()

  for (let sim = 0; sim < maxSimulations; sim++) {
    // Time budget check
    if (performance.now() - startTime > timeBudgetMs) break

    let node = root
    const scratch = cloneState(rootState)

    // Selection
    while (node.isExpanded && scratch.step < 16) {
      const currentTeam = DRAFT_ORDER[scratch.step][0]
      if (currentTeam === ourTeam) {
        // Our turn: UCB selection
        let bestScore = -Infinity
        let bestChild: MCTSNode | null = null
        for (const child of node.children.values()) {
          const score = ucbScore(child)
          if (score > bestScore) {
            bestScore = score
            bestChild = child
          }
        }
        if (!bestChild) break
        applyAction(scratch, bestChild.action)
        node = bestChild
      } else {
        // Opponent: GD model sample
        const { state: s, mask: m } = stateToTensors(scratch)
        const oppAction = await runGD(s, m)
        applyAction(scratch, oppAction)
        // Opponent nodes are pass-through
      }
    }

    let value: number
    if (scratch.step >= 16) {
      // Terminal: network value head (ourTeam in encoding, no flip needed)
      const { state: s, mask: m } = stateToTensors(scratch)
      const { value: v } = await runPolicy(s, m)
      value = v
    } else if (!node.isExpanded && DRAFT_ORDER[scratch.step][0] === ourTeam) {
      // Expand leaf
      const { state: s, mask: m } = stateToTensors(scratch)
      const { priors: leafPriors, value: v } = await runPolicy(s, m)
      value = v
      node.isExpanded = true
      for (let a = 0; a < NUM_HEROES; a++) {
        if (m[a] > 0) {
          node.children.set(a, createNode(a, node, leafPriors[a]))
        }
      }
    } else {
      // Leaf at opponent's turn — use network value
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

  // Build visit distribution
  const visits = new Float32Array(NUM_HEROES)
  for (const [action, child] of root.children) {
    visits[action] = child.visitCount
  }
  const visitSum = visits.reduce((a, b) => a + b, 0)
  if (visitSum > 0) {
    for (let i = 0; i < visits.length; i++) visits[i] /= visitSum
  }

  return { visits, value: rootValue }
}

// ── Worker message handling ──

async function loadModels() {
  // Load onnxruntime-web via importScripts (CDN)
  importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js')
  ort = (self as any).ort

  // Force WASM backend
  ort.env.wasm.numThreads = 1

  const tryLoad = async (int8: string, fp32: string) => {
    try { return await ort.InferenceSession.create(int8) }
    catch { return await ort.InferenceSession.create(fp32) }
  }
  const [p, g] = await Promise.all([
    tryLoad('/models/draft_policy_int8.onnx', '/models/draft_policy.onnx'),
    tryLoad('/models/generic_draft_0_int8.onnx', '/models/generic_draft_0.onnx'),
  ])
  policySession = p
  gdSession = g
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data

  if (msg.type === 'init') {
    try {
      await loadModels()
      self.postMessage({ type: 'ready' })
    } catch (err: any) {
      self.postMessage({ type: 'error', message: err.message })
    }
    return
  }

  if (msg.type === 'search') {
    try {
      const { team0Picks, team1Picks, bans, map, tier, step, ourTeam, takenHeroes } = msg

      // Convert hero names to indices for internal state
      const t0Idx = (team0Picks as string[]).map(h => HERO_TO_IDX[h]).filter(i => i !== undefined)
      const t1Idx = (team1Picks as string[]).map(h => HERO_TO_IDX[h]).filter(i => i !== undefined)
      const banIdx = (bans as string[]).map(h => HERO_TO_IDX[h]).filter(i => i !== undefined)
      const taken = new Set<number>()
      for (const h of takenHeroes as string[]) {
        const idx = HERO_TO_IDX[h]
        if (idx !== undefined) taken.add(idx)
      }

      const state: DraftMCTSState = {
        team0Picks: t0Idx, team1Picks: t1Idx, bans: banIdx,
        taken, step, map, tier, ourTeam,
      }

      // Adaptive simulation count: max(30, min(150, legalActions * 2))
      const legalActions = NUM_HEROES - taken.size
      const maxSims = Math.max(30, Math.min(150, legalActions * 2))

      const { visits, value } = await mctsSearch(state, ourTeam, maxSims, TIME_BUDGET_MS)

      // Build recommendations sorted by visit count
      const recommendations: { hero: string; visits: number; winProb: number }[] = []
      for (let i = 0; i < NUM_HEROES; i++) {
        if (visits[i] > 0) {
          recommendations.push({
            hero: HEROES[i],
            visits: visits[i],
            winProb: value, // base WP from root
          })
        }
      }
      recommendations.sort((a, b) => b.visits - a.visits)

      self.postMessage({
        type: 'result',
        recommendations: recommendations.slice(0, 15),
        valueEstimate: value,  // already from ourTeam's perspective
      })
    } catch (err: any) {
      self.postMessage({ type: 'error', message: err.message })
    }
  }
}
