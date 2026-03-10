/**
 * Browser-side ONNX inference for Draft Insights Hyper Pro Max.
 *
 * Loads onnxruntime-web from CDN (to avoid webpack bundling issues with
 * import.meta in the .mjs files), then loads three ONNX models:
 * - draft_policy.onnx: AlphaZero network (policy + value heads)
 * - generic_draft_0.onnx: Generic Draft model (opponent simulation)
 * - win_probability.onnx: Win Probability model (leaf evaluation)
 *
 * Hero encoding uses alphabetically sorted hero names (must match training).
 */
import { DRAFT_SEQUENCE } from './types'

// 90 heroes sorted alphabetically — must match training/shared.py exactly
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

const NUM_HEROES = HEROES.length // 90
const HERO_TO_IDX: Record<string, number> = {}
HEROES.forEach((h, i) => { HERO_TO_IDX[h] = i })

const MAPS = [
  "Alterac Pass", "Battlefield of Eternity", "Blackheart's Bay",
  "Braxis Holdout", "Cursed Hollow", "Dragon Shire",
  "Garden of Terror", "Hanamura Temple", "Infernal Shrines",
  "Sky Temple", "Tomb of the Spider Queen", "Towers of Doom",
  "Volskaya Foundry", "Warhead Junction",
]
const NUM_MAPS = MAPS.length // 14
const MAP_TO_IDX: Record<string, number> = {}
MAPS.forEach((m, i) => { MAP_TO_IDX[m] = i })

const SKILL_TIERS = ["low", "mid", "high"]
const NUM_TIERS = SKILL_TIERS.length // 3
const TIER_TO_IDX: Record<string, number> = {}
SKILL_TIERS.forEach((t, i) => { TIER_TO_IDX[t] = i })

// ── ORT loading via CDN ─────────────────────────────────────────────

let ort: any = null
let policySession: any = null
let gdSession: any = null
let wpSession: any = null
let loadPromise: Promise<void> | null = null

export function isAILoaded(): boolean {
  return policySession !== null && gdSession !== null && wpSession !== null
}

export async function loadAIModels(): Promise<void> {
  if (loadPromise) return loadPromise
  loadPromise = _loadModels()
  return loadPromise
}

async function loadOrtFromCDN(): Promise<any> {
  // Check if already loaded via script tag
  if ((window as any).ort) return (window as any).ort

  return new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js'
    script.onload = () => {
      const ortLib = (window as any).ort
      if (ortLib) resolve(ortLib)
      else reject(new Error('ort not found on window after loading script'))
    }
    script.onerror = () => reject(new Error('Failed to load onnxruntime-web from CDN'))
    document.head.appendChild(script)
  })
}

async function _loadModels(): Promise<void> {
  try {
    ort = await loadOrtFromCDN()

    const [p, g, w] = await Promise.all([
      ort.InferenceSession.create('/models/draft_policy.onnx'),
      ort.InferenceSession.create('/models/generic_draft_0.onnx'),
      ort.InferenceSession.create('/models/win_probability.onnx'),
    ])
    policySession = p
    gdSession = g
    wpSession = w
    console.log('[AI] All 3 ONNX models loaded')
  } catch (err) {
    console.error('[AI] Failed to load ONNX models:', err)
    loadPromise = null
    throw err
  }
}

// ── Encoding helpers ────────────────────────────────────────────────

function heroesToMultiHot(names: string[]): Float32Array {
  const vec = new Float32Array(NUM_HEROES)
  for (const name of names) {
    const idx = HERO_TO_IDX[name]
    if (idx !== undefined) vec[idx] = 1
  }
  return vec
}

function mapToOneHot(mapName: string): Float32Array {
  const vec = new Float32Array(NUM_MAPS)
  const idx = MAP_TO_IDX[mapName]
  if (idx !== undefined) vec[idx] = 1
  return vec
}

function tierToOneHot(tier: string): Float32Array {
  const vec = new Float32Array(NUM_TIERS)
  const idx = TIER_TO_IDX[tier]
  if (idx !== undefined) vec[idx] = 1
  return vec
}

// ── Draft state encoding ────────────────────────────────────────────

export interface AIDraftState {
  team0Picks: string[]
  team1Picks: string[]
  bans: string[]
  map: string
  tier: string
  step: number        // 0-15
  stepType: 'ban' | 'pick'
}

function encodeState(s: AIDraftState): Float32Array {
  const t0 = heroesToMultiHot(s.team0Picks)
  const t1 = heroesToMultiHot(s.team1Picks)
  const b = heroesToMultiHot(s.bans)
  const m = mapToOneHot(s.map)
  const t = tierToOneHot(s.tier)

  // 90*3 + 14 + 3 + 2 = 289
  const input = new Float32Array(289)
  let offset = 0
  input.set(t0, offset); offset += NUM_HEROES
  input.set(t1, offset); offset += NUM_HEROES
  input.set(b, offset); offset += NUM_HEROES
  input.set(m, offset); offset += NUM_MAPS
  input.set(t, offset); offset += NUM_TIERS
  input[offset++] = s.step / 15.0
  input[offset++] = s.stepType === 'pick' ? 1.0 : 0.0
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

// ── Inference ───────────────────────────────────────────────────────

export interface AIRecommendation {
  hero: string
  /** Policy prior probability (0-1) */
  prior: number
  /** Estimated win probability adjusted for player skill */
  winProb: number
  /** MAWP adjustment applied (percentage points) */
  mawpAdj: number
  /** Best player for this hero (if any) */
  suggestedPlayer: string | null
}

/** Per-hero stats for a player, matching DraftData.playerStats shape */
export interface PlayerHeroStat {
  games: number
  wins: number
  winRate: number
  mawp: number | null
}

/** Player MAWP data passed from the draft engine */
export interface PlayerMAWPData {
  /** battletag → hero → stats */
  playerStats: Record<string, Record<string, PlayerHeroStat>>
  /** Battletags available (not yet assigned to a pick) */
  availableBattletags: string[]
}

const MAWP_WEIGHT = 0.04 // per-player adjustment weight
const MAWP_MIN_GAMES = 10
const MAWP_CONFIDENCE_THRESHOLD = 30

function confidenceAdjustedMawp(mawp: number, games: number): number {
  if (games >= MAWP_CONFIDENCE_THRESHOLD) return mawp
  const weight = games / MAWP_CONFIDENCE_THRESHOLD
  return mawp * weight + 50 * (1 - weight)
}

/**
 * Find the best available player for a hero and compute the MAWP adjustment.
 */
function computePlayerAdjustment(
  hero: string,
  playerData: PlayerMAWPData | undefined,
): { adjustment: number; player: string | null } {
  if (!playerData || playerData.availableBattletags.length === 0) {
    return { adjustment: 0, player: null }
  }

  let bestAdj = 0
  let bestPlayer: string | null = null

  for (const bt of playerData.availableBattletags) {
    const stats = playerData.playerStats[bt]?.[hero]
    if (!stats || stats.games < MAWP_MIN_GAMES) continue

    const adjMawp = stats.mawp != null
      ? confidenceAdjustedMawp(stats.mawp, stats.games)
      : (stats.wins / stats.games) * 100

    const delta = (adjMawp - 50) / 100 // normalize to [-0.5, 0.5]
    const adj = MAWP_WEIGHT * delta

    if (adj > bestAdj || bestPlayer === null) {
      bestAdj = adj
      bestPlayer = bt
    }
  }

  return { adjustment: bestAdj, player: bestPlayer }
}

/**
 * Compute aggregate MAWP adjustment for all heroes already picked by our team
 * that have assigned players. This adjusts the baseline WP estimate.
 * Checks both team0 and team1 picks since playerData.availableBattletags
 * only contains our team's unassigned players.
 */
function computeTeamMawpAdjustment(
  draftState: AIDraftState,
  playerData?: PlayerMAWPData,
): number {
  if (!playerData) return 0

  let totalAdj = 0
  // Check all picked heroes — computePlayerAdjustment will only match
  // heroes that our available battletags have stats for, so it naturally
  // only applies to our team's heroes.
  const allPicked = [...draftState.team0Picks, ...draftState.team1Picks]
  for (const hero of allPicked) {
    const { adjustment } = computePlayerAdjustment(hero, playerData)
    totalAdj += adjustment
  }
  return totalAdj
}

/**
 * Get just the value estimate (WP) for the current state, with MAWP adjustment.
 * Used on opponent turns when we don't need full recommendations.
 */
export async function getValueEstimate(
  draftState: AIDraftState,
  playerData?: PlayerMAWPData,
): Promise<number> {
  if (!policySession || !ort) {
    throw new Error('AI models not loaded. Call loadAIModels() first.')
  }

  const state = encodeState(draftState)
  const mask = buildValidMask(new Set([
    ...draftState.team0Picks,
    ...draftState.team1Picks,
    ...draftState.bans,
  ]))
  const stateTensor = new ort.Tensor('float32', state, [1, 289])
  const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])
  const result = await policySession.run({ state: stateTensor, valid_mask: maskTensor })
  const rawWP = (result.value.data as Float32Array)[0]

  const mawpAdj = computeTeamMawpAdjustment(draftState, playerData)
  return Math.max(0, Math.min(1, rawWP + mawpAdj))
}

/**
 * Get AI draft recommendations for the current state.
 *
 * For each valid hero, simulates picking/banning it and evaluates the
 * resulting state with the policy network's value head to get a per-hero
 * expected win probability. Heroes are ranked by this WP estimate.
 */
export async function getAIRecommendations(
  draftState: AIDraftState,
  takenHeroes: Set<string>,
  /** Which team is making this pick/ban: 'A' or 'B' */
  currentTeam: 'A' | 'B',
  playerData?: PlayerMAWPData,
  topK = 15,
): Promise<{ recommendations: AIRecommendation[]; valueEstimate: number }> {
  if (!policySession || !ort) {
    throw new Error('AI models not loaded. Call loadAIModels() first.')
  }

  // Get baseline value estimate for current state
  const state = encodeState(draftState)
  const mask = buildValidMask(takenHeroes)
  const stateTensor = new ort.Tensor('float32', state, [1, 289])
  const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])
  const baseResult = await policySession.run({ state: stateTensor, valid_mask: maskTensor })
  const rawValueEstimate = (baseResult.value.data as Float32Array)[0]

  // Compute aggregate MAWP adjustment for all already-assigned players
  const teamMawpAdj = computeTeamMawpAdjustment(draftState, playerData)
  const baseValueEstimate = Math.max(0, Math.min(1, rawValueEstimate + teamMawpAdj))

  // Collect valid hero indices
  const validIndices: number[] = []
  for (let i = 0; i < NUM_HEROES; i++) {
    if (mask[i] > 0) validIndices.push(i)
  }

  // Build a batch of "what-if" states: one per valid hero
  const batchSize = validIndices.length
  const batchStates = new Float32Array(batchSize * 289)
  const batchMasks = new Float32Array(batchSize * NUM_HEROES)

  const isBanStep = draftState.stepType === 'ban'
  const nextStep = draftState.step + 1
  const nextStepType: 'ban' | 'pick' = nextStep < DRAFT_SEQUENCE.length
    ? DRAFT_SEQUENCE[nextStep].type
    : 'pick'

  // Pre-compute base vectors once
  const baseT0 = heroesToMultiHot(draftState.team0Picks)
  const baseT1 = heroesToMultiHot(draftState.team1Picks)
  const baseBans = heroesToMultiHot(draftState.bans)
  const mapVec = mapToOneHot(draftState.map)
  const tierVec = tierToOneHot(draftState.tier)

  // Pre-compute base mask
  const baseMask = new Float32Array(NUM_HEROES).fill(1)
  for (const hero of takenHeroes) {
    const idx = HERO_TO_IDX[hero]
    if (idx !== undefined) baseMask[idx] = 0
  }

  for (let b = 0; b < batchSize; b++) {
    const heroIdx = validIndices[b]

    // Clone base picks/bans and apply this hero
    const nextT0 = new Float32Array(baseT0)
    const nextT1 = new Float32Array(baseT1)
    const nextBans = new Float32Array(baseBans)

    if (isBanStep) {
      nextBans[heroIdx] = 1
    } else if (currentTeam === 'A') {
      // Team A = team 0
      nextT0[heroIdx] = 1
    } else {
      nextT1[heroIdx] = 1
    }

    // Encode the resulting state (one step ahead)
    const offset = b * 289
    batchStates.set(nextT0, offset)
    batchStates.set(nextT1, offset + NUM_HEROES)
    batchStates.set(nextBans, offset + NUM_HEROES * 2)
    batchStates.set(mapVec, offset + NUM_HEROES * 3)
    batchStates.set(tierVec, offset + NUM_HEROES * 3 + NUM_MAPS)
    batchStates[offset + NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS] = nextStep / 15.0
    batchStates[offset + NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS + 1] = nextStepType === 'pick' ? 1.0 : 0.0

    // Mask: taken heroes + this hero
    const heroMask = new Float32Array(baseMask)
    heroMask[heroIdx] = 0
    batchMasks.set(heroMask, b * NUM_HEROES)
  }

  // Single batched inference for all candidates
  const batchStateTensor = new ort.Tensor('float32', batchStates, [batchSize, 289])
  const batchMaskTensor = new ort.Tensor('float32', batchMasks, [batchSize, NUM_HEROES])
  const batchResult = await policySession.run({ state: batchStateTensor, valid_mask: batchMaskTensor })
  const batchValues = batchResult.value.data as Float32Array

  // Build recommendations ranked by per-hero WP
  const ranked: AIRecommendation[] = []
  for (let b = 0; b < batchSize; b++) {
    const heroIdx = validIndices[b]
    const heroWp = batchValues[b]

    const { adjustment, player } = isBanStep
      ? { adjustment: 0, player: null }
      : computePlayerAdjustment(HEROES[heroIdx], playerData)

    ranked.push({
      hero: HEROES[heroIdx],
      prior: 0,  // not used for ranking anymore
      winProb: Math.max(0, Math.min(1, heroWp + adjustment)),
      mawpAdj: adjustment,
      suggestedPlayer: player,
    })
  }

  // For picks: rank by highest WP (we want to maximize our win prob)
  // For bans: rank by highest WP too (ban the hero that gives the best state for the opponent,
  //   i.e. from team 0's perspective, ban the hero whose removal hurts the opponent most)
  if (isBanStep) {
    // Banning a hero the opponent would pick: we want to ban heroes where
    // the post-ban state has the HIGHEST WP for us (team 0)
    ranked.sort((a, b) => b.winProb - a.winProb)
  } else {
    ranked.sort((a, b) => b.winProb - a.winProb)
  }

  return {
    recommendations: ranked.slice(0, topK),
    valueEstimate: baseValueEstimate,
  }
}


/**
 * Get Generic Draft model predictions for the current state.
 * Used for predicting what the opponent will pick/ban.
 */
export async function getGenericDraftPredictions(
  draftState: AIDraftState,
  takenHeroes: Set<string>,
  topK = 15,
): Promise<{ hero: string; probability: number }[]> {
  if (!gdSession || !ort) {
    throw new Error('AI models not loaded. Call loadAIModels() first.')
  }

  const state = encodeState(draftState)
  const mask = buildValidMask(takenHeroes)

  const stateTensor = new ort.Tensor('float32', state, [1, 289])
  const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])

  const result = await gdSession.run({
    state: stateTensor,
    valid_mask: maskTensor,
  })

  const logits = result.hero_logits.data as Float32Array
  const probs = softmaxMasked(logits, mask)

  const ranked: { hero: string; probability: number }[] = []
  for (let i = 0; i < NUM_HEROES; i++) {
    if (mask[i] > 0 && probs[i] > 0.001) {
      ranked.push({ hero: HEROES[i], probability: probs[i] })
    }
  }
  ranked.sort((a, b) => b.probability - a.probability)
  return ranked.slice(0, topK)
}

/**
 * Get win probability estimate for a completed draft.
 */
export async function getWinProbability(
  team0Heroes: string[],
  team1Heroes: string[],
  map: string,
  tier: string,
): Promise<number> {
  if (!wpSession || !ort) {
    throw new Error('AI models not loaded. Call loadAIModels() first.')
  }

  const t0 = heroesToMultiHot(team0Heroes)
  const t1 = heroesToMultiHot(team1Heroes)
  const m = mapToOneHot(map)
  const t = tierToOneHot(tier)

  // 90 + 90 + 14 + 3 = 197
  const input = new Float32Array(197)
  let offset = 0
  input.set(t0, offset); offset += NUM_HEROES
  input.set(t1, offset); offset += NUM_HEROES
  input.set(m, offset); offset += NUM_MAPS
  input.set(t, offset); offset += NUM_TIERS

  const tensor = new ort.Tensor('float32', input, [1, 197])
  const result = await wpSession.run({ input: tensor })
  return (result.win_probability.data as Float32Array)[0]
}

// ── Utilities ───────────────────────────────────────────────────────

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

export { HEROES as AI_HEROES, HERO_TO_IDX as AI_HERO_TO_IDX }
