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
import { DRAFT_SEQUENCE, type DraftData } from './types'
import { HERO_ROLES } from '@/lib/data/hero-roles'

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
let mctsWorker: Worker | null = null
let mctsWorkerReady = false

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

    // Load quantized models (INT8) with float32 fallback
    const tryLoad = async (int8Path: string, floatPath: string) => {
      try {
        return await ort.InferenceSession.create(int8Path)
      } catch {
        console.warn(`[AI] INT8 model failed, falling back to float32: ${floatPath}`)
        return await ort.InferenceSession.create(floatPath)
      }
    }

    const [p, g, w] = await Promise.all([
      tryLoad('/models/draft_policy_int8.onnx', '/models/draft_policy.onnx'),
      tryLoad('/models/generic_draft_0_int8.onnx', '/models/generic_draft_0.onnx'),
      tryLoad('/models/win_probability_int8.onnx', '/models/win_probability.onnx'),
    ])
    policySession = p
    gdSession = g
    wpSession = w
    console.log('[AI] ONNX models loaded')

    // Initialize MCTS Web Worker
    try {
      mctsWorker = new Worker(
        new URL('./mcts-worker.ts', import.meta.url),
        { type: 'module' }
      )
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Worker init timeout')), 15000)
        mctsWorker!.onmessage = (e) => {
          if (e.data.type === 'ready') {
            clearTimeout(timeout)
            mctsWorkerReady = true
            resolve()
          } else if (e.data.type === 'error') {
            clearTimeout(timeout)
            reject(new Error(e.data.message))
          }
        }
        mctsWorker!.postMessage({ type: 'init' })
      })
      console.log('[AI] MCTS Web Worker ready')
    } catch (err) {
      console.warn('[AI] MCTS Worker failed to init, will use direct inference:', err)
      mctsWorker = null
      mctsWorkerReady = false
    }
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
  ourTeam: number     // 0 or 1 — whose perspective
}

function encodeState(s: AIDraftState): Float32Array {
  const t0 = heroesToMultiHot(s.team0Picks)
  const t1 = heroesToMultiHot(s.team1Picks)
  const b = heroesToMultiHot(s.bans)
  const m = mapToOneHot(s.map)
  const t = tierToOneHot(s.tier)

  // 90*3 + 14 + 3 + 2 + 1 = 290 (last = our_team indicator)
  const input = new Float32Array(290)
  let offset = 0
  input.set(t0, offset); offset += NUM_HEROES
  input.set(t1, offset); offset += NUM_HEROES
  input.set(b, offset); offset += NUM_HEROES
  input.set(m, offset); offset += NUM_MAPS
  input.set(t, offset); offset += NUM_TIERS
  input[offset++] = s.step / 15.0
  input[offset++] = s.stepType === 'pick' ? 1.0 : 0.0
  input[offset++] = s.ourTeam
  return input
}

/** Encode state without ourTeam indicator (289 dims) for GD model. */
function encodeStateForGD(s: AIDraftState): Float32Array {
  const full = encodeState(s)
  return full.slice(0, 289)
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
  const stateTensor = new ort.Tensor('float32', state, [1, 290])
  const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])
  const result = await policySession.run({ state: stateTensor, valid_mask: maskTensor })
  const rawWP = (result.value.data as Float32Array)[0]

  const mawpAdj = computeTeamMawpAdjustment(draftState, playerData)
  return Math.max(0, Math.min(1, rawWP + mawpAdj))
}

/**
 * Run MCTS via Web Worker. Returns recommendations ranked by visit count.
 */
async function mctsSearch(
  draftState: AIDraftState,
  takenHeroes: Set<string>,
  ourTeam: number,
): Promise<{ recommendations: { hero: string; visits: number; winProb: number }[]; valueEstimate: number }> {
  if (!mctsWorker || !mctsWorkerReady) {
    throw new Error('MCTS worker not available')
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('MCTS search timeout')), 5000)
    mctsWorker!.onmessage = (e) => {
      clearTimeout(timeout)
      if (e.data.type === 'result') {
        resolve({
          recommendations: e.data.recommendations,
          valueEstimate: e.data.valueEstimate,
        })
      } else if (e.data.type === 'error') {
        reject(new Error(e.data.message))
      }
    }
    mctsWorker!.postMessage({
      type: 'search',
      team0Picks: draftState.team0Picks,
      team1Picks: draftState.team1Picks,
      bans: draftState.bans,
      map: draftState.map,
      tier: draftState.tier,
      step: draftState.step,
      ourTeam,
      takenHeroes: Array.from(takenHeroes),
    })
  })
}

/**
 * Get AI draft recommendations for the current state.
 *
 * Uses MCTS via Web Worker when available, otherwise falls back to
 * batched value-head evaluation of what-if states.
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

  const isBanStep = draftState.stepType === 'ban'
  const teamMawpAdj = computeTeamMawpAdjustment(draftState, playerData)
  const ourTeamNum = currentTeam === 'A' ? 0 : 1

  // Try MCTS via Web Worker first
  if (mctsWorkerReady && mctsWorker) {
    try {
      const mctsResult = await mctsSearch(draftState, takenHeroes, ourTeamNum)
      const ranked: AIRecommendation[] = mctsResult.recommendations.map(r => {
        const { adjustment, player } = isBanStep
          ? { adjustment: 0, player: null }
          : computePlayerAdjustment(r.hero, playerData)
        return {
          hero: r.hero,
          prior: r.visits,
          winProb: Math.max(0, Math.min(1, mctsResult.valueEstimate + adjustment + teamMawpAdj)),
          mawpAdj: adjustment,
          suggestedPlayer: player,
        }
      }).sort(recommendationSorter)
      return {
        recommendations: ranked.slice(0, topK),
        valueEstimate: Math.max(0, Math.min(1, mctsResult.valueEstimate + teamMawpAdj)),
      }
    } catch (err) {
      console.warn('[AI] MCTS worker failed, falling back to direct inference:', err)
    }
  }

  // Use the policy head directly — it was trained via MCTS to know which
  // heroes are good picks. One forward pass, rank by policy probability.
  const state = encodeState(draftState)
  const mask = buildValidMask(takenHeroes)
  const stateTensor = new ort.Tensor('float32', state, [1, 290])
  const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])
  const result = await policySession.run({ state: stateTensor, valid_mask: maskTensor })

  const policyLogits = result.policy_logits.data as Float32Array
  const rawValueEstimate = (result.value.data as Float32Array)[0]
  const baseValueEstimate = Math.max(0, Math.min(1, rawValueEstimate + teamMawpAdj))

  // Softmax the masked policy logits to get pick probabilities
  const priors = softmaxMasked(policyLogits, mask)

  // Build recommendations ranked by policy probability + MAWP adjustment
  const ranked: AIRecommendation[] = []
  for (let i = 0; i < NUM_HEROES; i++) {
    if (mask[i] > 0 && priors[i] > 0.001) {
      const { adjustment, player } = isBanStep
        ? { adjustment: 0, player: null }
        : computePlayerAdjustment(HEROES[i], playerData)

      ranked.push({
        hero: HEROES[i],
        prior: priors[i],
        winProb: Math.max(0, Math.min(1, baseValueEstimate + adjustment)),
        mawpAdj: adjustment,
        suggestedPlayer: player,
      })
    }
  }
  ranked.sort(recommendationSorter)

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

  const state = encodeStateForGD(draftState)  // 289 dims — GD doesn't have ourTeam
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
// Two-lane maps (must match training/shared.py TWO_LANE_MAPS)
const TWO_LANE_MAPS = new Set(['Battlefield of Eternity', 'Braxis Holdout', 'Hanamura Temple'])

// Fine-grained roles for role_counts feature (must match training/shared.py HERO_ROLE_FINE)
const FINE_ROLES: Record<string, number> = {}
// Map official roles to fine-grained indices: 0=tank,1=bruiser,2=healer,3=ranged_aa,4=ranged_mage,5=melee_assassin,6=support_utility,7=varian,8=pusher
const FINE_ROLE_MAP: Record<string, number> = {
  // Tanks
  "Anub'arak":0,"Arthas":0,"Blaze":0,"Cho":0,"Diablo":0,"E.T.C.":0,"Garrosh":0,
  "Johanna":0,"Mal'Ganis":0,"Mei":0,"Muradin":0,"Stitches":0,"Tyrael":0,
  // Bruisers
  "Artanis":1,"Chen":1,"Deathwing":1,"Dehaka":1,"D.Va":1,"Gazlowe":1,"Hogger":1,
  "Imperius":1,"Leoric":1,"Malthael":1,"Ragnaros":1,"Rexxar":1,"Sonya":1,"Thrall":1,
  "Xul":1,"Yrel":1,
  // Healers
  "Alexstrasza":2,"Ana":2,"Anduin":2,"Auriel":2,"Brightwing":2,"Deckard":2,
  "Kharazim":2,"Li Li":2,"Lt. Morales":2,"Lúcio":2,"Malfurion":2,"Rehgar":2,
  "Stukov":2,"Tyrande":2,"Uther":2,"Whitemane":2,
  // Ranged AA
  "Cassia":3,"Falstad":3,"Fenix":3,"Greymane":3,"Hanzo":3,"Lunara":3,"Raynor":3,
  "Sgt. Hammer":3,"Sylvanas":3,"Tracer":3,"Tychus":3,"Valla":3,"Zul'jin":3,
  // Ranged Mage
  "Chromie":4,"Gall":4,"Genji":4,"Gul'dan":4,"Jaina":4,"Junkrat":4,"Kael'thas":4,
  "Kel'Thuzad":4,"Li-Ming":4,"Mephisto":4,"Nova":4,"Orphea":4,"Probius":4,"Tassadar":4,
  // Melee Assassin
  "Alarak":5,"Illidan":5,"Kerrigan":5,"Maiev":5,"Qhira":5,"Samuro":5,
  "The Butcher":5,"Valeera":5,"Zeratul":5,
  // Support Utility
  "Abathur":6,"Medivh":6,"Zarya":6,
  // Varian
  "Varian":7,
  // Pusher
  "Azmodan":8,"Nazeebo":8,"Zagara":8,"Murky":8,"The Lost Vikings":8,
}
const NUM_FINE_ROLES = 9

/**
 * Compute enriched features for the WP model.
 * Requires DraftData for hero stats and pairwise data.
 * Returns 82 enriched features matching the training groups:
 *   role_counts(18) + team_avg_wr(2) + map_delta(2) +
 *   pairwise_counters(2) + pairwise_synergies(2) + counter_detail(50) +
 *   meta_strength(4) + draft_diversity(2)
 */
function computeEnrichedFeatures(
  t0Heroes: string[],
  t1Heroes: string[],
  map: string,
  draftData: DraftData,
): Float32Array {
  const features = new Float32Array(82)
  let off = 0

  // Helper: get hero WR, prefer map-specific
  const getWR = (hero: string): number => {
    const mapData = draftData.heroMapWinRates[map]?.[hero]
    if (mapData && mapData.games >= 50) return mapData.winRate
    return draftData.heroStats[hero]?.winRate ?? 50
  }
  const getOverallWR = (hero: string): number => draftData.heroStats[hero]?.winRate ?? 50

  // 1. role_counts (18 = 9 per team)
  for (const h of t0Heroes) {
    const r = FINE_ROLE_MAP[h]
    if (r !== undefined) features[off + r] += 1
  }
  off += NUM_FINE_ROLES
  for (const h of t1Heroes) {
    const r = FINE_ROLE_MAP[h]
    if (r !== undefined) features[off + r] += 1
  }
  off += NUM_FINE_ROLES

  // 2. team_avg_wr (2)
  const t0wrs = t0Heroes.map(getWR)
  const t1wrs = t1Heroes.map(getWR)
  features[off++] = t0wrs.length > 0 ? t0wrs.reduce((a, b) => a + b, 0) / t0wrs.length : 50
  features[off++] = t1wrs.length > 0 ? t1wrs.reduce((a, b) => a + b, 0) / t1wrs.length : 50

  // 3. map_delta (2)
  let t0mapDelta = 0, t1mapDelta = 0
  for (const h of t0Heroes) {
    const mapWR = draftData.heroMapWinRates[map]?.[h]
    if (mapWR && mapWR.games >= 50) t0mapDelta += mapWR.winRate - getOverallWR(h)
  }
  for (const h of t1Heroes) {
    const mapWR = draftData.heroMapWinRates[map]?.[h]
    if (mapWR && mapWR.games >= 50) t1mapDelta += mapWR.winRate - getOverallWR(h)
  }
  features[off++] = t0mapDelta
  features[off++] = t1mapDelta

  // 4. pairwise_counters (2) — avg normalized counter delta per team
  const counterDelta = (ourH: string[], theirH: string[]): number => {
    let sum = 0, count = 0
    for (const a of ourH) {
      for (const b of theirH) {
        const d = draftData.counters[a]?.[b]
        if (!d || d.games < 30) continue
        const expected = getWR(a) + (100 - getWR(b)) - 50
        sum += d.winRate - expected
        count++
      }
    }
    return count > 0 ? sum / count : 0
  }
  features[off++] = counterDelta(t0Heroes, t1Heroes)
  features[off++] = counterDelta(t1Heroes, t0Heroes)

  // 5. pairwise_synergies (2) — avg normalized synergy delta per team
  const synergyDelta = (heroes: string[]): number => {
    let sum = 0, count = 0
    for (let i = 0; i < heroes.length; i++) {
      for (let j = i + 1; j < heroes.length; j++) {
        const d = draftData.synergies[heroes[i]]?.[heroes[j]]
        if (!d || d.games < 30) continue
        const expected = 50 + (getWR(heroes[i]) - 50) + (getWR(heroes[j]) - 50)
        sum += d.winRate - expected
        count++
      }
    }
    return count > 0 ? sum / count : 0
  }
  features[off++] = synergyDelta(t0Heroes)
  features[off++] = synergyDelta(t1Heroes)

  // 6. counter_detail (50 = 5×5×2) — all cross-team pairs
  for (const a of t0Heroes) {
    for (const b of t1Heroes) {
      const d = draftData.counters[a]?.[b]
      if (d && d.games >= 30) {
        features[off] = d.winRate - (getWR(a) + (100 - getWR(b)) - 50)
      }
      off++
    }
  }
  // Pad to 25 if fewer than 5 heroes per team
  while (off < 18 + 2 + 2 + 2 + 2 + 25) off++
  for (const b of t1Heroes) {
    for (const a of t0Heroes) {
      const d = draftData.counters[b]?.[a]
      if (d && d.games >= 30) {
        features[off] = d.winRate - (getWR(b) + (100 - getWR(a)) - 50)
      }
      off++
    }
  }
  // Pad to 50 total
  while (off < 18 + 2 + 2 + 2 + 2 + 50) off++

  // 7. meta_strength (4) — avg pick_rate and ban_rate per team
  const t0pr = t0Heroes.reduce((s, h) => s + (draftData.heroStats[h]?.pickRate ?? 0), 0) / Math.max(t0Heroes.length, 1)
  const t0br = t0Heroes.reduce((s, h) => s + (draftData.heroStats[h]?.banRate ?? 0), 0) / Math.max(t0Heroes.length, 1)
  const t1pr = t1Heroes.reduce((s, h) => s + (draftData.heroStats[h]?.pickRate ?? 0), 0) / Math.max(t1Heroes.length, 1)
  const t1br = t1Heroes.reduce((s, h) => s + (draftData.heroStats[h]?.banRate ?? 0), 0) / Math.max(t1Heroes.length, 1)
  features[off++] = t0pr
  features[off++] = t0br
  features[off++] = t1pr
  features[off++] = t1br

  // 8. draft_diversity (2) — std dev of hero WRs per team
  const stdDev = (vals: number[]) => {
    if (vals.length < 2) return 0
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length
    return Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length)
  }
  features[off++] = stdDev(t0wrs)
  features[off++] = stdDev(t1wrs)

  return features
}

/**
 * Get win probability using the enriched WP model.
 * Returns P(team0 wins). Requires DraftData for feature computation.
 */
export async function getWinProbability(
  team0Heroes: string[],
  team1Heroes: string[],
  map: string,
  tier: string,
  draftData?: DraftData,
): Promise<number> {
  if (!wpSession || !ort) {
    throw new Error('AI models not loaded. Call loadAIModels() first.')
  }

  const t0 = heroesToMultiHot(team0Heroes)
  const t1 = heroesToMultiHot(team1Heroes)
  const m = mapToOneHot(map)
  const t = tierToOneHot(tier)

  // Base features: 197
  const base = new Float32Array(197)
  let offset = 0
  base.set(t0, offset); offset += NUM_HEROES
  base.set(t1, offset); offset += NUM_HEROES
  base.set(m, offset); offset += NUM_MAPS
  base.set(t, offset); offset += NUM_TIERS

  // Enriched features: 76 (if DraftData available)
  let input: Float32Array
  if (draftData) {
    const enriched = computeEnrichedFeatures(team0Heroes, team1Heroes, map, draftData)
    input = new Float32Array(279)
    input.set(base, 0)
    input.set(enriched, 197)
  } else {
    // Fallback: pad with zeros (enriched model still works, just less accurate)
    input = new Float32Array(279)
    input.set(base, 0)
  }

  const tensor = new ort.Tensor('float32', input, [1, 279])
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

function recommendationSorter(a: AIRecommendation, b: AIRecommendation): number {
  if (b.prior !== a.prior) return b.prior - a.prior
  if (b.winProb !== a.winProb) return b.winProb - a.winProb
  return a.hero.localeCompare(b.hero)
}

export { HEROES as AI_HEROES, HERO_TO_IDX as AI_HERO_TO_IDX }
