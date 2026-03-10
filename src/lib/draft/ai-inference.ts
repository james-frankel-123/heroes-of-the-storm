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
  /** Estimated win probability for team 0 */
  winProb: number
}

/**
 * Get AI draft recommendations for the current state.
 * Returns heroes ranked by policy prior probability.
 */
export async function getAIRecommendations(
  draftState: AIDraftState,
  takenHeroes: Set<string>,
  topK = 15,
): Promise<{ recommendations: AIRecommendation[]; valueEstimate: number }> {
  if (!policySession || !ort) {
    throw new Error('AI models not loaded. Call loadAIModels() first.')
  }

  const state = encodeState(draftState)
  const mask = buildValidMask(takenHeroes)

  const stateTensor = new ort.Tensor('float32', state, [1, 289])
  const maskTensor = new ort.Tensor('float32', mask, [1, NUM_HEROES])

  const result = await policySession.run({
    state: stateTensor,
    valid_mask: maskTensor,
  })

  const policyLogits = result.policy_logits.data as Float32Array
  const valueEstimate = (result.value.data as Float32Array)[0]

  // Apply softmax with mask
  const priors = softmaxMasked(policyLogits, mask)

  // Sort by prior probability
  const ranked: AIRecommendation[] = []
  for (let i = 0; i < NUM_HEROES; i++) {
    if (mask[i] > 0 && priors[i] > 0.001) {
      ranked.push({
        hero: HEROES[i],
        prior: priors[i],
        winProb: valueEstimate,
      })
    }
  }
  ranked.sort((a, b) => b.prior - a.prior)

  return {
    recommendations: ranked.slice(0, topK),
    valueEstimate,
  }
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
