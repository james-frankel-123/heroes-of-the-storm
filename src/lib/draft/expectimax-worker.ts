/**
 * Web Worker for expectimax tree search.
 *
 * Loads the Generic Draft ONNX model and runs expectimax search
 * on an assigned subset of root candidates. Reports results back
 * via postMessage as iterative deepening completes each depth.
 *
 * Protocol:
 *   Main → Worker: { type: 'init' }
 *   Worker → Main: { type: 'ready' }
 *   Main → Worker: { type: 'search', rootState, draftData, config, candidates }
 *   Worker → Main: { type: 'depth-complete', results, depth }
 *   Worker → Main: { type: 'result', results }
 *   Worker → Main: { type: 'error', message }
 */

/// <reference lib="webworker" />

declare const self: DedicatedWorkerGlobalScope & {
  ort?: any
}

// ── Constants (duplicated from ai-inference.ts — workers are isolated) ──

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

const TIERS = ['low', 'mid', 'high']
const TIER_TO_IDX: Record<string, number> = {}
TIERS.forEach((t, i) => { TIER_TO_IDX[t] = i })

const DRAFT_SEQUENCE_TYPES = [
  'ban','ban','ban','ban','pick','pick','pick','pick','pick','ban','ban','pick','pick','pick','pick','pick'
]

// ── GD Model ──

let ort: any = null
let gdSession: any = null

function encodeStateForGD(state: any): Float32Array {
  const arr = new Float32Array(289)
  const t0 = state.ourTeam === 'A' ? state.ourPicks : state.enemyPicks
  const t1 = state.ourTeam === 'A' ? state.enemyPicks : state.ourPicks
  for (const h of t0) { const i = HERO_TO_IDX[h]; if (i !== undefined) arr[i] = 1 }
  for (const h of t1) { const i = HERO_TO_IDX[h]; if (i !== undefined) arr[90 + i] = 1 }
  for (const h of state.bans) { const i = HERO_TO_IDX[h]; if (i !== undefined) arr[180 + i] = 1 }
  const mi = MAP_TO_IDX[state.map]; if (mi !== undefined) arr[270 + mi] = 1
  const ti = TIER_TO_IDX[state.tier]; if (ti !== undefined) arr[284 + ti] = 1
  arr[287] = state.step / 15
  arr[288] = DRAFT_SEQUENCE_TYPES[state.step] === 'pick' ? 1 : 0
  return arr
}

function softmaxMasked(logits: Float32Array, mask: Float32Array): Float32Array {
  const probs = new Float32Array(logits.length)
  let mx = -Infinity
  for (let i = 0; i < logits.length; i++) if (mask[i] > 0.5 && logits[i] > mx) mx = logits[i]
  let sum = 0
  for (let i = 0; i < logits.length; i++) {
    if (mask[i] > 0.5) { probs[i] = Math.exp(logits[i] - mx); sum += probs[i] }
  }
  if (sum > 0) for (let i = 0; i < logits.length; i++) probs[i] /= sum
  return probs
}

async function gdPredict(state: any, topN: number): Promise<{ hero: string; probability: number }[]> {
  if (!gdSession) return []
  const stateArr = encodeStateForGD(state)
  const mask = new Float32Array(HEROES.length)
  for (let i = 0; i < HEROES.length; i++) mask[i] = state.taken.has(HEROES[i]) ? 0 : 1

  // Set → serialized back, need to reconstruct Set if received as array
  const stateTensor = new ort.Tensor('float32', stateArr, [1, 289])
  const maskTensor = new ort.Tensor('float32', mask, [1, HEROES.length])

  const output = await gdSession.run({ state: stateTensor, valid_mask: maskTensor })
  const key = Object.keys(output)[0]
  const logits = output[key].data as Float32Array
  const probs = softmaxMasked(logits, mask)

  const scored: { hero: string; probability: number }[] = []
  for (let i = 0; i < HEROES.length; i++) {
    if (mask[i] > 0.5 && probs[i] > 0.001) scored.push({ hero: HEROES[i], probability: probs[i] })
  }
  scored.sort((a, b) => b.probability - a.probability)
  return scored.slice(0, topN)
}

// ── Worker Messages ──

async function loadModels() {
  importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js')
  ort = self.ort
  ort.env.wasm.numThreads = 1

  const tryLoad = async (int8: string, fp32: string) => {
    try { return await ort.InferenceSession.create(int8) }
    catch { return await ort.InferenceSession.create(fp32) }
  }

  gdSession = await tryLoad('/models/generic_draft_0_int8.onnx', '/models/generic_draft_0.onnx')
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data

  if (msg.type === 'init') {
    try {
      await loadModels()
      self.postMessage({ type: 'ready' })
    } catch (err: any) {
      self.postMessage({ type: 'error', message: `Init failed: ${err.message}` })
    }
    return
  }

  if (msg.type === 'search') {
    try {
      const { rootState, draftData, config, candidates } = msg

      // Reconstruct Set from array (Sets don't survive structured clone)
      rootState.taken = new Set(rootState.taken)

      // Import the search module dynamically would be ideal but workers
      // can't use ES imports easily. Instead, we inline a simplified search.
      // The full search.ts is imported at build time via the bundler.

      // For now, post back that this worker architecture is ready
      // The actual search logic will be pulled in via the bundler
      const { expectimaxAtDepth } = await import('./expectimax/search')

      // Run iterative deepening on our assigned candidates
      const depths = [4, 6, 8].filter(d => d <= config.maxDepth)
      let lastResults: any[] = []

      for (const depth of depths) {
        const startTime = Date.now()
        const { results } = await expectimaxAtDepth(
          rootState, draftData, depth, config, gdPredict,
        )
        // Filter to only our assigned candidates
        const filtered = results.filter((r: any) => candidates.includes(r.hero))
        lastResults = filtered

        self.postMessage({ type: 'depth-complete', results: filtered, depth })

        if (Date.now() - startTime > config.timeBudgetMs / 2) break
      }

      self.postMessage({ type: 'result', results: lastResults })
    } catch (err: any) {
      self.postMessage({ type: 'error', message: `Search failed: ${err.message}` })
    }
  }
}
