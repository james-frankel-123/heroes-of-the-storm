/**
 * OpponentPredictor implementation using onnxruntime-node.
 * Loads the Generic Draft ONNX model and runs inference for opponent simulation.
 */

import * as ort from 'onnxruntime-node'
import * as path from 'path'
import { HERO_ROLES } from '@/lib/data/hero-roles'
import { DRAFT_SEQUENCE } from '@/lib/draft/types'
import type { OpponentPredictor, SearchState } from '@/lib/draft/expectimax/types'

const HEROES = Object.keys(HERO_ROLES).sort()
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

function encodeStateForGD(state: SearchState): Float32Array {
  // 289 dims: t0(90) + t1(90) + bans(90) + map(14) + tier(3) + step_norm(1) + is_pick(1)
  const arr = new Float32Array(289)

  // Team 0 = team A, Team 1 = team B
  const t0Picks = state.ourTeam === 'A' ? state.ourPicks : state.enemyPicks
  const t1Picks = state.ourTeam === 'A' ? state.enemyPicks : state.ourPicks

  for (const h of t0Picks) {
    const idx = HERO_TO_IDX[h]
    if (idx !== undefined) arr[idx] = 1
  }
  for (const h of t1Picks) {
    const idx = HERO_TO_IDX[h]
    if (idx !== undefined) arr[90 + idx] = 1
  }
  for (const h of state.bans) {
    const idx = HERO_TO_IDX[h]
    if (idx !== undefined) arr[180 + idx] = 1
  }

  const mapIdx = MAP_TO_IDX[state.map]
  if (mapIdx !== undefined) arr[270 + mapIdx] = 1

  const tierIdx = TIER_TO_IDX[state.tier]
  if (tierIdx !== undefined) arr[284 + tierIdx] = 1

  const step = DRAFT_SEQUENCE[state.step]
  arr[287] = state.step / 15
  arr[288] = step?.type === 'pick' ? 1 : 0

  return arr
}

function buildValidMask(taken: Set<string>): Float32Array {
  const mask = new Float32Array(HEROES.length)
  for (let i = 0; i < HEROES.length; i++) {
    mask[i] = taken.has(HEROES[i]) ? 0 : 1
  }
  return mask
}

function softmaxMasked(logits: Float32Array, mask: Float32Array): Float32Array {
  const probs = new Float32Array(logits.length)
  let maxVal = -Infinity
  for (let i = 0; i < logits.length; i++) {
    if (mask[i] > 0.5 && logits[i] > maxVal) maxVal = logits[i]
  }
  let sum = 0
  for (let i = 0; i < logits.length; i++) {
    if (mask[i] > 0.5) {
      probs[i] = Math.exp(logits[i] - maxVal)
      sum += probs[i]
    }
  }
  if (sum > 0) for (let i = 0; i < logits.length; i++) probs[i] /= sum
  return probs
}

export async function createNodeGDPredictor(
  modelPath?: string,
): Promise<OpponentPredictor> {
  const resolvedPath = modelPath ?? path.join(
    new URL('.', import.meta.url).pathname, '..', 'public', 'models', 'generic_draft_0.onnx'
  )

  console.log(`  Loading GD model from ${resolvedPath}`)
  const session = await ort.InferenceSession.create(resolvedPath)
  console.log(`  GD model loaded`)

  return async (state: SearchState, topN: number) => {
    const stateArr = encodeStateForGD(state)
    const mask = buildValidMask(state.taken)

    const stateTensor = new ort.Tensor('float32', stateArr, [1, 289])
    const maskTensor = new ort.Tensor('float32', mask, [1, HEROES.length])

    const output = await session.run({
      state: stateTensor,
      valid_mask: maskTensor,
    })

    // GD model outputs 'hero_logits' or first output
    const outputKey = Object.keys(output)[0]
    const logits = output[outputKey].data as Float32Array
    const probs = softmaxMasked(logits, mask)

    // Sort by probability, return top-N
    const scored: { hero: string; probability: number }[] = []
    for (let i = 0; i < HEROES.length; i++) {
      if (mask[i] > 0.5 && probs[i] > 0.001) {
        scored.push({ hero: HEROES[i], probability: probs[i] })
      }
    }
    scored.sort((a, b) => b.probability - a.probability)
    return scored.slice(0, topN)
  }
}
