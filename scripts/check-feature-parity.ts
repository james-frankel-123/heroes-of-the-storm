/** Asserts browser feature computation matches Python trainer goldens:
 * the hero/map/tier encoding lists, the base one-hot layout and the 86
 * enriched features. Run: npx tsx scripts/check-feature-parity.ts */
import { _testComputeEnrichedFeatures } from '../src/lib/draft/ai-inference'
import {
  HEROES, MAPS, SKILL_TIERS, HERO_TO_IDX, MAP_TO_IDX, TIER_TO_IDX,
  NUM_HEROES, NUM_MAPS, WP_BASE_DIM, ENRICHED_DIM,
} from '../src/lib/draft/encoding'
import draftStats from '../src/lib/data/draft-stats-decayed.json'
import modelCompositionsJson from '../src/lib/data/model-compositions.json'
import goldens from './feature-parity-goldens.json'

const g = goldens as any
let failures = 0
const same = (a: readonly string[], b: string[]) => a.length === b.length && a.every((x, i) => x === b[i])
for (const [name, ts, py] of [['heroes', HEROES, g.heroes], ['maps', MAPS, g.maps], ['tiers', SKILL_TIERS, g.tiers]] as const) {
  if (!same(ts, py)) { console.log(`ENCODING FAIL: ${name} differ from training/shared.py`); failures++ }
}
if (g.base_dim !== WP_BASE_DIM) { console.log(`ENCODING FAIL: base dim ${WP_BASE_DIM} vs ${g.base_dim}`); failures++ }

let worst = 0
let worstInfo = ''
for (const c of g.cases as any[]) {
  // base vector exactly as runWP builds it: t0 | t1 | map | tier
  const ones: number[] = []
  for (const h of c.t0) ones.push(HERO_TO_IDX[h])
  for (const h of c.t1) ones.push(NUM_HEROES + HERO_TO_IDX[h])
  ones.push(2 * NUM_HEROES + MAP_TO_IDX[c.map])
  ones.push(2 * NUM_HEROES + NUM_MAPS + TIER_TO_IDX[c.tier])
  ones.sort((a, b) => a - b)
  if (ones.join() !== c.base_ones.join()) { failures++; console.log(`BASE FAIL ${c.t0}|${c.t1} ${c.map}`) }

  const t = (draftStats as any).tiers[c.tier]
  const draftData: any = {
    heroStats: t.heroStats, heroMapWinRates: t.heroMapWinRates,
    synergies: t.synergies, counters: t.counters,
    playerStats: {}, playerMapStats: {},
    compositions: (modelCompositionsJson as any)[c.tier] ?? [], baselineCompWR: 50,
  }
  const ts = _testComputeEnrichedFeatures(c.t0, c.t1, c.map, c.tier, draftData)
  for (let i = 0; i < ENRICHED_DIM; i++) {
    const d = Math.abs(ts[i] - c.enriched[i])
    if (d > worst) { worst = d; worstInfo = `idx ${i} (${c.t0.length}v${c.t1.length}, ${c.map}/${c.tier}): ts=${ts[i].toFixed(3)} py=${c.enriched[i]}` }
    if (d > 0.05) failures++
  }
}
console.log(`${g.cases.length} cases; worst enriched |diff| = ${worst.toFixed(4)} at ${worstInfo}`)
console.log(failures === 0 ? 'PARITY OK' : `PARITY FAIL: ${failures} mismatches`)
process.exit(failures === 0 ? 0 : 1)
