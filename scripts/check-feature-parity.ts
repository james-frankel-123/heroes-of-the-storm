/** Asserts browser feature computation matches Python trainer goldens.
 * Run: npx tsx scripts/check-feature-parity.ts */
import { _testComputeEnrichedFeatures } from '../src/lib/draft/ai-inference'
import draftStats from '../src/lib/data/draft-stats-decayed.json'
import compositionsJson from '../src/lib/data/compositions.json'
import goldens from './feature-parity-goldens.json'

let worst = 0
let worstInfo = ''
let failures = 0
for (const c of goldens as any[]) {
  const t = (draftStats as any).tiers[c.tier]
  const draftData: any = {
    heroStats: t.heroStats, heroMapWinRates: t.heroMapWinRates,
    synergies: t.synergies, counters: t.counters,
    playerStats: {}, playerMapStats: {},
    compositions: (compositionsJson as any)[c.tier] ?? [], baselineCompWR: 50,
  }
  const ts = _testComputeEnrichedFeatures(c.t0, c.t1, c.map, draftData)
  for (let i = 0; i < 86; i++) {
    const d = Math.abs(ts[i] - c.enriched[i])
    if (d > worst) { worst = d; worstInfo = `idx ${i} (${c.t0.length}v${c.t1.length}, ${c.map}/${c.tier}): ts=${ts[i].toFixed(3)} py=${c.enriched[i]}` }
    if (d > 0.05) failures++
  }
}
console.log(`worst |diff| = ${worst.toFixed(4)} at ${worstInfo}`)
console.log(failures === 0 ? 'PARITY OK' : `PARITY FAIL: ${failures} feature values differ by > 0.05`)
process.exit(failures === 0 ? 0 : 1)
