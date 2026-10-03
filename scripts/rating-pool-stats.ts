/**
 * Frozen-pool statistics for the pre-registered expert rating study.
 * Reads data/rating-items.json and prints everything the prereg cites:
 *   - composition per block, and per block x tier / stratum / map
 *   - near-tie exclusion counts on the machine pairs at |consensus - 0.5|
 *     thresholds {0.01, 0.02, 0.05}, plus planned judgment counts (x3)
 *   - per-evaluator near-tie counts at 0.02 (sensitivity)
 *   - favored-side flips and near-tie changes between the uncorrected and
 *     structure-corrected labels (v6)
 *   - item-level worst-case power z at true agreement 0.60
 *   - Bradley-Terry comparison-graph connectivity over the strategies
 *   - OOD covariate (ood_var_max, ood_var_matchup) distribution summaries
 *
 * Consensus: provenance.wpTeam0Sym.consensus when present (v6: the
 * structure-corrected consensus written by
 * training/paper1_revision/oct2026_pool_judges.py), else the mean of the
 * evaluators listed in pool.judges (v5 and earlier: the four evaluators).
 *
 * Usage: npx tsx scripts/rating-pool-stats.ts [path]
 */
import fs from 'node:fs'
import path from 'node:path'

interface Item {
  id: number
  block: string
  tier: string
  map: string
  provenance: {
    source: string
    stratum: string
    team0Strategy?: string
    team1Strategy?: string
    wpTeam0Sym?: Record<string, number>
    wpTeam0Sym_uncorrected?: Record<string, number>
    ood_var_max?: number
    ood_var_matchup?: number
  }
}

const file = process.argv[2] ?? path.resolve(__dirname, '../data/rating-items.json')
const pool = JSON.parse(fs.readFileSync(file, 'utf8'))
const items: Item[] = pool.items
const judged = pool.judges?.wpTeam0Sym
// Evaluators are the individual judges; the consensus is reported separately.
const EVALUATORS: string[] = judged
  ? judged.specs.map((s: string) => s.split(':').pop()!.replace(/_sc$/, '')).filter((e: string) => e !== 'consensus')
  : ['naive', 'herostrength', 'enriched', 'augmented']
console.log(
  `pool seed ${pool.seed}, ${items.length} items; evaluators ${EVALUATORS.join(', ')}; ` +
    `consensus: ${judged ? judged.consensus.join('+') : 'mean of the evaluators'}`
)

const count = (xs: string[]) => {
  const m = new Map<string, number>()
  for (const x of xs) m.set(x, (m.get(x) ?? 0) + 1)
  return Object.fromEntries([...m.entries()].sort())
}
for (const block of ['screener', 'calibration', 'pairs', 'anchors', 'catch']) {
  const b = items.filter((i) => i.block === block)
  console.log(`[${block}] n=${b.length} tiers`, count(b.map((i) => i.tier)))
}
const pairs = items.filter((i) => i.provenance.source === 'tournament')
console.log('pairs by stratum', count(pairs.map((i) => i.provenance.stratum)))
console.log('all items by map', count(items.map((i) => i.map)))

function consensusOf(wp: Record<string, number> | undefined): number {
  if (!wp) return NaN
  if (typeof wp.consensus === 'number') return wp.consensus
  return EVALUATORS.reduce((s, e) => s + wp[e], 0) / EVALUATORS.length
}
const cons = (it: Item) => consensusOf(it.provenance.wpTeam0Sym)
const consU = (it: Item) => consensusOf(it.provenance.wpTeam0Sym_uncorrected)

console.log(`\nmachine pairs: ${pairs.length}`)
for (const thr of [0.01, 0.02, 0.05]) {
  const kept = pairs.filter((i) => Math.abs(cons(i) - 0.5) > thr)
  const z = 0.1 / Math.sqrt(0.25 / kept.length)
  const line = `  near-tie threshold ${thr}: exclude ${pairs.length - kept.length} -> ${kept.length} ` +
    `effective items, ${kept.length * 3} planned judgments; item-level worst-case z at 0.60 = ${z.toFixed(2)}`
  const keptU = pairs.filter((i) => Math.abs(consU(i) - 0.5) > thr)
  console.log(line + (Number.isNaN(consU(pairs[0])) ? '' : ` (uncorrected: ${keptU.length})`))
}
for (const ev of EVALUATORS) {
  const kept = pairs.filter((i) => Math.abs(i.provenance.wpTeam0Sym![ev] - 0.5) > 0.02)
  console.log(`  evaluator ${ev} alone (0.02): ${kept.length} effective items`)
}
if (!Number.isNaN(consU(pairs[0]))) {
  const flips = pairs.filter((i) => (cons(i) > 0.5) !== (consU(i) > 0.5))
  const tieChange = pairs.filter((i) => (Math.abs(cons(i) - 0.5) <= 0.02) !== (Math.abs(consU(i) - 0.5) <= 0.02))
  const nonTieFlips = flips.filter((i) => Math.abs(cons(i) - 0.5) > 0.02 && Math.abs(consU(i) - 0.5) > 0.02)
  console.log(`  correction: favored-side flips ${flips.length} (of which outside the 0.02 band in both: ` +
    `${nonTieFlips.length}); near-tie status changes at 0.02: ${tieChange.length}`)
}
const anchored = pairs.filter((i) => i.provenance.stratum === 'vs_anchored')
const anchoredKept = anchored.filter((i) => Math.abs(cons(i) - 0.5) > 0.02)
console.log(`  vs_anchored stratum: ${anchored.length} items, ${anchoredKept.length} after 0.02 ` +
  `exclusion (${anchoredKept.length * 3} judgments)`)

const parent = new Map<string, string>()
const find = (x: string): string => {
  if (!parent.has(x)) parent.set(x, x)
  const p = parent.get(x)!
  if (p === x) return x
  const r = find(p)
  parent.set(x, r)
  return r
}
const edges = new Set<string>()
for (const i of pairs) {
  const a = i.provenance.team0Strategy!
  const b = i.provenance.team1Strategy!
  edges.add([a, b].sort().join('|'))
  parent.set(find(a), find(b))
}
const strategies = new Set(pairs.flatMap((i) => [i.provenance.team0Strategy!, i.provenance.team1Strategy!]))
const comps = new Set([...strategies].map(find))
const possible = (strategies.size * (strategies.size - 1)) / 2
console.log(`\nBradley-Terry: ${strategies.size} strategies, ${edges.size} of ${possible} pairings present, ` +
  `${comps.size} connected component(s)`)
console.log('strategies', [...strategies].sort().join(', '))

function summary(name: string, xs: number[]) {
  const s = xs.filter((x) => typeof x === 'number').sort((a, b) => a - b)
  if (!s.length) {
    console.log(`${name}: ABSENT on all pairs`)
    return
  }
  if (s.length < xs.length) console.log(`${name}: missing on ${xs.length - s.length} pairs`)
  // linear interpolation between order statistics (numpy's default percentile)
  const q = (p: number) => {
    const h = (s.length - 1) * p
    const lo = Math.floor(h)
    return s[lo] + (h - lo) * ((s[Math.min(lo + 1, s.length - 1)] ?? s[lo]) - s[lo])
  }
  const mean = s.reduce((a, b) => a + b, 0) / s.length
  console.log(`${name}: n=${s.length} mean ${mean.toFixed(5)} median ${q(0.5).toFixed(5)} ` +
    `p90 ${q(0.9).toFixed(5)} p95 ${q(0.95).toFixed(5)} max ${s[s.length - 1].toFixed(5)}`)
}
summary('ood_var_max', pairs.map((i) => i.provenance.ood_var_max!))
summary('ood_var_matchup', pairs.map((i) => i.provenance.ood_var_matchup!))
