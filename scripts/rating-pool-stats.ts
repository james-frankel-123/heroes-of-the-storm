/**
 * Frozen-pool statistics for the pre-registered expert rating study.
 * Reads data/rating-items.json and prints everything the prereg cites:
 *   - composition per block
 *   - near-tie exclusion counts on core machine pairs at |consensus - 0.5|
 *     thresholds {0.01, 0.02, 0.05} (consensus = mean of the four frozen
 *     evaluator symmetrized WPs), plus planned judgment counts (x3)
 *   - item-level worst-case power z at true agreement 0.60
 *   - Bradley-Terry comparison-graph connectivity over the 11 strategies
 *     (core machine pairs only, and core + extended)
 *   - OOD covariate (ood_var_max) distribution summary
 *
 * Usage: npx tsx scripts/rating-pool-stats.ts
 */
import fs from 'node:fs'
import path from 'node:path'

interface Item {
  id: number
  block: string
  tier: string
  provenance: {
    source: string
    stratum: string
    team0Strategy?: string
    team1Strategy?: string
    wpTeam0Sym?: Record<string, number>
    ood_var_max?: number
  }
}

const EVALUATORS = ['naive', 'herostrength', 'enriched', 'augmented']

const { seed, items } = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../data/rating-items.json'), 'utf8')
) as { seed: number; items: Item[] }
console.log(`pool seed ${seed}, ${items.length} items`)

for (const block of ['screener', 'calibration', 'core', 'extended']) {
  const blockItems = items.filter((i) => i.block === block)
  const bySource = new Map<string, number>()
  for (const i of blockItems)
    bySource.set(i.provenance.source, (bySource.get(i.provenance.source) ?? 0) + 1)
  console.log(`[${block}] n=${blockItems.length}`, Object.fromEntries(bySource))
}

const consensus = (it: Item) => {
  const wp = it.provenance.wpTeam0Sym!
  return EVALUATORS.reduce((s, e) => s + wp[e], 0) / EVALUATORS.length
}

const corePairs = items.filter(
  (i) => i.block === 'core' && i.provenance.source === 'tournament'
)
console.log(`\ncore machine pairs: ${corePairs.length}`)
for (const thr of [0.01, 0.02, 0.05]) {
  const kept = corePairs.filter((i) => Math.abs(consensus(i) - 0.5) > thr)
  const excluded = corePairs.length - kept.length
  const z = 0.1 / Math.sqrt(0.25 / kept.length)
  console.log(
    `  near-tie threshold ${thr}: exclude ${excluded} -> ${kept.length} effective items, ` +
      `${kept.length * 3} planned judgments; item-level worst-case z at true 0.60 = ${z.toFixed(2)}`
  )
}
// Per-evaluator near-tie counts at the primary threshold (sensitivity).
for (const ev of EVALUATORS) {
  const kept = corePairs.filter((i) => Math.abs(i.provenance.wpTeam0Sym![ev] - 0.5) > 0.02)
  console.log(`  evaluator ${ev} alone (0.02): ${kept.length} effective items`)
}
// Anchored-vs-non-anchored stratum (named secondary), at primary threshold.
const anchoredStratum = corePairs.filter((i) => i.provenance.stratum === 'vs_anchored')
const anchoredKept = anchoredStratum.filter((i) => Math.abs(consensus(i) - 0.5) > 0.02)
console.log(
  `  vs_anchored core stratum: ${anchoredStratum.length} items, ` +
    `${anchoredKept.length} after 0.02 near-tie exclusion (${anchoredKept.length * 3} judgments)`
)

// Bradley-Terry connectivity: union-find over strategies.
function connectivity(pairs: Item[]): { strategies: number; components: number; edges: number } {
  const parent = new Map<string, string>()
  const find = (x: string): string => {
    if (!parent.has(x)) parent.set(x, x)
    let r = x
    while (parent.get(r) !== r) r = parent.get(r)!
    parent.set(x, r)
    return r
  }
  const edges = new Set<string>()
  for (const i of pairs) {
    const a = i.provenance.team0Strategy!
    const b = i.provenance.team1Strategy!
    edges.add([a, b].sort().join('__'))
    parent.set(find(a), find(b))
  }
  const strategies = new Set([...parent.keys()])
  const roots = new Set([...strategies].map(find))
  return { strategies: strategies.size, components: roots.size, edges: edges.size }
}
const coreConn = connectivity(corePairs)
const allPairs = items.filter((i) => i.provenance.source === 'tournament')
const allConn = connectivity(allPairs)
console.log(
  `\nBT connectivity (core only): ${coreConn.strategies} strategies, ` +
    `${coreConn.edges} distinct matchups, ${coreConn.components} connected component(s)`
)
console.log(
  `BT connectivity (core + extended): ${allConn.strategies} strategies, ` +
    `${allConn.edges} distinct matchups, ${allConn.components} connected component(s)`
)

// OOD covariate summary.
const ood = allPairs.map((i) => i.provenance.ood_var_max!).sort((a, b) => a - b)
const q = (p: number) => ood[Math.min(ood.length - 1, Math.floor((p / 100) * ood.length))]
console.log(
  `\nood_var_max over ${ood.length} machine pairs: ` +
    `mean=${(ood.reduce((s, x) => s + x, 0) / ood.length).toFixed(5)} ` +
    `median=${q(50).toFixed(5)} p90=${q(90).toFixed(5)} p95=${q(95).toFixed(5)} max=${ood[ood.length - 1].toFixed(5)}`
)
