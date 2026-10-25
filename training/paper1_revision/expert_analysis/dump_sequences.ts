/**
 * For each (rater code, slot) in config.json, the app's own assignment
 * (src/lib/rating/assignment.ts fullSequence) as an ordered id list, so the
 * analysis can check every rating against the rater's assigned slot and know
 * how far a partial rater got. No DB access.
 * Usage: npx tsx training/paper1_revision/expert_analysis/dump_sequences.ts <pool.json> <config.json> <out.json>
 */
import fs from 'node:fs'
import { fullSequence, type PoolIds } from '../../../src/lib/rating/assignment'

const [poolFile, cfgFile, outFile] = process.argv.slice(2)
const items: any[] = JSON.parse(fs.readFileSync(poolFile, 'utf8')).items
const cfg = JSON.parse(fs.readFileSync(cfgFile, 'utf8'))
const by = (b: string) => items.filter((i) => i.block === b).map((i) => i.id)
const anchorsByTier = new Map<string, number[]>()
for (const it of items.filter((i) => i.block === 'anchors')) {
  if (!anchorsByTier.has(it.tier)) anchorsByTier.set(it.tier, [])
  anchorsByTier.get(it.tier)!.push(it.id)
}
const P: PoolIds = { screener: by('screener'), calibration: by('calibration'), pairs: by('pairs'), anchorsByTier, catch: by('catch') }
const out: Record<string, { slot: number; sequence: number[] }> = {}
for (const [code, r] of Object.entries<any>(cfg.raters)) out[code] = { slot: r.slot, sequence: fullSequence(P, code, r.slot) }
fs.writeFileSync(outFile, JSON.stringify(out))
console.log(`sequences for ${Object.keys(out).length} codes -> ${outFile}`)
