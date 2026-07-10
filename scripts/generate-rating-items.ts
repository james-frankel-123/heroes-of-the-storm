/**
 * Generate the item set for the expert draft-rating study: a shared
 * CALIBRATION block of 20 items (rated by every rater, first), a CORE of 100
 * items (the pre-registered latin-square design), and an EXTENDED pool of 700
 * items (a volunteer arm for high-volume raters, served after core).
 *
 * Composition (seeded, reproducible given the same DB snapshot):
 *
 *   CALIBRATION — 20 items, ids 1..20 (every rater rates ALL of them before
 *   their latin-square core 30 → 50 items per rater total):
 *     - 20 real known-outcome ladder anchors, tier-stratified 7 low / 7 mid /
 *       6 high, distinct from every other item in the study.
 *
 *   CORE — 100 items, ids 21..120:
 *     - 55 tournament (machine-pair) drafts, same matchup strata as before,
 *       rebalanced to 55 in the original proportions (12:12:32:24 -> 8:8:22:17):
 *         A. constrained_mcts vs mcts            8 items
 *         B. mcts vs enriched                    8 items
 *         C. behavior-anchored vs non-anchored  22 items
 *            (anchored = gd, cql_naive_a1.0, cql_enr_a2.0, gourdeau_disc)
 *         D. all other strategy pairs           17 items
 *     - 45 real known-outcome ladder anchors (15 low / 15 mid / 15 high).
 *
 *   EXTENDED — 700 items, ids 121..820 (distinct replays/records from
 *   calibration and core):
 *     - 145 tournament (machine-pair) drafts in the same strata proportions
 *       (22:22:58:43).
 *     - 555 known-outcome ladder anchors, tier-stratified 185 / 185 / 185.
 *
 * Every item carries a `block` field ('calibration' | 'core' | 'extended').
 * Core keeps the original 10-slot latin square (30 items/rater, 3
 * ratings/item); calibration is served to everyone before core; extended is
 * served as an uncapped, coverage-prioritized pool after a rater finishes core.
 *
 * Output: data/rating-items.json — committed to the repo and seeded into the
 * rating_items table by scripts/seed-rating-items.ts. Provenance (strategy
 * labels, source) lives ONLY in this file / the DB provenance column; it is
 * never sent to the client.
 *
 * Usage: set -a && source .env && set +a && npx tsx scripts/generate-rating-items.ts
 */
import fs from 'node:fs'
import path from 'node:path'
import { neon } from '@neondatabase/serverless'
import { fnv1a, mulberry32, seededShuffle } from '../src/lib/rating/assignment'
import { HERO_ROLES } from '../src/lib/data/hero-roles'

// SEED history: 20260708 generated the original 500-item set (100 core + 400
// extended). 20260709 regenerates the full pool with the shared calibration
// block and the expanded extended arm (20 + 100 + 700 = 820 items).
const SEED = 20260709
const REPO = path.resolve(__dirname, '..')
const RESULT_DIRS = [
  path.join(REPO, 'training/rerun2026/results/roundrobin'),
  path.join(REPO, 'training/rerun2026/results/constrained/roundrobin'),
]
const OUT_PATH = path.join(REPO, 'data/rating-items.json')

const ANCHORED = new Set(['gd', 'cql_naive_a1.0', 'cql_enr_a2.0', 'gourdeau_disc'])
// Machine-pair matchup strata. `core` sums to 55 (rebalanced from the original
// 12:12:32:24=80 in the same proportions -> 8:8:22:17); `ext` sums to 145 in
// the same proportions (15:15:40:30 scaled x1.45 -> 22:22:58:43). Extended
// records are distinct from core.
const STRATA: { name: string; core: number; ext: number }[] = [
  { name: 'constrained_mcts_vs_mcts', core: 8, ext: 22 },
  { name: 'mcts_vs_enriched', core: 8, ext: 22 },
  { name: 'vs_anchored', core: 22, ext: 58 },
  { name: 'other_pairs', core: 17, ext: 43 },
]
// Known-outcome ladder anchors. Calibration = 7/7/6 (20, shared across all
// raters); core = 15/tier (45); extended = 185/tier (555); all distinct
// replays.
const LADDER_CALIB_COUNTS: Record<string, number> = { low: 7, mid: 7, high: 6 }
const LADDER_CORE_COUNTS: Record<string, number> = { low: 15, mid: 15, high: 15 }
const LADDER_EXT_COUNTS: Record<string, number> = { low: 185, mid: 185, high: 185 }

interface TournamentRecord {
  draft: number
  game_map: string
  tier: string
  team0: { picks: string[] }
  team1: { picks: string[] }
  wp: Record<string, { raw: number; sym: number }>
}

interface Candidate {
  map: string
  tier: string
  team0: string[]
  team1: string[]
  provenance: Record<string, unknown>
  pairKey: string
}

function stratumOf(s0: string, s1: string): string {
  const pair = new Set([s0, s1])
  if (pair.has('constrained_mcts') && pair.has('mcts')) return 'constrained_mcts_vs_mcts'
  if (pair.has('mcts') && pair.has('enriched')) return 'mcts_vs_enriched'
  const anchoredCount = [s0, s1].filter((s) => ANCHORED.has(s)).length
  if (anchoredCount === 1) return 'vs_anchored'
  return 'other_pairs'
}

function validTeams(team0: string[], team1: string[]): boolean {
  if (team0.length !== 5 || team1.length !== 5) return false
  const all = [...team0, ...team1]
  if (new Set(all).size !== 10) return false
  if (!all.every((h) => h === "Cho'gall" || HERO_ROLES[h] !== undefined)) return false
  // Exclude split Cho/Gall teams: impossible in a real lobby (the tournament
  // simulator allows them, ~18% of records), so they would (a) read as
  // nonsense to expert raters and (b) unblind tournament vs ladder items,
  // since real ladder drafts never contain them.
  for (const team of [team0, team1]) {
    if (team.includes('Cho') !== team.includes('Gall')) return false
  }
  return true
}

function loadTournamentCandidates(): Map<string, Candidate[]> {
  const byStratum = new Map<string, Candidate[]>()
  for (const dir of RESULT_DIRS) {
    for (const file of fs.readdirSync(dir).sort()) {
      if (!file.endsWith('.json')) continue
      const data = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'))
      const { team0_strategy: s0, team1_strategy: s1, records } = data
      if (!s0 || !s1 || !Array.isArray(records)) continue
      const stratum = stratumOf(s0, s1)
      const relDir = path.relative(REPO, dir)
      for (let i = 0; i < records.length; i++) {
        const r: TournamentRecord = records[i]
        if (!validTeams(r.team0.picks, r.team1.picks)) continue
        const wpSym: Record<string, number> = {}
        for (const [k, v] of Object.entries(r.wp ?? {})) wpSym[k] = v.sym
        const cand: Candidate = {
          map: r.game_map,
          tier: r.tier,
          team0: r.team0.picks,
          team1: r.team1.picks,
          pairKey: [s0, s1].sort().join('__'),
          provenance: {
            source: 'tournament',
            stratum,
            team0Strategy: s0,
            team1Strategy: s1,
            file: `${relDir}/${file}`,
            recordIndex: i,
            // Model-graded P(team0 wins), symmetrized, per evaluator — for
            // later expert-vs-model comparison.
            wpTeam0Sym: wpSym,
          },
        }
        if (!byStratum.has(stratum)) byStratum.set(stratum, [])
        byStratum.get(stratum)!.push(cand)
      }
    }
  }
  return byStratum
}

/**
 * Sample `count` candidates from a stratum, cycling over strategy pairs
 * (so every pair file contributes) and, within each pair, preferring the
 * tier/map least represented so far.
 */
function sampleStratum(cands: Candidate[], count: number, seed: number): Candidate[] {
  const rand = mulberry32(seed)
  const byPair = new Map<string, Candidate[]>()
  for (const c of cands) {
    if (!byPair.has(c.pairKey)) byPair.set(c.pairKey, [])
    byPair.get(c.pairKey)!.push(c)
  }
  const pairKeys = seededShuffle([...byPair.keys()].sort(), seed ^ 0x9e3779b9)
  for (const k of pairKeys) {
    byPair.set(k, seededShuffle(byPair.get(k)!, fnv1a(String(seed) + '|' + k)))
  }
  const picked: Candidate[] = []
  const tierCount = new Map<string, number>()
  const mapCount = new Map<string, number>()
  const usedKeys = new Set<string>()
  let pi = 0
  while (picked.length < count) {
    const pool = byPair.get(pairKeys[pi % pairKeys.length])!
    pi++
    // Best remaining candidate in this pair: least-used tier, then map, random tiebreak.
    let best: Candidate | null = null
    let bestScore = Infinity
    for (const c of pool) {
      const key = c.team0.join(',') + '|' + c.team1.join(',') + '|' + c.map
      if (usedKeys.has(key)) continue
      const score =
        (tierCount.get(c.tier) ?? 0) * 100 + (mapCount.get(c.map) ?? 0) * 3 + rand()
      if (score < bestScore) {
        bestScore = score
        best = c
      }
    }
    if (!best) continue
    const key = best.team0.join(',') + '|' + best.team1.join(',') + '|' + best.map
    usedKeys.add(key)
    tierCount.set(best.tier, (tierCount.get(best.tier) ?? 0) + 1)
    mapCount.set(best.map, (mapCount.get(best.map) ?? 0) + 1)
    picked.push(best)
  }
  return picked
}

/** Ladder anchors for one tier: `count` distinct valid drafts, map-spread. */
async function loadLadderTier(tier: string, count: number): Promise<Candidate[]> {
  const sql = neon(process.env.DATABASE_URL!)
  // Deterministic recent window; the committed JSON freezes the sample. Pull a
  // generous window so ~207/tier (7 calibration + 15 core + 185 extended)
  // valid drafts are available after filtering.
  const rows = (await sql`
    select replay_id, game_map, skill_tier, team0_heroes, team1_heroes, winner, game_date
    from replay_draft_data
    where skill_tier = ${tier}
    order by replay_id desc
    limit 6000
  `) as {
    replay_id: number
    game_map: string
    skill_tier: string
    team0_heroes: string[]
    team1_heroes: string[]
    winner: number
    game_date: string
  }[]
  const valid = rows.filter((r) => validTeams(r.team0_heroes, r.team1_heroes))
  if (valid.length < count) {
    throw new Error(`ladder tier ${tier}: only ${valid.length} valid drafts, need ${count}`)
  }
  const shuffled = seededShuffle(valid, fnv1a(String(SEED) + '|ladder|' + tier))
  const mapCount = new Map<string, number>()
  const chosen: typeof valid = []
  // Greedy map spread.
  while (chosen.length < count && shuffled.length > 0) {
    let bestIdx = 0
    let bestScore = Infinity
    for (let i = 0; i < shuffled.length; i++) {
      const score = mapCount.get(shuffled[i].game_map) ?? 0
      if (score < bestScore) {
        bestScore = score
        bestIdx = i
      }
    }
    const r = shuffled.splice(bestIdx, 1)[0]
    mapCount.set(r.game_map, (mapCount.get(r.game_map) ?? 0) + 1)
    chosen.push(r)
  }
  return chosen.map((r) => ({
    map: r.game_map,
    tier: r.skill_tier,
    team0: r.team0_heroes,
    team1: r.team1_heroes,
    pairKey: 'ladder',
    provenance: {
      source: 'ladder',
      stratum: 'ladder_anchor',
      replayId: r.replay_id,
      winner: r.winner, // real game outcome (0 = team0 won)
      gameDate: r.game_date,
    },
  }))
}

async function main() {
  if (!process.env.DATABASE_URL) {
    console.error('DATABASE_URL not set. Run: set -a && source .env && set +a')
    process.exit(1)
  }
  const byStratum = loadTournamentCandidates()
  const coreMachine: Candidate[] = []
  const extMachine: Candidate[] = []
  for (const { name, core, ext } of STRATA) {
    const cands = byStratum.get(name) ?? []
    console.log(`stratum ${name}: ${cands.length} candidates -> sampling ${core} core + ${ext} extended`)
    // Sample core+ext distinct records in one pass, then split so extended
    // never reuses a core record.
    const sampled = sampleStratum(cands, core + ext, SEED ^ fnv1a(name))
    coreMachine.push(...sampled.slice(0, core))
    extMachine.push(...sampled.slice(core))
  }

  const calibLadder: Candidate[] = []
  const coreLadder: Candidate[] = []
  const extLadder: Candidate[] = []
  for (const tier of Object.keys(LADDER_CORE_COUNTS)) {
    const calibN = LADDER_CALIB_COUNTS[tier]
    const coreN = LADDER_CORE_COUNTS[tier]
    const extN = LADDER_EXT_COUNTS[tier]
    // One pass per tier, sliced calibration → core → extended, so the three
    // blocks are guaranteed disjoint replays.
    const drafts = await loadLadderTier(tier, calibN + coreN + extN)
    calibLadder.push(...drafts.slice(0, calibN))
    coreLadder.push(...drafts.slice(calibN, calibN + coreN))
    extLadder.push(...drafts.slice(calibN + coreN))
    console.log(`ladder ${tier}: ${calibN} calibration + ${coreN} core + ${extN} extended`)
  }

  // CALIBRATION: shuffle so item id carries no tier ordering; ids 1..20.
  const calibration = seededShuffle(calibLadder, SEED ^ 0xca11b)
  // CORE: shuffle so item id carries no stratum information; ids follow calibration.
  const core = seededShuffle([...coreMachine, ...coreLadder], SEED)
  // EXTENDED: shuffle independently; ids continue after core.
  const extended = seededShuffle([...extMachine, ...extLadder], SEED ^ 0x5eed_e17e)

  const toItem = (c: Candidate, id: number, block: 'calibration' | 'core' | 'extended') => ({
    id,
    block,
    map: c.map,
    tier: c.tier,
    teams: { team0: c.team0, team1: c.team1 },
    provenance: c.provenance,
  })
  const items = [
    ...calibration.map((c, i) => toItem(c, i + 1, 'calibration')),
    ...core.map((c, i) => toItem(c, calibration.length + i + 1, 'core')),
    ...extended.map((c, i) =>
      toItem(c, calibration.length + core.length + i + 1, 'extended')
    ),
  ]

  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true })
  fs.writeFileSync(
    OUT_PATH,
    JSON.stringify({ seed: SEED, generatedAt: new Date().toISOString(), items }, null, 2)
  )
  console.log(
    `wrote ${items.length} items (${calibration.length} calibration + ${core.length} core + ${extended.length} extended) to ${OUT_PATH}`
  )

  // Composition report.
  for (const block of ['calibration', 'core', 'extended'] as const) {
    const strata = new Map<string, number>()
    const tiers = new Map<string, number>()
    const sources = new Map<string, number>()
    for (const it of items.filter((x) => x.block === block)) {
      const p = it.provenance as { stratum: string; source: string }
      strata.set(p.stratum, (strata.get(p.stratum) ?? 0) + 1)
      sources.set(p.source, (sources.get(p.source) ?? 0) + 1)
      tiers.set(it.tier, (tiers.get(it.tier) ?? 0) + 1)
    }
    console.log(`[${block}] sources:`, Object.fromEntries(sources))
    console.log(`[${block}] strata:`, Object.fromEntries(strata))
    console.log(`[${block}] tiers:`, Object.fromEntries(tiers))
  }
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
