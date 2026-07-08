/**
 * Generate the 100-item set for the expert draft-rating study.
 *
 * Composition (seeded, reproducible given the same DB snapshot):
 *   - 80 tournament drafts from training/rerun2026/results/{roundrobin,constrained/roundrobin},
 *     stratified with oversampling of the paper's key comparisons:
 *       A. constrained_mcts vs mcts            12 items
 *       B. mcts vs enriched                    12 items
 *       C. behavior-anchored vs non-anchored   32 items
 *          (anchored = gd, cql_naive_a1.0, cql_enr_a2.0, gourdeau_disc)
 *       D. all other strategy pairs            24 items
 *   - 20 real ladder drafts from replay_draft_data as anchors (7 low / 7 mid / 6 high).
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

const SEED = 20260708
const REPO = path.resolve(__dirname, '..')
const RESULT_DIRS = [
  path.join(REPO, 'training/rerun2026/results/roundrobin'),
  path.join(REPO, 'training/rerun2026/results/constrained/roundrobin'),
]
const OUT_PATH = path.join(REPO, 'data/rating-items.json')

const ANCHORED = new Set(['gd', 'cql_naive_a1.0', 'cql_enr_a2.0', 'gourdeau_disc'])
const STRATA: { name: string; count: number }[] = [
  { name: 'constrained_mcts_vs_mcts', count: 12 },
  { name: 'mcts_vs_enriched', count: 12 },
  { name: 'vs_anchored', count: 32 },
  { name: 'other_pairs', count: 24 },
]
const LADDER_TIER_COUNTS: Record<string, number> = { low: 7, mid: 7, high: 6 }

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

async function loadLadderAnchors(): Promise<Candidate[]> {
  const sql = neon(process.env.DATABASE_URL!)
  const picked: Candidate[] = []
  for (const [tier, count] of Object.entries(LADDER_TIER_COUNTS)) {
    // Deterministic recent window; the committed JSON freezes the sample.
    const rows = (await sql`
      select replay_id, game_map, skill_tier, team0_heroes, team1_heroes, winner, game_date
      from replay_draft_data
      where skill_tier = ${tier}
      order by replay_id desc
      limit 2000
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
    for (const r of chosen) {
      picked.push({
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
      })
    }
  }
  return picked
}

async function main() {
  if (!process.env.DATABASE_URL) {
    console.error('DATABASE_URL not set. Run: set -a && source .env && set +a')
    process.exit(1)
  }
  const byStratum = loadTournamentCandidates()
  const all: Candidate[] = []
  for (const { name, count } of STRATA) {
    const cands = byStratum.get(name) ?? []
    console.log(`stratum ${name}: ${cands.length} candidates -> sampling ${count}`)
    all.push(...sampleStratum(cands, count, SEED ^ fnv1a(name)))
  }
  const ladder = await loadLadderAnchors()
  console.log(`ladder anchors: ${ladder.length}`)
  all.push(...ladder)

  // Shuffle so item id carries no stratum information, then assign ids 1..N.
  const shuffled = seededShuffle(all, SEED)
  const items = shuffled.map((c, i) => ({
    id: i + 1,
    map: c.map,
    tier: c.tier,
    teams: { team0: c.team0, team1: c.team1 },
    provenance: c.provenance,
  }))

  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true })
  fs.writeFileSync(OUT_PATH, JSON.stringify({ seed: SEED, generatedAt: new Date().toISOString(), items }, null, 2))
  console.log(`wrote ${items.length} items to ${OUT_PATH}`)

  // Composition report.
  const strata = new Map<string, number>()
  const tiers = new Map<string, number>()
  for (const it of items) {
    const s = (it.provenance as { stratum: string }).stratum
    strata.set(s, (strata.get(s) ?? 0) + 1)
    tiers.set(it.tier, (tiers.get(it.tier) ?? 0) + 1)
  }
  console.log('strata:', Object.fromEntries(strata))
  console.log('tiers:', Object.fromEntries(tiers))
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
