/**
 * Generate the item set for the expert draft-rating study (v4: paid raters,
 * fixed 240-item assignment, 14 slots — see oss-export/docs/prereg_expert_study.md).
 *
 * Blocks (block labels live server-side; item ids are a GLOBAL shuffle of all
 * 901 items so neither id nor serving position leaks block membership):
 *
 *   SCREENER — 8 items:
 *     - each pairs a REAL recent ladder draft against a CONSTRUCTED
 *       role-degenerate team (5 tanks, 5 healers, no-healer+no-frontline
 *       stacks, …) built from HERO_ROLE_FINE like the synthetic-augmentation
 *       generator. Same map/tier framing as the real draft; sides blinded
 *       like every other item. A competent rater should approach 100%
 *       (pick the real draft); the inclusion gate is >= 7/8 correct.
 *       Served INTERLEAVED among the 40 calibration items in positions 1-48
 *       (v4: no consecutive screener run, to dilute "weird = fake" priming).
 *
 *   CALIBRATION — 40 items:
 *     - real known-outcome ladder anchors, tier-stratified 14 low / 13 mid /
 *       13 high, distinct from every other item in the study. Every rater
 *       rates all 40 within positions 1-48. Comparability block + per-rater
 *       calibration estimates (NOT a gate; excluded from S1/S3).
 *
 *   PAIRS — 280 items (all machine-vs-machine tournament pairs, merged v3
 *   core+extended; every one is confirmatory-eligible under fixed assignment):
 *         A. constrained_mcts vs mcts            42 items
 *         B. mcts vs enriched                    42 items
 *         C. behavior-anchored vs non-anchored  118 items
 *            (anchored = gd, cql_naive_a1.0, cql_enr_a2.0, gourdeau_disc)
 *         D. all other strategy pairs            78 items
 *     Assignment: 14 blocks of 20; slot s rates blocks {s, s+1, s+2} mod 14
 *     → 60 pairs/rater, every pair rated by exactly 3 raters.
 *
 *   ANCHORS — 570 items (190/tier), real known-outcome ladder drafts:
 *     Assignment: per tier, a seeded global order dealt cyclically — slot s
 *     takes positions [43s, 43s+43) of the sequence i % 190 — so each rater
 *     gets 43/tier (129 total) and each anchor is rated by exactly 3 raters
 *     except the first 32 per tier in the dealt order, which get 4.
 *
 *   CATCH — 3 items (1/tier), screener-grade real-vs-degenerate pairs served
 *     at fixed positions 121 / 181 / 231 of every rater's 240-item sequence
 *     (late-session attention checks; gate >= 2/3; no endpoint).
 *
 * Rater burden: 48 + 60 + 129 + 3 = 240 items, identical for every rater.
 *
 * ALL real ladder drafts (screener/catch reals + calibration + anchors)
 * POSTDATE the 2026-05-22 training snapshot (replay_id > 63653039), so no
 * anchor can appear in any model's training data.
 *
 * After generation, run training/rerun2026/rating_items_ood.py to freeze the
 * OOD ensemble-variance covariates (reference-probe AND matchup-level) into
 * every machine-pair item's provenance, THEN seed with
 * scripts/seed-rating-items.ts.
 *
 * Output: data/rating-items.json — committed to the repo and seeded into the
 * rating_items table by scripts/seed-rating-items.ts. Provenance (strategy
 * labels, source, winner, model WPs, OOD covariates) lives ONLY in this file /
 * the DB provenance column; it is never sent to the client.
 *
 * Usage: set -a && source .env && set +a && npx tsx scripts/generate-rating-items.ts
 */
import fs from 'node:fs'
import path from 'node:path'
import { neon } from '@neondatabase/serverless'
import { fnv1a, mulberry32, seededShuffle } from '../src/lib/rating/assignment'
import { HERO_ROLES } from '../src/lib/data/hero-roles'

// SEED history: 20260708 original 500-item set; 20260709 calibration block;
// 20260710 calibration grown to 40; 20260711 post-adversarial-review
// restructure (screener block, post-snapshot anchors, latin square 150).
// 20260712 is the v4 paid-rater redesign: core/extended merged into PAIRS
// (280) + ANCHORS (570), new 3-item CATCH block, GLOBAL id shuffle so ids
// carry no block information (required now that the screener is interleaved).
const SEED = 20260712
// The machine-pair sample is pinned to the v3 seed: v3 drew each stratum's
// core+ext records in a single sampleStratum pass, so sampling the same
// totals with the same seed reproduces the IDENTICAL 280 records — their
// frozen wpTeam0Sym and OOD provenances (and the prereg's near-tie counts)
// carry over unchanged across the v4 re-freeze.
const PAIR_SAMPLE_SEED = 20260711
// Last replay id in the 2026-05-22 model-training snapshot. Every real ladder
// draft in the study must be strictly newer, so no anchor is train-set data.
const TRAINING_SNAPSHOT_MAX_REPLAY_ID = 63653039
const REPO = path.resolve(__dirname, '..')
const RESULT_DIRS = [
  path.join(REPO, 'training/rerun2026/results/roundrobin'),
  path.join(REPO, 'training/rerun2026/results/constrained/roundrobin'),
]
const OUT_PATH = path.join(REPO, 'data/rating-items.json')

const ANCHORED = new Set(['gd', 'cql_naive_a1.0', 'cql_enr_a2.0', 'gourdeau_disc'])
// Machine-pair matchup strata (v4: single merged count per stratum = the v3
// core+ext totals; all 280 valid tournament records are used).
const STRATA: { name: string; count: number }[] = [
  { name: 'constrained_mcts_vs_mcts', count: 42 },
  { name: 'mcts_vs_enriched', count: 42 },
  { name: 'vs_anchored', count: 118 },
  { name: 'other_pairs', count: 78 },
]
// Known-outcome ladder anchors. Screener reals = 3/3/2 (8); catch reals =
// 1/1/1 (3); calibration = 14/13/13 (40, shared); anchor block = 190/tier
// (570); all distinct replays, all post-snapshot.
const LADDER_SCREENER_COUNTS: Record<string, number> = { low: 3, mid: 3, high: 2 }
const LADDER_CATCH_COUNTS: Record<string, number> = { low: 1, mid: 1, high: 1 }
const LADDER_CALIB_COUNTS: Record<string, number> = { low: 14, mid: 13, high: 13 }
const LADDER_ANCHOR_COUNTS: Record<string, number> = { low: 190, mid: 190, high: 190 }
// Extra per-tier drafts pulled so screener/catch slots can skip real drafts
// whose heroes collide with the constructed degenerate team.
const LADDER_BUFFER = 20

// ── Fine-grained hero roles (must match training/shared.py HERO_ROLE_FINE) ──
const HERO_ROLE_FINE: Record<string, string> = {
  // Tanks
  "Anub'arak": 'tank', Arthas: 'tank', Blaze: 'tank', Cho: 'tank',
  Diablo: 'tank', 'E.T.C.': 'tank', Garrosh: 'tank', Johanna: 'tank',
  "Mal'Ganis": 'tank', Mei: 'tank', Muradin: 'tank', Stitches: 'tank',
  Tyrael: 'tank',
  // Bruisers
  Artanis: 'bruiser', Chen: 'bruiser', Deathwing: 'bruiser',
  Dehaka: 'bruiser', 'D.Va': 'bruiser', Gazlowe: 'bruiser',
  Hogger: 'bruiser', Imperius: 'bruiser', Leoric: 'bruiser',
  Malthael: 'bruiser', Ragnaros: 'bruiser', Rexxar: 'bruiser',
  Sonya: 'bruiser', Thrall: 'bruiser', Xul: 'bruiser', Yrel: 'bruiser',
  // Healers
  Alexstrasza: 'healer', Ana: 'healer', Anduin: 'healer',
  Auriel: 'healer', Brightwing: 'healer', Deckard: 'healer',
  Kharazim: 'healer', 'Li Li': 'healer', 'Lt. Morales': 'healer',
  'Lúcio': 'healer', Malfurion: 'healer', Rehgar: 'healer',
  Stukov: 'healer', Tyrande: 'healer', Uther: 'healer',
  Whitemane: 'healer',
  // Ranged Assassins — AA-based
  Cassia: 'ranged_aa', Falstad: 'ranged_aa', Fenix: 'ranged_aa',
  Greymane: 'ranged_aa', Hanzo: 'ranged_aa', Lunara: 'ranged_aa',
  Raynor: 'ranged_aa', 'Sgt. Hammer': 'ranged_aa', Sylvanas: 'ranged_aa',
  Tracer: 'ranged_aa', Tychus: 'ranged_aa', Valla: 'ranged_aa',
  "Zul'jin": 'ranged_aa',
  // Ranged Assassins — Mage
  Chromie: 'ranged_mage', Gall: 'ranged_mage', Genji: 'ranged_mage',
  "Gul'dan": 'ranged_mage', Jaina: 'ranged_mage', Junkrat: 'ranged_mage',
  "Kael'thas": 'ranged_mage', "Kel'Thuzad": 'ranged_mage',
  'Li-Ming': 'ranged_mage', Mephisto: 'ranged_mage', Nova: 'ranged_mage',
  Orphea: 'ranged_mage', Probius: 'ranged_mage', Tassadar: 'ranged_mage',
  // Pushers / Specialists
  Azmodan: 'pusher', Nazeebo: 'pusher', Zagara: 'pusher',
  Murky: 'pusher', 'The Lost Vikings': 'pusher',
  // Melee Assassins
  Alarak: 'melee_assassin', Illidan: 'melee_assassin',
  Kerrigan: 'melee_assassin', Maiev: 'melee_assassin',
  Qhira: 'melee_assassin', Samuro: 'melee_assassin',
  'The Butcher': 'melee_assassin', Valeera: 'melee_assassin',
  Zeratul: 'melee_assassin',
  // Support / Utility
  Abathur: 'support_utility', Medivh: 'support_utility', Zarya: 'support_utility',
  // Varian — own category
  Varian: 'varian',
}

/**
 * Degenerate-team recipes for screener/catch items: each spec lists the
 * fine-role groups (with counts) the constructed team is sampled from, plus
 * optional fixed heroes. Cho and Gall are never sampled (pairing constraint).
 * A competent rater should see every one of these as clearly worse than a
 * real draft.
 */
interface ScreenerSpec {
  type: string
  tier: string
  fixed?: string[]
  sample?: { role: string; n: number }[]
}
const SCREENER_SPECS: ScreenerSpec[] = [
  { type: 'five_tanks', tier: 'low', sample: [{ role: 'tank', n: 5 }] },
  { type: 'five_healers', tier: 'low', sample: [{ role: 'healer', n: 5 }] },
  {
    type: 'no_healer_no_frontline', // all backline damage, zero sustain/front
    tier: 'low',
    sample: [
      { role: 'ranged_aa', n: 2 },
      { role: 'ranged_mage', n: 2 },
      { role: 'melee_assassin', n: 1 },
    ],
  },
  { type: 'five_ranged_mages', tier: 'mid', sample: [{ role: 'ranged_mage', n: 5 }] },
  {
    type: 'three_tanks_two_healers_no_damage',
    tier: 'mid',
    sample: [
      { role: 'tank', n: 3 },
      { role: 'healer', n: 2 },
    ],
  },
  {
    type: 'five_pushers', // all-specialist: no healer, no frontline
    tier: 'mid',
    fixed: ['Azmodan', 'Nazeebo', 'Zagara', 'Murky', 'The Lost Vikings'],
  },
  { type: 'five_melee_assassins', tier: 'high', sample: [{ role: 'melee_assassin', n: 5 }] },
  {
    type: 'support_pusher_stack', // utility troll comp: no healer, no frontline, no core damage
    tier: 'high',
    fixed: ['Abathur', 'Medivh', 'Zarya'],
    sample: [{ role: 'pusher', n: 2 }],
  },
]
// Late-session attention checks (v4). Distinct recipes from the screener so
// no rater sees the same degenerate shape twice.
const CATCH_SPECS: ScreenerSpec[] = [
  { type: 'five_bruisers', tier: 'low', sample: [{ role: 'bruiser', n: 5 }] }, // no healer, no range
  {
    type: 'four_healers_one_pusher',
    tier: 'mid',
    sample: [
      { role: 'healer', n: 4 },
      { role: 'pusher', n: 1 },
    ],
  },
  { type: 'five_ranged_aa', tier: 'high', sample: [{ role: 'ranged_aa', n: 5 }] }, // no healer, no frontline
]

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

/**
 * Build one role-degenerate team from a spec, avoiding `forbidden` heroes
 * (the paired real draft's ten heroes). Returns null if the spec cannot be
 * satisfied without collisions.
 */
function buildDegenTeam(
  spec: ScreenerSpec,
  forbidden: Set<string>,
  rand: () => number
): string[] | null {
  const team: string[] = []
  for (const h of spec.fixed ?? []) {
    if (forbidden.has(h)) return null
    team.push(h)
  }
  for (const { role, n } of spec.sample ?? []) {
    const pool = Object.keys(HERO_ROLE_FINE)
      .filter(
        (h) =>
          HERO_ROLE_FINE[h] === role &&
          h !== 'Cho' &&
          h !== 'Gall' &&
          !forbidden.has(h) &&
          !team.includes(h)
      )
      .sort()
    if (pool.length < n) return null
    // Seeded sample without replacement.
    for (let k = 0; k < n; k++) {
      const idx = Math.floor(rand() * pool.length)
      team.push(pool.splice(idx, 1)[0])
    }
  }
  return team.length === 5 ? team : null
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
            // later expert-vs-model comparison. FROZEN at generation time.
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
  let stall = 0
  while (picked.length < count && stall < pairKeys.length * 2) {
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
    if (!best) {
      stall++
      continue
    }
    stall = 0
    const key = best.team0.join(',') + '|' + best.team1.join(',') + '|' + best.map
    usedKeys.add(key)
    tierCount.set(best.tier, (tierCount.get(best.tier) ?? 0) + 1)
    mapCount.set(best.map, (mapCount.get(best.map) ?? 0) + 1)
    picked.push(best)
  }
  if (picked.length < count) {
    throw new Error(`stratum sampling exhausted at ${picked.length}/${count}`)
  }
  return picked
}

interface LadderDraft {
  replay_id: number
  game_map: string
  skill_tier: string
  team0_heroes: string[]
  team1_heroes: string[]
  winner: number
  game_date: string
}

/**
 * Ladder drafts for one tier: `count` distinct valid drafts, map-spread, ALL
 * strictly newer than the training snapshot (replay_id > 63653039).
 */
async function loadLadderTier(tier: string, count: number): Promise<LadderDraft[]> {
  const sql = neon(process.env.DATABASE_URL!)
  // Deterministic recent window; the committed JSON freezes the sample. Pull a
  // generous window so enough valid drafts survive filtering. The snapshot
  // floor guarantees no anchor was ever seen by any trained model.
  const rows = (await sql`
    select replay_id, game_map, skill_tier, team0_heroes, team1_heroes, winner, game_date
    from replay_draft_data
    where skill_tier = ${tier}
      and replay_id > ${TRAINING_SNAPSHOT_MAX_REPLAY_ID}
    order by replay_id desc
    limit 6000
  `) as LadderDraft[]
  const valid = rows.filter((r) => validTeams(r.team0_heroes, r.team1_heroes))
  if (valid.length < count) {
    throw new Error(
      `ladder tier ${tier}: only ${valid.length} valid post-snapshot drafts, need ${count}`
    )
  }
  const shuffled = seededShuffle(valid, fnv1a(String(SEED) + '|ladder|' + tier))
  const mapCount = new Map<string, number>()
  const chosen: LadderDraft[] = []
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
  return chosen
}

function ladderCandidate(r: LadderDraft): Candidate {
  return {
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
  }
}

/**
 * Build real-vs-degenerate items (screener or catch) for one tier's specs
 * from the front of that tier's draft list. Consumed replay_ids are added to
 * `used` so later blocks skip them.
 */
function buildDegenItems(
  specs: ScreenerSpec[],
  drafts: LadderDraft[],
  used: Set<number>,
  sourceLabel: 'screener' | 'catch'
): Candidate[] {
  const out: Candidate[] = []
  for (const spec of specs) {
    const rand = mulberry32(fnv1a(String(SEED) + '|' + sourceLabel + '|' + spec.type))
    let built: { real: LadderDraft; degen: string[] } | null = null
    for (const d of drafts) {
      if (used.has(d.replay_id)) continue
      // The real side shown is the WINNING team of a real game, so the
      // "correct" answer is unambiguous even beyond the degeneracy contrast.
      const realTeam = d.winner === 0 ? d.team0_heroes : d.team1_heroes
      const degen = buildDegenTeam(spec, new Set(realTeam), rand)
      if (degen && validTeams(realTeam, degen)) {
        built = { real: d, degen }
        break
      }
    }
    if (!built) throw new Error(`${sourceLabel} ${spec.type}: no compatible real draft found`)
    used.add(built.real.replay_id)
    const realTeam =
      built.real.winner === 0 ? built.real.team0_heroes : built.real.team1_heroes
    // Randomize which canonical side holds the real draft (display side is
    // additionally swapped per rater at serve time).
    const realSide = rand() < 0.5 ? 0 : 1
    const team0 = realSide === 0 ? realTeam : built.degen
    const team1 = realSide === 0 ? built.degen : realTeam
    out.push({
      map: built.real.game_map,
      tier: built.real.skill_tier,
      team0,
      team1,
      pairKey: sourceLabel,
      provenance: {
        source: sourceLabel,
        stratum: sourceLabel,
        degenType: spec.type,
        // `winner` = the REAL team's canonical side (the correct answer),
        // matching the anchor scoring convention.
        winner: realSide,
        replayId: built.real.replay_id,
        gameDate: built.real.game_date,
      },
    })
  }
  return out
}

async function main() {
  if (!process.env.DATABASE_URL) {
    console.error('DATABASE_URL not set. Run: set -a && source .env && set +a')
    process.exit(1)
  }
  const byStratum = loadTournamentCandidates()
  const pairs: Candidate[] = []
  for (const { name, count } of STRATA) {
    const cands = byStratum.get(name) ?? []
    console.log(`stratum ${name}: ${cands.length} candidates -> sampling ${count}`)
    pairs.push(...sampleStratum(cands, count, PAIR_SAMPLE_SEED ^ fnv1a(name)))
  }

  const screener: Candidate[] = []
  const catchItems: Candidate[] = []
  const calibLadder: Candidate[] = []
  const anchorLadder: Candidate[] = []
  for (const tier of Object.keys(LADDER_ANCHOR_COUNTS)) {
    const screenerSpecs = SCREENER_SPECS.filter((s) => s.tier === tier)
    const catchSpecs = CATCH_SPECS.filter((s) => s.tier === tier)
    const calibN = LADDER_CALIB_COUNTS[tier]
    const anchorN = LADDER_ANCHOR_COUNTS[tier]
    if (screenerSpecs.length !== LADDER_SCREENER_COUNTS[tier]) {
      throw new Error(`screener spec/tier count mismatch for ${tier}`)
    }
    if (catchSpecs.length !== LADDER_CATCH_COUNTS[tier]) {
      throw new Error(`catch spec/tier count mismatch for ${tier}`)
    }
    // One pass per tier (+ buffer for degen-compatibility scans), consumed
    // screener → catch → calibration → anchors, so all blocks are guaranteed
    // disjoint replays.
    const drafts = await loadLadderTier(
      tier,
      screenerSpecs.length + catchSpecs.length + calibN + anchorN + LADDER_BUFFER
    )
    const used = new Set<number>()
    screener.push(...buildDegenItems(screenerSpecs, drafts, used, 'screener'))
    catchItems.push(...buildDegenItems(catchSpecs, drafts, used, 'catch'))
    const remaining = drafts.filter((d) => !used.has(d.replay_id))
    calibLadder.push(...remaining.slice(0, calibN).map(ladderCandidate))
    anchorLadder.push(...remaining.slice(calibN, calibN + anchorN).map(ladderCandidate))
    console.log(
      `ladder ${tier}: ${screenerSpecs.length} screener + ${catchSpecs.length} catch + ` +
        `${calibN} calibration + ${anchorN} anchors (all post-snapshot)`
    )
  }

  // ── Verify all-unique replay/record usage across the whole pool ──
  const allCandidates: { c: Candidate; block: string }[] = [
    ...screener.map((c) => ({ c, block: 'screener' })),
    ...calibLadder.map((c) => ({ c, block: 'calibration' })),
    ...pairs.map((c) => ({ c, block: 'pairs' })),
    ...anchorLadder.map((c) => ({ c, block: 'anchors' })),
    ...catchItems.map((c) => ({ c, block: 'catch' })),
  ]
  const replayIds = allCandidates
    .map(({ c }) => c.provenance.replayId)
    .filter((x) => x !== undefined) as number[]
  if (new Set(replayIds).size !== replayIds.length) {
    throw new Error('duplicate ladder replay usage across blocks')
  }
  const recordKeys = allCandidates
    .filter(({ c }) => c.provenance.source === 'tournament')
    .map(({ c }) => `${c.provenance.file}#${c.provenance.recordIndex}`)
  if (new Set(recordKeys).size !== recordKeys.length) {
    throw new Error('duplicate tournament record usage across blocks')
  }
  console.log(
    `uniqueness verified: ${replayIds.length} distinct replays, ` +
      `${recordKeys.length} distinct tournament records`
  )

  // ── GLOBAL id shuffle: ids carry no block/stratum/source information ──
  // (v4 requirement: the screener is interleaved into positions 1-48, so
  // sequential screener ids would let an id-watching rater spot gate items.)
  const shuffled = seededShuffle(allCandidates, SEED ^ 0x91d5)
  const items = shuffled.map(({ c, block }, i) => ({
    id: i + 1,
    block: block as 'screener' | 'calibration' | 'pairs' | 'anchors' | 'catch',
    map: c.map,
    tier: c.tier,
    teams: { team0: c.team0, team1: c.team1 },
    provenance: c.provenance,
  }))

  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true })
  fs.writeFileSync(
    OUT_PATH,
    JSON.stringify(
      {
        seed: SEED,
        generatedAt: new Date().toISOString(),
        trainingSnapshotMaxReplayId: TRAINING_SNAPSHOT_MAX_REPLAY_ID,
        items,
      },
      null,
      2
    )
  )
  console.log(
    `wrote ${items.length} items (${screener.length} screener + ` +
      `${calibLadder.length} calibration + ${pairs.length} pairs + ` +
      `${anchorLadder.length} anchors + ${catchItems.length} catch) to ${OUT_PATH}`
  )

  // Composition report.
  for (const block of ['screener', 'calibration', 'pairs', 'anchors', 'catch'] as const) {
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
  // Post-snapshot verification line for the prereg.
  const minReplay = Math.min(...replayIds)
  console.log(
    `min anchor replay_id = ${minReplay} (> snapshot ${TRAINING_SNAPSHOT_MAX_REPLAY_ID}: ${
      minReplay > TRAINING_SNAPSHOT_MAX_REPLAY_ID
    })`
  )
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
