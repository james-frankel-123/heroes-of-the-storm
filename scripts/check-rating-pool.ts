/**
 * Offline verification of a frozen expert-rating pool (data/rating-items.json)
 * with the app's own assignment code (src/lib/rating/assignment.ts), WITHOUT
 * seeding or touching the database. Covers the design invariants that the
 * puppeteer suite (scripts/e2e-rate.mjs) checks against a seeded table, plus
 * pool-integrity checks:
 *   1. block counts 8 / 40 / 280 / 570 / 3 and 190 anchors per tier;
 *      calibration 14/13/13; ids 1..N unique
 *   2. for each of the 14 slots and two arbitrary rater names: 240 items, no
 *      duplicates; positions 1-48 = the 8 screener + 40 calibration items;
 *      positions 121/181/231 = the 3 catch items; the rest = 60 pairs +
 *      129 anchors (43 per tier)
 *   3. across all 14 slots: every pair covered exactly 3x; every anchor 3-4x
 *      (exactly 32 per tier with 4x)
 *   4. all real replays and tournament records distinct; every real game
 *      dated on/after the training cutoff; every item valid (5+5 distinct
 *      heroes)
 *   5. the test-rater smoke flow is 7 items
 *   6. (v6.1) every real game's DB game_date < pool.realGameEnd and build in
 *      pool.realGameBuilds; no sorted machine team in more than
 *      pool.maxTeamAppearances pairs (and the per-rater maximum exposure is
 *      printed); every pair carries finite labels (wpTeam0Sym,
 *      wpTeam0Sym_uncorrected) and finite, consistent OOD covariates
 * Not tested here (needs the DB or models): that labels/OOD values are the
 * right numbers, that real items match the DB, tier vs rank.
 * Usage: npx tsx scripts/check-rating-pool.ts [path] [--no-ood]
 */
import fs from 'node:fs'
import path from 'node:path'
import {
  fullSequence,
  CATCH_POSITIONS,
  NUM_SLOTS,
  SEQUENCE_LENGTH,
  type PoolIds,
} from '../src/lib/rating/assignment'

const args = process.argv.slice(2)
const noOod = args.includes('--no-ood')
const file = args.find((a) => !a.startsWith('--')) ?? path.resolve(__dirname, '../data/rating-items.json')
const pool = JSON.parse(fs.readFileSync(file, 'utf8'))
const items: any[] = pool.items
let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ok  ${name}`)
  else {
    failures++
    console.error(`FAIL  ${name}${detail ? ` - ${detail}` : ''}`)
  }
}

const byBlock = (b: string) => items.filter((i) => i.block === b)
const ids = items.map((i) => i.id)
check(`ids 1..${items.length} unique`, new Set(ids).size === items.length &&
  Math.min(...ids) === 1 && Math.max(...ids) === items.length)
const expect: Record<string, number> = { screener: 8, calibration: 40, pairs: 280, anchors: 570, catch: 3 }
for (const [b, n] of Object.entries(expect)) check(`${b}: ${n} items`, byBlock(b).length === n,
  `got ${byBlock(b).length}`)
for (const t of ['low', 'mid', 'high']) {
  check(`anchors ${t}: 190`, byBlock('anchors').filter((i) => i.tier === t).length === 190)
}
const cal = (t: string) => byBlock('calibration').filter((i) => i.tier === t).length
check(`calibration 14/13/13`, cal('low') === 14 && cal('mid') === 13 && cal('high') === 13)

const anchorsByTier = new Map<string, number[]>()
for (const it of byBlock('anchors')) {
  if (!anchorsByTier.has(it.tier)) anchorsByTier.set(it.tier, [])
  anchorsByTier.get(it.tier)!.push(it.id)
}
const P: PoolIds = {
  screener: byBlock('screener').map((i) => i.id),
  calibration: byBlock('calibration').map((i) => i.id),
  pairs: byBlock('pairs').map((i) => i.id),
  anchorsByTier,
  catch: byBlock('catch').map((i) => i.id),
}
const blockOf = new Map(items.map((i) => [i.id, i.block]))
const tierOf = new Map(items.map((i) => [i.id, i.tier]))
const pairCover = new Map<number, number>()
const anchorCover = new Map<number, number>()
let shapeOk = true
for (let slot = 0; slot < NUM_SLOTS; slot++) {
  for (const rater of [`rater-a-${slot}`, `Some Rater#${100 + slot}`]) {
    const seq = fullSequence(P, rater, slot)
    const first48 = seq.slice(0, 48)
    const catchAt = CATCH_POSITIONS.map((p) => blockOf.get(seq[p - 1]))
    const rest = seq.filter((_, i) => i >= 48 && !CATCH_POSITIONS.includes(i + 1))
    const restPairs = rest.filter((id) => blockOf.get(id) === 'pairs').length
    const restAnch = rest.filter((id) => blockOf.get(id) === 'anchors')
    const perTier = ['low', 'mid', 'high'].map((t) => restAnch.filter((id) => tierOf.get(id) === t).length)
    const ok =
      seq.length === SEQUENCE_LENGTH &&
      new Set(seq).size === seq.length &&
      first48.every((id) => ['screener', 'calibration'].includes(blockOf.get(id)!)) &&
      first48.filter((id) => blockOf.get(id) === 'screener').length === 8 &&
      catchAt.every((b) => b === 'catch') &&
      restPairs === 60 && restAnch.length === 129 && perTier.every((n) => n === 43)
    if (!ok) {
      shapeOk = false
      console.error(`  slot ${slot} rater ${rater}: len=${seq.length} pairs=${restPairs} anchors=${perTier}`)
    }
    if (rater.startsWith('rater-a-')) {
      for (const id of rest) {
        if (blockOf.get(id) === 'pairs') pairCover.set(id, (pairCover.get(id) ?? 0) + 1)
        if (blockOf.get(id) === 'anchors') anchorCover.set(id, (anchorCover.get(id) ?? 0) + 1)
      }
    }
  }
}
check('every slot: 240 items, 48-item opener, catch at 121/181/231, 60 pairs + 43/43/43 anchors', shapeOk)
check('every pair covered exactly 3x', P.pairs.every((id) => pairCover.get(id) === 3))
const anchCounts = [...anchorCover.values()]
check('every anchor covered 3-4x', P.anchorsByTier.size === 3 &&
  [...anchorsByTier.values()].flat().every((id) => [3, 4].includes(anchorCover.get(id) ?? 0)))
for (const [t, idsT] of anchorsByTier) {
  check(`anchors ${t}: exactly 32 covered 4x`, idsT.filter((id) => anchorCover.get(id) === 4).length === 32)
}
check('test smoke flow: 7 items', fullSequence(P, 'test-check', 0).length === 7)

const realIds = items.map((i) => i.provenance?.replayId).filter((x) => x != null)
check(`all ${realIds.length} real replays distinct`, new Set(realIds).size === realIds.length)
const recKeys = byBlock('pairs').map((i) => `${i.provenance.file}#${i.provenance.recordIndex}`)
check('all tournament records distinct', new Set(recKeys).size === recKeys.length)
const cutoff = pool.trainingSnapshotCutoff
check(`every real game dated >= ${cutoff}`, items.filter((i) => i.provenance?.gameDate)
  .every((i) => String(i.provenance.gameDate).slice(0, 10) >= cutoff))
const valid = (t: string[]) => Array.isArray(t) && t.length === 5
check('every item has 5+5 distinct heroes', items.every((i) =>
  valid(i.teams.team0) && valid(i.teams.team1) && new Set([...i.teams.team0, ...i.teams.team1]).size === 10))
// ── v6.1 checks ──
const reals = items.filter((i) => i.provenance?.replayId != null)
if (pool.realGameEnd) {
  check(`every real game DB date < ${pool.realGameEnd}`, reals.every((i) =>
    typeof i.provenance.gameDate === 'string' && !i.provenance.gameDate.endsWith('Z') &&
    i.provenance.gameDate < pool.realGameEnd))
} else console.log('  --  no realGameEnd in pool: upper date bound NOT tested')
if (pool.realGameBuilds) {
  check(`every real game on builds ${pool.realGameBuilds.join(', ')}`, reals.every((i) =>
    pool.realGameBuilds.includes(i.provenance.gameVersion)))
} else console.log('  --  no realGameBuilds in pool: build allow-list NOT tested')
const tkey = (t: string[]) => [...t].sort().join(',')
const teamUse = new Map<string, number>()
for (const it of byBlock('pairs')) for (const t of [it.teams.team0, it.teams.team1]) {
  teamUse.set(tkey(t), (teamUse.get(tkey(t)) ?? 0) + 1)
}
const maxUse = Math.max(...teamUse.values())
if (pool.maxTeamAppearances) {
  check(`no machine team in more than ${pool.maxTeamAppearances} pairs (max ${maxUse})`,
    maxUse <= pool.maxTeamAppearances)
} else console.log(`  --  no maxTeamAppearances in pool: team cap NOT tested (max ${maxUse})`)
const pairById = new Map(byBlock('pairs').map((i) => [i.id, i]))
const perSlot: number[] = []
for (let s = 0; s < NUM_SLOTS; s++) {
  const seq = fullSequence(P, `slot${s}-exposure`, s)
  const c = new Map<string, number>()
  for (const id of seq) {
    const it = pairById.get(id)
    if (!it) continue
    for (const t of [it.teams.team0, it.teams.team1]) c.set(tkey(t), (c.get(tkey(t)) ?? 0) + 1)
  }
  perSlot.push(Math.max(...c.values()))
}
console.log(`  ..  per-rater max exposure to one machine team (slots 0-13): ${perSlot.join(' ')}`)
const fin = (x: unknown) => typeof x === 'number' && Number.isFinite(x)
const labelKeys = ['naive', 'herostrength', 'enriched', 'consensus']
check('every pair has finite wpTeam0Sym and wpTeam0Sym_uncorrected (naive, herostrength, enriched, consensus)',
  byBlock('pairs').every((i) => ['wpTeam0Sym', 'wpTeam0Sym_uncorrected'].every((f) =>
    i.provenance[f] && labelKeys.every((k) => fin(i.provenance[f][k]) &&
      i.provenance[f][k] > 0 && i.provenance[f][k] < 1))))
if (!noOod) {
  const near = (a: number, b: number) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b))
  check('every pair has finite, consistent OOD covariates (team0/1, max, mean, matchup, refs)',
    byBlock('pairs').every((i) => {
      const p = i.provenance
      const ok = ['ood_var_team0', 'ood_var_team1', 'ood_var_max', 'ood_var_mean', 'ood_var_matchup']
        .every((k) => fin(p[k]) && p[k] >= 0)
      return ok && near(p.ood_var_max, Math.max(p.ood_var_team0, p.ood_var_team1)) &&
        near(p.ood_var_mean, 0.5 * (p.ood_var_team0 + p.ood_var_team1)) &&
        [p.ood_ref_team0, p.ood_ref_team1].every((r) => Array.isArray(r) && r.length === 5)
    }))
} else console.log('  --  --no-ood: OOD covariates NOT tested')

console.log(`\npool seed ${pool.seed}, ${items.length} items: ${failures ? failures + ' FAILURES' : 'ALL PASS'}`)
process.exit(failures ? 1 : 0)
