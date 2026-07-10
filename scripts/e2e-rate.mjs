/**
 * Puppeteer end-to-end suite for the /rate expert study instrument
 * (v4: fixed 240-item paid design, 14 slots).
 *
 * Verifies, against a running server (default http://localhost:3002) and the
 * REAL seeded pool (via a "testfull-" rater, whose ratings are flagged
 * is_test and deleted afterwards):
 *   1. assignment shape: 240 items; positions 1-48 = the 8 screener + 40
 *      calibration items interleaved; positions 121/181/231 = the 3 catch
 *      items; the rest = 60 pairs + 129 anchors (43/tier)
 *   2. design invariants across ALL 14 slots: every pair covered exactly 3x,
 *      every anchor 3-4x, per-slot per-tier anchor count = 43
 *   3. consent notice before the first item; not shown again on resume
 *   4. progress reads "1 / 240" and advances; resume anywhere (mid-first-48
 *      and mid-assigned reloads land on the exact next item)
 *   5. completion screen after 240 (no volunteer arm)
 *   6. gate metadata: draft_ratings rows carry the five block labels with
 *      counts 8/40/60/129/3, is_test=true, ms_taken recorded
 *   7. abbreviated "test-" rater smoke flow: 7 items
 *   8. no provenance or block labels leaked to the client
 *
 * Usage: set -a && source .env && set +a && node scripts/e2e-rate.mjs [baseUrl]
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer'
import { neon } from '@neondatabase/serverless'

const BASE = process.argv[2] ?? 'http://localhost:3002'
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const pool = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../data/rating-items.json'), 'utf8')
)
const blockById = new Map(pool.items.map((it) => [it.id, it.block]))
const tierById = new Map(pool.items.map((it) => [it.id, it.tier]))
const CATCH_IDX = [120, 180, 230] // 0-indexed positions 121/181/231

let failures = 0
function check(name, cond, detail = '') {
  if (cond) console.log(`  ok  ${name}`)
  else {
    failures++
    console.error(`FAIL  ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

async function currentItemId(page) {
  await page.waitForSelector('[data-testid="rate-item"], [data-testid="rate-complete"]', {
    timeout: 30_000,
  })
  const el = await page.$('[data-testid="rate-item"]')
  if (!el) return null
  return Number(await el.evaluate((n) => n.getAttribute('data-item-id')))
}

async function progressText(page) {
  return (await page.$eval('[data-testid="rate-progress"]', (n) => n.textContent)).trim()
}

async function consentIfShown(page) {
  await page.waitForSelector(
    '[data-testid="rate-consent"], [data-testid="rate-item"], [data-testid="rate-complete"]',
    { timeout: 30_000 }
  )
  const consent = await page.$('[data-testid="rate-consent-agree"]')
  if (consent) {
    await consent.click()
    return true
  }
  return false
}

/** Answer the currently shown item and wait until the NEXT item (or the
 * completion screen) is shown. */
async function rateCurrent(page, prevId, whichTeam) {
  // Q1 slider: one keypress marks it touched.
  await page.keyboard.press(whichTeam === 'A' ? 'ArrowLeft' : 'ArrowRight')
  // Q2 forced choice + Q3 confidence.
  await page.click(`[data-testid="rate-choice-${whichTeam.toLowerCase()}"]`)
  await page.click('[data-testid="rate-conf-3"]')
  // Auto-advance fires ~500ms after all three are answered.
  await page.waitForFunction(
    (prev) => {
      const item = document.querySelector('[data-testid="rate-item"]')
      if (!item) return !!document.querySelector('[data-testid="rate-complete"]')
      return Number(item.getAttribute('data-item-id')) !== prev
    },
    { timeout: 30_000 },
    prevId
  )
}

async function fetchAssignment(rater, slot) {
  const res = await fetch(`${BASE}/api/ratings/items?rater=${encodeURIComponent(rater)}&slot=${slot}`)
  if (!res.ok) throw new Error(`items API ${res.status}`)
  return res.json()
}

async function main() {
  if (!process.env.DATABASE_URL) throw new Error('DATABASE_URL not set')
  const sql = neon(process.env.DATABASE_URL)
  const stamp = Date.now()
  const fullRater = `testfull-e2e-${stamp}`
  const shortRater = `test-e2e-${stamp}`
  const slot = 3

  const browser = await puppeteer.launch({ headless: 'shell', args: ['--no-sandbox'] })
  const page = await browser.newPage()
  page.setDefaultTimeout(30_000)

  try {
    // ── Assignment API shape ─────────────────────────────────────────
    console.log('assignment API')
    const data = await fetchAssignment(fullRater, slot)
    const servedBlocks = data.items.map((it) => blockById.get(it.id))
    check('serves 240 items', data.items.length === 240, `got ${data.items.length}`)
    const first48 = servedBlocks.slice(0, 48)
    check(
      'positions 1-48 = 8 screener + 40 calibration',
      first48.filter((b) => b === 'screener').length === 8 &&
        first48.filter((b) => b === 'calibration').length === 40
    )
    check(
      'screener is interleaved (not a consecutive opening run)',
      !servedBlocks.slice(0, 8).every((b) => b === 'screener')
    )
    check(
      'catch items at positions 121/181/231',
      CATCH_IDX.every((i) => servedBlocks[i] === 'catch') &&
        servedBlocks.filter((b) => b === 'catch').length === 3
    )
    const assigned = servedBlocks.slice(48).filter((b) => b !== 'catch')
    check(
      'assigned region = 60 pairs + 129 anchors',
      assigned.filter((b) => b === 'pairs').length === 60 &&
        assigned.filter((b) => b === 'anchors').length === 129
    )
    const anchorTiers = data.items
      .filter((it) => blockById.get(it.id) === 'anchors')
      .map((it) => tierById.get(it.id))
    check(
      'anchors tier-balanced 43/43/43',
      ['low', 'mid', 'high'].every((t) => anchorTiers.filter((x) => x === t).length === 43)
    )
    check(
      'no provenance or block leaked',
      data.items.every((it) => !('provenance' in it) && !('winner' in it) && !('block' in it))
    )
    const expectedOrder = data.items.map((it) => it.id)

    // ── Design invariants across all 14 slots ────────────────────────
    console.log('cross-slot coverage invariants (14 slots)')
    const pairCoverage = new Map()
    const anchorCoverage = new Map()
    for (let s = 0; s < 14; s++) {
      const d = await fetchAssignment(`testfull-cov-${stamp}-${s}`, s)
      for (const it of d.items) {
        const b = blockById.get(it.id)
        if (b === 'pairs') pairCoverage.set(it.id, (pairCoverage.get(it.id) ?? 0) + 1)
        if (b === 'anchors') anchorCoverage.set(it.id, (anchorCoverage.get(it.id) ?? 0) + 1)
      }
    }
    const allPairIds = pool.items.filter((it) => it.block === 'pairs').map((it) => it.id)
    const allAnchorIds = pool.items.filter((it) => it.block === 'anchors').map((it) => it.id)
    check(
      'every pair covered by exactly 3 slots',
      allPairIds.every((id) => pairCoverage.get(id) === 3),
      `coverage counts: ${[...new Set(allPairIds.map((id) => pairCoverage.get(id) ?? 0))]}`
    )
    const anchorCounts = allAnchorIds.map((id) => anchorCoverage.get(id) ?? 0)
    check(
      'every anchor covered by 3 or 4 slots',
      anchorCounts.every((n) => n === 3 || n === 4),
      `distinct counts: ${[...new Set(anchorCounts)]}`
    )
    check(
      'anchor judgments total 1806 (602/tier)',
      anchorCounts.reduce((a, b) => a + b, 0) === 1806
    )

    // ── Full flow: consent, resume, completion ───────────────────────
    console.log('full 240-item flow (testfull rater)')
    await page.goto(`${BASE}/rate?rater=${encodeURIComponent(fullRater)}&slot=${slot}`, {
      waitUntil: 'networkidle0',
    })
    check('consent notice shown before first item', await consentIfShown(page))
    let id = await currentItemId(page)
    check('first item matches assignment', id === expectedOrder[0], `got ${id}`)
    check('progress starts 1 / 240', (await progressText(page)) === '1 / 240')

    const seen = []
    for (let i = 0; i < 240; i++) {
      id = await currentItemId(page)
      seen.push(id)
      await rateCurrent(page, id, i % 2 === 0 ? 'A' : 'B')

      if (i === 2) {
        // Mid-first-48 resume: reload after 3 ratings.
        await page.reload({ waitUntil: 'networkidle0' })
        check('no consent re-prompt on resume', !(await consentIfShown(page)))
        const resumedId = await currentItemId(page)
        check('mid-first-48 resume shows item 4', resumedId === expectedOrder[3], `got ${resumedId}`)
        check('mid-first-48 progress 4 / 240', (await progressText(page)) === '4 / 240')
      }
      if (i === 149) {
        // Mid-assigned resume: reload after 150 ratings.
        await page.reload({ waitUntil: 'networkidle0' })
        await consentIfShown(page)
        const resumedId = await currentItemId(page)
        check('mid-assigned resume shows item 151', resumedId === expectedOrder[150], `got ${resumedId}`)
        check('mid-assigned progress 151 / 240', (await progressText(page)) === '151 / 240')
      }
    }
    check(
      'served exactly the assignment order',
      JSON.stringify(seen) === JSON.stringify(expectedOrder)
    )

    // ── Completion (no volunteer arm) ────────────────────────────────
    await page.waitForSelector('[data-testid="rate-complete"]')
    check('completion screen after 240', true)
    check('no "keep rating" volunteer arm', (await page.$('[data-testid="rate-continue"]')) === null)

    // ── Gate metadata in the DB ──────────────────────────────────────
    console.log('gate metadata (draft_ratings)')
    const rows = await sql`
      select block, count(*)::int n,
             bool_and(is_test) all_test,
             bool_and(ms_taken is not null) has_ms,
             bool_and(better_team in ('A','B')) has_choice
      from draft_ratings where rater = ${fullRater} group by block order by block
    `
    const byBlock = Object.fromEntries(rows.map((r) => [r.block, r]))
    check('8 screener ratings recorded', byBlock.screener?.n === 8)
    check('40 calibration ratings recorded', byBlock.calibration?.n === 40)
    check('60 pair ratings recorded', byBlock.pairs?.n === 60)
    check('129 anchor ratings recorded', byBlock.anchors?.n === 129)
    check('3 catch ratings recorded', byBlock.catch?.n === 3)
    check(
      'all rows is_test with ms_taken + choice',
      rows.every((r) => r.all_test && r.has_ms && r.has_choice)
    )
    const [dup] = await sql`
      select count(*)::int n, count(distinct item_id)::int d
      from draft_ratings where rater = ${fullRater}
    `
    check('one rating per item (upsert)', dup.n === 240 && dup.d === 240)

    // ── Abbreviated smoke flow ───────────────────────────────────────
    console.log('abbreviated test rater (7-item smoke)')
    const shortData = await fetchAssignment(shortRater, 0)
    const shortBlocks = shortData.items.map((it) => blockById.get(it.id))
    check('7 items for test- rater', shortData.items.length === 7)
    check(
      'smoke order: 2 screener, 2 calibration, 3 assigned',
      JSON.stringify(shortBlocks.slice(0, 4)) ===
        JSON.stringify(['screener', 'screener', 'calibration', 'calibration']) &&
        shortBlocks.slice(4).every((b) => b === 'pairs' || b === 'anchors')
    )
    await page.goto(`${BASE}/rate?rater=${encodeURIComponent(shortRater)}&slot=0`, {
      waitUntil: 'networkidle0',
    })
    await consentIfShown(page)
    check('smoke progress starts 1 / 7', (await progressText(page)) === '1 / 7')
    for (let i = 0; i < 7; i++) {
      const cur = await currentItemId(page)
      await rateCurrent(page, cur, 'A')
    }
    await page.waitForSelector('[data-testid="rate-complete"]')
    check('smoke completion screen', true)
  } finally {
    await browser.close()
    // Cleanup: these raters are is_test by construction; scope strictly to
    // the unique names created by THIS run.
    await sql`
      delete from draft_ratings
      where rater in (${fullRater}, ${shortRater}) and is_test = true
    `
    console.log(`cleanup: deleted e2e is_test rows for ${fullRater}, ${shortRater}`)
  }

  if (failures > 0) {
    console.error(`\n${failures} check(s) FAILED`)
    process.exit(1)
  }
  console.log('\nall e2e checks passed')
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
