/**
 * Puppeteer end-to-end suite for the /rate expert study instrument.
 *
 * Verifies, against a running server (default http://localhost:3002) and the
 * REAL seeded pool (via a "testfull-" rater, whose ratings are flagged
 * is_test and deleted afterwards):
 *   1. serving order: 8 screener items FIRST, then 40 calibration, then 45
 *      latin-square core — matching /api/ratings/items exactly
 *   2. progress reads "1 / 93" and advances
 *   3. resume anywhere: mid-SCREENER reload and mid-core reload both resume
 *      at the exact next item with correct progress
 *   4. completion screen after 93, then the extended arm ("bonus round")
 *   5. gate metadata: draft_ratings rows carry block='screener'/'calibration'
 *      /'core'/'extended', is_test=true, ms_taken recorded
 *   6. abbreviated "test-" rater smoke flow: 7 items (2 screener + 2
 *      calibration + 3 core)
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
    check('serves 93 items', data.items.length === 93, `got ${data.items.length}`)
    check(
      'items 1-8 are screener',
      servedBlocks.slice(0, 8).every((b) => b === 'screener')
    )
    check(
      'items 9-48 are calibration',
      servedBlocks.slice(8, 48).every((b) => b === 'calibration')
    )
    check('items 49-93 are core', servedBlocks.slice(48).every((b) => b === 'core'))
    check(
      'no provenance leaked',
      data.items.every((it) => !('provenance' in it) && !('winner' in it))
    )
    check('extended pool present', data.extendedItems.length === 700)
    const expectedOrder = data.items.map((it) => it.id)

    // ── Full flow with mid-screener + mid-core resume ────────────────
    console.log('full 93-item flow (testfull rater)')
    await page.goto(`${BASE}/rate?rater=${encodeURIComponent(fullRater)}&slot=${slot}`, {
      waitUntil: 'networkidle0',
    })
    let id = await currentItemId(page)
    check('first item matches assignment', id === expectedOrder[0], `got ${id}`)
    check('progress starts 1 / 93', (await progressText(page)) === '1 / 93')

    const seen = []
    for (let i = 0; i < 93; i++) {
      id = await currentItemId(page)
      seen.push(id)
      await rateCurrent(page, id, i % 2 === 0 ? 'A' : 'B')

      if (i === 2) {
        // Mid-SCREENER resume: reload after 3 ratings.
        await page.reload({ waitUntil: 'networkidle0' })
        const resumedId = await currentItemId(page)
        check('mid-screener resume shows item 4', resumedId === expectedOrder[3], `got ${resumedId}`)
        check('mid-screener progress 4 / 93', (await progressText(page)) === '4 / 93')
      }
      if (i === 59) {
        // Mid-core resume: reload after 60 ratings.
        await page.reload({ waitUntil: 'networkidle0' })
        const resumedId = await currentItemId(page)
        check('mid-core resume shows item 61', resumedId === expectedOrder[60], `got ${resumedId}`)
        check('mid-core progress 61 / 93', (await progressText(page)) === '61 / 93')
      }
    }
    check(
      'served exactly the assignment order',
      JSON.stringify(seen) === JSON.stringify(expectedOrder)
    )

    // ── Completion + extended arm ────────────────────────────────────
    await page.waitForSelector('[data-testid="rate-complete"]')
    check('completion screen after 93', true)
    await page.click('[data-testid="rate-continue"]')
    const extId = await currentItemId(page)
    check('extended arm serves an extended item', blockById.get(extId) === 'extended')
    check('extended progress reads "94 rated"', (await progressText(page)) === '94 rated')
    await rateCurrent(page, extId, 'A')
    const extId2 = await currentItemId(page)
    check('second extended item', blockById.get(extId2) === 'extended' && extId2 !== extId)
    await rateCurrent(page, extId2, 'B')

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
    check('45 core ratings recorded', byBlock.core?.n === 45)
    check('2 extended ratings recorded', byBlock.extended?.n === 2)
    check(
      'all rows is_test with ms_taken + choice',
      rows.every((r) => r.all_test && r.has_ms && r.has_choice)
    )
    const [dup] = await sql`
      select count(*)::int n, count(distinct item_id)::int d
      from draft_ratings where rater = ${fullRater}
    `
    check('one rating per item (upsert)', dup.n === 95 && dup.d === 95)

    // ── Abbreviated smoke flow ───────────────────────────────────────
    console.log('abbreviated test rater (7-item smoke)')
    const shortData = await fetchAssignment(shortRater, 0)
    const shortBlocks = shortData.items.map((it) => blockById.get(it.id))
    check('7 items for test- rater', shortData.items.length === 7)
    check(
      'smoke order: 2 screener, 2 calibration, 3 core',
      JSON.stringify(shortBlocks) ===
        JSON.stringify(['screener', 'screener', 'calibration', 'calibration', 'core', 'core', 'core'])
    )
    await page.goto(`${BASE}/rate?rater=${encodeURIComponent(shortRater)}&slot=0`, {
      waitUntil: 'networkidle0',
    })
    check('smoke progress starts 1 / 7', (await progressText(page)) === '1 / 7')
    for (let i = 0; i < 7; i++) {
      const cur = await currentItemId(page)
      await rateCurrent(page, cur, 'A')
    }
    await page.waitForSelector('[data-testid="rate-complete"]')
    check('smoke completion screen', true)
  } finally {
    await browser.close()
    // Cleanup: these raters are is_test by construction.
    const del = await sql`
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
