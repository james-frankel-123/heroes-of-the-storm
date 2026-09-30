/**
 * Heroes Profile MMR-history pull (paper 3: at-game-time ratings).
 *
 * Endpoint: GET /api/external/v1/players/mmr/history?battletag=&region=&game_type=Storm League
 *   -> { history: [ { replayID, game_date, x_label, mmr_date_parsed, winner,
 *        mmr_ran, game_map, hero:{id,name,new_role,...}, hero_id,
 *        player_conservative_rating, player_change, hero_conservative_rating,
 *        hero_change, role_conservative_rating, role_change, mmr, mmr_change } ],
 *        league_tiers: [...] }
 *   One entry per match; ratings are POST-game in HP's processing order;
 *   mmr = 1800 + 40 * player_conservative_rating, mmr_change = player_change.
 *   Player, hero and role ratings all come in the one call, so one call per
 *   player (plus pages if the live endpoint paginates).
 *
 * Quota: the Player/MMR family has its own 50K/week pool (Developer plan),
 * separate from Replay/Data, Player/Match/History, etc. This job paces at
 * RATE calls/min and never lets its own calls in any rolling 7 days exceed
 * WEEKLY_BUDGET (default 45K, leaving headroom). It uses only the v1 key
 * (HEROES_PROFILE_V2_API_KEY), so the old-key backfills are untouched.
 *
 * Live gate: until live data is activated on the HP account, v1 answers
 * with fixture data (header x-hp-data-source: fixture). This job then logs
 * and exits WITHOUT storing anything, so its hourly cron entry is a no-op
 * until activation and starts the pull by itself afterwards.
 *
 * Files (training/personalization/cache/mmr_history/):
 *   panel.tsv              input, from training/personalization/p3_mmr_panel.py
 *   shards/<run>.jsonl.gz  one JSON record per account (rows in compact columns);
 *                          sync-flushed after each record, so a killed run's
 *                          shard is readable up to its last record
 *   done.jsonl             progress marks, appended AFTER the record is flushed
 *   ledger.json            call counts per hour (rolling-week budget)
 *   bench.json             set on quota_exceeded; runs exit until it passes
 * A record written but not marked (crash in between) is refetched; readers
 * dedupe on (region, blizz_id, replayID).
 *
 * Usage:
 *   npx tsx sync/fetch-mmr-history.ts              # run until done/benched
 *   npx tsx sync/fetch-mmr-history.ts --status
 *   npx tsx sync/fetch-mmr-history.ts --dry-run --limit 3 --out /tmp/x
 *        (fixture allowed; writes only under --out)
 *   --retry-errors   re-attempt accounts marked status=error
 */
import 'dotenv/config'
import * as fs from 'fs'
import * as path from 'path'
import * as zlib from 'zlib'
import { HeroesProfileApiV2 } from './api-client-v2'
import { isHpAccessPaused } from './hp-errors'
import { log } from './logger'

const ROOT = path.resolve(__dirname, '..')
const DEFAULT_DIR = path.join(ROOT, 'training/personalization/cache/mmr_history')
const RATE = Number(process.env.MMR_HISTORY_RATE ?? 5) // calls/min
const WEEKLY_BUDGET = Number(process.env.MMR_HISTORY_WEEKLY_BUDGET ?? 45_000)
const GAME_TYPE = 'Storm League'
const MAX_PAGES = 50
const MAX_CONSECUTIVE_FAILS = 20
const HOUR = 3_600_000
const WEEK = 7 * 24 * HOUR

const COLS = [
  'replayID', 'game_date', 'x_label', 'mmr_date_parsed', 'winner', 'mmr_ran',
  'game_map', 'hero_id', 'hero', 'role', 'player_conservative_rating',
  'player_change', 'hero_conservative_rating', 'hero_change',
  'role_conservative_rating', 'role_change', 'mmr', 'mmr_change',
] as const
const KNOWN = new Set<string>([...COLS, 'hero'])

interface PanelRow { order: number; region: number; blizzId: number; battletag: string; nGames: number }
interface Done { region: number; blizz_id: number; status: 'ok' | 'not_found' | 'error'; rows: number; calls: number; pages: number; at: string; err?: string }

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(name)
  return i >= 0 ? process.argv[i + 1] : undefined
}
const flag = (name: string) => process.argv.includes(name)

function readPanel(dir: string): PanelRow[] {
  const file = path.join(dir, 'panel.tsv')
  const lines = fs.readFileSync(file, 'utf8').trim().split('\n').slice(1)
  return lines.map(l => {
    const [order, region, blizz, tag, n] = l.split('\t')
    return { order: +order, region: +region, blizzId: +blizz, battletag: tag, nGames: +n }
  })
}

function readDone(dir: string): Map<string, Done> {
  const m = new Map<string, Done>()
  const f = path.join(dir, 'done.jsonl')
  if (!fs.existsSync(f)) return m
  for (const line of fs.readFileSync(f, 'utf8').split('\n')) {
    if (!line.trim()) continue
    try {
      const d = JSON.parse(line) as Done
      m.set(`${d.region}:${d.blizz_id}`, d)
    } catch { /* torn last line after a crash */ }
  }
  return m
}

class Ledger {
  private hours: Record<string, number> = {}
  constructor(private file: string) {
    if (fs.existsSync(file)) this.hours = JSON.parse(fs.readFileSync(file, 'utf8'))
  }
  add(n: number) {
    if (n <= 0) return
    const h = new Date(Math.floor(Date.now() / HOUR) * HOUR).toISOString()
    this.hours[h] = (this.hours[h] ?? 0) + n
  }
  lastWeek(): number {
    const cut = Date.now() - WEEK
    return Object.entries(this.hours).filter(([h]) => Date.parse(h) + HOUR > cut).reduce((s, [, n]) => s + n, 0)
  }
  /** ms until the rolling-week count drops below budget - need. */
  waitFor(need: number): number {
    let total = this.lastWeek()
    if (total + need <= WEEKLY_BUDGET) return 0
    const cut = Date.now() - WEEK
    const live = Object.entries(this.hours).filter(([h]) => Date.parse(h) + HOUR > cut).sort()
    for (const [h, n] of live) {
      total -= n
      if (total + need <= WEEKLY_BUDGET) return Date.parse(h) + HOUR + WEEK - Date.now()
    }
    return HOUR
  }
  save() {
    fs.writeFileSync(this.file + '.tmp', JSON.stringify(this.hours))
    fs.renameSync(this.file + '.tmp', this.file)
  }
}

function isFixture(api: HeroesProfileApiV2, body: any): boolean {
  if (api.lastDataSource === 'fixture') return true
  // Fixture fingerprint, in case the header ever goes missing.
  const h0 = body?.history?.[0]
  return !!h0 && Number(h0.replayID) >= 90_000_001 && h0.game_date === '2020-01-01 00:00:00'
}

function compact(entries: any[]): { rows: any[][]; extraKeys: string[] } {
  const extra = new Set<string>()
  const rows = entries.map(e => {
    for (const k of Object.keys(e)) if (!KNOWN.has(k)) extra.add(k)
    return COLS.map(c => {
      if (c === 'hero') return e.hero?.name ?? null
      if (c === 'role') return e.hero?.new_role ?? null
      if (c === 'hero_id') return e.hero_id ?? e.hero?.id ?? null
      return e[c] ?? null
    })
  })
  return { rows, extraKeys: [...extra] }
}

function nextPage(body: any, page: number): number | null {
  // v1 uses Laravel pagination on other player endpoints; follow it if present.
  const last = Number(body?.last_page ?? body?.pagination?.last_page ?? body?.meta?.last_page ?? 0)
  const nextUrl = body?.next_page_url ?? body?.pagination?.next_page_url ?? body?.links?.next ?? null
  if (nextUrl || (last && page < last)) return page + 1
  return null
}

async function status(dir: string) {
  const panel = readPanel(dir)
  const done = readDone(dir)
  const ledger = new Ledger(path.join(dir, 'ledger.json'))
  const by: Record<string, number> = {}
  let rows = 0
  for (const d of done.values()) { by[d.status] = (by[d.status] ?? 0) + 1; rows += d.rows }
  const left = panel.filter(p => !done.has(`${p.region}:${p.blizzId}`)).length
  const bench = path.join(dir, 'bench.json')
  const perDay = Math.min(RATE * 1440, WEEKLY_BUDGET / 7)
  console.log(`panel ${panel.length.toLocaleString()} | done ${done.size.toLocaleString()} ${JSON.stringify(by)} | left ${left.toLocaleString()}`)
  console.log(`history rows stored ${rows.toLocaleString()} | calls last 7d ${ledger.lastWeek().toLocaleString()} / budget ${WEEKLY_BUDGET.toLocaleString()}`)
  if (fs.existsSync(bench)) console.log(`benched: ${fs.readFileSync(bench, 'utf8')}`)
  console.log(`ETA at ~${Math.round(perDay).toLocaleString()} calls/day: ${(left / perDay).toFixed(1)} days of live pulling`)
}

async function main() {
  const dryRun = flag('--dry-run')
  const outDir = arg('--out')
  const dir = arg('--dir') ?? DEFAULT_DIR
  if (flag('--status')) return status(dir)
  if (dryRun && !outDir) throw new Error('--dry-run needs --out <dir> (never mixes test data into the real cache)')
  const writeDir = dryRun ? outDir! : dir
  const limit = Number(arg('--limit') ?? Infinity)
  const retryErrors = flag('--retry-errors')

  fs.mkdirSync(path.join(writeDir, 'shards'), { recursive: true })
  const benchFile = path.join(writeDir, 'bench.json')
  if (!dryRun && fs.existsSync(benchFile)) {
    const until = JSON.parse(fs.readFileSync(benchFile, 'utf8')).until
    if (Date.now() < Date.parse(until)) { log.info(`benched until ${until}; exiting`); return }
    fs.unlinkSync(benchFile) // our own marker file, not data
  }

  const key = process.env.HEROES_PROFILE_V2_API_KEY
  if (!key) throw new Error('HEROES_PROFILE_V2_API_KEY not set')
  const api = new HeroesProfileApiV2(key, RATE, 4)
  const ledger = new Ledger(path.join(writeDir, 'ledger.json'))

  const panel = readPanel(dir)
  const done = readDone(writeDir)
  const todo = panel.filter(p => {
    const d = done.get(`${p.region}:${p.blizzId}`)
    return !d || (retryErrors && d.status === 'error')
  }).slice(0, limit)
  log.info(`mmr-history: panel ${panel.length}, done ${done.size}, todo ${todo.length}, rate ${RATE}/min, ` +
    `week budget ${WEEKLY_BUDGET} (used ${ledger.lastWeek()})${dryRun ? ' [DRY RUN]' : ''}`)
  if (todo.length === 0) return

  const runId = new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '') + `-${process.pid}`
  let gz: zlib.Gzip | null = null
  let shard: fs.WriteStream | null = null
  const openShard = (): zlib.Gzip => {
    if (!gz) {
      gz = zlib.createGzip()
      shard = fs.createWriteStream(path.join(writeDir, 'shards', `${runId}.jsonl.gz`))
      gz.pipe(shard)
    }
    return gz
  }
  const doneFile = path.join(writeDir, 'done.jsonl')

  let stop = false
  const onSig = () => { log.warn('signal received; finishing current account'); stop = true }
  process.on('SIGTERM', onSig)
  process.on('SIGINT', onSig)

  let fails = 0, nOk = 0, nRows = 0
  const t0 = Date.now()
  for (const p of todo) {
    if (stop) break
    const wait = ledger.waitFor(1)
    if (wait > 0) {
      const until = new Date(Date.now() + wait).toISOString()
      log.info(`weekly budget reached (${ledger.lastWeek()}); benching until ${until}`)
      if (!dryRun) fs.writeFileSync(benchFile, JSON.stringify({ until, reason: 'own weekly budget' }))
      break
    }
    const before = api.getCallCount()
    const entries: any[] = []
    let pages = 0
    let meta: any = null
    let rec: Done | undefined
    try {
      for (let page: number | null = 1; page && pages < MAX_PAGES;) {
        const body: any = await api.fetch('players/mmr/history', {
          battletag: p.battletag,
          region: String(p.region),
          game_type: GAME_TYPE,
          pagination_page: page > 1 ? String(page) : undefined,
        })
        pages++
        if (isFixture(api, body) && !dryRun) {
          // fixture calls are not metered by HP, so they stay off the ledger
          log.info('v1 is still serving FIXTURE data (live data not activated on the HP account); storing nothing, exiting')
          stop = true
          break
        }
        entries.push(...(body?.history ?? []))
        if (pages === 1) meta = Object.fromEntries(Object.entries(body ?? {}).filter(([k]) => k !== 'history' && k !== 'league_tiers'))
        page = nextPage(body, page)
      }
      if (stop && entries.length === 0) break
      const { rows, extraKeys } = compact(entries)
      const record = {
        region: p.region, blizz_id: p.blizzId, battletag: p.battletag,
        fetched_at: new Date().toISOString(), data_source: api.lastDataSource,
        pages, meta, extra_keys: extraKeys, cols: COLS, rows,
      }
      const z = openShard()
      await new Promise<void>(res => z.write(JSON.stringify(record) + '\n', () => res()))
      await new Promise<void>(res => z.flush(zlib.constants.Z_SYNC_FLUSH, () => res()))
      rec = { region: p.region, blizz_id: p.blizzId, status: 'ok', rows: rows.length, calls: 0, pages, at: record.fetched_at }
      nOk++; nRows += rows.length; fails = 0
    } catch (err: any) {
      const msg = String(err?.message ?? err)
      if (msg.includes('Max calls')) {
        ledger.add(api.getCallCount() - before)
        ledger.save()
        const until = new Date(Date.now() + 6 * HOUR).toISOString()
        log.warn(`HP quota_exceeded on Player/MMR pool: ${msg}; benching until ${until}`)
        if (!dryRun) fs.writeFileSync(benchFile, JSON.stringify({ until, reason: msg.slice(0, 300) }))
        break
      }
      if (isHpAccessPaused(err)) {
        // account-level refusal (e.g. terms_not_accepted): don't mark this account; retry next run
        ledger.add(api.getCallCount() - before)
        log.error(`${msg}; stopping`)
        break
      }
      const m = msg.match(/API error (\d{3})/)
      const code = m ? Number(m[1]) : 0
      // 403 player_unavailable = the player went private: nothing to fetch.
      if (code === 404 || code === 422 || msg.includes('player_unavailable')) {
        rec = { region: p.region, blizz_id: p.blizzId, status: 'not_found', rows: 0, calls: 0, pages, at: new Date().toISOString(), err: msg.slice(0, 200) }
        fails = 0
      } else if (code >= 400 && code < 500) {
        rec = { region: p.region, blizz_id: p.blizzId, status: 'error', rows: 0, calls: 0, pages, at: new Date().toISOString(), err: msg.slice(0, 200) }
        if (code === 401 || code === 403) { log.error(`auth/plan error, stopping: ${msg}`); stop = true }
      } else {
        // network / 5xx after retries: leave unmarked, retry next run
        fails++
        log.warn(`transient failure for ${p.region}:${p.blizzId}: ${msg.slice(0, 200)}`)
        ledger.add(api.getCallCount() - before)
        if (fails >= MAX_CONSECUTIVE_FAILS) { log.error(`${fails} consecutive failures; stopping`); break }
        continue
      }
    }
    if (!rec) continue
    const used = api.getCallCount() - before
    rec.calls = used
    ledger.add(used)
    fs.appendFileSync(doneFile, JSON.stringify(rec) + '\n')
    if ((nOk + 1) % 25 === 0) {
      ledger.save()
      const mins = (Date.now() - t0) / 60_000
      log.info(`ok ${nOk} (${nRows.toLocaleString()} rows) | ${(api.getCallCount() / mins).toFixed(1)} calls/min | week ${ledger.lastWeek()}`)
    }
  }
  ledger.save()
  if (gz && shard) {
    const z: zlib.Gzip = gz, w: fs.WriteStream = shard
    const closed = new Promise<void>(res => w.on('close', () => res()))
    z.end()
    await closed
  }
  log.info(`mmr-history run done: ${nOk} accounts, ${nRows.toLocaleString()} rows, ${api.getCallCount()} calls`)
}

main().catch(err => { log.error(`fatal: ${err?.stack ?? err}`); process.exit(1) })
