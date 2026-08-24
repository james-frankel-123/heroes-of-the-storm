/**
 * Quick Match ingest worker (QM WP model — out-of-family OOD sanity check;
 * also feeds talents/personalization work).
 *
 * Enumerates Quick Match replays via Replay/Min_id (game_type filter, 1000
 * per call, effectively unmetered) walking ASCENDING from the oldest known
 * patch-2.55 upload (min replay_id in replay_draft_data — same upload
 * timeline as our ranked corpus). For every listed QM game on patch 2.55
 * that is parsed+valid, fetches Replay/Data and stores:
 *   - qm_games: replay-level row (map, date, version, region) + the listing
 *     row's avg_mmr / league_tier / rank (Replay/Data does not carry them)
 *   - replay_players: per-player rows via the shared storeReplayPlayers
 *     helper (hero, team, winner, MMRs, party, talents, scoreboard)
 *   - replay_extras: replay-level leftovers (fingerprint, xp breakdown, ...)
 *
 * Quota split (Max, 2026-07-14, training/QUOTA_ALLOCATION_PLAN.md): this
 * worker's share of key 1's 180/min is set PER BUCKET (recent bucket 80/min
 * in phase A, fills 40/min in phase B); sync/refetch-players.ts takes the
 * rest. Key 2 is left to the refetch + daemon (its replay-data quota is
 * small and usually exhausted).
 *
 * - Resumable: cursor checkpoint in qm_fetch_state (single row). Killed
 *   mid-batch → the listing page is re-read on resume and already-stored
 *   games are skipped by a qm_games existence check (idempotent).
 * - Off-patch / invalid / deleted listings are counted skipped; the cursor
 *   moves past them without spending Replay/Data quota.
 * - Weekly quota exhaustion ("non-JSON response" / "Max calls"): sleeps 60
 *   min, then re-probes.
 * - Caught up to the head of the listing: sleeps 60 min, then polls again
 *   (new QM games keep arriving; the worker is a long-running slurper).
 *
 * Usage:
 *   npx tsx sync/fetch-qm.ts             # run the ingest loop
 *   npx tsx sync/fetch-qm.ts --status    # progress + rate
 *
 * Detached launch:
 *   nohup npx tsx sync/fetch-qm.ts >> sync/logs/fetch-qm.log 2>&1 &
 */
import { sql, eq, inArray } from 'drizzle-orm'
import { createHpApi } from './hp-api'
import { createDb, SyncDb } from './db'
import { log } from './logger'
import { storeReplayPlayers } from './player-store'
import { qmGames, qmFetchState } from '../src/lib/db/schema'

const WORKERS = 3
const STRATA = 12 // per-bucket round-robin strata (era-uniform partial fills)
const PATCH_PREFIX = '2.55'
const QUOTA_BACKOFF_MS = 60 * 60_000
const CAUGHT_UP_SLEEP_MS = 60 * 60_000

/**
 * Era-bucket priority plan (2026-07-14, see training/QUOTA_ALLOCATION_PLAN.md).
 * 2021 (95.6K) and 2022 (128.2K) are already covered by the legacy ascending
 * scan. Buckets are matched by GAME DATE window (via a game_version -> date
 * map resolved from the ranked corpus at startup, since replay ids are
 * upload-ordered and eras overlap heavily in id space). `rate` is the
 * worker's own share of key 1's 180/min while that bucket is active:
 * the recent bucket is front-loaded (phase A), the fills run at 40/min
 * (phase B) — bump refetch-players to 140/min when phase A ends.
 */
interface EraBucket {
  name: string
  from: string // inclusive game-date window
  to: string   // exclusive
  target: number // qm_games rows with game_date in window
  rate: number
}
const BUCKETS: EraBucket[] = [
  { name: 'recent', from: '2025-07-01', to: '2027-01-01', target: 120_000, rate: 80 },
  { name: 'y2024', from: '2024-01-01', to: '2025-01-01', target: 90_000, rate: 40 },
  { name: 'y2023', from: '2023-01-01', to: '2024-01-01', target: 90_000, rate: 40 },
  { name: 'y2025h1', from: '2025-01-01', to: '2025-07-01', target: 60_000, rate: 40 },
]

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

interface ListingRow {
  replayID: number
  region: number | null
  parsed: number | null
  valid: number | null
  deleted: unknown
  game_type: string
  game_version: string | null
  game_map: string | null
  avg_mmr: number | null
  league_tier: number | null
  rank: string | null
}

interface QmState {
  cursor: number
  processedCount: number
  playerRowsUpserted: number
  skippedCount: number
  errorCount: number
  startedAt: Date
}

async function loadState(db: SyncDb, bucket: string): Promise<QmState | null> {
  const rows = await db.select().from(qmFetchState).where(eq(qmFetchState.bucket, bucket)).limit(1)
  if (rows.length === 0) return null
  const r = rows[0]
  return {
    cursor: r.cursor,
    processedCount: r.processedCount,
    playerRowsUpserted: r.playerRowsUpserted,
    skippedCount: r.skippedCount,
    errorCount: r.errorCount,
    startedAt: r.startedAt,
  }
}

async function initState(db: SyncDb, bucket: string, cursor: number): Promise<QmState> {
  const state: QmState = {
    cursor,
    processedCount: 0,
    playerRowsUpserted: 0,
    skippedCount: 0,
    errorCount: 0,
    startedAt: new Date(),
  }
  await db.insert(qmFetchState).values({ ...state, bucket })
  return state
}

async function saveState(db: SyncDb, bucket: string, state: QmState) {
  await db.update(qmFetchState).set({
    cursor: state.cursor,
    processedCount: state.processedCount,
    playerRowsUpserted: state.playerRowsUpserted,
    skippedCount: state.skippedCount,
    errorCount: state.errorCount,
    updatedAt: new Date(),
  }).where(eq(qmFetchState.bucket, bucket))
}

/**
 * game_version -> [min, max] game_date (ms) from the ranked corpus (same
 * builds). A bucket matches a build by RANGE OVERLAP with its window: a
 * build released in June that runs through August must count for a window
 * starting July 1 (min-date-only matching silently excluded every
 * window-straddling build and made the recent bucket match nothing).
 */
async function versionDateMap(db: SyncDb): Promise<Map<string, [number, number]>> {
  const rows = await db.execute(
    sql`SELECT game_version AS v, min(game_date) AS a, max(game_date) AS b
        FROM replay_draft_data GROUP BY 1`
  ).then(r => r.rows)
  return new Map(rows.map(r => [
    String(r.v),
    [new Date(String(r.a)).getTime(), new Date(String(r.b)).getTime()],
  ]))
}

/** Fetched-so-far count for a bucket (resume-safe: derived from qm_games). */
async function bucketProgress(db: SyncDb, b: EraBucket): Promise<number> {
  const [row] = await db.execute(
    sql`SELECT count(*) AS n FROM qm_games
        WHERE game_date >= ${b.from} AND game_date < ${b.to}`
  ).then(r => r.rows)
  return Number(row.n)
}

/**
 * Stratum start ids for a bucket: replay-id quantiles of ranked games in the
 * date window. Scanning strata round-robin instead of one ascending cursor
 * means a PARTIALLY filled bucket is an approximately uniform sample of its
 * era, not a biased prefix ("2024" never degrades to "Jan-Apr 2024" if the
 * quota gets cut mid-bucket). Quantiles (not a linear id split) give each
 * stratum equal game mass.
 */
async function strataBounds(db: SyncDb, b: EraBucket, n: number): Promise<number[]> {
  const fracs = Array.from({ length: n }, (_, i) => i / n)
  const rows = await db.execute(
    sql`SELECT unnest(percentile_disc(${sql.raw(`ARRAY[${fracs.join(',')}]`)})
          WITHIN GROUP (ORDER BY replay_id)) AS q
        FROM replay_draft_data
        WHERE game_date >= ${b.from} AND game_date < ${b.to}`
  ).then(r => r.rows)
  return rows.map(r => Number(r.q)).filter(q => Number.isFinite(q) && q > 0)
}

// ── Status ───────────────────────────────────────────────────────────

async function printStatus(db: SyncDb) {
  console.log('── QM ingest status (era buckets) ────────────────────')
  for (const b of BUCKETS) {
    const done = await bucketProgress(db, b)
    const [strataRow] = await db.execute(
      sql`SELECT count(*) AS n FROM qm_fetch_state WHERE bucket LIKE ${b.name + '#%'}`
    ).then(r => r.rows)
    const started = Number(strataRow.n) > 0
    console.log(
      `  ${b.name.padEnd(8)} [${b.from}..${b.to})  ` +
      `${done.toLocaleString()} / ${b.target.toLocaleString()}` +
      (done >= b.target ? '  DONE' : started ? `  (${strataRow.n} strata)` : '  (not started)')
    )
  }
  const [counts] = await db.execute(
    sql`SELECT count(*) AS games FROM qm_games`
  ).then(r => r.rows)
  console.log(`  total QM games stored: ${Number(counts.games).toLocaleString()}`)
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  const key1 = process.env.HEROES_PROFILE_API_KEY
  if (!process.env.DATABASE_URL) { log.error('DATABASE_URL required'); process.exit(1) }
  const db = createDb()

  if (process.argv.includes('--status')) {
    await printStatus(db)
    return
  }
  if (!key1) { log.error('HEROES_PROFILE_API_KEY required'); process.exit(1) }

  // --cron: one-shot mode — exit on quota exhaustion / caught-up instead of
  // sleeping, so the Neon endpoint can suspend between scheduled runs.
  const cron = process.argv.includes('--cron')

  const vmap = await versionDateMap(db)
  log.info(`version->date map: ${vmap.size} builds`)

  let stopping = false
  const requestStop = (sig: string) => {
    log.warn(`${sig} received — finishing in-flight calls, then checkpointing`)
    stopping = true
  }
  process.on('SIGINT', () => requestStop('SIGINT'))
  process.on('SIGTERM', () => requestStop('SIGTERM'))

  // Bucket loop: work the first bucket whose target is unmet.
  for (const bucket of BUCKETS) {
    if (stopping) break
    let done = await bucketProgress(db, bucket)
    if (done >= bucket.target) {
      log.info(`Bucket ${bucket.name}: ${done.toLocaleString()}/${bucket.target.toLocaleString()} — already met`)
      continue
    }
    const fromMs = new Date(bucket.from).getTime()
    const toMs = new Date(bucket.to).getTime()
    const inWindow = (v: string | null): boolean => {
      if (!v) return false
      const range = vmap.get(v)
      // Build's active range overlaps the window (edge builds included;
      // bucketProgress counts by real game_date, so targets stay exact).
      return range !== undefined && range[1] >= fromMs && range[0] < toMs
    }

    const api = createHpApi('key1', Math.min(bucket.rate, 110), 3)
    let exhaustedUntil = 0

    // Per-stratum cursors (state rows keyed `${bucket}#${i}`), scanned
    // round-robin so partial fills stay era-uniform.
    const bounds = await strataBounds(db, bucket, STRATA)
    interface Stratum { idx: number; key: string; state: QmState; end: number | null; exhausted: boolean }
    const strata: Stratum[] = []
    for (let i = 0; i < bounds.length; i++) {
      const key = `${bucket.name}#${i}`
      let st = await loadState(db, key)
      if (!st) st = await initState(db, key, bounds[i])
      const end = i + 1 < bounds.length ? bounds[i + 1] : null // last stratum open-ended
      strata.push({ idx: i, key, state: st, end, exhausted: end !== null && st.cursor >= end })
    }
    log.info(`Bucket ${bucket.name}: ${strata.length} strata, ` +
      `${done.toLocaleString()}/${bucket.target.toLocaleString()} fetched, rate ${bucket.rate}/min`)

    const runStart = Date.now()
    let processedThisRun = 0
    let turn = 0

    while (!stopping && done < bucket.target) {
      if (exhaustedUntil > Date.now()) {
        if (cron) {
          log.info(`Quota benched until ${new Date(exhaustedUntil).toISOString()} — exiting (cron mode)`)
          stopping = true
          continue
        }
        await sleep(Math.min(exhaustedUntil - Date.now(), 60_000))
        continue
      }

      const active = strata.filter(s => !s.exhausted)
      if (active.length === 0) {
        // Historical buckets have far more QM games than their targets, so
        // this normally only happens on the open-ended bucket: the head of
        // the listing was reached. Wait for new games, then re-open the
        // last stratum.
        if (cron) {
          log.info(`Bucket ${bucket.name}: all strata exhausted at ${done.toLocaleString()}/${bucket.target.toLocaleString()} — exiting (cron mode)`)
          break
        }
        log.info(`Bucket ${bucket.name}: all strata exhausted at ${done.toLocaleString()}/${bucket.target.toLocaleString()} — sleeping ${CAUGHT_UP_SLEEP_MS / 60000} min`)
        await sleep(CAUGHT_UP_SLEEP_MS)
        const last = strata[strata.length - 1]
        if (last.end === null) { last.exhausted = false; continue }
        break
      }
      const stratum = active[turn++ % active.length]
      const state = stratum.state

      let listing: ListingRow[]
      try {
        const raw = await api.getReplayMinId(state.cursor, 'Quick Match')
        listing = (Array.isArray(raw) ? raw : Object.values(raw)) as ListingRow[]
      } catch (err) {
        const msg = String(err)
        if (msg.includes('non-JSON response') || msg.includes('Max calls')) {
          exhaustedUntil = Date.now() + QUOTA_BACKOFF_MS
          log.warn(`Min_id quota/parse issue — sleeping 60 min: ${msg.slice(0, 200)}`)
        } else {
          log.warn(`Listing error at cursor ${state.cursor}: ${msg.slice(0, 200)} — retrying in 60s`)
          await sleep(60_000)
        }
        continue
      }
      if (listing.length === 0) {
        stratum.exhausted = true // head of listing (open stratum) or dead zone
        continue
      }

      // Eligible = valid QM game whose build falls in this bucket's window.
      const eligible = listing.filter(r =>
        (r.game_version ?? '').startsWith(PATCH_PREFIX) &&
        inWindow(r.game_version) &&
        r.valid === 1 && r.parsed === 1 && !r.deleted
      )
      const eligibleIds = eligible.map(r => r.replayID)
      const already = eligibleIds.length > 0
        ? new Set(
            (await db.select({ id: qmGames.replayId }).from(qmGames)
              .where(inArray(qmGames.replayId, eligibleIds))).map(r => r.id)
          )
        : new Set<number>()
      const remaining = bucket.target - done
      const todo = eligible.filter(r => !already.has(r.replayID)).slice(0, Math.max(remaining, 0))
      const pageSkipped = listing.length - todo.length

      const counters = { processed: 0, playerRows: 0, skipped: pageSkipped, errors: 0 }
      let idx = 0
      const retried = new Set<number>()
      const retries: ListingRow[] = []
      const take = (): ListingRow | undefined =>
        idx < todo.length ? todo[idx++] : retries.shift()

      const handleOne = async (row: ListingRow) => {
        try {
          const raw = await api.getReplayData(row.replayID)
          const replay = raw[String(row.replayID)] || raw
          if (!replay || typeof replay !== 'object') {
            counters.skipped++
            return
          }
          const stored = await storeReplayPlayers(db, row.replayID, replay)
          const gameDate = replay.game_date ? new Date(replay.game_date) : null
          await db.insert(qmGames).values({
            replayId: row.replayID,
            region: row.region ?? (Number.isFinite(Number(replay.region)) ? Number(replay.region) : null),
            gameMap: String(replay.game_map ?? row.game_map ?? 'unknown').slice(0, 80),
            gameDate: gameDate && !isNaN(gameDate.getTime()) ? gameDate : null,
            gameLength: Number.isFinite(Number(replay.game_length)) ? Number(replay.game_length) : null,
            gameVersion: String(replay.game_version ?? row.game_version ?? '').slice(0, 40) || null,
            gameType: String(replay.game_type ?? 'Quick Match').slice(0, 40),
            avgMmr: Number.isFinite(Number(row.avg_mmr)) ? Number(row.avg_mmr) : null,
            leagueTier: Number.isFinite(Number(row.league_tier)) ? Number(row.league_tier) : null,
            rank: row.rank ? String(row.rank).slice(0, 20) : null,
          }).onConflictDoNothing()
          counters.processed++
          counters.playerRows += stored.playerRows
        } catch (err) {
          const msg = String(err)
          const isQuota = msg.includes('non-JSON response') || msg.includes('Max calls')
          const isPermanent = /API error 4\d\d/.test(msg) || msg.includes('returned error')
          if (isQuota) {
            exhaustedUntil = Date.now() + QUOTA_BACKOFF_MS
            log.warn(`Replay/Data quota exhausted — benching 60 min`)
            retries.unshift(row)
            retried.delete(row.replayID)
          } else if (isPermanent) {
            counters.skipped++
          } else if (!retried.has(row.replayID)) {
            retried.add(row.replayID)
            retries.push(row)
          } else {
            counters.errors++
            if (counters.errors % 25 === 1) {
              log.warn(`QM fetch error for replay ${row.replayID}: ${msg.slice(0, 300)}`)
            }
          }
        }
      }

      const workerLoop = async () => {
        while (!stopping) {
          if (exhaustedUntil > Date.now()) return
          const row = take()
          if (row === undefined) return
          await handleOne(row)
        }
      }
      await Promise.all(Array.from({ length: WORKERS }, () => workerLoop()))

      if (!stopping && exhaustedUntil <= Date.now()) {
        state.cursor = listing[listing.length - 1].replayID + 1
        if (stratum.end !== null && state.cursor >= stratum.end) stratum.exhausted = true
      }
      state.processedCount += counters.processed
      state.playerRowsUpserted += counters.playerRows
      state.skippedCount += counters.skipped
      state.errorCount += counters.errors
      await saveState(db, stratum.key, state)
      done += counters.processed

      processedThisRun += counters.processed
      const rate = processedThisRun / Math.max((Date.now() - runStart) / 60_000, 0.1)
      log.info(
        `[${bucket.name}/s${stratum.idx}] +${counters.processed} QM games (${counters.playerRows} player rows, ` +
        `${counters.skipped} skipped, ${counters.errors} errors) | cursor=${state.cursor} | ` +
        `${done.toLocaleString()}/${bucket.target.toLocaleString()} | ${rate.toFixed(0)}/min`
      )
    }

    for (const s of strata) await saveState(db, s.key, s.state)
    if (done >= bucket.target) {
      log.info(`Bucket ${bucket.name} COMPLETE (${done.toLocaleString()}). ` +
        (bucket.name === 'recent'
          ? 'Phase A done — bump refetch-players to 140/min (see QUOTA_ALLOCATION_PLAN.md).'
          : ''))
    }
  }

  log.info(stopping ? 'Checkpoint saved. Bye.' : 'All QM era buckets complete. Exiting.')
}

main().catch(err => {
  log.error('Fatal QM ingest worker error', err)
  process.exit(1)
})
