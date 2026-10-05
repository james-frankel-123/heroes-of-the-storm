/**
 * Heroes Profile privacy feed (API terms s5, from 2026-10-01).
 *
 * Polls /players/privacy/changes and, within 24h of a player going private:
 *  - site data: deletes their player_match_history / player_hero_stats /
 *    player_hero_map_stats rows and blanks sync_log;
 *  - research data (replay_players), per PRIVACY_RESEARCH_MODE:
 *      retain (default): replay_players and player_history_marks untouched.
 *        Research store is never displayed;
 *        pending Heroes Profile's written approval (asked 2026-09-30).
 *      pseudonymize: every row of each matching (region, blizz_id) gets a
 *        fresh random negative blizz_id and battletag 'private'. No mapping
 *        is kept, so games/heroes/MMRs/outcomes and per-player panel
 *        structure survive but the identity link does not. Irreversible.
 *
 * Crash-safe order: each page is recorded in player_privacy BEFORE the feed
 * cursor advances; purges run from player_privacy rows with applied_at NULL.
 * Research file caches (training/personalization/cache) are NOT scrubbed here.
 *
 * Matching is by (battletag, region) because the feed carries no blizz_id: a
 * player renamed since we stored their games is only caught if we have seen
 * the new name at least once.
 *
 * Inert while the v1 account serves fixture data (never apply example rows).
 *
 * Site tables on Neon pick up the deletes when sync/publish-site.ts runs
 * right after this job (same cron line).
 *
 * Usage:
 *   npx tsx sync/privacy-sync.ts            # poll + apply
 *   npx tsx sync/privacy-sync.ts --dry-run  # poll + report, write nothing
 *   npx tsx sync/privacy-sync.ts --status
 */
import { randomInt } from 'crypto'
import { sql } from 'drizzle-orm'
import { HeroesProfileApiV2 } from './api-client-v2'
import { createDb, SyncDb } from './db'
import { isHpAccessPaused } from './hp-errors'
import { log } from './logger'

type Mode = 'pseudonymize' | 'retain'

interface Change {
  battletag: string
  region: number
  state: 'private' | 'public'
  changed_at: string
}

interface FeedPage {
  changes: Change[]
  next_since: string
  next_after_id: number
  has_more: boolean
}

const PAGE_LIMIT = 5000
const MAX_PAGES = 200 // first sync returns every account that ever changed state

function researchMode(): Mode {
  const m = process.env.PRIVACY_RESEARCH_MODE ?? 'retain'
  if (m !== 'pseudonymize' && m !== 'retain') {
    throw new Error(`PRIVACY_RESEARCH_MODE must be pseudonymize|retain, got ${m}`)
  }
  return m
}

async function ensureTables(db: SyncDb): Promise<void> {
  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS player_privacy (
      battletag varchar(100) NOT NULL,
      region integer NOT NULL,
      state varchar(10) NOT NULL,
      changed_at timestamptz NOT NULL,
      applied_at timestamptz,
      applied_mode varchar(20),
      research_ids integer,
      research_rows integer,
      PRIMARY KEY (battletag, region)
    )`)
  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS privacy_feed_state (
      id integer PRIMARY KEY DEFAULT 1 CHECK (id = 1),
      next_since text,
      next_after_id bigint,
      last_polled_at timestamptz
    )`)
}

async function loadCursor(db: SyncDb): Promise<{ since?: string; afterId?: number }> {
  const [row] = (await db.execute(sql`SELECT next_since, next_after_id FROM privacy_feed_state WHERE id = 1`)).rows as any[]
  if (!row?.next_since) return {}
  return { since: row.next_since, afterId: row.next_after_id == null ? undefined : Number(row.next_after_id) }
}

async function recordPage(db: SyncDb, page: FeedPage): Promise<void> {
  // Latest event per player wins; a newer event resets applied_at.
  const latest = new Map<string, Change>()
  for (const c of page.changes) {
    const k = `${c.battletag}\u0000${c.region}`
    const prev = latest.get(k)
    if (!prev || prev.changed_at < c.changed_at) latest.set(k, c)
  }
  const upserts = [...latest.values()].map(c => sql`
    INSERT INTO player_privacy (battletag, region, state, changed_at)
    VALUES (${c.battletag}, ${c.region}, ${c.state}, ${c.changed_at})
    ON CONFLICT (battletag, region) DO UPDATE
      SET state = excluded.state, changed_at = excluded.changed_at,
          applied_at = NULL, applied_mode = NULL, research_ids = NULL, research_rows = NULL
      WHERE excluded.changed_at > player_privacy.changed_at`)
  const cursor = sql`
    INSERT INTO privacy_feed_state (id, next_since, next_after_id, last_polled_at)
    VALUES (1, ${page.next_since}, ${page.next_after_id}, now())
    ON CONFLICT (id) DO UPDATE SET next_since = excluded.next_since,
      next_after_id = excluded.next_after_id, last_polled_at = excluded.last_polled_at`
  // One transaction: changes and cursor land together.
  await db.transaction(async tx => {
    for (const q of [...upserts, cursor]) await tx.execute(q)
  })
}

async function purge(db: SyncDb, battletag: string, region: number, mode: Mode) {
  await db.transaction(async tx => {
    await tx.execute(sql`DELETE FROM player_match_history WHERE battletag = ${battletag}`)
    await tx.execute(sql`DELETE FROM player_hero_stats WHERE battletag = ${battletag}`)
    await tx.execute(sql`DELETE FROM player_hero_map_stats WHERE battletag = ${battletag}`)
    await tx.execute(sql`UPDATE sync_log SET battletag = NULL WHERE battletag = ${battletag}`)
  })

  let researchIds = 0
  let researchRows = 0
  if (mode === 'pseudonymize') {
    // Research panel bookkeeping (enqueue-player-histories) goes with the rows.
    await db.execute(sql`DELETE FROM player_history_marks WHERE battletag = ${battletag}`)
    const ids = (await db.execute(sql`
      SELECT DISTINCT blizz_id FROM replay_players
      WHERE battletag = ${battletag} AND region = ${region}`)).rows.map((r: any) => Number(r.blizz_id))
    for (const id of ids) {
      if (id < 0) continue // already a surrogate
      // Random, not derived from the real id, so it cannot be reversed.
      const surrogate = -randomInt(1, 2 ** 47)
      const res = await db.execute(sql`
        UPDATE replay_players SET blizz_id = ${surrogate}, battletag = 'private'
        WHERE blizz_id = ${id} AND region = ${region}`)
      researchIds++
      researchRows += Number((res as any).rowCount ?? 0)
    }
  }

  await db.execute(sql`
    UPDATE player_privacy SET applied_at = now(), applied_mode = ${mode},
      research_ids = ${researchIds}, research_rows = ${researchRows}
    WHERE battletag = ${battletag} AND region = ${region}`)
  return { researchIds, researchRows }
}

async function status(db: SyncDb): Promise<void> {
  await ensureTables(db)
  const [s] = (await db.execute(sql`SELECT * FROM privacy_feed_state WHERE id = 1`)).rows as any[]
  const [c] = (await db.execute(sql`
    SELECT count(*) FILTER (WHERE state = 'private') AS private,
           count(*) FILTER (WHERE state = 'private' AND applied_at IS NULL) AS pending,
           coalesce(sum(research_rows), 0) AS research_rows
    FROM player_privacy`)).rows as any[]
  console.log(`last polled: ${s?.last_polled_at ?? 'never'} | cursor: ${s?.next_since ?? '-'} / ${s?.next_after_id ?? '-'}`)
  console.log(`private players: ${c.private} (pending purge: ${c.pending}) | research rows pseudonymized: ${c.research_rows}`)
}

async function main(): Promise<void> {
  const db = createDb()
  if (process.argv.includes('--status')) return status(db)
  const dryRun = process.argv.includes('--dry-run')
  const mode = researchMode()

  const key = process.env.HEROES_PROFILE_V2_API_KEY
  if (!key) throw new Error('HEROES_PROFILE_V2_API_KEY not set')
  const api = new HeroesProfileApiV2(key, 30, 3)

  await ensureTables(db)
  let { since, afterId } = await loadCursor(db)
  log.info(`privacy-sync: mode=${mode}${dryRun ? ' [DRY RUN]' : ''}, since=${since ?? '(first sync)'}`)

  let pages = 0
  let seen = 0
  for (;;) {
    const page = await api.fetch<FeedPage>('players/privacy/changes', {
      since,
      after_id: afterId === undefined ? undefined : String(afterId),
      limit: String(PAGE_LIMIT),
    })
    pages++
    if (api.lastDataSource === 'fixture') {
      log.info(`v1 is serving FIXTURE data (${page.changes?.length ?? 0} example changes); storing nothing`)
      return
    }
    seen += page.changes.length
    if (dryRun) {
      const priv = page.changes.filter(c => c.state === 'private').length
      log.info(`page ${pages}: ${page.changes.length} changes (${priv} private), has_more=${page.has_more}`)
    } else {
      await recordPage(db, page)
    }
    since = page.next_since
    afterId = page.next_after_id
    if (!page.has_more) break
    if (pages >= MAX_PAGES) {
      log.warn(`stopping after ${MAX_PAGES} pages; next run continues from the saved cursor`)
      break
    }
  }
  log.info(`feed: ${pages} page(s), ${seen} change(s)`)
  if (dryRun) return

  const pending = (await db.execute(sql`
    SELECT battletag, region FROM player_privacy
    WHERE state = 'private' AND applied_at IS NULL ORDER BY changed_at`)).rows as any[]
  let rows = 0
  for (const p of pending) {
    const r = await purge(db, p.battletag, Number(p.region), mode)
    rows += r.researchRows
  }
  // Public again: nothing to restore; mark handled.
  await db.execute(sql`
    UPDATE player_privacy SET applied_at = now(), applied_mode = 'public'
    WHERE state = 'public' AND applied_at IS NULL`)
  log.info(`purged ${pending.length} private player(s); research rows ${mode === 'retain' ? 'retained' : `pseudonymized: ${rows}`}`)
}

main().then(() => process.exit(0)).catch(err => {
  if (isHpAccessPaused(err)) log.error(err.message)
  else log.error(`privacy-sync failed: ${err?.stack ?? err}`)
  process.exit(1)
})
