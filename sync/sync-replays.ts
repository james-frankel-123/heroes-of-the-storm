/**
 * Replay data sync for Draft Insights Hyper Pro Max.
 *
 * Two phases:
 * 1. Discovery — scan Replay/Min_id to find Storm League replay IDs, enqueue them.
 *    (1M calls/week/key, returns up to 1000 per call)
 * 2. Fetch — pull full Replay/Data for queued IDs, extract draft order + compositions.
 *    (25K calls/week/key)
 *
 * Both phases are resumable via DB state (replay_sync_state + replay_fetch_queue).
 */
import { sql, eq } from 'drizzle-orm'
import {
  replayDraftData,
  replaySyncState,
  replayFetchQueue,
} from '../src/lib/db/schema'
import { MultiKeyApi } from './api-client'
import { createDb, SyncDb } from './db'
import { log } from './logger'

// ── Helpers ──────────────────────────────────────────────────────────

function leagueTierToSkillTier(tier: number | null): string {
  if (tier === null || tier === undefined) return 'mid'
  if (tier <= 2) return 'low'    // Bronze, Silver
  if (tier <= 4) return 'mid'    // Gold, Platinum
  return 'high'                   // Diamond, Master
}

// ── Phase 1: Discovery ──────────────────────────────────────────────

interface DiscoveryState {
  discoveryCursor: number
  maxKnownId: number
  discoveredCount: number
  fetchedCount: number
  backfillCursor: number
}

async function loadState(db: SyncDb): Promise<DiscoveryState> {
  const rows = await db.select().from(replaySyncState).limit(1)
  if (rows.length > 0) {
    return {
      discoveryCursor: rows[0].discoveryCursor,
      maxKnownId: rows[0].maxKnownId,
      discoveredCount: rows[0].discoveredCount,
      fetchedCount: rows[0].fetchedCount,
      backfillCursor: rows[0].backfillCursor,
    }
  }
  // Initialize
  await db.insert(replaySyncState).values({
    discoveryCursor: 0,
    maxKnownId: 0,
    discoveredCount: 0,
    fetchedCount: 0,
    backfillCursor: 0,
  })
  return { discoveryCursor: 0, maxKnownId: 0, discoveredCount: 0, fetchedCount: 0, backfillCursor: 0 }
}

async function saveState(db: SyncDb, state: DiscoveryState) {
  await db.update(replaySyncState).set({
    discoveryCursor: state.discoveryCursor,
    maxKnownId: state.maxKnownId,
    discoveredCount: state.discoveredCount,
    fetchedCount: state.fetchedCount,
    backfillCursor: state.backfillCursor,
    updatedAt: new Date(),
  })
}

/**
 * Discover replay IDs via Replay/Min_id. Scans backwards from maxId.
 * Enqueues Storm League replays into replay_fetch_queue.
 *
 * @param maxCalls - Max discovery API calls this run
 * @returns Number of new replays enqueued
 */
export async function discoverReplays(
  api: MultiKeyApi,
  db: SyncDb,
  maxCalls = 500,
): Promise<number> {
  const state = await loadState(db)

  // Refresh max ID
  const maxId = await api.next().getReplayMax()
  state.maxKnownId = Math.max(state.maxKnownId, maxId)

  // If cursor is 0, start scanning from a reasonable offset below max
  if (state.discoveryCursor === 0) {
    // Start 500K below max — roughly a few weeks of replays
    state.discoveryCursor = Math.max(1, maxId - 500_000)
    log.info(`Initializing discovery cursor to ${state.discoveryCursor} (maxId=${maxId})`)
  }

  let totalEnqueued = 0
  let callsMade = 0

  log.info(`Discovery: cursor=${state.discoveryCursor}, max=${maxId}, gap=${maxId - state.discoveryCursor}`)

  while (state.discoveryCursor < maxId && callsMade < maxCalls) {
    try {
      const batch = await api.next().getReplayMinId(state.discoveryCursor)
      callsMade++

      if (!Array.isArray(batch) || batch.length === 0) {
        // No more replays in this range, jump ahead but never past maxId
        state.discoveryCursor = Math.min(state.discoveryCursor + 1000, maxId)
        if (state.discoveryCursor >= maxId) break  // caught up
        continue
      }

      // Filter for valid Storm League replays on current major patch
      const MAJOR_PATCH = '2.55'
      const valid = batch.filter((r: any) =>
        r.game_type === 'Storm League' &&
        r.valid === 1 &&
        !r.deleted &&
        r.replayID &&
        (r.game_version || '').startsWith(MAJOR_PATCH)
      )

      if (valid.length > 0) {
        // Batch insert into queue, ignore conflicts (already queued)
        const queueRows = valid.map((r: any) => ({
          replayId: r.replayID,
          gameMap: r.game_map || null,
          leagueTier: r.league_tier || null,
          avgMmr: r.avg_mmr || null,
          gameVersion: r.game_version || null,
          fetched: false,
        }))

        for (let i = 0; i < queueRows.length; i += 100) {
          const chunk = queueRows.slice(i, i + 100)
          await db.insert(replayFetchQueue)
            .values(chunk)
            .onConflictDoNothing()
        }

        totalEnqueued += valid.length
        state.discoveredCount += valid.length
      }

      // Advance cursor past the last ID in the batch
      const lastId = batch[batch.length - 1].replayID
      state.discoveryCursor = lastId + 1

      if (callsMade % 50 === 0) {
        log.info(`  Discovery progress: ${callsMade} calls, ${totalEnqueued} enqueued, cursor=${state.discoveryCursor}`)
        await saveState(db, state)
      }
    } catch (err) {
      log.warn(`Discovery error at cursor=${state.discoveryCursor}: ${err}`)
      state.discoveryCursor += 100 // skip past problem area
    }
  }

  await saveState(db, state)
  log.info(`Discovery complete: ${callsMade} calls, ${totalEnqueued} new replays enqueued`)
  return totalEnqueued
}

// ── Phase 1b: Backfill discovery (scan backwards) ──────────────────

/**
 * Discover older replay IDs by scanning backwards from our oldest known replay.
 * Uses Replay/Min_id to find Storm League replays on current major patch.
 *
 * @param maxCalls - Max discovery API calls this run
 * @returns Number of new replays enqueued
 */
export async function discoverBackfill(
  api: MultiKeyApi,
  db: SyncDb,
  maxCalls = 500,
): Promise<number> {
  const state = await loadState(db)

  // Initialize backfill cursor to our oldest known replay ID
  if (state.backfillCursor === 0) {
    const oldest = await db.select({ minId: sql<number>`MIN(replay_id)` }).from(replayDraftData)
    const minId = oldest[0]?.minId ?? 60769081
    state.backfillCursor = minId
    log.info(`Backfill: initializing cursor to ${state.backfillCursor} (oldest known replay)`)
  }

  // Stop if we've gone below a reasonable floor (patch 2.55 started ~2021)
  const BACKFILL_FLOOR = 40_000_000
  if (state.backfillCursor <= BACKFILL_FLOOR) {
    log.info(`Backfill: reached floor (${BACKFILL_FLOOR}), nothing more to discover`)
    return 0
  }

  let totalEnqueued = 0
  let callsMade = 0
  const MAJOR_PATCH = '2.55'

  log.info(`Backfill: cursor=${state.backfillCursor}, floor=${BACKFILL_FLOOR}`)

  while (state.backfillCursor > BACKFILL_FLOOR && callsMade < maxCalls) {
    // Scan backwards: query a range ending at our cursor
    const queryStart = Math.max(BACKFILL_FLOOR, state.backfillCursor - 1000)
    try {
      const batch = await api.next().getReplayMinId(queryStart)
      callsMade++

      if (!Array.isArray(batch) || batch.length === 0) {
        state.backfillCursor = queryStart
        continue
      }

      // Filter for valid SL replays on current major patch
      const valid = batch.filter((r: any) =>
        r.game_type === 'Storm League' &&
        r.valid === 1 &&
        !r.deleted &&
        r.replayID &&
        r.replayID < state.backfillCursor &&
        (r.game_version || '').startsWith(MAJOR_PATCH)
      )

      if (valid.length > 0) {
        const queueRows = valid.map((r: any) => ({
          replayId: r.replayID,
          gameMap: r.game_map || null,
          leagueTier: r.league_tier || null,
          avgMmr: r.avg_mmr || null,
          gameVersion: r.game_version || null,
          fetched: false,
        }))

        for (let i = 0; i < queueRows.length; i += 100) {
          const chunk = queueRows.slice(i, i + 100)
          await db.insert(replayFetchQueue)
            .values(chunk)
            .onConflictDoNothing()
        }

        totalEnqueued += valid.length
        state.discoveredCount += valid.length
      }

      // Move cursor backwards
      state.backfillCursor = queryStart

      if (callsMade % 50 === 0) {
        log.info(`  Backfill progress: ${callsMade} calls, ${totalEnqueued} enqueued, cursor=${state.backfillCursor}`)
        await saveState(db, state)
      }
    } catch (err) {
      log.warn(`Backfill error at cursor=${state.backfillCursor}: ${err}`)
      state.backfillCursor -= 1000
    }
  }

  await saveState(db, state)
  log.info(`Backfill complete: ${callsMade} calls, ${totalEnqueued} new replays enqueued`)
  return totalEnqueued
}

// ── Phase 2: Fetch full replay data ─────────────────────────────────

/**
 * Fetch full Replay/Data for queued replays and extract draft data.
 *
 * @param maxCalls - Max Replay/Data API calls this run
 * @returns Number of replays successfully fetched
 */
export async function fetchReplayData(
  api: MultiKeyApi,
  db: SyncDb,
  maxCalls = 500,
): Promise<number> {
  // Get unfetched IDs from queue with metadata, randomized for tier spread
  const queue = await db
    .select({
      replayId: replayFetchQueue.replayId,
      leagueTier: replayFetchQueue.leagueTier,
      avgMmr: replayFetchQueue.avgMmr,
    })
    .from(replayFetchQueue)
    .where(eq(replayFetchQueue.fetched, false))
    .orderBy(sql`random()`)
    .limit(maxCalls)

  if (queue.length === 0) {
    log.info('Fetch: no unfetched replays in queue')
    return 0
  }

  log.info(`Fetch: processing ${queue.length} replays`)

  let fetched = 0
  let failed = 0
  let consecutiveQuotaErrors = 0
  const numKeys = api.keyCount()

  for (const queueItem of queue) {
    const replayId = queueItem.replayId
    try {
      const raw = await api.next().getReplayData(replayId)
      const replayKey = String(replayId)
      const replay = raw[replayKey] || raw

      if (!replay || !replay.draft_order || !Array.isArray(replay.draft_order)) {
        // No draft data — mark as fetched but don't store
        await db.update(replayFetchQueue)
          .set({ fetched: true })
          .where(eq(replayFetchQueue.replayId, replayId))
        failed++
        continue
      }

      // Extract team compositions and talents from player data
      const team0Heroes: string[] = []
      const team1Heroes: string[] = []
      const team0Talents: { hero: string; talents: Record<string, string> }[] = []
      const team1Talents: { hero: string; talents: Record<string, string> }[] = []
      let winner: number | null = null

      for (const [key, val] of Object.entries(replay)) {
        if (key === 'draft_order' || typeof val !== 'object' || val === null) continue
        const player = val as any
        if (player.hero && player.team !== undefined) {
          const talents = (player.talents && typeof player.talents === 'object') ? player.talents : {}
          if (player.team === 0) {
            team0Heroes.push(player.hero)
            team0Talents.push({ hero: player.hero, talents })
          } else if (player.team === 1) {
            team1Heroes.push(player.hero)
            team1Talents.push({ hero: player.hero, talents })
          }
          if (player.winner === true) winner = player.team
          else if (player.winner === false && winner === null) winner = player.team === 0 ? 1 : 0
        }
      }

      // Validate: need 5 heroes per team and a winner
      if (team0Heroes.length !== 5 || team1Heroes.length !== 5 || winner === null) {
        await db.update(replayFetchQueue)
          .set({ fetched: true })
          .where(eq(replayFetchQueue.replayId, replayId))
        failed++
        continue
      }

      // Extract bans from draft order
      const team0Bans: string[] = []
      const team1Bans: string[] = []
      for (const d of replay.draft_order) {
        if (d.type === '0' || d.type === 0) {
          // Ban — player_slot 1 = team that made the ban
          // Slots 0-4 = team 0, slots 5-9 = team 1
          // But for bans, player_slot is 1 or 2 indicating ban order team
          if (d.player_slot === 1) team0Bans.push(d.hero)
          else if (d.player_slot === 2) team1Bans.push(d.hero)
        }
      }

      // Use queue metadata for tier/mmr (Replay/Data doesn't include these)
      const leagueTier = queueItem.leagueTier
      const avgMmr = queueItem.avgMmr
      const skillTier = leagueTierToSkillTier(leagueTier)

      const talents = { team0: team0Talents, team1: team1Talents }

      await db.insert(replayDraftData)
        .values({
          replayId,
          region: replay.region || 0,
          gameMap: replay.game_map,
          gameDate: new Date(replay.game_date),
          gameLength: replay.game_length || null,
          gameVersion: replay.game_version || '',
          avgMmr: avgMmr || null,
          leagueTier: leagueTier || null,
          draftOrder: replay.draft_order,
          team0Heroes,
          team1Heroes,
          team0Bans,
          team1Bans,
          winner,
          skillTier,
          talents,
        })
        // If the row already exists (older fetch without talents), update with talents
        .onConflictDoUpdate({
          target: replayDraftData.replayId,
          set: { talents },
        })

      await db.update(replayFetchQueue)
        .set({ fetched: true })
        .where(eq(replayFetchQueue.replayId, replayId))

      fetched++
      consecutiveQuotaErrors = 0

      if (fetched % 100 === 0) {
        log.info(`  Fetch progress: ${fetched} succeeded, ${failed} skipped`)
      }
    } catch (err) {
      failed++
      const errMsg = String(err)
      const isQuota = errMsg.includes('non-JSON response') || errMsg.includes('Max calls')
      const isTransient =
        isQuota ||
        errMsg.includes('Rate limited') ||
        errMsg.includes('Server error') ||
        errMsg.includes('Network error')
      if (!isTransient) {
        // Permanent: bad replay ID / not Storm League / corrupt draft. Mark as
        // fetched so we don't retry forever.
        await db.update(replayFetchQueue)
          .set({ fetched: true })
          .where(eq(replayFetchQueue.replayId, replayId))
          .catch(() => {})
        consecutiveQuotaErrors = 0
      } else if (isQuota) {
        consecutiveQuotaErrors++
      }
      if (failed % 50 === 0) {
        log.warn(`Fetch error for replay ${replayId}: ${err}`)
      }
      // If every key has returned a quota error back-to-back, all keys are
      // exhausted — stop burning the queue and let the cycle end.
      if (consecutiveQuotaErrors >= numKeys) {
        log.warn(`All ${numKeys} API key(s) quota-exhausted for Replay/Data; aborting fetch cycle (${failed} failures)`)
        break
      }
    }
  }

  // Update state
  const state = await loadState(db)
  state.fetchedCount += fetched
  await saveState(db, state)

  log.info(`Fetch complete: ${fetched} succeeded, ${failed} skipped/failed`)
  return fetched
}

// ── Stats ────────────────────────────────────────────────────────────

/**
 * Re-queue replays that are missing talent data.
 * Only runs when the main fetch queue is empty (pending backlog cleared).
 * Finds replay IDs in replay_draft_data where talents is null or empty,
 * and inserts them back into the fetch queue (the normal fetch path
 * will re-fetch and update with talent data via onConflictDoUpdate).
 */
export async function backfillTalents(
  db: SyncDb,
  batchSize = 1000,
): Promise<number> {
  // Only run if the main queue is empty
  const [pendingCount] = await db.execute(
    sql`SELECT count(*) as c FROM replay_fetch_queue WHERE fetched = false`
  ).then(r => r.rows)

  if (Number(pendingCount.c) > 0) {
    return 0 // Main queue still has work — don't compete
  }

  // Find replays missing talents that aren't already re-queued
  const missing = await db.execute(
    sql`SELECT d.replay_id, d.league_tier, d.avg_mmr
        FROM replay_draft_data d
        LEFT JOIN replay_fetch_queue q ON d.replay_id = q.replay_id AND q.fetched = false
        WHERE (d.talents IS NULL OR d.talents = '{}' OR d.talents = 'null' OR d.talents::text = '{"team0":[],"team1":[]}')
          AND q.replay_id IS NULL
        LIMIT ${batchSize}`
  ).then(r => r.rows)

  if (missing.length === 0) {
    return 0
  }

  // Re-queue them — mark as unfetched so the normal fetch loop picks them up
  for (const row of missing) {
    await db.insert(replayFetchQueue)
      .values({
        replayId: Number(row.replay_id),
        leagueTier: row.league_tier != null ? Number(row.league_tier) : null,
        avgMmr: row.avg_mmr != null ? Number(row.avg_mmr) : null,
        fetched: false,
      })
      .onConflictDoUpdate({
        target: replayFetchQueue.replayId,
        set: { fetched: false },
      })
  }

  log.info(`Talent backfill: re-queued ${missing.length} replays missing talent data`)
  return missing.length
}

export async function getReplayStats(db: SyncDb) {
  const [draftCount] = await db.execute(
    sql`SELECT count(*) as c FROM replay_draft_data`
  ).then(r => r.rows)
  const [queueCount] = await db.execute(
    sql`SELECT count(*) as c FROM replay_fetch_queue WHERE fetched = false`
  ).then(r => r.rows)
  const [totalQueued] = await db.execute(
    sql`SELECT count(*) as c FROM replay_fetch_queue`
  ).then(r => r.rows)
  const state = await loadState(db)

  return {
    draftDataRows: Number(draftCount.c),
    pendingInQueue: Number(queueCount.c),
    totalQueued: Number(totalQueued.c),
    discoveryCursor: state.discoveryCursor,
    maxKnownId: state.maxKnownId,
    gapRemaining: state.maxKnownId - state.discoveryCursor,
  }
}
