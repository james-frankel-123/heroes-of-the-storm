/**
 * Test fetch: download exactly 20 recent Storm League Replay/Data entries
 * we don't already have. Logs every API call with timestamps, endpoints,
 * response sizes, and cumulative usage for quota tracking.
 *
 * Usage: set -a && source .env && set +a && npx tsx sync/fetch-test.ts
 */
import { HeroesProfileApi } from './api-client'
import { createDb } from './db'
import { replayDraftData, replayFetchQueue, replaySyncState } from '../src/lib/db/schema'
import { eq, sql, notInArray } from 'drizzle-orm'

const TARGET_FETCHES = 20

interface ApiCallLog {
  timestamp: string
  endpoint: string
  params: Record<string, string>
  durationMs: number
  statusCode: number
  responseSize: number
  success: boolean
  error?: string
}

const callLog: ApiCallLog[] = []

function ts(): string {
  return new Date().toISOString()
}

async function timedFetch(
  api: HeroesProfileApi,
  endpoint: string,
  params: Record<string, string>,
): Promise<{ data: any; log: ApiCallLog }> {
  const start = Date.now()
  const entry: ApiCallLog = {
    timestamp: ts(),
    endpoint,
    params,
    durationMs: 0,
    statusCode: 0,
    responseSize: 0,
    success: false,
  }

  try {
    let data: any
    if (endpoint === 'Replay/Max') {
      data = await api.getReplayMax()
      entry.responseSize = String(data).length
    } else if (endpoint === 'Replay/Min_id') {
      data = await api.getReplayMinId(parseInt(params.min_id), params.game_type || 'Storm League')
      entry.responseSize = JSON.stringify(data).length
    } else if (endpoint === 'Replay/Data') {
      data = await api.getReplayData(parseInt(params.replayID))
      entry.responseSize = JSON.stringify(data).length
    } else {
      throw new Error(`Unknown endpoint: ${endpoint}`)
    }

    entry.durationMs = Date.now() - start
    entry.statusCode = 200
    entry.success = true
    callLog.push(entry)
    return { data, log: entry }
  } catch (err: any) {
    entry.durationMs = Date.now() - start
    entry.error = err.message?.slice(0, 200)
    entry.success = false
    callLog.push(entry)
    throw err
  }
}

async function main() {
  const apiKey = process.env.HEROES_PROFILE_API_KEY
  if (!apiKey) { console.error('HEROES_PROFILE_API_KEY required'); process.exit(1) }
  if (!process.env.DATABASE_URL) { console.error('DATABASE_URL required'); process.exit(1) }

  // Use conservative rate limit for test
  const api = new HeroesProfileApi(apiKey, 55, 3)
  const db = createDb()

  console.log(`\n${'='.repeat(70)}`)
  console.log(`  API Fetch Test — ${TARGET_FETCHES} Replay/Data calls`)
  console.log(`  Started: ${ts()}`)
  console.log(`  Key: ...${apiKey.slice(-8)}`)
  console.log(`${'='.repeat(70)}\n`)

  // Step 1: Get current max replay ID
  console.log(`[${ts()}] Step 1: Replay/Max`)
  const { data: maxId } = await timedFetch(api, 'Replay/Max', {})
  console.log(`  Max replay ID: ${maxId}`)
  console.log(`  API calls so far: ${api.getCallCount()}`)

  // Step 2: Get existing replay IDs from our DB (last 50K range)
  const existingRows = await db
    .select({ replayId: replayDraftData.replayId })
    .from(replayDraftData)
    .where(sql`${replayDraftData.replayId} > ${maxId - 50000}`)
  const existingIds = new Set(existingRows.map(r => r.replayId))
  console.log(`  Already have ${existingIds.size} replays in range [${maxId - 50000}, ${maxId}]`)

  // Step 3: Discover recent replay IDs we don't have
  console.log(`\n[${ts()}] Step 2: Replay/Min_id (discover new IDs)`)
  let newIds: number[] = []
  let discoveryCursor = maxId - 10000 // Start 10K back
  let discoveryBatches = 0

  while (newIds.length < TARGET_FETCHES * 2 && discoveryBatches < 5) {
    discoveryBatches++
    const { data: batch, log: batchLog } = await timedFetch(api, 'Replay/Min_id', {
      min_id: String(discoveryCursor),
      game_type: 'Storm League',
    })
    console.log(`  Batch ${discoveryBatches}: cursor=${discoveryCursor}, returned=${Array.isArray(batch) ? batch.length : 0} entries (${batchLog.durationMs}ms, ${batchLog.responseSize} bytes)`)

    if (!Array.isArray(batch) || batch.length === 0) break

    for (const r of batch) {
      if (r.replayID && r.valid === 1 && !r.deleted && !existingIds.has(r.replayID)) {
        newIds.push(r.replayID)
      }
    }

    // Advance cursor past this batch
    const maxBatchId = Math.max(...batch.map((r: any) => r.replayID || 0))
    discoveryCursor = maxBatchId + 1
  }

  console.log(`  Found ${newIds.length} new replay IDs across ${discoveryBatches} discovery batches`)
  console.log(`  API calls so far: ${api.getCallCount()}`)

  // Step 4: Fetch exactly TARGET_FETCHES replays
  const toFetch = newIds.slice(0, TARGET_FETCHES)
  console.log(`\n[${ts()}] Step 3: Replay/Data (fetching ${toFetch.length} replays)`)

  let fetched = 0
  let failed = 0
  let stored = 0

  for (const replayId of toFetch) {
    try {
      const { data: raw, log: fetchLog } = await timedFetch(api, 'Replay/Data', {
        replayID: String(replayId),
      })

      const replayKey = String(replayId)
      const replay = raw[replayKey] || raw

      // Validate draft data exists
      if (!replay || !replay.draft_order || !Array.isArray(replay.draft_order)) {
        console.log(`  [${ts()}] Replay ${replayId}: no draft data (${fetchLog.durationMs}ms, ${fetchLog.responseSize} bytes)`)
        failed++
        continue
      }

      // Extract team compositions
      const team0Heroes: string[] = []
      const team1Heroes: string[] = []
      let winner: number | null = null

      for (const [key, val] of Object.entries(replay)) {
        if (key === 'draft_order' || typeof val !== 'object' || val === null) continue
        const player = val as any
        if (player.hero && player.team !== undefined) {
          if (player.team === 0) team0Heroes.push(player.hero)
          else if (player.team === 1) team1Heroes.push(player.hero)
          if (player.winner === true) winner = player.team
          else if (player.winner === false && winner === null) winner = player.team === 0 ? 1 : 0
        }
      }

      if (team0Heroes.length !== 5 || team1Heroes.length !== 5 || winner === null) {
        console.log(`  [${ts()}] Replay ${replayId}: incomplete (t0=${team0Heroes.length}, t1=${team1Heroes.length}, winner=${winner}) (${fetchLog.durationMs}ms)`)
        failed++
        continue
      }

      // Extract bans
      const team0Bans: string[] = []
      const team1Bans: string[] = []
      for (const d of replay.draft_order) {
        if (d.type === '0' || d.type === 0) {
          if (d.player_slot === 1) team0Bans.push(d.hero)
          else if (d.player_slot === 2) team1Bans.push(d.hero)
        }
      }

      console.log(`  [${ts()}] Replay ${replayId}: ${team0Heroes.join(',')} vs ${team1Heroes.join(',')} → team${winner} wins (${fetchLog.durationMs}ms, ${fetchLog.responseSize} bytes)`)

      fetched++
      stored++
    } catch (err: any) {
      console.log(`  [${ts()}] Replay ${replayId}: ERROR ${err.message?.slice(0, 100)}`)
      failed++
    }
  }

  // Summary
  console.log(`\n${'='.repeat(70)}`)
  console.log(`  RESULTS`)
  console.log(`${'='.repeat(70)}`)
  console.log(`  Completed: ${ts()}`)
  console.log(`  Total API calls: ${api.getCallCount()}`)
  console.log(`  Breakdown:`)
  const endpointCounts: Record<string, number> = {}
  for (const c of callLog) {
    endpointCounts[c.endpoint] = (endpointCounts[c.endpoint] || 0) + 1
  }
  for (const [ep, count] of Object.entries(endpointCounts)) {
    console.log(`    ${ep}: ${count} calls`)
  }
  console.log(`  Replays fetched: ${fetched}/${TARGET_FETCHES}`)
  console.log(`  Replays failed: ${failed}`)
  console.log(`  Replays stored: ${stored}`)
  console.log(`  Average Replay/Data latency: ${
    Math.round(callLog.filter(c => c.endpoint === 'Replay/Data').reduce((s, c) => s + c.durationMs, 0) /
    Math.max(1, callLog.filter(c => c.endpoint === 'Replay/Data').length)
  )}ms`)
  console.log(`  Average Replay/Data response size: ${
    Math.round(callLog.filter(c => c.endpoint === 'Replay/Data').reduce((s, c) => s + c.responseSize, 0) /
    Math.max(1, callLog.filter(c => c.endpoint === 'Replay/Data').length))
  } bytes`)

  // Full call log
  console.log(`\n  Full API Call Log:`)
  console.log(`  ${'─'.repeat(66)}`)
  console.log(`  ${'#'.padEnd(4)} ${'Time'.padEnd(24)} ${'Endpoint'.padEnd(16)} ${'ms'.padEnd(6)} ${'Bytes'.padEnd(8)} ${'OK?'}`)
  console.log(`  ${'─'.repeat(66)}`)
  for (let i = 0; i < callLog.length; i++) {
    const c = callLog[i]
    console.log(`  ${String(i + 1).padEnd(4)} ${c.timestamp.slice(11, 23).padEnd(24)} ${c.endpoint.padEnd(16)} ${String(c.durationMs).padEnd(6)} ${String(c.responseSize).padEnd(8)} ${c.success ? 'Y' : 'N'}`)
  }
  console.log(`${'='.repeat(70)}`)
}

main().catch(err => {
  console.error('Fatal error:', err)
  process.exit(1)
})
