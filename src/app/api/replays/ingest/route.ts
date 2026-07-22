import { NextResponse } from 'next/server'
import { timingSafeEqual } from 'crypto'
import { db } from '@/lib/db'
import { playerMatchHistory } from '@/lib/db/schema'
import { recomputeDerivedStats } from '@/lib/player-stats/derive'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const MAX_RECORDS = 10_000
const INSERT_BATCH = 500

interface RecordInput {
  battletag?: unknown
  replayId?: unknown
  hero?: unknown
  map?: unknown
  win?: unknown
  gameDate?: unknown // ISO string or epoch ms
  gameLength?: unknown
  kills?: unknown
  deaths?: unknown
  assists?: unknown
  heroDamage?: unknown
  siegeDamage?: unknown
  healing?: unknown
  experience?: unknown
  talents?: unknown
  gameMode?: unknown
  rank?: unknown
}

type MatchRow = typeof playerMatchHistory.$inferInsert

/** Constant-time bearer/token check against REPLAY_INGEST_TOKEN. */
function authorized(req: Request): boolean {
  const expected = process.env.REPLAY_INGEST_TOKEN
  if (!expected) return false // fail closed — no token configured means no writes
  const header = req.headers.get('x-ingest-token') ?? ''
  const a = Buffer.from(header)
  const b = Buffer.from(expected)
  return a.length === b.length && timingSafeEqual(a, b)
}

function str(v: unknown, max: number): string | null {
  if (typeof v !== 'string') return null
  const t = v.trim()
  return t.length > 0 && t.length <= max ? t : null
}

function intOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? Math.trunc(v) : null
}

function parseRecord(r: RecordInput, i: number): MatchRow | string {
  const battletag = str(r.battletag, 100)
  if (!battletag) return `record ${i}: battletag required (1-100 chars)`
  const replayId = str(r.replayId, 100)
  if (!replayId) return `record ${i}: replayId required (1-100 chars)`
  const hero = str(r.hero, 80)
  if (!hero) return `record ${i}: hero required (1-80 chars)`
  const map = str(r.map, 80)
  if (!map) return `record ${i}: map required (1-80 chars)`
  if (typeof r.win !== 'boolean') return `record ${i}: win must be a boolean`

  const gameDate = r.gameDate instanceof Object ? null : new Date(r.gameDate as string | number)
  if (!gameDate || Number.isNaN(gameDate.getTime())) return `record ${i}: gameDate must be an ISO date or epoch ms`

  return {
    battletag,
    replayId,
    hero,
    map,
    win: r.win,
    gameDate,
    gameLength: intOrNull(r.gameLength),
    kills: intOrNull(r.kills) ?? 0,
    deaths: intOrNull(r.deaths) ?? 0,
    assists: intOrNull(r.assists) ?? 0,
    heroDamage: intOrNull(r.heroDamage) ?? 0,
    siegeDamage: intOrNull(r.siegeDamage) ?? 0,
    healing: intOrNull(r.healing) ?? 0,
    experience: intOrNull(r.experience) ?? 0,
    talents: r.talents ?? null,
    gameMode: str(r.gameMode, 40),
    rank: str(r.rank, 40),
  }
}

/**
 * POST /api/replays/ingest
 * Auth: header `x-ingest-token: <REPLAY_INGEST_TOKEN>`.
 * Body: { records: MatchRow[] } — per-(battletag, replay) match records parsed
 * from local .StormReplay files. Upserts into player_match_history (dedup on
 * battletag+replayId) and recomputes derived per-hero MAWP for each battletag.
 */
export async function POST(req: Request) {
  if (!authorized(req)) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 })
  }

  let body: { records?: unknown }
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: 'invalid JSON body' }, { status: 400 })
  }

  if (!Array.isArray(body.records)) {
    return NextResponse.json({ error: 'records must be an array' }, { status: 400 })
  }
  if (body.records.length === 0) {
    return NextResponse.json({ ok: true, inserted: 0, battletags: 0, recomputed: [] })
  }
  if (body.records.length > MAX_RECORDS) {
    return NextResponse.json({ error: `too many records (max ${MAX_RECORDS} per request)` }, { status: 413 })
  }

  const rows: MatchRow[] = []
  for (let i = 0; i < body.records.length; i++) {
    const parsed = parseRecord(body.records[i] as RecordInput, i)
    if (typeof parsed === 'string') {
      return NextResponse.json({ error: parsed }, { status: 400 })
    }
    rows.push(parsed)
  }

  // Insert match rows, skipping any (battletag, replayId) already stored.
  let inserted = 0
  for (let i = 0; i < rows.length; i += INSERT_BATCH) {
    const batch = rows.slice(i, i + INSERT_BATCH)
    const res = await db
      .insert(playerMatchHistory)
      .values(batch)
      .onConflictDoNothing({ target: [playerMatchHistory.battletag, playerMatchHistory.replayId] })
      .returning({ id: playerMatchHistory.id })
    inserted += res.length
  }

  // Recompute derived per-hero stats (incl. MAWP) for each affected battletag.
  const battletags = Array.from(new Set(rows.map((r) => r.battletag)))
  const recomputed: Array<{ battletag: string; heroes: number }> = []
  for (const bt of battletags) {
    const { heroes } = await recomputeDerivedStats(bt)
    recomputed.push({ battletag: bt, heroes })
  }

  return NextResponse.json({ ok: true, inserted, battletags: battletags.length, recomputed })
}
