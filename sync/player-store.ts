/**
 * Shared per-player extraction + storage for Replay/Data payloads.
 *
 * Policy (Max, 2026-07-09): "save everything we can this time."
 * - Every player entry (keyed by battletag) → one replay_players row with all
 *   known fields extracted, the full `scores` payload as `scoreboard`, and any
 *   leftover per-player fields in `raw_extras`.
 * - Replay-level fields that replay_draft_data does not store (fingerprint,
 *   game_type, experience_breakdown, ...) → one replay_extras row.
 *
 * Used by both the daemon's new-game fetch path (sync/sync-replays.ts) and the
 * Phase 1 refetch worker (sync/refetch-players.ts).
 */
import { sql } from 'drizzle-orm'
import { replayPlayers, replayExtras } from '../src/lib/db/schema'
import { SyncDb } from './db'

// Per-player fields extracted into dedicated columns; everything else lands in raw_extras.
const EXTRACTED_PLAYER_FIELDS = new Set([
  'blizz_id', 'hero', 'team', 'winner', 'party',
  'player_mmr', 'hero_mmr', 'role_mmr', 'hero_level',
  'talents', 'scores',
])

// Replay-level fields already stored on replay_draft_data (or handled elsewhere).
const STORED_REPLAY_FIELDS = new Set([
  'draft_order', 'game_map', 'game_date', 'game_length', 'game_version', 'region',
])

function toNum(v: unknown): number | null {
  if (v === null || v === undefined || v === '') return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

/** True for the battletag-keyed player entries in a Replay/Data payload. */
function isPlayerEntry(val: unknown): val is Record<string, any> {
  return (
    typeof val === 'object' && val !== null && !Array.isArray(val) &&
    (val as any).hero !== undefined && (val as any).team !== undefined
  )
}

export interface PlayerStoreResult {
  playerRows: number
  extrasStored: boolean
}

/**
 * Extract and upsert all per-player rows + the replay-level extras row for one
 * Replay/Data payload. Idempotent (upserts). Returns row counts.
 *
 * @param replay - The unwrapped replay object (payload[replayId] || payload)
 */
export async function storeReplayPlayers(
  db: SyncDb,
  replayId: number,
  replay: Record<string, any>,
): Promise<PlayerStoreResult> {
  const region = toNum(replay.region)
  const now = new Date()

  // ── Per-player rows ────────────────────────────────────────────────
  const seen = new Set<number>()
  const rows: (typeof replayPlayers.$inferInsert)[] = []
  const extras: Record<string, any> = {}

  for (const [key, val] of Object.entries(replay)) {
    if (!isPlayerEntry(val)) {
      // Replay-level field — keep anything we don't already store.
      if (!STORED_REPLAY_FIELDS.has(key)) extras[key] = val
      continue
    }
    const p = val as Record<string, any>
    const blizzId = toNum(p.blizz_id)
    if (blizzId === null) continue // can't key the row; probe showed this never happens
    if (seen.has(blizzId)) continue // guard: ON CONFLICT can't touch a row twice
    seen.add(blizzId)

    const rawExtras: Record<string, any> = {}
    for (const [pk, pv] of Object.entries(p)) {
      if (!EXTRACTED_PLAYER_FIELDS.has(pk)) rawExtras[pk] = pv
    }

    rows.push({
      replayId,
      blizzId,
      battletag: String(key).slice(0, 100),
      hero: String(p.hero).slice(0, 80),
      team: toNum(p.team) ?? 0,
      winner: p.winner === true || p.winner === 1 || p.winner === '1',
      party: toNum(p.party),
      playerMmr: toNum(p.player_mmr),
      heroMmr: toNum(p.hero_mmr),
      roleMmr: toNum(p.role_mmr),
      heroLevel: toNum(p.hero_level),
      talents: (p.talents && typeof p.talents === 'object') ? p.talents : null,
      scoreboard: (p.scores && typeof p.scores === 'object') ? p.scores : null,
      rawExtras: Object.keys(rawExtras).length > 0 ? rawExtras : null,
      region,
      fetchedAt: now,
    })
  }

  if (rows.length > 0) {
    await db.insert(replayPlayers)
      .values(rows)
      .onConflictDoUpdate({
        target: [replayPlayers.replayId, replayPlayers.blizzId],
        set: {
          battletag: sql`excluded.battletag`,
          hero: sql`excluded.hero`,
          team: sql`excluded.team`,
          winner: sql`excluded.winner`,
          party: sql`excluded.party`,
          playerMmr: sql`excluded.player_mmr`,
          heroMmr: sql`excluded.hero_mmr`,
          roleMmr: sql`excluded.role_mmr`,
          heroLevel: sql`excluded.hero_level`,
          talents: sql`excluded.talents`,
          scoreboard: sql`excluded.scoreboard`,
          rawExtras: sql`excluded.raw_extras`,
          region: sql`excluded.region`,
          fetchedAt: sql`excluded.fetched_at`,
        },
      })
  }

  // ── Replay-level extras ────────────────────────────────────────────
  let extrasStored = false
  if (Object.keys(extras).length > 0) {
    await db.insert(replayExtras)
      .values({ replayId, extras, fetchedAt: now })
      .onConflictDoUpdate({
        target: replayExtras.replayId,
        set: { extras, fetchedAt: now },
      })
    extrasStored = true
  }

  return { playerRows: rows.length, extrasStored }
}
