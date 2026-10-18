/**
 * Heroes Profile v1 calls made by the site for player lookups, behind a quota guard.
 * Server only. Design and budgets: docs/personalization-site-design.md.
 *
 * Every call is charged to one HP allowance pool. The guard refuses a call when
 *  - the site's own calls in that pool over the last 7 days reached its weekly budget,
 *  - or the last seen x-hp-quota-remaining for the pool is below the pool's floor
 *    (the floor protects the sync jobs that share the key),
 *  - or the pool is switched off (budget 0).
 * Budgets are env-overridable so Max can move them without a deploy of new code.
 */
import { sql } from 'drizzle-orm'
import { db } from '@/lib/db'

const BASE_URL = 'https://www.heroesprofile.com/api/external/v1'
const TIMEOUT_MS = 12_000

export type Pool = 'player' | 'matches' | 'mmr'

interface PoolPolicy {
  weekly: number
  hourly: number
  floor: number
}

function envInt(name: string, fallback: number): number {
  const v = Number(process.env[name])
  return Number.isFinite(v) && process.env[name] !== '' && process.env[name] !== undefined ? v : fallback
}

/**
 * player:  /players (Player family, 50K/wk; no sync job uses it).
 * matches: /players/matches (Player/Match/History, 5K/wk; the tracked-player sync uses ~1.1K/wk).
 * mmr:     /players/mmr/history (Player/MMR, 50K/wk; the paper-3 tranche-2 pull uses all of it
 *          until about Oct 26, so the site budget starts at 0 and Max raises it after).
 */
export function policy(pool: Pool): PoolPolicy {
  switch (pool) {
    case 'player':
      return {
        weekly: envInt('PERSONAL_HP_PLAYER_WEEKLY', 3000),
        hourly: envInt('PERSONAL_HP_PLAYER_HOURLY', 120),
        floor: envInt('PERSONAL_HP_PLAYER_FLOOR', 5000),
      }
    case 'matches':
      return {
        weekly: envInt('PERSONAL_HP_MATCHES_WEEKLY', 1500),
        hourly: envInt('PERSONAL_HP_MATCHES_HOURLY', 60),
        floor: envInt('PERSONAL_HP_MATCHES_FLOOR', 1500),
      }
    case 'mmr':
      return {
        weekly: envInt('PERSONAL_HP_MMR_WEEKLY', 0),
        hourly: envInt('PERSONAL_HP_MMR_HOURLY', 60),
        floor: envInt('PERSONAL_HP_MMR_FLOOR', 2000),
      }
  }
}

export class QuotaGuardError extends Error {
  constructor(public pool: Pool, public reason: string) {
    super(`lookup budget for ${pool}: ${reason}`)
  }
}

export class HpUnavailableError extends Error {}

export type HpResult<T> =
  | { kind: 'ok'; data: T }
  | { kind: 'not_found' }
  | { kind: 'private' }

interface UsageRow {
  week: number
  hour: number
  remaining: number | null
}

async function usage(pool: Pool): Promise<UsageRow> {
  const rows = (await db.execute(sql`
    SELECT
      coalesce(sum(calls) FILTER (WHERE hour > now() - interval '7 days'), 0)::int AS week,
      coalesce(sum(calls) FILTER (WHERE hour >= date_trunc('hour', now())), 0)::int AS hour,
      (SELECT last_remaining FROM personal_api_usage
        WHERE pool = ${pool} AND last_remaining IS NOT NULL ORDER BY hour DESC LIMIT 1) AS remaining
    FROM personal_api_usage WHERE pool = ${pool}`)) as unknown as { rows?: UsageRow[] } | UsageRow[]
  const r = Array.isArray(rows) ? rows[0] : rows.rows?.[0]
  return { week: Number(r?.week ?? 0), hour: Number(r?.hour ?? 0), remaining: r?.remaining ?? null }
}

/** Throws QuotaGuardError when the pool cannot take `need` more calls now. */
export async function checkBudget(pool: Pool, need = 1): Promise<void> {
  const p = policy(pool)
  if (p.weekly <= 0) throw new QuotaGuardError(pool, 'switched off')
  const u = await usage(pool)
  if (u.week + need > p.weekly) throw new QuotaGuardError(pool, 'weekly budget used')
  if (u.hour + need > p.hourly) throw new QuotaGuardError(pool, 'hourly budget used')
  if (u.remaining !== null && u.remaining - need < p.floor) throw new QuotaGuardError(pool, 'pool below floor')
}

async function record(pool: Pool, remaining: number | null): Promise<void> {
  await db.execute(sql`
    INSERT INTO personal_api_usage (pool, hour, calls, last_remaining)
    VALUES (${pool}, date_trunc('hour', now()), 1, ${remaining})
    ON CONFLICT (pool, hour) DO UPDATE SET
      calls = personal_api_usage.calls + 1,
      last_remaining = coalesce(excluded.last_remaining, personal_api_usage.last_remaining)`)
  // keep the table to two weeks of hourly rows
  if (Math.random() < 0.02) await db.execute(sql`DELETE FROM personal_api_usage WHERE hour < now() - interval '14 days'`)
}

/** One guarded GET. 404 -> not_found, 403 player_unavailable -> private. */
export async function hpGet<T>(pool: Pool, path: string, params: Record<string, string | undefined>): Promise<HpResult<T>> {
  const key = process.env.HEROES_PROFILE_V2_API_KEY
  if (!key) throw new HpUnavailableError('HEROES_PROFILE_V2_API_KEY not set')
  await checkBudget(pool)
  const url = new URL(`${BASE_URL}/${path}`)
  for (const [k, v] of Object.entries(params)) if (v !== undefined) url.searchParams.set(k, v)
  let res: Response
  try {
    res = await fetch(url, {
      headers: { Authorization: `Bearer ${key}` },
      signal: AbortSignal.timeout(TIMEOUT_MS),
      cache: 'no-store',
    })
  } catch (err) {
    throw new HpUnavailableError(`network: ${String(err)}`)
  }
  const remainingHdr = res.headers.get('x-hp-quota-remaining')
  const remaining = remainingHdr === null ? null : Number(remainingHdr)
  // Count every call that may have been metered; 404s appear unmetered but are cheap to count.
  await record(pool, Number.isFinite(remaining) ? remaining : null)
  if (res.headers.get('x-hp-data-source') === 'fixture') throw new HpUnavailableError('fixture data')
  if (res.status === 404) return { kind: 'not_found' }
  const body: any = await res.json().catch(() => null)
  if (res.status === 403 && body?.error?.code === 'player_unavailable') return { kind: 'private' }
  if (res.status === 429) throw new QuotaGuardError(pool, 'Heroes Profile rate or quota limit')
  if (res.status === 202) throw new HpUnavailableError('Heroes Profile is preparing this data; try again shortly')
  if (!res.ok) throw new HpUnavailableError(`HP ${res.status} ${body?.error?.code ?? ''}`)
  return { kind: 'ok', data: body as T }
}

// ---------------------------------------------------------------- endpoint helpers

export const REGIONS: Record<number, string> = { 1: 'Americas', 2: 'Europe', 3: 'Asia', 5: 'China' }

export interface PlayerSummary {
  battletag: string
  region: number
  blizzId: number
  games: number
  wins: number
  lastPlayed: string | null
  mostPlayed: { hero: string; games: number }[]
  rank: string | null
}

interface HpHeroAgg {
  games_played?: number
  game_date?: string
  hero?: { name?: string }
  blizz_id?: number
}

/** /players for one battletag in one region (one Player-pool call when found). */
export async function fetchSummary(battletag: string, region: number): Promise<HpResult<PlayerSummary>> {
  const r = await hpGet<any>('player', 'players', { battletag, region: String(region), game_type: 'Storm League' })
  if (r.kind !== 'ok') return r
  const d = r.data ?? {}
  const aggs: HpHeroAgg[] = [
    ...(d.heroes_three_most_played ?? []),
    ...(d.heroes_three_latest_played ?? []),
    ...(d.heroes_three_highest_win_rate ?? []),
  ]
  const blizzId = Number(aggs.find(a => a.blizz_id)?.blizz_id ?? 0)
  const dates = aggs.map(a => a.game_date).filter((x): x is string => !!x).sort()
  const wins = Number(d.wins ?? 0)
  const games = wins + Number(d.losses ?? 0)
  if (!blizzId || games === 0) return { kind: 'not_found' }
  return {
    kind: 'ok',
    data: {
      battletag,
      region,
      blizzId,
      games,
      wins,
      lastPlayed: dates.length ? dates[dates.length - 1].slice(0, 10) : null,
      mostPlayed: (d.heroes_three_most_played ?? []).map((a: HpHeroAgg) => ({
        hero: a.hero?.name ?? '?',
        games: Number(a.games_played ?? 0),
      })),
      rank: d.sl_mmr_data?.rank_tier ?? null,
    },
  }
}

export interface HpGame {
  hero: string
  win: boolean
  date: string
}

function isWin(x: unknown): boolean {
  return x === true || x === 1 || x === '1' || String(x).toLowerCase() === 'true'
}

/** Full SL history from /players/mmr/history (one Player/MMR call, rarely more). */
export async function fetchHistoryMmr(battletag: string, region: number): Promise<HpResult<HpGame[]>> {
  const out: HpGame[] = []
  for (let page = 1; page <= 5; page++) {
    const r = await hpGet<any>('mmr', 'players/mmr/history', {
      battletag,
      region: String(region),
      game_type: 'Storm League',
      pagination_page: page > 1 ? String(page) : undefined,
    })
    if (r.kind !== 'ok') return page === 1 ? r : { kind: 'ok', data: out }
    for (const e of r.data?.history ?? []) {
      if (!e?.game_date) continue
      out.push({ hero: e.hero?.name ?? e.hero, win: isWin(e.winner), date: String(e.game_date) })
    }
    const last = Number(r.data?.last_page ?? r.data?.pagination?.last_page ?? 0)
    if (!r.data?.next_page_url && !(last && page < last)) break
  }
  return { kind: 'ok', data: out }
}

/** The most recent `pages` x 100 SL games from /players/matches (one call per page). */
export async function fetchHistoryMatches(battletag: string, region: number, pages: number): Promise<HpResult<HpGame[]>> {
  const out: HpGame[] = []
  for (let page = 1; page <= pages; page++) {
    const r = await hpGet<any>('matches', 'players/matches', {
      battletag,
      region: String(region),
      game_type: 'Storm League',
      pagination_page: String(page),
    })
    if (r.kind !== 'ok') return page === 1 ? r : { kind: 'ok', data: out }
    for (const e of r.data?.data ?? []) {
      if (!e?.game_date) continue
      out.push({ hero: e.hero?.name ?? e.hero, win: isWin(e.winner), date: String(e.game_date) })
    }
    if (!r.data?.next_page_url) break
  }
  return { kind: 'ok', data: out }
}
