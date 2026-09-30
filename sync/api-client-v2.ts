/**
 * Heroes Profile API v1 client ("v2" from our side: the successor to
 * sync/api-client.ts, which speaks the retiring api.heroesprofile.com API).
 *
 * Design: DROP-IN COMPATIBLE with the old client's surface. Every worker
 * calls the same methods (getReplayData, getReplayMinId, getPlayerReplays,
 * getHeroStats, ...) and receives the LEGACY response shapes; this client
 * translates v1's named objects back into them, so cutover is a client
 * swap behind the HP_API env flag (see sync/hp-api.ts), not a rewrite of
 * every parser. See sync/docs/hp-v1-migration-notes.md for the mapping.
 *
 * v1 semantics handled here:
 *  - Authorization: Bearer header (never ?api_token=)
 *  - error envelope {error:{code,message}} with real status codes; QUOTA
 *    (429 quota_exceeded) throws an error whose message contains
 *    "Max calls" so existing worker quota-benching logic keeps working
 *  - /replays cursor paging (exclusive `after`): getReplayMinId emulates
 *    the old inclusive min_id contract and aggregates several pages
 *  - 202 + job polling for global statistics endpoints (poll costs no
 *    quota; Retry-After honored)
 *  - fixture mode surfaced via lastDataSource ("fixture" until the
 *    account activates live data — never store fixture data)
 */
import { HpAccessPausedError } from './hp-errors'
import { log } from './logger'

const BASE_URL = 'https://www.heroesprofile.com/api/external/v1'

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

class RateLimiter {
  private stamps: number[] = []
  constructor(private maxPerMinute: number) {}
  async acquire(): Promise<void> {
    for (;;) {
      const now = Date.now()
      this.stamps = this.stamps.filter(t => now - t < 60_000)
      if (this.stamps.length < this.maxPerMinute) {
        this.stamps.push(now)
        return
      }
      const wait = 60_000 - (now - this.stamps[0]) + 50
      await sleep(wait)
    }
  }
}

export class HeroesProfileApiV2 {
  private rateLimiter: RateLimiter
  private callCount = 0
  /** "fixture" until live data is activated on the account; "live" after. */
  lastDataSource: string | null = null
  /**
   * Production workers set this: a fixture response then throws
   * HpAccessPausedError, which every worker already treats as "stop, don't
   * store, don't advance cursors".
   */
  refuseFixture = false

  constructor(
    private apiKey: string,
    maxCallsPerMinute = 55,
    private maxRetries = 5,
  ) {
    this.rateLimiter = new RateLimiter(maxCallsPerMinute)
  }

  getCallCount(): number {
    return this.callCount
  }

  private buildUrl(path: string, params: Record<string, string | undefined>): string {
    const url = new URL(`${BASE_URL}/${path}`)
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) url.searchParams.set(k, v)
    }
    return url.toString()
  }

  /**
   * Core fetch: rate limiting, retries on 429-rate/5xx/network, error
   * envelope decoding, and transparent 202-job polling.
   */
  async fetch<T = any>(path: string, params: Record<string, string | undefined> = {}): Promise<T> {
    const url = this.buildUrl(path, params)
    let delay = 2_000

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      await this.rateLimiter.acquire()
      this.callCount++

      let response: Response
      try {
        response = await fetch(url, {
          headers: { Authorization: `Bearer ${this.apiKey}` },
          signal: AbortSignal.timeout(360_000),
        })
      } catch (err) {
        if (attempt === this.maxRetries) {
          throw new Error(`Network error after ${this.maxRetries} retries for ${path}: ${err}`)
        }
        await sleep(delay + Math.random() * 1_000)
        delay = Math.min(delay * 2, 300_000)
        continue
      }

      this.lastDataSource = response.headers.get('x-hp-data-source')
      if (this.refuseFixture && this.lastDataSource === 'fixture') {
        throw new HpAccessPausedError(response.status, 'fixture_data', `${path}: live data not activated on the account`)
      }

      if (response.status === 202) {
        // Global-statistics job: poll Location until 200. Polls cost no quota.
        const body: any = await response.json().catch(() => ({}))
        const jobPath = response.headers.get('location') ?? (body.job_id ? `jobs/${body.job_id}` : null)
        const retryAfter = Number(response.headers.get('retry-after')) || 10
        if (!jobPath) throw new Error(`202 without job location for ${path}`)
        return this.pollJob<T>(jobPath.replace(/^.*\/v1\//, ''), retryAfter)
      }

      if (response.ok) {
        return response.json() as Promise<T>
      }

      const errBody: any = await response.json().catch(() => null)
      const code = errBody?.error?.code ?? `http_${response.status}`
      const message = errBody?.error?.message ?? response.statusText

      if (response.status === 429) {
        if (code === 'quota_exceeded') {
          // Weekly per-endpoint allowance spent. "Max calls" keeps every
          // worker's existing quota-benching string check working.
          throw new Error(`Max calls (v1 quota_exceeded) for ${path}: ${message}`)
        }
        // Per-minute rate limit: back off and retry.
        if (attempt === this.maxRetries) throw new Error(`Rate limited (429) after retries for ${path}`)
        await sleep(delay + Math.random() * 1_000)
        delay = Math.min(delay * 2, 300_000)
        continue
      }

      if (response.status >= 500) {
        if (attempt === this.maxRetries) throw new Error(`API error ${response.status} for ${path}: ${message}`)
        await sleep(delay + Math.random() * 1_000)
        delay = Math.min(delay * 2, 300_000)
        continue
      }

      // 401 bad key / 403 not in plan, terms_not_accepted,
      // project_details_required: account-level, never item-level.
      if (response.status === 401 || response.status === 403) {
        throw new HpAccessPausedError(response.status, code, `${path}: ${message}`)
      }

      // 4xx (404 / 422 incl. timeframe_too_wide, timeframe_unavailable): don't retry.
      throw new Error(`API error ${response.status} (${code}) for ${path}: ${message}`)
    }
    throw new Error(`unreachable retry loop for ${path}`)
  }

  private async pollJob<T>(jobPath: string, retryAfter: number): Promise<T> {
    const deadline = Date.now() + 15 * 60_000
    for (;;) {
      if (Date.now() > deadline) throw new Error(`Job ${jobPath} did not finish within 15 min`)
      await sleep(Math.max(retryAfter, 5) * 1000)
      const response = await fetch(`${BASE_URL}/${jobPath}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
        signal: AbortSignal.timeout(60_000),
      })
      if (response.status === 202) continue
      if (response.status === 429) {
        // Per-minute limit on /jobs under concurrent polling: back off, keep the job.
        retryAfter = Math.max(Number(response.headers.get('retry-after')) || 0, retryAfter * 2, 10)
        continue
      }
      if (response.status === 401 || response.status === 403) {
        const b: any = await response.json().catch(() => null)
        throw new HpAccessPausedError(response.status, b?.error?.code ?? `http_${response.status}`, `${jobPath}: ${b?.error?.message ?? ''}`)
      }
      if (response.status === 404) throw new Error(`Job ${jobPath} expired; restart the original call`)
      if (response.status === 500) {
        const b: any = await response.json().catch(() => null)
        throw new Error(`Job ${jobPath} failed: ${b?.error ?? 'unknown'}`)
      }
      if (!response.ok) throw new Error(`Job ${jobPath} unexpected status ${response.status}`)
      return response.json() as Promise<T>
    }
  }

  // ── Legacy-shape methods (drop-in for HeroesProfileApi) ─────────────

  /**
   * Old Replay/Data contract: an object whose keys are battletags (player
   * entries) plus replay-level fields. Built from v1 /replay/{id}.
   * NOTE: v1 detail has no game_version; workers take version from the
   * listing row, which still carries it.
   */
  async getReplayData(replayId: number): Promise<Record<string, any>> {
    const m: any = await this.fetch(`replay/${replayId}`)
    const legacy: Record<string, any> = {
      region: m.region,
      game_type: m.game_type,
      game_date: m.game_date,
      game_map: m.game_map?.name ?? m.game_map,
      game_length: m.game_length,
      winner: m.winner,
      draft_order: (m.draft_order ?? []).map((e: any) => ({
        ...e,
        hero: e.hero?.name ?? e.hero,
      })),
      replay_bans: (m.replay_bans ?? []).map((teamBans: any[]) =>
        (teamBans ?? []).map((b: any) => ({ ...b, hero: b.hero?.name ?? b.hero }))),
      experience_breakdown: m.experience_breakdown,
    }
    for (const team of m.players ?? []) {
      for (const p of team ?? []) {
        if (!p?.battletag) continue
        legacy[p.battletag] = {
          ...p,
          hero: p.hero?.name ?? p.hero,
          scores: p.score ?? p.scores ?? null,
        }
      }
    }
    return legacy
  }

  /**
   * Old Replay/Min_id contract: listing rows from an INCLUSIVE min id. v1
   * /replays pages ~25 rows with an EXCLUSIVE `after` cursor, so this
   * aggregates pages (each page = one metered replay_index call). Stops at
   * maxRows, or before the first row with id >= beforeId (range scans).
   */
  async getReplayMinId(
    minId: number,
    gameType = 'Storm League',
    maxRows = 200,
    beforeId?: number,
    majorPatch?: string,
  ): Promise<any[]> {
    const rows: any[] = []
    let after = minId - 1 // inclusive -> exclusive
    while (rows.length < maxRows) {
      const d: any = await this.fetch('replays', {
        after: String(after),
        game_type: gameType,
        // Server-side patch filter: far fewer pages against the 20K/wk replay_index allowance.
        timeframe_type: majorPatch ? 'major' : undefined,
        timeframe: majorPatch,
      })
      const page: any[] = d.replays ?? []
      if (page.length === 0) break
      for (const r of page) {
        if (beforeId !== undefined && r.replayID >= beforeId) return rows
        // Old rows had `valid`; v1 dropped it. Synthesize so existing
        // `valid === 1` filters keep their meaning (parsed and present).
        rows.push({ ...r, valid: r.parsed && !r.deleted ? 1 : 0 })
      }
      if (d.next_after === null || d.next_after === undefined) break
      after = Number(d.next_after)
    }
    return rows
  }

  /** Old Replay/Max contract: highest stored replay id, as a number. */
  async getReplayMax(): Promise<number> {
    const d: any = await this.fetch('replays', { after: '999999999' })
    if (typeof d.max_replay_id === 'number') return d.max_replay_id
    const d2: any = await this.fetch('replays', { after: '0' })
    return Number(d2.max_replay_id)
  }

  /**
   * Old Player/Replays contract: { "Storm League": { "<id>": {...} } } with
   * hero/map/game_type as names and level_* as talent titles. Built from v1
   * /players/matches (paginated, 100/page; may answer 202 + job).
   */
  async getPlayerReplays(
    battletag: string,
    region: number,
    startDate?: string,
    endDate?: string,
    gameType = 'Storm League',
  ): Promise<Record<string, any>> {
    const inner: Record<string, any> = {}
    let page = 1
    for (;;) {
      const d: any = await this.fetch('players/matches', {
        battletag,
        region: String(region),
        game_type: gameType,
        pagination_page: String(page),
      })
      for (const row of d.data ?? []) {
        const when = String(row.game_date ?? '')
        if (startDate && when && when.slice(0, 10) < startDate) continue
        if (endDate && when && when.slice(0, 10) >= endDate) continue
        const flat: Record<string, any> = {
          ...row,
          hero: row.hero?.name ?? row.hero,
          game_map: row.game_map?.name ?? row.game_map,
          game_type: row.game_type?.name ?? row.game_type,
        }
        for (const lvl of TALENT_LEVEL_KEYS) {
          if (row[lvl] && typeof row[lvl] === 'object') flat[lvl] = row[lvl].title ?? null
        }
        inner[String(row.replayID)] = flat
      }
      if (!d.next_page_url || page >= Number(d.last_page ?? page)) break
      page++
    }
    return { [gameType]: inner }
  }

  /** Old Heroes/Stats contract: an array of rows with `name` = hero name. */
  async getHeroStats(timeframeType: string, timeframe: string, leagueTier?: string, hero?: string) {
    const d: any = await this.fetch('heroes/stats', {
      timeframe_type: timeframeType,
      timeframe,
      game_type: 'Storm League',
      league_tier: leagueTier,
      hero,
    })
    const rows: any[] = d.data ?? []
    // TODO(max): confirm v1 heroes/stats row fields against the docs; the
    // fixture rows carry only name/role/wins. Refuse rather than let
    // sync-global store zeros for games/ban_rate/popularity.
    if (rows.length > 0 && rows.every(r => r.games_played === undefined && r.games === undefined)) {
      throw new Error('heroes/stats rows have no games_played field; v1 row schema unconfirmed')
    }
    // v1 `popularity` is pick + ban rate (Rehgar 2.55: 24.78 + 20.21 = 45);
    // the old API's `popularity`, which sync-global stores as pickRate, was
    // the pick rate alone. Hand the legacy field the v1 pick_rate.
    return rows.map(r => ({
      ...r,
      name: r.name ?? r.hero?.name ?? r.hero,
      popularity: r.pick_rate ?? r.popularity,
    }))
  }

  /**
   * Old Heroes/Matchups contract: { "<HeroB>": { ally: {wins, losses,
   * win_rate}, enemy: {wins, losses, win_rate} } }, where enemy stats are
   * the OPPONENT's (sync-global inverts them).
   */
  async getHeroMatchups(hero: string, timeframeType: string, timeframe: string, leagueTier?: string) {
    if (V1_ENEMY_ROWS_ARE_OPPONENT_PERSPECTIVE === null) {
      // TODO(max): confirm in the v1 docs whose wins `enemy` rows count.
      // Guessing wrong silently inverts every counter-pick stat.
      throw new Error('heroes/matchups enemy-row perspective unconfirmed; not syncing matchups')
    }
    const d: any = await this.fetch('heroes/matchups', {
      timeframe_type: timeframeType,
      timeframe,
      game_type: 'Storm League',
      hero,
      league_tier: leagueTier,
    })
    const out: Record<string, { ally?: any; enemy?: any }> = {}
    const conv = (r: any) => {
      const wins = Number(r.wins ?? 0)
      const losses = Number(r.losses ?? 0)
      const games = wins + losses
      return { wins, losses, win_rate: r.win_rate ?? (games > 0 ? (100 * wins) / games : 0) }
    }
    for (const r of d.ally ?? []) {
      const name = r.hero?.name ?? r.hero
      if (name) (out[name] ??= {}).ally = conv(r)
    }
    for (const r of d.enemy ?? []) {
      const name = r.hero?.name ?? r.hero
      if (!name) continue
      const c = conv(r)
      ;(out[name] ??= {}).enemy = V1_ENEMY_ROWS_ARE_OPPONENT_PERSPECTIVE
        ? c
        : { wins: c.losses, losses: c.wins, win_rate: 100 - c.win_rate }
    }
    return out
  }

  /** Old Patches contract: { "2.55": ["2.55.17.97771", ...], ... }. */
  async getPatches(): Promise<Record<string, string[]>> {
    const d: any = await this.fetch('patches')
    const out: Record<string, string[]> = {}
    for (const p of d.patches ?? []) {
      if (p.valid_globals === false || !p.game_version) continue
      ;(out[`${p.major}.${p.minor}`] ??= []).push(p.game_version)
    }
    return out
  }

  /**
   * Old Heroes/Talents/Details contract (same argument order as the old
   * client): { "<tier>": [{ title, games_played, wins, win_rate, popularity }] }.
   * v1 requires a hero; rows carry the HERO name in `name` and the talent
   * under talentInfo.
   */
  async getTalentDetails(timeframeType: string, timeframe: string, leagueTier?: string, hero?: string) {
    if (!hero) throw new Error('v1 heroes/talents/details needs a hero')
    const d: any = await this.fetch('heroes/talents/details', {
      timeframe_type: timeframeType,
      timeframe,
      game_type: 'Storm League',
      hero,
      league_tier: leagueTier,
    })
    const out: Record<string, any[]> = {}
    for (const [tier, rows] of Object.entries(d ?? {})) {
      if (!Array.isArray(rows)) continue
      out[tier] = rows.map((r: any) => ({
        title: r.talentInfo?.title ?? r.title,
        games_played: r.games_played,
        wins: r.wins,
        losses: r.losses,
        win_rate: r.win_rate,
        popularity: r.popularity,
      }))
    }
    return out
  }

  /**
   * Old Heroes/Stats group_by_map contract. Only sync/run-once.ts uses it; v1
   * serves it as heroes/maps, one hero per call (Heroes/Map/Stats allowance
   * is 1K/wk), so it is deliberately not ported yet.
   */
  async getHeroMapStats(_timeframeType: string, _timeframe: string, _leagueTier?: string): Promise<never> {
    throw new Error('hero-map stats are not ported to v1 (heroes/maps needs one call per hero)')
  }

  /** Composition win rates: same row shape the internal site endpoint returned. */
  async getCompositions(params: {
    timeframeType: string
    timeframe: string
    gameType: string
    leagueTier?: string
    minimumGames?: number
  }): Promise<any[]> {
    const d: any = await this.fetch('compositions', {
      timeframe_type: params.timeframeType,
      timeframe: params.timeframe,
      game_type: params.gameType,
      league_tier: params.leagueTier,
      minimum_games: params.minimumGames === undefined ? undefined : String(params.minimumGames),
    })
    return Array.isArray(d) ? d : d.data ?? []
  }
}

const TALENT_LEVEL_KEYS = [
  'level_one', 'level_four', 'level_seven', 'level_ten',
  'level_thirteen', 'level_sixteen', 'level_twenty',
] as const

/**
 * Whose wins do v1 heroes/matchups `enemy` rows count? The opponent's, as in
 * the old API: confirmed 2026-09-30 by sync/check-v1-live.ts (Jaina, patch
 * 2.55: r = -0.966 against her win rate per opponent in replay_draft_data).
 */
const V1_ENEMY_ROWS_ARE_OPPONENT_PERSPECTIVE: boolean | null = true
