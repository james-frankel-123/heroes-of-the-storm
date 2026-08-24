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

      // 4xx (401 bad key / 403 not in plan / 404 / 422): don't retry.
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
   * Old Replay/Min_id contract: up to ~maxRows listing rows from an
   * INCLUSIVE min id. v1 /replays pages ~25 rows with an EXCLUSIVE
   * `after` cursor, so this aggregates pages (each page = one metered
   * call against the replay index allowance).
   */
  async getReplayMinId(
    minId: number,
    gameType?: string,
    maxRows = 200,
  ): Promise<any[]> {
    const rows: any[] = []
    let after = minId - 1 // inclusive -> exclusive
    while (rows.length < maxRows) {
      const d: any = await this.fetch('replays', {
        after: String(after),
        game_type: gameType,
      })
      const page: any[] = d.replays ?? []
      if (page.length === 0) break
      for (const r of page) {
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
   * Old Player/Replays contract: { "Storm League": { "<id>": {...} } }.
   * Built from v1 /players/matches (paginated, 100/page). Date-window
   * params are honored by filtering rows, so the enumerator's chunking
   * keeps working — though with pagination it no longer needs chunks.
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
        inner[String(row.replayID)] = row
      }
      if (!d.next_page_url || page >= Number(d.last_page ?? page)) break
      page++
    }
    return { [gameType]: inner }
  }

  /** Old Heroes/Stats contract: v1 returns averages + data rows; legacy
   * callers consumed the rows. */
  async getHeroStats(timeframeType: string, timeframe: string, leagueTier?: string, hero?: string) {
    const d: any = await this.fetch('heroes/stats', {
      timeframe_type: timeframeType,
      timeframe,
      game_type: 'Storm League',
      league_tier: leagueTier,
      hero,
    })
    return d.data ?? d
  }

  async getHeroMatchups(hero: string, timeframeType: string, timeframe: string, leagueTier?: string) {
    return this.fetch('heroes/matchups', {
      timeframe_type: timeframeType,
      timeframe,
      game_type: 'Storm League',
      hero,
      league_tier: leagueTier,
    })
  }

  async getPatches() {
    const d: any = await this.fetch('patches')
    return d.patches ?? d
  }

  async getTalentDetails(hero: string, timeframeType: string, timeframe: string, leagueTier?: string) {
    return this.fetch('heroes/talents/details', {
      timeframe_type: timeframeType,
      timeframe,
      game_type: 'Storm League',
      hero,
      league_tier: leagueTier,
    })
  }
}
