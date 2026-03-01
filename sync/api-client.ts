import { log } from './logger'

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Concurrency-safe sliding-window rate limiter.
 * Serializes acquire() calls via a promise chain so concurrent workers
 * can't race past the limit.
 */
class RateLimiter {
  private timestamps: number[] = []
  private chain: Promise<void> = Promise.resolve()

  constructor(private maxPerMinute: number) {}

  acquire(): Promise<void> {
    this.chain = this.chain.then(() => this.doAcquire())
    return this.chain
  }

  private async doAcquire(): Promise<void> {
    while (true) {
      const now = Date.now()
      this.timestamps = this.timestamps.filter(t => now - t < 60_000)

      if (this.timestamps.length < this.maxPerMinute) {
        this.timestamps.push(Date.now())
        return
      }

      const waitMs = 60_000 - (now - this.timestamps[0]) + 200
      log.info(`Rate limiter: waiting ${Math.round(waitMs / 1000)}s (${this.timestamps.length}/${this.maxPerMinute} calls in window)`)
      await sleep(waitMs)
    }
  }
}

const BASE_URL = 'https://api.heroesprofile.com/api'

export class HeroesProfileApi {
  private rateLimiter: RateLimiter
  private maxRetries: number
  private callCount = 0

  constructor(
    private apiKey: string,
    maxCallsPerMinute = 55, // Stay safely under 60/min
    maxRetries = 5,
  ) {
    this.rateLimiter = new RateLimiter(maxCallsPerMinute)
    this.maxRetries = maxRetries
  }

  getCallCount(): number {
    return this.callCount
  }

  private buildUrl(path: string, params: Record<string, string | undefined>): string {
    const url = new URL(`${BASE_URL}/${path}`)
    url.searchParams.set('api_token', this.apiKey)
    for (const [key, val] of Object.entries(params)) {
      if (val !== undefined) url.searchParams.set(key, val)
    }
    return url.toString()
  }

  /**
   * Core fetch with rate limiting, exponential backoff, and jitter.
   * Retries on 429 (rate limited) and 5xx (server error).
   * Throws immediately on 4xx (client error, except 429).
   */
  async fetch<T = any>(path: string, params: Record<string, string | undefined> = {}): Promise<T> {
    const url = this.buildUrl(path, params)
    // Redact api_token from logs
    const logUrl = url.replace(/api_token=[^&]+/, 'api_token=***')
    let delay = 2_000

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      await this.rateLimiter.acquire()
      this.callCount++

      let response: Response
      try {
        response = await fetch(url, { signal: AbortSignal.timeout(360_000) })
      } catch (err) {
        if (attempt === this.maxRetries) {
          throw new Error(`Network error after ${this.maxRetries} retries for ${logUrl}: ${err}`)
        }
        const jitter = Math.random() * 1_000
        log.warn(`Network error for ${logUrl}, retry ${attempt + 1}/${this.maxRetries} in ${Math.round((delay + jitter) / 1000)}s`)
        await sleep(delay + jitter)
        delay = Math.min(delay * 2, 300_000)
        continue
      }

      if (response.ok) {
        const contentType = response.headers.get('content-type') ?? ''
        if (!contentType.includes('json')) {
          const remaining = response.headers.get('x-ratelimit-remaining')
          throw new Error(
            `API returned non-JSON response (content-type: ${contentType}) for ${logUrl}. ` +
            `This usually means the API key is invalid or the subscription has expired. ` +
            `Rate limit remaining: ${remaining ?? 'unknown'}`,
          )
        }
        const data = await response.json()
        this.validateResponse(data, logUrl)
        return data as T
      }

      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After')
        const waitMs = retryAfter ? parseInt(retryAfter, 10) * 1_000 : delay
        const jitter = Math.random() * 2_000

        if (attempt === this.maxRetries) {
          throw new Error(`Rate limited (429) after ${this.maxRetries} retries for ${logUrl}`)
        }

        log.warn(`Rate limited (429) for ${logUrl}. Waiting ${Math.round((waitMs + jitter) / 1000)}s (attempt ${attempt + 1}/${this.maxRetries})`)
        await sleep(waitMs + jitter)
        delay = Math.min(delay * 2, 300_000)
        continue
      }

      if (response.status >= 500) {
        if (attempt === this.maxRetries) {
          throw new Error(`Server error ${response.status} after ${this.maxRetries} retries for ${logUrl}`)
        }
        const jitter = Math.random() * 1_000
        log.warn(`Server error ${response.status} for ${logUrl}, retry ${attempt + 1}/${this.maxRetries} in ${Math.round((delay + jitter) / 1000)}s`)
        await sleep(delay + jitter)
        delay = Math.min(delay * 2, 300_000)
        continue
      }

      // 4xx client error (not 429) — don't retry
      const body = await response.text().catch(() => '(could not read body)')
      throw new Error(`API error ${response.status} for ${logUrl}: ${body.slice(0, 500)}`)
    }

    throw new Error('Unreachable')
  }

  private validateResponse(data: any, url: string): void {
    if (data === null || data === undefined) {
      throw new Error(`Empty response from ${url}`)
    }
    // Some APIs wrap errors in a JSON body
    if (typeof data === 'object' && !Array.isArray(data) && data.error) {
      throw new Error(`API returned error for ${url}: ${JSON.stringify(data.error)}`)
    }
  }

  // ── Endpoint methods ──────────────────────────────────────────────

  async getPatches() {
    return this.fetch('Patches')
  }

  async getHeroStats(timeframeType: string, timeframe: string, leagueTier?: string) {
    return this.fetch('Heroes/Stats', {
      timeframe_type: timeframeType,
      timeframe,
      game_type: 'Storm League',
      league_tier: leagueTier,
    })
  }

  async getHeroMapStats(timeframeType: string, timeframe: string, leagueTier?: string) {
    return this.fetch('Heroes/Stats', {
      timeframe_type: timeframeType,
      timeframe,
      game_type: 'Storm League',
      league_tier: leagueTier,
      group_by_map: 'true',
    })
  }

  async getTalentDetails(timeframeType: string, timeframe: string, leagueTier?: string, hero?: string) {
    return this.fetch('Heroes/Talents/Details', {
      timeframe_type: timeframeType,
      timeframe,
      game_type: 'Storm League',
      league_tier: leagueTier,
      hero,
    })
  }

  async getHeroMatchups(hero: string, timeframeType: string, timeframe: string, leagueTier?: string) {
    return this.fetch('Heroes/Matchups', {
      timeframe_type: timeframeType,
      timeframe,
      game_type: 'Storm League',
      hero,
      league_tier: leagueTier,
    })
  }

  async getPlayerReplays(battletag: string, region: number, startDate?: string) {
    return this.fetch('Player/Replays', {
      battletag,
      region: String(region),
      mode: 'json',
      game_type: 'Storm League',
      start_date: startDate,
    })
  }

  async getPlayerHeroAll(battletag: string, region: number) {
    return this.fetch('Player/Hero/All', {
      battletag,
      region: String(region),
      game_type: 'Storm League',
      mode: 'json',
    })
  }
}
