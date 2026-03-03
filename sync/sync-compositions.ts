/**
 * Scrape composition win-rate data from Heroes Profile internal API.
 *
 * Unlike the public API (api.heroesprofile.com), this uses the internal
 * Laravel endpoint at heroesprofile.com/api/v1/global/compositions which
 * requires a CSRF token from a session cookie.
 *
 * Writes normalized data to src/lib/data/compositions.json.
 */

import { writeFileSync } from 'fs'
import { resolve } from 'path'
import { log } from './logger'

const HP_BASE = 'https://www.heroesprofile.com'

interface RawRole {
  mmr_type_table_id: number
  mmr_type_id: number
  name: string
}

interface RawComposition {
  composition_id: number
  wins: number
  losses: number
  win_rate: number
  games_played: number
  popularity: number
  role_one: RawRole
  role_two: RawRole
  role_three: RawRole
  role_four: RawRole
  role_five: RawRole
}

interface NormalizedComposition {
  roles: string[]
  winRate: number
  games: number
  popularity: number
}

// Tier → league_tier codes for the API
const TIER_CODES: Record<string, number[]> = {
  low: [1, 2],    // Bronze, Silver
  mid: [3, 4],    // Gold, Platinum
  high: [5, 6],   // Diamond, Master
}

/**
 * Fetch a page from HP to get the session cookie and CSRF token.
 * The CSRF token is embedded in a <meta name="csrf-token"> tag and
 * the session cookie is set via Set-Cookie.
 */
async function getSession(): Promise<{ cookie: string; csrfToken: string }> {
  log.info('Fetching HP session and CSRF token...')

  const resp = await fetch(`${HP_BASE}/Global/Compositions`, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
      Accept: 'text/html',
    },
    redirect: 'manual',
  })

  // Collect cookies from Set-Cookie headers
  const setCookies = resp.headers.getSetCookie?.() ?? []
  const cookies = setCookies
    .map((c) => c.split(';')[0])
    .filter(Boolean)
    .join('; ')

  if (!cookies) {
    throw new Error('No session cookies received from Heroes Profile')
  }

  // Extract CSRF token from HTML
  const html = await resp.text()
  const csrfMatch = html.match(/meta\s+name="csrf-token"\s+content="([^"]+)"/)
  if (!csrfMatch) {
    throw new Error('Could not find CSRF token in Heroes Profile page')
  }

  log.info('Session established')
  return { cookie: cookies, csrfToken: csrfMatch[1] }
}

/**
 * Fetch compositions for a single tier from the internal API.
 */
async function fetchTierCompositions(
  tierCodes: number[],
  cookie: string,
  csrfToken: string
): Promise<RawComposition[]> {
  const body = JSON.stringify({
    league_tier: tierCodes,
    game_type: ['sl'],
    minimum_games: 25,
  })

  const resp = await fetch(`${HP_BASE}/api/v1/global/compositions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      Cookie: cookie,
      'X-CSRF-TOKEN': csrfToken,
      'X-Requested-With': 'XMLHttpRequest',
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
      Referer: `${HP_BASE}/Global/Compositions`,
    },
    body,
  })

  if (!resp.ok) {
    const text = await resp.text().catch(() => '(unreadable)')
    throw new Error(`HP compositions API returned ${resp.status}: ${text.slice(0, 500)}`)
  }

  return resp.json()
}

/**
 * Normalize a raw composition: extract the 5 role names and sort alphabetically.
 */
function normalizeComposition(raw: RawComposition): NormalizedComposition {
  const roles = [
    raw.role_one.name,
    raw.role_two.name,
    raw.role_three.name,
    raw.role_four.name,
    raw.role_five.name,
  ].sort()

  return {
    roles,
    winRate: Math.round(raw.win_rate * 100) / 100,
    games: raw.games_played,
    popularity: Math.round(raw.popularity * 100) / 100,
  }
}

export async function syncCompositions(): Promise<void> {
  log.info('── Syncing composition data ──')

  const { cookie, csrfToken } = await getSession()

  const result: Record<string, NormalizedComposition[]> = {}

  for (const [tier, codes] of Object.entries(TIER_CODES)) {
    log.info(`Fetching ${tier} tier compositions (league_tier=${JSON.stringify(codes)})...`)
    const raw = await fetchTierCompositions(codes, cookie, csrfToken)
    result[tier] = raw.map(normalizeComposition)
    log.info(`  ${tier}: ${result[tier].length} compositions`)
  }

  const outPath = resolve(__dirname, '../src/lib/data/compositions.json')
  writeFileSync(outPath, JSON.stringify(result, null, 2) + '\n')
  log.info(`Wrote composition data to ${outPath}`)
}
