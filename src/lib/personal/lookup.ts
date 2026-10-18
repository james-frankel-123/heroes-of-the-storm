/**
 * Player lookup orchestration: search with disambiguation, history fetch, skill cards.
 * Server only (API routes under src/app/api/personal). Design: docs/personalization-site-design.md.
 */
import {
  fetchHistoryMatches,
  fetchHistoryMmr,
  fetchSummary,
  HpUnavailableError,
  policy,
  QuotaGuardError,
  REGIONS,
  type HpGame,
  type HpResult,
  type PlayerSummary,
} from './hp'
import { ARTIFACT, computeProfile, dayOf, heroIndex, type HistoryRow, type Profile } from './model'
import {
  findByName,
  findByTag,
  forget,
  isFresh,
  metricEvidence,
  PROFILE_TTL_H,
  saveProfile,
  saveSummary,
  SUMMARY_TTL_H,
} from './store'

export const MATCH_PAGES = Number(process.env.PERSONAL_MATCH_PAGES ?? 3)

const TAG_RE = /^[^\s#]{2,12}#\d{3,6}$/u
const NAME_RE = /^[^\s#]{2,12}$/u

export function parseQuery(q: string): { kind: 'tag'; battletag: string } | { kind: 'name'; name: string } | null {
  const s = q.trim()
  if (TAG_RE.test(s)) return { kind: 'tag', battletag: s }
  if (NAME_RE.test(s)) return { kind: 'name', name: s }
  return null
}

export interface Candidate {
  battletag: string
  region: number
  regionName: string
  games: number
  wins: number
  lastPlayed: string | null
  mostPlayed: { hero: string; games: number }[]
  rank: string | null
}

export interface SearchResult {
  candidates: Candidate[]
  /** Regions where the account is private on Heroes Profile (nothing is shown for them). */
  privateRegions: string[]
  unavailable?: string
  hint?: string
}

function toCandidate(s: PlayerSummary): Candidate {
  return {
    battletag: s.battletag,
    region: s.region,
    regionName: REGIONS[s.region] ?? `Region ${s.region}`,
    games: s.games,
    wins: s.wins,
    lastPlayed: s.lastPlayed,
    mostPlayed: s.mostPlayed,
    rank: s.rank,
  }
}

function unavailableMessage(err: unknown): string {
  if (err instanceof QuotaGuardError) {
    return 'New lookups are paused for now because we have used this week\'s share of our Heroes Profile allowance. Players looked up recently still work. Try again in a few hours.'
  }
  if (err instanceof HpUnavailableError) return 'Heroes Profile did not answer. Try again in a few minutes.'
  throw err
}

/**
 * "Name#1234": one /players probe per region (cached for a day). "Name": only accounts
 * someone has looked up before (Heroes Profile has no search by name).
 */
export async function search(q: string): Promise<SearchResult> {
  const parsed = parseQuery(q)
  if (!parsed) return { candidates: [], privateRegions: [], hint: 'Enter a BattleTag like Name#1234.' }
  if (parsed.kind === 'name') {
    const rows = await findByName(parsed.name)
    return {
      candidates: rows.map(r => toCandidate(r.summary)),
      privateRegions: [],
      hint:
        rows.length === 0
          ? 'Add the number after the # (for example Name#1234) to look up a player we have not seen yet.'
          : 'These are players looked up here before. Add the #number to find someone else.',
    }
  }
  const out: SearchResult = { candidates: [], privateRegions: [] }
  const regions = Object.keys(REGIONS).map(Number)
  const results = await Promise.all(
    regions.map(async region => {
      const cached = await findByTag(parsed.battletag, region)
      if (cached && isFresh(cached.summaryAt, SUMMARY_TTL_H)) return { region, r: { kind: 'ok', data: cached.summary } as HpResult<PlayerSummary> }
      try {
        const r = await fetchSummary(parsed.battletag, region)
        if (r.kind === 'ok') await saveSummary(r.data)
        if (r.kind === 'private') await forget(parsed.battletag, region)
        return { region, r }
      } catch (err) {
        if (cached) return { region, r: { kind: 'ok', data: cached.summary } as HpResult<PlayerSummary> }
        return { region, err }
      }
    }),
  )
  for (const x of results) {
    if ('err' in x) {
      out.unavailable = unavailableMessage(x.err)
      continue
    }
    if (x.r.kind === 'ok') out.candidates.push(toCandidate(x.r.data))
    if (x.r.kind === 'private') out.privateRegions.push(REGIONS[x.region])
  }
  out.candidates.sort((a, b) => (b.lastPlayed ?? '').localeCompare(a.lastPlayed ?? '') || b.games - a.games)
  return out
}

export interface PlayerResult {
  status: 'ok' | 'private' | 'not_found' | 'unavailable'
  message?: string
  candidate?: Candidate
  profile?: Profile
  historySource?: string | null
  computedAt?: string | null
  stale?: boolean
  metricPending?: boolean
}

function toRows(games: HpGame[]): { rows: HistoryRow[]; unknownHeroes: number } {
  const rows: HistoryRow[] = []
  let unknownHeroes = 0
  for (const g of games) {
    const hero = heroIndex(g.hero)
    const day = dayOf(g.date.replace(' ', 'T') + (g.date.length > 10 && !/[zZ+]/.test(g.date) ? 'Z' : ''))
    if (hero < 0 || !Number.isFinite(day)) {
      unknownHeroes++
      continue
    }
    rows.push({ hero, win: g.win, day })
  }
  return { rows, unknownHeroes }
}

async function fetchHistory(battletag: string, region: number): Promise<{ r: HpResult<HpGame[]>; source: string }> {
  if (policy('mmr').weekly > 0) {
    try {
      return { r: await fetchHistoryMmr(battletag, region), source: 'mmr' }
    } catch (err) {
      if (!(err instanceof QuotaGuardError)) throw err
    }
  }
  const r = await fetchHistoryMatches(battletag, region, MATCH_PAGES)
  return { r, source: r.kind === 'ok' ? `matches:${r.data.length}` : 'matches' }
}

/** Skill cards for one account; served from the cache while fresh. */
export async function player(battletag: string, region: number): Promise<PlayerResult> {
  if (!TAG_RE.test(battletag) || !REGIONS[region]) return { status: 'not_found', message: 'Unknown player.' }
  const cached = await findByTag(battletag, region)
  const serveCached = (extra: Partial<PlayerResult> = {}): PlayerResult => ({
    status: 'ok',
    candidate: toCandidate(cached!.summary),
    profile: cached!.profile!,
    historySource: cached!.historySource,
    computedAt: cached!.profileAt,
    metricPending: cached!.wantsMetric,
    ...extra,
  })
  if (cached?.profile && isFresh(cached.profileAt, PROFILE_TTL_H) && cached.profile.modelVersion === ARTIFACT.version) {
    // a fresh imp2 card is upgraded as soon as the corpus evidence arrives
    if (!(cached.wantsMetric && (await metricEvidence(region, cached.blizzId, ARTIFACT.version)))) return serveCached()
  }
  try {
    let summary = cached?.summary
    if (!summary || !isFresh(cached!.summaryAt, SUMMARY_TTL_H)) {
      const s = await fetchSummary(battletag, region)
      if (s.kind === 'private') {
        await forget(battletag, region)
        return { status: 'private', message: 'This player\'s Heroes Profile account is private, so we show nothing for it.' }
      }
      if (s.kind === 'not_found') return { status: 'not_found', message: 'No Storm League games found for this BattleTag in this region.' }
      summary = s.data
      await saveSummary(summary)
    }
    const { r, source } = await fetchHistory(battletag, region)
    if (r.kind === 'private') {
      await forget(battletag, region)
      return { status: 'private', message: 'This player\'s Heroes Profile account is private, so we show nothing for it.' }
    }
    if (r.kind === 'not_found') return { status: 'not_found', message: 'No Storm League games found for this BattleTag in this region.' }
    const { rows } = toRows(r.data)
    const evidence = await metricEvidence(region, summary.blizzId, ARTIFACT.version)
    const asOf = dayOf(new Date()) + 1
    const profile = computeProfile(rows, asOf, evidence)
    await saveProfile(region, summary.blizzId, profile, source, !evidence)
    return {
      status: 'ok',
      candidate: toCandidate(summary),
      profile,
      historySource: source,
      computedAt: new Date().toISOString(),
      metricPending: !evidence,
    }
  } catch (err) {
    const message = unavailableMessage(err)
    if (cached?.profile) return serveCached({ stale: true, message })
    return { status: 'unavailable', message }
  }
}

// ---------------------------------------------------------------- per-IP limit

const hits = new Map<string, number[]>()

/** Soft per-instance limit on lookups that may call Heroes Profile. */
export function allowIp(ip: string, max = Number(process.env.PERSONAL_IP_PER_10MIN ?? 20)): boolean {
  const now = Date.now()
  const list = (hits.get(ip) ?? []).filter(t => now - t < 600_000)
  if (list.length >= max) {
    hits.set(ip, list)
    return false
  }
  list.push(now)
  hits.set(ip, list)
  if (hits.size > 5000) hits.clear()
  return true
}
