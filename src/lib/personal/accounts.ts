/**
 * Reads of the published player table (personal_accounts on Neon). Server only.
 * Neon egress per request: one row (~350 B) for a lookup; for a name match, the
 * candidates' small columns, then the cards of at most 4.
 */
import { sql } from 'drizzle-orm'
import { db } from '@/lib/db'
import { decodeCard, draftByName, HERO_NAMES, type Card } from './card'
import { nfc, parseTag, rankCandidates, type MatchContext } from './match'

export const REGIONS: Record<number, string> = { 1: 'Americas', 2: 'Europe', 3: 'Asia', 5: 'China' }

let through: { value: string; at: number } | null = null

/** The data's last day (latest last_played in the table), cached for an hour per instance. */
export async function dataThrough(): Promise<string> {
  if (through && Date.now() - through.at < 3_600_000) return through.value
  const r = rowsOf<{ d: string | null }>(await db.execute(sql`SELECT max(last_played)::text AS d FROM personal_accounts`))[0]
  through = { value: r?.d ?? new Date().toISOString().slice(0, 10), at: Date.now() }
  return through.value
}

export interface AccountSummary {
  /** Visible name + #number + region identify the account to the page; blizz ids never leave the server. */
  name: string
  discriminator: number
  region: number
  regionName: string
  mmr: number | null
  lastPlayed: string
  games: number
  recentGames: number
  topHeroes: string[]
}

export interface AccountWithCard extends AccountSummary {
  card: Card
}

function rowsOf<T>(res: unknown): T[] {
  if (Array.isArray(res)) return res as T[]
  return ((res as { rows?: T[] })?.rows ?? []) as T[]
}

function bytesOf(v: unknown): Uint8Array {
  if (v instanceof Uint8Array) return v
  if (typeof v === 'string') {
    const hex = v.startsWith('\\x') ? v.slice(2) : v
    return Uint8Array.from(Buffer.from(hex, 'hex'))
  }
  throw new Error('unexpected bytea value')
}

function toSummary(r: any): AccountSummary {
  const top = bytesOf(r.top_heroes)
  return {
    name: r.name,
    discriminator: Number(r.discriminator),
    region: Number(r.region),
    regionName: REGIONS[Number(r.region)] ?? `Region ${r.region}`,
    mmr: r.mmr === null || r.mmr === undefined ? null : Number(r.mmr),
    lastPlayed: String(r.last_played).slice(0, 10),
    games: Number(r.games),
    recentGames: Number(r.recent_games),
    topHeroes: Array.from(top).map(h => HERO_NAMES[h]),
  }
}

/** Decode a published card; an invalid one is logged and left out rather than failing the request. */
function cardOrNull(v: unknown, where: string): Card | null {
  try {
    return decodeCard(bytesOf(v))
  } catch (err) {
    console.error(`[personal] invalid card skipped (${where}):`, err instanceof Error ? err.message : err)
    return null
  }
}

const COLS = sql`region, name, discriminator, mmr, last_played::text AS last_played, games, recent_games, top_heroes`

/** Every account with this exact BattleTag (any region). */
export async function byTag(name: string, discriminator: number): Promise<AccountWithCard[]> {
  const res = await db.execute(sql`
    SELECT ${COLS}, card FROM personal_accounts
    WHERE lower(name) = lower(${nfc(name)}) AND discriminator = ${discriminator}
    ORDER BY last_played DESC LIMIT 8`)
  return rowsOf<any>(res).flatMap(r => {
    const card = cardOrNull(r.card, `tag region ${r.region}`)
    return card ? [{ ...toSummary(r), card }] : []
  })
}

export { nfc, parseTag } from './match'

export type DraftTerms = ReturnType<typeof draftByName>

export interface MatchResult {
  query: string
  status: 'matched' | 'choose' | 'none'
  /** Up to 4 candidates, best first; [0] is the selection when matched. */
  candidates: (AccountSummary & { draft: DraftTerms })[]
}

/** Every published account with this visible name, most recently active first (name-only search). */
export async function byName(rawName: string, limit = 50): Promise<AccountSummary[]> {
  const name = nfc(rawName).trim()
  if (!/^[^\s#]{2,12}$/u.test(name)) return []
  const res = await db.execute(sql`
    SELECT ${COLS} FROM personal_accounts WHERE lower(name) = lower(${name})
    ORDER BY last_played DESC, games DESC LIMIT ${limit}`)
  return rowsOf<any>(res).map(toSummary)
}

/**
 * Match a visible name (no #number) given the party's context. A full BattleTag is looked up
 * exactly; it is selected without asking unless it exists in several regions and the context
 * does not settle which.
 */
export async function matchName(query: string, ctx: MatchContext): Promise<MatchResult> {
  const name = nfc(query).trim()
  const tag = parseTag(name)
  if (tag) {
    const accts = (await byTag(tag.name, tag.discriminator)).map(({ card, ...a }) => ({ ...a, draft: draftByName(card) }))
    if (accts.length === 0) return { query, status: 'none', candidates: [] }
    const { ranked, clear } = rankCandidates(accts, ctx)
    const ok = accts.length === 1 || clear
    return { query, status: ok ? 'matched' : 'choose', candidates: ranked.slice(0, ok ? 1 : 4).map(x => x.item) }
  }
  if (!/^[^\s#]{2,12}$/u.test(name)) return { query, status: 'none', candidates: [] }
  const res = await db.execute(sql`
    SELECT ${COLS}, blizz_id FROM personal_accounts WHERE lower(name) = lower(${name}) LIMIT 1000`)
  const rows = rowsOf<any>(res)
  if (rows.length === 0) return { query, status: 'none', candidates: [] }
  const { ranked, clear } = rankCandidates(
    rows.map(r => ({ ...r, region: Number(r.region), mmr: r.mmr === null ? null : Number(r.mmr), lastPlayed: String(r.last_played) })),
    ctx,
  )
  const top = ranked.slice(0, clear ? 1 : 4).map(x => x.item)
  const cards = rowsOf<any>(
    await db.execute(sql`
      SELECT region, blizz_id, card FROM personal_accounts
      WHERE (region, blizz_id) IN (${sql.join(top.map(t => sql`(${t.region}, ${t.blizz_id})`), sql`, `)})`),
  )
  const cardOf = new Map(cards.map(c => [`${c.region}:${c.blizz_id}`, cardOrNull(c.card, `match region ${c.region}`)]))
  const candidates = top.flatMap(t => {
    const card = cardOf.get(`${t.region}:${t.blizz_id}`)
    return card ? [{ ...toSummary(t), draft: draftByName(card) }] : []
  })
  if (candidates.length === 0) return { query, status: 'none', candidates: [] }
  // a clear match whose card was invalid is not silently replaced by a weaker candidate
  return { query, status: clear && candidates.length === top.length ? 'matched' : 'choose', candidates }
}

/** Party context: region (most common) and mean MMR of the party accounts found. */
export async function partyContext(
  tags: string[],
): Promise<{ ctx: MatchContext; party: (AccountSummary & { tag: string; draft: DraftTerms })[] }> {
  const party: (AccountSummary & { tag: string; draft: DraftTerms })[] = []
  for (const t of tags.slice(0, 10)) {
    const p = parseTag(t)
    if (!p) continue
    const accts = await byTag(p.name, p.discriminator)
    if (accts.length === 0) continue
    const { card, ...a } = accts[0] // most recently active region
    party.push({ ...a, tag: t, draft: draftByName(card) })
  }
  const counts = new Map<number, number>()
  for (const a of party) counts.set(a.region, (counts.get(a.region) ?? 0) + 1)
  const region = [...counts.entries()].sort((x, y) => y[1] - x[1])[0]?.[0] ?? null
  const mmrs = party.map(a => a.mmr).filter((x): x is number => x !== null)
  const mmr = mmrs.length ? mmrs.reduce((a, b) => a + b, 0) / mmrs.length : null
  return { ctx: { region, mmr, asOf: await dataThrough() }, party }
}
