import { NextResponse } from 'next/server'
import { dataThrough, matchName, parseTag, partyContext, REGIONS } from '@/lib/personal/accounts'
import { ARTIFACT, draftCoefs, NO_ESTIMATE_HEROES } from '@/lib/personal/card'
import { lobbyRequestError, splitList } from '@/lib/personal/match'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

/**
 * GET /api/personal/match?names=a,b,Name#1234&party=Name#1234,Other#5678[&region=1]
 * In-game mode of the draft page: match visible names (no #number) or full BattleTags against
 * the published name index, using the party's (and any full tags') region and MMR as context.
 * At most 10 names; more is a 400, never a silent cut. Reads Neon only.
 */
export async function GET(req: Request) {
  try {
    return await handle(req)
  } catch (err) {
    // database or decoder failure: a controlled JSON error, details only in the server log
    console.error('[personal/match] failed:', err)
    return NextResponse.json(
      { error: 'Player data is unavailable right now. Try again in a minute.' },
      { status: 503, headers: { 'Cache-Control': 'no-store' } },
    )
  }
}

async function handle(req: Request): Promise<Response> {
  const sp = new URL(req.url).searchParams
  const names = splitList(sp.get('names'))
  const partyTags = splitList(sp.get('party'))
  const err = lobbyRequestError(names, partyTags)
  if (err) return NextResponse.json({ error: err }, { status: 400 })
  const regionParam = sp.get('region')
  const region = regionParam ? Number(regionParam) : null
  if (region !== null && !REGIONS[region]) {
    return NextResponse.json({ error: 'Unknown region.' }, { status: 400 })
  }
  // Context from the registered party plus any full BattleTags typed in the lobby;
  // only the registered party is returned as `party` (typed tags come back as results).
  const typedTags = names.filter(n => parseTag(n) !== null)
  const { ctx, party: all } = await partyContext([...partyTags, ...typedTags])
  const party = all.filter(p => partyTags.includes(p.tag))
  if (region !== null) ctx.region = region
  const results = await Promise.all(names.map(n => matchName(n, ctx)))
  return NextResponse.json(
    { through: await dataThrough(), estimatesAsOf: ARTIFACT.estimatesAsOf, noEstimate: NO_ESTIMATE_HEROES, context: ctx, party, results, coefs: draftCoefs() },
    { headers: { 'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400' } },
  )
}
