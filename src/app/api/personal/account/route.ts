import { NextResponse } from 'next/server'
import { byName, byTag, dataThrough, parseTag } from '@/lib/personal/accounts'
import { ARTIFACT, NO_ESTIMATE_HEROES } from '@/lib/personal/card'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

/**
 * GET /api/personal/account?tag=Name#1234, or ?name=Name (every account with that visible name)
 * Published cards for every account with this BattleTag (one per region). Reads Neon only;
 * no Heroes Profile call is ever made from the site (docs/personalization-site-design.md).
 */
export async function GET(req: Request) {
  try {
    return await handle(req)
  } catch (err) {
    // database or decoder failure: a controlled JSON error, details only in the server log
    console.error('[personal/account] failed:', err)
    return NextResponse.json(
      { error: 'Player data is unavailable right now. Try again in a minute.' },
      { status: 503, headers: { 'Cache-Control': 'no-store' } },
    )
  }
}

async function handle(req: Request): Promise<Response> {
  const sp = new URL(req.url).searchParams
  const rawName = sp.get('name')
  const name = rawName === null ? null : rawName.normalize('NFC')
  if (name !== null) {
    if (!/^[^\s#]{2,12}$/u.test(name.trim())) {
      return NextResponse.json({ error: 'Enter a name of 2 to 12 characters.' }, { status: 400 })
    }
    return NextResponse.json(
      { through: await dataThrough(), estimatesAsOf: ARTIFACT.estimatesAsOf, matches: await byName(name) },
      { headers: { 'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400' } },
    )
  }
  const tag = sp.get('tag') ?? ''
  const p = parseTag(tag)
  if (!p) return NextResponse.json({ error: 'Enter a full BattleTag, like Name#1234.' }, { status: 400 })
  const accounts = await byTag(p.name, p.discriminator)
  return NextResponse.json(
    { through: await dataThrough(), estimatesAsOf: ARTIFACT.estimatesAsOf, accounts, validation: ARTIFACT.validation, noEstimate: NO_ESTIMATE_HEROES },
    { headers: { 'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400' } },
  )
}
