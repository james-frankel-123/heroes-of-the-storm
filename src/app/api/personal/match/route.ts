import { NextResponse } from 'next/server'
import { DATA_THROUGH, matchName, partyContext, REGIONS } from '@/lib/personal/accounts'
import { draftCoefs } from '@/lib/personal/card'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

/**
 * GET /api/personal/match?names=a,b,c&party=Name#1234,Other#5678[&region=1]
 * In-game mode of the draft page: match visible names (no #number) against the published
 * name index, using the party's region and MMR as context. Reads Neon only.
 */
export async function GET(req: Request) {
  const sp = new URL(req.url).searchParams
  const names = (sp.get('names') ?? '').split(',').map(s => s.trim()).filter(Boolean).slice(0, 9)
  const partyTags = (sp.get('party') ?? '').split(',').map(s => s.trim()).filter(Boolean).slice(0, 5)
  if (names.length === 0 && partyTags.length === 0) {
    return NextResponse.json({ error: 'names or party required' }, { status: 400 })
  }
  const { ctx, party } = await partyContext(partyTags)
  const region = Number(sp.get('region'))
  if (REGIONS[region]) ctx.region = region
  const results = await Promise.all(names.map(n => matchName(n, ctx)))
  return NextResponse.json(
    { through: DATA_THROUGH, context: ctx, party, results, coefs: draftCoefs() },
    { headers: { 'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400' } },
  )
}
