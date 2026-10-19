import { NextResponse } from 'next/server'
import { byTag, DATA_THROUGH, parseTag } from '@/lib/personal/accounts'
import { ARTIFACT } from '@/lib/personal/card'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

/**
 * GET /api/personal/account?tag=Name#1234
 * Published cards for every account with this BattleTag (one per region). Reads Neon only;
 * no Heroes Profile call is ever made from the site (docs/personalization-site-design.md).
 */
export async function GET(req: Request) {
  const tag = new URL(req.url).searchParams.get('tag') ?? ''
  const p = parseTag(tag)
  if (!p) return NextResponse.json({ error: 'Enter a full BattleTag, like Name#1234.' }, { status: 400 })
  const accounts = await byTag(p.name, p.discriminator)
  return NextResponse.json(
    { through: DATA_THROUGH, accounts, validation: ARTIFACT.validation },
    { headers: { 'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400' } },
  )
}
