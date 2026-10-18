import { NextResponse } from 'next/server'
import { allowIp, search } from '@/lib/personal/lookup'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

function ipOf(req: Request): string {
  return req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || 'unknown'
}

/**
 * GET /api/personal/search?q=Name#1234 (or q=Name)
 * Candidates across regions for the player lookup (docs/personalization-site-design.md).
 */
export async function GET(req: Request) {
  const q = new URL(req.url).searchParams.get('q')?.trim() ?? ''
  if (q.length < 2 || q.length > 24) {
    return NextResponse.json({ error: 'Enter a BattleTag like Name#1234.' }, { status: 400 })
  }
  if (q.includes('#') && !allowIp(ipOf(req))) {
    return NextResponse.json({ error: 'Too many lookups from this address. Try again in a few minutes.' }, { status: 429 })
  }
  const res = await search(q)
  return NextResponse.json(res, { headers: { 'Cache-Control': 'private, max-age=60' } })
}
