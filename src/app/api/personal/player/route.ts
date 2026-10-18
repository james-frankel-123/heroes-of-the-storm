import { NextResponse } from 'next/server'
import { allowIp, player } from '@/lib/personal/lookup'
import { ARTIFACT } from '@/lib/personal/model'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const maxDuration = 30

function ipOf(req: Request): string {
  return req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || 'unknown'
}

/**
 * GET /api/personal/player?tag=Name#1234&region=1
 * Per-hero skill cards for one account (docs/personalization-site-design.md).
 */
export async function GET(req: Request) {
  const sp = new URL(req.url).searchParams
  const tag = sp.get('tag')?.trim() ?? ''
  const region = Number(sp.get('region'))
  if (!tag || !Number.isInteger(region)) {
    return NextResponse.json({ status: 'not_found', message: 'tag and region are required' }, { status: 400 })
  }
  if (!allowIp(ipOf(req))) {
    return NextResponse.json(
      { status: 'unavailable', message: 'Too many lookups from this address. Try again in a few minutes.' },
      { status: 429 },
    )
  }
  const res = await player(tag, region)
  const code = res.status === 'ok' ? 200 : res.status === 'unavailable' ? 503 : res.status === 'private' ? 403 : 404
  return NextResponse.json({ ...res, validation: ARTIFACT.validation }, { status: code, headers: { 'Cache-Control': 'private, max-age=300' } })
}
