import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// Basic-auth gate for the roadmap page only. The password lives in the
// ROADMAP_PASSWORD environment variable (set in Vercel), never in source.
// If the env var is unset we fail open (the page was public before) rather
// than lock everyone out.
export function middleware(req: NextRequest) {
  const expected = process.env.ROADMAP_PASSWORD?.trim()
  if (!expected) return NextResponse.next()

  const header = req.headers.get('authorization')
  if (header?.startsWith('Basic ')) {
    try {
      const decoded = atob(header.slice(6))
      const password = decoded.slice(decoded.indexOf(':') + 1)
      if (password === expected) return NextResponse.next()
    } catch {
      // fall through to 401
    }
  }

  return new NextResponse('Authentication required.', {
    status: 401,
    headers: {
      'WWW-Authenticate': 'Basic realm="HotS Fever Roadmap", charset="UTF-8"',
    },
  })
}

// Every roadmap URL, including the static files the rewrites point at — those are
// reachable directly, so gating only the pretty URL would leave the document open.
// /roadmap/:path* is a catch-all so a future roadmap page is gated by default
// rather than by remembering to add it here.
export const config = {
  matcher: [
    '/roadmap',
    '/roadmap.html',
    '/roadmap/:path*',
    '/analysis-roadmap.html',
  ],
}
