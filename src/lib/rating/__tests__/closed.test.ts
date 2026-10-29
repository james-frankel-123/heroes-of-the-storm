import { describe, expect, it, vi } from 'vitest'

// Any database access fails the test: a closed study must answer before touching it.
vi.mock('@/lib/db', () => ({
  db: new Proxy({}, { get: () => { throw new Error('database touched') } }),
}))

describe('closed survey', () => {
  it('rejects a submission with 410 before reading the body or the database', async () => {
    const { POST } = await import('@/app/api/ratings/route')
    const body = { rater: 'x', itemId: 1, pTeamA: 50, betterTeam: 'A', confidence: 3 }
    const res = await POST(new Request('http://t/api/ratings', { method: 'POST', body: JSON.stringify(body) }))
    expect(res.status).toBe(410)
    expect((await res.json()).error).toMatch(/closed on October 26, 2026/)
    const bad = await POST(new Request('http://t/api/ratings', { method: 'POST', body: 'not json' }))
    expect(bad.status).toBe(410)
  })

  it('serves no items and rejects other write methods', async () => {
    const items = await import('@/app/api/ratings/items/route')
    expect((await items.GET(new Request('http://t/api/ratings/items?rater=a'))).status).toBe(410)
    const r = await import('@/app/api/ratings/route')
    expect((await r.PUT()).status).toBe(410)
    expect((await r.DELETE()).status).toBe(410)
  })
})
