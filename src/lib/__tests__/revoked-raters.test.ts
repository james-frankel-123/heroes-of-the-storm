import { describe, expect, it } from 'vitest'
import { isRevokedRater } from '@/lib/rating/assignment'

describe('revoked rater links', () => {
  it('blocks a withdrawn code name in any case or spacing', () => {
    expect(isRevokedRater('rater05')).toBe(true)
    expect(isRevokedRater(' RATER05 ')).toBe(true)
  })

  it('leaves active raters alone', () => {
    for (const r of ['rater01', 'rater04', 'rater06', 'rater13', 'rater15', 'testtenth-rater05']) {
      expect(isRevokedRater(r)).toBe(false)
    }
  })
})
