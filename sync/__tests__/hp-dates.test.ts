// Run as if on the sync box (America/New_York), whatever the CI host's zone is.
// Node applies a runtime TZ change to later Date calls.
process.env.TZ = 'America/New_York'

import { describe, expect, it } from 'vitest'
import { parseUtcDate, requireUtcDate } from '../hp-dates'

describe('parseUtcDate', () => {
  it('runs under a non-UTC zone, so a local-time parse would be caught', () => {
    // The pre-fix code did new Date(hpString): local time, 4 h late in summer.
    expect(new Date('2026-07-09 20:14:30').toISOString()).toBe('2026-07-10T00:14:30.000Z')
  })

  it('reads zone-less HP strings as UTC in summer and winter', () => {
    expect(parseUtcDate('2026-07-09 20:14:30')?.toISOString()).toBe('2026-07-09T20:14:30.000Z')
    expect(parseUtcDate('2026-01-15 03:04:05')?.toISOString()).toBe('2026-01-15T03:04:05.000Z')
    expect(parseUtcDate('2026-07-09T20:14:30')?.toISOString()).toBe('2026-07-09T20:14:30.000Z')
    expect(parseUtcDate('2026-07-09')?.toISOString()).toBe('2026-07-09T00:00:00.000Z')
  })

  it('is exact across the US DST changes', () => {
    expect(parseUtcDate('2026-03-08 06:30:00')?.toISOString()).toBe('2026-03-08T06:30:00.000Z')
    expect(parseUtcDate('2026-03-08 07:30:00')?.toISOString()).toBe('2026-03-08T07:30:00.000Z')
    expect(parseUtcDate('2026-11-01 05:30:00')?.toISOString()).toBe('2026-11-01T05:30:00.000Z')
    expect(parseUtcDate('2026-11-01 06:30:00')?.toISOString()).toBe('2026-11-01T06:30:00.000Z')
  })

  it('keeps an explicit zone', () => {
    expect(parseUtcDate('2026-07-09T20:14:30Z')?.toISOString()).toBe('2026-07-09T20:14:30.000Z')
    expect(parseUtcDate('2026-07-09T16:14:30-04:00')?.toISOString()).toBe('2026-07-09T20:14:30.000Z')
    expect(parseUtcDate('2026-07-09 22:14:30+0200')?.toISOString()).toBe('2026-07-09T20:14:30.000Z')
  })

  it('returns null for empty or bad input and passes Dates through', () => {
    expect(parseUtcDate(null)).toBeNull()
    expect(parseUtcDate(undefined)).toBeNull()
    expect(parseUtcDate('')).toBeNull()
    expect(parseUtcDate('not a date')).toBeNull()
    const d = new Date(Date.UTC(2026, 6, 9))
    expect(parseUtcDate(d)).toBe(d)
  })

  it('requireUtcDate throws on missing input', () => {
    expect(() => requireUtcDate(undefined)).toThrow(/game_date/)
    expect(requireUtcDate('2026-07-09 20:14:30').toISOString()).toBe('2026-07-09T20:14:30.000Z')
  })
})
