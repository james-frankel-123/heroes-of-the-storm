// Heroes Profile sends game_date as a zone-less UTC string ("2026-07-09 20:14:30").
// `new Date()` reads such a string as HOST LOCAL time, which on this box
// (America/New_York) stored every game 4-5 h late. See sync/docs/timezone-fix-2026-10.md.
// The same holds for timestamp-without-time-zone values read back from Postgres
// as text (drizzle raw `execute`): they are UTC wall clock.

const HAS_ZONE = /(?:Z|[+-]\d{2}(?::?\d{2})?)$/i
const DATE_ONLY = /^\d{4}-\d{2}-\d{2}$/

/**
 * Parse an HP (or DB timestamp-without-time-zone) date string as UTC.
 * Strings that carry an explicit offset or 'Z' keep it. Returns null for
 * empty or unparseable input; never depends on the host time zone.
 */
export function parseUtcDate(value: unknown): Date | null {
  if (value === null || value === undefined) return null
  if (value instanceof Date) return isNaN(value.getTime()) ? null : value
  const s = String(value).trim()
  if (!s) return null
  let iso = s.replace(' ', 'T')
  if (DATE_ONLY.test(iso)) iso += 'T00:00:00'
  if (!HAS_ZONE.test(iso)) iso += 'Z'
  const d = new Date(iso)
  return isNaN(d.getTime()) ? null : d
}

/** parseUtcDate that throws on missing or unparseable input (NOT NULL columns). */
export function requireUtcDate(value: unknown, what = 'game_date'): Date {
  const d = parseUtcDate(value)
  if (!d) throw new Error(`unparseable ${what}: ${JSON.stringify(value)}`)
  return d
}
