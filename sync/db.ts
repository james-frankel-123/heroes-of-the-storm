import { drizzle } from 'drizzle-orm/node-postgres'
import pg from 'pg'
import * as schema from '../src/lib/db/schema'

// Sync jobs write to the local primary store (DATABASE_URL on this box, see
// sync/docs/data-architecture-plan.md). node-postgres works for both the local
// store and Neon (NEON_DATABASE_URL, used only by publish/study tooling).

function needsSsl(url: string): boolean {
  return /sslmode=(require|verify)/.test(url) || /\.neon\.tech/.test(url)
}

export function createPool(url: string, max = 8): pg.Pool {
  const ssl = needsSsl(url)
  // TLS is configured below (verify-full); drop libpq-only URL params pg warns about.
  const connectionString = ssl
    ? url.replace(/([?&])(sslmode|channel_binding)=[^&]*&?/g, '$1').replace(/[?&]$/, '')
    : url
  return new pg.Pool({
    connectionString,
    max,
    ssl: ssl ? { rejectUnauthorized: true } : undefined,
    // Let one-shot scripts exit without an explicit pool.end().
    allowExitOnIdle: true,
  })
}

export function createDb(url: string | undefined = process.env.DATABASE_URL) {
  if (!url) {
    throw new Error('DATABASE_URL environment variable is required')
  }
  return drizzle(createPool(url), { schema })
}

export type SyncDb = ReturnType<typeof createDb>

/** Neon connection for the site publish step and the expert-study tables. */
export function createNeonDb() {
  const url = process.env.NEON_DATABASE_URL
  if (!url) throw new Error('NEON_DATABASE_URL environment variable is required')
  return createDb(url)
}

/**
 * Tagged-template query helper with the same call shape as neon()'s
 * (`await sql\`select ... ${x}\`` resolves to the row array), on node-postgres.
 */
export function pgSql(url: string) {
  const pool = createPool(url, 2)
  return async function sql<T = any>(strings: TemplateStringsArray, ...values: unknown[]): Promise<T[]> {
    const text = strings.reduce((acc, s, i) => acc + (i ? `$${i}` : '') + s, '')
    return (await pool.query(text, values as any[])).rows as T[]
  }
}
