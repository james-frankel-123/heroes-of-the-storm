/**
 * Publish the website's tables from the local primary store to Neon.
 *
 * Only SITE_TABLES (sync/table-groups.ts) are ever written; the study tables
 * and everything else are refused. Per table:
 *  - skip when local and Neon already hold the same rows (count + content
 *    hash, computed server-side; nothing bulk leaves Neon);
 *  - gate: refuse when the local count fell >20% below Neon's (or is 0), so a
 *    broken sync cannot blank the site;
 *  - otherwise replace the Neon rows. All changed tables swap in ONE Neon
 *    transaction, so the site sees the old set or the new set, never a mix.
 * Values travel as Postgres text (no JS type parsing), so they round-trip exactly.
 *
 * Usage (source .env): npx tsx sync/publish-site.ts [--dry-run]
 * Runs at the end of run-sync.sh and after each privacy-sync (crontab).
 */
import pg from 'pg'
import { createPool } from './db'
import { SITE_TABLES } from './table-groups'
import { log } from './logger'

const GATE_DROP = 0.2
const BATCH_ROWS = 1000
const RAW: pg.CustomTypesConfig = { getTypeParser: () => (v: string) => v } as any

async function q(c: pg.PoolClient | pg.Pool, text: string, values?: unknown[]) {
  return c.query({ text, values, types: RAW })
}

async function columnsOf(c: pg.Pool, table: string): Promise<string[]> {
  const r = await q(c, `SELECT column_name FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = $1 ORDER BY ordinal_position`, [table])
  return r.rows.map((x: any) => x.column_name)
}

async function fingerprint(c: pg.Pool, table: string, cols: string[]) {
  const row = `ROW(${cols.map(x => `"${x}"`).join(', ')})::text`
  const r = await q(c, `SELECT count(*) AS n, coalesce(sum(hashtextextended(${row}, 0)), 0)::text AS h
    FROM public."${table}"`)
  return { n: Number(r.rows[0].n), h: String(r.rows[0].h) }
}

async function main() {
  const dryRun = process.argv.includes('--dry-run')
  const localUrl = process.env.DATABASE_URL
  const neonUrl = process.env.NEON_DATABASE_URL
  if (!localUrl || !neonUrl) throw new Error('DATABASE_URL (local) and NEON_DATABASE_URL required')
  if (/neon\.tech/.test(localUrl)) throw new Error('DATABASE_URL points at Neon; publish reads the local store')

  const local = createPool(localUrl, 2)
  const neon = createPool(neonUrl, 2)

  const plan: { table: string; cols: string[]; rows: any[] }[] = []
  let gated = 0
  for (const table of SITE_TABLES) {
    const cols = await columnsOf(neon, table)
    const localCols = new Set(await columnsOf(local, table))
    const missing = cols.filter(c => !localCols.has(c))
    if (missing.length) throw new Error(`${table}: local store lacks columns ${missing.join(', ')}`)
    // Site tables have no timestamptz columns, so row text is session-TimeZone independent.
    const [lf, nf] = await Promise.all([fingerprint(local, table, cols), fingerprint(neon, table, cols)])
    if (lf.n === nf.n && lf.h === nf.h) {
      log.info(`  ${table}: unchanged (${lf.n} rows)`)
      continue
    }
    if (lf.n === 0 || lf.n < nf.n * (1 - GATE_DROP)) {
      log.error(`  ${table}: GATED, local ${lf.n} rows vs Neon ${nf.n}; not published`)
      gated++
      continue
    }
    const r = await q(local, `SELECT ${cols.map(c => `"${c}"`).join(', ')} FROM public."${table}"`)
    plan.push({ table, cols, rows: r.rows })
    log.info(`  ${table}: ${nf.n} -> ${lf.n} rows${dryRun ? ' (dry run)' : ''}`)
  }

  if (!dryRun && plan.length) {
    const c = await neon.connect()
    try {
      await c.query('BEGIN')
      for (const { table, cols, rows } of plan) {
        await c.query(`DELETE FROM public."${table}"`)
        const colList = cols.map(x => `"${x}"`).join(', ')
        for (let i = 0; i < rows.length; i += BATCH_ROWS) {
          const chunk = rows.slice(i, i + BATCH_ROWS)
          const values: unknown[] = []
          const tuples = chunk.map(row => `(${cols.map(col => { values.push(row[col]); return `$${values.length}` }).join(', ')})`)
          await c.query(`INSERT INTO public."${table}" (${colList}) VALUES ${tuples.join(', ')}`, values)
        }
      }
      await c.query('COMMIT')
      log.info(`published ${plan.length} table(s) to Neon`)
    } catch (err) {
      await c.query('ROLLBACK').catch(() => {})
      throw err
    } finally {
      c.release()
    }
  } else {
    log.info(dryRun ? `dry run: ${plan.length} table(s) would change` : 'Neon already current')
  }
  await Promise.all([local.end(), neon.end()])
  if (gated) process.exit(2)
}

main().catch(err => {
  log.error(`publish-site failed: ${err?.message ?? err}`)
  process.exit(1)
})
