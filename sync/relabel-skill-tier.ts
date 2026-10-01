/**
 * One-off relabel of replay_draft_data.skill_tier into the site's scheme
 * (low = Bronze+Silver, mid = Gold+Platinum, high = Diamond+Master), the
 * same rule as leagueTierToSkillTier in sync/sync-replays.ts.
 *
 * Rows written before 2026-09-30 used a shifted scheme in which a NULL
 * league_tier (Master games) fell through to 'mid'. Pinned research
 * snapshots reference those old labels, so --backup must run first:
 *   - table replay_draft_skill_tier_backup_20260930 (replay_id, skill_tier,
 *     league_tier, avg_mmr) in the same database
 *   - backups/replay_draft_skill_tier_20260930.csv.gz
 *
 * Usage (in order):
 *   npx tsx sync/relabel-skill-tier.ts --backup
 *   npx tsx sync/relabel-skill-tier.ts --apply    # refuses without a full backup
 *   npx tsx sync/relabel-skill-tier.ts --verify
 * --apply is idempotent (a pure function of league_tier and avg_mmr).
 */
import { createWriteStream } from 'fs'
import { join } from 'path'
import { createGzip } from 'zlib'
import { sql } from 'drizzle-orm'
import { createDb, SyncDb } from './db'
import { log } from './logger'

const BACKUP_TABLE = 'replay_draft_skill_tier_backup_20260930'
const BACKUP_CSV = join(__dirname, '..', 'backups', 'replay_draft_skill_tier_20260930.csv.gz')
const BATCH = 200_000

const NEW_TIER = sql.raw(`CASE
  WHEN league_tier IS NULL THEN CASE WHEN avg_mmr IS NULL THEN 'unknown' ELSE 'high' END
  WHEN league_tier <= 3 THEN 'low'
  WHEN league_tier <= 5 THEN 'mid'
  ELSE 'high' END`)

async function count(db: SyncDb, table: string): Promise<number> {
  const [r] = (await db.execute(sql.raw(`SELECT count(*) AS n FROM ${table}`))).rows as any[]
  return Number(r.n)
}

async function backup(db: SyncDb): Promise<void> {
  await db.execute(sql.raw(`CREATE TABLE IF NOT EXISTS ${BACKUP_TABLE} AS
    SELECT replay_id, skill_tier, league_tier, avg_mmr, now() AS backed_up_at FROM replay_draft_data`))
  const [n, total] = [await count(db, BACKUP_TABLE), await count(db, 'replay_draft_data')]
  log.info(`backup table ${BACKUP_TABLE}: ${n.toLocaleString()} rows (replay_draft_data has ${total.toLocaleString()})`)

  const gz = createGzip()
  const out = createWriteStream(BACKUP_CSV)
  gz.pipe(out)
  gz.write('replay_id,skill_tier,league_tier,avg_mmr\n')
  let after = -1
  let written = 0
  for (;;) {
    const rows = (await db.execute(sql.raw(`SELECT replay_id, skill_tier, league_tier, avg_mmr FROM ${BACKUP_TABLE}
      WHERE replay_id > ${after} ORDER BY replay_id LIMIT ${BATCH}`))).rows as any[]
    if (rows.length === 0) break
    gz.write(rows.map(r => `${r.replay_id},${r.skill_tier},${r.league_tier ?? ''},${r.avg_mmr ?? ''}`).join('\n') + '\n')
    written += rows.length
    after = Number(rows[rows.length - 1].replay_id)
  }
  await new Promise<void>(res => { out.on('finish', () => res()); gz.end() })
  log.info(`backup CSV ${BACKUP_CSV}: ${written.toLocaleString()} rows`)
}

async function apply(db: SyncDb): Promise<void> {
  const exists = (await db.execute(sql`SELECT to_regclass(${BACKUP_TABLE}) IS NOT NULL AS ok`)).rows as any[]
  if (!exists[0]?.ok) throw new Error(`no ${BACKUP_TABLE}; run --backup first`)
  const [missing] = (await db.execute(sql.raw(`SELECT count(*) AS n FROM replay_draft_data d
    WHERE NOT EXISTS (SELECT 1 FROM ${BACKUP_TABLE} b WHERE b.replay_id = d.replay_id)
      AND d.fetched_at < (SELECT min(backed_up_at) FROM ${BACKUP_TABLE})`))).rows as any[]
  if (Number(missing.n) > 0) throw new Error(`${missing.n} pre-backup rows are missing from ${BACKUP_TABLE}; refusing`)

  const [{ lo, hi }] = (await db.execute(sql`SELECT min(replay_id) AS lo, max(replay_id) AS hi FROM replay_draft_data`)).rows as any[]
  let changed = 0
  for (let start = Number(lo); start <= Number(hi); start += BATCH) {
    const res: any = await db.execute(sql`UPDATE replay_draft_data SET skill_tier = ${NEW_TIER}
      WHERE replay_id >= ${start} AND replay_id < ${start + BATCH} AND skill_tier IS DISTINCT FROM ${NEW_TIER}`)
    changed += Number(res.rowCount ?? 0)
  }
  log.info(`relabelled ${changed.toLocaleString()} rows`)
}

async function verify(db: SyncDb): Promise<void> {
  const [bad] = (await db.execute(sql`SELECT count(*) AS n FROM replay_draft_data WHERE skill_tier IS DISTINCT FROM ${NEW_TIER}`)).rows as any[]
  const dist = (await db.execute(sql`SELECT skill_tier, count(*) AS n FROM replay_draft_data GROUP BY 1 ORDER BY 1`)).rows as any[]
  const moves = (await db.execute(sql.raw(`SELECT b.skill_tier AS old, d.skill_tier AS new, count(*) AS n
    FROM replay_draft_data d JOIN ${BACKUP_TABLE} b USING (replay_id) GROUP BY 1, 2 ORDER BY 1, 2`))).rows as any[]
  console.log(`rows not matching the rule: ${bad.n}`)
  console.log('current:', dist.map(r => `${r.skill_tier}=${Number(r.n).toLocaleString()}`).join(' '))
  console.log('old -> new:', moves.map(r => `${r.old}->${r.new} ${Number(r.n).toLocaleString()}`).join(' | '))
}

async function main(): Promise<void> {
  const db = createDb()
  if (process.argv.includes('--backup')) await backup(db)
  else if (process.argv.includes('--apply')) await apply(db)
  else if (process.argv.includes('--verify')) await verify(db)
  else throw new Error('pass --backup, --apply or --verify')
  process.exit(0)
}

main().catch(err => { log.error(String(err?.stack ?? err)); process.exit(1) })
