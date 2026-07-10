/**
 * Seed data/rating-items.json into the rating_items table.
 *
 * Reseeding an item-pool regeneration (new seed / new counts) REPLACES the
 * table: any is_test pilot ratings referencing old item ids are deleted first
 * (they are disposable smoke-test rows; deleting them avoids FK orphans), then
 * all rating_items rows are dropped and the new pool inserted. The script
 * ABORTS if any non-test rating exists — real study data must never be
 * orphaned or deleted by a reseed.
 *
 * Usage: set -a && source .env && set +a && npx tsx scripts/seed-rating-items.ts
 */
import fs from 'node:fs'
import path from 'node:path'
import { neon } from '@neondatabase/serverless'

async function main() {
  if (!process.env.DATABASE_URL) {
    console.error('DATABASE_URL not set. Run: set -a && source .env && set +a')
    process.exit(1)
  }
  const sql = neon(process.env.DATABASE_URL)
  const file = path.resolve(__dirname, '../data/rating-items.json')
  const { items } = JSON.parse(fs.readFileSync(file, 'utf8'))

  // Safety: never reseed over real (non-test) study data.
  const [{ count: realCount }] = (await sql`
    select count(*) as count from draft_ratings where is_test = false
  `) as { count: string }[]
  if (Number(realCount) > 0) {
    console.error(
      `ABORT: ${realCount} non-test ratings exist; reseeding would orphan real study data.`
    )
    process.exit(1)
  }
  const [{ count: testCount }] = (await sql`
    select count(*) as count from draft_ratings where is_test = true
  `) as { count: string }[]
  await sql`delete from draft_ratings where is_test = true`
  await sql`delete from rating_items`
  console.log(
    `cleared rating_items and ${testCount} disposable is_test pilot ratings (FK safety)`
  )

  for (const it of items) {
    const block = it.block ?? 'core'
    await sql`
      insert into rating_items (id, block, teams, map, tier, provenance)
      values (${it.id}, ${block}, ${JSON.stringify(it.teams)}::jsonb, ${it.map}, ${it.tier}, ${JSON.stringify(it.provenance)}::jsonb)
      on conflict (id) do update set
        block = excluded.block,
        teams = excluded.teams,
        map = excluded.map,
        tier = excluded.tier,
        provenance = excluded.provenance
    `
  }
  const [{ count }] = (await sql`select count(*) from rating_items`) as { count: string }[]
  console.log(`seeded ${items.length} items; rating_items now has ${count} rows`)
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
