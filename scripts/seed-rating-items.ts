/**
 * Seed data/rating-items.json into the rating_items table (upsert by id).
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
