#!/usr/bin/env bash
# Cheap pipeline status: planner estimates + small exact tables only.
# Never scans replay_players (16M rows; exact coverage counts take minutes
# on a cold endpoint). Usage: bash sync/status.sh
set -euo pipefail
cd /home/max/heroes-of-the-storm
set -a; . ./.env; set +a
npx tsx --eval "
import { createDb } from './sync/db'
import { sql } from 'drizzle-orm'
async function main() {
  const db = createDb()
  const est = await db.execute(sql\`SELECT relname, reltuples::bigint r FROM pg_class
    WHERE relname IN ('replay_players','replay_draft_data','qm_games','player_fetch_queue','draft_ratings')\`)
  const m = Object.fromEntries(est.rows.map((x: any) => [x.relname, Number(x.r)]))
  console.log('replays covered (est):', Math.round(m.replay_players / 10).toLocaleString(),
              'of', m.replay_draft_data.toLocaleString(), 'drafts')
  console.log('qm games (est):', m.qm_games.toLocaleString())
  const q = await db.execute(sql\`SELECT count(*) n FROM player_fetch_queue\`)
  const marks = await db.execute(sql\`SELECT count(*) n FROM player_history_marks\`)
  console.log('queue pending:', Number(q.rows[0].n).toLocaleString(),
              '| players enumerated:', Number(marks.rows[0].n).toLocaleString())
  // draft_ratings is tiny and correctness-critical (prereg): count exactly.
  const dr = await db.execute(sql\\`SELECT count(*) n, count(*) FILTER (WHERE NOT is_test) real FROM draft_ratings\\`)
  console.log('ratings collected:', Number(dr.rows[0].real).toLocaleString(),
              'real,', Number(dr.rows[0].n).toLocaleString(), 'total')
  process.exit(0)
}
main()
"
