/**
 * Map and hero-map aggregates from our own replay corpus (replay_draft_data).
 *
 * map_stats_aggregate drives the site's map lists (getAllMaps: the draft
 * page's map picker, the maps page). It used to be filled once from Heroes
 * Profile's hero-map endpoint (sync/run-once.ts), which the HP v1 API no
 * longer serves in bulk, so it was frozen at 2026-03-03: Haunted Mines
 * (in ranked rotation since 2026-07-20) never appeared, and Hanamura Temple
 * and Warhead Junction (out since the same day) never left.
 *
 * Now both tables are rebuilt daily from ranked games in the last
 * WINDOW_DAYS, so they follow the rotation: a map with no games in the window
 * drops out of the lists. games = actual games (map table) and hero-games
 * (hero-map table) in our sample, per site tier.
 *
 * Run standalone: npx tsx sync/derive-map-stats.ts [--dry-run]
 */
import { sql } from 'drizzle-orm'
import { createDb, type SyncDb } from './db'
import { log } from './logger'

export const WINDOW_DAYS = 60

const GAMES = sql.raw(`
  SELECT game_map AS map, skill_tier AS tier, winner,
         team0_heroes::jsonb AS t0, team1_heroes::jsonb AS t1
  FROM replay_draft_data
  WHERE game_date > now() - interval '${WINDOW_DAYS} days'
    AND skill_tier IN ('low', 'mid', 'high')
    AND winner IN (0, 1)
    AND game_map IS NOT NULL
    AND jsonb_array_length(team0_heroes::jsonb) = 5
    AND jsonb_array_length(team1_heroes::jsonb) = 5`)

export async function deriveMapStats(
  db: SyncDb, dryRun = false,
): Promise<{ maps: number; heroMaps: number }> {
  log.info(`Deriving map stats from replay_draft_data (last ${WINDOW_DAYS} days)`)
  if (dryRun) {
    const r: any = await db.execute(sql`
      WITH g AS (${GAMES})
      SELECT map, tier, count(*)::int AS games FROM g GROUP BY map, tier ORDER BY map, tier`)
    for (const row of r.rows) log.info(`  ${row.map} / ${row.tier}: ${row.games}`)
    return { maps: r.rows.length, heroMaps: 0 }
  }
  // cutoff from the DB clock (rows written below get now() > cutoff)
  const t: any = await db.execute(sql`SELECT clock_timestamp()::timestamp::text AS t`)
  const cutoff: string = t.rows[0].t

  const maps = await db.execute(sql`
    WITH g AS (${GAMES})
    INSERT INTO map_stats_aggregate (map, skill_tier, games, updated_at)
    SELECT map, tier::skill_tier, count(*), now() FROM g GROUP BY map, tier
    ON CONFLICT (map, skill_tier) DO UPDATE
      SET games = excluded.games, updated_at = excluded.updated_at`)

  const heroMaps = await db.execute(sql`
    WITH g AS (${GAMES}),
    x AS (
      SELECT h.hero, g.map, g.tier, (g.winner = 0)::int AS won
      FROM g, jsonb_array_elements_text(g.t0) AS h(hero)
      UNION ALL
      SELECT h.hero, g.map, g.tier, (g.winner = 1)::int AS won
      FROM g, jsonb_array_elements_text(g.t1) AS h(hero)
    )
    INSERT INTO hero_map_stats_aggregate (hero, map, skill_tier, games, wins, win_rate, updated_at)
    SELECT hero, map, tier::skill_tier, count(*), sum(won),
           round(100.0 * sum(won) / count(*), 2), now()
    FROM x GROUP BY hero, map, tier
    ON CONFLICT (hero, map, skill_tier) DO UPDATE
      SET games = excluded.games, wins = excluded.wins,
          win_rate = excluded.win_rate, updated_at = excluded.updated_at`)

  const nMaps = (maps as any).rowCount ?? 0
  const nHeroMaps = (heroMaps as any).rowCount ?? 0
  if (nMaps === 0) {
    // never empty the lists because a query matched nothing
    log.warn('No games in the window; leaving map tables unchanged')
    return { maps: 0, heroMaps: 0 }
  }
  // rows not refreshed this run: maps out of rotation, heroes unplayed there
  await db.execute(sql`DELETE FROM map_stats_aggregate WHERE updated_at < ${cutoff}::timestamp`)
  await db.execute(sql`DELETE FROM hero_map_stats_aggregate WHERE updated_at < ${cutoff}::timestamp`)
  log.info(`  map_stats_aggregate: ${nMaps} rows; hero_map_stats_aggregate: ${nHeroMaps} rows`)
  return { maps: nMaps, heroMaps: nHeroMaps }
}

if (process.argv[1]?.endsWith('derive-map-stats.ts')) {
  deriveMapStats(createDb(), process.argv.includes('--dry-run'))
    .then(r => { log.info(`done: ${JSON.stringify(r)}`); process.exit(0) })
    .catch(err => { log.error('derive-map-stats failed', err); process.exit(1) })
}
