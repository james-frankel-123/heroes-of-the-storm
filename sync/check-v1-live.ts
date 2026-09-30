/**
 * One-shot check to run right after the HP account is migrated to live v1
 * data. Resolves what the v1 docs leave open, before HP_API=v2 goes into cron:
 *
 *  1. heroes/stats row fields (sync-global needs games_played, wins,
 *     win_rate, ban_rate, popularity).
 *  2. heroes/matchups `enemy` perspective: correlates v1 enemy win rates for
 *     one hero with that hero's win rate vs each opponent in our own
 *     replay_draft_data. Positive => rows are the hero's; negative => the
 *     opponent's (the legacy convention sync-global inverts).
 *  3. /replays listing fields (league_tier, avg_mmr, game_version).
 *
 * Costs ~3 calls. Writes nothing.
 *   npx tsx sync/check-v1-live.ts [hero]
 */
import { sql } from 'drizzle-orm'
import { HeroesProfileApiV2 } from './api-client-v2'
import { createDb } from './db'

function pearson(xs: number[], ys: number[]): number {
  const n = xs.length
  const mx = xs.reduce((a, b) => a + b, 0) / n
  const my = ys.reduce((a, b) => a + b, 0) / n
  let sxy = 0, sxx = 0, syy = 0
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my)
    sxx += (xs[i] - mx) ** 2
    syy += (ys[i] - my) ** 2
  }
  return sxy / Math.sqrt(sxx * syy)
}

async function main(): Promise<void> {
  const key = process.env.HEROES_PROFILE_V2_API_KEY
  if (!key) throw new Error('HEROES_PROFILE_V2_API_KEY not set')
  const api = new HeroesProfileApiV2(key, 30, 2)
  const hero = process.argv[2] ?? 'Jaina'
  const patch = '2.55'

  const stats: any = await api.fetch('heroes/stats', { timeframe_type: 'major', timeframe: patch, game_type: 'Storm League' })
  if (api.lastDataSource === 'fixture') {
    console.log('Still fixture data: migrate the account first.')
    return
  }
  console.log('1. heroes/stats row keys:', Object.keys(stats.data?.[0] ?? {}).join(', '))
  console.log('   sample:', JSON.stringify(stats.data?.[0]).slice(0, 400))

  const m: any = await api.fetch('heroes/matchups', { timeframe_type: 'major', timeframe: patch, game_type: 'Storm League', hero })
  console.log('2. matchups enemy row keys:', Object.keys(m.enemy?.[0] ?? {}).join(', '))
  const db = createDb()
  const ours = (await db.execute(sql`
    WITH g AS (
      SELECT team0_heroes t0, team1_heroes t1, winner FROM replay_draft_data
      WHERE game_version LIKE ${patch + '.%'}
        AND (team0_heroes ? ${hero} OR team1_heroes ? ${hero})
    ), sides AS (
      SELECT CASE WHEN t0 ? ${hero} THEN t1 ELSE t0 END AS enemies,
             (CASE WHEN t0 ? ${hero} THEN 0 ELSE 1 END) = winner AS won
      FROM g
    )
    SELECT e AS enemy, count(*) AS games, avg(won::int) * 100 AS wr
    FROM sides, jsonb_array_elements_text(enemies) e
    GROUP BY e HAVING count(*) >= 200`)).rows as any[]
  const oursBy = new Map(ours.map(r => [r.enemy, Number(r.wr)]))
  const xs: number[] = [], ys: number[] = []
  for (const r of m.enemy ?? []) {
    const name = r.hero?.name ?? r.hero
    const theirs = r.win_rate ?? (100 * r.wins) / (r.wins + r.losses)
    if (oursBy.has(name) && Number.isFinite(theirs)) { xs.push(Number(theirs)); ys.push(oursBy.get(name)!) }
  }
  const r = pearson(xs, ys)
  console.log(`   ${hero}: ${xs.length} opponents compared, r = ${r.toFixed(3)} =>`,
    r > 0.3 ? "enemy rows are the HERO's perspective (set V1_ENEMY_ROWS_ARE_OPPONENT_PERSPECTIVE = false)"
      : r < -0.3 ? "enemy rows are the OPPONENT's perspective (set V1_ENEMY_ROWS_ARE_OPPONENT_PERSPECTIVE = true)"
        : 'inconclusive: try another hero')

  const l: any = await api.fetch('replays', { after: '0', game_type: 'Storm League', timeframe_type: 'major', timeframe: patch })
  console.log('3. /replays row keys:', Object.keys(l.replays?.[0] ?? {}).join(', '))
  console.log('   page size:', l.replays?.length, '| league_tier present:', 'league_tier' in (l.replays?.[0] ?? {}))
  process.exit(0)
}

main().catch(err => { console.error(err); process.exit(1) })
