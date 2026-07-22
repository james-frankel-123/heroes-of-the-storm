/**
 * DB-backed recompute of derived per-player stats (per-hero MAWP + per-hero-per-map
 * win rates) from `player_match_history`. Uses the pure aggregation in
 * `./aggregate` (which mirrors sync/compute-derived.ts). Kept separate from the
 * aggregation so the pure functions stay unit-testable without a DB connection.
 */
import { eq, desc, sql } from 'drizzle-orm'
import { db } from '@/lib/db'
import { playerMatchHistory, playerHeroStats, playerHeroMapStats } from '@/lib/db/schema'
import { aggregateByHero, aggregateByHeroMap } from './aggregate'

export { aggregateByHero, aggregateByHeroMap } from './aggregate'
export type { DerivedMatch, HeroAggregation, HeroMapAggregation } from './aggregate'

/**
 * Recompute and upsert derived stats for a single battletag from its full
 * match history. Idempotent (upsert on the unique indexes).
 */
export async function recomputeDerivedStats(battletag: string): Promise<{ heroes: number; heroMaps: number }> {
  const matches = await db
    .select({
      hero: playerMatchHistory.hero,
      map: playerMatchHistory.map,
      win: playerMatchHistory.win,
      gameDate: playerMatchHistory.gameDate,
      kills: playerMatchHistory.kills,
      deaths: playerMatchHistory.deaths,
      assists: playerMatchHistory.assists,
    })
    .from(playerMatchHistory)
    .where(eq(playerMatchHistory.battletag, battletag))
    .orderBy(desc(playerMatchHistory.gameDate))

  if (matches.length === 0) return { heroes: 0, heroMaps: 0 }

  const heroStats = aggregateByHero(matches)
  for (const hs of heroStats) {
    await db
      .insert(playerHeroStats)
      .values({
        battletag,
        hero: hs.hero,
        games: hs.games,
        wins: hs.wins,
        winRate: hs.winRate,
        mawp: hs.mawp,
        avgKills: hs.avgKills,
        avgDeaths: hs.avgDeaths,
        avgAssists: hs.avgAssists,
        recentWinRate: hs.recentWinRate,
        trend: hs.trend,
        updatedAt: new Date(),
      })
      .onConflictDoUpdate({
        target: [playerHeroStats.battletag, playerHeroStats.hero],
        set: {
          games: sql.raw('excluded.games'),
          wins: sql.raw('excluded.wins'),
          winRate: sql.raw('excluded.win_rate'),
          mawp: sql.raw('excluded.mawp'),
          avgKills: sql.raw('excluded.avg_kills'),
          avgDeaths: sql.raw('excluded.avg_deaths'),
          avgAssists: sql.raw('excluded.avg_assists'),
          recentWinRate: sql.raw('excluded.recent_win_rate'),
          trend: sql.raw('excluded.trend'),
          updatedAt: sql`now()`,
        },
      })
  }

  const heroMapStats = aggregateByHeroMap(matches)
  const BATCH = 100
  for (let i = 0; i < heroMapStats.length; i += BATCH) {
    const batch = heroMapStats.slice(i, i + BATCH)
    await db
      .insert(playerHeroMapStats)
      .values(batch.map((hm) => ({ battletag, hero: hm.hero, map: hm.map, games: hm.games, wins: hm.wins, winRate: hm.winRate, updatedAt: new Date() })))
      .onConflictDoUpdate({
        target: [playerHeroMapStats.battletag, playerHeroMapStats.hero, playerHeroMapStats.map],
        set: {
          games: sql.raw('excluded.games'),
          wins: sql.raw('excluded.wins'),
          winRate: sql.raw('excluded.win_rate'),
          updatedAt: sql`now()`,
        },
      })
  }

  return { heroes: heroStats.length, heroMaps: heroMapStats.length }
}
