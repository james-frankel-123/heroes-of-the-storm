import { NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { playerMatchHistory } from '@/lib/db/schema'
import { sql, desc, and, gte, eq } from 'drizzle-orm'
import { getPlayerSeasonBreakdown } from '@/lib/data/queries'

/**
 * GET /api/player-search?q=SirWatsonII
 *
 * Searches player_match_history for battletags matching the query, then
 * returns hero stats, map stats, and season stats for the first match.
 * Used by the Heroes tab "Search" feature for non-tracked players.
 */
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const q = searchParams.get('q')?.trim()
  if (!q || q.length < 2) {
    return NextResponse.json({ error: 'Query too short' }, { status: 400 })
  }

  // Find matching battletags
  const matches = await db
    .selectDistinct({ battletag: playerMatchHistory.battletag })
    .from(playerMatchHistory)
    .where(sql`${playerMatchHistory.battletag} ILIKE ${'%' + q + '%'}`)
    .limit(10)

  if (matches.length === 0) {
    return NextResponse.json({ error: 'No players found', results: [] })
  }

  // If exact match or single result, return full stats
  const battletag = matches.length === 1
    ? matches[0].battletag
    : matches.find(m => m.battletag.toLowerCase().startsWith(q.toLowerCase() + '#'))?.battletag
      ?? null

  if (!battletag) {
    return NextResponse.json({
      results: matches.map(m => m.battletag),
    })
  }

  // Fetch hero stats (career)
  const heroStats = await db
    .select({
      hero: playerMatchHistory.hero,
      games: sql<number>`count(*)::int`,
      wins: sql<number>`sum(case when ${playerMatchHistory.win} then 1 else 0 end)::int`,
    })
    .from(playerMatchHistory)
    .where(sql`${playerMatchHistory.battletag} = ${battletag}`)
    .groupBy(playerMatchHistory.hero)
    .orderBy(desc(sql`count(*)`))

  // Fetch map stats (career)
  const mapStats = await db
    .select({
      map: playerMatchHistory.map,
      games: sql<number>`count(*)::int`,
      wins: sql<number>`sum(case when ${playerMatchHistory.win} then 1 else 0 end)::int`,
    })
    .from(playerMatchHistory)
    .where(sql`${playerMatchHistory.battletag} = ${battletag}`)
    .groupBy(playerMatchHistory.map)
    .orderBy(desc(sql`count(*)`))

  // Fetch season stats (current year) + last 3 seasons
  const year = new Date().getFullYear()
  const seasonStart = new Date(year, 0, 1)
  const threeSeasonStart = new Date(year - 2, 0, 1)
  const seasonStats = await db
    .select({
      hero: playerMatchHistory.hero,
      games: sql<number>`count(*)::int`,
      wins: sql<number>`sum(case when ${playerMatchHistory.win} then 1 else 0 end)::int`,
    })
    .from(playerMatchHistory)
    .where(and(
      sql`${playerMatchHistory.battletag} = ${battletag}`,
      gte(playerMatchHistory.gameDate, seasonStart),
    ))
    .groupBy(playerMatchHistory.hero)
    .orderBy(desc(sql`count(*)`))

  // Season map stats
  const seasonMapStats = await db
    .select({
      map: playerMatchHistory.map,
      games: sql<number>`count(*)::int`,
      wins: sql<number>`sum(case when ${playerMatchHistory.win} then 1 else 0 end)::int`,
    })
    .from(playerMatchHistory)
    .where(and(
      sql`${playerMatchHistory.battletag} = ${battletag}`,
      gte(playerMatchHistory.gameDate, seasonStart),
    ))
    .groupBy(playerMatchHistory.map)
    .orderBy(desc(sql`count(*)`))

  return NextResponse.json({
    battletag,
    heroStats: heroStats.map(h => ({
      hero: h.hero,
      games: h.games,
      wins: h.wins,
      winRate: h.games > 0 ? Math.round((h.wins / h.games) * 1000) / 10 : 0,
    })),
    mapStats: mapStats.map(m => ({
      map: m.map,
      games: m.games,
      wins: m.wins,
      winRate: m.games > 0 ? Math.round((m.wins / m.games) * 1000) / 10 : 0,
    })),
    seasonHeroStats: seasonStats.map(h => ({
      hero: h.hero,
      games: h.games,
      wins: h.wins,
    })),
    seasonMapStats: seasonMapStats.map(m => ({
      map: m.map,
      games: m.games,
      wins: m.wins,
      winRate: m.games > 0 ? Math.round((m.wins / m.games) * 1000) / 10 : 0,
    })),
    threeSeasonHeroStats: await (async () => {
      const rows = await db
        .select({
          hero: playerMatchHistory.hero,
          games: sql<number>`count(*)::int`,
          wins: sql<number>`sum(case when ${playerMatchHistory.win} then 1 else 0 end)::int`,
        })
        .from(playerMatchHistory)
        .where(and(
          sql`${playerMatchHistory.battletag} = ${battletag}`,
          gte(playerMatchHistory.gameDate, threeSeasonStart),
        ))
        .groupBy(playerMatchHistory.hero)
        .orderBy(desc(sql`count(*)`))
      return rows.map(h => ({ hero: h.hero, games: h.games, wins: h.wins }))
    })(),
    threeSeasonMapStats: await (async () => {
      const rows = await db
        .select({
          map: playerMatchHistory.map,
          games: sql<number>`count(*)::int`,
          wins: sql<number>`sum(case when ${playerMatchHistory.win} then 1 else 0 end)::int`,
        })
        .from(playerMatchHistory)
        .where(and(
          sql`${playerMatchHistory.battletag} = ${battletag}`,
          gte(playerMatchHistory.gameDate, threeSeasonStart),
        ))
        .groupBy(playerMatchHistory.map)
        .orderBy(desc(sql`count(*)`))
      return rows.map(m => ({
        map: m.map, games: m.games, wins: m.wins,
        winRate: m.games > 0 ? Math.round((m.wins / m.games) * 1000) / 10 : 0,
      }))
    })(),
    seasonBreakdown: await getPlayerSeasonBreakdown(battletag),
  })
}
