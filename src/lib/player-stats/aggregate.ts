/**
 * Pure per-player stat aggregation (no DB dependency, unit-testable).
 *
 * Mirrors the aggregation in `sync/compute-derived.ts` so that stats ingested
 * via the API are identical to those the offline HeroesProfile sync produces.
 * MAWP comes from the shared reference impl in `@/lib/mawp`. Keep in sync with
 * compute-derived.ts.
 */
import { computeMAWP } from '@/lib/mawp'

export interface DerivedMatch {
  hero: string
  map: string
  win: boolean
  gameDate: Date
  kills: number | null
  deaths: number | null
  assists: number | null
}

export interface HeroAggregation {
  hero: string
  games: number
  wins: number
  winRate: number
  mawp: number
  recentWinRate: number | null
  trend: number | null
  avgKills: number
  avgDeaths: number
  avgAssists: number
}

/** Per-hero aggregate with MAWP. Mirrors sync/compute-derived.ts:aggregateByHero. */
export function aggregateByHero(matches: DerivedMatch[]): HeroAggregation[] {
  const grouped: Record<string, DerivedMatch[]> = {}
  for (const m of matches) {
    ;(grouped[m.hero] ??= []).push(m)
  }

  return Object.entries(grouped).map(([hero, heroMatches]) => {
    const games = heroMatches.length
    const wins = heroMatches.filter((m) => m.win).length
    const winRate = games > 0 ? (wins / games) * 100 : 0
    const mawp = computeMAWP(heroMatches) * 100

    const sorted = [...heroMatches].sort((a, b) => b.gameDate.getTime() - a.gameDate.getTime())
    const recent20 = sorted.slice(0, 20)
    const recentWinRate =
      recent20.length >= 5 ? (recent20.filter((m) => m.win).length / recent20.length) * 100 : null
    const trend = recentWinRate !== null ? recentWinRate - winRate : null

    const avgKills = heroMatches.reduce((s, m) => s + (m.kills ?? 0), 0) / games
    const avgDeaths = heroMatches.reduce((s, m) => s + (m.deaths ?? 0), 0) / games
    const avgAssists = heroMatches.reduce((s, m) => s + (m.assists ?? 0), 0) / games

    return { hero, games, wins, winRate, mawp, recentWinRate, trend, avgKills, avgDeaths, avgAssists }
  })
}

export interface HeroMapAggregation {
  hero: string
  map: string
  games: number
  wins: number
  winRate: number
}

/** Per-hero-per-map aggregate. Mirrors sync/compute-derived.ts:aggregateByHeroMap. */
export function aggregateByHeroMap(matches: Array<{ hero: string; map: string; win: boolean }>): HeroMapAggregation[] {
  const grouped: Record<string, { hero: string; map: string; games: number; wins: number }> = {}
  for (const m of matches) {
    const key = `${m.hero}|${m.map}`
    grouped[key] ??= { hero: m.hero, map: m.map, games: 0, wins: 0 }
    grouped[key].games++
    if (m.win) grouped[key].wins++
  }
  return Object.values(grouped).map((g) => ({ ...g, winRate: g.games > 0 ? (g.wins / g.games) * 100 : 0 }))
}
