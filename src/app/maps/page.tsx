'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Map, TrendingUp, Loader2, AlertTriangle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatPercent, getWinRateColor } from '@/lib/utils'
import { usePlayerData } from '@/lib/hooks/use-data'

export default function MapsPage() {
  const { data, isLoading, error } = usePlayerData()

  // Calculate top heroes per map with confidence weighting
  const mapDataWithTopHeroes = React.useMemo(() => {
    if (!data) return []

    return data.mapStats.map((mapStat) => {
      // Get heroes played on this specific map with confidence weighting
      const heroesOnMap = (mapStat.heroes || [])
        .filter(h => h.games >= 5) // Minimum 5 games for statistical significance
        .map(h => ({
          hero: h.hero,
          games: h.games,
          winRate: h.winRate,
          // Confidence-weighted score: Multiply win rate by confidence factor
          // More games = higher confidence = win rate matters more
          score: h.winRate * (1 + Math.log(h.games + 1) / 8)
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 3) // Top 3 heroes

      // High potential heroes: 2-4 games with >60% win rate
      const highPotentialHeroes = (mapStat.heroes || [])
        .filter(h => h.games >= 2 && h.games < 5 && h.winRate >= 60)
        .map(h => ({
          hero: h.hero,
          games: h.games,
          winRate: h.winRate,
        }))
        .sort((a, b) => b.winRate - a.winRate)
        .slice(0, 2) // Show top 2 potential heroes

      // Heroes to avoid: 5+ games with win rate significantly below map average
      const heroesToAvoid = (mapStat.heroes || [])
        .filter(h => {
          const gamesThreshold = 5
          const winRateDifference = mapStat.winRate - h.winRate
          return h.games >= gamesThreshold && winRateDifference >= 10 // 10%+ below map average
        })
        .map(h => {
          // Calculate what map win rate would be without this hero
          const gamesWithoutHero = mapStat.games - h.games
          const winsWithoutHero = mapStat.wins - h.wins
          const hypotheticalWinRate = gamesWithoutHero > 0
            ? (winsWithoutHero / gamesWithoutHero) * 100
            : mapStat.winRate

          return {
            hero: h.hero,
            games: h.games,
            winRate: h.winRate,
            hypotheticalMapWinRate: hypotheticalWinRate,
            improvement: hypotheticalWinRate - mapStat.winRate,
          }
        })
        .sort((a, b) => b.improvement - a.improvement) // Sort by biggest improvement potential
        .slice(0, 3) // Show top 3 worst performers

      return {
        map: mapStat.map,
        games: mapStat.games,
        wins: mapStat.wins,
        losses: mapStat.losses,
        winRate: mapStat.winRate,
        topHeroes: heroesOnMap, // Keep full hero objects with stats
        highPotential: highPotentialHeroes,
        heroesToAvoid: heroesToAvoid,
      }
    })
  }, [data])

  // Show loading state
  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="text-center">
          <Loader2 className="mx-auto h-12 w-12 animate-spin text-primary-500" />
          <p className="mt-4 text-sm text-muted-foreground">Loading map analytics...</p>
        </div>
      </div>
    )
  }

  // Show error state
  if (error) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <Card className="glass border-gaming-danger/30">
          <CardContent className="p-6 text-center">
            <AlertTriangle className="mx-auto h-12 w-12 text-gaming-danger" />
            <h3 className="mt-4 text-xl font-semibold">Error Loading Map Data</h3>
            <p className="mt-2 text-sm text-muted-foreground">{error}</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  const bestMaps = mapDataWithTopHeroes.slice(0, 2)
  const worstMaps = [...mapDataWithTopHeroes].sort((a, b) => a.winRate - b.winRate).slice(0, 2)
  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-4xl font-bold tracking-tight glow">Map Analytics</h1>
        <p className="mt-2 text-muted-foreground">
          Analyze your performance across all battlegrounds
        </p>
      </motion.div>

      {/* Map Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {mapDataWithTopHeroes.map((map, index) => (
          <motion.div
            key={map.map}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="glass group h-full border-primary-500/30 transition-all hover:scale-[1.02] hover:border-primary-500/60 hover:shadow-lg hover:shadow-primary-500/20">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <Map className="h-5 w-5 text-primary-500" />
                      <CardTitle className="text-lg">{map.map}</CardTitle>
                    </div>
                  </div>
                  <div className={`text-3xl font-bold ${getWinRateColor(map.winRate)}`}>
                    {formatPercent(map.winRate, 1)}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground">Total Games</p>
                    <p className="text-lg font-semibold">{map.games}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Record</p>
                    <p className="text-lg font-semibold">
                      {map.wins}-{map.losses}
                    </p>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gaming-success">Wins</span>
                    <span className="text-gaming-danger">Losses</span>
                  </div>
                  <div className="relative h-2 overflow-hidden rounded-full bg-muted">
                    <div
                      className={`h-full ${
                        map.winRate >= 52
                          ? 'bg-gaming-success'
                          : map.winRate >= 48
                          ? 'bg-gaming-warning'
                          : 'bg-gaming-danger'
                      }`}
                      style={{ width: `${map.winRate}%` }}
                    />
                  </div>
                </div>

                {/* Top Heroes */}
                {map.topHeroes.length > 0 && (
                  <div className="space-y-2 rounded-lg bg-primary-500/5 p-3">
                    <p className="text-xs font-semibold text-muted-foreground">Top Heroes (5+ games)</p>
                    <div className="space-y-2">
                      {map.topHeroes.map((heroData) => (
                        <div
                          key={heroData.hero}
                          className="flex items-center justify-between rounded-lg bg-primary-500/10 px-3 py-2"
                        >
                          <span className="text-xs font-medium">{heroData.hero}</span>
                          <div className="flex items-center gap-2 text-xs">
                            <span className={getWinRateColor(heroData.winRate)}>
                              {heroData.winRate.toFixed(1)}%
                            </span>
                            <span className="text-muted-foreground">
                              {heroData.games}g
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* High Potential Heroes */}
                {map.highPotential && map.highPotential.length > 0 && (
                  <div className="space-y-2 rounded-lg bg-accent-cyan/5 p-3">
                    <p className="text-xs font-semibold text-accent-cyan">High Potential (2-4 games)</p>
                    <div className="space-y-2">
                      {map.highPotential.map((heroData) => (
                        <div
                          key={heroData.hero}
                          className="flex items-center justify-between rounded-lg bg-accent-cyan/10 px-3 py-2"
                        >
                          <span className="text-xs font-medium">{heroData.hero}</span>
                          <div className="flex items-center gap-2 text-xs">
                            <span className={getWinRateColor(heroData.winRate)}>
                              {heroData.winRate.toFixed(1)}%
                            </span>
                            <span className="text-muted-foreground">
                              {heroData.games}g
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Heroes to Avoid */}
                {map.heroesToAvoid && map.heroesToAvoid.length > 0 && (
                  <div className="space-y-2 rounded-lg bg-gaming-danger/5 p-3">
                    <p className="text-xs font-semibold text-gaming-danger">Avoid These Heroes</p>
                    <div className="space-y-2">
                      {map.heroesToAvoid.map((heroData) => (
                        <div
                          key={heroData.hero}
                          className="rounded-lg border border-gaming-danger/30 bg-gaming-danger/10 px-3 py-2"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium">{heroData.hero}</span>
                            <div className="flex items-center gap-2 text-xs">
                              <span className="text-gaming-danger">
                                {heroData.winRate.toFixed(1)}%
                              </span>
                              <span className="text-muted-foreground">
                                {heroData.games}g
                              </span>
                            </div>
                          </div>
                          <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
                            <span>Without: {heroData.hypotheticalMapWinRate.toFixed(1)}%</span>
                            <span className="text-gaming-success">
                              (+{heroData.improvement.toFixed(1)}%)
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Map Recommendations */}
      {mapDataWithTopHeroes.length > 0 && (
        <Card className="glass border-primary-500/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-gaming-success" />
              Map Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {bestMaps.length > 0 && (
              <div className="rounded-lg border border-gaming-success/30 bg-gaming-success/5 p-4">
                <p className="font-semibold text-gaming-success">Best Maps</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Focus on {bestMaps.map(m => m.map).join(' and ')} - your win rate is{' '}
                  {bestMaps[0].winRate.toFixed(1)}% or better on these maps!
                </p>
              </div>
            )}
            {worstMaps.length > 0 && worstMaps[0].winRate < 48 && (
              <div className="rounded-lg border border-gaming-danger/30 bg-gaming-danger/5 p-4">
                <p className="font-semibold text-gaming-danger">Practice Needed</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  {worstMaps[0].map} ({worstMaps[0].winRate.toFixed(1)}% WR) needs attention. Consider practicing specific heroes or strategies for this map.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
