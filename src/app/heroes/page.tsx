'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Search, Filter, TrendingUp, TrendingDown, Loader2, AlertTriangle, Trophy, Award, Shield, Star, Sparkles } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Heatmap } from '@/components/charts/heatmap'
import { formatPercent, getWinRateColor } from '@/lib/utils'
import { usePlayerData } from '@/lib/hooks/use-data'
import { useHeroCommentary } from '@/lib/hooks/use-hero-commentary'
import { StreamingText } from '@/components/commentary/streaming-text'
import { HeroStats, PlayerData } from '@/types'

interface HeroCardProps {
  hero: HeroStats
  index: number
  playerData?: PlayerData
}

function HeroCard({ hero, index, playerData }: HeroCardProps) {
  const { commentary, isStreaming, error } = useHeroCommentary(
    hero.hero,
    playerData,
    { autoFetch: true }
  )

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
    >
      <Card className="glass group border-primary-500/30 transition-all hover:scale-[1.02] hover:border-primary-500/60 hover:shadow-lg hover:shadow-primary-500/20">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div>
              <CardTitle className="text-lg">{hero.hero}</CardTitle>
              <Badge variant={hero.role.toLowerCase().replace(' ', '') as any} className="mt-2 text-xs">
                {hero.role}
              </Badge>
            </div>
            <div className={`text-2xl font-bold ${getWinRateColor(hero.winRate)}`}>
              {formatPercent(hero.winRate, 1)}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Games</span>
            <span className="font-semibold">{hero.games}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Record</span>
            <span className="font-semibold">
              {hero.wins}-{hero.losses}
            </span>
          </div>
          <div className="relative h-1.5 overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full rounded-full ${
                hero.winRate >= 55
                  ? 'bg-gaming-success'
                  : hero.winRate >= 50
                  ? 'bg-gaming-warning'
                  : 'bg-gaming-danger'
              }`}
              style={{ width: `${hero.winRate}%` }}
            />
          </div>

          {/* AI Commentary */}
          {playerData && (
            <div className="pt-3 border-t border-primary-500/20">
              <div className="flex items-center gap-1 mb-2">
                <Sparkles className="h-3 w-3 text-blue-400" />
                <span className="text-xs font-medium text-blue-400">AI Analysis</span>
              </div>
              {error ? (
                <p className="text-xs text-red-400">{error}</p>
              ) : (
                <StreamingText
                  text={commentary}
                  isStreaming={isStreaming}
                  className="text-xs text-muted-foreground leading-relaxed"
                  showCursor={true}
                />
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default function HeroesPage() {
  const { data, isLoading, error } = usePlayerData()
  const [searchQuery, setSearchQuery] = React.useState('')
  const [sortBy, setSortBy] = React.useState<'winRate' | 'games'>('winRate')
  const [selectedRole, setSelectedRole] = React.useState<string | null>(null)

  // Generate heatmap data from mapStats
  const heatmapData = React.useMemo(() => {
    if (!data) return []

    const heatmap: Array<{ hero: string; map: string; winRate: number; games: number }> = []

    // For each map, get top heroes and their stats
    data.mapStats.forEach((mapStat) => {
      const topHeroes = (mapStat.heroes || [])
        .filter(h => h.games >= 5) // Only include heroes with significant games
        .sort((a, b) => b.winRate - a.winRate)
        .slice(0, 10) // Top 10 heroes per map

      topHeroes.forEach(heroOnMap => {
        heatmap.push({
          hero: heroOnMap.hero,
          map: mapStat.map,
          winRate: heroOnMap.winRate,
          games: heroOnMap.games,
        })
      })
    })

    return heatmap
  }, [data])

  // Calculate mastery badges: heroes completed on all maps and maps completed with all heroes
  const masteryBadges = React.useMemo(() => {
    if (!data) return null

    const totalMaps = data.mapStats.length
    const totalHeroes = data.heroStats.length

    // Build a map of hero -> set of maps played
    const heroToMaps = new Map<string, Set<string>>()
    data.mapStats.forEach((mapStat) => {
      (mapStat.heroes || []).forEach((heroOnMap) => {
        if (heroOnMap.games > 0) {
          if (!heroToMaps.has(heroOnMap.hero)) {
            heroToMaps.set(heroOnMap.hero, new Set())
          }
          heroToMaps.get(heroOnMap.hero)!.add(mapStat.map)
        }
      })
    })

    // Build a map of map -> set of heroes played
    const mapToHeroes = new Map<string, Set<string>>()
    data.mapStats.forEach((mapStat) => {
      const heroesOnThisMap = new Set<string>()
      ;(mapStat.heroes || []).forEach((heroOnMap) => {
        if (heroOnMap.games > 0) {
          heroesOnThisMap.add(heroOnMap.hero)
        }
      })
      mapToHeroes.set(mapStat.map, heroesOnThisMap)
    })

    // Get all map names
    const allMaps = new Set(data.mapStats.map(m => m.map))

    // Find heroes that have been played on ALL maps
    const completedHeroes = Array.from(heroToMaps.entries())
      .filter(([hero, maps]) => maps.size === totalMaps)
      .map(([hero]) => hero)

    // Find heroes near completion (played on some maps but not all)
    const nearCompletionHeroes = Array.from(heroToMaps.entries())
      .filter(([hero, maps]) => maps.size > 0 && maps.size < totalMaps)
      .map(([hero, maps]) => {
        const missingMaps = Array.from(allMaps).filter(map => !maps.has(map))
        return {
          hero,
          mapsCompleted: maps.size,
          mapsRemaining: totalMaps - maps.size,
          completionPercentage: (maps.size / totalMaps) * 100,
          missingMaps,
        }
      })
      .sort((a, b) => b.mapsCompleted - a.mapsCompleted) // Sort by most maps completed

    // Find maps that have had ALL heroes played on them
    const completedMaps = Array.from(mapToHeroes.entries())
      .filter(([map, heroes]) => heroes.size === totalHeroes)
      .map(([map]) => map)

    // Get all hero names
    const allHeroNames = new Set(data.heroStats.map(h => h.hero))

    // Find maps near completion (have some heroes but not all)
    const nearCompletionMaps = Array.from(mapToHeroes.entries())
      .filter(([map, heroes]) => heroes.size > 0 && heroes.size < totalHeroes)
      .map(([map, heroes]) => {
        const missingHeroes = Array.from(allHeroNames).filter(hero => !heroes.has(hero))
        return {
          map,
          heroesCompleted: heroes.size,
          heroesRemaining: totalHeroes - heroes.size,
          completionPercentage: (heroes.size / totalHeroes) * 100,
          missingHeroes,
        }
      })
      .sort((a, b) => b.heroesCompleted - a.heroesCompleted) // Sort by most heroes completed

    // Check if ALL heroes have been played on ALL maps
    const isGrandmaster = completedHeroes.length === totalHeroes && completedMaps.length === totalMaps

    return {
      completedHeroes,
      completedMaps,
      nearCompletionHeroes,
      nearCompletionMaps,
      totalHeroes,
      totalMaps,
      isGrandmaster,
      heroCompletionPercentage: (completedHeroes.length / totalHeroes) * 100,
      mapCompletionPercentage: (completedMaps.length / totalMaps) * 100,
    }
  }, [data])

  // Filter and sort heroes
  const filteredHeroes = React.useMemo(() => {
    if (!data) return []

    return data.heroStats
      .filter((hero) => {
        const matchesSearch = hero.hero.toLowerCase().includes(searchQuery.toLowerCase())
        const matchesRole = !selectedRole || hero.role === selectedRole
        return matchesSearch && matchesRole
      })
      .sort((a, b) => {
        if (sortBy === 'winRate') return b.winRate - a.winRate
        return b.games - a.games
      })
  }, [data, searchQuery, selectedRole, sortBy])

  // Show loading state
  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="text-center">
          <Loader2 className="mx-auto h-12 w-12 animate-spin text-primary-500" />
          <p className="mt-4 text-sm text-muted-foreground">Loading hero analytics...</p>
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
            <h3 className="mt-4 text-xl font-semibold">Error Loading Hero Data</h3>
            <p className="mt-2 text-sm text-muted-foreground">{error}</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-4xl font-bold tracking-tight glow">Hero Analytics</h1>
        <p className="mt-2 text-muted-foreground">
          Detailed performance analysis for all your heroes
        </p>
      </motion.div>

      {/* Search and Filters */}
      <Card className="glass border-primary-500/30">
        <CardContent className="pt-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search heroes..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-10 w-full rounded-lg border border-input bg-background pl-10 pr-4 text-sm focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
              />
            </div>
            <div className="flex gap-2">
              <Button
                variant={sortBy === 'winRate' ? 'gaming' : 'outline'}
                size="sm"
                onClick={() => setSortBy('winRate')}
              >
                Sort by Win Rate
              </Button>
              <Button
                variant={sortBy === 'games' ? 'gaming' : 'outline'}
                size="sm"
                onClick={() => setSortBy('games')}
              >
                Sort by Games
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Hero Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filteredHeroes.map((hero, index) => (
          <HeroCard key={hero.hero} hero={hero} index={index} playerData={data || undefined} />
        ))}
      </div>

      {/* Mastery Badges */}
      {masteryBadges && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-6"
        >
          {/* Grandmaster Badge - Only show if complete mastery achieved */}
          {masteryBadges.isGrandmaster && (
            <Card className="glass border-2 border-accent-gold/50 bg-gradient-to-br from-accent-gold/10 to-accent-purple/10">
              <CardContent className="pt-6">
                <div className="flex items-center gap-6">
                  <div className="rounded-full bg-accent-gold/20 p-6">
                    <Trophy className="h-12 w-12 text-accent-gold" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <h3 className="text-2xl font-bold text-accent-gold">Nexus Grandmaster</h3>
                      <Badge variant="outline" className="border-accent-gold text-accent-gold">
                        100% Complete
                      </Badge>
                    </div>
                    <p className="mt-2 text-sm text-muted-foreground">
                      🎉 Legendary achievement! You&apos;ve played every single hero on every single map. True mastery of the Nexus!
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Hero & Map Mastery Grid */}
          <div className="grid gap-6 md:grid-cols-2">
            {/* Hero Mastery Badges */}
            <Card className="glass border-primary-500/30">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Award className="h-5 w-5 text-primary-500" />
                    <CardTitle>Hero Mastery</CardTitle>
                  </div>
                  <Badge variant="outline" className="text-primary-500">
                    {masteryBadges.completedHeroes.length}/{masteryBadges.totalHeroes}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <p className="mb-4 text-xs text-muted-foreground">
                  Heroes played on all {masteryBadges.totalMaps} maps
                </p>

                {/* Progress Bar */}
                <div className="mb-4 space-y-2">
                  <div className="relative h-2 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary-500 transition-all duration-500"
                      style={{ width: `${masteryBadges.heroCompletionPercentage}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {masteryBadges.heroCompletionPercentage.toFixed(1)}% Complete
                  </p>
                </div>

                {/* Completed Heroes */}
                {masteryBadges.completedHeroes.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-gaming-success">Completed Heroes:</p>
                    <div className="flex flex-wrap gap-1.5">
                      {masteryBadges.completedHeroes.map((hero) => (
                        <Badge
                          key={hero}
                          variant="outline"
                          className="border-gaming-success/30 bg-gaming-success/10 text-gaming-success"
                        >
                          <Trophy className="mr-1 h-3 w-3" />
                          {hero}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Near Completion Heroes */}
                {masteryBadges.nearCompletionHeroes.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <p className="text-xs font-semibold text-gaming-warning">Near Mastery (Top 10):</p>
                    <div className="max-h-64 space-y-2 overflow-y-auto pr-2">
                      {masteryBadges.nearCompletionHeroes.slice(0, 10).map((heroData) => (
                        <div
                          key={heroData.hero}
                          className="rounded-lg border border-primary-500/20 bg-primary-500/5 p-2"
                        >
                          <div className="mb-1 flex items-center justify-between">
                            <span className="text-xs font-medium">{heroData.hero}</span>
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-muted-foreground">
                                {heroData.mapsCompleted}/{masteryBadges.totalMaps} maps
                              </span>
                              <Badge variant="outline" className="text-[10px]">
                                {heroData.completionPercentage.toFixed(0)}%
                              </Badge>
                            </div>
                          </div>
                          <div className="flex flex-wrap gap-1">
                            <span className="text-[10px] text-muted-foreground">Missing:</span>
                            {heroData.missingMaps.map((map) => (
                              <Badge
                                key={map}
                                variant="outline"
                                className="border-gaming-danger/30 bg-gaming-danger/5 text-[10px] text-gaming-danger"
                              >
                                {map}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {masteryBadges.completedHeroes.length === 0 && masteryBadges.nearCompletionHeroes.length === 0 && (
                  <p className="text-center text-xs italic text-muted-foreground">
                    Play a hero on all maps to earn your first hero mastery badge!
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Map Mastery Badges */}
            <Card className="glass border-accent-cyan/30">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Shield className="h-5 w-5 text-accent-cyan" />
                    <CardTitle>Map Mastery</CardTitle>
                  </div>
                  <Badge variant="outline" className="text-accent-cyan">
                    {masteryBadges.completedMaps.length}/{masteryBadges.totalMaps}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <p className="mb-4 text-xs text-muted-foreground">
                  Maps with all {masteryBadges.totalHeroes} heroes played
                </p>

                {/* Progress Bar */}
                <div className="mb-4 space-y-2">
                  <div className="relative h-2 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-accent-cyan transition-all duration-500"
                      style={{ width: `${masteryBadges.mapCompletionPercentage}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {masteryBadges.mapCompletionPercentage.toFixed(1)}% Complete
                  </p>
                </div>

                {/* Completed Maps */}
                {masteryBadges.completedMaps.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-accent-cyan">Completed Maps:</p>
                    <div className="flex flex-wrap gap-1.5">
                      {masteryBadges.completedMaps.map((map) => (
                        <Badge
                          key={map}
                          variant="outline"
                          className="border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan"
                        >
                          <Shield className="mr-1 h-3 w-3" />
                          {map}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Near Completion Maps */}
                {masteryBadges.nearCompletionMaps.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <p className="text-xs font-semibold text-gaming-warning">Near Mastery (Top 10):</p>
                    <div className="max-h-64 space-y-2 overflow-y-auto pr-2">
                      {masteryBadges.nearCompletionMaps.slice(0, 10).map((mapData) => (
                        <div
                          key={mapData.map}
                          className="rounded-lg border border-accent-cyan/20 bg-accent-cyan/5 p-2"
                        >
                          <div className="mb-1 flex items-center justify-between">
                            <span className="text-xs font-medium">{mapData.map}</span>
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-muted-foreground">
                                {mapData.heroesCompleted}/{masteryBadges.totalHeroes} heroes
                              </span>
                              <Badge variant="outline" className="text-[10px]">
                                {mapData.completionPercentage.toFixed(0)}%
                              </Badge>
                            </div>
                          </div>
                          <div className="flex flex-wrap gap-1">
                            <span className="text-[10px] text-muted-foreground">Missing:</span>
                            {mapData.missingHeroes.slice(0, 15).map((hero) => (
                              <Badge
                                key={hero}
                                variant="outline"
                                className="border-gaming-danger/30 bg-gaming-danger/5 text-[10px] text-gaming-danger"
                              >
                                {hero}
                              </Badge>
                            ))}
                            {mapData.missingHeroes.length > 15 && (
                              <Badge variant="outline" className="text-[10px] text-muted-foreground">
                                +{mapData.missingHeroes.length - 15} more
                              </Badge>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {masteryBadges.completedMaps.length === 0 && masteryBadges.nearCompletionMaps.length === 0 && (
                  <p className="text-center text-xs italic text-muted-foreground">
                    Play all heroes on a map to earn your first map mastery badge!
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </motion.div>
      )}

      {/* Heatmap */}
      <Card className="glass border-primary-500/30">
        <CardHeader>
          <CardTitle>Hero vs Map Win Rate Matrix</CardTitle>
        </CardHeader>
        <CardContent>
          <Heatmap data={heatmapData} />
        </CardContent>
      </Card>
    </div>
  )
}
