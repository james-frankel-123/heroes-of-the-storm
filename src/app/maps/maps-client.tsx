'use client'

import { useState, useMemo } from 'react'
import { TierSelector, getTierLabel } from '@/components/shared/tier-selector'
import { MapCard } from '@/components/maps/map-card'
import { MapDetailModal } from '@/components/maps/map-detail-modal'
import { confidenceAdjustedWinRate } from '@/lib/utils'
import type {
  SkillTier,
  MapStats,
  HeroMapStats,
  PlayerMatch,
} from '@/lib/types'

interface PersonalMapData {
  battletag: string
  mapStats: { map: string; games: number; wins: number; winRate: number }[]
  matches: PlayerMatch[]
}

interface MapsClientProps {
  mapStatsByTier: Record<SkillTier, MapStats[]>
  heroMapByTier: Record<SkillTier, Record<string, HeroMapStats[]>>
  personalData: PersonalMapData[]
}

export function MapsClient({
  mapStatsByTier,
  heroMapByTier,
  personalData,
}: MapsClientProps) {
  const [tier, setTier] = useState<SkillTier>('mid')
  const [selectedMap, setSelectedMap] = useState<string | null>(null)

  const maps = mapStatsByTier[tier]

  // Build personal stats lookup for each map
  const personalByMap: Record<
    string,
    { battletag: string; games: number; wins: number; winRate: number }[]
  > = {}
  for (const p of personalData) {
    for (const ms of p.mapStats) {
      if (!personalByMap[ms.map]) personalByMap[ms.map] = []
      personalByMap[ms.map].push({
        battletag: p.battletag,
        games: ms.games,
        wins: ms.wins,
        winRate: ms.winRate,
      })
    }
  }

  // Pre-compute confidence-adjusted top/bottom heroes per map.
  // Threshold = 50: heroes with <50 games get phantom 50% games padded to 50.
  const CONFIDENCE_THRESHOLD = 50

  const heroRankingsByMap = useMemo(() => {
    const result: Record<string, { top: HeroMapStats[]; bottom: HeroMapStats[] }> = {}
    for (const mapStat of maps) {
      const heroesOnMap = heroMapByTier[tier][mapStat.map] ?? []
      const sorted = [...heroesOnMap].sort((a, b) => {
        const adjA = confidenceAdjustedWinRate(a.wins, a.games, CONFIDENCE_THRESHOLD)
        const adjB = confidenceAdjustedWinRate(b.wins, b.games, CONFIDENCE_THRESHOLD)
        return adjB - adjA
      })
      result[mapStat.map] = {
        top: sorted.slice(0, 5),
        bottom: sorted.slice(-3).reverse(),
      }
    }
    return result
  }, [maps, heroMapByTier, tier])

  // Data for selected map's detail modal
  const detailHeroStats = selectedMap
    ? heroMapByTier[tier][selectedMap] ?? []
    : []

  const detailPersonal = selectedMap
    ? personalByMap[selectedMap] ?? []
    : []

  const detailMatches = selectedMap
    ? personalData.map((p) => ({
        battletag: p.battletag,
        matches: p.matches.filter((m) => m.map === selectedMap),
      }))
    : []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Maps</h1>
          <p className="text-muted-foreground mt-1">
            Aggregate map stats &mdash; {getTierLabel(tier)}
          </p>
        </div>
        <TierSelector value={tier} onChange={setTier} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {[...maps]
          .sort((a, b) => b.games - a.games)
          .map((mapStat) => {
            const rankings = heroRankingsByMap[mapStat.map] ?? { top: [], bottom: [] }
            return (
              <MapCard
                key={mapStat.map}
                mapStat={mapStat}
                personalStats={personalByMap[mapStat.map] ?? []}
                topHeroes={rankings.top}
                bottomHeroes={rankings.bottom}
                onClick={() => setSelectedMap(mapStat.map)}
              />
            )
          })}
      </div>

      {selectedMap && (
        <MapDetailModal
          open={!!selectedMap}
          onClose={() => setSelectedMap(null)}
          mapName={selectedMap}
          heroStats={detailHeroStats}
          personalStats={detailPersonal}
          personalMatches={detailMatches}
          currentTier={tier}
        />
      )}
    </div>
  )
}
