'use client'

import { useState } from 'react'
import { TierSelector, getTierLabel } from '@/components/shared/tier-selector'
import { MapCard } from '@/components/maps/map-card'
import { MapDetailModal } from '@/components/maps/map-detail-modal'
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
        {maps
          .sort((a, b) => b.games - a.games)
          .map((mapStat) => (
            <MapCard
              key={mapStat.map}
              mapStat={mapStat}
              personalStats={personalByMap[mapStat.map] ?? []}
              topHeroes={(heroMapByTier[tier][mapStat.map] ?? []).slice(0, 5)}
              bottomHeroes={(heroMapByTier[tier][mapStat.map] ?? [])
                .sort((a, b) => a.winRate - b.winRate)
                .slice(0, 3)}
              onClick={() => setSelectedMap(mapStat.map)}
            />
          ))}
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
