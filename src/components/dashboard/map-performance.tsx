'use client'

import * as React from 'react'
import { Map, ChevronRight } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { getWinRateColor, formatPercent } from '@/lib/utils'
import { MapStats, PlayerData } from '@/types'
import { MapDetailsModal } from '@/components/modals/map-details-modal'

interface MapPerformanceProps {
  mapData: MapStats[]
  playerData?: PlayerData
}

interface MapItemProps {
  map: MapStats
  playerData?: PlayerData
}

function MapItem({ map, playerData }: MapItemProps) {
  const [modalOpen, setModalOpen] = React.useState(false)

  if (!playerData) {
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">{map.map}</span>
          <span className={`text-sm font-bold ${getWinRateColor(map.winRate)}`}>
            {formatPercent(map.winRate, 1)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full rounded-full transition-all ${
                map.winRate >= 52
                  ? 'bg-gaming-success'
                  : map.winRate >= 48
                  ? 'bg-gaming-warning'
                  : 'bg-gaming-danger'
              }`}
              style={{ width: `${map.winRate}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground">
            {map.games} games
          </span>
        </div>
      </div>
    )
  }

  return (
    <>
      <button
        onClick={() => setModalOpen(true)}
        className="group w-full space-y-2 text-left rounded-lg p-2 -m-2 transition-all hover:bg-primary-500/5 cursor-pointer"
      >
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">{map.map}</span>
          <div className="flex items-center gap-2">
            <span className={`text-sm font-bold ${getWinRateColor(map.winRate)}`}>
              {formatPercent(map.winRate, 1)}
            </span>
            <ChevronRight className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full rounded-full transition-all ${
                map.winRate >= 52
                  ? 'bg-gaming-success'
                  : map.winRate >= 48
                  ? 'bg-gaming-warning'
                  : 'bg-gaming-danger'
              }`}
              style={{ width: `${map.winRate}%` }}
            />
          </div>
          <span className="text-xs text-muted-foreground">
            {map.games} games
          </span>
        </div>
        <p className="text-xs text-muted-foreground opacity-0 group-hover:opacity-70 transition-opacity">
          Click for AI analysis
        </p>
      </button>

      <MapDetailsModal
        map={map}
        playerData={playerData}
        open={modalOpen}
        onOpenChange={setModalOpen}
      />
    </>
  )
}

export function MapPerformance({ mapData, playerData }: MapPerformanceProps) {
  return (
    <Card className="glass border-primary-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Map className="h-5 w-5 text-primary-500" />
          Map Performance
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {mapData.map((map) => (
            <MapItem key={map.map} map={map} playerData={playerData} />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
