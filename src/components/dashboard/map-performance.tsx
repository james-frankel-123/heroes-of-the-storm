'use client'

import * as React from 'react'
import { Map, Sparkles } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { getWinRateColor, formatPercent } from '@/lib/utils'
import { MapStats, PlayerData } from '@/types'
import { useMapCommentary } from '@/lib/hooks/use-map-commentary'
import { StreamingText } from '@/components/commentary/streaming-text'

interface MapPerformanceProps {
  mapData: MapStats[]
  playerData?: PlayerData
}

interface MapItemProps {
  map: MapStats
  playerData?: PlayerData
}

function MapItem({ map, playerData }: MapItemProps) {
  const { commentary, isStreaming, error } = useMapCommentary(
    map.map,
    playerData,
    { autoFetch: true }
  )

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

      {/* AI Commentary */}
      {playerData && (
        <div className="mt-2 pl-1">
          <div className="flex items-center gap-1 mb-1">
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
    </div>
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
