'use client'

import * as React from 'react'
import { Map } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { getWinRateColor, formatPercent } from '@/lib/utils'
import { MapStats } from '@/types'

interface MapPerformanceProps {
  mapData: MapStats[]
}

export function MapPerformance({ mapData }: MapPerformanceProps) {
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
            <div key={map.map} className="space-y-2">
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
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
