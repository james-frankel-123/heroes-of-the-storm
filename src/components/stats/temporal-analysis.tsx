'use client'

import * as React from 'react'
import { Sparkles } from 'lucide-react'
import { TemporalPattern, DayOfWeekPattern } from '@/lib/data/statistics'
import { ContextData } from './clickable-metric'
import { Card } from '@/components/ui/card'

interface TemporalAnalysisProps {
  hourlyPerformance: TemporalPattern[]
  dailyPerformance: DayOfWeekPattern[]
  onCellClick: (context: ContextData) => void
}

export function TemporalAnalysis({
  hourlyPerformance,
  dailyPerformance,
  onCellClick,
}: TemporalAnalysisProps) {
  const getHeatmapColor = (winRate: number, games: number) => {
    if (games === 0) return 'bg-muted/20'
    if (games < 3) return 'bg-muted/40' // Not enough data

    if (winRate >= 65) return 'bg-green-500/80'
    if (winRate >= 55) return 'bg-green-500/50'
    if (winRate >= 50) return 'bg-primary-500/50'
    if (winRate >= 45) return 'bg-yellow-500/50'
    if (winRate >= 35) return 'bg-orange-500/50'
    return 'bg-red-500/50'
  }

  const handleHourlyClick = (hour: TemporalPattern) => {
    onCellClick({
      type: 'time-period',
      label: `Performance at ${hour.hour}:00`,
      value: hour.winRate,
      relatedMetrics: {
        games: hour.games,
        wins: hour.wins,
        losses: hour.losses,
        timeSlot: `${hour.hour}:00 - ${hour.hour + 1}:00`,
      },
    })
  }

  const handleDailyClick = (day: DayOfWeekPattern) => {
    onCellClick({
      type: 'time-period',
      label: `Performance on ${day.dayName}`,
      value: day.winRate,
      relatedMetrics: {
        games: day.games,
        wins: day.wins,
        losses: day.losses,
        dayOfWeek: day.dayName,
      },
    })
  }

  // Day names for display
  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

  // Group hours into 4-hour blocks for better visualization
  const hourBlocks = [
    { label: 'Night', start: 0, end: 6 },
    { label: 'Morning', start: 6, end: 12 },
    { label: 'Afternoon', start: 12, end: 18 },
    { label: 'Evening', start: 18, end: 24 },
  ]

  return (
    <div className="space-y-6">
      {/* Time of Day Heatmap */}
      <Card className="p-6 glass-card">
        <div className="mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            Performance by Time of Day
            <Sparkles className="h-4 w-4 text-primary-500/50" />
          </h3>
          <p className="text-sm text-muted-foreground">
            Click any time slot to analyze temporal patterns
          </p>
        </div>

        <div className="grid grid-cols-12 gap-2">
          {hourlyPerformance.map((hour) => {
            const games = hour.games
            const hasEnoughData = games >= 3

            return (
              <div
                key={hour.hour}
                onClick={() => hasEnoughData && handleHourlyClick(hour)}
                className={`
                  aspect-square rounded-md flex flex-col items-center justify-center
                  ${getHeatmapColor(hour.winRate, games)}
                  ${hasEnoughData ? 'cursor-pointer hover:ring-2 hover:ring-primary-500 hover:scale-105' : 'opacity-50'}
                  transition-all group relative
                `}
                title={`${hour.hour}:00 - ${hour.winRate.toFixed(1)}% WR (${games} games)`}
              >
                <span className="text-xs font-medium">
                  {hour.hour}
                </span>
                {hasEnoughData && (
                  <>
                    <span className="text-[10px] opacity-80">
                      {hour.winRate.toFixed(0)}%
                    </span>
                    <Sparkles className="h-2 w-2 absolute top-0.5 right-0.5 opacity-0 group-hover:opacity-70 transition-opacity text-primary-500" />
                  </>
                )}
              </div>
            )
          })}
        </div>

        {/* Legend */}
        <div className="mt-4 flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-green-500/80" />
            <span>High WR (65%+)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-primary-500/50" />
            <span>Average</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-red-500/50" />
            <span>Low WR (35%)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-muted/40" />
            <span>Limited data</span>
          </div>
        </div>

        {/* Best/Worst Times */}
        {hourlyPerformance.filter(h => h.games >= 3).length > 0 && (
          <div className="mt-4 grid grid-cols-2 gap-4">
            {(() => {
              const withEnoughData = hourlyPerformance.filter(h => h.games >= 3)
              const best = withEnoughData.reduce((prev, curr) =>
                curr.winRate > prev.winRate ? curr : prev
              )
              const worst = withEnoughData.reduce((prev, curr) =>
                curr.winRate < prev.winRate ? curr : prev
              )

              return (
                <>
                  <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/30">
                    <p className="text-xs text-green-500 font-medium mb-1">Best Time</p>
                    <p className="text-sm font-semibold">
                      {best.hour}:00 - {best.hour + 1}:00
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {best.winRate.toFixed(1)}% WR ({best.games} games)
                    </p>
                  </div>
                  <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                    <p className="text-xs text-red-500 font-medium mb-1">Challenging Time</p>
                    <p className="text-sm font-semibold">
                      {worst.hour}:00 - {worst.hour + 1}:00
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {worst.winRate.toFixed(1)}% WR ({worst.games} games)
                    </p>
                  </div>
                </>
              )
            })()}
          </div>
        )}
      </Card>

      {/* Day of Week Performance */}
      <Card className="p-6 glass-card">
        <div className="mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            Performance by Day of Week
            <Sparkles className="h-4 w-4 text-primary-500/50" />
          </h3>
          <p className="text-sm text-muted-foreground">
            Click any day to analyze weekly patterns
          </p>
        </div>

        <div className="grid grid-cols-7 gap-3">
          {dailyPerformance.map((day) => {
            const hasEnoughData = day.games >= 5

            return (
              <div
                key={day.dayOfWeek}
                onClick={() => hasEnoughData && handleDailyClick(day)}
                className={`
                  p-4 rounded-lg flex flex-col items-center justify-center
                  ${getHeatmapColor(day.winRate, day.games)}
                  ${hasEnoughData ? 'cursor-pointer hover:ring-2 hover:ring-primary-500 hover:scale-105' : 'opacity-50'}
                  transition-all group relative
                `}
              >
                <span className="text-sm font-semibold mb-2">
                  {day.dayName}
                </span>
                {hasEnoughData && (
                  <>
                    <span className="text-2xl font-bold">
                      {day.winRate.toFixed(0)}%
                    </span>
                    <span className="text-xs text-muted-foreground mt-1">
                      {day.games} games
                    </span>
                    <Sparkles className="h-3 w-3 absolute top-2 right-2 opacity-0 group-hover:opacity-70 transition-opacity text-primary-500" />
                  </>
                )}
                {!hasEnoughData && (
                  <span className="text-xs text-muted-foreground">
                    {day.games} games
                  </span>
                )}
              </div>
            )
          })}
        </div>

        {/* Best/Worst Days */}
        {dailyPerformance.filter(d => d.games >= 5).length > 0 && (
          <div className="mt-4 grid grid-cols-2 gap-4">
            {(() => {
              const withEnoughData = dailyPerformance.filter(d => d.games >= 5)
              const best = withEnoughData.reduce((prev, curr) =>
                curr.winRate > prev.winRate ? curr : prev
              )
              const worst = withEnoughData.reduce((prev, curr) =>
                curr.winRate < prev.winRate ? curr : prev
              )

              return (
                <>
                  <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/30">
                    <p className="text-xs text-green-500 font-medium mb-1">Best Day</p>
                    <p className="text-sm font-semibold">{best.dayName}</p>
                    <p className="text-xs text-muted-foreground">
                      {best.winRate.toFixed(1)}% WR ({best.games} games)
                    </p>
                  </div>
                  <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                    <p className="text-xs text-red-500 font-medium mb-1">Challenging Day</p>
                    <p className="text-sm font-semibold">{worst.dayName}</p>
                    <p className="text-xs text-muted-foreground">
                      {worst.winRate.toFixed(1)}% WR ({worst.games} games)
                    </p>
                  </div>
                </>
              )
            })()}
          </div>
        )}
      </Card>
    </div>
  )
}
