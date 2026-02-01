'use client'

import * as React from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  TooltipProps,
} from 'recharts'
import { TimeSeriesPoint } from '@/lib/data/statistics'
import { ContextData } from './clickable-metric'
import { Card } from '@/components/ui/card'
import { Sparkles } from 'lucide-react'

interface PerformanceTrendsProps {
  winRateOverTime: TimeSeriesPoint[]
  gamesOverTime: TimeSeriesPoint[]
  onPointClick: (context: ContextData) => void
}

export function PerformanceTrends({
  winRateOverTime,
  gamesOverTime,
  onPointClick,
}: PerformanceTrendsProps) {
  const handleWinRateClick = (data: any) => {
    if (!data || !data.activePayload || data.activePayload.length === 0) return

    const point = data.activePayload[0].payload as TimeSeriesPoint
    onPointClick({
      type: 'chart-point',
      label: `Win Rate on ${new Date(point.date).toLocaleDateString()}`,
      value: point.value,
      timeRange: {
        start: point.date,
        end: point.date,
        granularity: 'daily',
      },
      relatedMetrics: {
        games: point.games,
        wins: Math.round((point.value / 100) * point.games),
        losses: point.games - Math.round((point.value / 100) * point.games),
      },
    })
  }

  const handleGamesClick = (data: any) => {
    if (!data) return

    const point = data as TimeSeriesPoint
    onPointClick({
      type: 'chart-point',
      label: `Games Played on ${new Date(point.date).toLocaleDateString()}`,
      value: point.games,
      timeRange: {
        start: point.date,
        end: point.date,
        granularity: 'daily',
      },
      relatedMetrics: {
        winRate: point.value,
      },
    })
  }

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: TooltipProps<number, string>) => {
    if (!active || !payload || payload.length === 0) return null

    const data = payload[0].payload as TimeSeriesPoint
    const isWinRate = payload[0].dataKey === 'value'

    return (
      <div className="bg-card/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg">
        <p className="text-sm font-medium mb-2">
          {new Date(data.date).toLocaleDateString()}
        </p>
        {isWinRate ? (
          <>
            <p className="text-xs text-muted-foreground">
              Win Rate: <span className="font-semibold text-primary-500">{data.value.toFixed(1)}%</span>
            </p>
            <p className="text-xs text-muted-foreground">
              Games: {data.games}
            </p>
          </>
        ) : (
          <>
            <p className="text-xs text-muted-foreground">
              Games: <span className="font-semibold text-primary-500">{data.games}</span>
            </p>
            <p className="text-xs text-muted-foreground">
              Win Rate: {data.value.toFixed(1)}%
            </p>
          </>
        )}
        <p className="text-xs text-primary-500/70 mt-2 flex items-center gap-1">
          <Sparkles className="h-3 w-3" />
          Click to analyze
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Win Rate Over Time */}
      <Card className="p-6 glass-card">
        <div className="mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            Win Rate Over Time
            <Sparkles className="h-4 w-4 text-primary-500/50" />
          </h3>
          <p className="text-sm text-muted-foreground">
            Click any point to analyze performance trends
          </p>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={winRateOverTime}
            onClick={handleWinRateClick}
            className="cursor-pointer"
          >
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis
              dataKey="date"
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <YAxis
              domain={[0, 100]}
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="value"
              stroke="hsl(var(--primary-500))"
              strokeWidth={3}
              dot={{ fill: 'hsl(var(--primary-500))', r: 5, strokeWidth: 2, stroke: 'hsl(var(--background))' }}
              activeDot={{ r: 8, fill: 'hsl(var(--primary-400))', stroke: 'hsl(var(--primary-600))', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Games Played Over Time */}
      <Card className="p-6 glass-card">
        <div className="mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            Games Played Over Time
            <Sparkles className="h-4 w-4 text-primary-500/50" />
          </h3>
          <p className="text-sm text-muted-foreground">
            Click any bar to analyze activity patterns
          </p>
        </div>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={gamesOverTime}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis
              dataKey="date"
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <YAxis
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar
              dataKey="games"
              fill="hsl(var(--primary-500))"
              radius={[4, 4, 0, 0]}
              onClick={handleGamesClick}
              className="cursor-pointer hover:opacity-80 transition-opacity"
            />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}
