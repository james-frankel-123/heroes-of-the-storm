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
      value: point.winRate,
      timeRange: {
        start: point.date,
        end: point.date,
        granularity: 'daily',
      },
      relatedMetrics: {
        games: point.games,
        wins: point.wins,
        losses: point.losses,
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
        winRate: point.winRate,
      },
    })
  }

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || payload.length === 0) return null

    const data = payload[0].payload as TimeSeriesPoint
    const isWinRate = payload[0].dataKey === 'winRate'

    return (
      <div className="bg-slate-900/95 backdrop-blur-sm border border-slate-700 rounded-lg p-3 shadow-lg">
        <p className="text-sm font-medium mb-2 text-slate-100">
          {new Date(data.date).toLocaleDateString()}
        </p>
        {isWinRate ? (
          <>
            <p className="text-xs text-slate-300">
              Win Rate: <span className="font-semibold text-blue-400">{data.winRate.toFixed(1)}%</span>
            </p>
            <p className="text-xs text-slate-300">
              Games: {data.games}
            </p>
          </>
        ) : (
          <>
            <p className="text-xs text-slate-300">
              Games: <span className="font-semibold text-blue-400">{data.games}</span>
            </p>
            <p className="text-xs text-slate-300">
              Win Rate: {data.winRate.toFixed(1)}%
            </p>
          </>
        )}
        <p className="text-xs text-blue-400/80 mt-2 flex items-center gap-1">
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
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
            <XAxis
              dataKey="date"
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              stroke="rgba(148, 163, 184, 0.8)"
              fontSize={12}
              tick={{ fill: 'rgba(148, 163, 184, 0.9)' }}
            />
            <YAxis
              domain={[0, 100]}
              stroke="rgba(148, 163, 184, 0.8)"
              fontSize={12}
              tickFormatter={(value) => `${value}%`}
              tick={{ fill: 'rgba(148, 163, 184, 0.9)' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="winRate"
              stroke="#60a5fa"
              strokeWidth={4}
              dot={{ fill: '#3b82f6', r: 6, strokeWidth: 2, stroke: '#1e40af' }}
              activeDot={{ r: 9, fill: '#60a5fa', stroke: '#1e3a8a', strokeWidth: 3 }}
              filter="drop-shadow(0 0 8px rgba(96, 165, 250, 0.4))"
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
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
            <XAxis
              dataKey="date"
              tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              stroke="rgba(148, 163, 184, 0.8)"
              fontSize={12}
              tick={{ fill: 'rgba(148, 163, 184, 0.9)' }}
            />
            <YAxis
              stroke="rgba(148, 163, 184, 0.8)"
              fontSize={12}
              tick={{ fill: 'rgba(148, 163, 184, 0.9)' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar
              dataKey="games"
              fill="#3b82f6"
              radius={[4, 4, 0, 0]}
              onClick={handleGamesClick}
              className="cursor-pointer hover:opacity-80 transition-opacity"
              stroke="#60a5fa"
              strokeWidth={1}
            />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}
