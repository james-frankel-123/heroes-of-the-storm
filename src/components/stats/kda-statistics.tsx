'use client'

import * as React from 'react'
import { Sparkles, Target, Skull, HandHelping } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { ClickableMetric, ContextData } from './clickable-metric'

interface KDAStatisticsProps {
  overallKDA: {
    kills: number
    deaths: number
    assists: number
    kda: number
    avgKills: number
    avgDeaths: number
    avgAssists: number
    totalGames: number
  }
  onAddToChat: (context: ContextData) => void
}

export function KDAStatistics({ overallKDA, onAddToChat }: KDAStatisticsProps) {
  return (
    <Card className="p-6 glass-card">
      <div className="mb-6">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          Combat Statistics
          <Sparkles className="h-4 w-4 text-primary-500/50" />
        </h3>
        <p className="text-sm text-muted-foreground">
          Overall KDA performance across all games
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Overall KDA Ratio */}
        <ClickableMetric
          context={{
            type: 'metric',
            label: 'Overall KDA Ratio',
            value: overallKDA.kda,
            relatedMetrics: {
              avgKills: overallKDA.avgKills,
              avgDeaths: overallKDA.avgDeaths,
              avgAssists: overallKDA.avgAssists,
              totalGames: overallKDA.totalGames,
            },
          }}
          onAddToChat={onAddToChat}
        >
          <div className="p-4 rounded-lg border border-border bg-card/30 hover:bg-card/50 transition-colors">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">KDA Ratio</span>
              <Target className="h-4 w-4 text-primary-500" />
            </div>
            <div
              className={`text-3xl font-bold ${
                overallKDA.kda >= 3
                  ? 'text-green-500'
                  : overallKDA.kda >= 2
                  ? 'text-primary-500'
                  : 'text-muted-foreground'
              }`}
            >
              {overallKDA.kda.toFixed(2)}
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              {overallKDA.avgKills.toFixed(1)} / {overallKDA.avgDeaths.toFixed(1)} /{' '}
              {overallKDA.avgAssists.toFixed(1)}
            </div>
          </div>
        </ClickableMetric>

        {/* Average Kills */}
        <ClickableMetric
          context={{
            type: 'metric',
            label: 'Average Kills per Game',
            value: overallKDA.avgKills,
            relatedMetrics: {
              totalKills: overallKDA.kills,
              totalGames: overallKDA.totalGames,
              kda: overallKDA.kda,
            },
          }}
          onAddToChat={onAddToChat}
        >
          <div className="p-4 rounded-lg border border-border bg-card/30 hover:bg-card/50 transition-colors">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Avg Kills</span>
              <Target className="h-4 w-4 text-green-500" />
            </div>
            <div className="text-3xl font-bold text-green-500">
              {overallKDA.avgKills.toFixed(1)}
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              {overallKDA.kills.toLocaleString()} total
            </div>
          </div>
        </ClickableMetric>

        {/* Average Deaths */}
        <ClickableMetric
          context={{
            type: 'metric',
            label: 'Average Deaths per Game',
            value: overallKDA.avgDeaths,
            relatedMetrics: {
              totalDeaths: overallKDA.deaths,
              totalGames: overallKDA.totalGames,
              kda: overallKDA.kda,
            },
          }}
          onAddToChat={onAddToChat}
        >
          <div className="p-4 rounded-lg border border-border bg-card/30 hover:bg-card/50 transition-colors">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Avg Deaths</span>
              <Skull className="h-4 w-4 text-red-500" />
            </div>
            <div className="text-3xl font-bold text-red-500">
              {overallKDA.avgDeaths.toFixed(1)}
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              {overallKDA.deaths.toLocaleString()} total
            </div>
          </div>
        </ClickableMetric>

        {/* Average Assists */}
        <ClickableMetric
          context={{
            type: 'metric',
            label: 'Average Assists per Game',
            value: overallKDA.avgAssists,
            relatedMetrics: {
              totalAssists: overallKDA.assists,
              totalGames: overallKDA.totalGames,
              kda: overallKDA.kda,
            },
          }}
          onAddToChat={onAddToChat}
        >
          <div className="p-4 rounded-lg border border-border bg-card/30 hover:bg-card/50 transition-colors">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Avg Assists</span>
              <HandHelping className="h-4 w-4 text-blue-500" />
            </div>
            <div className="text-3xl font-bold text-blue-500">
              {overallKDA.avgAssists.toFixed(1)}
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              {overallKDA.assists.toLocaleString()} total
            </div>
          </div>
        </ClickableMetric>
      </div>

      {/* Combat Style Indicator */}
      <div className="mt-6 p-4 rounded-lg border border-border bg-muted/20">
        <p className="text-sm font-medium mb-2">Combat Style</p>
        <p className="text-xs text-muted-foreground">
          {overallKDA.avgKills > overallKDA.avgAssists
            ? '⚔️ Aggressive - You secure more kills than assists'
            : overallKDA.avgAssists > overallKDA.avgKills * 2
            ? '🤝 Supportive - You excel at enabling your team'
            : '⚖️ Balanced - You mix kills and assists effectively'}
        </p>
      </div>
    </Card>
  )
}
