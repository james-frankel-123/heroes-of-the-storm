'use client'

import * as React from 'react'
import { Loader2 } from 'lucide-react'
import { PasswordGate } from '@/components/auth/password-gate'
import { Skeleton } from '@/components/ui/skeleton'
import { usePlayerData } from '@/lib/hooks/use-data'
import { useStatisticsData } from '@/lib/hooks/use-statistics-data'
import { useStatisticsAnalysis } from '@/lib/hooks/use-statistics-analysis'
import { StreakTracker } from '@/components/stats/streak-tracker'
import { SummaryDashboard } from '@/components/stats/summary-dashboard'
import { AIAnalysisPanel } from '@/components/stats/ai-analysis-panel'
import { ClickableMetric } from '@/components/stats/clickable-metric'
import { PerformanceTrends } from '@/components/stats/performance-trends'
import { HeroAnalytics } from '@/components/stats/hero-analytics'
import { KDAStatistics } from '@/components/stats/kda-statistics'
import { TemporalAnalysis } from '@/components/stats/temporal-analysis'

export default function StatsPage() {
  const { data: playerData } = usePlayerData()
  const { statistics, isLoading, error } = useStatisticsData()
  const {
    contextCards,
    addContextCard,
    removeContextCard,
    sendQuestion,
    isStreaming,
    currentResponse,
    conversationHistory,
    clearAll,
  } = useStatisticsAnalysis()

  if (isLoading) {
    return (
      <PasswordGate requiredPassword="ronpaul2012" storageKey="protected_pages_auth">
        <div className="flex h-screen">
          <div className="flex-1 overflow-y-auto pr-6 space-y-8">
            <div>
              <Skeleton className="h-12 w-96" />
              <Skeleton className="mt-2 h-6 w-64" />
            </div>
            <div className="grid gap-6 md:grid-cols-2">
              {[1, 2].map((i) => (
                <Skeleton key={i} className="h-64" />
              ))}
            </div>
          </div>
          <div className="w-96 border-l border-border">
            <Skeleton className="h-full" />
          </div>
        </div>
      </PasswordGate>
    )
  }

  if (error || !statistics) {
    return (
      <PasswordGate requiredPassword="ronpaul2012" storageKey="protected_pages_auth">
        <div className="flex h-screen items-center justify-center">
          <div className="text-center">
            <p className="text-muted-foreground">
              {error ? 'Error loading statistics. Please try again.' : 'No data available.'}
            </p>
          </div>
        </div>
      </PasswordGate>
    )
  }

  return (
    <PasswordGate requiredPassword="ronpaul2012" storageKey="protected_pages_auth">
      <div className="flex h-screen">
        {/* Main Content - 70% width */}
        <div className="flex-1 overflow-y-auto pr-6 space-y-8 pb-8">
          <div>
            <h1 className="text-4xl font-bold tracking-tight glow">Statistics</h1>
            <p className="mt-2 text-muted-foreground">
              Advanced statistical analysis and trends
            </p>
            <p className="mt-1 text-sm text-primary-500">
              ✨ Click any statistic to analyze it with AI
            </p>
          </div>

          {/* Summary Cards */}
          <div className="grid gap-6 md:grid-cols-2">
            {/* Streak Tracker */}
            <ClickableMetric
              context={{
                type: 'streak',
                label: statistics.currentStreak
                  ? `${statistics.currentStreak.type === 'win' ? 'Win' : 'Loss'} Streak`
                  : 'No Streak',
                value: statistics.currentStreak?.length || 0,
                relatedMetrics: {
                  longestWinStreak: statistics.longestWinStreak?.length || 0,
                  longestLossStreak: statistics.longestLossStreak?.length || 0,
                },
              }}
              onAddToChat={addContextCard}
            >
              <StreakTracker
                currentStreak={statistics.currentStreak}
                longestWinStreak={statistics.longestWinStreak}
                longestLossStreak={statistics.longestLossStreak}
              />
            </ClickableMetric>

            {/* Summary Dashboard */}
            <ClickableMetric
              context={{
                type: 'metric',
                label: 'Performance Summary',
                value: statistics.last10WinRate,
                relatedMetrics: {
                  last10WinRate: statistics.last10WinRate,
                  last20WinRate: statistics.last20WinRate,
                  last50WinRate: statistics.last50WinRate,
                  consistencyScore: statistics.consistencyScore,
                  kda: statistics.overallKDA.kda,
                },
              }}
              onAddToChat={addContextCard}
            >
              <SummaryDashboard statistics={statistics} />
            </ClickableMetric>
          </div>

          {/* Performance Trends */}
          <PerformanceTrends
            winRateOverTime={statistics.winRateOverTime}
            gamesOverTime={statistics.gamesOverTime}
            onPointClick={addContextCard}
          />

          {/* KDA Statistics */}
          <KDAStatistics
            overallKDA={statistics.overallKDA}
            onAddToChat={addContextCard}
          />

          {/* Hero Analytics */}
          <HeroAnalytics
            kdaByHero={statistics.kdaByHero}
            onRowClick={addContextCard}
          />

          {/* Temporal Analysis */}
          <TemporalAnalysis
            hourlyPerformance={statistics.hourlyPerformance}
            dailyPerformance={statistics.dailyPerformance}
            onCellClick={addContextCard}
          />
        </div>

        {/* AI Chat Panel */}
        <AIAnalysisPanel
          contextCards={contextCards}
          onRemoveCard={removeContextCard}
          onSendQuestion={(question) => sendQuestion(question, playerData)}
          conversationHistory={conversationHistory}
          isStreaming={isStreaming}
          currentResponse={currentResponse}
          onClearAll={clearAll}
        />
      </div>
    </PasswordGate>
  )
}
