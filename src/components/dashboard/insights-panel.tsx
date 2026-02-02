'use client'

import * as React from 'react'
import { Sparkles, TrendingUp, AlertTriangle, Lightbulb, Info, Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { Insight } from '@/types'

interface InsightsPanelProps {
  insights: Insight[]
  isLoading?: boolean
}

const iconMap = {
  success: TrendingUp,
  warning: AlertTriangle,
  info: Info,
  tip: Lightbulb,
}

const colorMap = {
  success: 'text-gaming-success border-gaming-success/30 bg-gaming-success/5',
  warning: 'text-gaming-warning border-gaming-warning/30 bg-gaming-warning/5',
  info: 'text-primary-500 border-primary-500/30 bg-primary-500/5',
  tip: 'text-accent-cyan border-accent-cyan/30 bg-accent-cyan/5',
}

export function InsightsPanel({ insights, isLoading = false }: InsightsPanelProps) {
  return (
    <Card className="glass border-primary-500/30 h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-accent-cyan" />
          Insights
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="rounded-lg border border-dashed border-primary-500/50 p-4 text-center">
            <Loader2 className="mx-auto h-8 w-8 text-accent-cyan animate-spin" />
            <p className="mt-2 text-sm font-medium">Analyzing your performance...</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Reviewing your stats
            </p>
          </div>
        ) : insights.length > 0 ? (
          insights.map((insight, index) => {
            const Icon = iconMap[insight.type]
            return (
              <div
                key={index}
                className={cn(
                  'rounded-lg border p-4',
                  colorMap[insight.type]
                )}
              >
                <div className="flex gap-3">
                  <div className="mt-0.5">
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="flex-1 space-y-1">
                    <p className="font-semibold text-sm">{insight.title}</p>
                    <p className="text-xs text-muted-foreground">
                      {insight.description}
                    </p>
                  </div>
                </div>
              </div>
            )
          })
        ) : (
          <div className="rounded-lg border border-dashed border-primary-500/50 p-4 text-center">
            <Sparkles className="mx-auto h-8 w-8 text-accent-cyan" />
            <p className="mt-2 text-sm font-medium">No insights available</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Play more games to unlock analytics
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
