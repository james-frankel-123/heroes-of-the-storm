'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import {
  Sparkles,
  TrendingUp,
  AlertTriangle,
  Lightbulb,
  Target,
  Trophy,
  Star,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { usePlayerData } from '@/lib/hooks/use-data'
import { generateInsights } from '@/lib/data/transform'
import { Loader2 } from 'lucide-react'
import { PlayerDataError } from '@/components/error-boundary/player-data-error'

const colorMap = {
  success: {
    border: 'border-gaming-success/30',
    bg: 'bg-gaming-success/5',
    text: 'text-gaming-success',
  },
  warning: {
    border: 'border-gaming-warning/30',
    bg: 'bg-gaming-warning/5',
    text: 'text-gaming-warning',
  },
  info: {
    border: 'border-primary-500/30',
    bg: 'bg-primary-500/5',
    text: 'text-primary-500',
  },
  tip: {
    border: 'border-accent-cyan/30',
    bg: 'bg-accent-cyan/5',
    text: 'text-accent-cyan',
  },
}

// Icon mapping for generated insights
const iconMap: Record<string, any> = {
  '🗺️': TrendingUp,
  '🏆': Trophy,
  '⚠️': AlertTriangle,
  '✨': Sparkles,
  '🎯': Target,
  '⭐': Star,
}

export default function InsightsPage() {
  const { data, isLoading, error } = usePlayerData()

  // Generate dynamic insights from player data and adapt to UI format
  const insights = React.useMemo(() => {
    if (!data) return []

    const generatedInsights = generateInsights(data)

    // Transform generated insights to include icon components and additional fields
    return generatedInsights.map((insight) => ({
      ...insight,
      icon: iconMap[insight.icon || '✨'] || Sparkles,
      recommendation: 'Continue building on this performance.',
      stats: { winRate: 0, games: 0 }, // Stats are embedded in description
    }))
  }, [data])

  // Show loading state
  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="text-center">
          <Loader2 className="mx-auto h-12 w-12 animate-spin text-primary-500" />
          <p className="mt-4 text-sm text-muted-foreground">Loading insights...</p>
        </div>
      </div>
    )
  }

  // Show error state
  if (error || !data) {
    return <PlayerDataError error={error} reset={() => window.location.reload()} />
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3">
          <Sparkles className="h-10 w-10 text-accent-cyan" />
          <div>
            <h1 className="text-4xl font-bold tracking-tight glow">
              Smart Insights
            </h1>
            <p className="mt-2 text-muted-foreground">
              AI-powered recommendations to improve your gameplay
            </p>
          </div>
        </div>
      </motion.div>

      {/* Insights Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {insights.map((insight, index) => {
          const colors = colorMap[insight.type as keyof typeof colorMap]
          const Icon = insight.icon

          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card
                className={`glass group h-full border-2 transition-all hover:scale-[1.02] ${colors.border}`}
              >
                <CardHeader>
                  <div className="flex items-start gap-4">
                    <div
                      className={`rounded-lg p-3 ${colors.bg} ${colors.text}`}
                    >
                      <Icon className="h-6 w-6" />
                    </div>
                    <div className="flex-1">
                      <CardTitle className="text-lg">{insight.title}</CardTitle>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    {insight.description}
                  </p>
                  <div
                    className={`rounded-lg border p-3 ${colors.border} ${colors.bg}`}
                  >
                    <div className="flex items-start gap-2">
                      <Lightbulb className={`h-4 w-4 ${colors.text} mt-0.5`} />
                      <div>
                        <p className={`text-xs font-semibold ${colors.text}`}>
                          Recommendation
                        </p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {insight.recommendation}
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )
        })}
      </div>

      {/* Coming Soon */}
      <Card className="glass border-primary-500/30">
        <CardContent className="py-12 text-center">
          <Sparkles className="mx-auto h-12 w-12 text-accent-cyan" />
          <h3 className="mt-4 text-xl font-semibold">More Insights Coming Soon</h3>
          <p className="mt-2 text-sm text-muted-foreground">
            We&apos;re working on advanced AI-powered analytics including:
          </p>
          <ul className="mt-4 space-y-2 text-sm text-muted-foreground">
            <li>🎯 Draft phase suggestions and counter-picks</li>
            <li>📊 Win rate prediction before matches</li>
            <li>🔥 Hot streak detection and momentum tracking</li>
            <li>🌟 Talent build optimization recommendations</li>
            <li>👥 Team synergy analysis</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
