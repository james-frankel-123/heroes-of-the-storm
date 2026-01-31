'use client'

import * as React from 'react'
import { Zap, Sparkles } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatPercent } from '@/lib/utils'
import { PowerPick, PlayerData } from '@/types'
import { useStreamingCommentary } from '@/lib/hooks/use-streaming-commentary'
import { StreamingText } from '@/components/commentary/streaming-text'

interface PowerPicksProps {
  powerPicks: PowerPick[]
  playerData?: PlayerData
}

interface PowerPickCardProps {
  pick: PowerPick
  index: number
  playerData?: PlayerData
}

function PowerPickCard({ pick, index, playerData }: PowerPickCardProps) {
  const { commentary, isStreaming, error, fetchCommentary } = useStreamingCommentary()

  React.useEffect(() => {
    if (playerData) {
      fetchCommentary('/api/commentary/power-pick', {
        powerPick: pick,
        playerData,
      })
    }
  }, [pick.hero, pick.map, playerData])

  return (
    <div
      className="group relative overflow-hidden rounded-lg border border-gaming-success/30 bg-gaming-success/5 p-4 transition-all hover:scale-[1.02] hover:border-gaming-success/60"
    >
      <div className="absolute right-2 top-2 text-4xl font-bold text-gaming-success/10">
        {index + 1}
      </div>
      <div className="relative z-10 space-y-2">
        <p className="text-lg font-bold">{pick.hero}</p>
        <p className="text-sm text-muted-foreground">{pick.map}</p>
        <div className="flex items-end gap-2">
          <span className="text-2xl font-bold text-gaming-success">
            {formatPercent(pick.winRate, 1)}
          </span>
          <span className="mb-1 text-xs text-muted-foreground">
            {pick.games} games
          </span>
        </div>

        {/* AI Commentary */}
        {playerData && (
          <div className="mt-3 pt-3 border-t border-gaming-success/20">
            <div className="flex items-center gap-1 mb-2">
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
    </div>
  )
}

export function PowerPicks({ powerPicks, playerData }: PowerPicksProps) {
  if (powerPicks.length === 0) {
    return null
  }

  return (
    <Card className="glass border-primary-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-gaming-success" />
          Power Picks (65%+ Win Rate)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {powerPicks.map((pick, index) => (
            <PowerPickCard
              key={`${pick.hero}-${pick.map}`}
              pick={pick}
              index={index}
              playerData={playerData}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
