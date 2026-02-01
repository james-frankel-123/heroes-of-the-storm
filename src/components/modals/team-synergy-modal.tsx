'use client'

import * as React from 'react'
import { PlayerData } from '@/types'
import { DuoStats, parseDuoHeroes } from '@/lib/data/team-compositions'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { StreamingText } from '@/components/commentary/streaming-text'
import { useStreamingCommentary } from '@/lib/hooks/use-streaming-commentary'
import { Trophy, Target, TrendingUp, Sparkles, Users } from 'lucide-react'
import { getWinRateColor } from '@/lib/utils'
import { classifyHeroRole } from '@/lib/data/transform'

interface TeamSynergyModalProps {
  synergy: DuoStats
  playerData: PlayerData
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function TeamSynergyModal({
  synergy,
  playerData,
  open,
  onOpenChange,
}: TeamSynergyModalProps) {
  const { commentary, isStreaming, error, fetchCommentary } = useStreamingCommentary()

  const [hero1, hero2] = parseDuoHeroes(synergy.heroes)
  const role1 = classifyHeroRole(hero1)
  const role2 = classifyHeroRole(hero2)

  React.useEffect(() => {
    if (open && synergy && playerData) {
      fetchCommentary('/api/commentary/synergy', {
        hero1,
        hero2,
        synergy,
        playerData,
      })
    }
  }, [open, hero1, hero2, playerData?.playerName])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[75vw] max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-2xl">
            <Users className="h-6 w-6 text-primary-500" />
            <span>Duo Synergy</span>
          </DialogTitle>
        </DialogHeader>

        <div className="flex gap-6 max-h-[calc(90vh-8rem)] overflow-hidden">
          {/* Left Column: Stats and Details */}
          <div className="flex-1 space-y-6 overflow-y-auto pr-2">
            {/* Hero Pair */}
            <div className="flex items-center justify-center gap-4">
              <div className="flex flex-col items-center gap-2">
                <span className="text-lg font-bold">{hero1}</span>
                <Badge variant={role1.toLowerCase().replace(' ', '') as any} className="text-xs">
                  {role1}
                </Badge>
              </div>
              <Users className="h-8 w-8 text-primary-500" />
              <div className="flex flex-col items-center gap-2">
                <span className="text-lg font-bold">{hero2}</span>
                <Badge variant={role2.toLowerCase().replace(' ', '') as any} className="text-xs">
                  {role2}
                </Badge>
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Trophy className="h-4 w-4" />
                  <span className="text-xs font-medium">Games</span>
                </div>
                <p className="text-2xl font-bold">{synergy.games}</p>
              </div>

              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Target className="h-4 w-4" />
                  <span className="text-xs font-medium">Win Rate</span>
                </div>
                <p className={`text-2xl font-bold ${getWinRateColor(synergy.winRate)}`}>
                  {synergy.winRate.toFixed(1)}%
                </p>
              </div>

              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <TrendingUp className="h-4 w-4" />
                  <span className="text-xs font-medium">Wins</span>
                </div>
                <p className="text-2xl font-bold text-gaming-success">{synergy.wins}</p>
              </div>

              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <TrendingUp className="h-4 w-4 rotate-180" />
                  <span className="text-xs font-medium">Losses</span>
                </div>
                <p className="text-2xl font-bold text-gaming-danger">{synergy.losses}</p>
              </div>
            </div>
          </div>

          {/* Right Column: AI Analysis */}
          <div className="flex-1 flex flex-col">
            <div className="glass border border-accent-cyan/30 rounded-lg p-4 flex flex-col h-full overflow-hidden">
              <div className="flex items-center gap-2 mb-3 flex-shrink-0">
                <Sparkles className="h-4 w-4 text-accent-cyan" />
                <h3 className="font-semibold">AI Analysis</h3>
              </div>
              <div className="overflow-y-auto flex-1">
                {error ? (
                  <p className="text-sm text-gaming-danger">{error}</p>
                ) : (
                  <StreamingText
                    text={commentary}
                    isStreaming={isStreaming}
                    className="text-sm text-muted-foreground leading-relaxed"
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
