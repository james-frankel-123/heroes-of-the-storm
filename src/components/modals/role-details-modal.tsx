'use client'

import * as React from 'react'
import { PlayerData } from '@/types'
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

interface RoleStats {
  role: string
  wins: number
  games: number
  winRate: number
}

interface RoleDetailsModalProps {
  role: string
  stats: RoleStats
  playerData: PlayerData
  open: boolean
  onOpenChange: (open: boolean) => void
}

const roleVariantMap: Record<string, any> = {
  'Ranged Assassin': 'ranged',
  'Tank': 'tank',
  'Healer': 'healer',
  'Bruiser': 'bruiser',
  'Melee Assassin': 'melee',
  'Support': 'support',
  'Unknown': 'outline',
}

export function RoleDetailsModal({
  role,
  stats,
  playerData,
  open,
  onOpenChange,
}: RoleDetailsModalProps) {
  const { commentary, isStreaming, error, fetchCommentary } = useStreamingCommentary()

  // Get heroes in this role
  const heroesInRole = React.useMemo(() => {
    return playerData.heroStats
      .filter(h => h.role === role)
      .sort((a, b) => b.winRate - a.winRate)
  }, [playerData.heroStats, role])

  React.useEffect(() => {
    if (open && role && playerData) {
      fetchCommentary('/api/commentary/role', {
        role,
        stats,
        heroesInRole,
        playerData,
      })
    }
  }, [open, role, playerData?.playerName])

  const losses = stats.games - stats.wins

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[75vw] max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-2xl">
            <Users className="h-6 w-6 text-primary-500" />
            <span>{role}</span>
            <Badge variant={roleVariantMap[role] || 'outline'} className="text-xs">
              Role Performance
            </Badge>
          </DialogTitle>
        </DialogHeader>

        <div className="flex gap-6 max-h-[calc(90vh-8rem)] overflow-hidden">
          {/* Left Column: Stats and Details */}
          <div className="flex-1 space-y-6 overflow-y-auto pr-2">
            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Trophy className="h-4 w-4" />
                  <span className="text-xs font-medium">Games</span>
                </div>
                <p className="text-2xl font-bold">{stats.games}</p>
              </div>

              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Target className="h-4 w-4" />
                  <span className="text-xs font-medium">Win Rate</span>
                </div>
                <p className={`text-2xl font-bold ${getWinRateColor(stats.winRate)}`}>
                  {stats.winRate.toFixed(1)}%
                </p>
              </div>

              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <TrendingUp className="h-4 w-4" />
                  <span className="text-xs font-medium">Wins</span>
                </div>
                <p className="text-2xl font-bold text-gaming-success">{stats.wins}</p>
              </div>

              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <TrendingUp className="h-4 w-4 rotate-180" />
                  <span className="text-xs font-medium">Losses</span>
                </div>
                <p className="text-2xl font-bold text-gaming-danger">{losses}</p>
              </div>
            </div>

            {/* Heroes in Role */}
            {heroesInRole.length > 0 && (
              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Users className="h-4 w-4 text-primary-500" />
                  <h3 className="font-semibold">Heroes Played ({heroesInRole.length})</h3>
                </div>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {heroesInRole.map((hero) => (
                    <div
                      key={hero.hero}
                      className="flex items-center justify-between text-sm border-b border-border/50 pb-2 last:border-0"
                    >
                      <span className="font-medium">{hero.hero}</span>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-muted-foreground">
                          {hero.wins}W - {hero.losses}L
                        </span>
                        <span className={`font-semibold ${getWinRateColor(hero.winRate)}`}>
                          {hero.winRate.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
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
