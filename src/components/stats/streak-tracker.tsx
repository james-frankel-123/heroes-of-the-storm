'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { Streak } from '@/lib/data/statistics'

interface StreakTrackerProps {
  currentStreak: Streak | null
  longestWinStreak: Streak | null
  longestLossStreak: Streak | null
  onClick?: () => void
}

export function StreakTracker({
  currentStreak,
  longestWinStreak,
  longestLossStreak,
  onClick,
}: StreakTrackerProps) {
  return (
    <Card
      className="glass border-primary-500/30 cursor-pointer hover:border-primary-500/50 transition-all"
      onClick={onClick}
    >
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          {currentStreak?.type === 'win' ? (
            <TrendingUp className="h-5 w-5 text-gaming-success" />
          ) : (
            <TrendingDown className="h-5 w-5 text-gaming-danger" />
          )}
          Current Streak
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {currentStreak ? (
          <>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Status:</span>
              <Badge
                variant={currentStreak.type === 'win' ? 'default' : 'destructive'}
                className="text-lg px-4 py-1"
              >
                {currentStreak.type === 'win' ? 'W' : 'L'}{currentStreak.length}
              </Badge>
            </div>

            <div className="pt-4 border-t border-border space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Best Win Streak:</span>
                <span className="font-medium text-gaming-success">
                  {longestWinStreak ? `W${longestWinStreak.length}` : 'N/A'}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Longest Loss Streak:</span>
                <span className="font-medium text-gaming-danger">
                  {longestLossStreak ? `L${longestLossStreak.length}` : 'N/A'}
                </span>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center text-muted-foreground py-4">
            No active streak (need 2+ consecutive games)
          </div>
        )}
      </CardContent>
    </Card>
  )
}
