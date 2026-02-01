'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Ban, User, Clock } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { DraftTurn } from '@/lib/draft/draft-sequence'
import { getHeroRole } from '@/lib/data/hero-roles'

interface DraftHistoryTimelineProps {
  draftHistory: Array<{
    turn: DraftTurn
    hero: string
    timestamp: number
    battletag?: string
  }>
  compact?: boolean
}

export function DraftHistoryTimeline({
  draftHistory,
  compact = false
}: DraftHistoryTimelineProps) {
  if (draftHistory.length === 0) {
    return (
      <Card className="border-dashed border-border">
        <CardContent className="p-6 text-center">
          <Clock className="h-8 w-8 text-muted-foreground mx-auto mb-2 opacity-50" />
          <p className="text-sm text-muted-foreground">
            Draft history will appear here as actions are taken.
          </p>
        </CardContent>
      </Card>
    )
  }

  if (compact) {
    return (
      <div className="space-y-1">
        {draftHistory.map((action, idx) => {
          const isBan = action.turn.action === 'ban'
          const teamColor = action.turn.team === 'blue'
            ? 'text-blue-400'
            : 'text-red-400'
          const bgColor = action.turn.team === 'blue'
            ? 'bg-blue-500/10 border-blue-500/30'
            : 'bg-red-500/10 border-red-500/30'

          return (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.02 }}
              className={`flex items-center gap-2 p-2 rounded-lg border ${bgColor}`}
            >
              {isBan ? (
                <Ban className={`h-3 w-3 ${teamColor}`} />
              ) : (
                <User className={`h-3 w-3 ${teamColor}`} />
              )}
              <span className="text-xs font-medium flex-1">{action.hero}</span>
              <span className={`text-xs font-bold ${teamColor}`}>
                {action.turn.team === 'blue' ? 'B' : 'R'}
              </span>
            </motion.div>
          )
        })}
      </div>
    )
  }

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Clock className="h-4 w-4" />
          Draft History
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {draftHistory.map((action, idx) => {
            const isBan = action.turn.action === 'ban'
            const teamColor = action.turn.team === 'blue'
              ? 'text-blue-400'
              : 'text-red-400'
            const bgColor = action.turn.team === 'blue'
              ? 'bg-blue-500/10 border-blue-500/30'
              : 'bg-red-500/10 border-red-500/30'

            const role = getHeroRole(action.hero)
            const playerName = action.battletag?.split('#')[0]

            // Format timestamp
            const time = new Date(action.timestamp)
            const formattedTime = time.toLocaleTimeString('en-US', {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit'
            })

            return (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05 }}
                className={`p-3 rounded-lg border ${bgColor}`}
              >
                <div className="flex items-start gap-3">
                  {/* Turn Number */}
                  <div className="flex-shrink-0 flex items-center justify-center h-8 w-8 rounded-full bg-background/50 border border-border">
                    <span className="text-xs font-bold">{idx + 1}</span>
                  </div>

                  {/* Action Details */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      {isBan ? (
                        <Ban className={`h-4 w-4 ${teamColor}`} />
                      ) : (
                        <User className={`h-4 w-4 ${teamColor}`} />
                      )}
                      <Badge
                        variant="outline"
                        className={`text-xs ${teamColor} border-current`}
                      >
                        {action.turn.team.toUpperCase()} {isBan ? 'BAN' : 'PICK'}
                      </Badge>
                      {action.turn.phase && (
                        <Badge variant="outline" className="text-xs">
                          Phase {action.turn.phase}
                        </Badge>
                      )}
                    </div>

                    <div className="mt-1">
                      <span className="font-bold text-sm">{action.hero}</span>
                      {role && (
                        <span className="text-xs text-muted-foreground ml-2">
                          • {role}
                        </span>
                      )}
                    </div>

                    {playerName && (
                      <div className="text-xs text-muted-foreground mt-1">
                        Picked by {playerName}
                      </div>
                    )}

                    <div className="text-xs text-muted-foreground/50 mt-1">
                      {formattedTime}
                    </div>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
