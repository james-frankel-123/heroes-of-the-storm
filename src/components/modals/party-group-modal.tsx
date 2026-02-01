'use client'

import * as React from 'react'
import { Users, Sparkles } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { StatCard } from '@/components/ui/stat-card'
import { Badge } from '@/components/ui/badge'
import { StreamingText } from '@/components/commentary/streaming-text'
import { useStreamingCommentary } from '@/lib/hooks/use-streaming-commentary'
import { PartyGroup, PlayerData } from '@/types'
import { getWinRateColor, cn } from '@/lib/utils'

interface PartyGroupModalProps {
  group: PartyGroup
  playerData: PlayerData
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function PartyGroupModal({ group, playerData, open, onOpenChange }: PartyGroupModalProps) {
  const { commentary, isStreaming, error, fetchCommentary } = useStreamingCommentary()

  React.useEffect(() => {
    if (open && group && playerData) {
      fetchCommentary('/api/commentary/party-group', {
        group,
        playerData,
      })
    }
  }, [open, group.membershipKey, playerData?.playerName])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[75vw] max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-2xl">
            <Users className="h-6 w-6 text-accent-cyan" />
            <span>
              {group.partySize === 2 && 'Duo'}
              {group.partySize === 3 && 'Trio'}
              {group.partySize === 4 && 'Quadruple'}
              {group.partySize === 5 && 'Quintuple'}
              {' '}Party
            </span>
          </DialogTitle>
        </DialogHeader>

        <div className="flex gap-6 max-h-[calc(90vh-8rem)] overflow-hidden">
          {/* Left Column: Stats and Details */}
          <div className="flex-1 space-y-6 overflow-y-auto pr-2">
            {/* Party Members */}
            <div className="glass border border-primary-500/30 rounded-lg p-4">
              <h3 className="font-semibold mb-3">Party Members</h3>
              <div className="space-y-2">
                {group.members.map((battletag, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <Badge
                      variant={battletag === playerData.playerName ? 'default' : 'secondary'}
                      className="text-xs font-mono"
                    >
                      {battletag === playerData.playerName ? '★ ' : ''}{battletag}
                    </Badge>
                  </div>
                ))}
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid gap-4 md:grid-cols-2">
              <StatCard label="Games Played" value={group.totalGames} />
              <StatCard
                label="Win Rate"
                value={`${group.winRate.toFixed(1)}%`}
                valueColor={getWinRateColor(group.winRate)}
              />
              <StatCard
                label="Wins"
                value={group.totalWins}
                valueColor="text-gaming-success"
              />
              <StatCard
                label="Losses"
                value={group.totalLosses}
                valueColor="text-gaming-danger"
              />
            </div>

            {/* Top Heroes for Each Member */}
            {group.memberHeroes && Object.keys(group.memberHeroes).length > 0 && (
              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <h3 className="font-semibold mb-3">Top Heroes by Member</h3>
                <div className="space-y-4">
                  {group.members.map((battletag) => {
                    const memberHeroes = group.memberHeroes[battletag]
                    if (!memberHeroes || memberHeroes.length === 0) return null

                    return (
                      <div key={battletag}>
                        <p className="text-xs font-medium text-muted-foreground mb-2">
                          {battletag === playerData.playerName ? (
                            <span className="text-primary-500">★ {battletag.split('#')[0]}</span>
                          ) : (
                            battletag.split('#')[0]
                          )}
                        </p>
                        <div className="space-y-1 pl-3 border-l-2 border-primary-500/20">
                          {memberHeroes.slice(0, 3).map((heroStat) => (
                            <div key={heroStat.hero} className="flex justify-between text-sm">
                              <span className="text-sm">{heroStat.hero}</span>
                              <span className="text-xs text-muted-foreground">
                                Win Rate: {heroStat.winRate.toFixed(1)}% • {heroStat.games}g
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Team Compositions */}
            {group.compositions && group.compositions.length > 0 && (
              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <h3 className="font-semibold mb-3">Hero Compositions</h3>
                <div className="space-y-2">
                  {group.compositions.slice(0, 5).map((comp, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between items-start">
                        <span className="flex-1 text-sm leading-relaxed">{comp.composition}</span>
                      </div>
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-muted-foreground">
                          {comp.wins}W - {comp.losses}L
                        </span>
                        <span className="text-muted-foreground">
                          Win Rate: <span className={cn("font-semibold", getWinRateColor(comp.winRate))}>
                            {comp.winRate.toFixed(1)}%
                          </span> • {comp.games}g
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Best Maps */}
            {group.bestMaps.length > 0 && (
              <div className="glass border border-primary-500/30 rounded-lg p-4">
                <h3 className="font-semibold mb-3">Maps With This Party</h3>
                <div className="space-y-2">
                  {group.bestMaps.slice(0, 5).map((mapStat) => (
                    <div key={mapStat.map} className="flex justify-between text-sm">
                      <span>{mapStat.map}</span>
                      <span className="text-xs text-muted-foreground">
                        Win Rate: <span className={cn("font-semibold", getWinRateColor(mapStat.winRate))}>
                          {mapStat.winRate.toFixed(1)}%
                        </span> • {mapStat.games}g
                      </span>
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
                <h3 className="font-semibold">Party Dynamics Analysis</h3>
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
