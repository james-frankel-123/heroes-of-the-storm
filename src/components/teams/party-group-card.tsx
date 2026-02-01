'use client'

import * as React from 'react'
import { ChevronRight } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn, getWinRateColor } from '@/lib/utils'
import { PartyGroup, PlayerData } from '@/types'
import { PartyGroupModal } from '@/components/modals/party-group-modal'

interface PartyGroupCardProps {
  group: PartyGroup
  playerData: PlayerData
}

export function PartyGroupCard({ group, playerData }: PartyGroupCardProps) {
  const [selectedGroup, setSelectedGroup] = React.useState<PartyGroup | null>(null)

  return (
    <>
      <button
        onClick={() => setSelectedGroup(group)}
        className="group relative rounded-lg border border-primary-500/30 bg-primary-500/5 p-4 transition-all hover:border-primary-500/60 hover:scale-[1.02] text-left w-full"
      >
        {/* Header: Party Members */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex-1">
            <div className="flex items-center gap-1 mb-1">
              <Badge variant="outline" className="text-xs">
                {group.partySize} Players
              </Badge>
            </div>
            <div className="space-y-0.5">
              {group.members.map((battletag, idx) => (
                <p
                  key={idx}
                  className={cn(
                    "text-xs",
                    battletag === playerData.playerName
                      ? "font-semibold text-primary-500"
                      : "font-medium text-muted-foreground"
                  )}
                >
                  {battletag}
                </p>
              ))}
            </div>
          </div>
          <ChevronRight className="h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
        </div>

        {/* Stats Grid */}
        <div className="space-y-2 mt-3 pt-3 border-t border-primary-500/20">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Games</span>
            <span className="font-medium">{group.totalGames}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Win Rate</span>
            <span className={cn("font-medium", getWinRateColor(group.winRate))}>
              {group.winRate.toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Record</span>
            <span className="font-medium">
              {group.totalWins}W - {group.totalLosses}L
            </span>
          </div>
        </div>

        {/* Top Heroes */}
        {group.commonHeroes.length > 0 && (
          <div className="mt-3 pt-3 border-t border-primary-500/20">
            <p className="text-xs text-muted-foreground mb-1">Top Heroes</p>
            <div className="flex flex-wrap gap-1">
              {group.commonHeroes.slice(0, 3).map((heroStat) => (
                <Badge key={heroStat.hero} variant="secondary" className="text-xs">
                  {heroStat.hero}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </button>

      {selectedGroup && (
        <PartyGroupModal
          group={selectedGroup}
          playerData={playerData}
          open={!!selectedGroup}
          onOpenChange={(open) => !open && setSelectedGroup(null)}
        />
      )}
    </>
  )
}
