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
import { Trophy, Target, TrendingUp, Users } from 'lucide-react'
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
  const [hero1, hero2] = parseDuoHeroes(synergy.heroes)
  const role1 = classifyHeroRole(hero1)
  const role2 = classifyHeroRole(hero2)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[75vw] max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-2xl">
            <Users className="h-6 w-6 text-primary-500" />
            <span>Duo Synergy</span>
          </DialogTitle>
        </DialogHeader>

        <div className="max-h-[calc(90vh-8rem)] overflow-hidden">
          {/* Stats and Details */}
          <div className="space-y-6 overflow-y-auto pr-2">
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
        </div>
      </DialogContent>
    </Dialog>
  )
}
