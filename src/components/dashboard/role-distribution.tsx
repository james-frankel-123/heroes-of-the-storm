'use client'

import * as React from 'react'
import { Users, ChevronRight } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { getWinRateColor, formatPercent } from '@/lib/utils'
import { RoleDetailsModal } from '@/components/modals/role-details-modal'
import { PlayerData } from '@/types'

interface RoleDistributionProps {
  roleData: Record<string, { wins: number; games: number; winRate: number }>
  playerData?: PlayerData
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

export function RoleDistribution({ roleData, playerData }: RoleDistributionProps) {
  const [selectedRole, setSelectedRole] = React.useState<{ role: string; stats: any } | null>(null)

  const roleEntries = Object.entries(roleData)
    .sort((a, b) => b[1].games - a[1].games) // Sort by games played
    .filter(([_, stats]) => stats.games >= 10) // Only show roles with 10+ games

  const totalGames = roleEntries.reduce((sum, [_, stats]) => sum + stats.games, 0)

  return (
    <>
      <Card className="glass border-primary-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5 text-primary-500" />
            Role Distribution
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {roleEntries.map(([role, stats]) => {
            const percentage = totalGames > 0 ? (stats.games / totalGames) * 100 : 0
            return (
              <button
                key={role}
                onClick={() => playerData && setSelectedRole({ role, stats })}
                className={`group w-full space-y-2 text-left rounded-lg p-2 -m-2 transition-all ${
                  playerData ? 'cursor-pointer hover:bg-primary-500/5' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <Badge variant={roleVariantMap[role] || 'outline'} className="text-xs">
                    {role}
                  </Badge>
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-muted-foreground">
                      {stats.games} games
                    </span>
                    <span
                      className={`text-sm font-bold ${getWinRateColor(stats.winRate)}`}
                    >
                      {formatPercent(stats.winRate, 1)}
                    </span>
                    {playerData && (
                      <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                    )}
                  </div>
                </div>
                <div className="relative h-2 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary-500/50 transition-all"
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </button>
            )
          })}
        </CardContent>
      </Card>

      {selectedRole && playerData && (
        <RoleDetailsModal
          role={selectedRole.role}
          stats={selectedRole.stats}
          playerData={playerData}
          open={!!selectedRole}
          onOpenChange={(open) => !open && setSelectedRole(null)}
        />
      )}
    </>
  )
}
