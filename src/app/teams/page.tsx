'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Users, TrendingUp, TrendingDown, Shield, ChevronRight } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { formatPercent, getWinRateColor } from '@/lib/utils'
import { TEAM_COMPOSITIONS, DuoStats } from '@/lib/data/team-compositions'
import { TeamSynergyModal } from '@/components/modals/team-synergy-modal'
import { PartyHistory } from '@/components/teams/party-history'
import { usePlayerData } from '@/lib/hooks/use-data'

// Use DuoStats type from team-compositions module
type TeamComposition = DuoStats

// Use imported data instead of duplicating it
const teamCompositions: TeamComposition[] = TEAM_COMPOSITIONS

export default function TeamsPage() {
  const { data: playerData } = usePlayerData()
  const [selectedSynergy, setSelectedSynergy] = React.useState<DuoStats | null>(null)

  const bestSynergies = teamCompositions
    .filter((comp) => comp.games >= 2)
    .sort((a, b) => b.winRate - a.winRate)
    .slice(0, 6)

  const worstSynergies = teamCompositions
    .filter((comp) => comp.games >= 2)
    .sort((a, b) => a.winRate - b.winRate)
    .slice(0, 6)

  const allCompositions = [...teamCompositions].sort((a, b) => b.winRate - a.winRate)

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3">
          <Users className="h-10 w-10 text-primary-500" />
          <div>
            <h1 className="text-4xl font-bold tracking-tight glow">
              Team Compositions
            </h1>
            <p className="mt-2 text-muted-foreground">
              Analyze hero synergies and duo performance
            </p>
          </div>
        </div>
      </motion.div>

      {/* Best Synergies */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="glass border-gaming-success/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-gaming-success" />
              Best Synergies
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {bestSynergies.map((comp, index) => (
                <button
                  key={comp.heroes}
                  onClick={() => playerData && setSelectedSynergy(comp)}
                  className={`group relative overflow-hidden rounded-lg border border-gaming-success/30 bg-gaming-success/5 p-4 transition-all hover:border-gaming-success/60 text-left w-full ${
                    playerData ? 'cursor-pointer hover:scale-[1.02]' : ''
                  }`}
                >
                  <div className="absolute right-2 top-2 text-3xl font-bold text-gaming-success/10">
                    #{index + 1}
                  </div>
                  <div className="relative z-10 space-y-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-start gap-2 flex-1">
                        <Users className="h-5 w-5 text-gaming-success mt-0.5" />
                        <div className="flex-1">
                          <p className="font-bold text-sm leading-tight">
                            {comp.heroes.split(' + ').join(' + ')}
                          </p>
                        </div>
                      </div>
                      {playerData && (
                        <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                      )}
                    </div>
                    <div className="flex items-end justify-between">
                      <div>
                        <p className="text-2xl font-bold text-gaming-success">
                          {formatPercent(comp.winRate, 1)}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {comp.wins}-{comp.losses} ({comp.games} games)
                        </p>
                      </div>
                      {comp.winRate >= 75 && (
                        <Badge variant="outline" className="border-gaming-success text-gaming-success">
                          Hot Duo
                        </Badge>
                      )}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Worst Synergies */}
      {worstSynergies.some(comp => comp.winRate < 40) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="glass border-gaming-danger/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingDown className="h-5 w-5 text-gaming-danger" />
                Avoid These Combinations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {worstSynergies.map((comp) => (
                  <button
                    key={comp.heroes}
                    onClick={() => playerData && setSelectedSynergy(comp)}
                    className={`group rounded-lg border border-gaming-danger/30 bg-gaming-danger/5 p-4 transition-all text-left w-full ${
                      playerData ? 'cursor-pointer hover:scale-[1.02]' : ''
                    }`}
                  >
                    <div className="space-y-3">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-start gap-2 flex-1">
                          <Users className="h-5 w-5 text-gaming-danger mt-0.5" />
                          <div className="flex-1">
                            <p className="font-bold text-sm leading-tight">
                              {comp.heroes}
                            </p>
                          </div>
                        </div>
                        {playerData && (
                          <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                        )}
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-gaming-danger">
                          {formatPercent(comp.winRate, 1)}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {comp.wins}-{comp.losses} ({comp.games} games)
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* All Compositions Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="glass border-primary-500/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary-500" />
              All Team Compositions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-hidden rounded-lg border border-border/50">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border/50 bg-muted/50">
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Rank
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Heroes
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Games
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Record
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Win Rate
                    </th>
                    <th className="px-4 py-3 w-10"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {allCompositions.map((comp, index) => (
                    <tr
                      key={comp.heroes}
                      onClick={() => playerData && setSelectedSynergy(comp)}
                      className={`group transition-colors ${
                        playerData
                          ? 'cursor-pointer hover:bg-primary-500/5'
                          : ''
                      }`}
                    >
                      <td className="px-4 py-3 text-sm text-muted-foreground">
                        {index + 1}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <Users className="h-4 w-4 text-primary-500" />
                          <span className="font-medium">{comp.heroes}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right text-sm text-muted-foreground">
                        {comp.games}
                      </td>
                      <td className="px-4 py-3 text-right text-sm">
                        {comp.wins}-{comp.losses}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span
                          className={`font-bold ${getWinRateColor(comp.winRate)}`}
                        >
                          {formatPercent(comp.winRate, 1)}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        {playerData && (
                          <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <Card className="glass border-primary-500/30">
          <CardHeader>
            <CardTitle>Team Synergy Insights</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-lg border border-gaming-success/30 bg-gaming-success/5 p-4">
              <p className="font-semibold text-gaming-success">Strong Partnerships</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Anduin + Stitches, Diablo + Deckard, and Anduin + Nazeebo all have 100% win rates. These are your power duos!
              </p>
            </div>
            <div className="rounded-lg border border-primary-500/30 bg-primary-500/5 p-4">
              <p className="font-semibold text-primary-500">Falstad Synergies</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Falstad works well with Li-Ming (75% WR) and shows decent synergy with Nazeebo and Garrosh (60% each).
              </p>
            </div>
            <div className="rounded-lg border border-gaming-danger/30 bg-gaming-danger/5 p-4">
              <p className="font-semibold text-gaming-danger">Avoid These Pairs</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Several combinations have 0% win rates. Consider alternative hero pairings when drafting.
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Party History Section */}
      {playerData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <PartyHistory playerData={playerData} />
        </motion.div>
      )}

      {/* Team Synergy Modal */}
      {selectedSynergy && playerData && (
        <TeamSynergyModal
          synergy={selectedSynergy}
          playerData={playerData}
          open={!!selectedSynergy}
          onOpenChange={(open) => !open && setSelectedSynergy(null)}
        />
      )}
    </div>
  )
}
