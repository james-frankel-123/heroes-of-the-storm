'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Users } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { PartyGroupCard } from '@/components/teams/party-group-card'
import { useReplays } from '@/lib/hooks/use-replays'
import { PlayerData, PartyGroup } from '@/types'

interface PartyHistoryProps {
  playerData: PlayerData
}

export function PartyHistory({ playerData }: PartyHistoryProps) {
  const { data: replayData, isLoading } = useReplays(playerData?.playerName)

  if (isLoading) {
    return (
      <Card className="glass border-primary-500/30">
        <CardContent className="p-8 text-center">
          <p className="text-muted-foreground">Loading party data...</p>
        </CardContent>
      </Card>
    )
  }

  if (!replayData || replayData.partyGames === 0) {
    return (
      <Card className="glass border-primary-500/30">
        <CardContent className="p-8 text-center">
          <p className="text-muted-foreground">
            No party data available. Play some games with friends to see your party statistics!
          </p>
        </CardContent>
      </Card>
    )
  }

  const { partyStats } = replayData

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Card className="glass border-primary-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5 text-accent-cyan" />
            Your Party History
          </CardTitle>
          <CardDescription>
            Performance with specific party groups ({replayData.partyGames} party games)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="duos" className="space-y-6">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="duos">
                Duos {partyStats.duos.length > 0 && `(${partyStats.duos.length})`}
              </TabsTrigger>
              <TabsTrigger value="trios">
                Trios {partyStats.trios.length > 0 && `(${partyStats.trios.length})`}
              </TabsTrigger>
              <TabsTrigger value="quadruples">
                Quadruples {partyStats.quadruples.length > 0 && `(${partyStats.quadruples.length})`}
              </TabsTrigger>
              <TabsTrigger value="quintuples">
                Quintuples {partyStats.quintuples.length > 0 && `(${partyStats.quintuples.length})`}
              </TabsTrigger>
            </TabsList>

            <TabsContent value="duos">
              <PartyGroupGrid groups={partyStats.duos} playerData={playerData} emptyMessage="No duos found. Play with a friend to see duo statistics!" />
            </TabsContent>

            <TabsContent value="trios">
              <PartyGroupGrid groups={partyStats.trios} playerData={playerData} emptyMessage="No trios found. Play with two friends to see trio statistics!" />
            </TabsContent>

            <TabsContent value="quadruples">
              <PartyGroupGrid groups={partyStats.quadruples} playerData={playerData} emptyMessage="No quadruples found. Play with three friends to see statistics!" />
            </TabsContent>

            <TabsContent value="quintuples">
              <PartyGroupGrid groups={partyStats.quintuples} playerData={playerData} emptyMessage="No full premades found. Play with a 5-stack to see statistics!" />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </motion.div>
  )
}

interface PartyGroupGridProps {
  groups: PartyGroup[]
  playerData: PlayerData
  emptyMessage: string
}

function PartyGroupGrid({ groups, playerData, emptyMessage }: PartyGroupGridProps) {
  if (groups.length === 0) {
    return (
      <div className="p-8 text-center">
        <p className="text-sm text-muted-foreground">{emptyMessage}</p>
      </div>
    )
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {groups.slice(0, 9).map((group) => (
        <PartyGroupCard
          key={group.membershipKey}
          group={group}
          playerData={playerData}
        />
      ))}
    </div>
  )
}
