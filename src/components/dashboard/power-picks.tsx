'use client'

import * as React from 'react'
import { Zap, ChevronRight } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatPercent } from '@/lib/utils'
import { PowerPick, PlayerData } from '@/types'
import { PowerPickModal } from '@/components/modals/power-pick-modal'

interface PowerPicksProps {
  powerPicks: PowerPick[]
  playerData?: PlayerData
}

interface PowerPickCardProps {
  pick: PowerPick
  index: number
  playerData?: PlayerData
}

function PowerPickCard({ pick, index, playerData }: PowerPickCardProps) {
  const [modalOpen, setModalOpen] = React.useState(false)

  if (!playerData) {
    return null
  }

  return (
    <>
      <button
        onClick={() => setModalOpen(true)}
        className="group relative overflow-hidden rounded-lg border border-gaming-success/30 bg-gaming-success/5 p-4 transition-all hover:scale-[1.02] hover:border-gaming-success/60 w-full text-left cursor-pointer"
      >
        <div className="absolute right-2 top-2 text-4xl font-bold text-gaming-success/10">
          {index + 1}
        </div>
        <div className="relative z-10 space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-lg font-bold">{pick.hero}</p>
            <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          <p className="text-sm text-muted-foreground">{pick.map}</p>
          <div className="flex items-end gap-2">
            <span className="text-2xl font-bold text-gaming-success">
              {formatPercent(pick.winRate, 1)}
            </span>
            <span className="mb-1 text-xs text-muted-foreground">
              {pick.games} games
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-2 opacity-70 group-hover:opacity-100 transition-opacity">
            Click for AI analysis
          </p>
        </div>
      </button>

      <PowerPickModal
        powerPick={pick}
        playerData={playerData}
        open={modalOpen}
        onOpenChange={setModalOpen}
      />
    </>
  )
}

export function PowerPicks({ powerPicks, playerData }: PowerPicksProps) {
  if (powerPicks.length === 0) {
    return null
  }

  return (
    <Card className="glass border-primary-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-gaming-success" />
          Power Picks (65%+ Win Rate)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {powerPicks.map((pick, index) => (
            <PowerPickCard
              key={`${pick.hero}-${pick.map}`}
              pick={pick}
              index={index}
              playerData={playerData}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
