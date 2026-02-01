'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Sparkles } from 'lucide-react'
import { PasswordGate } from '@/components/auth/password-gate'
import { DraftConfigModal, PartyMember } from '@/components/draft/draft-config-modal'
import { DraftBoard } from '@/components/draft/draft-board'
import {
  DRAFT_SEQUENCE,
  DraftTurn,
  DraftTeam,
  getCurrentTurn,
  getNextTurn
} from '@/lib/draft/draft-sequence'

interface DraftAction {
  turn: DraftTurn
  hero: string
  timestamp: number
  battletag?: string
}

export default function DraftPage() {
  // Draft configuration
  const [isConfigured, setIsConfigured] = React.useState(false)
  const [selectedMap, setSelectedMap] = React.useState<string>('')
  const [yourTeam, setYourTeam] = React.useState<DraftTeam>('blue')
  const [partyRoster, setPartyRoster] = React.useState<PartyMember[]>([])

  // Draft state
  const [turnIndex, setTurnIndex] = React.useState(0)
  const [bluePicks, setBluePicks] = React.useState<(string | null)[]>([null, null, null, null, null])
  const [redPicks, setRedPicks] = React.useState<(string | null)[]>([null, null, null, null, null])
  const [blueBans, setBlueBans] = React.useState<string[]>([])
  const [redBans, setRedBans] = React.useState<string[]>([])
  const [draftHistory, setDraftHistory] = React.useState<DraftAction[]>([])

  // Get current turn
  const currentTurn = getCurrentTurn(turnIndex)

  // Handle draft configuration
  const handleConfigComplete = (
    map: string,
    team: DraftTeam,
    roster: PartyMember[]
  ) => {
    setSelectedMap(map)
    setYourTeam(team)
    setPartyRoster(roster)
    setIsConfigured(true)
  }

  // Handle hero selection
  const handleHeroSelect = (hero: string) => {
    if (!currentTurn) return

    const turn = currentTurn

    // Add to appropriate bans or picks
    if (turn.action === 'ban') {
      if (turn.team === 'blue') {
        setBlueBans([...blueBans, hero])
      } else {
        setRedBans([...redBans, hero])
      }
    } else {
      // Pick action
      if (turn.team === 'blue') {
        const newPicks = [...bluePicks]
        newPicks[turn.pickSlot!] = hero
        setBluePicks(newPicks)
      } else {
        const newPicks = [...redPicks]
        newPicks[turn.pickSlot!] = hero
        setRedPicks(newPicks)
      }
    }

    // Add to history
    const battletag = turn.team === yourTeam && turn.action === 'pick' && turn.pickSlot !== undefined
      ? partyRoster[turn.pickSlot]?.battletag
      : undefined

    setDraftHistory([
      ...draftHistory,
      {
        turn,
        hero,
        timestamp: Date.now(),
        battletag
      }
    ])

    // Advance turn
    const nextTurn = getNextTurn(turnIndex)
    if (nextTurn) {
      setTurnIndex(turnIndex + 1)
    }
  }

  // Handle undo
  const handleUndo = () => {
    if (draftHistory.length === 0) return

    const lastAction = draftHistory[draftHistory.length - 1]

    // Remove from bans or picks
    if (lastAction.turn.action === 'ban') {
      if (lastAction.turn.team === 'blue') {
        setBlueBans(blueBans.slice(0, -1))
      } else {
        setRedBans(redBans.slice(0, -1))
      }
    } else {
      // Pick action
      if (lastAction.turn.team === 'blue') {
        const newPicks = [...bluePicks]
        newPicks[lastAction.turn.pickSlot!] = null
        setBluePicks(newPicks)
      } else {
        const newPicks = [...redPicks]
        newPicks[lastAction.turn.pickSlot!] = null
        setRedPicks(newPicks)
      }
    }

    // Remove from history
    setDraftHistory(draftHistory.slice(0, -1))

    // Go back one turn
    setTurnIndex(Math.max(0, turnIndex - 1))
  }

  // Handle reset
  const handleReset = () => {
    setTurnIndex(0)
    setBluePicks([null, null, null, null, null])
    setRedPicks([null, null, null, null, null])
    setBlueBans([])
    setRedBans([])
    setDraftHistory([])
  }

  // Can undo if there's history
  const canUndo = draftHistory.length > 0

  if (!isConfigured) {
    return (
      <PasswordGate requiredPassword="ronpaul2012" storageKey="protected_pages_auth">
        <DraftConfigModal
          onComplete={handleConfigComplete}
        />
      </PasswordGate>
    )
  }

  return (
    <PasswordGate requiredPassword="ronpaul2012" storageKey="protected_pages_auth">
      <div className="space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-3">
            <Sparkles className="h-10 w-10 text-accent-cyan" />
            <div>
              <h1 className="text-4xl font-bold tracking-tight glow">Draft Assistant</h1>
              <p className="mt-2 text-muted-foreground">
                Storm League draft sequence with real-time AI coaching
              </p>
            </div>
          </div>
        </motion.div>

        {/* Draft Board */}
        {currentTurn && (
          <DraftBoard
            selectedMap={selectedMap}
            currentTurn={currentTurn}
            yourTeam={yourTeam}
            bluePicks={bluePicks}
            redPicks={redPicks}
            blueBans={blueBans}
            redBans={redBans}
            partyRoster={partyRoster}
            onHeroSelect={handleHeroSelect}
            onUndo={handleUndo}
            onReset={handleReset}
            canUndo={canUndo}
          />
        )}
      </div>
    </PasswordGate>
  )
}
