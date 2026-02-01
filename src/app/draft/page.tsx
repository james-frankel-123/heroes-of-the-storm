'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, Radio, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { PasswordGate } from '@/components/auth/password-gate'
import { DraftConfigModal, PartyMember } from '@/components/draft/draft-config-modal'
import { DraftBoard } from '@/components/draft/draft-board'
import { AICoachPanel } from '@/components/draft/ai-coach-panel'
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

  // Mobile AI panel state
  const [showMobileAI, setShowMobileAI] = React.useState(false)

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
      <div className="flex flex-col h-screen overflow-hidden">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 border-b border-border"
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

        {/* Two-Panel Layout */}
        {currentTurn && (
          <>
            <div className="flex flex-1 overflow-hidden">
              {/* Left Panel: Draft Board (60%) */}
              <div className="flex-1 lg:w-3/5 overflow-y-auto">
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
              </div>

              {/* Right Panel: AI Coach (40%) - Fixed/Sticky (Desktop) */}
              <div className="hidden lg:block lg:w-2/5 overflow-hidden">
                <AICoachPanel
                  draftState={{
                    selectedMap,
                    yourTeam,
                    currentTurn,
                    bluePicks,
                    redPicks,
                    blueBans,
                    redBans
                  }}
                  partyRoster={partyRoster}
                  draftHistory={draftHistory}
                />
              </div>
            </div>

            {/* Mobile AI Coach Toggle Button */}
            <Button
              onClick={() => setShowMobileAI(true)}
              className="fixed bottom-6 right-6 lg:hidden rounded-full h-14 w-14 shadow-lg"
              variant="gaming"
            >
              <Radio className="h-6 w-6" />
            </Button>

            {/* Mobile AI Coach Panel (Drawer) */}
            <AnimatePresence>
              {showMobileAI && (
                <>
                  {/* Backdrop */}
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={() => setShowMobileAI(false)}
                    className="fixed inset-0 bg-black/50 z-40 lg:hidden"
                  />

                  {/* Drawer */}
                  <motion.div
                    initial={{ x: '100%' }}
                    animate={{ x: 0 }}
                    exit={{ x: '100%' }}
                    transition={{ type: 'spring', damping: 30, stiffness: 300 }}
                    className="fixed right-0 top-0 bottom-0 w-full sm:w-96 z-50 lg:hidden"
                  >
                    <div className="relative h-full flex flex-col bg-card">
                      {/* Close Button */}
                      <Button
                        onClick={() => setShowMobileAI(false)}
                        variant="ghost"
                        size="icon"
                        className="absolute top-4 right-4 z-10"
                      >
                        <X className="h-5 w-5" />
                      </Button>

                      {/* AI Coach Panel */}
                      <AICoachPanel
                        draftState={{
                          selectedMap,
                          yourTeam,
                          currentTurn,
                          bluePicks,
                          redPicks,
                          blueBans,
                          redBans
                        }}
                        partyRoster={partyRoster}
                        draftHistory={draftHistory}
                      />
                    </div>
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </>
        )}
      </div>
    </PasswordGate>
  )
}
