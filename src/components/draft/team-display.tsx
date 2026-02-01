'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { User, Users, CheckCircle } from 'lucide-react'
import { DraftTeam, DraftTurn } from '@/lib/draft/draft-sequence'
import { PartyMember } from './draft-config-modal'
import { Badge } from '@/components/ui/badge'

interface TeamDisplayProps {
  team: DraftTeam
  picks: (string | null)[]
  yourTeam: DraftTeam
  currentTurn: DraftTurn
  partyRoster?: PartyMember[]
}

export function TeamDisplay({
  team,
  picks,
  yourTeam,
  currentTurn,
  partyRoster
}: TeamDisplayProps) {
  const isYourTeam = team === yourTeam
  const teamColor = team === 'blue' ? 'blue' : 'red'
  const bgColor = team === 'blue' ? 'bg-blue-500/10' : 'bg-red-500/10'
  const borderColor = team === 'blue' ? 'border-blue-500/30' : 'border-red-500/30'
  const textColor = team === 'blue' ? 'text-blue-500' : 'text-red-500'

  // Determine which pick slot is currently active
  const isPickingNow = currentTurn.action === 'pick' && currentTurn.team === team
  const activePickSlot = isPickingNow ? currentTurn.pickSlot : null

  return (
    <div className="space-y-3">
      {/* Team Header */}
      <div className="flex items-center gap-2">
        <div className={`h-3 w-3 rounded-full ${team === 'blue' ? 'bg-blue-500' : 'bg-red-500'}`}></div>
        <span className={`font-bold ${textColor}`}>
          {team.toUpperCase()} TEAM
        </span>
        {isYourTeam && (
          <Badge variant="outline" className="ml-auto">
            Your Team
          </Badge>
        )}
      </div>

      {/* Pick Slots */}
      <div className="space-y-2">
        {picks.map((pick, index) => (
          <PickSlot
            key={`${team}-${index}`}
            pickNumber={index + 1}
            hero={pick}
            teamColor={teamColor}
            bgColor={bgColor}
            borderColor={borderColor}
            textColor={textColor}
            isActive={activePickSlot === index}
            battletag={isYourTeam && partyRoster ? partyRoster[index]?.battletag : undefined}
          />
        ))}
      </div>
    </div>
  )
}

interface PickSlotProps {
  pickNumber: number
  hero: string | null
  teamColor: 'blue' | 'red'
  bgColor: string
  borderColor: string
  textColor: string
  isActive: boolean
  battletag?: string
}

function PickSlot({
  pickNumber,
  hero,
  teamColor,
  bgColor,
  borderColor,
  textColor,
  isActive,
  battletag
}: PickSlotProps) {
  const isEmpty = !hero

  return (
    <motion.div
      initial={{ opacity: 0, x: teamColor === 'blue' ? -20 : 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: pickNumber * 0.05 }}
      className={`relative rounded-lg border-2 ${
        isEmpty
          ? isActive
            ? `${borderColor} ${bgColor} ring-2 ring-primary-500`
            : 'border-dashed border-border'
          : `${borderColor} ${bgColor}`
      } p-3 transition-all`}
    >
      <div className="flex items-center gap-3">
        {/* Pick Number Badge */}
        <div
          className={`flex-shrink-0 flex items-center justify-center h-8 w-8 rounded-full border-2 ${borderColor} ${bgColor}`}
        >
          <span className={`text-sm font-bold ${textColor}`}>{pickNumber}</span>
        </div>

        {/* Pick Content */}
        <div className="flex-1 min-w-0">
          {isEmpty ? (
            <div className="space-y-1">
              {isActive ? (
                <div className="flex items-center gap-2">
                  <motion.div
                    animate={{
                      scale: [1, 1.2, 1],
                      opacity: [0.5, 1, 0.5],
                    }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      ease: 'easeInOut',
                    }}
                  >
                    <User className="h-4 w-4 text-primary-500" />
                  </motion.div>
                  <span className="text-sm font-medium text-primary-500">Selecting...</span>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <User className="h-4 w-4 text-muted-foreground opacity-30" />
                  <span className="text-sm text-muted-foreground">Awaiting pick</span>
                </div>
              )}
              {battletag && (
                <div className="text-xs text-muted-foreground truncate">{battletag}</div>
              )}
            </div>
          ) : (
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                className="space-y-1"
              >
                <div className="flex items-center gap-2">
                  <CheckCircle className={`h-4 w-4 ${textColor}`} />
                  <span className="font-medium text-sm truncate">{hero}</span>
                </div>
                {battletag && (
                  <div className="text-xs text-muted-foreground truncate">{battletag}</div>
                )}
              </motion.div>
            </AnimatePresence>
          )}
        </div>
      </div>

      {/* Active indicator glow */}
      {isActive && (
        <motion.div
          className="absolute inset-0 rounded-lg bg-primary-500/10 -z-10 pointer-events-none"
          animate={{
            opacity: [0, 0.5, 0],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      )}
    </motion.div>
  )
}
