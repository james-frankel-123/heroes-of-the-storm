'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { DraftTurn, DraftTeam, formatTurnIndicator } from '@/lib/draft/draft-sequence'
import { ChevronRight } from 'lucide-react'

interface TurnIndicatorProps {
  currentTurn: DraftTurn
  yourTeam: DraftTeam
}

export function TurnIndicator({ currentTurn, yourTeam }: TurnIndicatorProps) {
  const { isYourTurn, teamLabel, actionLabel, phaseLabel, description } = formatTurnIndicator(
    currentTurn,
    yourTeam
  )

  const teamColor = currentTurn.team === 'blue' ? 'text-blue-500' : 'text-red-500'
  const bgColor = currentTurn.team === 'blue' ? 'bg-blue-500/10' : 'bg-red-500/10'
  const borderColor = currentTurn.team === 'blue' ? 'border-blue-500/30' : 'border-red-500/30'

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="w-full"
    >
      <div
        className={`relative rounded-lg border-2 ${borderColor} ${bgColor} p-6 ${
          isYourTurn ? 'ring-2 ring-primary-500 ring-offset-2 ring-offset-background' : ''
        }`}
      >
        {/* Phase indicator at top */}
        <div className="absolute top-2 right-4 text-xs text-muted-foreground">
          {phaseLabel}
        </div>

        {/* Main turn info */}
        <div className="flex items-center justify-center gap-4">
          {isYourTurn && (
            <motion.div
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.5, 1, 0.5],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            >
              <ChevronRight className="h-8 w-8 text-primary-500" />
            </motion.div>
          )}

          <div className="text-center">
            <div className={`text-sm font-medium ${teamColor} mb-1`}>{teamLabel}</div>
            <div className="text-3xl font-bold">{actionLabel}</div>
            <div className="text-sm text-muted-foreground mt-2">{description}</div>
          </div>

          {isYourTurn && (
            <motion.div
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.5, 1, 0.5],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            >
              <ChevronRight className="h-8 w-8 text-primary-500 rotate-180" />
            </motion.div>
          )}
        </div>

        {/* Pulsing background when your turn */}
        {isYourTurn && (
          <motion.div
            className="absolute inset-0 rounded-lg bg-primary-500/5 -z-10"
            animate={{
              opacity: [0, 0.3, 0],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
        )}
      </div>

      {/* Turn progress bar */}
      <div className="mt-2 h-1 bg-border rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-primary-500"
          initial={{ width: 0 }}
          animate={{ width: `${((currentTurn.turnIndex + 1) / 16) * 100}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>
      <div className="text-xs text-muted-foreground text-center mt-1">
        Turn {currentTurn.turnIndex + 1} of 16
      </div>
    </motion.div>
  )
}
