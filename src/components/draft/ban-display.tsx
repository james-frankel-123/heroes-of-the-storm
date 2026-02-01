'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Ban, Shield } from 'lucide-react'
import { DraftTeam } from '@/lib/draft/draft-sequence'
import { Badge } from '@/components/ui/badge'

interface BanDisplayProps {
  blueBans: string[]
  redBans: string[]
  yourTeam: DraftTeam
}

export function BanDisplay({ blueBans, redBans, yourTeam }: BanDisplayProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-sm font-semibold text-muted-foreground">
        <Ban className="h-4 w-4" />
        Banned Heroes
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Blue Bans */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <div className="h-3 w-3 rounded-full bg-blue-500"></div>
            <span className="text-sm font-medium text-blue-500">
              BLUE TEAM {yourTeam === 'blue' && '(You)'}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[0, 1, 2].map((index) => (
              <BanSlot
                key={`blue-${index}`}
                hero={blueBans[index]}
                phaseNumber={index + 1}
                teamColor="blue"
              />
            ))}
          </div>
        </div>

        {/* Red Bans */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <div className="h-3 w-3 rounded-full bg-red-500"></div>
            <span className="text-sm font-medium text-red-500">
              RED TEAM {yourTeam === 'red' && '(You)'}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[0, 1, 2].map((index) => (
              <BanSlot
                key={`red-${index}`}
                hero={redBans[index]}
                phaseNumber={index + 1}
                teamColor="red"
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

interface BanSlotProps {
  hero: string | undefined
  phaseNumber: number
  teamColor: 'blue' | 'red'
}

function BanSlot({ hero, phaseNumber, teamColor }: BanSlotProps) {
  const isEmpty = !hero

  const bgColor = teamColor === 'blue' ? 'bg-blue-500/10' : 'bg-red-500/10'
  const borderColor = teamColor === 'blue' ? 'border-blue-500/30' : 'border-red-500/30'
  const textColor = teamColor === 'blue' ? 'text-blue-500' : 'text-red-500'

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className={`relative aspect-square rounded-lg border-2 ${
        isEmpty ? 'border-dashed border-border' : `${borderColor} ${bgColor}`
      } flex flex-col items-center justify-center p-2 overflow-hidden`}
    >
      {isEmpty ? (
        <div className="flex flex-col items-center gap-1">
          <Shield className="h-6 w-6 text-muted-foreground opacity-30" />
          <span className="text-xs text-muted-foreground">Ban {phaseNumber}</span>
        </div>
      ) : (
        <AnimatePresence>
          <motion.div
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className="flex flex-col items-center gap-1 text-center"
          >
            <div className="relative">
              <Ban className={`h-6 w-6 ${textColor}`} />
              <div className="absolute -top-1 -right-1">
                <Badge
                  variant="outline"
                  className="text-[10px] px-1 py-0 h-4 bg-background"
                >
                  {phaseNumber}
                </Badge>
              </div>
            </div>
            <div className="text-xs font-medium line-clamp-2 leading-tight">{hero}</div>
          </motion.div>
        </AnimatePresence>
      )}

      {/* Diagonal strike-through effect when banned */}
      {!isEmpty && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className={`w-full h-0.5 ${textColor} opacity-50 rotate-45`}></div>
        </div>
      )}
    </motion.div>
  )
}
