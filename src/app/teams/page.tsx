'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Users } from 'lucide-react'
import { PartyHistory } from '@/components/teams/party-history'
import { usePlayerData } from '@/lib/hooks/use-data'

export default function TeamsPage() {
  const { data: playerData } = usePlayerData()

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

      {/* Party History Section */}
      {playerData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <PartyHistory playerData={playerData} />
        </motion.div>
      )}
    </div>
  )
}
