'use client'

import React from 'react'
import { BattletagEntryModal } from '@/components/modals/battletag-entry-modal'
import { usePlayer } from '@/contexts/player-context'

export function BattletagGate({ children }: { children: React.ReactNode }) {
  const { battletag, setBattletag, isLoading } = usePlayer()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <>
      <BattletagEntryModal
        open={!battletag}
        onSubmit={setBattletag}
      />
      {children}
    </>
  )
}
