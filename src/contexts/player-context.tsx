'use client'

import React, { createContext, useContext, useState, useEffect } from 'react'

interface PlayerContextType {
  battletag: string | null
  setBattletag: (battletag: string) => void
  isLoading: boolean
}

const PlayerContext = createContext<PlayerContextType | undefined>(undefined)

const STORAGE_KEY = 'hots_player_battletag'

export function PlayerProvider({ children }: { children: React.ReactNode }) {
  const [battletag, setBattletagState] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Load battletag from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      setBattletagState(stored)
    }
    setIsLoading(false)
  }, [])

  // Save to localStorage when battletag changes
  const setBattletag = (newBattletag: string) => {
    setBattletagState(newBattletag)
    localStorage.setItem(STORAGE_KEY, newBattletag)
  }

  return (
    <PlayerContext.Provider value={{ battletag, setBattletag, isLoading }}>
      {children}
    </PlayerContext.Provider>
  )
}

export function usePlayer() {
  const context = useContext(PlayerContext)
  if (context === undefined) {
    throw new Error('usePlayer must be used within a PlayerProvider')
  }
  return context
}
