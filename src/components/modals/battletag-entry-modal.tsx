'use client'

import React, { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

interface BattletagEntryModalProps {
  open: boolean
  onSubmit: (battletag: string) => void
}

export function BattletagEntryModal({ open, onSubmit }: BattletagEntryModalProps) {
  const [input, setInput] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = () => {
    const trimmed = input.trim()

    // Validate format: Name#1234
    if (!trimmed.includes('#')) {
      setError('Battletag must include # and discriminator (e.g., PlayerName#1234)')
      return
    }

    const [name, discriminator] = trimmed.split('#')
    if (!name || !discriminator || !/^\d+$/.test(discriminator)) {
      setError('Invalid battletag format. Expected: PlayerName#1234')
      return
    }

    setError('')
    onSubmit(trimmed)
  }

  return (
    <Dialog open={open} onOpenChange={() => {}}>
      <DialogContent className="sm:max-w-md" onEscapeKeyDown={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>Welcome to HotS Analytics</DialogTitle>
          <DialogDescription>
            Enter your Heroes of the Storm battletag to view your stats
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium">Battletag</label>
            <Input
              placeholder="PlayerName#1234"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
              className="mt-1"
            />
            {error && <p className="text-sm text-red-500 mt-1">{error}</p>}
            <p className="text-xs text-muted-foreground mt-1">
              Include your full battletag with # and number
            </p>
          </div>

          <Button onClick={handleSubmit} className="w-full">
            Continue
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
