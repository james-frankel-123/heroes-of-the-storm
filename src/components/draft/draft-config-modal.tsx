'use client'

import * as React from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Map as MapIcon, Users, ChevronRight, ChevronLeft, Loader2 } from 'lucide-react'
import { PlayerData } from '@/types'
import { DraftTeam } from '@/lib/draft/draft-sequence'

const MAPS = [
  'Infernal Shrines',
  'Braxis Holdout',
  'Garden of Terror',
  'Cursed Hollow',
  'Volskaya Foundry',
  'Alterac Pass',
  'Battlefield of Eternity',
  'Sky Temple',
  'Tomb of the Spider Queen',
  'Dragon Shire',
  'Towers of Doom',
  'Warhead Junction',
  'Hanamura Temple',
  'Blackheart\'s Bay'
].sort()

export interface PartyMember {
  battletag: string
  playerStats: PlayerData | null
  loading: boolean
  slot: number
}

interface DraftConfigModalProps {
  open: boolean
  onComplete: (config: {
    selectedMap: string
    yourTeam: DraftTeam
    partyRoster: PartyMember[]
  }) => void
}

export function DraftConfigModal({ open, onComplete }: DraftConfigModalProps) {
  const [step, setStep] = React.useState(1)

  // Step 1: Map and Team
  const [selectedMap, setSelectedMap] = React.useState('')
  const [yourTeam, setYourTeam] = React.useState<DraftTeam>('blue')

  // Step 2: Party Roster
  const [partyRoster, setPartyRoster] = React.useState<PartyMember[]>([
    { battletag: '', playerStats: null, loading: false, slot: 0 },
    { battletag: '', playerStats: null, loading: false, slot: 1 },
    { battletag: '', playerStats: null, loading: false, slot: 2 },
    { battletag: '', playerStats: null, loading: false, slot: 3 },
    { battletag: '', playerStats: null, loading: false, slot: 4 },
  ])

  const canProceedStep1 = selectedMap !== ''
  const canProceedStep2 = partyRoster.some(m => m.battletag !== '' && m.playerStats !== null)
  const isAnyLoading = partyRoster.some(m => m.loading)

  async function fetchPlayerStats(battletag: string): Promise<PlayerData | null> {
    try {
      const response = await fetch(`/api/data?battletag=${encodeURIComponent(battletag)}`)
      if (!response.ok) {
        return null
      }
      const data = await response.json()

      // Try case-insensitive match
      const exactMatch = data[battletag]
      if (exactMatch) return exactMatch

      const lowerBattletag = battletag.toLowerCase()
      const matchedKey = Object.keys(data).find(key => key.toLowerCase() === lowerBattletag)
      if (matchedKey) return data[matchedKey]

      return null
    } catch (error) {
      console.error('Error fetching player stats:', error)
      return null
    }
  }

  function handleBattletagChange(slot: number, value: string) {
    const updated = [...partyRoster]
    updated[slot] = {
      ...updated[slot],
      battletag: value,
      playerStats: null  // Clear stats when battletag changes
    }
    setPartyRoster(updated)
  }

  async function handleBattletagBlur(slot: number) {
    const member = partyRoster[slot]
    if (!member.battletag || member.battletag.trim() === '') {
      return
    }

    // Set loading state
    const updated = [...partyRoster]
    updated[slot] = { ...updated[slot], loading: true }
    setPartyRoster(updated)

    // Fetch stats
    const stats = await fetchPlayerStats(member.battletag)

    // Update with results
    const final = [...partyRoster]
    final[slot] = {
      ...final[slot],
      playerStats: stats,
      loading: false
    }
    setPartyRoster(final)
  }

  function handleStartDraft() {
    onComplete({
      selectedMap,
      yourTeam,
      partyRoster: partyRoster.filter(m => m.battletag !== '')
    })
  }

  return (
    <Dialog open={open} onOpenChange={() => {}}>
      <DialogContent className="sm:max-w-2xl" onEscapeKeyDown={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Draft Configuration
            <Badge variant="outline">Step {step} of 3</Badge>
          </DialogTitle>
          <DialogDescription>
            Configure your draft settings before starting
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Step 1: Map and Team */}
          {step === 1 && (
            <div className="space-y-6">
              {/* Map Selection */}
              <div>
                <label className="text-sm font-semibold flex items-center gap-2 mb-3">
                  <MapIcon className="h-4 w-4" />
                  Select Battleground
                </label>
                <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto p-1">
                  {MAPS.map((map) => (
                    <button
                      key={map}
                      onClick={() => setSelectedMap(map)}
                      className={`text-left px-3 py-2 rounded-lg border transition-colors text-sm ${
                        selectedMap === map
                          ? 'border-primary-500 bg-primary-500/10 text-primary-500'
                          : 'border-border hover:border-primary-500/50 hover:bg-accent'
                      }`}
                    >
                      {map}
                    </button>
                  ))}
                </div>
              </div>

              {/* Team Selection */}
              <div>
                <label className="text-sm font-semibold flex items-center gap-2 mb-3">
                  <Users className="h-4 w-4" />
                  Your Team Side
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setYourTeam('blue')}
                    className={`flex flex-col items-center justify-center p-4 rounded-lg border-2 transition-all ${
                      yourTeam === 'blue'
                        ? 'border-blue-500 bg-blue-500/10'
                        : 'border-border hover:border-blue-500/50'
                    }`}
                  >
                    <div className="text-3xl mb-2">🔵</div>
                    <div className="font-bold text-blue-500">BLUE TEAM</div>
                    <div className="text-xs text-muted-foreground mt-1">First pick advantage</div>
                  </button>

                  <button
                    onClick={() => setYourTeam('red')}
                    className={`flex flex-col items-center justify-center p-4 rounded-lg border-2 transition-all ${
                      yourTeam === 'red'
                        ? 'border-red-500 bg-red-500/10'
                        : 'border-border hover:border-red-500/50'
                    }`}
                  >
                    <div className="text-3xl mb-2">🔴</div>
                    <div className="font-bold text-red-500">RED TEAM</div>
                    <div className="text-xs text-muted-foreground mt-1">First ban, double picks</div>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Party Roster */}
          {step === 2 && (
            <div className="space-y-4">
              <div className="text-sm text-muted-foreground">
                Enter battletags for your party members. Stats will auto-load when you finish typing.
                Leave slots empty if playing with unknown players.
              </div>

              {partyRoster.map((member, index) => (
                <div key={index} className="space-y-2">
                  <label className="text-sm font-medium">
                    Slot {index + 1} {index === 0 && yourTeam === 'blue' ? '(First pick)' : ''}
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="text"
                      placeholder="PlayerName#1234"
                      value={member.battletag}
                      onChange={(e) => handleBattletagChange(index, e.target.value)}
                      onBlur={() => handleBattletagBlur(index)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleBattletagBlur(index)
                        }
                      }}
                      className="flex-1 px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary-500"
                      disabled={member.loading}
                    />
                    {member.loading && (
                      <Loader2 className="h-4 w-4 animate-spin text-primary-500" />
                    )}
                    {member.playerStats && (
                      <Badge variant="outline" className="text-green-500 border-green-500">
                        ✓ Loaded
                      </Badge>
                    )}
                    {member.battletag && !member.loading && !member.playerStats && (
                      <Badge variant="outline" className="text-red-500 border-red-500">
                        Not found
                      </Badge>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Step 3: Review */}
          {step === 3 && (
            <div className="space-y-4">
              <div className="rounded-lg border border-border p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-semibold">Battleground:</div>
                  <div className="text-sm">{selectedMap}</div>
                </div>
                <div className="flex items-center justify-between">
                  <div className="text-sm font-semibold">Your Team:</div>
                  <Badge variant={yourTeam === 'blue' ? 'default' : 'destructive'}>
                    {yourTeam.toUpperCase()}
                  </Badge>
                </div>
              </div>

              <div className="space-y-3">
                <div className="text-sm font-semibold">Party Roster:</div>
                {partyRoster.filter(m => m.battletag !== '').map((member, index) => (
                  <div key={index} className="rounded-lg border border-border p-3 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="font-medium text-sm">
                        Slot {member.slot + 1}: {member.battletag}
                      </div>
                      {member.playerStats && (
                        <Badge variant="outline" className="text-green-500 border-green-500">
                          Stats loaded
                        </Badge>
                      )}
                    </div>
                    {member.playerStats && (
                      <div className="text-xs text-muted-foreground space-y-1">
                        <div>
                          Overall: {member.playerStats.overallWinRate.toFixed(1)}% WR,{' '}
                          {member.playerStats.totalGames} games
                        </div>
                        <div>
                          Top heroes:{' '}
                          {member.playerStats.heroStats
                            .slice(0, 3)
                            .map((h) => `${h.hero} (${h.winRate}%)`)
                            .join(', ')}
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {partyRoster.filter(m => m.battletag !== '').length === 0 && (
                  <div className="text-sm text-muted-foreground italic">
                    No party members entered. AI recommendations will be generic.
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between pt-4 border-t border-border">
          {step > 1 ? (
            <Button
              variant="outline"
              onClick={() => setStep(step - 1)}
              disabled={isAnyLoading}
            >
              <ChevronLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
          ) : (
            <div></div>
          )}

          {step < 3 ? (
            <Button
              onClick={() => setStep(step + 1)}
              disabled={(step === 1 && !canProceedStep1) || (step === 2 && !canProceedStep2) || isAnyLoading}
            >
              Next
              <ChevronRight className="h-4 w-4 ml-2" />
            </Button>
          ) : (
            <Button onClick={handleStartDraft} disabled={isAnyLoading}>
              Start Draft
            </Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
