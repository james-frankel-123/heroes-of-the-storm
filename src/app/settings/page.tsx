'use client'

import * as React from 'react'
import { Settings, User } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { usePlayer } from '@/contexts/player-context'

export default function SettingsPage() {
  const { battletag, setBattletag } = usePlayer()
  const [input, setInput] = React.useState(battletag || '')
  const [error, setError] = React.useState('')
  const [saved, setSaved] = React.useState(false)

  // Update input when battletag changes from context
  React.useEffect(() => {
    if (battletag) {
      setInput(battletag)
    }
  }, [battletag])

  const handleSave = () => {
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
    setBattletag(trimmed)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)

    // Reload the page to fetch new player data
    setTimeout(() => {
      window.location.reload()
    }, 500)
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold tracking-tight glow">Settings</h1>
        <p className="mt-2 text-muted-foreground">
          Customize your experience
        </p>
      </div>

      <Card className="glass border-primary-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="h-5 w-5" />
            Player Profile
          </CardTitle>
          <CardDescription>
            Change your Heroes of the Storm battletag
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Battletag</label>
            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="PlayerName#1234"
                className="flex-1"
              />
              <Button onClick={handleSave} disabled={saved}>
                {saved ? 'Saved!' : 'Save'}
              </Button>
            </div>
            {error && <p className="text-sm text-red-500">{error}</p>}
            {battletag && !error && (
              <p className="text-xs text-muted-foreground">
                Current: {battletag}
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="glass border-primary-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Application Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="py-12 text-center">
          <p className="text-muted-foreground">
            Additional settings coming soon with theme preferences, data sources, and more.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
