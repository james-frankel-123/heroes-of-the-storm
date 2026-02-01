'use client'

import { AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { usePlayer } from '@/contexts/player-context'

interface PlayerDataErrorProps {
  error?: Error
  reset?: () => void
}

export function PlayerDataError({ error, reset }: PlayerDataErrorProps) {
  const { battletag } = usePlayer()

  const handleTryDifferentBattletag = () => {
    // Clear current battletag to show entry modal
    localStorage.removeItem('hots_player_battletag')
    window.location.reload()
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] p-4">
      <div className="max-w-md text-center space-y-4">
        <div className="flex justify-center">
          <div className="rounded-full bg-red-500/10 p-4">
            <AlertCircle className="h-12 w-12 text-red-500" />
          </div>
        </div>

        <h2 className="text-2xl font-bold">Unable to Load Player Data</h2>

        <p className="text-muted-foreground">
          We couldn't find data for <span className="font-mono font-semibold">{battletag}</span>
        </p>

        <div className="bg-muted/50 rounded-lg p-4 text-left">
          <p className="text-sm font-medium mb-2">This could mean:</p>
          <ul className="text-sm text-muted-foreground space-y-1 list-disc pl-5">
            <li>The battletag doesn't exist</li>
            <li>The player hasn't played any Storm League games</li>
            <li>Heroes Profile doesn't have data for this player</li>
            <li>There was a temporary network issue</li>
          </ul>
        </div>

        {error && (
          <details className="bg-muted/50 rounded-lg p-4 text-left">
            <summary className="text-sm font-medium cursor-pointer">Error Details</summary>
            <p className="text-xs text-muted-foreground mt-2 font-mono">
              {error.message}
            </p>
          </details>
        )}

        <div className="flex gap-2 justify-center pt-4">
          <Button onClick={handleTryDifferentBattletag}>
            Try Different Battletag
          </Button>
          {reset && (
            <Button variant="outline" onClick={reset}>
              Retry
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
