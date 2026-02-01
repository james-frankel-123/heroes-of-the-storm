'use client'

import * as React from 'react'
import { Map as MapIcon, RotateCcw, Undo } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { DraftTurn, getDraftPhaseName } from '@/lib/draft/draft-sequence'

interface DraftPhaseHeaderProps {
  selectedMap: string
  currentTurn: DraftTurn
  onUndo?: () => void
  onReset?: () => void
  canUndo: boolean
}

export function DraftPhaseHeader({
  selectedMap,
  currentTurn,
  onUndo,
  onReset,
  canUndo
}: DraftPhaseHeaderProps) {
  const phaseName = getDraftPhaseName(currentTurn.phase)

  return (
    <div className="flex items-center justify-between py-4 border-b border-border">
      {/* Left: Map info */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <MapIcon className="h-5 w-5 text-primary-500" />
          <div>
            <div className="text-xs text-muted-foreground">Battleground</div>
            <div className="font-semibold">{selectedMap}</div>
          </div>
        </div>

        <div className="h-8 w-px bg-border"></div>

        <div>
          <div className="text-xs text-muted-foreground">Draft Phase</div>
          <Badge variant="outline" className="font-semibold">
            {phaseName} (Phase {currentTurn.phase})
          </Badge>
        </div>
      </div>

      {/* Right: Controls */}
      <div className="flex items-center gap-2">
        {onUndo && (
          <Button
            variant="outline"
            size="sm"
            onClick={onUndo}
            disabled={!canUndo}
            title="Undo last action"
          >
            <Undo className="h-4 w-4 mr-2" />
            Undo
          </Button>
        )}
        {onReset && (
          <Button
            variant="outline"
            size="sm"
            onClick={onReset}
            title="Reset draft"
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </Button>
        )}
      </div>
    </div>
  )
}
