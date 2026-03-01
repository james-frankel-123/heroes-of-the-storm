'use client'

import { cn } from '@/lib/utils'
import type { PlayerSlot } from '@/lib/draft/types'

interface PlayerSlotsProps {
  slots: PlayerSlot[]
  registeredBattletags: string[]
  onSetPlayer: (slotIndex: number, battletag: string | null) => void
}

export function PlayerSlots({
  slots,
  registeredBattletags,
  onSetPlayer,
}: PlayerSlotsProps) {
  // Track which battletags are already assigned
  const assignedBattletags = new Set(
    slots.map((s) => s.battletag).filter(Boolean)
  )

  return (
    <div className="space-y-1.5">
      {slots.map((slot, idx) => {
        // Available battletags = registered minus already assigned (except this slot)
        const available = registeredBattletags.filter(
          (bt) => !assignedBattletags.has(bt) || bt === slot.battletag
        )

        return (
          <div key={idx} className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-16 shrink-0">
              Player {idx + 1}
            </span>
            <select
              value={slot.battletag ?? ''}
              onChange={(e) =>
                onSetPlayer(idx, e.target.value || null)
              }
              className={cn(
                'flex-1 h-8 px-2 rounded-md border text-sm bg-background transition-colors',
                slot.battletag
                  ? 'border-primary/40 text-foreground'
                  : 'border-border text-muted-foreground'
              )}
            >
              <option value="">Not assigned</option>
              {available.map((bt) => (
                <option key={bt} value={bt}>
                  {bt}
                </option>
              ))}
            </select>
          </div>
        )
      })}
      {registeredBattletags.length === 0 && (
        <p className="text-[10px] text-muted-foreground">
          No registered battletags. Generic stats will be used.
        </p>
      )}
    </div>
  )
}
