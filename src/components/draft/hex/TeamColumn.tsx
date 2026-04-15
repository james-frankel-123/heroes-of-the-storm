'use client'

import { HexTile } from './HexTile'
import type { SlotView } from './draft-view-model'
import type { TeamAccent } from './constants'

interface TeamColumnProps {
  picks: SlotView[]
  accent: TeamAccent
  label: string
  hexSize?: number
  zigzag?: number
}

/**
 * Vertical stack of 5 pick slots with alternating horizontal offset (zigzag).
 * Blue team shifts even picks left / odd picks right; red mirrors.
 */
export function TeamColumn({
  picks, accent, label, hexSize = 128, zigzag = 24,
}: TeamColumnProps) {
  const tint = accent === 'blue' ? 'text-[#6b8dd4]' : 'text-[#d46b6b]'
  return (
    <div className="flex flex-col items-center gap-2">
      <div className={`text-xs tracking-[0.3em] font-light mb-1 ${tint}`}>{label}</div>
      {picks.map((slot, i) => {
        const even = i % 2 === 0
        const shift = accent === 'blue'
          ? (even ? -zigzag : zigzag)
          : (even ? zigzag : -zigzag)
        return (
          <div key={slot.stepIndex} style={{ transform: `translateX(${shift}px)` }}>
            <HexTile
              hero={slot.hero}
              size={hexSize}
              accent={accent}
              isCurrent={slot.isCurrent}
              isSkipped={slot.isSkipped}
              bannerSide={accent === 'blue' ? 'lower-left' : 'lower-right'}
            />
          </div>
        )
      })}
    </div>
  )
}
