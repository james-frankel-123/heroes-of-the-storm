'use client'

import { HexTile } from './HexTile'
import type { SlotView } from './draft-view-model'

interface BanBarProps {
  bansA: SlotView[]
  bansB: SlotView[]
  ourTeam: 'A' | 'B'
  hexSize?: number
}

/**
 * Horizontal ban strip. Our-team bans sit on the left with blue accent; enemy
 * bans on the right with red accent. Each side has 3 ban hexes.
 */
export function BanBar({ bansA, bansB, ourTeam, hexSize = 56 }: BanBarProps) {
  const ourBans = ourTeam === 'A' ? bansA : bansB
  const enemyBans = ourTeam === 'A' ? bansB : bansA

  return (
    <div className="flex items-center justify-center gap-8 pb-6 border-b border-[#d6dbe0]/20">
      <BanGroup label="YOUR BANS" bans={ourBans} accent="blue" hexSize={hexSize} />
      <div className="text-[#d6dbe0]/50 text-xs tracking-widest">VS</div>
      <BanGroup label="ENEMY BANS" bans={enemyBans} accent="red" hexSize={hexSize} />
    </div>
  )
}

function BanGroup({
  label, bans, accent, hexSize,
}: { label: string; bans: SlotView[]; accent: 'blue' | 'red'; hexSize: number }) {
  const tint = accent === 'blue' ? 'text-[#6b8dd4]' : 'text-[#d46b6b]'
  return (
    <div className="flex flex-col items-center gap-2">
      <div className={`text-[10px] tracking-[0.3em] font-light ${tint}`}>{label}</div>
      <div className="flex gap-3">
        {bans.map((slot) => (
          <HexTile
            key={slot.stepIndex}
            hero={slot.hero}
            size={hexSize}
            accent={accent}
            banned={!!slot.hero}
            isCurrent={slot.isCurrent}
          />
        ))}
      </div>
    </div>
  )
}
