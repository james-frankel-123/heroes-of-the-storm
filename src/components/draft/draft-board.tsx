'use client'

import * as React from 'react'
import { DraftTeam, DraftTurn, getAvailableHeroes, isYourTurn } from '@/lib/draft/draft-sequence'
import { PartyMember } from './draft-config-modal'
import { DraftPhaseHeader } from './draft-phase-header'
import { TurnIndicator } from './turn-indicator'
import { BanDisplay } from './ban-display'
import { TeamDisplay } from './team-display'
import { HeroSelector } from './hero-selector'

const ALL_HEROES = [
  'Abathur', 'Alarak', 'Alexstrasza', 'Ana', 'Anduin', "Anub'arak",
  'Artanis', 'Arthas', 'Auriel', 'Azmodan', 'Blaze', 'Brightwing',
  'Cassia', 'Chen', 'Cho', 'Chromie', 'D.Va', 'Deckard',
  'Dehaka', 'Diablo', 'E.T.C.', 'Falstad', 'Fenix', 'Gall',
  'Garrosh', 'Gazlowe', 'Genji', 'Greymane', 'Gul\'dan', 'Hanzo',
  'Hogger', 'Illidan', 'Imperius', 'Jaina', 'Johanna', 'Junkrat',
  'Kael\'thas', 'Kel\'Thuzad', 'Kerrigan', 'Kharazim', 'Leoric', 'Li Li',
  'Li-Ming', 'Lt. Morales', 'Lúcio', 'Lunara', 'Maiev', 'Malfurion',
  'Mal\'Ganis', 'Medivh', 'Mei', 'Mephisto', 'Muradin', 'Murky',
  'Nazeebo', 'Nova', 'Orphea', 'Probius', 'Qhira', 'Ragnaros',
  'Raynor', 'Rehgar', 'Rexxar', 'Samuro', 'Sgt. Hammer', 'Sonya',
  'Stitches', 'Stukov', 'Sylvanas', 'Tassadar', 'The Butcher', 'The Lost Vikings',
  'Thrall', 'Tracer', 'Tychus', 'Tyrael', 'Tyrande', 'Uther',
  'Valeera', 'Valla', 'Varian', 'Whitemane', 'Xul', 'Yrel',
  'Zagara', 'Zarya', 'Zeratul', 'Zul\'jin'
].sort()

interface DraftBoardProps {
  selectedMap: string
  currentTurn: DraftTurn
  yourTeam: DraftTeam
  bluePicks: (string | null)[]
  redPicks: (string | null)[]
  blueBans: string[]
  redBans: string[]
  partyRoster: PartyMember[]
  onHeroSelect: (hero: string) => void
  onUndo?: () => void
  onReset?: () => void
  canUndo: boolean
}

export function DraftBoard({
  selectedMap,
  currentTurn,
  yourTeam,
  bluePicks,
  redPicks,
  blueBans,
  redBans,
  partyRoster,
  onHeroSelect,
  onUndo,
  onReset,
  canUndo
}: DraftBoardProps) {
  // Calculate available heroes
  const availableHeroes = React.useMemo(() => {
    return getAvailableHeroes(
      ALL_HEROES,
      { blue: blueBans, red: redBans },
      { blue: bluePicks, red: redPicks }
    )
  }, [blueBans, redBans, bluePicks, redPicks])

  // Check if it's your turn
  const yourTurn = isYourTurn(currentTurn, yourTeam)

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <DraftPhaseHeader
        selectedMap={selectedMap}
        currentTurn={currentTurn}
        onUndo={onUndo}
        onReset={onReset}
        canUndo={canUndo}
      />

      {/* Turn Indicator */}
      <TurnIndicator currentTurn={currentTurn} yourTeam={yourTeam} />

      {/* Bans Display */}
      <BanDisplay blueBans={blueBans} redBans={redBans} yourTeam={yourTeam} />

      {/* Teams Display - Side by Side */}
      <div className="grid grid-cols-2 gap-6">
        <TeamDisplay
          team="blue"
          picks={bluePicks}
          yourTeam={yourTeam}
          currentTurn={currentTurn}
          partyRoster={yourTeam === 'blue' ? partyRoster : undefined}
        />
        <TeamDisplay
          team="red"
          picks={redPicks}
          yourTeam={yourTeam}
          currentTurn={currentTurn}
          partyRoster={yourTeam === 'red' ? partyRoster : undefined}
        />
      </div>

      {/* Hero Selector */}
      <HeroSelector
        availableHeroes={availableHeroes}
        onHeroSelect={onHeroSelect}
        disabled={!yourTurn}
      />
    </div>
  )
}
