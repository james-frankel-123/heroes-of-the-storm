'use client'

import * as React from 'react'
import { motion } from 'framer-motion'
import { Search, SortAsc } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { getAvailableHeroes, DraftTeam, DraftTurn } from '@/lib/draft/draft-sequence'

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

interface HeroSelectorProps {
  availableHeroes: string[]
  onHeroSelect: (hero: string) => void
  disabled?: boolean
  currentTurn?: DraftTurn
  yourTeam?: DraftTeam
}

export function HeroSelector({
  availableHeroes,
  onHeroSelect,
  disabled = false,
  currentTurn,
  yourTeam
}: HeroSelectorProps) {
  const [searchQuery, setSearchQuery] = React.useState('')
  const [sortBy, setSortBy] = React.useState<'name' | 'role'>('name')

  // Filter heroes by search query
  const filteredHeroes = React.useMemo(() => {
    if (!searchQuery) return availableHeroes

    const query = searchQuery.toLowerCase()
    return availableHeroes.filter(hero =>
      hero.toLowerCase().includes(query)
    )
  }, [availableHeroes, searchQuery])

  // Sort heroes
  const sortedHeroes = React.useMemo(() => {
    return [...filteredHeroes].sort((a, b) => {
      if (sortBy === 'name') {
        return a.localeCompare(b)
      }
      // Could add role sorting if we had role data
      return a.localeCompare(b)
    })
  }, [filteredHeroes, sortBy])

  const isYourTurn = currentTurn && yourTeam ? currentTurn.team === yourTeam : true
  const turnTeamLabel = currentTurn ? currentTurn.team.toUpperCase() : ''
  const actionLabel = currentTurn ? (currentTurn.action === 'ban' ? 'BAN' : 'PICK') : ''

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold">
          {currentTurn ? (
            <>
              Select hero for <span className={currentTurn.team === 'blue' ? 'text-blue-400' : 'text-red-400'}>
                {turnTeamLabel} TEAM {actionLabel}
              </span> ({filteredHeroes.length} available)
            </>
          ) : (
            <>Select Hero ({filteredHeroes.length} available)</>
          )}
        </div>
        {currentTurn && (
          <Badge
            variant="outline"
            className={isYourTurn ? 'text-green-400 border-green-400/30' : 'text-yellow-400 border-yellow-400/30'}
          >
            {isYourTurn ? 'Your Team' : 'Enemy Team'}
          </Badge>
        )}
      </div>

      {/* Search and filters */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search heroes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
            disabled={disabled}
          />
        </div>
        <Button
          variant="outline"
          size="icon"
          onClick={() => setSortBy(sortBy === 'name' ? 'role' : 'name')}
          disabled={disabled}
          title="Sort heroes"
        >
          <SortAsc className="h-4 w-4" />
        </Button>
      </div>

      {/* Hero Grid */}
      <div className="max-h-96 overflow-y-auto">
        {sortedHeroes.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            {searchQuery ? 'No heroes found' : 'All heroes have been picked or banned'}
          </div>
        ) : (
          <div className="grid grid-cols-4 sm:grid-cols-5 md:grid-cols-6 gap-2">
            {sortedHeroes.map((hero, index) => (
              <HeroButton
                key={hero}
                hero={hero}
                onSelect={onHeroSelect}
                disabled={disabled}
                index={index}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

interface HeroButtonProps {
  hero: string
  onSelect: (hero: string) => void
  disabled: boolean
  index: number
}

function HeroButton({ hero, onSelect, disabled, index }: HeroButtonProps) {
  return (
    <motion.button
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.01, duration: 0.2 }}
      whileHover={disabled ? {} : { scale: 1.05 }}
      whileTap={disabled ? {} : { scale: 0.95 }}
      onClick={() => !disabled && onSelect(hero)}
      disabled={disabled}
      className={`
        relative aspect-square rounded-lg border-2 p-2 flex items-center justify-center
        text-xs font-medium text-center transition-all
        ${
          disabled
            ? 'border-border bg-background/50 text-muted-foreground cursor-not-allowed opacity-50'
            : 'border-border hover:border-primary-500 hover:bg-primary-500/10 hover:text-primary-500 cursor-pointer'
        }
      `}
    >
      <span className="line-clamp-2 leading-tight">{hero}</span>
    </motion.button>
  )
}
