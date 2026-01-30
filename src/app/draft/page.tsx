'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Sparkles,
  Users,
  Target,
  Ban,
  Map as MapIcon,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  X,
  Eye,
  Loader2,
  UserPlus
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { formatPercent, getWinRateColor } from '@/lib/utils'
import { fetchPlayerHeroStats, PlayerStats } from '@/lib/api/heroes-profile'
import { TEAM_COMPOSITIONS, getDuoWinRate } from '@/lib/data/team-compositions'

// Mock hero list - will be populated from real data
const ALL_HEROES = [
  'Nazeebo', 'Azmodan', 'Valla', 'Tracer', 'Li-Ming', 'Mei', 'Leoric',
  'Anub\'arak', 'Deckard', 'Ana', 'Garrosh', 'Muradin', 'Diablo',
  'Falstad', 'Anduin', 'Stitches', 'Stukov', 'Dehaka', 'Raynor',
  'Cassia', 'Lúcio', 'Tychus', 'Nova', 'Kel\'Thuzad', 'Brightwing',
  'Johanna', 'Malfurion', 'Rehgar', 'Uther', 'Tyrande', 'Zeratul',
  'Illidan', 'Kerrigan', 'Alarak', 'Arthas', 'E.T.C.', 'Sonya'
].sort()

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
  'Dragon Shire'
]

// Default stats for current user
const DEFAULT_HERO_STATS: Record<string, { winRate: number; games: number; role: string }> = {
  'Nazeebo': { winRate: 56.4, games: 431, role: 'Ranged Assassin' },
  'Azmodan': { winRate: 56.2, games: 265, role: 'Ranged Assassin' },
  'Valla': { winRate: 58.2, games: 55, role: 'Ranged Assassin' },
  'Tracer': { winRate: 54.3, games: 223, role: 'Ranged Assassin' },
  'Li-Ming': { winRate: 53.2, games: 94, role: 'Ranged Assassin' },
  'Mei': { winRate: 62.1, games: 29, role: 'Unknown' },
  'Leoric': { winRate: 58.8, games: 34, role: 'Bruiser' },
  'Anub\'arak': { winRate: 57.9, games: 19, role: 'Tank' },
  'Deckard': { winRate: 57.1, games: 35, role: 'Healer' },
  'Ana': { winRate: 52.5, games: 160, role: 'Healer' },
  'Garrosh': { winRate: 52.5, games: 99, role: 'Tank' },
  'Muradin': { winRate: 51.0, games: 42, role: 'Tank' },
}

const SYNERGIES: Record<string, string[]> = {
  'Anduin': ['Stitches', 'Nazeebo'],
  'Diablo': ['Deckard'],
  'Falstad': ['Li-Ming', 'Nazeebo', 'Garrosh'],
  'Nazeebo': ['Anduin', 'Falstad', 'Dehaka'],
}

interface TeamSlot {
  hero: string | null
  showing: string | null
  battletag: string
  playerStats: PlayerStats | null
  loading: boolean
}

export default function DraftPage() {
  const [selectedMap, setSelectedMap] = React.useState<string>('')
  const [yourTeam, setYourTeam] = React.useState<TeamSlot[]>([
    { hero: null, showing: null, battletag: 'AzmoDonTrump#1139', playerStats: null, loading: false },
    { hero: null, showing: null, battletag: 'Django#1458', playerStats: null, loading: false },
    { hero: null, showing: null, battletag: 'SirWatsonII#1400', playerStats: null, loading: false },
    { hero: null, showing: null, battletag: '', playerStats: null, loading: false },
    { hero: null, showing: null, battletag: '', playerStats: null, loading: false },
  ])
  const [enemyTeam, setEnemyTeam] = React.useState<string[]>([])
  const [bannedHeroes, setBannedHeroes] = React.useState<string[]>([])
  const [searchQuery, setSearchQuery] = React.useState('')
  const [activeSlot, setActiveSlot] = React.useState<number>(0)
  const [activeTab, setActiveTab] = React.useState<'your' | 'enemy' | 'ban'>('your')

  // Auto-fetch stats for pre-filled battletags on mount
  const [hasAutoFetched, setHasAutoFetched] = React.useState(false)

  React.useEffect(() => {
    if (!hasAutoFetched) {
      const fetchInitialStats = async () => {
        const battletags = [
          { idx: 0, tag: 'AzmoDonTrump#1139' },
          { idx: 1, tag: 'Django#1458' },
          { idx: 2, tag: 'SirWatsonII#1400' },
        ]

        console.log('Auto-fetching stats for pre-filled battletags...')

        for (const { idx, tag } of battletags) {
          // Inline fetch logic to avoid dependency issues
          console.log('Fetching stats for slot', idx, ':', tag)

          setYourTeam(prev => {
            const newTeam = [...prev]
            newTeam[idx].loading = true
            return newTeam
          })

          const stats = await fetchPlayerHeroStats(tag)

          setYourTeam(prev => {
            const updatedTeam = [...prev]
            updatedTeam[idx].playerStats = stats
            updatedTeam[idx].loading = false
            return updatedTeam
          })

          // Small delay between requests to avoid rate limiting
          await new Promise(resolve => setTimeout(resolve, 500))
        }

        console.log('Auto-fetch complete!')
        setHasAutoFetched(true)
      }

      fetchInitialStats()
    }
  }, [hasAutoFetched])

  const allPickedHeroes = [
    ...yourTeam.map(slot => slot.hero).filter(Boolean) as string[],
    ...yourTeam.map(slot => slot.showing).filter(Boolean) as string[],
    ...enemyTeam
  ]

  const availableHeroesUnsorted = ALL_HEROES.filter(
    hero =>
      !allPickedHeroes.includes(hero) &&
      !bannedHeroes.includes(hero) &&
      hero.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleFetchPlayerStats = async (slotIndex: number, battletag: string) => {
    if (!battletag || !battletag.includes('#')) {
      console.log('Invalid battletag format:', battletag)
      return
    }

    console.log('Starting fetch for slot', slotIndex, 'battletag:', battletag)

    // Set loading state
    setYourTeam(prev => {
      const newTeam = [...prev]
      newTeam[slotIndex].loading = true
      return newTeam
    })

    const stats = await fetchPlayerHeroStats(battletag)
    console.log('Stats received for', battletag, ':', stats)

    // Update with stats and clear loading
    setYourTeam(prev => {
      const updatedTeam = [...prev]
      updatedTeam[slotIndex].playerStats = stats
      updatedTeam[slotIndex].loading = false
      return updatedTeam
    })
  }

  const handleHeroClick = (hero: string) => {
    if (activeTab === 'your') {
      const newTeam = [...yourTeam]
      const slot = newTeam[activeSlot]

      if (slot.hero) {
        // Already locked, ignore
        return
      }

      // If showing something else, replace it
      slot.showing = hero
      setYourTeam(newTeam)
    } else if (activeTab === 'enemy' && enemyTeam.length < 5) {
      setEnemyTeam([...enemyTeam, hero])
    } else if (activeTab === 'ban' && bannedHeroes.length < 6) {
      setBannedHeroes([...bannedHeroes, hero])
    }
  }

  const lockInHero = (slotIndex: number) => {
    const newTeam = [...yourTeam]
    const slot = newTeam[slotIndex]
    if (slot.showing && !slot.hero) {
      slot.hero = slot.showing
      slot.showing = null
    }
    setYourTeam(newTeam)
  }

  const clearSlot = (slotIndex: number) => {
    const newTeam = [...yourTeam]
    newTeam[slotIndex].hero = null
    newTeam[slotIndex].showing = null
    setYourTeam(newTeam)
  }

  const removeFromEnemyTeam = (hero: string) => {
    setEnemyTeam(enemyTeam.filter(h => h !== hero))
  }

  const removeBan = (hero: string) => {
    setBannedHeroes(bannedHeroes.filter(h => h !== hero))
  }

  const getHeroWinRate = (hero: string, slotIndex: number, includeTeamSynergies: boolean = true): number => {
    const slot = yourTeam[slotIndex]

    // Get base win rate - ONLY use player-specific stats (no fallback to DEFAULT_HERO_STATS)
    let baseWinRate = 50
    const playerGames = slot.playerStats?.heroStats[hero]?.games || 0

    if (slot.playerStats?.heroStats[hero]) {
      baseWinRate = slot.playerStats.heroStats[hero].winRate
    }
    // If no player stats, always use neutral 50%

    if (!includeTeamSynergies) {
      return baseWinRate
    }

    // Calculate actual duo win rates with locked teammates
    const lockedHeroes = yourTeam
      .map((s, idx) => ({ hero: s.hero, idx }))
      .filter(item => item.hero && item.idx !== slotIndex) as Array<{ hero: string; idx: number }>

    if (lockedHeroes.length === 0) {
      return baseWinRate
    }

    // If we have duo data with ANY locked teammate, use that instead of solo stats
    // Duo data is more specific to the actual team composition
    let duoAdjustedWR = baseWinRate
    let foundDuoData = false

    lockedHeroes.forEach(({ hero: teammateHero }) => {
      const duoWR = getDuoWinRate(hero, teammateHero, TEAM_COMPOSITIONS)
      if (duoWR !== null) {
        const duoData = TEAM_COMPOSITIONS.find(d => {
          const heroes = d.heroes.split(' + ').map(h => h.trim())
          return (heroes[0] === hero && heroes[1] === teammateHero) ||
                 (heroes[0] === teammateHero && heroes[1] === hero)
        })
        if (duoData && duoData.games >= 2) {
          // Use duo win rate directly - it's the actual performance with this teammate
          // Apply a small adjustment towards the mean (regression to mean)
          // More games = trust the duo rate more, fewer games = pull towards solo rate
          const confidence = Math.min(duoData.games / 10, 1) // 10+ games = full confidence
          duoAdjustedWR = (duoWR * confidence) + (baseWinRate * (1 - confidence))
          foundDuoData = true
          return // Use first duo match found
        }
      }
    })

    // If no duo data found, just return base win rate
    return foundDuoData ? duoAdjustedWR : baseWinRate
  }

  const calculateProjectedWinRate = () => {
    const lockedHeroes = yourTeam
      .map((slot, idx) => ({ hero: slot.hero, idx }))
      .filter(item => item.hero !== null) as Array<{ hero: string; idx: number }>

    if (lockedHeroes.length === 0) return 50

    // Calculate average win rate - getHeroWinRate already includes duo adjustments
    const heroWinRates = lockedHeroes.map(({ hero, idx }) =>
      getHeroWinRate(hero, idx, true) // Already includes team synergies via duo data
    )

    const avgWinRate = heroWinRates.reduce((a, b) => a + b, 0) / heroWinRates.length

    // Don't add synergy bonuses on top - the duo data already accounts for this
    // Just return the average of individual hero win rates
    return Math.round(avgWinRate * 10) / 10
  }

  const projectedWinRate = calculateProjectedWinRate()

  const getSynergiesForTeam = () => {
    const lockedHeroes = yourTeam
      .map(slot => slot.hero)
      .filter(Boolean) as string[]

    const synergies: Array<{ hero1: string; hero2: string }> = []
    lockedHeroes.forEach((hero1, i) => {
      lockedHeroes.forEach((hero2, j) => {
        if (i < j && SYNERGIES[hero1]?.includes(hero2)) {
          synergies.push({ hero1, hero2 })
        }
      })
    })
    return synergies
  }

  const teamSynergies = getSynergiesForTeam()

  interface TeamComposition {
    heroes: Array<{ slot: number; hero: string; winRate: number; games: number }>
    winRate: number
    explanation: string
  }

  const getTeamCompositionRecommendations = (): TeamComposition[] => {
    // Get currently locked/showing heroes
    const currentPicks = yourTeam.map((slot, idx) => ({
      slot: idx,
      hero: slot.hero || slot.showing || null
    }))

    const emptySlots = currentPicks
      .map((p, idx) => ({ ...p, slot: idx }))
      .filter(p => !p.hero)

    if (emptySlots.length === 0) return []

    // Generate candidate compositions
    const compositions: TeamComposition[] = []

    // For each empty slot, get top 5 heroes
    const candidatesBySlot = emptySlots.map(emptySlot => {
      const slot = yourTeam[emptySlot.slot]
      const candidates = availableHeroesUnsorted
        .filter(hero => {
          // Hero must not be picked by anyone
          return !currentPicks.some(p => p.hero === hero)
        })
        .map(hero => ({
          hero,
          slotIndex: emptySlot.slot,
          baseWR: slot.playerStats?.heroStats[hero]?.winRate || 50,
          games: slot.playerStats?.heroStats[hero]?.games || 0
        }))
        .sort((a, b) => {
          // Prioritize heroes with more games and higher win rate
          // Multiply by confidence factor (more games = higher weight)
          const aScore = a.baseWR * (1 + Math.log(a.games + 1) / 8)
          const bScore = b.baseWR * (1 + Math.log(b.games + 1) / 8)
          return bScore - aScore
        })
        .slice(0, 5)

      return candidates
    })

    // Generate compositions by trying different combinations
    const maxCompositions = 20
    let compositionCount = 0

    // Helper to calculate team win rate and return hero details
    const calculateTeamWinRate = (picks: Array<{ slot: number; hero: string | null }>): {
      winRate: number;
      heroDetails: Array<{ slot: number; hero: string; winRate: number; games: number }>
    } => {
      const heroes = picks.filter(p => p.hero).map(p => ({ slot: p.slot, hero: p.hero! }))

      if (heroes.length === 0) return { winRate: 50, heroDetails: [] }

      // Calculate average hero win rate with duo adjustments
      let totalWR = 0
      let count = 0
      const heroDetails: Array<{ slot: number; hero: string; winRate: number; games: number }> = []

      heroes.forEach(({ slot, hero }) => {
        const playerStats = yourTeam[slot].playerStats?.heroStats[hero]
        const baseWR = playerStats?.winRate || 50
        const games = playerStats?.games || 0

        // Check duo win rates with other heroes
        let duoAdjustedWR = baseWR
        let foundDuoData = false

        heroes.forEach(other => {
          if (other.slot !== slot && !foundDuoData) {
            const duoWR = getDuoWinRate(hero, other.hero, TEAM_COMPOSITIONS)
            if (duoWR !== null) {
              const duoData = TEAM_COMPOSITIONS.find(d => {
                const heroes = d.heroes.split(' + ').map(h => h.trim())
                return (heroes[0] === hero && heroes[1] === other.hero) ||
                       (heroes[0] === other.hero && heroes[1] === hero)
              })
              if (duoData && duoData.games >= 2) {
                // Use duo win rate with confidence adjustment
                const confidence = Math.min(duoData.games / 10, 1)
                duoAdjustedWR = (duoWR * confidence) + (baseWR * (1 - confidence))
                foundDuoData = true
              }
            }
          }
        })

        // Simply use the win rate - no multiplication by confidence factors
        totalWR += duoAdjustedWR
        count++

        heroDetails.push({ slot, hero, winRate: duoAdjustedWR, games })
      })

      // Simple average - no additional weighting or caps needed
      const avgWinRate = count > 0 ? totalWR / count : 50
      return {
        winRate: Math.round(avgWinRate * 10) / 10,
        heroDetails
      }
    }

    // Try combinations (limited to avoid performance issues)
    if (emptySlots.length === 1) {
      // Only 1 slot empty - show top 3 heroes for that slot
      candidatesBySlot[0].slice(0, 3).forEach(candidate => {
        const testPicks = currentPicks.map(p => ({ ...p }))
        testPicks[candidate.slotIndex] = { slot: candidate.slotIndex, hero: candidate.hero }

        const result = calculateTeamWinRate(testPicks)
        const playerName = yourTeam[candidate.slotIndex].battletag.split('#')[0]

        compositions.push({
          heroes: result.heroDetails,
          winRate: result.winRate,
          explanation: `${playerName} → ${candidate.hero} (${candidate.games}g, ${candidate.baseWR.toFixed(1)}%)`
        })
      })
    } else if (emptySlots.length === 2) {
      // 2 slots empty - try combinations
      const slot1Candidates = candidatesBySlot[0].slice(0, 3)
      const slot2Candidates = candidatesBySlot[1].slice(0, 3)

      slot1Candidates.forEach(c1 => {
        slot2Candidates.forEach(c2 => {
          if (c1.hero !== c2.hero) {
            const testPicks = currentPicks.map(p => ({ ...p }))
            testPicks[c1.slotIndex] = { slot: c1.slotIndex, hero: c1.hero }
            testPicks[c2.slotIndex] = { slot: c2.slotIndex, hero: c2.hero }

            const result = calculateTeamWinRate(testPicks)
            const player1 = yourTeam[c1.slotIndex].battletag.split('#')[0]
            const player2 = yourTeam[c2.slotIndex].battletag.split('#')[0]

            compositions.push({
              heroes: result.heroDetails,
              winRate: result.winRate,
              explanation: `${player1} → ${c1.hero}, ${player2} → ${c2.hero}`
            })
          }
        })
      })
    } else {
      // 3+ slots empty - show top combinations
      const slot1Candidates = candidatesBySlot[0]?.slice(0, 2) || []
      const slot2Candidates = candidatesBySlot[1]?.slice(0, 2) || []
      const slot3Candidates = candidatesBySlot[2]?.slice(0, 2) || []

      slot1Candidates.forEach(c1 => {
        slot2Candidates.forEach(c2 => {
          slot3Candidates.forEach(c3 => {
            const heroes = [c1.hero, c2.hero, c3.hero]
            if (new Set(heroes).size === heroes.length) { // All unique
              const testPicks = currentPicks.map(p => ({ ...p }))
              testPicks[c1.slotIndex] = { slot: c1.slotIndex, hero: c1.hero }
              testPicks[c2.slotIndex] = { slot: c2.slotIndex, hero: c2.hero }
              testPicks[c3.slotIndex] = { slot: c3.slotIndex, hero: c3.hero }

              const result = calculateTeamWinRate(testPicks)
              const player1 = yourTeam[c1.slotIndex].battletag.split('#')[0] || `P${c1.slotIndex + 1}`
              const player2 = yourTeam[c2.slotIndex].battletag.split('#')[0] || `P${c2.slotIndex + 1}`
              const player3 = yourTeam[c3.slotIndex].battletag.split('#')[0] || `P${c3.slotIndex + 1}`

              compositions.push({
                heroes: result.heroDetails,
                winRate: result.winRate,
                explanation: `${player1} → ${c1.hero}, ${player2} → ${c2.hero}, ${player3} → ${c3.hero}`
              })
            }
          })
        })
      })
    }

    // Sort by win rate and return top 3
    return compositions
      .sort((a, b) => b.winRate - a.winRate)
      .slice(0, 3)
  }

  const teamCompositions = getTeamCompositionRecommendations()

  // Sort available heroes by confidence-weighted score (for ordering only)
  const availableHeroes = [...availableHeroesUnsorted].sort((a, b) => {
    // Sort by projected win rate for active slot when in 'your' tab
    if (activeTab === 'your') {
      const currentSlot = yourTeam[activeSlot]

      // Get win rates and game counts
      const aWinRate = getHeroWinRate(a, activeSlot, true)
      const bWinRate = getHeroWinRate(b, activeSlot, true)
      const aGames = currentSlot.playerStats?.heroStats[a]?.games || 0
      const bGames = currentSlot.playerStats?.heroStats[b]?.games || 0

      // Calculate confidence-weighted scores FOR SORTING ONLY
      // This doesn't change the displayed win rate, just affects order
      // Formula: Multiply win rate by confidence factor (more games = higher weight)
      const aScore = aWinRate * (1 + Math.log(aGames + 1) / 8)
      const bScore = bWinRate * (1 + Math.log(bGames + 1) / 8)

      // Primary sort: by confidence-adjusted score (descending)
      if (Math.abs(bScore - aScore) > 0.1) {
        return bScore - aScore
      }

      // Secondary sort: heroes with player stats first
      const aHasStats = currentSlot.playerStats?.heroStats[a] ? 1 : 0
      const bHasStats = currentSlot.playerStats?.heroStats[b] ? 1 : 0
      if (bHasStats !== aHasStats) {
        return bHasStats - aHasStats
      }

      // Tertiary sort: by games played (descending)
      return bGames - aGames
    }

    // Default alphabetical sort for enemy/ban tabs
    return a.localeCompare(b)
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-3">
          <Sparkles className="h-10 w-10 text-accent-cyan" />
          <div>
            <h1 className="text-4xl font-bold tracking-tight glow">Draft Assistant</h1>
            <p className="mt-2 text-muted-foreground">
              Real-time draft simulator with teammate stats integration
            </p>
          </div>
        </div>
      </motion.div>

      {/* Map Selection */}
      <Card className="glass border-primary-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MapIcon className="h-5 w-5" />
            Select Battleground
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {MAPS.map(map => (
              <Button
                key={map}
                variant={selectedMap === map ? 'gaming' : 'outline'}
                size="sm"
                onClick={() => setSelectedMap(selectedMap === map ? '' : map)}
              >
                {map}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Bans - Full Width Above Teams */}
      <Card className={`glass transition-all ${activeTab === 'ban' ? 'border-gaming-warning' : 'border-primary-500/30'}`}>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Ban className="h-5 w-5 text-gaming-warning" />
              Bans
            </div>
            <Badge variant="outline">{bannedHeroes.length}/6</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {bannedHeroes.length === 0 ? (
            <div
              className="rounded-lg border border-dashed border-border p-4 text-center cursor-pointer hover:border-gaming-warning/50"
              onClick={() => setActiveTab('ban')}
            >
              <p className="text-xs text-muted-foreground">
                Add bans
              </p>
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {bannedHeroes.map(hero => (
                <div
                  key={hero}
                  className="flex items-center gap-2 rounded-lg border border-gaming-warning/30 bg-gaming-warning/5 px-3 py-2"
                >
                  <p className="font-semibold text-sm">{hero}</p>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-5 w-5"
                    onClick={() => removeBan(hero)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Draft Area */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Your Team */}
        <div>
          <Card className="glass border-gaming-success/30">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Users className="h-5 w-5 text-gaming-success" />
                  Your Team
                </div>
                <Badge variant="outline">
                  {yourTeam.filter(s => s.hero).length}/5 Locked
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {yourTeam.map((slot, idx) => (
                <div
                  key={idx}
                  className={`rounded-lg border-2 p-4 transition-all cursor-pointer ${
                    activeSlot === idx && activeTab === 'your'
                      ? 'border-gaming-success bg-gaming-success/10'
                      : slot.hero
                      ? 'border-gaming-success/50 bg-gaming-success/5'
                      : slot.showing
                      ? 'border-gaming-warning/50 bg-gaming-warning/5'
                      : 'border-border bg-card/50'
                  }`}
                  onClick={() => {
                    setActiveSlot(idx)
                    setActiveTab('your')
                  }}
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary-500/20 text-sm font-bold">
                      {idx + 1}
                    </div>

                    <div className="flex-1">
                      {/* BattleTag Input */}
                      <div className="flex items-center gap-2 mb-2">
                        <input
                          type="text"
                          placeholder="BattleTag#1234"
                          value={slot.battletag}
                          onChange={(e) => {
                            const newTeam = [...yourTeam]
                            newTeam[idx].battletag = e.target.value
                            // Clear stats when battletag changes
                            newTeam[idx].playerStats = null
                            setYourTeam(newTeam)
                          }}
                          onBlur={() => {
                            if (slot.battletag && slot.battletag.includes('#') && !slot.playerStats && !slot.loading) {
                              handleFetchPlayerStats(idx, slot.battletag)
                            }
                          }}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && slot.battletag && slot.battletag.includes('#')) {
                              e.preventDefault()
                              handleFetchPlayerStats(idx, slot.battletag)
                            }
                          }}
                          className="h-7 flex-1 rounded border border-input bg-background px-2 text-xs focus:border-primary-500 focus:outline-none"
                          onClick={(e) => e.stopPropagation()}
                        />
                        {slot.loading && <Loader2 className="h-4 w-4 animate-spin text-primary-500" />}
                        {!slot.loading && slot.battletag && slot.battletag.includes('#') && !slot.playerStats && (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleFetchPlayerStats(idx, slot.battletag)
                            }}
                            title="Fetch player stats"
                          >
                            <UserPlus className="h-3 w-3" />
                          </Button>
                        )}
                        {slot.playerStats && !slot.loading && !slot.playerStats.error && (
                          <div className="flex items-center gap-1" title={`Loaded ${Object.keys(slot.playerStats.heroStats).length} heroes`}>
                            <CheckCircle className="h-4 w-4 text-gaming-success" />
                            <span className="text-[10px] font-semibold text-gaming-success">
                              {Object.keys(slot.playerStats.heroStats).length}
                            </span>
                          </div>
                        )}
                        {slot.playerStats?.error && (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleFetchPlayerStats(idx, slot.battletag)
                            }}
                            title={`Error: ${slot.playerStats.error}. Click to retry.`}
                          >
                            <AlertCircle className="h-4 w-4 text-gaming-danger" />
                          </Button>
                        )}
                      </div>

                      {/* Hero Display */}
                      {slot.hero ? (
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-bold text-gaming-success">{slot.hero}</p>
                            <p className="text-xs text-muted-foreground">
                              {(() => {
                                const baseWR = getHeroWinRate(slot.hero, idx, false)
                                const adjustedWR = getHeroWinRate(slot.hero, idx, true)
                                const diff = adjustedWR - baseWR
                                const hasDuoEffect = Math.abs(diff) > 0.5
                                return (
                                  <>
                                    {adjustedWR.toFixed(1)}% WR
                                    {hasDuoEffect && (
                                      <span className={diff > 0 ? "text-gaming-success" : "text-gaming-danger"}>
                                        {' '}({diff > 0 ? '+' : ''}{diff.toFixed(1)}% duo)
                                      </span>
                                    )}
                                    {slot.playerStats && ' • Player Stats'}
                                  </>
                                )
                              })()}
                            </p>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={(e) => {
                              e.stopPropagation()
                              clearSlot(idx)
                            }}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ) : slot.showing ? (
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Eye className="h-4 w-4 text-gaming-warning" />
                            <div>
                              <p className="font-semibold text-gaming-warning">{slot.showing}</p>
                              <p className="text-xs text-muted-foreground">
                                {(() => {
                                  const baseWR = getHeroWinRate(slot.showing, idx, false)
                                  const adjustedWR = getHeroWinRate(slot.showing, idx, true)
                                  const diff = adjustedWR - baseWR
                                  const hasDuoBonus = Math.abs(diff) > 0.5
                                  return (
                                    <>
                                      Pre-pick • {adjustedWR.toFixed(1)}% WR
                                      {hasDuoBonus && (
                                        <span className={diff > 0 ? "text-gaming-success" : "text-gaming-danger"}>
                                          {' '}({diff > 0 ? '+' : ''}{diff.toFixed(1)}% duo)
                                        </span>
                                      )}
                                    </>
                                  )
                                })()}
                              </p>
                            </div>
                          </div>
                          <div className="flex gap-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 text-xs"
                              onClick={(e) => {
                                e.stopPropagation()
                                lockInHero(idx)
                              }}
                            >
                              Lock In
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6"
                              onClick={(e) => {
                                e.stopPropagation()
                                clearSlot(idx)
                              }}
                            >
                              <X className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground">
                          {activeSlot === idx && activeTab === 'your'
                            ? 'Select a hero below...'
                            : 'Click to select hero'}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Empty Middle Column */}
        <div className="hidden lg:block"></div>

        {/* Enemy Team */}
        <div>
          <Card className={`glass transition-all ${activeTab === 'enemy' ? 'border-gaming-danger' : 'border-primary-500/30'}`}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-gaming-danger" />
                  Enemy Team
                </div>
                <Badge variant="outline">{enemyTeam.length}/5</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {enemyTeam.length === 0 ? (
                <div
                  className="rounded-lg border border-dashed border-border p-4 text-center cursor-pointer hover:border-gaming-danger/50"
                  onClick={() => setActiveTab('enemy')}
                >
                  <p className="text-xs text-muted-foreground">
                    Add enemy picks
                  </p>
                </div>
              ) : (
                enemyTeam.map(hero => (
                  <div
                    key={hero}
                    className="flex items-center justify-between rounded-lg border border-gaming-danger/30 bg-gaming-danger/5 p-3"
                  >
                    <p className="font-semibold">{hero}</p>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6"
                      onClick={() => removeFromEnemyTeam(hero)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Win Rate Projection */}
      <Card className="glass border-primary-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Projected Win Rate
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-8">
            <div className="flex-1">
              <div className={`text-6xl font-bold ${getWinRateColor(projectedWinRate)}`}>
                {formatPercent(projectedWinRate, 1)}
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Based on {yourTeam.filter(s => s.hero).length} locked pick{yourTeam.filter(s => s.hero).length !== 1 ? 's' : ''}
                {yourTeam.some(s => s.playerStats) && ' • Using player stats'}
              </p>
            </div>

            {teamSynergies.length > 0 && (
              <div className="flex-1">
                <div className="rounded-lg border border-gaming-success/30 bg-gaming-success/5 p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle className="h-5 w-5 text-gaming-success" />
                    <p className="font-semibold text-gaming-success">Team Synergies!</p>
                  </div>
                  {teamSynergies.map((syn, i) => (
                    <p key={i} className="text-sm text-muted-foreground">
                      • {syn.hero1} + {syn.hero2} (+3%)
                    </p>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Recommendations */}
      {/* AI Team Composition Recommendations */}
      {teamCompositions.length > 0 && (
        <Card className="glass border-accent-cyan/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-accent-cyan" />
              Recommended Team Compositions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {teamCompositions.map((comp, idx) => (
                <div
                  key={idx}
                  className="rounded-lg border border-accent-cyan/30 bg-accent-cyan/5 p-4 transition-all hover:border-accent-cyan/60"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-cyan/20 text-sm font-bold text-accent-cyan">
                        #{idx + 1}
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-accent-cyan">
                          {comp.winRate.toFixed(1)}% Projected Win Rate
                        </p>
                        <p className="text-xs text-muted-foreground">{comp.explanation}</p>
                      </div>
                    </div>
                    <div className={`text-2xl font-bold ${getWinRateColor(comp.winRate)}`}>
                      {comp.winRate.toFixed(0)}%
                    </div>
                  </div>

                  {/* Team Lineup */}
                  <div className="grid grid-cols-5 gap-2">
                    {[0, 1, 2, 3, 4].map(slotIdx => {
                      const hero = comp.heroes.find(h => h.slot === slotIdx)
                      const currentHero = yourTeam[slotIdx].hero || yourTeam[slotIdx].showing
                      const displayHero = hero?.hero || currentHero
                      const playerName = yourTeam[slotIdx].battletag.split('#')[0] || `Slot ${slotIdx + 1}`
                      const isRecommendation = hero && !currentHero

                      return (
                        <div
                          key={slotIdx}
                          className={`rounded border p-2 text-center text-[11px] ${
                            currentHero
                              ? 'border-primary-500/50 bg-primary-500/10'
                              : 'border-accent-cyan/30 bg-accent-cyan/5'
                          }`}
                        >
                          <div className="font-bold text-[9px] text-muted-foreground mb-1 truncate">
                            {playerName}
                          </div>
                          <div className="font-semibold truncate">
                            {displayHero || '—'}
                          </div>
                          {hero && (
                            <div className={`text-[9px] mt-0.5 ${getWinRateColor(hero.winRate)}`}>
                              {hero.winRate.toFixed(1)}% • {hero.games}g
                            </div>
                          )}
                          {isRecommendation && displayHero && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-5 w-full mt-1 text-[9px] px-1"
                              onClick={() => {
                                if (displayHero) {
                                  setActiveSlot(slotIdx)
                                  setActiveTab('your')
                                  handleHeroClick(displayHero)
                                }
                              }}
                            >
                              Pick
                            </Button>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Hero Selection */}
      <Card className="glass border-primary-500/30">
        <CardHeader>
          <div className="space-y-4">
            <CardTitle>Available Heroes</CardTitle>

            {/* Tab Selection */}
            <div className="flex gap-2">
              <Button
                variant={activeTab === 'your' ? 'gaming' : 'outline'}
                size="sm"
                onClick={() => setActiveTab('your')}
              >
                Pre-Pick (Slot {activeSlot + 1})
              </Button>
              <Button
                variant={activeTab === 'enemy' ? 'gaming' : 'outline'}
                size="sm"
                onClick={() => setActiveTab('enemy')}
                disabled={enemyTeam.length >= 5}
              >
                Add Enemy ({enemyTeam.length}/5)
              </Button>
              <Button
                variant={activeTab === 'ban' ? 'gaming' : 'outline'}
                size="sm"
                onClick={() => setActiveTab('ban')}
                disabled={bannedHeroes.length >= 6}
              >
                Ban ({bannedHeroes.length}/6)
              </Button>
            </div>

            {/* Search */}
            <input
              type="text"
              placeholder="Search heroes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-10 w-full rounded-lg border border-input bg-background px-4 text-sm focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
            />
          </div>
        </CardHeader>
        <CardContent>
          {activeTab === 'your' && yourTeam[activeSlot].playerStats && Object.keys(yourTeam[activeSlot].playerStats!.heroStats).length > 0 && (
            <div className="mb-4 rounded-lg border border-gaming-success/30 bg-gaming-success/5 p-3">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-gaming-success" />
                <p className="text-sm font-semibold text-gaming-success">
                  Using {yourTeam[activeSlot].battletag.split('#')[0]}&apos;s stats ({Object.keys(yourTeam[activeSlot].playerStats!.heroStats).length} heroes)
                </p>
              </div>
            </div>
          )}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            {availableHeroes.map(hero => {
              const currentSlot = yourTeam[activeSlot]
              const baseWR = activeTab === 'your' ? getHeroWinRate(hero, activeSlot, false) : DEFAULT_HERO_STATS[hero]?.winRate || 50
              const adjustedWR = activeTab === 'your' ? getHeroWinRate(hero, activeSlot, true) : baseWR
              const hasSynergy = adjustedWR > baseWR
              const hasPlayerStats = activeTab === 'your' && currentSlot.playerStats?.heroStats[hero]
              const hasMinGames = hasPlayerStats && currentSlot.playerStats!.heroStats[hero].games >= 5

              return (
                <Button
                  key={hero}
                  variant="outline"
                  size="sm"
                  onClick={() => handleHeroClick(hero)}
                  className={`h-auto flex-col items-start p-3 text-left relative ${
                    hasPlayerStats && hasMinGames ? 'border-primary-500/50' : ''
                  } ${!hasPlayerStats && activeTab === 'your' ? 'opacity-60' : ''}`}
                >
                  {hasPlayerStats && hasMinGames && (
                    <div className="absolute top-1 right-1">
                      <CheckCircle className="h-3 w-3 text-primary-500" />
                    </div>
                  )}
                  <span className="font-semibold text-xs">{hero}</span>
                  <span className={`text-xs ${getWinRateColor(adjustedWR)}`}>
                    {adjustedWR.toFixed(1)}%
                    {hasSynergy && activeTab === 'your' && (
                      <span className={adjustedWR > baseWR ? "text-gaming-success" : "text-gaming-danger"}>
                        {' '}{adjustedWR > baseWR ? '+' : ''}{(adjustedWR - baseWR).toFixed(0)}
                      </span>
                    )}
                  </span>
                  {hasPlayerStats ? (
                    <span className="text-[10px] text-muted-foreground">
                      {currentSlot.playerStats!.heroStats[hero].games}g
                    </span>
                  ) : activeTab === 'your' && currentSlot.playerStats ? (
                    <span className="text-[10px] text-muted-foreground/50">
                      no data
                    </span>
                  ) : null}
                </Button>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
