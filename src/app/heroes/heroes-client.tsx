'use client'

import { useState, useMemo, useCallback } from 'react'
import { TierSelector, getTierLabel } from '@/components/shared/tier-selector'
import { HeroTable } from '@/components/heroes/hero-table'
import type { HeroRow } from '@/components/heroes/hero-table'
import { HeroDetailModal } from '@/components/heroes/hero-detail-modal'
import { cn } from '@/lib/utils'
import { heroImageSrc } from '@/lib/data/hero-images'
import { mapImageSrc } from '@/lib/data/map-images'
import type {
  SkillTier,
  HeroStats,
  HeroMapStats,
  HeroTalentStats,
  HeroPairwiseStats,
  PlayerHeroStats,
  PlayerMatch,
} from '@/lib/types'

/** 'global' for aggregate data, or a battletag string for personal */
type ViewMode = 'global' | string

interface HeroesClientProps {
  heroStatsByTier: Record<SkillTier, HeroStats[]>
  heroStatsByName: Record<string, HeroStats[]>
  talentsByTier: Record<SkillTier, Record<string, HeroTalentStats[]>>
  pairwiseByTier: Record<
    SkillTier,
    Record<string, { synergies: HeroPairwiseStats[]; counters: HeroPairwiseStats[] }>
  >
  heroMapByTier: Record<SkillTier, Record<string, HeroMapStats[]>>
  personalData: {
    battletag: string
    heroStats: PlayerHeroStats[]
    matches: PlayerMatch[]
    mapStats: { map: string; games: number; wins: number; winRate: number }[]
    seasonHeroStats: { hero: string; games: number; wins: number }[]
    seasonMapStats: { map: string; games: number; wins: number; winRate: number }[]
    threeSeasonHeroStats: { hero: string; games: number; wins: number }[]
    threeSeasonMapStats: { map: string; games: number; wins: number; winRate: number }[]
  }[]
}

/** Convert PlayerHeroStats to HeroRow so the same table component works */
function toHeroRow(p: PlayerHeroStats): HeroRow {
  return {
    hero: p.hero,
    skillTier: 'mid' as SkillTier, // not tier-bucketed
    games: p.games,
    wins: p.wins,
    winRate: p.winRate,
    banRate: 0,
    pickRate: 0,
    avgKills: p.avgKills,
    avgDeaths: p.avgDeaths,
    avgAssists: p.avgAssists,
    avgHeroDamage: 0,
    avgSiegeDamage: 0,
    avgHealing: 0,
    avgExperience: 0,
    avgDamageSoaked: 0,
    avgMercCaptures: 0,
    avgSelfHealing: 0,
    avgTimeDead: 0,
    patchTag: null,
    mawp: p.mawp,
  }
}

export function HeroesClient({
  heroStatsByTier,
  heroStatsByName,
  talentsByTier,
  pairwiseByTier,
  heroMapByTier,
  personalData,
}: HeroesClientProps) {
  const [tier, setTier] = useState<SkillTier>('mid')
  const [viewMode, setViewMode] = useState<ViewMode>('global')
  const [selectedHero, setSelectedHero] = useState<string | null>(null)

  // Search state
  const [showSearch, setShowSearch] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [searchSuggestions, setSearchSuggestions] = useState<string[] | null>(null)
  const [searchedPlayers, setSearchedPlayers] = useState<typeof personalData>([])

  const allPersonalData = useMemo(
    () => [...personalData, ...searchedPlayers],
    [personalData, searchedPlayers]
  )

  const handleSearch = useCallback(async (query: string) => {
    if (!query || query.length < 2) return
    setSearchLoading(true)
    setSearchError(null)
    setSearchSuggestions(null)
    try {
      const res = await fetch(`/api/player-search?q=${encodeURIComponent(query)}`)
      const data = await res.json()
      if (data.error === 'No players found') {
        setSearchError('No players found')
        return
      }
      if (data.results && !data.battletag) {
        setSearchSuggestions(data.results)
        return
      }
      if (!data.battletag) {
        setSearchError('No results')
        return
      }
      // Check if already loaded
      if (!allPersonalData.some(p => p.battletag === data.battletag)) {
        setSearchedPlayers(prev => [...prev, {
          battletag: data.battletag,
          heroStats: data.heroStats.map((h: any) => ({
            hero: h.hero, games: h.games, wins: h.wins,
            winRate: h.winRate, mawp: null,
            avgKills: 0, avgDeaths: 0, avgAssists: 0,
            recentWinRate: null, trend: null,
          })),
          matches: [],
          mapStats: data.mapStats,
          seasonHeroStats: data.seasonHeroStats,
          seasonMapStats: data.seasonMapStats ?? [],
          threeSeasonHeroStats: data.threeSeasonHeroStats ?? [],
          threeSeasonMapStats: data.threeSeasonMapStats ?? [],
        }])
      }
      setViewMode(data.battletag)
      setShowSearch(false)
      setSearchQuery('')
    } catch {
      setSearchError('Search failed')
    } finally {
      setSearchLoading(false)
    }
  }, [allPersonalData])

  const isPersonal = viewMode !== 'global'

  // Build rows for current view
  const heroes: HeroRow[] = useMemo(() => {
    if (!isPersonal) return heroStatsByTier[tier]
    const pd = allPersonalData.find((p) => p.battletag === viewMode)
    if (!pd) return []
    return pd.heroStats.map(toHeroRow)
  }, [isPersonal, viewMode, tier, heroStatsByTier, allPersonalData])

  // Subtitle text
  const subtitle = isPersonal
    ? `Personal stats for ${viewMode}`
    : `Aggregate hero stats \u2014 ${getTierLabel(tier)}`

  // Build detail data for selected hero
  const detailStatsByTier = selectedHero
    ? heroStatsByName[selectedHero] ?? []
    : []

  const detailTalents = selectedHero
    ? talentsByTier[tier][selectedHero] ?? []
    : []

  const detailMapStats = selectedHero
    ? heroMapByTier[tier][selectedHero] ?? []
    : []

  const detailPairwise = selectedHero
    ? pairwiseByTier[tier][selectedHero] ?? { synergies: [], counters: [] }
    : { synergies: [], counters: [] }

  const detailPersonal = selectedHero
    ? allPersonalData.map((p) => ({
        battletag: p.battletag,
        stats:
          p.heroStats.find((h) => h.hero === selectedHero) ?? null,
      }))
    : []

  const detailMatches = selectedHero
    ? allPersonalData.map((p) => ({
        battletag: p.battletag,
        matches: p.matches.filter((m) => m.hero === selectedHero),
      }))
    : []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Heroes</h1>
          <p className="text-muted-foreground mt-1">{subtitle}</p>
        </div>
        {/* Only show tier selector in global mode */}
        {!isPersonal && <TierSelector value={tier} onChange={setTier} />}
      </div>

      {/* View mode selector */}
      <div className="space-y-2">
        <div className="flex items-center gap-1 p-1 rounded-lg bg-muted w-fit flex-wrap">
          <button
            onClick={() => { setViewMode('global'); setShowSearch(false) }}
            className={cn(
              'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
              viewMode === 'global' && !showSearch
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            Global
          </button>
          {allPersonalData.map((p) => (
            <button
              key={p.battletag}
              onClick={() => { setViewMode(p.battletag); setShowSearch(false) }}
              className={cn(
                'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                viewMode === p.battletag && !showSearch
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {p.battletag.split('#')[0]}
            </button>
          ))}
          <button
            onClick={() => setShowSearch(!showSearch)}
            className={cn(
              'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
              showSearch
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            Search
          </button>
        </div>

        {showSearch && (
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSearch(searchQuery)
              }}
              placeholder="Enter battletag (e.g. SirWatsonII)"
              className="px-3 py-1.5 rounded-md text-sm border border-border bg-background text-foreground placeholder:text-muted-foreground w-64"
              autoFocus
            />
            <button
              onClick={() => handleSearch(searchQuery)}
              disabled={searchLoading || searchQuery.length < 2}
              className={cn(
                'px-4 py-1.5 rounded-md text-sm font-medium transition-colors',
                searchLoading
                  ? 'bg-muted text-muted-foreground cursor-wait'
                  : 'bg-primary text-primary-foreground hover:bg-primary/90'
              )}
            >
              {searchLoading ? 'Searching...' : 'Go'}
            </button>
            {searchError && (
              <span className="text-xs text-gaming-danger">{searchError}</span>
            )}
          </div>
        )}

        {showSearch && searchSuggestions && (
          <div className="flex items-center gap-1 flex-wrap">
            <span className="text-xs text-muted-foreground">Did you mean:</span>
            {searchSuggestions.map((bt) => (
              <button
                key={bt}
                onClick={() => { setSearchQuery(bt); handleSearch(bt) }}
                className="text-xs text-primary hover:underline"
              >
                {bt}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Player Snapshot — top 5 heroes + top 3 maps */}
      {isPersonal && (() => {
        const pd = allPersonalData.find((p) => p.battletag === viewMode)
        if (!pd) return null
        const topHeroes = [...pd.heroStats]
          .filter((h) => h.games >= 20)
          .sort((a, b) => b.winRate - a.winRate)
          .slice(0, 5)
        const topMaps = [...pd.mapStats]
          .filter((m) => m.games >= 20)
          .sort((a, b) => b.winRate - a.winRate)
          .slice(0, 3)
        const seasonTopHeroes = pd.seasonHeroStats
          .filter((h) => h.games >= 5)
          .map((h) => ({ ...h, winRate: h.games > 0 ? Math.round((h.wins / h.games) * 1000) / 10 : 0 }))
          .sort((a, b) => b.winRate - a.winRate)
          .slice(0, 5)
        const seasonTopMaps = (pd.seasonMapStats ?? [])
          .filter((m) => m.games >= 5)
          .sort((a, b) => b.winRate - a.winRate)
          .slice(0, 3)
        const threeSeasonTopHeroes = (pd.threeSeasonHeroStats ?? [])
          .filter((h) => h.games >= 10)
          .map((h) => ({ ...h, winRate: h.games > 0 ? Math.round((h.wins / h.games) * 1000) / 10 : 0 }))
          .sort((a, b) => b.winRate - a.winRate)
          .slice(0, 5)
        const threeSeasonTopMaps = (pd.threeSeasonMapStats ?? [])
          .filter((m) => m.games >= 10)
          .sort((a, b) => b.winRate - a.winRate)
          .slice(0, 3)
        return (
          <PlayerSnapshot
            player={viewMode}
            topHeroes={topHeroes}
            topMaps={topMaps}
            seasonTopHeroes={seasonTopHeroes}
            seasonTopMaps={seasonTopMaps}
            threeSeasonTopHeroes={threeSeasonTopHeroes}
            threeSeasonTopMaps={threeSeasonTopMaps}
          />
        )
      })()}

      {/* Career Snapshot — top/bottom 5 by win surplus */}
      {isPersonal && (() => {
        const pd = allPersonalData.find((p) => p.battletag === viewMode)
        if (!pd) return null
        const withDiff = pd.heroStats
          .filter((h) => h.games >= 5)
          .map((h) => ({ ...h, diff: h.wins - (h.games - h.wins) }))
        const best = [...withDiff].sort((a, b) => b.diff - a.diff).slice(0, 5).filter(h => h.diff > 0)
        const worst = [...withDiff].sort((a, b) => a.diff - b.diff).slice(0, 5).filter(h => h.diff < 0)
        // WAR: player winRate vs global average winRate for each hero
        const globalStats = heroStatsByTier['mid']
        const globalMap: Record<string, number> = {}
        for (const g of globalStats) globalMap[g.hero] = g.winRate
        const war = pd.heroStats
          .filter((h) => h.games >= 10 && globalMap[h.hero] !== undefined)
          .map((h) => ({
            hero: h.hero,
            games: h.games,
            playerWR: h.winRate,
            globalWR: globalMap[h.hero],
            delta: Math.round((h.winRate - globalMap[h.hero]) * 10) / 10,
          }))
          .sort((a, b) => b.delta - a.delta)
          .slice(0, 10)
        return <CareerSnapshot best={best} worst={worst} war={war} />
      })()}

      {/* This Season Snapshot */}
      {isPersonal && (() => {
        const pd = allPersonalData.find((p) => p.battletag === viewMode)
        if (!pd || pd.seasonHeroStats.length === 0) return null
        const withDiff = pd.seasonHeroStats
          .filter((h) => h.games >= 3)
          .map((h) => ({ ...h, diff: h.wins - (h.games - h.wins) }))
        const best = [...withDiff].sort((a, b) => b.diff - a.diff).slice(0, 5).filter(h => h.diff > 0)
        const worst = [...withDiff].sort((a, b) => a.diff - b.diff).slice(0, 5).filter(h => h.diff < 0)
        if (best.length === 0 && worst.length === 0) return null
        return <SeasonSnapshot best={best} worst={worst} />
      })()}

      <HeroTable
        heroes={heroes}
        onHeroClick={setSelectedHero}
        personal={isPersonal}
      />

      {selectedHero && (
        <HeroDetailModal
          open={!!selectedHero}
          onClose={() => setSelectedHero(null)}
          heroName={selectedHero}
          statsByTier={detailStatsByTier}
          mapStats={detailMapStats}
          talents={detailTalents}
          synergies={detailPairwise.synergies}
          counters={detailPairwise.counters}
          personalStats={detailPersonal}
          recentMatches={detailMatches}
          currentTier={tier}
        />
      )}
    </div>
  )
}

function PlayerSnapshot({
  player,
  topHeroes,
  topMaps,
  seasonTopHeroes,
  seasonTopMaps,
}: {
  player: string
  topHeroes: PlayerHeroStats[]
  topMaps: { map: string; games: number; wins: number; winRate: number }[]
  seasonTopHeroes: { hero: string; games: number; wins: number; winRate: number }[]
  seasonTopMaps: { map: string; games: number; wins: number; winRate: number }[]
  threeSeasonTopHeroes: { hero: string; games: number; wins: number; winRate: number }[]
  threeSeasonTopMaps: { map: string; games: number; wins: number; winRate: number }[]
}) {
  const name = player.split('#')[0]
  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-4">
      <h3 className="text-sm font-semibold text-white">
        Player Snapshot &mdash; {name}
      </h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        {/* Top heroes */}
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
            Top Heroes
          </p>
          <div className="flex gap-3">
            {topHeroes.map((h) => {
              const wrColor = h.winRate >= 55
                ? 'text-gaming-success'
                : h.winRate >= 50
                  ? 'text-gaming-warning'
                  : 'text-gaming-danger'
              return (
                <div key={h.hero} className="flex flex-col items-center gap-1 w-14">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={heroImageSrc(h.hero)}
                    alt={h.hero}
                    className="w-12 h-12 rounded-md object-cover border border-border"
                  />
                  <span className="text-[10px] text-muted-foreground truncate w-full text-center">
                    {h.hero}
                  </span>
                  <span className={cn('text-xs font-bold tabular-nums', wrColor)}>
                    {h.winRate.toFixed(1)}%
                  </span>
                  <span className="text-[9px] text-muted-foreground tabular-nums">
                    {h.games}g
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Top maps */}
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
            Top Maps
          </p>
          <div className="flex gap-3">
            {topMaps.map((m) => {
              const img = mapImageSrc(m.map)
              const wrColor = m.winRate >= 55
                ? 'text-gaming-success'
                : m.winRate >= 50
                  ? 'text-gaming-warning'
                  : 'text-gaming-danger'
              return (
                <div key={m.map} className="flex flex-col items-center gap-1 w-20">
                  {img && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={img}
                      alt={m.map}
                      className="w-full h-10 rounded-md object-cover border border-border"
                    />
                  )}
                  <span className="text-[10px] text-muted-foreground truncate w-full text-center">
                    {m.map}
                  </span>
                  <span className={cn('text-xs font-bold tabular-nums', wrColor)}>
                    {m.winRate.toFixed(1)}%
                  </span>
                  <span className="text-[9px] text-muted-foreground tabular-nums">
                    {m.games}g
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* This Season sections */}
      {(seasonTopHeroes.length > 0 || seasonTopMaps.length > 0) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-2 border-t border-border">
          {/* Season top heroes */}
          {seasonTopHeroes.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
                This Season Top Heroes
              </p>
              <div className="flex gap-3">
                {seasonTopHeroes.map((h) => {
                  const wrColor = h.winRate >= 55
                    ? 'text-gaming-success'
                    : h.winRate >= 50
                      ? 'text-gaming-warning'
                      : 'text-gaming-danger'
                  return (
                    <div key={h.hero} className="flex flex-col items-center gap-1 w-14">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={heroImageSrc(h.hero)}
                        alt={h.hero}
                        className="w-12 h-12 rounded-md object-cover border border-border"
                      />
                      <span className="text-[10px] text-muted-foreground truncate w-full text-center">
                        {h.hero}
                      </span>
                      <span className={cn('text-xs font-bold tabular-nums', wrColor)}>
                        {h.winRate.toFixed(1)}%
                      </span>
                      <span className="text-[9px] text-muted-foreground tabular-nums">
                        {h.games}g
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Season top maps */}
          {seasonTopMaps.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
                This Season Top Maps
              </p>
              <div className="flex gap-3">
                {seasonTopMaps.map((m) => {
                  const img = mapImageSrc(m.map)
                  const wrColor = m.winRate >= 55
                    ? 'text-gaming-success'
                    : m.winRate >= 50
                      ? 'text-gaming-warning'
                      : 'text-gaming-danger'
                  return (
                    <div key={m.map} className="flex flex-col items-center gap-1 w-20">
                      {img && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={img}
                          alt={m.map}
                          className="w-full h-10 rounded-md object-cover border border-border"
                        />
                      )}
                      <span className="text-[10px] text-muted-foreground truncate w-full text-center">
                        {m.map}
                      </span>
                      <span className={cn('text-xs font-bold tabular-nums', wrColor)}>
                        {m.winRate.toFixed(1)}%
                      </span>
                      <span className="text-[9px] text-muted-foreground tabular-nums">
                        {m.games}g
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Last 3 Seasons sections */}
      {(threeSeasonTopHeroes.length > 0 || threeSeasonTopMaps.length > 0) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-2 border-t border-border">
          {threeSeasonTopHeroes.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
                Last 3 Seasons Top Heroes
              </p>
              <div className="flex gap-3">
                {threeSeasonTopHeroes.map((h) => {
                  const wrColor = h.winRate >= 55
                    ? 'text-gaming-success'
                    : h.winRate >= 50
                      ? 'text-gaming-warning'
                      : 'text-gaming-danger'
                  return (
                    <div key={h.hero} className="flex flex-col items-center gap-1 w-14">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={heroImageSrc(h.hero)}
                        alt={h.hero}
                        className="w-12 h-12 rounded-md object-cover border border-border"
                      />
                      <span className="text-[10px] text-muted-foreground truncate w-full text-center">
                        {h.hero}
                      </span>
                      <span className={cn('text-xs font-bold tabular-nums', wrColor)}>
                        {h.winRate.toFixed(1)}%
                      </span>
                      <span className="text-[9px] text-muted-foreground tabular-nums">
                        {h.games}g
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {threeSeasonTopMaps.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
                Last 3 Seasons Top Maps
              </p>
              <div className="flex gap-3">
                {threeSeasonTopMaps.map((m) => {
                  const img = mapImageSrc(m.map)
                  const wrColor = m.winRate >= 55
                    ? 'text-gaming-success'
                    : m.winRate >= 50
                      ? 'text-gaming-warning'
                      : 'text-gaming-danger'
                  return (
                    <div key={m.map} className="flex flex-col items-center gap-1 w-20">
                      {img && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={img}
                          alt={m.map}
                          className="w-full h-10 rounded-md object-cover border border-border"
                        />
                      )}
                      <span className="text-[10px] text-muted-foreground truncate w-full text-center">
                        {m.map}
                      </span>
                      <span className={cn('text-xs font-bold tabular-nums', wrColor)}>
                        {m.winRate.toFixed(1)}%
                      </span>
                      <span className="text-[9px] text-muted-foreground tabular-nums">
                        {m.games}g
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function CareerSnapshot({
  best,
  worst,
  war,
}: {
  best: (PlayerHeroStats & { diff: number })[]
  worst: (PlayerHeroStats & { diff: number })[]
  war: { hero: string; games: number; playerWR: number; globalWR: number; delta: number }[]
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-4">
      <h3 className="text-sm font-semibold text-white">Career Snapshot</h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        {/* Best — positive career +/- */}
        {best.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
              Biggest Winners
            </p>
            <div className="space-y-1.5">
              {best.map((h) => (
                <div key={h.hero} className="flex items-center gap-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={heroImageSrc(h.hero)}
                    alt=""
                    className="w-8 h-8 rounded object-cover border border-border"
                  />
                  <span className="text-sm text-foreground flex-1 truncate">{h.hero}</span>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {h.wins}W-{h.games - h.wins}L
                  </span>
                  <span className="text-sm font-bold tabular-nums text-gaming-success">
                    +{h.diff}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Worst — negative career +/- */}
        {worst.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
              Biggest Losers
            </p>
            <div className="space-y-1.5">
              {worst.map((h) => (
                <div key={h.hero} className="flex items-center gap-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={heroImageSrc(h.hero)}
                    alt=""
                    className="w-8 h-8 rounded object-cover border border-border"
                  />
                  <span className="text-sm text-foreground flex-1 truncate">{h.hero}</span>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {h.wins}W-{h.games - h.wins}L
                  </span>
                  <span className="text-sm font-bold tabular-nums text-gaming-danger">
                    {h.diff}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Wins Above Replacement */}
      {war.length > 0 && (
        <div className="space-y-2 pt-2 border-t border-border">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
            Wins Above Replacement (top 10)
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1.5">
            {war.map((h) => {
              const deltaColor = h.delta > 0 ? 'text-gaming-success' : h.delta < 0 ? 'text-gaming-danger' : 'text-muted-foreground'
              return (
                <div key={h.hero} className="flex items-center gap-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={heroImageSrc(h.hero)}
                    alt=""
                    className="w-8 h-8 rounded object-cover border border-border"
                  />
                  <span className="text-sm text-foreground flex-1 truncate">{h.hero}</span>
                  <span className="text-[10px] text-muted-foreground tabular-nums">
                    {h.playerWR.toFixed(1)}% vs {h.globalWR.toFixed(1)}%
                  </span>
                  <span className={cn('text-sm font-bold tabular-nums', deltaColor)}>
                    {h.delta > 0 ? '+' : ''}{h.delta.toFixed(1)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function SeasonSnapshot({
  best,
  worst,
}: {
  best: { hero: string; games: number; wins: number; diff: number }[]
  worst: { hero: string; games: number; wins: number; diff: number }[]
}) {
  const year = new Date().getFullYear()
  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-4">
      <h3 className="text-sm font-semibold text-white">
        This Season Snapshot &mdash; {year}
      </h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        {best.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
              Biggest Winners
            </p>
            <div className="space-y-1.5">
              {best.map((h) => (
                <div key={h.hero} className="flex items-center gap-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={heroImageSrc(h.hero)}
                    alt=""
                    className="w-8 h-8 rounded object-cover border border-border"
                  />
                  <span className="text-sm text-foreground flex-1 truncate">{h.hero}</span>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {h.wins}W-{h.games - h.wins}L
                  </span>
                  <span className="text-sm font-bold tabular-nums text-gaming-success">
                    +{h.diff}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {worst.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
              Biggest Losers
            </p>
            <div className="space-y-1.5">
              {worst.map((h) => (
                <div key={h.hero} className="flex items-center gap-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={heroImageSrc(h.hero)}
                    alt=""
                    className="w-8 h-8 rounded object-cover border border-border"
                  />
                  <span className="text-sm text-foreground flex-1 truncate">{h.hero}</span>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {h.wins}W-{h.games - h.wins}L
                  </span>
                  <span className="text-sm font-bold tabular-nums text-gaming-danger">
                    {h.diff}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
