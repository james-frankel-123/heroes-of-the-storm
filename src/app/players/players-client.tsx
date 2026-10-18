'use client'

import { useCallback, useEffect, useState } from 'react'
import Image from 'next/image'
import { useRouter, useSearchParams } from 'next/navigation'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { heroImageSrc } from '@/lib/data/hero-images'
import { cn } from '@/lib/utils'
import type { Candidate, PlayerResult, SearchResult } from '@/lib/personal/lookup'
import type { HeroEstimate, Suggestion, Validation } from '@/lib/personal/model'

type PlayerResponse = PlayerResult & { validation?: Validation }

const pp = (x: number) => `${x >= 0 ? '+' : ''}${(100 * x).toFixed(1)}`
const pct = (x: number) => `${Math.round(100 * x)}%`

function dayToDate(day: number): string {
  return new Date(day * 86_400_000).toISOString().slice(0, 10)
}

function verdict(lo: number, hi: number): { text: string; className: string } {
  if (lo > 0) return { text: 'Above average', className: 'text-gaming-success' }
  if (hi < 0) return { text: 'Below average', className: 'text-gaming-danger' }
  return { text: 'Not clearly different from average', className: 'text-muted-foreground' }
}

function HeroIcon({ hero }: { hero: string }) {
  return (
    <Image src={heroImageSrc(hero)} alt="" width={40} height={40} className="h-10 w-10 shrink-0 rounded-md object-cover" unoptimized />
  )
}

function CandidateCard({ c, onPick }: { c: Candidate; onPick: (c: Candidate) => void }) {
  return (
    <button onClick={() => onPick(c)} className="text-left">
      <Card className="p-4 hover:border-primary/60">
        <div className="flex items-baseline justify-between gap-2">
          <span className="font-semibold">{c.battletag}</span>
          <span className="text-sm text-muted-foreground">{c.regionName}</span>
        </div>
        <div className="mt-1 text-sm text-muted-foreground">
          {c.games.toLocaleString()} Storm League games, {c.games ? pct(c.wins / c.games) : '-'} won
          {c.rank ? `, ${c.rank}` : ''}
        </div>
        <div className="mt-1 text-sm text-muted-foreground">Last played {c.lastPlayed ?? 'unknown'}</div>
        {c.mostPlayed.length > 0 && (
          <div className="mt-2 flex items-center gap-2 text-sm">
            {c.mostPlayed.map(h => (
              <span key={h.hero} className="flex items-center gap-1">
                <Image src={heroImageSrc(h.hero)} alt="" width={20} height={20} className="h-5 w-5 rounded" unoptimized />
                {h.hero} ({h.games})
              </span>
            ))}
          </div>
        )}
      </Card>
    </button>
  )
}

function HeroCard({ h }: { h: HeroEstimate }) {
  const v = verdict(h.lo, h.hi)
  return (
    <Card className="flex gap-3 p-3">
      <HeroIcon hero={h.hero} />
      <div className="min-w-0">
        <div className="flex items-baseline gap-2">
          <span className="font-semibold">{h.hero}</span>
          <span className="text-lg font-bold tabular-nums">{pp(h.mean)} pp</span>
        </div>
        <div className="text-xs text-muted-foreground">
          {h.games} games, {pct(h.wins / h.games)} won, last {dayToDate(h.lastDay)}
        </div>
        <div className="text-xs text-muted-foreground tabular-nums">
          80% range {pp(h.lo)} to {pp(h.hi)}
        </div>
        <div className={cn('text-xs', v.className)}>{v.text}</div>
      </div>
    </Card>
  )
}

function SuggestionCard({ s }: { s: Suggestion }) {
  return (
    <Card className="flex gap-3 p-3">
      <HeroIcon hero={s.hero} />
      <div>
        <div className="flex items-baseline gap-2">
          <span className="font-semibold">{s.hero}</span>
          <span className="font-bold tabular-nums">{pp(s.mean)} pp</span>
        </div>
        <div className="text-xs text-muted-foreground tabular-nums">
          80% range {pp(s.lo)} to {pp(s.hi)}, once learned
        </div>
      </div>
    </Card>
  )
}

function Profile({ r }: { r: PlayerResponse }) {
  const [showAll, setShowAll] = useState(false)
  const p = r.profile!
  const c = r.candidate!
  const main = p.model === 'main'
  const recent = r.historySource?.startsWith('matches:') ? Number(r.historySource.split(':')[1]) : null
  const heroes = showAll ? p.heroes : p.heroes.slice(0, 18)
  const val = r.validation
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">{c.battletag}</h2>
        <p className="text-sm text-muted-foreground">
          {c.regionName}, {c.games.toLocaleString()} Storm League games on Heroes Profile
          {recent !== null ? `; these estimates use the ${recent} most recent` : ''}.
          {r.computedAt ? ` Updated ${r.computedAt.slice(0, 10)}.` : ''}
        </p>
        {r.stale && r.message && <p className="mt-1 text-sm text-gaming-warning">{r.message}</p>}
      </div>

      <section className="space-y-2">
        <h3 className="text-lg font-semibold">Skill by hero</h3>
        <p className="text-sm text-muted-foreground">
          Each number is how many percentage points more (or fewer) games we expect this player to win on the hero than an
          average player in the same draft. {main
            ? 'It uses wins, losses and scoreboard stats from past games.'
            : 'It uses wins and losses only.'}
          {!main && r.metricPending ? ' Scoreboard stats from our replay data are added within a day if we have any for this player.' : ''}
        </p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {heroes.map(h => (
            <HeroCard key={h.hero} h={h} />
          ))}
        </div>
        {p.heroes.length > 18 && (
          <Button variant="outline" size="sm" onClick={() => setShowAll(!showAll)}>
            {showAll ? 'Show fewer' : `Show all ${p.heroes.length} heroes`}
          </Button>
        )}
      </section>

      {p.suggestions.length > 0 && (
        <section className="space-y-2">
          <h3 className="text-lg font-semibold">Heroes you might be good at</h3>
          <p className="text-sm text-muted-foreground">
            Heroes this player has not played, ranked by what players with similar records do on them. The number is
            the expected edge once the hero is learned; first games on a new hero usually run about 3 points worse.
          </p>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {p.suggestions.map(s => (
              <SuggestionCard key={s.hero} s={s} />
            ))}
          </div>
        </section>
      )}

      <section className="space-y-1 text-sm text-muted-foreground">
        <h3 className="text-base font-semibold text-foreground">How far to trust this</h3>
        <p>
          One game says little: a single win or loss is mostly noise. The ranges are 80% ranges. When we tested them on
          games the model never saw, the true value fell inside about {val ? pct(main ? val.cover80.main : val.cover80.imp2) : '80%'} of
          the time, so roughly 1 range in 5 misses.
        </p>
        {val && (
          <p>
            Combined with the draft, estimates like these for all ten players called the winner of{' '}
            {(100 * val.accuracy.v2.main).toFixed(1)}% of held-out games with scoreboard stats and{' '}
            {(100 * val.accuracy.v2.imp2).toFixed(1)}% with wins and losses only. Real gains, and a reminder that
            the draft and the players leave most of a game unexplained.
          </p>
        )}
        <p>Private Heroes Profile accounts are never shown. Data provided by Heroes Profile.</p>
      </section>
    </div>
  )
}

export function PlayersClient() {
  const router = useRouter()
  const params = useSearchParams()
  const [q, setQ] = useState(params.get('q') ?? '')
  const [search, setSearch] = useState<SearchResult | null>(null)
  const [player, setPlayer] = useState<PlayerResponse | null>(null)
  const [busy, setBusy] = useState<'search' | 'player' | null>(null)
  const [error, setError] = useState<string | null>(null)

  const loadPlayer = useCallback(async (tag: string, region: number) => {
    setBusy('player')
    setError(null)
    setPlayer(null)
    try {
      const res = await fetch(`/api/personal/player?tag=${encodeURIComponent(tag)}&region=${region}`)
      const body = (await res.json()) as PlayerResponse
      if (body.status === 'ok') setPlayer(body)
      else setError(body.message ?? 'Lookup failed.')
    } catch {
      setError('Lookup failed. Try again.')
    } finally {
      setBusy(null)
    }
  }, [])

  const runSearch = useCallback(async (query: string) => {
    setBusy('search')
    setError(null)
    setPlayer(null)
    try {
      const res = await fetch(`/api/personal/search?q=${encodeURIComponent(query)}`)
      const body = await res.json()
      if (!res.ok) {
        setError(body.error ?? 'Search failed.')
        setSearch(null)
      } else {
        setSearch(body as SearchResult)
      }
    } catch {
      setError('Search failed. Try again.')
    } finally {
      setBusy(null)
    }
  }, [])

  const tagParam = params.get('tag')
  const regionParam = params.get('region')
  const qParam = params.get('q')
  useEffect(() => {
    if (tagParam && regionParam) loadPlayer(tagParam, Number(regionParam))
    else if (qParam) runSearch(qParam)
  }, [tagParam, regionParam, qParam, loadPlayer, runSearch])

  const pick = (c: Candidate) => {
    router.push(`/players?tag=${encodeURIComponent(c.battletag)}&region=${c.region}`)
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Player lookup</h1>
        <p className="text-muted-foreground">
          Per-hero skill estimates for any public Heroes Profile account. Enter a BattleTag with its number, like Name#1234.
        </p>
      </div>
      <form
        className="flex gap-2"
        onSubmit={e => {
          e.preventDefault()
          if (q.trim()) router.push(`/players?q=${encodeURIComponent(q.trim())}`)
        }}
      >
        <Input value={q} onChange={e => setQ(e.target.value)} placeholder="Name#1234" maxLength={24} aria-label="BattleTag" />
        <Button type="submit" disabled={busy !== null}>
          {busy === 'search' ? 'Searching...' : 'Search'}
        </Button>
      </form>

      {error && <p className="text-sm text-gaming-danger">{error}</p>}

      {search && !player && busy !== 'player' && (
        <section className="space-y-3">
          {search.unavailable && <p className="text-sm text-gaming-warning">{search.unavailable}</p>}
          {search.candidates.length > 1 && <p className="text-sm text-muted-foreground">More than one account matches. Pick one:</p>}
          {search.candidates.length === 0 && !search.unavailable && (
            <p className="text-sm text-muted-foreground">No public Storm League account found.</p>
          )}
          <div className="grid gap-3 sm:grid-cols-2">
            {search.candidates.map(c => (
              <CandidateCard key={`${c.region}:${c.battletag}`} c={c} onPick={pick} />
            ))}
          </div>
          {search.privateRegions.length > 0 && (
            <p className="text-sm text-muted-foreground">
              Private on Heroes Profile in {search.privateRegions.join(', ')}; nothing is shown for those.
            </p>
          )}
          {search.hint && <p className="text-sm text-muted-foreground">{search.hint}</p>}
        </section>
      )}

      {busy === 'player' && <p className="text-sm text-muted-foreground">Loading games from Heroes Profile...</p>}
      {player && <Profile r={player} />}
    </div>
  )
}
