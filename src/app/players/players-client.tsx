'use client'

import { useEffect, useState } from 'react'
import Image from 'next/image'
import { useRouter, useSearchParams } from 'next/navigation'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { heroImageSrc } from '@/lib/data/hero-images'
import { cn } from '@/lib/utils'
import type { AccountWithCard } from '@/lib/personal/accounts'
import type { PlayedHero, SuggestedHero } from '@/lib/personal/card'

interface Validation {
  accuracy: { v2: Record<string, number> }
  cover80: Record<string, number>
}

interface AccountResponse {
  through: string
  accounts: AccountWithCard[]
  validation: Validation
  error?: string
}

const Z80 = 1.2815515655446004
const pp = (x: number) => `${x >= 0 ? '+' : ''}${(100 * x).toFixed(1)}`
const pct = (x: number) => `${Math.round(100 * x)}%`

function verdict(lo: number, hi: number): { text: string; className: string } {
  if (lo > 0) return { text: 'Above average', className: 'text-gaming-success' }
  if (hi < 0) return { text: 'Below average', className: 'text-gaming-danger' }
  return { text: 'Not clearly different from average', className: 'text-muted-foreground' }
}

function HeroIcon({ hero, size = 40 }: { hero: string; size?: number }) {
  return (
    <Image
      src={heroImageSrc(hero)}
      alt=""
      width={size}
      height={size}
      className="shrink-0 rounded-md object-cover"
      style={{ width: size, height: size }}
      unoptimized
    />
  )
}

export function AccountChoice({ a, onPick }: { a: AccountWithCard | Omit<AccountWithCard, 'card'>; onPick?: () => void }) {
  return (
    <button onClick={onPick} className="w-full text-left" type="button">
      <Card className="p-3 hover:border-primary/60">
        <div className="flex items-baseline justify-between gap-2">
          <span className="font-semibold">
            {a.name}#{a.discriminator}
          </span>
          <span className="text-sm text-muted-foreground">{a.regionName}</span>
        </div>
        <div className="mt-1 text-xs text-muted-foreground">
          {a.games.toLocaleString()} games in our data, last played {a.lastPlayed}
          {a.mmr !== null ? `, MMR ${a.mmr.toLocaleString()}` : ''}
        </div>
        {a.topHeroes.length > 0 && (
          <div className="mt-1 flex items-center gap-1 text-xs">
            {a.topHeroes.map(h => (
              <span key={h} className="flex items-center gap-1">
                <HeroIcon hero={h} size={18} />
                {h}
              </span>
            ))}
          </div>
        )}
      </Card>
    </button>
  )
}

function HeroCard({ h }: { h: PlayedHero }) {
  const lo = h.mean - Z80 * h.sd
  const hi = h.mean + Z80 * h.sd
  const v = verdict(lo, hi)
  return (
    <Card className="flex gap-3 p-3">
      <HeroIcon hero={h.hero} />
      <div className="min-w-0">
        <div className="flex items-baseline gap-2">
          <span className="font-semibold">{h.hero}</span>
          <span className="text-lg font-bold tabular-nums">{pp(h.mean)} pp</span>
        </div>
        <div className="text-xs text-muted-foreground">
          {h.games} games, last {h.lastPlayed}
        </div>
        <div className="text-xs tabular-nums text-muted-foreground">
          80% range {pp(lo)} to {pp(hi)}
        </div>
        <div className={cn('text-xs', v.className)}>{v.text}</div>
      </div>
    </Card>
  )
}

function SuggestionCard({ s }: { s: SuggestedHero }) {
  return (
    <Card className="flex gap-3 p-3">
      <HeroIcon hero={s.hero} />
      <div>
        <div className="flex items-baseline gap-2">
          <span className="font-semibold">{s.hero}</span>
          <span className="font-bold tabular-nums">{pp(s.mean)} pp</span>
        </div>
        <div className="text-xs tabular-nums text-muted-foreground">
          80% range {pp(s.mean - Z80 * s.sd)} to {pp(s.mean + Z80 * s.sd)}, once learned
        </div>
      </div>
    </Card>
  )
}

function Profile({ a, through, val }: { a: AccountWithCard; through: string; val: Validation }) {
  const [showAll, setShowAll] = useState(false)
  const main = a.card.model === 'main'
  const heroes = showAll ? a.card.played : a.card.played.slice(0, 18)
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">
          {a.name}#{a.discriminator}
        </h2>
        <p className="text-sm text-muted-foreground">
          {a.regionName}, {a.games.toLocaleString()} Storm League games in our data, last played {a.lastPlayed}
          {a.mmr !== null ? `, MMR ${a.mmr.toLocaleString()}` : ''}. Estimates as of {through}.
        </p>
      </div>

      <section className="space-y-2">
        <h3 className="text-lg font-semibold">Skill by hero</h3>
        <p className="text-sm text-muted-foreground">
          Each number is how many percentage points more (or fewer) games we expect this player to win on the hero
          than an average player in the same draft.{' '}
          {main ? 'It uses wins, losses and scoreboard stats from past games.' : 'It uses wins and losses only.'}
        </p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {heroes.map(h => (
            <HeroCard key={h.hero} h={h} />
          ))}
        </div>
        {a.card.played.length > 18 && (
          <Button variant="outline" size="sm" onClick={() => setShowAll(!showAll)}>
            {showAll ? 'Show fewer' : `Show all ${a.card.played.length} heroes`}
          </Button>
        )}
      </section>

      {a.card.suggestions.length > 0 && (
        <section className="space-y-2">
          <h3 className="text-lg font-semibold">Heroes you might be good at</h3>
          <p className="text-sm text-muted-foreground">
            Heroes this player has not played, ranked by what players with similar records do on them. The number is
            the expected edge once the hero is learned; first games on a new hero usually run about 3 points worse.
          </p>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {a.card.suggestions.map(s => (
              <SuggestionCard key={s.hero} s={s} />
            ))}
          </div>
        </section>
      )}

      <section className="space-y-1 text-sm text-muted-foreground">
        <h3 className="text-base font-semibold text-foreground">How far to trust this</h3>
        <p>
          One game says little: a single win or loss is mostly noise. The ranges are 80% ranges. Tested on games the
          model never saw, the true value fell inside about {pct(main ? val.cover80.main : val.cover80.imp2)} of the
          time, so roughly 1 range in 5 misses.
        </p>
        <p>
          Combined with the draft, estimates like these for all ten players called the winner of{' '}
          {(100 * val.accuracy.v2.main).toFixed(1)}% of held-out games with scoreboard stats and{' '}
          {(100 * val.accuracy.v2.imp2).toFixed(1)}% with wins and losses only. Real gains, and a reminder that the
          draft and the players leave most of a game unexplained.
        </p>
        <p>
          These estimates come from our own replay data and are updated weekly. Private Heroes Profile accounts are
          never shown. Data provided by Heroes Profile.
        </p>
      </section>
    </div>
  )
}

export function PlayersClient() {
  const router = useRouter()
  const params = useSearchParams()
  const tagParam = params.get('tag') ?? ''
  const regionParam = Number(params.get('region'))
  const [q, setQ] = useState(tagParam)
  const [res, setRes] = useState<AccountResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!tagParam) return
    let cancelled = false
    setBusy(true)
    setError(null)
    fetch(`/api/personal/account?tag=${encodeURIComponent(tagParam)}`)
      .then(r => r.json())
      .then((body: AccountResponse) => {
        if (cancelled) return
        if (body.error) setError(body.error)
        else setRes(body)
      })
      .catch(() => !cancelled && setError('Lookup failed. Try again.'))
      .finally(() => !cancelled && setBusy(false))
    return () => {
      cancelled = true
    }
  }, [tagParam])

  const shown = res?.accounts.filter(a => !regionParam || a.region === regionParam) ?? []
  const one = shown.length === 1 ? shown[0] : null

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Player lookup</h1>
        <p className="text-muted-foreground">
          Per-hero skill estimates for accounts in our replay data. Enter a full BattleTag, like Name#1234.
        </p>
      </div>
      <form
        className="flex gap-2"
        onSubmit={e => {
          e.preventDefault()
          if (q.trim()) router.push(`/players?tag=${encodeURIComponent(q.trim())}`)
        }}
      >
        <Input value={q} onChange={e => setQ(e.target.value)} placeholder="Name#1234" maxLength={24} aria-label="BattleTag" />
        <Button type="submit" disabled={busy}>
          {busy ? 'Looking up...' : 'Look up'}
        </Button>
      </form>

      {error && <p className="text-sm text-gaming-danger">{error}</p>}

      {res && !busy && res.accounts.length === 0 && (
        <p className="text-sm text-muted-foreground">
          {tagParam} is not in our data. We have cards for accounts with at least 3 Storm League games in our replay
          data in the 180 days up to {res.through}. That covers most active players, but not everyone.
        </p>
      )}

      {res && !busy && shown.length > 1 && (
        <section className="space-y-2">
          <p className="text-sm text-muted-foreground">This BattleTag exists in more than one region. Pick one:</p>
          <div className="grid gap-3 sm:grid-cols-2">
            {shown.map(a => (
              <AccountChoice
                key={a.region}
                a={a}
                onPick={() => router.push(`/players?tag=${encodeURIComponent(tagParam)}&region=${a.region}`)}
              />
            ))}
          </div>
        </section>
      )}

      {res && one && <Profile a={one} through={res.through} val={res.validation} />}
    </div>
  )
}
