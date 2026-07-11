'use client'

/**
 * Expert draft-rating study client (v4: fixed 240-item paid design).
 *
 * Per item: map + skill tier prominently at top, two blinded 5-hero teams
 * side by side, then three questions:
 *   (a) slider P(Team A wins) 0-100%
 *   (b) forced choice: which team drafted better (A/B)
 *   (c) confidence 1-5
 * Auto-advances once all three are answered.
 *
 * A consent-and-data-handling notice is shown before a rater's first item
 * (skipped on resume — a rater with saved ratings has already consented).
 *
 * Keyboard: arrows = slider, A/B = choice, 1-5 = confidence, Enter = next.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { cn } from '@/lib/utils'
import { heroImageSrc } from '@/lib/data/hero-images'
import { mapImageSrc } from '@/lib/data/map-images'
import { getHeroRole } from '@/lib/data/hero-roles'
import { RoleIcon } from '@/components/shared/role-icon'

interface RatingItem {
  id: number
  map: string
  tier: string
  teamA: string[]
  teamB: string[]
}

interface ItemsResponse {
  rater: string
  slot: number
  isTest: boolean
  items: RatingItem[]
  ratedItemIds: number[]
  ratedCount: number
}

const RATER_STORAGE_KEY = 'draft-rating-rater'
const SLOT_STORAGE_KEY = 'draft-rating-slot'
const CONSENT_STORAGE_KEY = 'draft-rating-consented'

const TIER_META: Record<string, { label: string; ranks: string; className: string }> = {
  low: {
    label: 'LOW TIER',
    ranks: 'Bronze – Silver',
    className: 'bg-emerald-500/20 text-emerald-300 border-emerald-400/50',
  },
  mid: {
    label: 'MID TIER',
    ranks: 'Gold – Platinum',
    className: 'bg-amber-500/20 text-amber-300 border-amber-400/50',
  },
  high: {
    label: 'HIGH TIER',
    ranks: 'Diamond – Master',
    className: 'bg-fuchsia-500/20 text-fuchsia-300 border-fuchsia-400/50',
  },
}

const CONFIDENCE_LABELS = ['Guess', 'Low', 'Medium', 'High', 'Certain']

export function RateClient() {
  const searchParams = useSearchParams()

  const [rater, setRater] = useState<string | null>(null)
  const [slot, setSlot] = useState<number | null>(null)
  const [nameInput, setNameInput] = useState('')
  const [initialized, setInitialized] = useState(false)
  const [consented, setConsented] = useState(false)

  // One fixed sequence of 240 items (7 for test raters). Resume works
  // anywhere: the server returns already-rated ids and they are filtered out.
  const [queue, setQueue] = useState<RatingItem[]>([])
  const [totalItems, setTotalItems] = useState(0)
  const [ratedBase, setRatedBase] = useState(0) // rated before this session
  const [done, setDone] = useState(0) // submitted this session
  const [idx, setIdx] = useState(0)
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  // Per-item answers
  const [p, setP] = useState(50)
  const [pTouched, setPTouched] = useState(false)
  const [choice, setChoice] = useState<'A' | 'B' | null>(null)
  const [confidence, setConfidence] = useState<number | null>(null)
  const shownAtRef = useRef<number>(Date.now())

  // Resolve rater from ?rater= or localStorage
  useEffect(() => {
    const qpRater = searchParams.get('rater')?.trim()
    const qpSlot = searchParams.get('slot')
    let nextRater: string | null = null
    let nextSlot: number | null = null
    if (qpRater) {
      nextRater = qpRater
      localStorage.setItem(RATER_STORAGE_KEY, qpRater)
      if (qpSlot !== null && /^(0|1[0-3]?|[2-9])$/.test(qpSlot)) {
        nextSlot = Number(qpSlot)
        localStorage.setItem(SLOT_STORAGE_KEY, qpSlot)
      } else {
        localStorage.removeItem(SLOT_STORAGE_KEY)
      }
    } else {
      const stored = localStorage.getItem(RATER_STORAGE_KEY)
      if (stored) {
        nextRater = stored
        const storedSlot = localStorage.getItem(SLOT_STORAGE_KEY)
        if (storedSlot !== null && /^\d+$/.test(storedSlot)) nextSlot = Number(storedSlot)
      }
    }
    setRater(nextRater)
    setSlot(nextSlot)
    if (nextRater && localStorage.getItem(CONSENT_STORAGE_KEY) === nextRater.toLowerCase()) {
      setConsented(true)
    }
    setInitialized(true)
  }, [searchParams])

  // Load assignment
  useEffect(() => {
    if (!rater) return
    let cancelled = false
    setLoading(true)
    setLoadError(null)
    const params = new URLSearchParams({ rater })
    if (slot !== null) params.set('slot', String(slot))
    fetch(`/api/ratings/items?${params}`)
      .then(async (res) => {
        if (!res.ok) throw new Error((await res.json()).error ?? `HTTP ${res.status}`)
        return res.json() as Promise<ItemsResponse>
      })
      .then((data) => {
        if (cancelled) return
        const ratedSet = new Set(data.ratedItemIds)
        const remaining = data.items.filter((it) => !ratedSet.has(it.id))
        setQueue(remaining)
        setTotalItems(data.items.length)
        setRatedBase(data.ratedCount ?? data.items.length - remaining.length)
        setDone(0)
        setIdx(0)
        // A rater with saved ratings has already consented (cross-device resume).
        if ((data.ratedCount ?? 0) > 0) setConsented(true)
        shownAtRef.current = Date.now()
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [rater, slot])

  const current: RatingItem | undefined = queue[idx]
  const complete = pTouched && choice !== null && confidence !== null
  const progressDone = Math.min(totalItems, ratedBase + done)

  const resetAnswers = useCallback(() => {
    setP(50)
    setPTouched(false)
    setChoice(null)
    setConfidence(null)
    setSubmitError(null)
    shownAtRef.current = Date.now()
  }, [])

  const submit = useCallback(async () => {
    if (!current || !rater || !complete || submitting) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const res = await fetch('/api/ratings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rater,
          slot,
          itemId: current.id,
          pTeamA: p,
          betterTeam: choice,
          confidence,
          msTaken: Date.now() - shownAtRef.current,
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.error ?? `HTTP ${res.status}`)
      }
      setDone((n) => n + 1)
      setIdx((i) => i + 1)
      resetAnswers()
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }, [current, rater, slot, complete, submitting, p, choice, confidence, resetAnswers])

  // Auto-advance shortly after all three questions are answered
  useEffect(() => {
    if (!complete || submitting || !current) return
    const t = setTimeout(() => submit(), 500)
    return () => clearTimeout(t)
  }, [complete, submitting, current, submit])

  // Keyboard shortcuts
  useEffect(() => {
    if (!current || !consented) return
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
        if (!(target as HTMLInputElement).type || (target as HTMLInputElement).type !== 'range') return
      }
      // The slider is displayed with Team A on the LEFT (matching the team
      // panels), so moving the handle left must INCREASE P(Team A wins).
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
        e.preventDefault()
        setP((v) => Math.min(100, v + (e.shiftKey ? 5 : 1)))
        setPTouched(true)
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
        e.preventDefault()
        setP((v) => Math.max(0, v - (e.shiftKey ? 5 : 1)))
        setPTouched(true)
      } else if (e.key === 'a' || e.key === 'A') {
        setChoice('A')
      } else if (e.key === 'b' || e.key === 'B') {
        setChoice('B')
      } else if (/^[1-5]$/.test(e.key)) {
        setConfidence(Number(e.key))
      } else if (e.key === 'Enter') {
        e.preventDefault()
        submit()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [current, consented, submit])

  // ---------------------------------------------------------------- screens

  if (!initialized) return null

  if (!rater) {
    return (
      <div className="mx-auto max-w-md pt-16">
        <div className="rounded-xl border bg-card p-6 shadow-lg">
          <h1 className="text-2xl font-bold">Draft Rating Study</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Enter the rater name from your invite (or open the personalized link you were sent).
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault()
              const name = nameInput.trim()
              if (!name) return
              localStorage.setItem(RATER_STORAGE_KEY, name)
              setRater(name)
            }}
            className="mt-4 flex gap-2"
          >
            <input
              autoFocus
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              placeholder="Your rater name"
              maxLength={100}
              className="h-11 flex-1 rounded-md border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-primary"
            />
            <button
              type="submit"
              disabled={!nameInput.trim()}
              className="h-11 rounded-md bg-primary px-5 text-sm font-semibold text-primary-foreground disabled:opacity-50"
            >
              Start
            </button>
          </form>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="pt-24 text-center text-muted-foreground" data-testid="rate-loading">
        Loading your assigned drafts…
      </div>
    )
  }

  if (loadError) {
    return (
      <div className="mx-auto max-w-md pt-16 text-center">
        <p className="text-destructive">{loadError}</p>
        <button
          onClick={() => setRater((r) => (r ? r + '' : r))}
          className="mt-4 rounded-md border px-4 py-2 text-sm"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!consented && current) {
    return (
      <div className="mx-auto max-w-2xl pt-12" data-testid="rate-consent">
        <div className="rounded-xl border bg-card p-6 shadow-lg sm:p-8">
          <h1 className="text-2xl font-bold">Before you begin</h1>
          <div className="mt-4 space-y-3 text-sm leading-relaxed text-muted-foreground">
            <p>
              This is a <span className="font-medium text-foreground">research study</span> on
              how experts judge Heroes of the Storm drafts. You will rate{' '}
              <span className="font-medium text-foreground">{totalItems} pairs of drafts</span>{' '}
              (most raters take 1.5–2.5 hours in total). You can stop and resume anytime, on any
              device — your progress saves after every item.
            </p>
            <p>
              <span className="font-medium text-foreground">What we collect:</span> your ratings,
              response times, and the rater name on your invite. Results will be reported in
              academic publications in aggregate or pseudonymous form; your name will never be
              published without your permission.
            </p>
            <p>
              <span className="font-medium text-foreground">Compensation:</span> $100 via Venmo on
              completion of all {totalItems} items. Partial completion is not compensated.
            </p>
            <p>
              <span className="font-medium text-foreground">Voluntary:</span> you may stop at any
              time. Questions or concerns: max@segan.com.
            </p>
            <div className="rounded-lg border bg-background/60 p-4" data-testid="rate-instructions">
              <p className="font-semibold text-foreground">How to rate</p>
              <p className="mt-2">
                Each item shows two five-hero drafts (Team A and Team B) for a stated map and
                skill tier (Low = Bronze–Silver, Mid = Gold–Platinum, High = Diamond–Master).
                Judge the whole picture: the composition, the specific map, and what actually
                works at that skill level.
              </p>
              <ol className="mt-2 list-decimal space-y-1 pl-5">
                <li>
                  <span className="font-medium text-foreground">Win probability</span> — drag the
                  slider to the chance each team wins the game, judging only from the drafts, map,
                  and tier (assume otherwise equal players).
                </li>
                <li>
                  <span className="font-medium text-foreground">Better draft</span> — pick the team
                  you think drafted better, even when it&apos;s close. This will usually match your
                  slider, but answer it in its own right; the slider can sit at exactly 50/50, the
                  pick cannot.
                </li>
                <li>
                  <span className="font-medium text-foreground">Confidence</span> — how sure you
                  are (1 = guess, 5 = certain).
                </li>
              </ol>
              <p className="mt-2">
                The page advances automatically once all three are answered. There is no time
                limit — most pairs take 15–60 seconds, and some are genuinely hard; use your
                judgment and don&apos;t overthink.
              </p>
            </div>
          </div>
          <button
            onClick={() => {
              localStorage.setItem(CONSENT_STORAGE_KEY, rater.toLowerCase())
              setConsented(true)
              shownAtRef.current = Date.now()
            }}
            data-testid="rate-consent-agree"
            className="mt-6 h-12 w-full rounded-lg bg-primary text-sm font-bold text-primary-foreground shadow"
          >
            I agree — start rating
          </button>
        </div>
      </div>
    )
  }

  if (!current) {
    return (
      <div className="mx-auto max-w-lg pt-16 text-center" data-testid="rate-complete">
        <div className="rounded-xl border bg-card p-8 shadow-lg">
          <div className="text-5xl">🏆</div>
          <h1 className="mt-4 text-2xl font-bold">All done — thank you!</h1>
          <p className="mt-3 text-muted-foreground">
            You rated all {progressDone} assigned drafts. Your judgments are a huge help in
            validating our draft models against real expert intuition.
          </p>
          <p className="mt-2 text-sm text-muted-foreground">
            We&apos;ll be in touch about your compensation shortly. You can close this page now,{' '}
            <span className="font-medium">{rater}</span>.
          </p>
        </div>
      </div>
    )
  }

  const tierMeta = TIER_META[current.tier] ?? TIER_META.mid
  const mapImg = mapImageSrc(current.map)

  return (
    <div
      className="mx-auto max-w-4xl pb-16"
      data-testid="rate-item"
      data-item-id={current.id}
    >
      {/* Progress */}
      <div className="mb-4">
        <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
          <span>
            Rater: <span className="font-medium text-foreground">{rater}</span>
          </span>
          <span data-testid="rate-progress">
            {progressDone + 1} / {totalItems}
          </span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-all"
            style={{ width: `${(progressDone / Math.max(1, totalItems)) * 100}%` }}
          />
        </div>
      </div>

      {/* Map + tier banner — deliberately large */}
      <div className="relative mb-4 overflow-hidden rounded-xl border">
        {mapImg && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={mapImg}
            alt={current.map}
            className="absolute inset-0 h-full w-full object-cover opacity-40"
          />
        )}
        <div className="absolute inset-0 bg-gradient-to-r from-background/80 via-background/40 to-background/80" />
        <div className="relative flex flex-col items-center gap-2 px-4 py-5 sm:flex-row sm:justify-between sm:py-6">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
              Map
            </div>
            <div className="text-2xl font-extrabold leading-tight sm:text-3xl" data-testid="rate-map">
              {current.map}
            </div>
          </div>
          <div
            className={cn(
              'rounded-lg border px-4 py-2 text-center font-bold shadow-sm',
              tierMeta.className
            )}
            data-testid="rate-tier"
          >
            <div className="text-base leading-tight sm:text-lg">{tierMeta.label}</div>
            <div className="text-[11px] font-medium opacity-90">{tierMeta.ranks}</div>
          </div>
        </div>
      </div>

      {/* Teams */}
      <div className="mb-5 grid grid-cols-2 gap-3 sm:gap-4">
        <TeamPanel side="A" heroes={current.teamA} selected={choice === 'A'} />
        <TeamPanel side="B" heroes={current.teamB} selected={choice === 'B'} />
      </div>

      {/* Q1: win probability slider.
          Team A is displayed on the LEFT, so the slider is rendered with the
          left end = 100% Team A: display position = 100 - p. Dragging toward
          a team increases that team's win probability. Stored semantics
          (p = P(Team A wins)) are unchanged. */}
      <div className="mb-4 rounded-xl border bg-card p-4 sm:p-5">
        <div className="mb-1 text-sm font-semibold">
          1. How likely is each team to win this game?
        </div>
        <div className="mb-3 text-xs text-muted-foreground">
          Both teams at the skill level shown above, on this map. Drag toward the stronger team.
        </div>
        <div className="flex items-center gap-3 sm:gap-4">
          <span className="w-14 shrink-0 text-right text-xs font-medium text-sky-400">
            Team A wins
          </span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={100 - p}
            onChange={(e) => {
              setP(100 - Number(e.target.value))
              setPTouched(true)
            }}
            onPointerUp={() => setPTouched(true)}
            aria-label="Win probability (left = Team A wins, right = Team B wins)"
            data-testid="rate-slider"
            className="h-2 flex-1 cursor-pointer accent-sky-400"
          />
          <span className="w-14 shrink-0 text-xs font-medium text-rose-400">Team B wins</span>
        </div>
        <div className="mt-2 text-center">
          <span
            className={cn(
              'inline-block rounded-md px-3 py-1 text-lg font-bold tabular-nums',
              pTouched
                ? p >= 50
                  ? 'bg-sky-500/15 text-sky-300'
                  : 'bg-rose-500/15 text-rose-300'
                : 'bg-muted text-muted-foreground'
            )}
            data-testid="rate-p-label"
          >
            {pTouched ? (
              <>
                <span className="text-sky-300">A {p}%</span>{' '}
                <span className="opacity-60">—</span>{' '}
                <span className="text-rose-300">{100 - p}% B</span>
              </>
            ) : (
              'Move the slider to answer'
            )}
          </span>
        </div>
      </div>

      {/* Q2: forced choice */}
      <div className="mb-4 rounded-xl border bg-card p-4 sm:p-5">
        <div className="mb-3 text-sm font-semibold">
          2. Which team <span className="underline decoration-dotted">drafted better</span>?
          <span className="ml-2 font-normal text-muted-foreground">(pick one, even if close)</span>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => setChoice('A')}
            data-testid="rate-choice-a"
            className={cn(
              'h-14 rounded-lg border-2 text-lg font-bold transition-colors',
              choice === 'A'
                ? 'border-sky-400 bg-sky-500/20 text-sky-200'
                : 'border-border bg-background hover:border-sky-400/50'
            )}
          >
            Team A
          </button>
          <button
            onClick={() => setChoice('B')}
            data-testid="rate-choice-b"
            className={cn(
              'h-14 rounded-lg border-2 text-lg font-bold transition-colors',
              choice === 'B'
                ? 'border-rose-400 bg-rose-500/20 text-rose-200'
                : 'border-border bg-background hover:border-rose-400/50'
            )}
          >
            Team B
          </button>
        </div>
      </div>

      {/* Q3: confidence */}
      <div className="mb-4 rounded-xl border bg-card p-4 sm:p-5">
        <div className="mb-3 text-sm font-semibold">3. How confident are you?</div>
        <div className="grid grid-cols-5 gap-2">
          {[1, 2, 3, 4, 5].map((c) => (
            <button
              key={c}
              onClick={() => setConfidence(c)}
              data-testid={`rate-conf-${c}`}
              className={cn(
                'flex h-14 flex-col items-center justify-center rounded-lg border-2 transition-colors',
                confidence === c
                  ? 'border-primary bg-primary/15 text-foreground'
                  : 'border-border bg-background hover:border-primary/50'
              )}
            >
              <span className="text-lg font-bold leading-none">{c}</span>
              <span className="mt-1 text-[10px] text-muted-foreground">
                {CONFIDENCE_LABELS[c - 1]}
              </span>
            </button>
          ))}
        </div>
      </div>

      {submitError && (
        <div className="mb-3 rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          Failed to save: {submitError} — press Enter or the button to retry.
        </div>
      )}

      <div className="flex items-center justify-between">
        <div className="hidden text-xs text-muted-foreground sm:block">
          Shortcuts: ← → slider (Shift = ±5) · A / B choice · 1-5 confidence · Enter next
        </div>
        <button
          onClick={() => submit()}
          disabled={!complete || submitting}
          data-testid="rate-next"
          className="h-12 rounded-lg bg-primary px-8 text-sm font-bold text-primary-foreground shadow disabled:opacity-40"
        >
          {submitting ? 'Saving…' : complete ? 'Next →' : 'Answer all 3 to continue'}
        </button>
      </div>
    </div>
  )
}

function TeamPanel({
  side,
  heroes,
  selected,
}: {
  side: 'A' | 'B'
  heroes: string[]
  selected: boolean
}) {
  const isA = side === 'A'
  return (
    <div
      className={cn(
        'rounded-xl border-2 bg-card p-2.5 sm:p-4 transition-colors',
        isA ? 'border-sky-500/40' : 'border-rose-500/40',
        selected && (isA ? 'border-sky-400 bg-sky-500/5' : 'border-rose-400 bg-rose-500/5')
      )}
      data-testid={`team-${side.toLowerCase()}`}
    >
      <div
        className={cn(
          'mb-2 text-center text-sm font-extrabold uppercase tracking-widest sm:text-base',
          isA ? 'text-sky-400' : 'text-rose-400'
        )}
      >
        Team {side}
      </div>
      <ul className="space-y-1.5 sm:space-y-2">
        {heroes.map((hero) => {
          const role = getHeroRole(hero)
          return (
            <li key={hero} className="flex items-center gap-2 sm:gap-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={heroImageSrc(hero)}
                alt={hero}
                width={40}
                height={40}
                className={cn(
                  'h-8 w-8 shrink-0 rounded-full border object-cover sm:h-10 sm:w-10',
                  isA ? 'border-sky-500/40' : 'border-rose-500/40'
                )}
              />
              <div className="min-w-0">
                <div className="truncate text-xs font-semibold sm:text-sm">{hero}</div>
                {role && (
                  <div className="flex items-center gap-1 text-[10px] text-muted-foreground sm:text-[11px]">
                    <RoleIcon role={role} size={10} />
                    <span className="truncate">{role}</span>
                  </div>
                )}
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
