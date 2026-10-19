'use client'

import { useState } from 'react'
import { cn } from '@/lib/utils'
import type { AccountSummary, DraftTerms, MatchResult } from '@/lib/personal/accounts'
import type { DraftCoefs, PersonalPlayer } from '@/lib/personal/draft-personal'

/**
 * In-game mode (docs/personalization-site-design.md, "Draft page"): type the visible names of
 * the other players (no #number). Names are matched against our published name index using
 * the party's region and MMR; a clear winner is selected, otherwise 2 to 4 candidates are
 * shown. Matched players' skill feeds the suggestions; unmatched players count as average.
 */

export interface PersonalDraftState {
  ours: PersonalPlayer[]
  theirs: PersonalPlayer[]
  coefs: DraftCoefs
}

type Candidate = AccountSummary & { draft: DraftTerms }

interface MatchResponse {
  through: string
  context: { region: number | null; mmr: number | null }
  party: (AccountSummary & { tag: string; draft: DraftTerms })[]
  results: MatchResult[]
  coefs: DraftCoefs
}

const REGION_OPTIONS: [string, string][] = [
  ['', 'Party region'],
  ['1', 'Americas'],
  ['2', 'Europe'],
  ['3', 'Asia'],
  ['5', 'China'],
]

function toPlayer(label: string, d: DraftTerms): PersonalPlayer {
  return { label, s: d.s, offRole: d.offRole, b2: d.b2, b3: d.b3 }
}

function CandidateLine({ c, onPick, selected }: { c: Candidate; onPick?: () => void; selected?: boolean }) {
  return (
    <button
      type="button"
      onClick={onPick}
      className={cn(
        'w-full rounded border px-2 py-1 text-left text-[11px]',
        selected ? 'border-primary/60 bg-primary/10' : 'border-border hover:border-primary/40',
      )}
    >
      <span className="font-medium">
        {c.name}#{c.discriminator}
      </span>
      <span className="text-muted-foreground">
        {' '}
        · {c.regionName}
        {c.mmr !== null ? ` · MMR ${c.mmr}` : ''} · last {c.lastPlayed} · {c.games} games
        {c.topHeroes.length ? ` · ${c.topHeroes.join(', ')}` : ''}
      </span>
    </button>
  )
}

export function InGamePlayers({
  partyTags,
  onChange,
}: {
  partyTags: string[]
  onChange: (s: PersonalDraftState | null) => void
}) {
  const [allies, setAllies] = useState<string[]>(['', '', '', ''])
  const [enemies, setEnemies] = useState<string[]>(['', '', '', '', ''])
  const [region, setRegion] = useState('')
  const [res, setRes] = useState<MatchResponse | null>(null)
  const [picked, setPicked] = useState<Record<string, number | null>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const emit = (r: MatchResponse | null, pk: Record<string, number | null>) => {
    if (!r) return onChange(null)
    const chosen = (side: 'a' | 'e', names: string[]) =>
      names
        .map((n, i) => {
          const m = r.results.find(x => x.query === n.trim())
          const k = pk[`${side}${i}`]
          const c = k === undefined ? (m?.status === 'matched' ? m.candidates[0] : undefined) : k === null ? undefined : m?.candidates[k]
          return c ? toPlayer(c.name, c.draft) : null
        })
        .filter((x): x is PersonalPlayer => x !== null)
    const ours = [...r.party.map(p => toPlayer(p.name, p.draft)), ...chosen('a', allies)]
    const theirs = chosen('e', enemies)
    onChange(ours.length + theirs.length > 0 ? { ours, theirs, coefs: r.coefs } : null)
  }

  const run = async () => {
    setBusy(true)
    setError(null)
    try {
      const names = [...allies, ...enemies].map(s => s.trim()).filter(Boolean)
      const qs = new URLSearchParams({ names: names.join(','), party: partyTags.join(',') })
      if (region) qs.set('region', region)
      const r = await fetch(`/api/personal/match?${qs}`)
      const body = await r.json()
      if (!r.ok) throw new Error(body.error ?? 'match failed')
      setRes(body)
      setPicked({})
      emit(body, {})
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Matching failed')
    } finally {
      setBusy(false)
    }
  }

  const pick = (key: string, k: number | null) => {
    const pk = { ...picked, [key]: k }
    setPicked(pk)
    emit(res, pk)
  }

  const row = (side: 'a' | 'e', i: number, value: string, set: (v: string) => void) => {
    const key = `${side}${i}`
    const m = res?.results.find(x => x.query === value.trim())
    const k = picked[key]
    const sel = k === undefined ? (m?.status === 'matched' ? 0 : null) : k
    return (
      <div key={key} className="space-y-1">
        <input
          value={value}
          onChange={e => set(e.target.value)}
          placeholder={side === 'a' ? `Teammate ${i + 1}` : `Enemy ${i + 1}`}
          maxLength={12}
          className="h-7 w-full rounded-md border border-border bg-background px-2 text-xs"
        />
        {m && value.trim() && (
          <div className="space-y-1">
            {m.status === 'none' && <p className="text-[10px] text-muted-foreground">Not in our data: counted as an average player.</p>}
            {m.status === 'matched' && sel === 0 && (
              <>
                <CandidateLine c={m.candidates[0]} selected />
                <button type="button" className="text-[10px] text-muted-foreground underline" onClick={() => pick(key, null)}>
                  Not them
                </button>
              </>
            )}
            {(m.status === 'choose' || (m.status === 'matched' && sel !== 0)) && (
              <>
                <p className="text-[10px] text-muted-foreground">
                  {m.status === 'choose' ? 'More than one possible account. Pick one, or leave it as an average player:' : 'Counted as an average player.'}
                </p>
                {m.status === 'choose' &&
                  m.candidates.map((c, ci) => (
                    <CandidateLine key={`${c.region}:${c.discriminator}`} c={c} selected={sel === ci} onPick={() => pick(key, sel === ci ? null : ci)} />
                  ))}
                {m.status === 'matched' && (
                  <button type="button" className="text-[10px] text-muted-foreground underline" onClick={() => pick(key, 0)}>
                    Use {m.candidates[0].name}#{m.candidates[0].discriminator}
                  </button>
                )}
              </>
            )}
          </div>
        )}
      </div>
    )
  }

  const nMatched = res
    ? res.party.length +
      [...allies.map((n, i) => ['a', n, i] as const), ...enemies.map((n, i) => ['e', n, i] as const)].filter(([side, n, i]) => {
        const m = res.results.find(x => x.query === n.trim())
        const k = picked[`${side}${i}`]
        return k === undefined ? m?.status === 'matched' : k !== null
      }).length
    : 0

  return (
    <div className="space-y-2 rounded-sm border border-border p-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium">In-game players</h3>
        <select value={region} onChange={e => setRegion(e.target.value)} className="h-7 rounded-md border border-border bg-background px-1 text-xs">
          {REGION_OPTIONS.map(([v, l]) => (
            <option key={v} value={v}>
              {l}
            </option>
          ))}
        </select>
      </div>
      <p className="text-[10px] text-muted-foreground">
        Type the names you see in the loading screen, without the #number. We match them against our replay data using
        your party&apos;s region and MMR, and add each matched player&apos;s skill on each hero to the suggestions.
      </p>
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <p className="text-[10px] font-medium uppercase text-muted-foreground">Your team</p>
          {partyTags.map(t => (
            <p key={t} className="text-[11px]">
              {t.split('#')[0]} <span className="text-muted-foreground">(party)</span>
            </p>
          ))}
          {allies.slice(0, Math.max(0, 5 - partyTags.length)).map((v, i) =>
            row('a', i, v, x => setAllies(allies.map((y, j) => (j === i ? x : y)))),
          )}
        </div>
        <div className="space-y-2">
          <p className="text-[10px] font-medium uppercase text-muted-foreground">Enemy team</p>
          {enemies.map((v, i) => row('e', i, v, x => setEnemies(enemies.map((y, j) => (j === i ? x : y)))))}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={run}
          disabled={busy}
          className="h-7 rounded-md border border-primary/50 px-3 text-xs hover:bg-primary/10 disabled:opacity-50"
        >
          {busy ? 'Matching...' : 'Match players'}
        </button>
        {res && (
          <span className="text-[10px] text-muted-foreground">
            {nMatched} player{nMatched === 1 ? '' : 's'} with personal skill; data as of {res.through}
          </span>
        )}
        {res && (
          <button
            type="button"
            className="text-[10px] text-muted-foreground underline"
            onClick={() => {
              setRes(null)
              setPicked({})
              onChange(null)
            }}
          >
            Clear
          </button>
        )}
      </div>
      {error && <p className="text-[10px] text-gaming-danger">{error}</p>}
    </div>
  )
}
