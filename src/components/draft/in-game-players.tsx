'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { cn } from '@/lib/utils'
import type { MatchResult } from '@/lib/personal/accounts'
import type { DraftCoefs, PersonalPlayer } from '@/lib/personal/draft-personal'
import { resolveLobby, slotKey, type Candidate, type MatchSnapshot, type PartyAccount, type Side } from '@/lib/personal/lobby'

/**
 * In-game mode (docs/personalization-site-design.md, "Draft page"): type the visible names of
 * the players (no #number), or a full BattleTag for an exact lookup. Names are matched against
 * our published name index using the party's region and MMR; a clear winner is selected,
 * otherwise the candidates are shown. Matched players' skill feeds the suggestions; unmatched
 * players count as average. Edits stop a slot counting until it is matched again.
 */

export interface PersonalDraftState {
  ours: PersonalPlayer[]
  theirs: PersonalPlayer[]
  coefs: DraftCoefs
}

interface MatchResponse {
  through: string
  estimatesAsOf?: string
  context: { region: number | null; mmr: number | null }
  party: PartyAccount[]
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

const slotLabel = (key: string) => `${key[0] === 'a' ? 'Teammate' : 'Enemy'} ${Number(key.slice(1)) + 1}`

function CandidateLine({ c, onPick, selected }: { c: Candidate; onPick?: () => void; selected?: boolean }) {
  return (
    <button
      type="button"
      onClick={onPick}
      aria-pressed={selected ?? false}
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
  const nAllies = Math.max(0, 5 - partyTags.length)
  const [allies, setAllies] = useState<string[]>(() => Array(nAllies).fill(''))
  const [enemies, setEnemies] = useState<string[]>(['', '', '', '', ''])
  const [region, setRegion] = useState('')
  const [res, setRes] = useState<(MatchResponse & { snap: MatchSnapshot }) | null>(null)
  const [picked, setPicked] = useState<Record<string, number | null>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const reqId = useRef(0)

  const lobby = useMemo(
    () => resolveLobby(res?.snap ?? null, allies, enemies, region, picked, slotLabel),
    [res, allies, enemies, region, picked],
  )
  const { ours, theirs, count } = lobby
  const coefs = res?.coefs
  useEffect(() => {
    onChange(coefs && count > 0 ? { ours, theirs, coefs } : null)
  }, [ours, theirs, count, coefs, onChange])

  const stale = Object.values(lobby.slots).some(s => s.kind === 'pending')

  const run = async () => {
    const id = ++reqId.current
    setBusy(true)
    setError(null)
    const inputs: Record<string, string> = {}
    allies.forEach((v, i) => v.trim() && (inputs[slotKey('a', i)] = v.trim()))
    enemies.forEach((v, i) => v.trim() && (inputs[slotKey('e', i)] = v.trim()))
    try {
      const names = Array.from(new Set(Object.values(inputs)))
      const qs = new URLSearchParams({ names: names.join(','), party: partyTags.join(',') })
      if (region) qs.set('region', region)
      const r = await fetch(`/api/personal/match?${qs}`)
      const body = await r.json()
      if (!r.ok) throw new Error(body.error ?? 'Matching failed')
      if (id !== reqId.current) return
      const m = body as MatchResponse
      setRes({ ...m, snap: { results: m.results, party: m.party, coefs: m.coefs, region, inputs } })
      setPicked({})
    } catch (e) {
      if (id !== reqId.current) return
      // a failed rematch must not leave the previous lobby's players in place
      setRes(null)
      setPicked({})
      setError(e instanceof Error ? e.message : 'Matching failed')
    } finally {
      if (id === reqId.current) setBusy(false)
    }
  }

  const pick = (key: string, k: number | null) => setPicked(p => ({ ...p, [key]: k }))

  const row = (side: Side, i: number, value: string, set: (v: string) => void) => {
    const key = slotKey(side, i)
    const st = lobby.slots[key] ?? { kind: 'empty' }
    const label = slotLabel(key)
    const result = 'result' in st ? st.result : null
    const choose = result?.status === 'choose'
    const sel = st.kind === 'selected' || st.kind === 'duplicate' ? st.index : null
    return (
      <div key={key} className="space-y-1">
        <input
          value={value}
          onChange={e => set(e.target.value)}
          placeholder={label}
          aria-label={`${label}: name or BattleTag`}
          maxLength={19}
          className="h-7 w-full rounded-md border border-border bg-background px-2 text-xs"
        />
        {st.kind === 'pending' && res && (
          <p className="text-[10px] text-yellow-400">Changed: press Match players to use this player.</p>
        )}
        {st.kind === 'none' && <p className="text-[10px] text-muted-foreground">Not in our data: counted as an average player.</p>}
        {st.kind === 'duplicate' && (
          <p className="text-[10px] text-yellow-400">
            Same account as {st.usedBy}: counted once. Pick another account or leave this slot as an average player.
          </p>
        )}
        {!choose && st.kind === 'selected' && (
          <>
            <CandidateLine c={st.candidate} selected />
            <button type="button" className="text-[10px] text-muted-foreground underline" onClick={() => pick(key, null)}>
              Not them
            </button>
          </>
        )}
        {st.kind === 'unselected' && !choose && (
          <>
            <p className="text-[10px] text-muted-foreground">Counted as an average player.</p>
            <button type="button" className="text-[10px] text-muted-foreground underline" onClick={() => pick(key, 0)}>
              Use {st.result.candidates[0].name}#{st.result.candidates[0].discriminator}
            </button>
          </>
        )}
        {choose && result && (
          <div role="group" aria-label={`${label}: possible accounts`} className="space-y-1">
            <p className="text-[10px] text-muted-foreground">
              {result.candidates.length === 1
                ? 'Possible account. Confirm it is this player, or leave them as an average player:'
                : 'More than one possible account. Pick one, or leave it as an average player:'}
            </p>
            {result.candidates.map((c, ci) => (
              <CandidateLine key={`${c.region}:${c.discriminator}`} c={c} selected={sel === ci} onPick={() => pick(key, sel === ci ? null : ci)} />
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-2 rounded-sm border border-border p-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium">In-game players</h3>
        <select
          value={region}
          onChange={e => setRegion(e.target.value)}
          aria-label="Region for matching"
          className="h-7 rounded-md border border-border bg-background px-1 text-xs"
        >
          {REGION_OPTIONS.map(([v, l]) => (
            <option key={v} value={v}>
              {l}
            </option>
          ))}
        </select>
      </div>
      <p className="text-[10px] text-muted-foreground">
        Type the names you see in the loading screen, without the #number, or a full BattleTag like Name#1234 for
        yourself and anyone you know. We match them against our replay data using your party&apos;s region and MMR, and
        add each matched player&apos;s skill on each hero to the suggestions.
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="space-y-2">
          <p className="text-[10px] font-medium uppercase text-muted-foreground">Your team</p>
          {partyTags.map(t => (
            <p key={t} className="text-[11px]">
              {t.split('#')[0]} <span className="text-muted-foreground">(party)</span>
            </p>
          ))}
          {allies.slice(0, nAllies).map((v, i) => row('a', i, v, x => setAllies(a => a.map((y, j) => (j === i ? x : y)))))}
        </div>
        <div className="space-y-2">
          <p className="text-[10px] font-medium uppercase text-muted-foreground">Enemy team</p>
          {enemies.map((v, i) => row('e', i, v, x => setEnemies(a => a.map((y, j) => (j === i ? x : y)))))}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={run}
          disabled={busy || (partyTags.length === 0 && ![...allies, ...enemies].some(v => v.trim()))}
          className="h-7 rounded-md border border-primary/50 px-3 text-xs hover:bg-primary/10 disabled:opacity-50"
        >
          {busy ? 'Matching...' : 'Match players'}
        </button>
        <span role="status" aria-live="polite" className="text-[10px] text-muted-foreground">
          {res &&
            `${count} player${count === 1 ? '' : 's'} with personal skill${stale ? ' (some changes not matched yet)' : ''}. Records through ${res.through}${res.estimatesAsOf ? `; skill estimates use games through ${res.estimatesAsOf}` : ''}.`}
        </span>
        {res && (
          <button
            type="button"
            className="text-[10px] text-muted-foreground underline"
            onClick={() => {
              reqId.current++
              setRes(null)
              setPicked({})
            }}
          >
            Clear
          </button>
        )}
      </div>
      {error && (
        <p role="alert" className="text-[10px] text-gaming-danger">
          {error}
        </p>
      )}
    </div>
  )
}
