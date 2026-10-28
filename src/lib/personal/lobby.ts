/**
 * In-game lobby state for the draft page (client safe, pure).
 *
 * A match response is a snapshot of the inputs it answered. Each slot only contributes while
 * its text (and the region) still equal what was matched; an account counts once per lobby,
 * across the party and both teams. The players handed to the scorer and the "N matched" count
 * come from the same list.
 */
import type { AccountSummary, DraftTerms, MatchResult } from './accounts'
import type { DraftCoefs, PersonalPlayer } from './draft-personal'

export type Side = 'a' | 'e'

export type PartyAccount = AccountSummary & { tag: string; draft: DraftTerms }
export type Candidate = AccountSummary & { draft: DraftTerms }

export interface MatchSnapshot {
  results: MatchResult[]
  party: PartyAccount[]
  coefs: DraftCoefs
  /** Region select value at match time ('' = party region) */
  region: string
  /** Slot key ('a0', 'e3', ...) -> trimmed text at match time */
  inputs: Record<string, string>
}

export type SlotStatus =
  | { kind: 'empty' }
  /** Typed or edited since the last match (or the region changed): not counted until matched again */
  | { kind: 'pending' }
  | { kind: 'none' }
  /** Candidates exist but none is selected: counted as an average player */
  | { kind: 'unselected'; result: MatchResult }
  | { kind: 'selected'; result: MatchResult; index: number; candidate: Candidate }
  /** The selected account is already used by the party or an earlier slot */
  | { kind: 'duplicate'; result: MatchResult; index: number; candidate: Candidate; usedBy: string }

export interface Lobby {
  slots: Record<string, SlotStatus>
  ours: PersonalPlayer[]
  theirs: PersonalPlayer[]
  count: number
}

/** Stable identity of a published account: region + BattleTag (names are not unique). */
export function accountKey(a: { region: number; name: string; discriminator: number }): string {
  return `${a.region}:${a.name.normalize('NFC').toLowerCase()}#${a.discriminator}`
}

export function toPlayer(a: AccountSummary & { draft: DraftTerms; tag?: string }): PersonalPlayer {
  const d = a.draft
  return { id: accountKey(a), label: a.name, tag: a.tag, s: d.s, offRole: d.offRole, b2: d.b2, b3: d.b3 }
}

export const slotKey = (side: Side, i: number) => `${side}${i}`

/**
 * Resolve every slot against the snapshot. `picked[key]`: undefined = default (the clear match,
 * if any), null = "average player", k = candidate k. `labels[key]` names a slot for messages.
 */
export function resolveLobby(
  snap: MatchSnapshot | null,
  allies: string[],
  enemies: string[],
  region: string,
  picked: Record<string, number | null>,
  labels: (key: string) => string = k => k,
): Lobby {
  const slots: Record<string, SlotStatus> = {}
  const ours: PersonalPlayer[] = []
  const theirs: PersonalPlayer[] = []
  const usedBy = new Map<string, string>()
  if (snap) {
    for (const p of snap.party) {
      const pl = toPlayer(p)
      if (usedBy.has(pl.id)) continue
      usedBy.set(pl.id, 'your party')
      ours.push(pl)
    }
  }
  const sides: [Side, string[], PersonalPlayer[]][] = [
    ['a', allies, ours],
    ['e', enemies, theirs],
  ]
  for (const [side, names, out] of sides) {
    names.forEach((raw, i) => {
      const key = slotKey(side, i)
      const text = raw.trim()
      if (!text) return void (slots[key] = { kind: 'empty' })
      if (!snap || snap.region !== region || snap.inputs[key] !== text) return void (slots[key] = { kind: 'pending' })
      const result = snap.results.find(r => r.query.trim() === text)
      if (!result || result.status === 'none' || result.candidates.length === 0) return void (slots[key] = { kind: 'none' })
      const k = picked[key]
      const index = k === undefined ? (result.status === 'matched' ? 0 : null) : k
      if (index === null || index < 0 || index >= result.candidates.length) {
        return void (slots[key] = { kind: 'unselected', result })
      }
      const candidate = result.candidates[index]
      const id = accountKey(candidate)
      const prior = usedBy.get(id)
      if (prior !== undefined) return void (slots[key] = { kind: 'duplicate', result, index, candidate, usedBy: prior })
      usedBy.set(id, labels(key))
      out.push(toPlayer(candidate))
      slots[key] = { kind: 'selected', result, index, candidate }
    })
  }
  return { slots, ours, theirs, count: ours.length + theirs.length }
}
