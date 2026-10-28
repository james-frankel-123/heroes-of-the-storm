/**
 * Who played which hero (the draft board's "Who?" selects). Pure; tested in
 * src/lib/draft/__tests__/assignments.test.ts.
 *
 * Values are a registered BattleTag (Name#1234) or a matched in-game player id
 * (lobby.ts accountKey, "region:name#1234"). A player plays one hero, so assigning them
 * to a pick moves them off any other pick.
 */
import type { PersonalPlayer } from '@/lib/personal/draft-personal'

export function assignPlayer(
  assignments: Record<number, string>,
  stepIndex: number,
  value: string,
): Record<number, string> {
  const next: Record<number, string> = {}
  const key = value.toLowerCase()
  for (const [k, v] of Object.entries(assignments)) {
    if (Number(k) !== stepIndex && v.toLowerCase() !== key) next[Number(k)] = v
  }
  if (value) next[stepIndex] = value
  return next
}

/** The player an assignment value names: by id, or a party member by BattleTag. */
export function playerFor(value: string, players: PersonalPlayer[]): PersonalPlayer | undefined {
  const v = value.toLowerCase()
  return players.find(p => p.id === value || p.tag?.toLowerCase() === v)
}

/** Hero -> player id pins for one team, from that team's picks and their step indices. */
export function pinsFor(
  picks: { hero: string; stepIndex: number }[],
  assignments: Record<number, string>,
  players: PersonalPlayer[],
): Record<string, string> {
  const fixed: Record<string, string> = {}
  for (const { hero, stepIndex } of picks) {
    const v = assignments[stepIndex]
    const p = v ? playerFor(v, players) : undefined
    if (p) fixed[hero] = p.id
  }
  return fixed
}

export interface AssignOption {
  value: string
  label: string
}

/**
 * Options for one team's selects: every assignable player of that team, whether or not
 * they already have a hero (picking them again moves them). Ours: registered party tags,
 * then matched teammates not already listed by tag. Theirs: matched enemies.
 */
export function assignOptionsFor(
  side: 'ours' | 'theirs',
  registeredTags: string[],
  players: PersonalPlayer[],
): AssignOption[] {
  const opts: AssignOption[] = side === 'ours' ? registeredTags.map(bt => ({ value: bt, label: bt.split('#')[0] })) : []
  const listed = new Set(registeredTags.map(t => t.toLowerCase()))
  for (const p of players) {
    if (side === 'ours' && p.tag && listed.has(p.tag.toLowerCase())) continue
    opts.push({ value: p.id, label: p.label })
  }
  return opts
}
