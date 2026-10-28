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
 * Display names for players, told apart when several share a visible name: region and games
 * in our data (what the confirmation cards already showed), never a #number the user did
 * not type. Ids stay the identity.
 */
export function displayLabels(players: PersonalPlayer[], alsoTaken: string[] = []): Record<string, string> {
  const count = new Map<string, number>()
  const bump = (l: string) => count.set(l.toLowerCase(), (count.get(l.toLowerCase()) ?? 0) + 1)
  alsoTaken.forEach(bump)
  players.forEach(p => bump(p.label))
  const out: Record<string, string> = {}
  for (const p of players) {
    const clash = (count.get(p.label.toLowerCase()) ?? 0) > 1
    const detail = [p.regionName, p.games !== undefined ? `${p.games} games` : undefined].filter(Boolean).join(', ')
    out[p.id] = clash && detail ? `${p.label} (${detail})` : p.label
  }
  return out
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
  const tags = side === 'ours' ? registeredTags : []
  const listed = new Set(tags.map(t => t.toLowerCase()))
  const shown = players.filter(p => !(side === 'ours' && p.tag && listed.has(p.tag.toLowerCase())))
  const tagNames = tags.map(bt => bt.split('#')[0])
  const labels = displayLabels(shown, tagNames)
  const clashWithParty = new Set(shown.map(p => p.label.toLowerCase()))
  const opts: AssignOption[] = tags.map((bt, i) => ({
    value: bt,
    label: clashWithParty.has(tagNames[i].toLowerCase()) ? `${tagNames[i]} (party)` : tagNames[i],
  }))
  for (const p of shown) opts.push({ value: p.id, label: labels[p.id] })
  return opts
}

const REGION_ABBR: Record<string, string> = { Americas: 'AM', Europe: 'EU', Asia: 'AS', China: 'CN' }

/**
 * Compact form of a display label for the closed (narrow) select: "Max (Americas, 668 games)"
 * becomes "Max (AM 668g)". The full label stays in the open list and the tooltip.
 */
export function compactLabel(label: string): string {
  return label.replace(/\((Americas|Europe|Asia|China), (\d+) games\)/, (_m, r: string, g: string) => `(${REGION_ABBR[r]} ${g}g)`)
}
