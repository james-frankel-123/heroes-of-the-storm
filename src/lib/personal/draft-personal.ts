/**
 * Personal terms in the draft (P3_DRAFTER.md, docs/personalization-site-design.md "Draft page").
 *
 *   V = sigmoid(b0 + b1 logit WP + sum_ours b2_p s(p, h) - sum_theirs b2_p s(p, h)
 *               + b3 (O_ours - O_theirs))
 *
 * WP is the page's own search value. Which player takes which hero is not known, so each
 * team's term is its best assignment of its picked heroes to its players (unmatched players
 * add 0: population values). b0, b1 are the main model's V1 combiner; b2 and b3 come with
 * each player's model (refit with the off-role count, p3_site_offrole.py).
 * Pure functions; client safe (no model artifact import).
 */

export interface DraftCoefs {
  b0: number
  b1: number
}

export interface PersonalPlayer {
  /** Stable account key (region + BattleTag, see lobby.ts accountKey); identity in assignments */
  id: string
  /** Display label, e.g. the visible name (not unique: two accounts can share a name) */
  label: string
  /** Full BattleTag when known (party members), to match registered-tag assignments */
  tag?: string
  /** Hero name -> s(p, h) (win-probability units) */
  s: Record<string, number>
  /** Hero name -> off-role flag */
  offRole: Record<string, boolean>
  /** Weight on s (b2 of the player's model) */
  b2: number
  /** Off-role weight (b3 of the player's model, refit by p3_site_offrole.py) */
  b3: number
}

export interface Assignment {
  value: number
  /** hero -> player id (only heroes given to matched players) */
  who: Record<string, string>
}

function playerValue(p: PersonalPlayer, hero: string): number {
  return p.b2 * (p.s[hero] ?? 0) + (p.offRole[hero] ? p.b3 : 0)
}

/**
 * Best total personal logit for a team: heroes to distinct players, team size 5, players
 * beyond the matched ones are unknown (0). `fixed` pins a hero to a player id.
 */
export function bestAssignment(
  heroes: string[],
  players: PersonalPlayer[],
  c: DraftCoefs,
  fixed: Record<string, string> = {},
): Assignment {
  const slots = Math.max(5, heroes.length)
  const ps = players.slice(0, slots)
  let best: Assignment = { value: 0, who: {} }
  let bestSet = false
  const used = new Array<boolean>(slots).fill(false)
  const cur: number[] = []
  const rec = (i: number, acc: number) => {
    if (i === heroes.length) {
      if (!bestSet || acc > best.value) {
        const who: Record<string, string> = {}
        cur.forEach((slot, k) => {
          if (slot < ps.length) who[heroes[k]] = ps[slot].id
        })
        best = { value: acc, who }
        bestSet = true
      }
      return
    }
    const pin = fixed[heroes[i]]
    for (let slot = 0; slot < slots; slot++) {
      if (used[slot]) continue
      const p = slot < ps.length ? ps[slot] : null
      if (pin !== undefined && p?.id !== pin) continue
      used[slot] = true
      cur.push(slot)
      rec(i + 1, acc + (p ? playerValue(p, heroes[i]) : 0))
      cur.pop()
      used[slot] = false
    }
  }
  rec(0, 0)
  if (!bestSet) return bestAssignment(heroes, players, c) // a pin that cannot be honoured is dropped
  return best
}

const logit = (p: number) => Math.log(p / (1 - p))
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

/** Win chance (0-1) with personal terms on top of a population value q (0-1). */
export function personalValue(
  q: number,
  ours: string[],
  theirs: string[],
  oursPlayers: PersonalPlayer[],
  theirPlayers: PersonalPlayer[],
  c: DraftCoefs,
  fixed: Record<string, string> = {},
  fixedTheirs: Record<string, string> = {},
): { value: number; ours: Assignment; theirs: Assignment } {
  const qq = Math.min(Math.max(q, 1e-4), 1 - 1e-4)
  const a = bestAssignment(ours, oursPlayers, c, fixed)
  const b = bestAssignment(theirs, theirPlayers, c, fixedTheirs)
  return { value: sigmoid(c.b0 + c.b1 * logit(qq) + a.value - b.value), ours: a, theirs: b }
}

/**
 * Value of banning `hero` against the enemy's matched players: how much their best
 * personal option drops (logit units), over players not yet given a hero. `who` is a player id.
 */
export function banGain(
  hero: string,
  available: string[],
  theirPlayers: PersonalPlayer[],
  assigned: Set<string>,
  c: DraftCoefs,
): { gain: number; who: string | null } {
  let gain = 0
  let who: string | null = null
  for (const p of theirPlayers) {
    if (assigned.has(p.id)) continue
    let best = -Infinity
    let without = -Infinity
    for (const h of available) {
      const v = playerValue(p, h)
      best = Math.max(best, v)
      if (h !== hero) without = Math.max(without, v)
    }
    const g = Number.isFinite(without) ? best - without : 0
    if (g > gain) {
      gain = g
      who = p.id
    }
  }
  return { gain, who }
}
