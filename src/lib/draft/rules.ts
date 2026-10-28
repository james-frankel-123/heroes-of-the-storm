/**
 * Draft rules shared by the board reducer, the in-browser search (mcts-search.ts) and the
 * enemy-impact projections, so all three see the same legal actions and transitions.
 * Pure; tested in src/lib/draft/__tests__/rules.test.ts.
 *
 * A draft position is the board's own representation: `selections` (step index -> hero) and
 * `step` (the next step to fill). Cho'gall is two heroes, Cho and Gall, drafted together:
 *  - picking either fills the team's next pick slot with the other; that slot is skipped;
 *  - either is pickable only when the team has two consecutive open pick slots now;
 *  - banning or picking either makes both unavailable.
 */
import { DRAFT_SEQUENCE, type Team } from './types'

export type Selections = Record<number, string>

export const CHO = 'Cho'
export const GALL = 'Gall'

export function companionOf(hero: string): string | null {
  return hero === CHO ? GALL : hero === GALL ? CHO : null
}

/** Open pick slots the team has in a row starting at `step` (a prefilled companion ends the run). */
export function consecutivePicksRemaining(step: number, team: Team, selections: Selections): number {
  let count = 0
  for (let i = step; i < DRAFT_SEQUENCE.length; i++) {
    const s = DRAFT_SEQUENCE[i]
    if (s.type !== 'pick' || s.team !== team) break
    if (selections[i]) break
    count++
  }
  return count
}

/** Heroes no step can take any more: every selected hero, with Cho and Gall as one. */
export function takenHeroes(selections: Selections): Set<string> {
  const taken = new Set(Object.values(selections))
  if (taken.has(CHO) || taken.has(GALL)) {
    taken.add(CHO)
    taken.add(GALL)
  }
  return taken
}

/** Heroes the step at `step` may not take: the taken ones, plus Cho/Gall without two open slots. */
export function unavailableAt(selections: Selections, step: number): Set<string> {
  const out = takenHeroes(selections)
  const s = DRAFT_SEQUENCE[step]
  if (s && s.type === 'pick' && consecutivePicksRemaining(step, s.team, selections) < 2) {
    out.add(CHO)
    out.add(GALL)
  }
  return out
}

export function isLegal(selections: Selections, step: number, hero: string): boolean {
  return step < DRAFT_SEQUENCE.length && !selections[step] && !unavailableAt(selections, step).has(hero)
}

/**
 * Select `hero` at `step` (the board's SELECT_HERO): for a Cho/Gall pick the companion fills
 * the team's next pick slot, then the step advances past every prefilled slot.
 */
export function applySelection(selections: Selections, step: number, hero: string): { selections: Selections; step: number } {
  const next: Selections = { ...selections, [step]: hero }
  const cur = DRAFT_SEQUENCE[step]
  const companion = companionOf(hero)
  if (cur?.type === 'pick' && companion) {
    for (let i = step + 1; i < DRAFT_SEQUENCE.length; i++) {
      const s = DRAFT_SEQUENCE[i]
      if (s.type === 'pick' && s.team === cur.team) {
        next[i] = companion
        break
      }
    }
  }
  let n = step + 1
  while (n < DRAFT_SEQUENCE.length && next[n]) n++
  return { selections: next, step: n }
}

/** Picks per team and bans of a position (every filled step, companions included). */
export function lineups(selections: Selections): { A: string[]; B: string[]; bans: string[] } {
  const out = { A: [] as string[], B: [] as string[], bans: [] as string[] }
  for (let i = 0; i < DRAFT_SEQUENCE.length; i++) {
    const h = selections[i]
    if (!h) continue
    const s = DRAFT_SEQUENCE[i]
    if (s.type === 'ban') out.bans.push(h)
    else out[s.team].push(h)
  }
  return out
}

/**
 * A position is legal when no hero appears twice (Cho and Gall count as one hero only when
 * paired on the same team), Cho and Gall are never split across teams or between a pick and
 * nothing, and every filled step is a real step. Used to assert before evaluating.
 */
export function positionProblems(selections: Selections, complete = false): string[] {
  const problems: string[] = []
  const seen = new Map<string, number>()
  for (const [k, h] of Object.entries(selections)) {
    const i = Number(k)
    if (!(i >= 0 && i < DRAFT_SEQUENCE.length)) problems.push(`step ${k} out of range`)
    if (seen.has(h)) problems.push(`${h} at steps ${seen.get(h)} and ${i}`)
    seen.set(h, i)
  }
  const { A, B, bans } = lineups(selections)
  for (const team of [A, B]) {
    const cho = team.includes(CHO)
    const gall = team.includes(GALL)
    if (cho !== gall) problems.push(`unpaired ${cho ? CHO : GALL}`)
  }
  if ((A.includes(CHO) || A.includes(GALL)) && (B.includes(CHO) || B.includes(GALL))) problems.push('Cho/Gall on both teams')
  if ((bans.includes(CHO) || bans.includes(GALL)) && [...A, ...B].some(h => h === CHO || h === GALL)) {
    problems.push('Cho/Gall banned and picked')
  }
  if (complete && (A.length !== 5 || B.length !== 5)) problems.push(`complete draft has ${A.length} + ${B.length} picks`)
  return problems
}
