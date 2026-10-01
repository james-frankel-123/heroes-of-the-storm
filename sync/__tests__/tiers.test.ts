import { describe, expect, it } from 'vitest'
import { avgPlayerMmr, legacyListingTier } from '../listing-fields'
import { leagueTierToSkillTier } from '../sync-replays'

describe('leagueTierToSkillTier', () => {
  it("maps listing tier ids (one above the tier names) to the site's scheme", () => {
    expect(leagueTierToSkillTier(1, 1800)).toBe('low') // Wood
    expect(leagueTierToSkillTier(2, 2300)).toBe('low') // Bronze
    expect(leagueTierToSkillTier(3, 2500)).toBe('low') // Silver
    expect(leagueTierToSkillTier(4, 2650)).toBe('mid') // Gold
    expect(leagueTierToSkillTier(5, 2750)).toBe('mid') // Platinum
    expect(leagueTierToSkillTier(6, 2900)).toBe('high') // Diamond
  })

  it('treats a null tier with an MMR as Master (high), not mid', () => {
    expect(leagueTierToSkillTier(null, 3050)).toBe('high')
  })

  it('reports unknown when there is neither a tier nor an MMR', () => {
    expect(leagueTierToSkillTier(null, null)).toBe('unknown')
  })
})

describe('legacyListingTier', () => {
  it('shifts v1 tier names to the old listing ids', () => {
    expect(legacyListingTier('Bronze 1', 'sl')).toBe(2)
    expect(legacyListingTier('Silver 5', 'sl')).toBe(3)
    expect(legacyListingTier('Diamond 3', 'qm')).toBe(6)
  })

  it('encodes Master as null for Storm League and 0 for Quick Match', () => {
    expect(legacyListingTier('Master', 'sl')).toBeNull()
    expect(legacyListingTier('Master', 'qm')).toBe(0)
  })

  it('returns null for an unrecognised tier name', () => {
    expect(legacyListingTier('Plastic 2', 'sl')).toBeNull()
  })
})

describe('avgPlayerMmr', () => {
  const player = (player_mmr: number | null) => ({ hero: 'Jaina', team: 0, player_mmr })

  it('averages player MMRs and ignores replay-level fields', () => {
    expect(avgPlayerMmr({ region: 1, 'A#1': player(2800), 'B#2': player(3000) })).toBe(2900)
  })

  it('leaves private players (null MMR) out instead of counting them as 0', () => {
    expect(avgPlayerMmr({ 'A#1': player(2800), 'private:0:1': player(null), 'B#2': player(3000) })).toBe(2900)
  })

  it('returns null when no player has an MMR', () => {
    expect(avgPlayerMmr({ 'private:0:0': player(null) })).toBeNull()
  })
})
