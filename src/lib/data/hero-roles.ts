// Hero role classifications for Heroes of the Storm
// Based on official role classifications

export type HeroRole = 'Tank' | 'Bruiser' | 'Melee Assassin' | 'Ranged Assassin' | 'Healer' | 'Support'

export const HERO_ROLES: Record<string, HeroRole> = {
  // Tanks
  'Anub\'arak': 'Tank',
  'Arthas': 'Tank',
  'Blaze': 'Tank',
  'Diablo': 'Tank',
  'E.T.C.': 'Tank',
  'Garrosh': 'Tank',
  'Johanna': 'Tank',
  'Mal\'Ganis': 'Tank',
  'Mei': 'Tank',
  'Muradin': 'Tank',
  'Stitches': 'Tank',
  'Tyrael': 'Tank',

  // Bruisers
  'Artanis': 'Bruiser',
  'Chen': 'Bruiser',
  'Cho': 'Bruiser',
  'Dehaka': 'Bruiser',
  'D.Va': 'Bruiser',
  'Gazlowe': 'Bruiser',
  'Hogger': 'Bruiser',
  'Imperius': 'Bruiser',
  'Leoric': 'Bruiser',
  'Malthael': 'Bruiser',
  'Ragnaros': 'Bruiser',
  'Rexxar': 'Bruiser',
  'Sonya': 'Bruiser',
  'Thrall': 'Bruiser',
  'Varian': 'Bruiser',
  'Xul': 'Bruiser',
  'Yrel': 'Bruiser',

  // Melee Assassins
  'Alarak': 'Melee Assassin',
  'Illidan': 'Melee Assassin',
  'Kerrigan': 'Melee Assassin',
  'Maiev': 'Melee Assassin',
  'Murky': 'Melee Assassin',
  'Qhira': 'Melee Assassin',
  'Samuro': 'Melee Assassin',
  'The Butcher': 'Melee Assassin',
  'Valeera': 'Melee Assassin',
  'Zeratul': 'Melee Assassin',

  // Ranged Assassins
  'Azmodan': 'Ranged Assassin',
  'Cassia': 'Ranged Assassin',
  'Chromie': 'Ranged Assassin',
  'Falstad': 'Ranged Assassin',
  'Fenix': 'Ranged Assassin',
  'Gall': 'Ranged Assassin',
  'Genji': 'Ranged Assassin',
  'Greymane': 'Ranged Assassin',
  'Gul\'dan': 'Ranged Assassin',
  'Hanzo': 'Ranged Assassin',
  'Jaina': 'Ranged Assassin',
  'Junkrat': 'Ranged Assassin',
  'Kael\'thas': 'Ranged Assassin',
  'Kel\'Thuzad': 'Ranged Assassin',
  'Li-Ming': 'Ranged Assassin',
  'Lunara': 'Ranged Assassin',
  'Mephisto': 'Ranged Assassin',
  'Nazeebo': 'Ranged Assassin',
  'Nova': 'Ranged Assassin',
  'Orphea': 'Ranged Assassin',
  'Probius': 'Ranged Assassin',
  'Raynor': 'Ranged Assassin',
  'Sgt. Hammer': 'Ranged Assassin',
  'Sylvanas': 'Ranged Assassin',
  'Tracer': 'Ranged Assassin',
  'Tychus': 'Ranged Assassin',
  'Valla': 'Ranged Assassin',
  'Zul\'jin': 'Ranged Assassin',

  // Healers
  'Alexstrasza': 'Healer',
  'Ana': 'Healer',
  'Anduin': 'Healer',
  'Auriel': 'Healer',
  'Brightwing': 'Healer',
  'Deckard': 'Healer',
  'Kharazim': 'Healer',
  'Li Li': 'Healer',
  'Lt. Morales': 'Healer',
  'Lúcio': 'Healer',
  'Malfurion': 'Healer',
  'Rehgar': 'Healer',
  'Stukov': 'Healer',
  'Tyrande': 'Healer',
  'Uther': 'Healer',
  'Whitemane': 'Healer',

  // Support
  'Abathur': 'Support',
  'Medivh': 'Support',
  'The Lost Vikings': 'Support',
  'Tassadar': 'Support',
  'Zarya': 'Support',
}

export function getHeroRole(hero: string): HeroRole | null {
  return HERO_ROLES[hero] || null
}

export interface RoleBalance {
  tank: number
  bruiser: number
  meleeAssassin: number
  rangedAssassin: number
  healer: number
  support: number
}

export function calculateRoleBalance(heroes: (string | null)[]): RoleBalance {
  const balance: RoleBalance = {
    tank: 0,
    bruiser: 0,
    meleeAssassin: 0,
    rangedAssassin: 0,
    healer: 0,
    support: 0,
  }

  heroes.forEach(hero => {
    if (!hero) return
    const role = getHeroRole(hero)
    if (!role) return

    switch (role) {
      case 'Tank':
        balance.tank++
        break
      case 'Bruiser':
        balance.bruiser++
        break
      case 'Melee Assassin':
        balance.meleeAssassin++
        break
      case 'Ranged Assassin':
        balance.rangedAssassin++
        break
      case 'Healer':
        balance.healer++
        break
      case 'Support':
        balance.support++
        break
    }
  })

  return balance
}

export interface RoleNeed {
  role: string
  priority: 'critical' | 'important' | 'nice-to-have'
}

export function analyzeRoleNeeds(balance: RoleBalance): RoleNeed[] {
  const needs: RoleNeed[] = []

  // Critical needs
  if (balance.tank === 0) {
    needs.push({ role: 'Tank', priority: 'critical' })
  }
  if (balance.healer === 0) {
    needs.push({ role: 'Healer', priority: 'critical' })
  }

  // Important needs
  const totalDamage = balance.meleeAssassin + balance.rangedAssassin
  if (totalDamage === 0) {
    needs.push({ role: 'Damage', priority: 'critical' })
  } else if (balance.rangedAssassin === 0) {
    needs.push({ role: 'Ranged Assassin', priority: 'important' })
  }

  // Nice to have
  if (balance.bruiser === 0 && balance.tank === 1) {
    needs.push({ role: 'Bruiser', priority: 'nice-to-have' })
  }

  return needs
}
