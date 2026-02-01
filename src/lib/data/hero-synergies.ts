// Hero synergy data for Heroes of the Storm
// Represents strong 2-hero combinations with explanations

export interface HeroSynergy {
  heroes: [string, string]
  reason: string
  strength: 'high' | 'medium'
}

export const HERO_SYNERGIES: HeroSynergy[] = [
  // Tank + Healer Synergies
  { heroes: ['Diablo', 'Anduin'], reason: 'Diablo flip into Anduin root', strength: 'high' },
  { heroes: ['Stitches', 'Anduin'], reason: 'Hook into root combo', strength: 'high' },
  { heroes: ['Garrosh', 'Deckard'], reason: 'Throw into Deckard root', strength: 'high' },
  { heroes: ['Muradin', 'Rehgar'], reason: 'Stun setup for Ancestral timing', strength: 'medium' },

  // Tank + Damage Synergies
  { heroes: ['E.T.C.', 'Jaina'], reason: 'Mosh pit into Ring of Frost', strength: 'high' },
  { heroes: ['E.T.C.', 'Kael\'thas'], reason: 'Mosh pit into Pyroblast/Flamestrike', strength: 'high' },
  { heroes: ['Johanna', 'Kael\'thas'], reason: 'Blessed Shield into Flamestrike', strength: 'medium' },
  { heroes: ['Muradin', 'Valla'], reason: 'Stun for Valla follow-up', strength: 'medium' },

  // Dive Compositions
  { heroes: ['Illidan', 'Abathur'], reason: 'Hat Illidan dive comp', strength: 'high' },
  { heroes: ['Illidan', 'Tyrael'], reason: 'Sanctification for dive protection', strength: 'high' },
  { heroes: ['Kerrigan', 'Medivh'], reason: 'Portal combos with Kerrigan engage', strength: 'high' },
  { heroes: ['Zeratul', 'Tyrande'], reason: 'Hunter\'s Mark for burst damage', strength: 'medium' },

  // Siege/Push Synergies
  { heroes: ['Azmodan', 'Johanna'], reason: 'Frontline protection for Azmo stacks', strength: 'medium' },
  { heroes: ['Nazeebo', 'Johanna'], reason: 'Peel for Nazeebo stacking', strength: 'medium' },
  { heroes: ['Zagara', 'Dehaka'], reason: 'Global presence and vision control', strength: 'medium' },
  { heroes: ['Sylvanas', 'Zagara'], reason: 'Push and structure damage', strength: 'medium' },

  // Protect the Hypercarry
  { heroes: ['Ana', 'Genji'], reason: 'Nano boost on Genji', strength: 'high' },
  { heroes: ['Ana', 'Illidan'], reason: 'Nano boost on Illidan', strength: 'high' },
  { heroes: ['Uther', 'Greymane'], reason: 'Divine Shield for aggressive dives', strength: 'high' },
  { heroes: ['Zarya', 'Tracer'], reason: 'Shield on Tracer for aggressive plays', strength: 'medium' },

  // Poke Compositions
  { heroes: ['Chromie', 'Hanzo'], reason: 'Long-range poke composition', strength: 'medium' },
  { heroes: ['Li-Ming', 'Junkrat'], reason: 'Poke and resets', strength: 'medium' },
  { heroes: ['Deckard', 'Valla'], reason: 'Poke sustain with potions', strength: 'medium' },

  // AOE Lockdown
  { heroes: ['Malfurion', 'E.T.C.'], reason: 'Entangling roots + Mosh pit', strength: 'high' },
  { heroes: ['Malfurion', 'Kerrigan'], reason: 'Root into Kerrigan combo', strength: 'high' },
  { heroes: ['Xul', 'Johanna'], reason: 'Bone prison + Condemn lockdown', strength: 'medium' },

  // Double Support
  { heroes: ['Abathur', 'Rehgar'], reason: 'Global presence with heals', strength: 'medium' },
  { heroes: ['Tyrande', 'Malfurion'], reason: 'Double support sustain', strength: 'medium' },
  { heroes: ['Zarya', 'Tassadar'], reason: 'Shield synergy', strength: 'medium' },

  // Wombo Combo
  { heroes: ['Zarya', 'Jaina'], reason: 'Graviton Surge into Ring of Frost', strength: 'high' },
  { heroes: ['Zarya', 'Kael\'thas'], reason: 'Graviton into Flamestrike', strength: 'high' },
  { heroes: ['Ley Line', 'Medivh'], reason: 'Portal into Ley Line combo', strength: 'high' },

  // Specialist Synergies
  { heroes: ['Abathur', 'The Lost Vikings'], reason: 'Hat Vikings for global pressure', strength: 'medium' },
  { heroes: ['Cho', 'Gall'], reason: 'Two-headed ogre synergy (required)', strength: 'high' },
  { heroes: ['Medivh', 'Valla'], reason: 'Portal plays with Valla mobility', strength: 'medium' },

  // Execute Combos
  { heroes: ['Kael\'thas', 'Jaina'], reason: 'Burst mage combo', strength: 'medium' },
  { heroes: ['Greymane', 'Valla'], reason: 'AA damage synergy', strength: 'medium' },
  { heroes: ['Valla', 'Tyrande'], reason: 'Hunter\'s Mark with Valla AA', strength: 'medium' },

  // Front to Back
  { heroes: ['Johanna', 'Li-Ming'], reason: 'Peel for Li-Ming poke', strength: 'medium' },
  { heroes: ['Muradin', 'Raynor'], reason: 'Frontline with sustained backline', strength: 'medium' },
  { heroes: ['Garrosh', 'Stukov'], reason: 'Throw setup with Stukov silence', strength: 'medium' },

  // Macro Play
  { heroes: ['Falstad', 'Brightwing'], reason: 'Global rotation dominance', strength: 'high' },
  { heroes: ['Dehaka', 'Falstad'], reason: 'Global presence', strength: 'medium' },
  { heroes: ['Abathur', 'Falstad'], reason: 'Split push with global', strength: 'medium' },
]

export function findSynergies(heroes: (string | null)[]): HeroSynergy[] {
  const pickedHeroes = heroes.filter(h => h !== null) as string[]
  const synergies: HeroSynergy[] = []

  // Check all pairs of picked heroes
  for (let i = 0; i < pickedHeroes.length; i++) {
    for (let j = i + 1; j < pickedHeroes.length; j++) {
      const hero1 = pickedHeroes[i]
      const hero2 = pickedHeroes[j]

      // Find synergy in either direction
      const synergy = HERO_SYNERGIES.find(s =>
        (s.heroes[0] === hero1 && s.heroes[1] === hero2) ||
        (s.heroes[0] === hero2 && s.heroes[1] === hero1)
      )

      if (synergy) {
        synergies.push(synergy)
      }
    }
  }

  return synergies
}

export function getSynergyScore(heroes: (string | null)[]): number {
  const synergies = findSynergies(heroes)
  let score = 0

  synergies.forEach(s => {
    score += s.strength === 'high' ? 2 : 1
  })

  return score
}
