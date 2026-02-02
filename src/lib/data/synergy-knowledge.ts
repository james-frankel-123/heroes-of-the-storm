/**
 * Synergy Knowledge Database
 *
 * Defines hero duos that work well together with mechanical explanations.
 * This enables LLM commentary to explain WHY duos work rather than just
 * stating "you have high win rate together."
 */

export interface AbilityCombo {
  abilities: string[]      // e.g., ["Hook", "Root"]
  description: string      // What the combo does
  timing?: string          // Execution timing
}

export interface HeroSynergy {
  duo: [string, string]    // Hero names (order doesn't matter)
  category: string         // Type of synergy
  rating: 'S' | 'A' | 'B'  // S = godly, A = strong, B = decent

  // Why they work together
  reason: string

  // Mechanical combos
  combos: AbilityCombo[]

  // How to execute
  execution: string[]

  // Draft considerations
  draftOrder: string
  pickStrategy: string

  // Counters to this duo
  counters?: string[]
}

/**
 * Synergy database for hero duos
 * Focus on impactful and common synergies
 */
export const HERO_SYNERGIES: HeroSynergy[] = [
  // === TANK + HEALER COMBOS ===

  {
    duo: ["Stitches", "Anduin"],
    category: "Playmaker Combo",
    rating: "S",
    reason: "Hook into Root creates 2.5+ seconds of lockdown. Pull saves hooked allies. Lightbomb extends hook combo.",
    combos: [
      {
        abilities: ["Hook", "Root"],
        description: "Hook pulls target, Anduin immediately roots for extended lockdown",
        timing: "Root as soon as hook lands"
      },
      {
        abilities: ["Gorge", "Lightbomb"],
        description: "Lightbomb Stitches as he walks toward team with Gorged target",
        timing: "Lightbomb right before Gorge ends"
      },
      {
        abilities: ["Hook", "Pull"],
        description: "Pull hooked ally to safety if hook was on ally",
        timing: "Immediate - hook can be on ally"
      }
    ],
    execution: [
      "Stitches positions for hook on priority target",
      "Anduin positions to follow up with root immediately",
      "Root the hooked target the moment they stop moving",
      "Team collapses for kill",
      "Save Pull for hooked allies or diving enemies"
    ],
    draftOrder: "Pick Anduin after Stitches to threaten hook combos",
    pickStrategy: "First-pick Stitches, then secure Anduin. Forces enemy to draft mobile heroes or cleanse.",
    counters: ["Cleanse", "Tracer", "Genji", "Unstoppable effects"]
  },

  {
    duo: ["Diablo", "Anduin"],
    category: "Displacement Combo",
    rating: "A",
    reason: "Flip into Root locks down targets. Pull repositions Diablo after dive. Lightbomb amplifies Diablo engage.",
    combos: [
      {
        abilities: ["Shadow Charge", "Overpower", "Root"],
        description: "Diablo flips target into wall, Anduin roots for extended CC",
        timing: "Root immediately after Overpower"
      },
      {
        abilities: ["Shadow Charge", "Lightbomb"],
        description: "Lightbomb Diablo as he charges into enemy team",
        timing: "Lightbomb mid-charge for AOE followup"
      }
    ],
    execution: [
      "Diablo charges and flips priority target into wall",
      "Anduin roots the flipped target",
      "Pull Diablo to safety if overextended",
      "Lightbomb for AOE damage in teamfights"
    ],
    draftOrder: "Pick Anduin after Diablo to maximize CC chain",
    pickStrategy: "Strong vs immobile comps. Requires terrain for Diablo flips.",
    counters: ["Mobile heroes", "Cleanse", "Spell armor"]
  },

  {
    duo: ["ETC", "Rehgar"],
    category: "Wombo Combo",
    rating: "S",
    reason: "Mosh Pit with Bloodlust is devastating. Ancestral saves ETC during/after Mosh. Stage Dive into teamfight.",
    combos: [
      {
        abilities: ["Mosh Pit", "Bloodlust"],
        description: "Bloodlust team while enemies are moshed for massive damage",
        timing: "Bloodlust immediately after Mosh starts"
      },
      {
        abilities: ["Mosh Pit", "Ancestral Healing"],
        description: "Ancestral on ETC during Mosh to keep him alive",
        timing: "Ancestral when ETC reaches ~30% health during Mosh"
      },
      {
        abilities: ["Stage Dive", "Wolf Run"],
        description: "Both engage simultaneously with mobility",
        timing: "Coordinate engages"
      }
    ],
    execution: [
      "ETC lands Mosh Pit on 3+ targets",
      "Rehgar immediately casts Bloodlust on team",
      "Team unloads damage on moshed enemies",
      "Ancestral on ETC if he's low during Mosh",
      "Lightning Shield on ETC for extra damage"
    ],
    draftOrder: "Can pick either first. ETC + Rehgar = wombo threat",
    pickStrategy: "Forces enemy to draft Mosh counters. Bloodlust with melee-heavy comps.",
    counters: ["Stun ETC during Mosh", "Displacements", "Cleanse"]
  },

  // === TANK + DPS COMBOS ===

  {
    duo: ["Stitches", "Kerrigan"],
    category: "Hook Combo",
    rating: "A",
    reason: "Hook into Kerrigan combo for instant deletion. Both excel at picks and ganks.",
    combos: [
      {
        abilities: ["Hook", "Ravage", "Primal Grasp", "Impaling Blades"],
        description: "Hook pulls target, Kerrigan lands full combo for massive burst",
        timing: "Kerrigan combos as hook lands"
      },
      {
        abilities: ["Gorge", "Kerrigan Combo"],
        description: "Gorge and walk toward Kerrigan, she combos as target is released",
        timing: "Kerrigan positions for combo before Gorge ends"
      }
    ],
    execution: [
      "Stitches hooks priority target",
      "Kerrigan immediately follows with Ravage → Primal Grasp → Impaling Blades",
      "Team follows up for kill",
      "Works on any squishy target"
    ],
    draftOrder: "Pick Stitches first, Kerrigan after",
    pickStrategy: "Godlike pick potential. Forces enemy to group and play safe.",
    counters: ["Cleanse", "Spell armor", "Don't get hooked"]
  },

  {
    duo: ["ETC", "Jaina"],
    category: "AOE Burst",
    rating: "S",
    reason: "Mosh Pit locks down enemies, Jaina unloads AOE burst. Ring of Frost + Mosh is GG.",
    combos: [
      {
        abilities: ["Mosh Pit", "Ring of Frost"],
        description: "Ring of Frost on moshed enemies for extended lockdown + burst",
        timing: "Ring as Mosh starts"
      },
      {
        abilities: ["Mosh Pit", "Blizzard", "Cone of Cold"],
        description: "Full Jaina burst on helpless enemies",
        timing: "Blizzard immediately, Cone when in range"
      }
    ],
    execution: [
      "ETC lands Mosh Pit on grouped enemies",
      "Jaina immediately casts Ring of Frost + Blizzard",
      "Cone of Cold for massive burst",
      "Team wipes enemy",
      "Water Elemental for extra damage"
    ],
    draftOrder: "Pick ETC first, Jaina after",
    pickStrategy: "Wombo combo comp. Requires enemy to group. Devastating in teamfights.",
    counters: ["Spread formation", "Stun ETC", "Spell armor"]
  },

  // === DOUBLE SUPPORT ===

  {
    duo: ["Illidan", "Rehgar"],
    category: "Enabler Duo",
    rating: "S",
    reason: "Ancestral keeps Illidan alive. Bloodlust amplifies Illidan's damage. Lightning Shield adds damage.",
    combos: [
      {
        abilities: ["The Hunt", "Ancestral Healing"],
        description: "Illidan hunts backline, Rehgar ancestrals when low",
        timing: "Ancestral when Illidan reaches ~25% HP"
      },
      {
        abilities: ["Metamorphosis", "Bloodlust"],
        description: "Bloodlust Illidan during Meta for unstoppable damage",
        timing: "Bloodlust as Meta starts"
      },
      {
        abilities: ["Dive", "Lightning Shield"],
        description: "Lightning Shield on Illidan for extra damage",
        timing: "Shield before Illidan dives"
      }
    ],
    execution: [
      "Rehgar shields Illidan before dive",
      "Illidan dives backline",
      "Rehgar follows in wolf form",
      "Ancestral when Illidan gets low",
      "Bloodlust for sustained fights"
    ],
    draftOrder: "Pick Illidan, then Rehgar to enable him",
    pickStrategy: "Illidan requires enablers. Rehgar is top tier with him. Draft second support.",
    counters: ["Mages", "Hard CC", "Polymorph"]
  },

  {
    duo: ["Illidan", "Abathur"],
    category: "Hypercarry",
    rating: "S",
    reason: "Symbiote makes Illidan unkillable. Clone creates two Illidans. Ultimate Evolution doubles carry threat.",
    combos: [
      {
        abilities: ["Dive", "Symbiote"],
        description: "Abathur hats Illidan during dives for shields and damage",
        timing: "Hat immediately when Illidan engages"
      },
      {
        abilities: ["Metamorphosis", "Ultimate Evolution"],
        description: "Two Illidans with Meta is unstoppable",
        timing: "Clone Illidan during key teamfights"
      }
    ],
    execution: [
      "Abathur permanently hats Illidan",
      "Illidan dives with hat shields",
      "Abathur spikes and shields during dive",
      "Clone Illidan for teamfights",
      "Two Illidans win game"
    ],
    draftOrder: "Pick Illidan, then Abathur. Signals hypercarry comp.",
    pickStrategy: "Requires draft built around Illidan. Team must enable him.",
    counters: ["Mages", "Artanis", "Johanna"]
  },

  // === DPS DUOS ===

  {
    duo: ["Tracer", "Tassadar"],
    category: "Dive Duo",
    rating: "A",
    reason: "Force Wall enables Tracer kills. Lifesteal keeps Tracer healthy. Oracle reveals enemies for Pulse Bomb.",
    combos: [
      {
        abilities: ["Pulse Bomb", "Force Wall"],
        description: "Force Wall traps enemies, Tracer Pulse Bombs trapped target",
        timing: "Coordinate wall placement with bomb"
      },
      {
        abilities: ["Blink", "Shield"],
        description: "Shield Tracer as she dives",
        timing: "Shield before Tracer engages"
      }
    ],
    execution: [
      "Tassadar shields Tracer before dive",
      "Tracer dives backline",
      "Tassadar walls off escape routes",
      "Pulse Bomb on trapped target",
      "Oracle for vision"
    ],
    draftOrder: "Pick Tracer, then Tassadar",
    pickStrategy: "Strong dive comp. Requires team to enable Tracer.",
    counters: ["Hard CC", "Blinds", "Targeted CC"]
  },

  // === GLOBALS ===

  {
    duo: ["Falstad", "Brightwing"],
    category: "Global Duo",
    rating: "A",
    reason: "Double global enables superior rotations. Constant map presence. Gust + Emerald Wind for disengage.",
    combos: [
      {
        abilities: ["Flight", "Phase Shift"],
        description: "Both respond to ganks or objectives instantly",
        timing: "Coordinate globals to save allies or secure kills"
      },
      {
        abilities: ["Gust", "Emerald Wind"],
        description: "Double knockback for ultimate disengage",
        timing: "Layer knockbacks to push enemies far"
      }
    ],
    execution: [
      "Play aggressive knowing you have global backup",
      "Both respond to ganks instantly",
      "Rotate to objectives together",
      "Double knockback for peel",
      "Force enemy to group or get picked"
    ],
    draftOrder: "Can pick either first. Double global forces respect.",
    pickStrategy: "Strong on large maps. Enables split-push with safety.",
    counters: ["4-1 Split push", "Hard engage", "Dive past globals"]
  },

  // === SPLIT PUSH ===

  {
    duo: ["Azmodan", "Abathur"],
    category: "Push Duo",
    rating: "B",
    reason: "Abathur hats Azmodan's demon warriors. Double soak and push pressure. Ultimate Evolution clones Azmodan for siege.",
    combos: [
      {
        abilities: ["Demon Warriors", "Symbiote"],
        description: "Hat demon warriors for extra push power",
        timing: "Hat strongest demon"
      },
      {
        abilities: ["Black Pool", "Ultimate Evolution"],
        description: "Two Azmodans with Black Pool melts structures",
        timing: "Clone for key structure pushes"
      }
    ],
    execution: [
      "Azmodan stacks quest",
      "Abathur hats demons for push",
      "Both soak lanes",
      "Clone Azmodan for teamfights or siege",
      "Overwhelming macro pressure"
    ],
    draftOrder: "Pick Abathur first, then Azmodan",
    pickStrategy: "Macro comp. Requires strong 3-man to stall. Weak to hard engage.",
    counters: ["Dive comps", "Hard engage", "4-1 pressure"]
  },

  // === MAGE COMBOS ===

  {
    duo: ["Jaina", "Kael'thas"],
    category: "Double Mage",
    rating: "A",
    reason: "Combined AOE burst deletes teams. Chill + Gravity Lapse CC chain. Overwhelming teamfight damage.",
    combos: [
      {
        abilities: ["Ring of Frost", "Flamestrike"],
        description: "Ring locks down enemies, Kael Flamestrikes for damage",
        timing: "Flamestrike immediately on Ring"
      },
      {
        abilities: ["Blizzard", "Phoenix"],
        description: "Layered AOE denial zones",
        timing: "Coordinate to cover objective"
      }
    ],
    execution: [
      "Draft strong frontline to protect mages",
      "Jaina slows with Chill",
      "Kael follows with burst",
      "Ring of Frost for setup",
      "Combined AOE wins teamfights"
    ],
    draftOrder: "Pick mages last after securing frontline",
    pickStrategy: "Strong vs grouped comps. Weak vs dive. Requires peel.",
    counters: ["Dive", "Spread formation", "Spell armor"]
  }
]

/**
 * Get synergy knowledge for a hero duo
 * Order doesn't matter - will find synergy regardless
 */
export function getHeroSynergy(hero1: string, hero2: string): HeroSynergy | undefined {
  return HERO_SYNERGIES.find(
    synergy =>
      (synergy.duo[0] === hero1 && synergy.duo[1] === hero2) ||
      (synergy.duo[0] === hero2 && synergy.duo[1] === hero1)
  )
}

/**
 * Get all synergies for a specific hero
 */
export function getSynergiesForHero(heroName: string): HeroSynergy[] {
  return HERO_SYNERGIES.filter(
    synergy => synergy.duo[0] === heroName || synergy.duo[1] === heroName
  )
}

/**
 * Get synergies by rating
 */
export function getSynergiesByRating(rating: 'S' | 'A' | 'B'): HeroSynergy[] {
  return HERO_SYNERGIES.filter(synergy => synergy.rating === rating)
}

/**
 * Check if two heroes have defined synergy
 */
export function hasSynergy(hero1: string, hero2: string): boolean {
  return getHeroSynergy(hero1, hero2) !== undefined
}

/**
 * Get synergy category description
 */
export function getSynergyDescription(hero1: string, hero2: string): string | null {
  const synergy = getHeroSynergy(hero1, hero2)
  if (!synergy) return null

  return `${synergy.category} (${synergy.rating}-tier): ${synergy.reason}`
}
