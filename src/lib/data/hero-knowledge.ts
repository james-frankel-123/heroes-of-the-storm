/**
 * Hero Knowledge Database
 *
 * Provides mechanical, strategic, and positioning knowledge for heroes.
 * This data enables LLM commentary to give specific, actionable advice
 * rather than generic stat prose.
 */

export interface HeroAbility {
  key: string
  name: string
  description: string
  cooldown?: number
  synergies?: string[]  // Other abilities/heroes that combo well
  counters?: string[]   // What this ability is weak against
}

export interface HeroKnowledge {
  hero: string
  role: string

  // Core strengths and playstyle
  strengths: string[]
  weaknesses: string[]
  playstyle: string  // Brief description of optimal playstyle

  // Key abilities (focus on impactful ones)
  abilities?: HeroAbility[]

  // Matchups
  counters: string[]      // Heroes that counter this hero
  strongAgainst: string[]  // Heroes this hero counters

  // Map recommendations
  bestMaps: string[]   // Maps where this hero excels
  worstMaps: string[]  // Maps where this hero struggles
  mapStrategy: string  // Why certain maps work/don't work

  // Positioning and macro
  positioning: string
  objectiveValue: string  // How to play around objectives

  // Draft considerations
  draftStrategy: string
  pickTiming: 'early' | 'mid' | 'late' | 'flex'  // When to pick in draft

  // Skill requirements
  skillFloor: 'low' | 'medium' | 'high'
  skillCeiling: 'low' | 'medium' | 'high'
}

/**
 * Hero knowledge database
 * Start with popular heroes across all roles
 */
export const HERO_KNOWLEDGE: Record<string, HeroKnowledge> = {
  // === TANKS ===

  "Stitches": {
    hero: "Stitches",
    role: "Tank",
    strengths: [
      "Pick potential with Hook",
      "High health pool and sustain",
      "Gorge removes enemies from fights",
      "Strong zoning presence"
    ],
    weaknesses: [
      "Vulnerable to percentage-based damage",
      "Limited mobility",
      "Relies on skillshot hook",
      "Weak vs poke comps"
    ],
    playstyle: "Playmaker tank that creates picks with Hook. Position in brush or flanks to land hooks on backline. Use Gorge to remove key targets or reposition enemies.",

    abilities: [
      {
        key: "Q",
        name: "Hook",
        description: "Pulls enemy to Stitches",
        cooldown: 16,
        synergies: ["Anduin Root", "Diablo Flip", "Kerrigan Combo"],
        counters: ["Tracer Recall", "Genji Deflect", "Iceblock"]
      },
      {
        key: "W",
        name: "Slam",
        description: "AOE damage around Stitches",
        cooldown: 4,
        synergies: ["Hook followup"],
      },
      {
        key: "R1",
        name: "Gorge",
        description: "Swallow enemy, remove from fight",
        cooldown: 65,
        synergies: ["Anduin Lightbomb", "Walking toward team"],
        counters: ["Cleanse", "Unstoppable effects"]
      }
    ],

    counters: ["Tychus", "Malthael", "Leoric", "Tracer", "Genji"],
    strongAgainst: ["Immobile mages", "Backline assassins", "Healers without mobility"],

    bestMaps: ["Cursed Hollow", "Dragon Shire", "Infernal Shrines"],
    worstMaps: ["Braxis Holdout", "Volskaya Foundry"],
    mapStrategy: "Excels on large maps with brush near objectives. Struggles on small maps where brush positioning is limited and poke is prevalent.",

    positioning: "Flank through brush for hooks. Don't frontline - stay at medium range looking for pick opportunities. Save Gorge for high-value targets or repositioning enemies into your team.",
    objectiveValue: "Creates picks before objectives start. Hook priority targets during teamfights. Gorge can secure shrine kills or remove defenders.",

    draftStrategy: "Strong first-pick tank. Forces enemy to draft mobile heroes or cleanse. Ban Tracer/Genji if showing Stitches early.",
    pickTiming: "early",

    skillFloor: "medium",
    skillCeiling: "high"
  },

  "Muradin": {
    hero: "Muradin",
    role: "Tank",
    strengths: [
      "High survivability with trait",
      "Strong engage and peel",
      "Avatar makes him nearly unkillable",
      "Versatile talent builds"
    ],
    weaknesses: [
      "Percentage damage negates trait",
      "Limited kill pressure",
      "Can be kited",
      "Trait interrupted by any damage"
    ],
    playstyle: "Durable tank with strong engage. Jump backline, stun priority targets, then jump out to heal. Use terrain for Storm Bolt angles.",

    counters: ["Tychus", "Malthael", "Leoric"],
    strongAgainst: ["Dive assassins", "Immobile mages"],

    bestMaps: ["Infernal Shrines", "Battlefield of Eternity", "Tomb of the Spider Queen"],
    worstMaps: ["Warhead Junction", "Garden of Terror"],
    mapStrategy: "Excels on small maps with tight chokes. Struggles on large maps where poke and globals have more value.",

    positioning: "Frontline tank. Use Dwarf Toss to engage or escape. Position to land Storm Bolt on multiple targets. Kite back when low to trigger trait.",
    objectiveValue: "Strong at contesting objectives. Avatar timing for key teamfights. Thunder Clap for zone control.",

    draftStrategy: "Safe first-pick tank. Flexible into most comps. Consider Avatar vs burst, Haymaker vs immobile backline.",
    pickTiming: "early",

    skillFloor: "low",
    skillCeiling: "medium"
  },

  // === BRUISERS ===

  "Sonya": {
    hero: "Sonya",
    role: "Bruiser",
    strengths: [
      "Strong solo laner",
      "Self-sustain through damage",
      "High sustained damage",
      "Camp clear and macro pressure"
    ],
    weaknesses: [
      "No hard engage",
      "Vulnerable to CC chains",
      "Needs to deal damage to heal",
      "Weak vs percentage damage"
    ],
    playstyle: "Solo lane bruiser focused on sustain and pressure. Spin to waveclear and heal. Spear for engage/disengage. Wrath of the Berserker to become unkillable in fights.",

    counters: ["Leoric", "Malthael", "Hard CC chains"],
    strongAgainst: ["Artanis", "Chen", "Other melee bruisers"],

    bestMaps: ["Braxis Holdout", "Dragon Shire", "Volskaya Foundry"],
    worstMaps: ["Cursed Hollow", "Sky Temple"],
    mapStrategy: "Dominant on maps with solo lane importance. Struggles on large maps where rotations matter more than 1v1 pressure.",

    positioning: "Solo lane or offlane. In teamfights, dive backline when Wrath is active. Use Spear to engage or escape.",
    objectiveValue: "Strong at camps and wave pressure. Wrath for objective teamfights. Can solo bosses mid-late game.",

    draftStrategy: "Show in solo lane phase of draft. Punishes weak solo laners. Ban Leoric/Malthael if Sonya is priority.",
    pickTiming: "mid",

    skillFloor: "medium",
    skillCeiling: "high"
  },

  // === HEALERS ===

  "Anduin": {
    hero: "Anduin",
    role: "Healer",
    strengths: [
      "Pull can save allies or setup kills",
      "Root enables combos",
      "Lightbomb is versatile",
      "Flash Heal for burst healing"
    ],
    weaknesses: [
      "Limited AOE healing",
      "Vulnerable to dive",
      "Mana intensive",
      "Root is skillshot"
    ],
    playstyle: "Setup healer with strong peel. Use Pull to save allies or position enemies. Root to enable combos. Lightbomb for engage, disengage, or burst protection.",

    abilities: [
      {
        key: "E",
        name: "Chastise",
        description: "Root enemies in a line",
        cooldown: 10,
        synergies: ["Stitches Hook", "Diablo Flip", "Kerrigan Combo"],
        counters: ["Cleanse", "Unstoppable"]
      },
      {
        key: "D",
        name: "Leap of Faith",
        description: "Pull ally to Anduin",
        cooldown: 70,
        synergies: ["Saving overextended allies", "Pulling divers away"],
      },
      {
        key: "R2",
        name: "Lightbomb",
        description: "Shield ally, AOE damage and knockback",
        cooldown: 60,
        synergies: ["Stitches Gorge", "Illidan Dive", "Muradin Jump"],
        counters: ["Spread formation", "Bait then disengage"]
      }
    ],

    counters: ["Illidan", "Tracer", "Zeratul"],
    strongAgainst: ["Stitches combos", "Diablo setups", "Melee-heavy comps"],

    bestMaps: ["Infernal Shrines", "Tomb of the Spider Queen", "Alterac Pass"],
    worstMaps: ["Warhead Junction", "Cursed Hollow"],
    mapStrategy: "Strong on maps with frequent teamfights. Pull and Root are less valuable on large maps with split objectives.",

    positioning: "Backline healer. Stay near tank for Pull saves. Position to hit Root on multiple enemies. Lightbomb for peel or engage.",
    objectiveValue: "Root locks down objective guardians. Pull saves allies from dangerous positions. Lightbomb for teamfight impact.",

    draftStrategy: "Pair with playmaker tanks (Stitches, Diablo). Pick after seeing enemy dive threats. Consider Uther if facing heavy dive.",
    pickTiming: "mid",

    skillFloor: "medium",
    skillCeiling: "high"
  },

  "Rehgar": {
    hero: "Rehgar",
    role: "Healer",
    strengths: [
      "Ancestral Healing saves allies",
      "Wolf form mobility",
      "Bloodlust for sustained fights",
      "Strong waveclear"
    ],
    weaknesses: [
      "Limited peel",
      "Ancestral can be interrupted",
      "Weak vs burst damage",
      "No hard CC"
    ],
    playstyle: "Mobile healer with game-changing ultimate. Use Wolf form for rotations and bite trades. Chain Heal for AOE healing. Time Ancestral to save allies from lethal damage.",

    counters: ["Burst damage", "CC during Ancestral cast", "Poke comps"],
    strongAgainst: ["Sustained damage", "Melee comps", "Low CC teams"],

    bestMaps: ["Dragon Shire", "Sky Temple", "Battlefield of Eternity"],
    worstMaps: ["Volskaya Foundry", "Braxis Holdout"],
    mapStrategy: "Wolf form enables fast rotations on multi-lane maps. Struggles when team needs constant peel.",

    positioning: "Mid-range healer. Wolf form to rotate or chase. Position to Chain Heal multiple allies. Save Ancestral for lethal damage.",
    objectiveValue: "Bloodlust during objective teamfights. Ancestral to save carries. Lightning Shield on divers.",

    draftStrategy: "Pick with auto-attack carries or melee-heavy comps. Bloodlust synergizes with sustained damage. Ancestral vs burst.",
    pickTiming: "mid",

    skillFloor: "low",
    skillCeiling: "high"
  },

  // === RANGED ASSASSINS ===

  "Nazeebo": {
    hero: "Nazeebo",
    role: "Ranged Assassin",
    strengths: [
      "Voodoo Ritual quest scaling",
      "Strong sustained damage",
      "Zone control with Zombie Wall",
      "Excellent waveclear"
    ],
    weaknesses: [
      "Immobile",
      "Vulnerable to dive",
      "Weak early game",
      "Relies on hitting skillshots"
    ],
    playstyle: "Late-game hypercarry. Focus on completing Voodoo Ritual quest early. Use spiders and toads for poke. Zombie Wall for peel or trap. Ravenous Spirit for sustained damage in fights.",

    counters: ["Illidan", "Kerrigan", "Zeratul", "Tracer"],
    strongAgainst: ["Immobile frontlines", "Low mobility backlines", "Tank-heavy comps"],

    bestMaps: ["Cursed Hollow", "Infernal Shrines", "Sky Temple"],
    worstMaps: ["Braxis Holdout", "Hanamura Temple"],
    mapStrategy: "Excels on large maps with long laning phases for quest stacking. Struggles on small maps against aggressive solo laners.",

    positioning: "Backline mage. Maintain distance from dive threats. Use Zombie Wall defensively. Stack quest on lanes and mercs.",
    objectiveValue: "Zone control with Zombie Wall. Ravenous Spirit on stationary objectives. Focus on stacking until level 20.",

    draftStrategy: "Last-pick into safe comps. Ban dive threats. Pair with peel supports (Anduin, Uther). Avoid blind-picking.",
    pickTiming: "late",

    skillFloor: "medium",
    skillCeiling: "high"
  },

  "Jaina": {
    hero: "Jaina",
    role: "Ranged Assassin",
    strengths: [
      "High burst damage",
      "AOE slow with Chill",
      "Ring of Frost lockdown",
      "Strong waveclear"
    ],
    weaknesses: [
      "No mobility",
      "Squishy",
      "High mana cost",
      "Vulnerable to dive"
    ],
    playstyle: "Burst mage with powerful AOE. Apply Chill to slow enemies. Blizzard for zone control. Frostbolt for poke. Cone of Cold for close-range burst. Ring of Frost for setup.",

    counters: ["Illidan", "Genji", "Tracer", "Zeratul"],
    strongAgainst: ["Immobile heroes", "Grouped enemies", "Melee-heavy comps"],

    bestMaps: ["Infernal Shrines", "Tomb of the Spider Queen", "Alterac Pass"],
    worstMaps: ["Warhead Junction", "Garden of Terror"],
    mapStrategy: "Strong on small maps with choke points. AOE shines in tight teamfights. Struggles when enemies can split push.",

    positioning: "Max range backline. Use Chill slow to kite. Save Ice Block for dive. Position for multi-target Blizzard.",
    objectiveValue: "AOE damage on shrine guardians. Ring of Frost for objective teamfights. Zone control with Blizzard.",

    draftStrategy: "Pick with strong peel. Avoid vs heavy dive. Ice Block talent is mandatory vs dive. Ring of Frost vs grouped comps.",
    pickTiming: "mid",

    skillFloor: "medium",
    skillCeiling: "high"
  },

  // === MELEE ASSASSINS ===

  "Illidan": {
    hero: "Illidan",
    role: "Melee Assassin",
    strengths: [
      "High mobility with Dive and Sweeping Strike",
      "Evasion negates auto-attacks",
      "The Hunt for global pressure",
      "Metamorphosis for survivability"
    ],
    weaknesses: [
      "Vulnerable to CC",
      "Weak vs mages and skillshots",
      "Requires healing support",
      "Low range"
    ],
    playstyle: "Hyper-mobile melee carry. Dive backline, use Evasion vs auto-attackers. Hunt for picks on split-push or low targets. Requires team to peel for him and enable dives.",

    counters: ["Mages", "Hard CC", "Blinds", "Polymorph"],
    strongAgainst: ["Ranged auto-attackers", "Low mobility backlines", "Squishy heroes"],

    bestMaps: ["Dragon Shire", "Cursed Hollow", "Sky Temple"],
    worstMaps: ["Infernal Shrines", "Tomb of the Spider Queen"],
    mapStrategy: "The Hunt enables cross-map pressure. Strong on large maps. Struggles in tight teamfights against mage poke.",

    positioning: "Backline diver. Wait for enemy cooldowns before diving. Evasion to negate auto-attack damage. Metamorphosis to survive burst.",
    objectiveValue: "Hunt to pressure split-pushers. Dive backline during objective fights. Requires team to engage first.",

    draftStrategy: "Pick with double support or Abathur. Requires draft built around him. Ban Artanis/Cassia/Johanna. Pair with Rehgar/Uther.",
    pickTiming: "mid",

    skillFloor: "high",
    skillCeiling: "high"
  },

  "Zeratul": {
    hero: "Zeratul",
    role: "Melee Assassin",
    strengths: [
      "Permanent stealth",
      "Void Prison for setup/disengage",
      "High burst damage",
      "Blink for mobility"
    ],
    weaknesses: [
      "Squishy",
      "Skill-dependent",
      "Weak waveclear",
      "Long cooldowns"
    ],
    playstyle: "Stealth assassin focused on picks. Use stealth to position. Burst combo: Singularity Spike → auto → Cleave → Blink out. Void Prison for fight control or saving allies.",

    counters: ["Reveal mechanics", "AOE damage", "Blinds"],
    strongAgainst: ["Backline mages", "Healers", "Squishy heroes"],

    bestMaps: ["Tomb of the Spider Queen", "Infernal Shrines", "Alterac Pass"],
    worstMaps: ["Warhead Junction", "Cursed Hollow"],
    mapStrategy: "Strong on small maps with frequent ganks. Struggles on large maps where waveclear and macro matter more.",

    positioning: "Flanker. Use stealth to find isolated targets. Wormhole to escape after burst. Void Prison to isolate enemies or save team.",
    objectiveValue: "Pick off low targets near objectives. Void Prison for setup or disengage. Stealth ganks on rotations.",

    draftStrategy: "Last pick vs squishy backlines. Weak vs reveal or tanky comps. Void Prison requires team coordination.",
    pickTiming: "late",

    skillFloor: "high",
    skillCeiling: "high"
  },

  "Kerrigan": {
    hero: "Kerrigan",
    role: "Melee Assassin",
    strengths: [
      "Combo burst damage",
      "Shield sustain from damage",
      "Strong at camps and bosses",
      "Ultralisk adds pressure"
    ],
    weaknesses: [
      "Combo reliant",
      "No escape",
      "Vulnerable to CC",
      "Weak vs tanky comps"
    ],
    playstyle: "Combo assassin. Land Ravage → Primal Grasp → Impaling Blades for massive burst. Generate shields from damage. Ultralisk for frontline pressure.",

    counters: ["Tanks", "Cleanse", "Unstoppable effects"],
    strongAgainst: ["Immobile mages", "Squishy backlines", "Low CC teams"],

    bestMaps: ["Volskaya Foundry", "Dragon Shire", "Battlefield of Eternity"],
    worstMaps: ["Cursed Hollow", "Warhead Junction"],
    mapStrategy: "Strong on maps with camps and bosses. Struggles on large maps where poke is prevalent.",

    positioning: "Flanker or solo laner. Wait for CC to be used before engaging. Land combo on grouped or isolated targets.",
    objectiveValue: "Combo burst on objective. Strong at camps pre-objective. Ultralisk to zone enemies.",

    draftStrategy: "Pick with follow-up damage. Pair with CC (Stitches, ETC). Weak vs cleanse. Consider after seeing enemy comp.",
    pickTiming: "mid",

    skillFloor: "high",
    skillCeiling: "high"
  },

  // === SUPPORT ===

  "Abathur": {
    hero: "Abathur",
    role: "Support",
    strengths: [
      "Global presence with Symbiote",
      "Soaks XP safely",
      "Locust push pressure",
      "Clone high-value heroes"
    ],
    weaknesses: [
      "No teamfight presence",
      "Vulnerable to ganks",
      "Team plays 4v5",
      "Requires specific comps"
    ],
    playstyle: "Macro specialist. Body-soak lanes while Symbiote assists team. Push lanes with Locusts. Clone carries in teamfights. Ultimate Evolution for critical fights.",

    counters: ["Dive comps", "Global heroes", "Strong 4v5 teams"],
    strongAgainst: ["Poke comps", "Weak waveclear", "Low gank pressure"],

    bestMaps: ["Cursed Hollow", "Sky Temple", "Warhead Junction"],
    worstMaps: ["Infernal Shrines", "Tomb of the Spider Queen"],
    mapStrategy: "Excels on large multi-lane maps. Struggles on small maps with constant teamfights.",

    positioning: "Safe soak in bushes. Move between lanes for XP. Clone carries in teamfights. Backdoor pressure with Locusts.",
    objectiveValue: "Soak XP while team contests. Clone for objective teamfights. Locust nest for lane pressure.",

    draftStrategy: "Pick with strong 4-man. Requires hypercarry to clone. Draft strong waveclear and self-sufficient heroes.",
    pickTiming: "early",

    skillFloor: "high",
    skillCeiling: "high"
  }
}

/**
 * Get hero knowledge by hero name
 */
export function getHeroKnowledge(heroName: string): HeroKnowledge | undefined {
  return HERO_KNOWLEDGE[heroName]
}

/**
 * Get all heroes with knowledge entries
 */
export function getAllKnownHeroes(): string[] {
  return Object.keys(HERO_KNOWLEDGE)
}

/**
 * Check if hero has knowledge entry
 */
export function hasHeroKnowledge(heroName: string): boolean {
  return heroName in HERO_KNOWLEDGE
}
