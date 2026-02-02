/**
 * Map Knowledge Database
 *
 * Provides strategic information about each battleground including:
 * - Objective mechanics
 * - Macro strategy (laning, rotations, camps)
 * - Hero recommendations
 * - Map-specific tips
 *
 * This enables LLM commentary to explain WHY certain heroes work on specific maps
 * rather than just stating "you have high win rate here."
 */

export interface MapKnowledge {
  map: string

  // Basic info
  size: 'small' | 'medium' | 'large'
  laneCount: 2 | 3

  // Objective
  objective: string
  objectiveDescription: string
  objectiveStrategy: string
  objectiveTiming: string  // When objectives spawn

  // Macro strategy
  macro: {
    earlyGame: string   // Pre-level 10 strategy
    midGame: string     // Level 10-16 strategy
    lateGame: string    // Level 16+ strategy
  }

  // Camps
  campImportance: 'low' | 'medium' | 'high'
  campTiming: string  // When to take camps

  // Hero preferences
  goodHeroes: {
    category: string      // e.g., "Stackers", "Globals", "Sustain"
    heroes: string[]
    reason: string
  }[]

  badHeroes: {
    category: string
    heroes: string[]
    reason: string
  }[]

  // Key strategic tips
  tips: string[]
}

/**
 * Map knowledge database for all Storm League maps
 */
export const MAP_KNOWLEDGE: Record<string, MapKnowledge> = {
  "Cursed Hollow": {
    map: "Cursed Hollow",
    size: "large",
    laneCount: 3,

    objective: "Tributes",
    objectiveDescription: "Collect 3 tributes to curse enemy team. Curse prevents enemies from attacking, minions have 1 HP.",
    objectiveStrategy: "Secure tributes through teamfights. Curse enables free structure damage. Position for tribute spawns 30 seconds early.",
    objectiveTiming: "First tribute around 3:00. Spawns every 2:30-3:00 after collection.",

    macro: {
      earlyGame: "Focus on lane XP and quest stacking. Large map rewards global presence. Rotate to tribute spawns early.",
      midGame: "Group for tributes. Take camps before tribute spawns to pressure during curse. Control boss area.",
      lateGame: "Curse wins games. Wipe enemy team then curse to end. Boss is game-ending with curse."
    },

    campImportance: "medium",
    campTiming: "Take camps 30 seconds before tribute spawn. Siege giants push hard during curse.",

    goodHeroes: [
      {
        category: "Stackers",
        heroes: ["Nazeebo", "Azmodan", "Butcher"],
        reason: "Large map with long laning phase enables quest completion. Many minions to stack."
      },
      {
        category: "Globals",
        heroes: ["Falstad", "Dehaka", "Brightwing"],
        reason: "Global abilities provide rotations advantage on large 3-lane map."
      },
      {
        category: "Sustain",
        heroes: ["Zagara", "Xul", "Johanna"],
        reason: "Prolonged laning and teamfights reward sustain heroes."
      }
    ],

    badHeroes: [
      {
        category: "Short Range",
        heroes: ["Nova", "Valeera", "Samuro"],
        reason: "Large map makes ganks difficult. Tributes force teamfights where stealth is less valuable."
      }
    ],

    tips: [
      "Position near next tribute spawn 30 seconds early",
      "Take siege giants before tribute for curse push",
      "Boss after won teamfight + curse = game over",
      "Long lanes favor quest stacking heroes",
      "Save heroics for tribute teamfights"
    ]
  },

  "Infernal Shrines": {
    map: "Infernal Shrines",
    size: "small",
    laneCount: 3,

    objective: "Shrines",
    objectiveDescription: "Kill 40 guardians at shrine to summon Punisher. Punisher sieges lane and stuns heroes.",
    objectiveStrategy: "Burst damage and AOE clear guardians fast. Punisher is extremely strong - prioritize winning shrines.",
    objectiveTiming: "First shrine around 1:30. Spawns continuously.",

    macro: {
      earlyGame: "Rush to shrine. Waveclear and burst damage critical. Team with better clear wins early shrines.",
      midGame: "Group for shrines. Clear guardians fast then force fights. Punisher push is devastating.",
      lateGame: "Punisher + deathball ends game. Win shrine then push with Punisher."
    },

    campImportance: "low",
    campTiming: "Camps are low value compared to shrines. Only take if shrine is down.",

    goodHeroes: [
      {
        category: "AOE Burst",
        heroes: ["Jaina", "Kael'thas", "Gul'dan"],
        reason: "Fast guardian clear wins shrines. AOE burst damage is king."
      },
      {
        category: "Sustained DPS",
        heroes: ["Raynor", "Valla", "Greymane"],
        reason: "Consistent damage to clear guardians and burn Punisher."
      },
      {
        category: "Frontline",
        heroes: ["Johanna", "Muradin", "Blaze"],
        reason: "Tank shrine guardians while team clears. Punisher stuns make tanks valuable."
      }
    ],

    badHeroes: [
      {
        category: "Weak Waveclear",
        heroes: ["Nova", "Zeratul", "Illidan"],
        reason: "Can't clear guardians efficiently. Team will lose shrines."
      },
      {
        category: "Globals/Split-push",
        heroes: ["Falstad", "Zagara", "Azmodan"],
        reason: "Small map with constant teamfights. Globals less valuable, can't split-push."
      }
    ],

    tips: [
      "Draft AOE damage for guardian clear",
      "Punisher targets closest hero - tank controls it",
      "Top and bot shrines more dangerous (closer to keeps)",
      "Focus Punisher after lost shrine to minimize damage",
      "Ignore camps - shrines are everything"
    ]
  },

  "Braxis Holdout": {
    map: "Braxis Holdout",
    size: "small",
    laneCount: 2,

    objective: "Beacons",
    objectiveDescription: "Capture and hold top/bottom beacons. 100% charge spawns Zerg wave that pushes lane.",
    objectiveStrategy: "Win both solo lanes to control beacons. Zerg waves are devastating - prioritize beacon control.",
    objectiveTiming: "Beacons activate around 2:00. Stay active until wave spawns.",

    macro: {
      earlyGame: "Solo lane matchup determines everything. Win lanes = win beacons. Gank losing lane.",
      midGame: "Rotate between lanes to support weak side. 4-man can overwhelm solo laner.",
      lateGame: "100% Zerg wave ends game. Focus on winning one lane completely."
    },

    campImportance: "medium",
    campTiming: "Take bruiser camp after winning beacon to double push pressure.",

    goodHeroes: [
      {
        category: "Solo Laners",
        heroes: ["Sonya", "Dehaka", "Malthael", "Yrel"],
        reason: "Strong 1v1 matchup wins beacon control. Solo lane dominance is critical."
      },
      {
        category: "Waveclear",
        heroes: ["Xul", "Jaina", "Gul'dan"],
        reason: "Clear massive Zerg waves to defend. Poor waveclear loses to Zerg."
      },
      {
        category: "4-man Roam",
        heroes: ["ETC", "Stitches", "Arthas"],
        reason: "Strong 4-man can support weak solo lane. Ganks determine beacon control."
      }
    ],

    badHeroes: [
      {
        category: "Weak Solo",
        heroes: ["Valla", "Jaina", "Kael'thas"],
        reason: "Can't solo lane. Team forced into unfavorable compositions."
      },
      {
        category: "Stackers",
        heroes: ["Nazeebo", "Azmodan"],
        reason: "Can't stack safely vs strong solo laners. Will feed and lose beacon."
      }
    ],

    tips: [
      "Solo lane winner controls game",
      "4-man should gank losing lane repeatedly",
      "Zerg wave + bruiser camp = unstoppable push",
      "Waveclear is mandatory on this map",
      "100% Zerg ends game - play around beacon percentages"
    ]
  },

  "Dragon Shire": {
    map: "Dragon Shire",
    size: "medium",
    laneCount: 3,

    objective: "Dragon Knight",
    objectiveDescription: "Capture top and bot shrines simultaneously. Control both to activate Dragon Knight in mid.",
    objectiveStrategy: "Win solo lane top/bot. 4-man controls other shrine. Coordinate to cap both simultaneously.",
    objectiveTiming: "Shrines activate around 1:45. Reset if not both captured.",

    macro: {
      earlyGame: "Win top or bot solo lane. 4-man controls opposite shrine. Rotate to cap both at once.",
      midGame: "Dragon Knight is strong - protect pilot. Enemies will try to kick pilot out.",
      lateGame: "Late-game Dragon Knight can end. Focus on getting DK then pushing core."
    },

    campImportance: "medium",
    campTiming: "Take knights when enemies have Dragon Knight to pressure other lanes.",

    goodHeroes: [
      {
        category: "Solo Laners",
        heroes: ["Sonya", "Dehaka", "Zagara"],
        reason: "Win solo lane to secure shrine control. Macro pressure in solo lane critical."
      },
      {
        category: "Sustain",
        heroes: ["Johanna", "Arthas", "Thrall"],
        reason: "Hold shrines against enemy team. Sustain enables shrine captures."
      },
      {
        category: "DK Pilot",
        heroes: ["Tanks"],
        reason: "Tanks pilot Dragon Knight safely. Hard to kick out of DK."
      }
    ],

    badHeroes: [
      {
        category: "No Solo",
        heroes: ["Murky", "Abathur"],
        reason: "Weak solo lane matchups lose shrine control early."
      }
    ],

    tips: [
      "Coordinate shrine caps - need both simultaneously",
      "Tank pilots Dragon Knight for survivability",
      "Kick enemy out of DK with stuns/displacements",
      "Mid lane rotation is key to shrine control",
      "Dragon Knight gets stronger late game"
    ]
  },

  "Sky Temple": {
    map: "Sky Temple",
    size: "large",
    laneCount: 3,

    objective: "Temples",
    objectiveDescription: "Capture temples to fire lasers at enemy structures. Up to 3 temples active at once.",
    objectiveStrategy: "Split team to capture multiple temples. Temples directly damage structures - prioritize control.",
    objectiveTiming: "First temples around 2:00. Rotate spawn locations.",

    macro: {
      earlyGame: "Soak all lanes. Prepare to split for temples. Vision control around temple spawns.",
      midGame: "Split efficiently to cap multiple temples. Trade temples based on position. Push with temples.",
      lateGame: "Single temple can end game. Coordinate to cap one temple fully then rotate."
    },

    campImportance: "medium",
    campTiming: "Take siege giants when temples are active on other side of map.",

    goodHeroes: [
      {
        category: "Globals",
        heroes: ["Falstad", "Dehaka", "Brightwing"],
        reason: "Global presence enables temple rotations. Can assist multiple temples."
      },
      {
        category: "Split-push",
        heroes: ["Zagara", "Azmodan", "Sylvanas"],
        reason: "Force enemies to defend lanes while team takes temples."
      },
      {
        category: "Poke",
        heroes: ["Chromie", "Junkrat", "Hanzo"],
        reason: "Poke enemies off temples from range. Large map rewards long-range poke."
      }
    ],

    badHeroes: [
      {
        category: "Melee Dive",
        heroes: ["Illidan", "Kerrigan", "Butcher"],
        reason: "Large map makes dive risky. Hard to create picks when enemies can split."
      }
    ],

    tips: [
      "Split team to contest multiple temples",
      "Trading temples is okay - focus on damage dealt",
      "Bottom temple is most dangerous (near core)",
      "Siege giants push while enemies at temples",
      "Late game temples end games fast"
    ]
  },

  "Battlefield of Eternity": {
    map: "Battlefield of Eternity",
    size: "small",
    laneCount: 2,

    objective: "Immortals",
    objectiveDescription: "Kill enemy Immortal faster than enemies kill yours. Winner's Immortal pushes a lane.",
    objectiveStrategy: "Race to kill enemy Immortal. Heroes with sustained DPS are critical. Immortal shield protects it.",
    objectiveTiming: "First Immortals around 2:30. Continuous spawns.",

    macro: {
      earlyGame: "Soak XP. Draft sustained damage for Immortal race. Vision control center area.",
      midGame: "Race Immortals. Can disrupt enemy team but focus is DPS race. Immortal push is powerful.",
      lateGame: "Immortal + team ends game. Winning Immortal race critical."
    },

    campImportance: "low",
    campTiming: "Ignore camps. Immortal race is everything.",

    goodHeroes: [
      {
        category: "Sustained DPS",
        heroes: ["Raynor", "Valla", "Greymane", "Tychus"],
        reason: "Burn Immortal fast. DPS race determines winner. Percentage damage shreds shields."
      },
      {
        category: "Burst",
        heroes: ["Jaina", "Kael'thas", "Li-Ming"],
        reason: "Burst down shields. AOE hits Immortal and enemies."
      },
      {
        category: "Frontline",
        heroes: ["Muradin", "Arthas", "ETC"],
        reason: "Body-block enemy DPS from Immortal. Peel for your DPS."
      }
    ],

    badHeroes: [
      {
        category: "Low DPS",
        heroes: ["Medivh", "Abathur", "Murky"],
        reason: "Can't contribute to Immortal race. Team will lose objective."
      },
      {
        category: "Globals/Split-push",
        heroes: ["Falstad", "Dehaka", "Azmodan"],
        reason: "Small map. Can't split-push during Immortal phase."
      }
    ],

    tips: [
      "Draft 2-3 sustained DPS heroes",
      "Body-block enemy team from Immortal",
      "Percentage damage (Tychus, Malthael) melts shields",
      "Protect own Immortal push with team",
      "Late game Immortals end games instantly"
    ]
  },

  "Tomb of the Spider Queen": {
    map: "Tomb of the Spider Queen",
    size: "small",
    laneCount: 3,

    objective: "Gems",
    objectiveDescription: "Collect gems from minion kills. Turn in 50 gems to fire web blasts at enemy structures.",
    objectiveStrategy: "Farm gems from lanes. Turn in at altars safely. Deny enemy turn-ins. Gems drop on death - forcing trades is key.",
    objectiveTiming: "Continuous - gems always available. Altars in each lane.",

    macro: {
      earlyGame: "Farm gems. Rotate to turn in with advantage. Deny enemy turn-ins. Trading kills wins gems.",
      midGame: "Web blasts are powerful. Force turn-ins when ahead. Gank high-gem targets.",
      lateGame: "Web blasts + push ends game. Coordinate turn-ins after won fights."
    },

    campImportance: "low",
    campTiming: "Only take when safe. Dying with gems loses objective.",

    goodHeroes: [
      {
        category: "Waveclear",
        heroes: ["Xul", "Jaina", "Gul'dan"],
        reason: "Fast waveclear = more gems. Small map enables constant lane clear."
      },
      {
        category: "Gankers",
        heroes: ["Zeratul", "Nova", "Kerrigan"],
        reason: "Kill high-gem targets. Small map enables frequent ganks."
      },
      {
        category: "Sustain",
        heroes: ["Johanna", "Thrall", "Sonya"],
        reason: "Stay healthy to turn in gems safely. Sustain prevents backing."
      }
    ],

    badHeroes: [
      {
        category: "Globals",
        heroes: ["Falstad", "Dehaka", "Brightwing"],
        reason: "Small map makes globals less valuable. Better to have strong teamfight."
      },
      {
        category: "Poor Waveclear",
        heroes: ["Nova", "Illidan"],
        reason: "Can't farm gems efficiently. Team will fall behind on turn-ins."
      }
    ],

    tips: [
      "Clear lanes constantly to farm gems",
      "Turn in with team protection",
      "Gank heroes with most gems",
      "Gems drop on death - force trades when behind",
      "Web blasts pressure all lanes simultaneously"
    ]
  },

  "Volskaya Foundry": {
    map: "Volskaya Foundry",
    size: "medium",
    laneCount: 2,

    objective: "Gunner Protector",
    objectiveDescription: "Capture point to control Protector. Gunner shoots, pilot uses abilities. Protector is extremely powerful.",
    objectiveStrategy: "Win teamfight at point. Coordinate gunner and pilot. Protector can solo keeps.",
    objectiveTiming: "First Protector around 2:15. Spawns after previous destroyed.",

    macro: {
      earlyGame: "Soak lanes. Prepare for point fight. Vision control critical.",
      midGame: "Win teamfight for Protector. Protect pilot. Protector push wins forts.",
      lateGame: "Protector can end game. Coordinate pilot + gunner + team push."
    },

    campImportance: "high",
    campTiming: "Turret camps are strong. Take before Protector to double pressure.",

    goodHeroes: [
      {
        category: "Solo Laners",
        heroes: ["Sonya", "Dehaka", "Malthael"],
        reason: "Win solo lane while 4-man controls point. Dual pilot for Protector."
      },
      {
        category: "Teamfight",
        heroes: ["ETC", "Malfurion", "Jaina"],
        reason: "Win point teamfight. Teamfight winners control Protector."
      },
      {
        category: "Pilot",
        heroes: ["Tanks", "Bruisers"],
        reason: "Durable pilots survive longer. Pilot abilities are impactful."
      }
    ],

    badHeroes: [
      {
        category: "Weak Solo",
        heroes: ["Valla", "Li-Ming"],
        reason: "Can't solo lane on 2-lane map. Forced into 4-man."
      }
    ],

    tips: [
      "Win point teamfight for Protector",
      "Tank pilots for survivability",
      "Gunner focuses structures > heroes",
      "Turret camps + Protector = unstoppable",
      "Coordinate pilot abilities with gunner damage"
    ]
  },

  "Towers of Doom": {
    map: "Towers of Doom",
    size: "medium",
    laneCount: 3,

    objective: "Altars",
    objectiveDescription: "Capture altars to fire your team's towers at enemy core. Core cannot be attacked directly - only via towers.",
    objectiveStrategy: "Control altars through teamfights. Sappers can capture enemy towers. Core damage only through objectives.",
    objectiveTiming: "Altars spawn regularly. Top and bottom alternate with center.",

    macro: {
      earlyGame: "Soak lanes. Position for altar spawns. Sappers push lanes.",
      midGame: "Win altars for tower shots. Control enemy towers with sappers. Each tower shot matters.",
      lateGame: "Altars directly determine winner. Final altars end game."
    },

    campImportance: "high",
    campTiming: "Sappers are critical - they can capture enemy towers. Take before altars.",

    goodHeroes: [
      {
        category: "Teamfight",
        heroes: ["ETC", "Malfurion", "Valla"],
        reason: "Altars won through teamfights. Teamfight comps dominate."
      },
      {
        category: "Sapper Clear",
        heroes: ["Xul", "Johanna", "Jaina"],
        reason: "Clear enemy sappers before they cap towers. Waveclear critical."
      },
      {
        category: "Sustain",
        heroes: ["Thrall", "Rehgar", "Zagara"],
        reason: "Prolonged fights at altars. Sustain enables altar control."
      }
    ],

    badHeroes: [
      {
        category: "Structure Damage",
        heroes: ["Azmodan", "Sylvanas"],
        reason: "Cannot attack core directly. Structure damage less valuable."
      }
    ],

    tips: [
      "Core cannot be attacked - only via altars",
      "Sappers can capture enemy towers",
      "Clear enemy sappers immediately",
      "Control own towers near enemy base",
      "Each tower shot is precious - win altars"
    ]
  }
}

/**
 * Get map knowledge by map name
 */
export function getMapKnowledge(mapName: string): MapKnowledge | undefined {
  return MAP_KNOWLEDGE[mapName]
}

/**
 * Get all maps with knowledge entries
 */
export function getAllKnownMaps(): string[] {
  return Object.keys(MAP_KNOWLEDGE)
}

/**
 * Check if map has knowledge entry
 */
export function hasMapKnowledge(mapName: string): boolean {
  return mapName in MAP_KNOWLEDGE
}

/**
 * Get map knowledge for why a hero works on a map
 */
export function getHeroMapSynergy(heroName: string, mapName: string): string | null {
  const mapKnowledge = getMapKnowledge(mapName)
  if (!mapKnowledge) return null

  // Check if hero is in goodHeroes
  for (const category of mapKnowledge.goodHeroes) {
    if (category.heroes.includes(heroName)) {
      return `${category.category}: ${category.reason}`
    }
  }

  // Check if hero is in badHeroes
  for (const category of mapKnowledge.badHeroes) {
    if (category.heroes.includes(heroName)) {
      return `${category.category}: ${category.reason}`
    }
  }

  return null
}
