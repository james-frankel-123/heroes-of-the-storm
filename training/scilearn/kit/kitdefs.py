"""Shared definitions for the kit table: axes, ranges and fixed team-identity weights."""

AXES = ["hard_cc", "soft_cc_peel", "healing_throughput", "self_sustain_shields", "burst",
        "sustained_damage", "range", "mobility", "waveclear", "siege", "global_presence", "early_vs_late"]

AXIS_RANGE = {a: (0, 4) for a in AXES}
AXIS_RANGE["early_vs_late"] = (-2, 2)

AXIS_DEFINITIONS = {
    "hard_cc": "Stun, root, silence, sleep, polymorph, fear, taunt, time stop, knock-up, knockback, pull, grab.",
    "soft_cc_peel": "Slows, blinds, attack-speed reductions, area denial, and ally-protection tools that are not healing "
                    "(cleanse, unstoppable or invulnerability on an ally, moving allies, displacing attackers).",
    "healing_throughput": "Healing and shields given to allies.",
    "self_sustain_shields": "Self-healing, lifesteal, self-shields, armor, damage reduction, self invulnerability, extra lives.",
    "burst": "Damage deliverable in about 2 s (combos, empowered attacks, executes, percent-health damage).",
    "sustained_damage": "Damage per second kept up over about 10 s (basic attack DPS, repeatable abilities, DoTs, summons).",
    "range": "Distance from which most damage and effects are applied (0 melee, 4 very long range).",
    "mobility": "Dashes, blinks, teleports, leaps, flight, speed boosts, unstoppable movement, repositioning stealth.",
    "waveclear": "Speed of killing minion waves (area, piercing, bonus vs non-heroes, minion-fighting summons).",
    "siege": "Damaging and pressuring structures (pushing summons, out-ranging towers, structure bonuses, disables).",
    "global_presence": "Acting across the map (global teleports, map-wide abilities or vision, remote soaking, fast travel).",
    "early_vs_late": "Power timing from mechanics only: -2 early (high base, no scaling) to +2 late (quests, stacking, high-level gates).",
}

# Fixed team identities: per-hero score = sum(weight * axis value) on the 0-4 axes; a team's
# identity is the mean of its five heroes' scores. Weights are fixed and never fit.
IDENTITY_WEIGHTS = {
    "poke": {"range": 0.50, "burst": 0.30, "sustained_damage": 0.20},
    "dive": {"mobility": 0.35, "hard_cc": 0.25, "burst": 0.25, "self_sustain_shields": 0.15},
    "sustain": {"healing_throughput": 0.50, "self_sustain_shields": 0.30, "soft_cc_peel": 0.20},
    "push": {"siege": 0.40, "waveclear": 0.35, "global_presence": 0.25},
}

SD_FLAG = 1.0
