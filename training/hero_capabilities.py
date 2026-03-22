"""
Per-hero capability counts for enriched WP model features.

Each hero gets a count (0, 1, 2, 3) for each capability dimension,
representing baseline kit + key talent upgrades that provide the capability.

DRAFT — needs human audit before use in training.

Dimensions:
  stuns:          hard stuns (0.5s+ stun, not mini-stuns)
  cleanses:       cleanse/unstoppable effects (remove/prevent CC)
  self_protect:   self-only damage prevention (ice block, evasion, deflect, stasis)
  ally_protect:   protect ALLIES (divine shield, force of will, barriers on others)
  roots:          root effects (immobilize without stun)
  blinds:         blind effects (miss auto-attacks)

Proposed additional dimensions:
  silences:       prevent ability use
  displacement:   knockback/pull/swap (positional disruption)
  global:         global presence (mount, teleport, global ability)
  waveclear:      PvE waveclear strength (0=poor, 1=ok, 2=good, 3=excellent)
  burst:          burst damage potential (0=low, 1=moderate, 2=high, 3=extreme)
  sustain_dmg:    sustained damage output
  self_sustain:   self-healing / survivability without healer
  percent_damage:  % max HP damage (tank busters)
"""

# Counts represent: baseline kit abilities + commonly picked talents
# 0 = none, 1 = one source, 2 = two sources or very reliable, 3 = exceptional/multiple
HERO_CAPABILITIES = {
    # ── Tanks ──
    "Anub'arak":    {"stuns": 2, "cleanses": 0, "self_protect": 1, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 2, "dive": 2},
    "Arthas":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Blaze":        {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 1, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 1, "dive": 1},
    "Cho":          {"stuns": 1, "cleanses": 0, "self_protect": 1, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 2, "self_sustain": 2, "percent_damage": 0, "mobility": 2, "dive": 2},
    "Diablo":       {"stuns": 2, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 2, "global": 0, "waveclear": 1, "burst": 2, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 2},
    "E.T.C.":       {"stuns": 2, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 1, "waveclear": 1, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 2},
    "Garrosh":      {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 0, "burst": 1, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Johanna":      {"stuns": 1, "cleanses": 1, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 1, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 0, "sustain_dmg": 0, "self_sustain": 2, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Mal'Ganis":    {"stuns": 2, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 1, "displacement": 0, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 1, "self_sustain": 3, "percent_damage": 0, "mobility": 1, "dive": 1},
    "Mei":          {"stuns": 1, "cleanses": 0, "self_protect": 1, "ally_protect": 0, "roots": 0, "blinds": 1, "silences": 0, "displacement": 1, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 0, "self_sustain": 2, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Muradin":      {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 0, "self_sustain": 2, "percent_damage": 0, "mobility": 2, "dive": 2},
    "Stitches":     {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 2, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Tyrael":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 1, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 3, "dive": 2},

    # ── Bruisers ──
    "Artanis":      {"stuns": 0, "cleanses": 0, "self_protect": 1, "ally_protect": 0, "roots": 0, "blinds": 1, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 2, "self_sustain": 2, "percent_damage": 0, "mobility": 1, "dive": 1},
    "Chen":         {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 2, "dive": 2},
    "Deathwing":    {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 1, "waveclear": 2, "burst": 2, "sustain_dmg": 2, "self_sustain": 2, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Dehaka":       {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 1, "displacement": 1, "global": 2, "waveclear": 2, "burst": 1, "sustain_dmg": 2, "self_sustain": 2, "percent_damage": 0, "mobility": 2, "dive": 1},
    "D.Va":         {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 2, "dive": 1},
    "Gazlowe":      {"stuns": 2, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 3, "burst": 2, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Hogger":       {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 2, "dive": 1},
    "Imperius":     {"stuns": 1, "cleanses": 0, "self_protect": 1, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 1, "dive": 2},
    "Leoric":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 2, "self_sustain": 2, "percent_damage": 2, "mobility": 1, "dive": 0},
    "Malthael":     {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 1, "displacement": 0, "global": 0, "waveclear": 3, "burst": 1, "sustain_dmg": 2, "self_sustain": 3, "percent_damage": 2, "mobility": 2, "dive": 2},
    "Ragnaros":     {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 3, "burst": 2, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Rexxar":       {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 0, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Sonya":        {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 2, "self_sustain": 2, "percent_damage": 0, "mobility": 1, "dive": 1},
    "Thrall":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 2, "self_sustain": 2, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Xul":          {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 3, "burst": 1, "sustain_dmg": 1, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Yrel":         {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 0, "sustain_dmg": 0, "self_sustain": 2, "percent_damage": 0, "mobility": 1, "dive": 1},

    # ── Healers ──
    "Alexstrasza":  {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Ana":          {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 0, "burst": 0, "sustain_dmg": 0, "self_sustain": 0, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Anduin":       {"stuns": 1, "cleanses": 1, "self_protect": 0, "ally_protect": 1, "roots": 1, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 0, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Auriel":       {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 1, "roots": 0, "blinds": 1, "silences": 0, "displacement": 1, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Brightwing":   {"stuns": 0, "cleanses": 1, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 2, "waveclear": 1, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Deckard":      {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Kharazim":     {"stuns": 0, "cleanses": 1, "self_protect": 0, "ally_protect": 1, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 3, "dive": 2},
    "Li Li":        {"stuns": 0, "cleanses": 1, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 1, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Lt. Morales":  {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 1, "waveclear": 0, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Lúcio":        {"stuns": 0, "cleanses": 1, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 0, "self_sustain": 2, "percent_damage": 0, "mobility": 2, "dive": 0},
    "Malfurion":    {"stuns": 0, "cleanses": 1, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 1, "displacement": 0, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Rehgar":       {"stuns": 0, "cleanses": 1, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Stukov":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 1, "displacement": 1, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Tyrande":      {"stuns": 1, "cleanses": 1, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 1, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Uther":        {"stuns": 2, "cleanses": 1, "self_protect": 0, "ally_protect": 1, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 0, "burst": 1, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Whitemane":    {"stuns": 0, "cleanses": 1, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 1, "self_sustain": 2, "percent_damage": 0, "mobility": 0, "dive": 0},

    # ── Ranged AA ──
    "Cassia":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 1, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Falstad":      {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 3, "global": 2, "waveclear": 2, "burst": 2, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 1, "mobility": 2, "dive": 0},
    "Fenix":        {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Greymane":     {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 3, "sustain_dmg": 2, "self_sustain": 0, "percent_damage": 1, "mobility": 2, "dive": 1},
    "Hanzo":        {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 2, "self_sustain": 0, "percent_damage": 1, "mobility": 1, "dive": 0},
    "Lunara":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 0, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 2, "dive": 0},
    "Raynor":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 3, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Sgt. Hammer":  {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 3, "self_sustain": 1, "percent_damage": 1, "mobility": 0, "dive": 0},
    "Sylvanas":     {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 1, "displacement": 0, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 2, "self_sustain": 0, "percent_damage": 0, "mobility": 1, "dive": 2},
    "Tracer":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 2, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 3, "dive": 2},
    "Tychus":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 2, "mobility": 1, "dive": 0},
    "Valla":        {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 3, "self_sustain": 0, "percent_damage": 1, "mobility": 1, "dive": 0},
    "Zul'jin":      {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 3, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},

    # ── Ranged Mage ──
    "Chromie":      {"stuns": 0, "cleanses": 0, "self_protect": 1, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 3, "sustain_dmg": 2, "self_sustain": 0, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Gall":         {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 3, "sustain_dmg": 2, "self_sustain": 0, "percent_damage": 0, "mobility": 1, "dive": 1},
    "Genji":        {"stuns": 0, "cleanses": 0, "self_protect": 2, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 2, "sustain_dmg": 1, "self_sustain": 0, "percent_damage": 0, "mobility": 3, "dive": 3},
    "Gul'dan":      {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 3, "burst": 2, "sustain_dmg": 3, "self_sustain": 2, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Jaina":        {"stuns": 0, "cleanses": 0, "self_protect": 1, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 3, "burst": 3, "sustain_dmg": 1, "self_sustain": 0, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Junkrat":      {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 2, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 1, "self_sustain": 0, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Kael'thas":    {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 3, "burst": 3, "sustain_dmg": 1, "self_sustain": 0, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Kel'Thuzad":   {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 3, "sustain_dmg": 0, "self_sustain": 0, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Li-Ming":      {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 3, "sustain_dmg": 3, "self_sustain": 0, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Mephisto":     {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 1, "blinds": 0, "silences": 1, "displacement": 0, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 2},
    "Nova":         {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 0, "burst": 3, "sustain_dmg": 0, "self_sustain": 0, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Orphea":       {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 0},
    "Probius":      {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 3, "burst": 1, "sustain_dmg": 2, "self_sustain": 0, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Tassadar":     {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 3, "burst": 1, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},

    # ── Melee Assassins ──
    "Alarak":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 1, "displacement": 1, "global": 0, "waveclear": 1, "burst": 3, "sustain_dmg": 1, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 1},
    "Illidan":      {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 1, "waveclear": 2, "burst": 1, "sustain_dmg": 2, "self_sustain": 2, "percent_damage": 0, "mobility": 3, "dive": 3},
    "Kerrigan":     {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 1, "burst": 2, "sustain_dmg": 1, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 2},
    "Maiev":        {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 1, "self_sustain": 0, "percent_damage": 0, "mobility": 2, "dive": 2},
    "Qhira":        {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 2, "sustain_dmg": 1, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 2},
    "Samuro":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 2, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 3, "dive": 2},
    "The Butcher":  {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 1, "displacement": 0, "global": 0, "waveclear": 1, "burst": 3, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 2},
    "Valeera":      {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 1, "displacement": 0, "global": 0, "waveclear": 1, "burst": 3, "sustain_dmg": 1, "self_sustain": 0, "percent_damage": 0, "mobility": 1, "dive": 1},
    "Zeratul":      {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 3, "sustain_dmg": 1, "self_sustain": 0, "percent_damage": 0, "mobility": 3, "dive": 3},

    # ── Support / Utility ──
    "Abathur":      {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 1, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 3, "waveclear": 2, "burst": 0, "sustain_dmg": 1, "self_sustain": 0, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Medivh":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 2, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 1, "self_sustain": 0, "percent_damage": 0, "mobility": 2, "dive": 0},
    "Zarya":        {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 1, "roots": 0, "blinds": 0, "silences": 0, "displacement": 1, "global": 0, "waveclear": 1, "burst": 1, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},

    # ── Varian ──
    "Varian":       {"stuns": 1, "cleanses": 0, "self_protect": 2, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 1, "burst": 2, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 1, "dive": 2},

    # ── Pushers ──
    "Azmodan":      {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 3, "burst": 1, "sustain_dmg": 2, "self_sustain": 0, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Nazeebo":      {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 3, "burst": 1, "sustain_dmg": 2, "self_sustain": 1, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Zagara":       {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 1, "waveclear": 3, "burst": 1, "sustain_dmg": 2, "self_sustain": 0, "percent_damage": 0, "mobility": 0, "dive": 0},
    "Murky":        {"stuns": 0, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 0, "waveclear": 2, "burst": 1, "sustain_dmg": 1, "self_sustain": 3, "percent_damage": 0, "mobility": 1, "dive": 0},
    "The Lost Vikings": {"stuns": 1, "cleanses": 0, "self_protect": 0, "ally_protect": 0, "roots": 0, "blinds": 0, "silences": 0, "displacement": 0, "global": 2, "waveclear": 2, "burst": 0, "sustain_dmg": 0, "self_sustain": 1, "percent_damage": 0, "mobility": 3, "dive": 0},
}

# All capability dimension names
CAPABILITY_DIMS = [
    # CC and utility:
    "stuns", "cleanses", "self_protect", "ally_protect", "roots", "blinds",
    "silences", "displacement",
    # Mobility and engagement:
    "mobility", "dive",
    # Strategic:
    "global", "waveclear",
    # Damage profile:
    "burst", "sustain_dmg", "percent_damage",
    # Survivability:
    "self_sustain",
]

if __name__ == "__main__":
    # Verification
    import sys
    sys.path.insert(0, '.')
    from shared import HEROES

    missing = [h for h in HEROES if h not in HERO_CAPABILITIES]
    extra = [h for h in HERO_CAPABILITIES if h not in HEROES]
    print(f"Heroes: {len(HEROES)}, Mapped: {len(HERO_CAPABILITIES)}")
    if missing: print(f"MISSING: {missing}")
    if extra: print(f"EXTRA: {extra}")

    # Print summary table
    print(f"\n{'Hero':<22} {'stun':>4} {'clns':>4} {'sprt':>4} {'aprt':>4} {'root':>4} {'blnd':>4} {'slnc':>4} {'disp':>4} {'mobl':>4} {'dive':>4} {'glob':>4} {'wave':>4} {'brst':>4} {'sdmg':>4} {'%dmg':>4} {'self':>4}")
    print("-" * 80)
    for hero in HEROES:
        c = HERO_CAPABILITIES.get(hero, {})
        vals = [c.get(d, 0) for d in CAPABILITY_DIMS]
        print(f"{hero:<22} " + " ".join(f"{v:>4}" for v in vals))

    # Team totals sanity check
    print(f"\nSample team: Muradin, Brightwing, Valla, Sonya, Jaina")
    team = ["Muradin", "Brightwing", "Valla", "Sonya", "Jaina"]
    for dim in CAPABILITY_DIMS:
        total = sum(HERO_CAPABILITIES[h].get(dim, 0) for h in team)
        print(f"  {dim}: {total}")
