# Kit coding rubric (kit table v2.57)

You code what a hero's kit can do, read from its ability and talent text. You do not judge how good the hero is.

## Forbidden

- No judgment of strength, balance, win rate, pick rate, tier lists, the meta, popularity or how often anything is played.
- Do not try to identify the hero, and do not use anything you may know about a particular hero. The texts are anonymized on purpose. Code only from the text and from general game rules (minions, structures, mounts, mercenaries, levels).
- Do not compare heroes with each other by reputation. Two texts with the same mechanics get the same codes.

## How to read a text

- Ability slots are shown as tokens: [Q], [W], [E], [D] (trait), [R1] and [R2] (the two heroic choices, one is picked at level 10), [alt-Q] and similar (abilities of an alternate form or a second unit), [T4.2] (talent 2 at level 4). "term A", "summon A", "unit A" stand for named summons, forms or companions. "this hero" is the hero being coded; "another hero" is any other hero.
- "LATER PATCH CHANGES" update the text above them, oldest first. Where they conflict with the base text, the later change wins. A note that a hero was "fully reworked" replaces the earlier text.
- Numbers are level-1 values unless stated; "(+4% per level)" means the value grows about 4% per level. Basic attack lines give range (in game units; melee is about 1 to 2, typical ranged is 5.5), attack interval and damage.

## Scope rule

Code the hero at a typical mid-game state (levels 10 to 16) with a sensible talent build:

- The base kit (basic abilities, trait, basic attack) counts fully.
- Heroics: use the heroic option that matters most for the axis, but a capability that exists only in one heroic usually moves the score by at most 1 point, because the heroic has a long cooldown and is one of two choices.
- Talents: one talent per level is chosen, so a capability that exists only in one talent usually moves the score by at most 1 point. Quest talents count at their completed reward if the quest is easy to complete.
- Mechanics only: count what the text says an ability does (damage numbers, durations, cooldowns, ranges, area or single target, reliability such as skillshot versus targeted).

## Scale

All axes are integers from 0 to 4, except early_vs_late, which is an integer from -2 to +2. Use the full range across heroes. 0 means the capability is absent or negligible; 4 means it is a central, repeatable part of the kit at the top of what the text format allows.

## Axes

**hard_cc.** Effects that stop an enemy acting or moving: stun, root, silence, sleep, polymorph, fear, taunt, time stop, knock-up, knockback, pull, grab, and similar displacement.
- 0: none in base kit or heroics.
- 1: one short (under about 1 s) or conditional or hard-to-land source, or only via one heroic or talent.
- 2: one reliable basic-ability source (about 1 s or more, cooldown about 12 s or less) or a strong area heroic.
- 3: two or more base-kit sources, or one area source that can catch several heroes.
- 4: several reliable sources including area effects on short cooldowns; disabling enemies is a main job of the kit.

**soft_cc_peel.** Slows, blinds, attack-speed reductions, area denial, and tools that protect allies from attackers without healing them: cleanse, unstoppable or invulnerability given to an ally, moving allies out of danger, displacing attackers off an ally, damage reduction applied to enemies.
- 0: none. 1: one minor slow or one situational tool. 2: a reliable slow or blind, or one ally-protection tool. 3: several such tools, or a strong area slow plus a protection tool. 4: protecting allies and impairing attackers is a main job, with several frequent tools.

**healing_throughput.** Healing and shields given to allies (not to this hero alone).
- 0: none. 1: incidental or talent-only ally healing or shielding. 2: meaningful but secondary ally healing or shields (for example an area shield on a long cooldown, or small heals on hits). 3: sustained ally healing as a main function, but limited in amount, targets or uptime. 4: several frequent, large heal or shield sources; keeping allies alive is the kit's core.

**self_sustain_shields.** Ways this hero keeps itself alive: self-healing, lifesteal, self-shields, armor, damage reduction, self invulnerability or untargetability, extra lives, large health regeneration.
- 0: none. 1: one minor or long-cooldown source. 2: one reliable source or two minor ones. 3: several reliable sources. 4: surviving through own healing, shields or armor is central and repeatable.

**burst.** Damage this hero can deal in a short window (about 2 s), counting abilities usable together, empowered attacks, execute effects and percent-of-max-health damage. Rough guide using level-1 numbers of a full combo against one hero: 0 under about 150; 1 about 150 to 300; 2 about 300 to 500; 3 about 500 to 750; 4 above about 750, or strong execute or percent-health effects. Lower the score if most of the damage depends on hard-to-land conditions.

**sustained_damage.** Damage per second this hero can keep up over about 10 s of fighting: basic attack damage per second (damage divided by attack interval), repeatable ability damage, damage over time and summons. Rough guide using basic attack DPS at level 1: 0 under about 50 with little ability damage; 1 about 50 to 90; 2 about 90 to 130; 3 about 130 to 180; 4 above about 180, or strong stacking attack-speed or damage-over-time engines. Adjust up or down for ability damage that repeats quickly or for attacks that only work in some forms.

**range.** The distance from which this hero applies most of its damage and effects.
- 0: melee basic attack and short-range abilities only. 1: melee or short-range attacker with some medium-range abilities. 2: ranged basic attack (about 4.5 to 5.5) with typical ability ranges. 3: ranged attacker whose core damage comes from long-range abilities (about 10 or more). 4: very long range is the core of the kit (artillery, snipes, map-wide or very long targeting).

**mobility.** Moving faster than the enemy or crossing terrain: dashes, blinks, teleports, leaps, flight, movement-speed boosts, unstoppable movement, stealth used to reposition. Normal mount does not count; a special mount does.
- 0: none. 1: one minor speed boost or one long-cooldown movement tool. 2: one reliable dash, blink or strong speed boost on a moderate cooldown. 3: several tools, or one frequent one. 4: mobility is the core of the kit (several short-cooldown dashes or blinks).

**waveclear.** Killing minion waves quickly: area damage, piercing or line damage, bonus damage against minions or non-heroes, damage that spreads, summons that fight minions.
- 0: single-target only and slow. 1: limited area damage. 2: one good area ability or a strong basic attack against waves. 3: several area tools, or one that clears most of a wave. 4: fast, repeated area damage that clears full waves.

**siege.** Damaging and pressuring structures: summons that push or attack structures, attacks that out-range towers, bonus damage to structures, disabling structures, sustained damage while standing near forts.
- 0: none beyond a normal basic attack. 1: some structure damage from abilities or summons. 2: one clear siege tool. 3: several siege tools or strong unattended lane pressure. 4: taking structures is a main purpose of the kit.

**global_presence.** Affecting distant parts of the map: global or long-distance teleports, map-wide abilities or vision, remote lane soaking (summons or units that push lanes alone), fast travel such as special mounts or travel abilities.
- 0: none. 1: one minor tool (for example faster travel). 2: one global or semi-global tool. 3: several such tools. 4: the kit is built around acting across the whole map.

**early_vs_late** (-2 to +2). Where the mechanics put this hero's power over the game, read only from the text.
- -2: high base numbers and lane or early-fight tools, little or no scaling, nothing important gated behind levels.
- -1: leans early.
- 0: neutral.
- +1: leans late: some quests or stacking rewards, or key effects at levels 16 or 20.
- +2: heavy scaling: large quest rewards, permanent stacking bonuses, or core capabilities gated behind high-level talents or heroics, with weak base numbers.

## Output

Return strict JSON only, no prose, no code fences. One object keyed by hero code. Each value has the 12 integer fields and an optional list "low_confidence" naming axes where the text was unclear:

```
{"K123": {"hard_cc": 2, "soft_cc_peel": 1, "healing_throughput": 0, "self_sustain_shields": 1,
          "burst": 3, "sustained_damage": 2, "range": 2, "mobility": 1, "waveclear": 2, "siege": 1,
          "global_presence": 0, "early_vs_late": 0, "low_confidence": []}}
```
