# replay-probe

A throwaway-turned-kept diagnostic that answers one question: **what is actually
recorded in a `.StormReplay`, and at what resolution?**

It exists because the post-game analysis roadmap (`/roadmap/analysis`) needed that
answer as a fact rather than an assumption. The headline finding is that
`Heroes.ReplayParser` exposes a hero position for every second of the game, but
most of those are interpolated — `Position.IsEstimated` marks them — and the
positions the replay genuinely recorded are a median of **15 seconds apart**.
That single number is why the roadmap scopes to macro analysis and rules out
mechanical/micro feedback.

## Run it

```
dotnet run -c Release -- "<path to a .StormReplay>"
```

Replays live under
`%USERPROFILE%\Documents\Heroes of the Storm\Accounts\<account>\<toon>\Replays\Multiplayer`.

## What it prints

- map, map grid size, length, frame count
- counts of units, tracker events and game events
- per player: total position points, how many were actually recorded rather than
  interpolated, and the median / p90 gap between real fixes
- the most common game-event types (the dense movement and camera streams)
- hero deaths with exact time, coordinates, killer and killing unit

Hero units are emitted **one instance per life**, so a player who died ten times
has eleven entries; the per-player table stitches those segments back together.

## Sample output

From a 20:49 Volskaya Foundry Storm League game:

```
map               : Volskaya Foundry  size 248x208
length            : 00:20:49  (19984 frames)
units (all)       : 4809
tracker events    : 10750
game events       : 136482

player            all pts    real  real dt med  real dt p90
IAmTheJames           900      58        15.0s        30.0s
SmugClarence          962      68        15.0s        30.0s

top game-event types:
  CCmdUpdateTargetPointEvent     37431
  CCmdEvent                      25217
  CCameraUpdateEvent             13484

hero deaths       : 43
  01:16  HeroSylvanas   at (106,46)  killed by otherworld (HeroSamuro)
```
