# Proposer guide

You propose hypotheses about what makes a Heroes of the Storm draft win. A harness fits and
tests them. You never see raw outcome data. Every look at outcomes goes through `hyp`, is
charged against your budget and is written to your ledger.

## The model you are improving

Win probability for team A against team B is

    eta(A, B) = T(A | ctx) - T(B | ctx) + X(A, B) - X(B, A),    P(A wins) = sigmoid(eta)

T is team value, X is cross-team value, ctx is map, tier, build and date. The incumbent is
the kit-anchored backbone K0: hero strength per build (a random walk across builds), team
sums of the 12 kit axes, map class by kit axis, and player-hero experience. Each card you get
accepted adds one term to your incumbent. Numbers are always fitted by the harness; a card
buys a functional form, not a value.

## Budgets (per cycle)

| Item | Cap |
|---|---|
| Sandbox outcome numbers (Q1 on S) | 5,000 |
| Dev bits (Q1 and Q2 on your dev tranche) | 120 |
| Bits per cell group (a hero, a pair, an axis, a structure term) | 8; then the group is burned |
| Cards tested | 40 |

A Q2 test costs 1.5 bits. A noisy dev cell costs 0.16 bits. A card on a cell group passes the
dev gate when z >= 3.0 + 0.59 * sqrt(bits already spent on that group). Untouched groups need
z >= 3.0; a group probed for 4 bits needs z >= 4.2. So write cards from theory and from the
sandbox, and spend dev bits sparingly.

Suggested split of dev bits by family: composition 20, strength 12, solo_lane 14, macro 14,
team_fight 16, ganking 10, scaling 14, others 20.

## Tranches

- **S** (sandbox): older builds, about 1.9M games. Exploration is cheap here: exact win rates
  to 0.1pp in cells of 200 or more teams, charged to your 5,000 sandbox numbers. Results on S
  never support a claim by themselves.
- **Your dev tranche** (`hyp meta` shows which): cells of 500 or more teams only, Gaussian
  noise with sd = 2 SE, rounded to 0.5pp, charged in bits.
- Everything newer is closed to queries.

## Commands

```
hyp meta
hyp budget
hyp query '{"class":"Q0","tranche":"S","stat":"count","filter":{"hero":"<hero>"},"group_by":["map"]}'
hyp query '{"class":"Q1","tranche":"S","stat":"win_rate","filter":{"role_count":{"Healer":0}},"group_by":["tier"]}'
hyp card submit /work/cards/my_card.yaml
hyp test <card_id>
hyp test-status
hyp ledger --mine
```

Query fields: `filter` takes `hero`, `ally` (both on the same team), `enemy`, `map`, `tier`,
`build`, `role_count` ({role: count}). `group_by` takes at most 2 of `map`, `tier`, `build`,
`map_class`, `hero`, `enemy_hero`, `first_pick`, `role_count:<Role>`. Q1 returns at most 20 cells.
Identical queries return the cached answer.

## Card format (YAML)

See `card_template.yaml`. Required: `id`, `family`, `theory` (2 to 5 sentences),
`expected_effect` (sign, size in pp and in log loss, share of games affected), `term`.
Also give `mechanism` (the intermediate, its direction, time window and one negative-control
mediator), `sources` (your ledger entries, kit fields), `alternatives_considered`, and `tag`
(structural or meta). The card is hashed into your ledger when you submit it.

## Term types

| type | fields | notes |
|---|---|---|
| lookup | keys: [hero] or [hero, map / map_class / tier] | shrunk |
| state | (none) | hero strength by build, random walk |
| multiset | min_teams | role-multiset effect |
| indicator | predicates (all must hold), sign | capped monotone structural flag |
| attr_sum | axis, shape (monotone_inc, monotone_dec, concave_inc, free), n_knots | f(team sum of an axis) |
| bilinear | entries: [{a: axis, b: axis, sign}] (at most 30) | A_A' W A_B - A_B' W A_A |
| product | axis, by (map_class, map, tier) | axis sum by context |
| pair_family | min_games (>= 500) | within-team pairs, only as a family |
| cross_family | min_games | hero-vs-hero pairs, only as a family |
| decay | half_life_days, sign | player-hero experience, as-of |
| rule | predicates (at most 3), sign | capped at 0.15 logit |

Predicates: `{field: role_count, key: <Role>, op: ">=", value: 2}`,
`{field: kit_sum, key: <axis>, op: "<=", value: 3}`, `{field: map_class, value: lanes2}`,
`{field: map, value: <map>}`, `{field: tier, value: high}`, `{field: hero_in_team, value: <hero>}`.

Only draft-time inputs exist. Fields such as game length, MMR, talents or scoreboard are
refused.

## What a test returns

`accept` or `reject`, plus the gain in dev log loss rounded to 0.0001 with noise. Accepted
cards join your incumbent. A near miss (z between 2.0 and the threshold) may be retried once
in a materially different form on the next tranche at double charge. No mechanism data is
available yet (parsed replays are still arriving), so accepted terms carry an `unexplained` flag.

## Rules

- Meta facts ("hero X is strong this patch") are parameter values. Do not submit them as cards.
- Work only in /work. Keep notes there; they are part of the record.
- Do not try to reach data any other way. The sandbox has no network except the model API.
