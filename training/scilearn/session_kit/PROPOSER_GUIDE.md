# Proposer guide

You propose hypotheses about what makes a Heroes of the Storm draft win. A harness fits and
tests them. You never see raw outcome data. Every look at outcomes goes through `hyp`, is
charged against your budget and is written to your ledger.

## The model you are improving

Win probability for team A against team B is

    eta(A, B) = T(A | ctx) - T(B | ctx) + X(A, B) - X(B, A),    P(A wins) = sigmoid(eta)

T is team value, X is cross-team value, ctx is map, tier, build and date. The backbone is
K0: hero strength per build (a random walk across builds), team sums of the 12 kit axes, map
class by kit axis, and player-hero experience. Each card you get accepted adds one term to
your incumbent. Numbers are always fitted by the harness; a card buys a functional form, not
a value. `hyp meta` lists your incumbent's terms, including any lookup tables accepted in an
earlier cycle.

## Budgets (per cycle)

| Item | Cap |
|---|---|
| Sandbox outcome numbers (Q1 on S) | 5,000 |
| Dev bits (Q1 and Q2 on your dev tranche) | 120 |
| Bits per cell group (a hero, a pair, an axis, a structure term) | 8; then the group is burned |
| Cards tested | 40 |
| Mediator bits (Q3 cells and mechanism tests on M-test) | 60 |
| Inspected games (M-explore) | 300 |

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
in a materially different form at double charge. Accepted terms carry an `unexplained` flag
until a mechanism test passes (below). If a test ends in `error`, the harness failed, nothing
was charged, and you can run `hyp test` on the same card again.

Timing: a test takes 1 to 4 minutes once it runs. After an accept, the incumbent is refit
(about 100 minutes), and every test or mechanism test you start after the accept waits for
that refit. `hyp test-status` shows a `stage` for each pending card (waiting for a refit,
queued for a slot, or running). A pending test is never lost: its result and charge land in
your ledger when it finishes, so wait for it rather than stopping.

## Explanation cards (cycle 2)

A lookup table (hero x map, hero x tier, ally pairs, hero-vs-hero counters) predicts well and
explains nothing. An explanation card proposes a small kit- or role-based function that
reproduces the table's pattern. It adds no predictive term. It earns its place by explaining
the table, by transferring to heroes it was not fit on, and by not hurting early-patch games.

```yaml
id: e01
family: team_fight
theory: >
  Hard crowd control pins down mobile heroes, so heroes with hard CC should do better
  against mobile enemies than their overall strength says.
expected_effect: {sign: "+", share: 0.05}
term:
  type: explain
  table: c06_cross_family          # card id of an accepted table in your incumbent
  features:
    - {kit_pair: [hard_cc, mobility], sign: "+"}
```

Features by table type (at most 30 parameters in total):

| table | feature | parameters |
|---|---|---|
| hero x map | `{kit: <axis>, by: map}` / `by: map_class` / `by: lanes2` | 15 / 2 / 1 |
| hero x map | `{role: <Role>, by: map}` (or map_class, lanes2) | 15 / 2 / 1 |
| hero x tier | `{kit: <axis>, by: tier}`, `{role: <Role>, by: tier}` | 4 |
| ally pairs | `{kit_pair: [a, b]}` (a of one hero times b of the other, symmetric) | 1 |
| ally pairs | `{kit_sum: <axis>}` (both heroes high on the same axis: stacking) | 1 |
| ally pairs | `{same_role: true}`, `{role_pair: true}` | 1, 21 |
| counters | `{kit_pair: [a, b]}`: a of my hero times b of the enemy, minus the reverse | 1 |
| counters | `{role_pair: true}` (role vs role) | 15 |

Any feature may take `sign: "+"` or `"-"`. Hero and context main effects are removed from the
table before scoring (hero strength and map means live elsewhere in the model), so a theory
is credited only for interaction structure.

How it is scored (`hyp test <id>`, 1.5 bits from your dev budget, charged to the group
`explain:<table>`):

1. **Explain.** The table is refit on data disjoint from the sandbox and from your dev
   tranche. The theory is fit on 9/10 of the cells and scored on the other tenth, ten times,
   with cells weighted by their precision. Released: the share of the table's signal it
   explains. Gate: z >= 3.0 + 0.59 sqrt(bits already on the group), and the theory must
   compress (the bits it saves on held-out cells exceed 0.5 k log2(cells) for k parameters).
2. **Transfer.** Fit without any cell of 20 held-out heroes, scored on their cells. Gate: z >= 2.
3. **Onset.** The table shrunk toward the theory instead of toward zero, scored on the first
   10K games of each dev build. Gate: no worse than z = -2.
4. **Kit permutations.** The kit axes you named are permuted within role 200 times. The real
   axes must beat 95% of the permutations. A kit theory whose content is only role-level
   fails here; write it as a role theory instead.

`hyp table <card_id>` shows a table's largest cells from an S-only fit of your incumbent
(hero and context main effects removed), at one sandbox number per cell. Explore there; the
test uses a different fit.

## Mechanism cards and the M tranche (cycle 2)

Parsed replays of 2.57 games form M, split once by keyed hash into M-explore (for reading)
and M-test (for aggregate tests). `hyp meta` shows how many games each holds.

Mediators (team perspective; `lead` is own minus opponent): `level_lead_5`, `level_lead_7`,
`level_lead_10`, `level_lead_15`, `first_to_10`, `xp_lead_10`, `minion_xp_share_10`,
`creep_xp_share_10`, `hero_xp_share_10`, `deaths_10`, `isolated_deaths_10`, `first_blood`,
`fights_won_10`, `fights_won_post10`, `towers_lost_15`, `forts_lost_15`, `first_fort_lost`,
`mercs_10`. Definitions are in `hyp meta`.

- **Q3:** `hyp query '{"class":"Q3","mediator":"minion_xp_share_10","filter":{"role_count":{"Healer":0}},"group_by":["map_class"]}'`
  returns mediator means in cells of at least 200 team rows on M-test, with noise sd = 2 SE.
  Each number costs 0.16 rho^2 mediator bits (rho is the mediator's correlation with winning).
- **Inspect (Q5):** `hyp inspect --hero <hero>` returns one M-explore game: levels, XP by
  source, deaths with positions, structures, mercs, map events. The game is spent. Use
  `--interesting` for games the draft model found surprising, `--random` for controls.
- **Mechanism test:** give a card a mechanism block with names from the list above,
  ```yaml
  mechanism:
    mediator: minion_xp_share_10
    direction: "+"                     # the team the term favours has more of it
    negative_control: hero_xp_share_10
  ```
  test the card as usual, then run `hyp mtest <card_id>` (1.5 mediator bits, once per card).
  The card's fitted term must predict the mediator in the declared direction on M-test
  (z >= 3.0 + 0.59 sqrt(bits on `mech:<mediator>`)) and leave the negative control alone
  (|z| < 2). A pass replaces `unexplained` with `mechanism_ok`.
- **Mechanism card** for a term already in your incumbent (an inherited table, say): the
  same header and mechanism block, with
  ```yaml
  term: {type: mechanism, of: c06_cross_family}
  ```
  Submit it, then `hyp mtest <id>`. No dev bits; 1.5 mediator bits. M-test is small (see
  `hyp meta`); a mechanism test needs at least 2,000 M-test games and is refused before.

## Replacement cards (cycle 3)

A predictive card may replace a term already in your incumbent, for example a pair table with
a different support floor. Add `replaces:` with the card id of that term:

```yaml
id: r01
family: composition
theory: >
  ...
expected_effect: {sign: "+", pp: 0.3, logloss: 0.0001, frac_games: 0.3}
replaces: c05_pair_family          # card id of a term in your incumbent (`hyp meta` lists them)
term: {type: pair_family, min_games: 2000}
```

- **One test, one card.** `hyp test r01` costs 1.5 dev bits, like any predictive card, charged
  to the cell groups of both the new and the replaced term. The gate is the usual one, on the
  groups' bits so far.
- **What is compared.** The new term is fit with the rest of your incumbent fixed and the old
  term removed. The dev comparison is your incumbent as it is against your incumbent with the
  old term swapped for the new one. The gain can be negative: a replacement that predicts
  worse than what it replaces is rejected.
- **Parameter cap.** The 20,000-parameter cap is checked on the result (incumbent minus the old
  term plus the new one). A replacement that would exceed it is refused before any result is
  computed, and nothing is charged.
- **If accepted,** the old term leaves your incumbent and the new one joins it. A table that
  was replaced can no longer be named by explanation or mechanism cards; the new one can.
- Only predictive terms can replace, and only terms that came from a card (not the K0
  backbone).

## Rules

- Meta facts ("hero X is strong this patch") are parameter values. Do not submit them as cards.
- Work only in /work. Keep notes there; they are part of the record.
- Do not try to reach data any other way. The sandbox has no network except the model API.
