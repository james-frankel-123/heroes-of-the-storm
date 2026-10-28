# Scoreboard profiles for the base draft WP (2026-10-26)

**Question (Max).** Can the scoreboard help the base model? Average each hero's scoreboard (stun time, self-healing, XP soak and so on) for its hero, map and tier, sum it over the team, and look at what complements and what is redundant. The scoreboard was a useful extra signal for personalization, so it might help the underlying system too.

**Answer.** It adds almost nothing.
- **Inside paper 1's own split** (the WP's training period), profiles add nothing: -0.10 ×1000 log loss (CI -0.34 to +0.12).
- **After the snapshot, against a stale base**, they add +0.54 ×1000 (0.33 to 0.83) and +0.17pp accuracy.
- **An as-of hero win rate does more**, +1.97 ×1000 (1.69 to 2.15). On top of it, profiles add +0.13 (-0.05 to +0.36).
- **The composition terms** (coverage, redundancy, interactions) carry no lift in either design.

Recommendation: do not add profiles to the base WP or the production refresh.

Code: `training/personalization/base_profiles/` (bsp_export, bsp_wp, bsp_profiles, bsp_eval). Numbers: `results/base_profiles/bsp_results.json`. Everything except the export ran on the 5090's CPU (nice 19, 4 threads, under 8 GB, watchdog in `bsp_run.sh`).

## Headline

All lifts are measured against the recalibrated base: the same offset model with only an intercept and a slope on logit(base WP), fit on the same rows. Positive means better. CIs are 95% week-block bootstraps.

| design | test games | base ll ×1000 / acc | profiles (pre-planned set): ll lift ×1000 | acc lift pp | as-of hero WR (control) | profiles on top of the control |
|---|---|---|---|---|---|---|
| POST, time split (primary) | 118,810 (7 weeks) | 681.65 / 56.09% | **+0.54 (0.33, 0.83)** | +0.17 (0.04, 0.30) | +1.97 (1.69, 2.15) | +0.13 (-0.05, 0.36) |
| P1, paper 1's own test set | 38,869 (233 weeks) | 674.40 / 57.65% | **-0.10 (-0.34, 0.12)** | +0.01 (-0.20, 0.22) | +0.32 (0.10, 0.54) | -0.12 (-0.34, 0.10) |

- The richer regularized set (420 terms, L2 chosen on validation) is worse than the small set in both designs: POST +0.34 (0.17, 0.52), P1 -0.03 (-0.23, 0.18).
- In P1, its validation loss falls monotonically as the penalty grows.
- On validation the small set was selected over the rich set in both designs.
- The recalibration alone is worth +0.13 ×1000 in POST (the base's slope there is 0.97) and 0 in P1.

## What was built

**Data.** One read-only export from the local research store, with no player identity. It covers 2,497,483 Storm League games and 24,958,096 player scoreboards dated before 2026-09-28, with no 2.57 build. The 5,000 sealed replays were excluded inside the SQL, so they were never read.

**Base WP.** Paper 1's leak-free site WP `enriched`: 283 inputs, out-of-fold statistics, selected seed 42, swap-symmetrized as in `paper1_revision/train_wp.evaluate`.
- It reproduces paper 1's test numbers: log loss 0.6744 and accuracy 57.6%.
- Post-snapshot games are featurized with the frozen deploy statistics, as deployed. There the base scores 0.6803 and 56.4%.
- The consensus judge (CONS) was not usable as a base here: it was trained on post-snapshot games.

**Profiles.** These use 22 per-minute dimensions: hero damage, siege, structure, healing, self-healing, shields, damage taken, XP, CC, stun, root, silence, mercs, deaths, time dead, escapes, globes, teamfight damage dealt, taken and healed, takedowns and outnumbered deaths.
- CC is stun + root + silence seconds per minute. The store's `time_cc_enemy_heroes` is 0 on every row, and slows and displacement are not on the scoreboard.
- Each dimension is winsorized at its 99.9th percentile.
- Sums are kept by week and cell (hero × map × tier × outcome).
- A game's profile uses only weeks strictly before its own week: the trailing 26 weeks by default.

**Outcome.** Winners' and losers' means are shrunk separately and then averaged with equal weight: hero × map × tier toward hero × map toward hero, 40 games of prior weight. So a profile does not encode how often the hero wins.

**Sanity checks.**
- The top CC heroes are Mal'Ganis, Deckard, Stukov, Malfurion, Ana and Arthas.
- The top healers are Li Li, Alexstrasza, Brightwing, Lúcio and Whitemane.
- The top self-healers are Illidan, Chen, Tyrael and Varian.
- The top soakers are The Lost Vikings, Ragnaros, Murky and Azmodan.
- Median cell size is about 1,000 games.

**Team features** (every feature is team 0 minus team 1, so the model is antisymmetric). The pre-planned set has 31 terms in 4 families:
- **SUM** (12): team totals of damage, siege, healing, self-healing, shields, damage taken, XP, CC, mercs, deaths, escapes and teamfight damage.
- **MAX** (5): the team's top value of damage, healing, CC, damage taken and self-healing.
- **COV** (9): coverage and redundancy.
  - Healers, CC heroes and sustain heroes are defined from the profile: half the median team-top value. A real healer is ≥ 1,956 healing/min, heavy CC is ≥ 1.06 CC-s/min, a sustain hero is ≥ 736 self-healing/min, and a frontliner is ≥ 2,201 damage taken/min.
  - The terms: number of real healers, no healer, 2+ healers, number of heavy-CC heroes, the second-highest CC, CC concentration, the second-highest self-healing, 2+ sustain heroes, and no frontliner.
  - By these definitions, 96.8% of teams have exactly one real healer, 1.3% have none, 63% have 2+ heavy-CC heroes and 78% have 2+ sustain heroes.
- **INT** (5): CC × damage, healing × damage taken, self-healing × healing, own CC vs enemy escapes, and own damage vs enemy healing.

The rich set adds 420 terms: sum, max, min and second-highest of 18 dimensions; all within-team and cross-team products of 12 team sums; and map × sum and tier × sum.

**Model and splits.** The model is logit P = b + a·logit(base) + w·x, with L2 on w only.
- **POST** (time split): offset-train 05-23 to 07-12 (108,108 games), validation 07-13 to 08-09 (60,833), test 08-10 to 09-27 (118,810). These are games paper 1's WP never saw.
- **P1**: offset fit on paper 1's validation games (38,100), choices by 5-fold week-block CV inside them, test once on paper 1's test games (38,869).
- Paper 1's split is a replay-hash split inside the snapshot. It is not a time split, so POST is the time-based design.
- Each test set was scored once, after selection.

## Which families carry it

Ablations are on the test sets: ll lift ×1000 against the recalibrated base.

| set | POST | P1 |
|---|---|---|
| all 31 (selected) | +0.54 (0.33, 0.83) | -0.10 (-0.34, 0.12) |
| SUM only | **+0.66 (0.54, 0.79)** | +0.05 (-0.12, 0.23) |
| MAX only | -0.29 (-0.42, -0.17) | +0.05 (-0.00, 0.11) |
| COV only | +0.02 (-0.04, 0.09) | -0.19 (-0.37, -0.01) |
| INT only | -0.10 (-0.14, -0.05) | -0.04 (-0.12, 0.03) |
| drop SUM | -0.26 (-0.43, -0.03) | -0.09 (-0.25, 0.07) |
| drop MAX | +0.39 (0.17, 0.63) | -0.07 (-0.29, 0.15) |
| drop COV | +0.39 (0.17, 0.63) | +0.01 (-0.20, 0.22) |
| drop INT | +0.63 (0.45, 0.88) | -0.08 (-0.30, 0.13) |

- All of the POST lift sits in the team sums, which are additive in the heroes. Coverage, redundancy and interaction terms add nothing alone, and removing the interactions improves the test score.
- An additive hero-level profile can only re-weight hero strengths along a few directions, refit on the new period's outcomes. That is a cheap update of a stale base. It explains why the lift appears after the snapshot, vanishes inside it, and is beaten 4 to 1 by an as-of hero × map × tier win rate (one term).

**By tier** (POST, pre-planned set, ll ×1000):

| tier | lift |
|---|---|
| low | +0.75 (0.42, 1.07) |
| mid | +0.32 (-0.10, 0.80) |
| high | +0.55 (-0.24, 1.23) |

In P1, every tier is within ±0.21 with CIs that span 0.

**By map** (POST): the lift is positive on 9 of 11 maps.
- Largest: Braxis Holdout +1.32 (0.55, 2.15), Volskaya +1.11 (0.42, 1.92), Towers of Doom +0.87 (-0.03, 1.68), Cursed Hollow +0.85 (0.39, 1.34).
- Negative: Dragon Shire -0.46 (-1.93, 0.91) and Infernal Shrines -0.11.

In P1, no map is clearly positive; Infernal Shrines is -0.85 (-1.61, -0.12). The map pattern does not replicate between designs, so I read it as noise around a small additive update.

## Patch drift

| POST test, pre-planned set | ll lift ×1000 vs recalibrated base | vs rolling 26-week profiles |
|---|---|---|
| rolling 26 weeks (main) | +0.54 (0.33, 0.83) | |
| rolling 8 weeks | +0.61 (0.38, 0.90) | +0.06 better |
| all history | +0.29 (0.10, 0.52) | -0.26 |
| frozen at the snapshot (as of 2026-05-23) | +0.36 (0.14, 0.65) | -0.18 (-0.22, -0.14) |

**Updating the profiles is worth +0.18 ×1000.** The CI is tight, but it is only a third of an already small lift. Shorter windows help slightly.

**Right after a patch.** The window holds the tail of 97039 and four new builds: 97605 (from 07-20), 97650 (07-24), 97771 (08-12) and 98025 (09-12). The test period covers the last two releases. Two measures:
- Updated vs frozen profiles: +0.20 in a build's first 14 days and +0.16 after.
- Profiles vs base: +0.53 and +0.56.

There is no extra gain right after a patch.

**Heroes whose profiles changed.** For each game I measured how far its ten heroes' profiles moved since the snapshot, in standardized units.
- The most-moved heroes are Abathur, Chen, Mal'Ganis, Qhira, Yrel and Murky.
- Games in the top quarter of movement gain +0.06 (-0.49, 0.79) from profiles and +0.05 from updating them.
- The other games gain +0.70 and +0.23.

Profiles help least exactly where they changed most. A measured profile change does not track a change in hero strength well enough to replace a retrain or a refreshed win rate.

## Interpretable effects

These are coefficients of the pre-planned set, in logit per SD with SE, plus win-probability points at 50%. They are effects on top of a base that already knows the heroes and their roles.

| effect | POST | P1 (in-snapshot) | game sense |
|---|---|---|---|
| top CC hero's CC (MAX:cc) | +0.075 (0.022); +1.2pp per CC-s/min | -0.014 (0.022) | one heavy-CC hero matters |
| second heavy-CC hero's CC (COV:second_cc) | -0.026 (0.013); -0.65pp per CC-s/min | -0.002 (0.019) | diminishing returns to stacking CC |
| second self-sustain hero (COV:second_self_heal) | -0.033 (0.014); about -0.8pp per 500 HP/min | +0.020 (0.021) | double bruiser costs damage |
| 2+ sustain heroes (indicator) | +0.002 (0.011) | -0.033 (0.017); -1.3pp | same idea, other design |
| no real healer | -0.073 (0.028) with n_healers -0.095 (0.045): about -2.4pp vs one healer | +0.034 (0.023), opposite sign | not stable |
| team escapes total | -0.037 (0.007) | +0.000 (0.012) | not stable |
| team XP soak total | +0.070 (0.016) | +0.038 (0.022) | soak is good, the same sign in both |

**The sensible story is POST-only.** Max's two examples get the expected signs after the snapshot: a second heavy-CC hero is worth less than the first, and a second self-sustain bruiser is a mild negative. Both effects are near z = 2 and do not replicate inside the snapshot, where the second-CC term is 0 and the second-sustain term flips sign.
- Healer coverage flips sign between designs. That fits the base already pricing healer count through its role features (it has a role-count group and a composition win-rate group).
- The rich model's largest terms (max CC, CC × CC within team, the team's weakest teamfight damage) are each worth under 0.016 logit per SD.

## Comparison with paper 3's kit capabilities

Paper 3 tested LLM-rated per-hero kit axes (hard CC, burst, healing, mobility, waveclear, and others) as composition terms. Kit-pair, kit-by-context and kit-conjunction terms all gained about 0 (around 1e-6 per game), and kit axes explained 0 to 5% of the pair and counter tables (`paper/hypothesis_system/CYCLE2_RESULTS.md`).

Measured profiles land in the same place for composition. Data-measured CC, sustain and healer coverage by hero, map, tier and time add no composition signal beyond what the hero-level WP already holds. The one new thing they offer is a time-varying additive term. It helps a stale base by about 0.5 ×1000, and an as-of win rate already does that four times better.

**Why the scoreboard helped personalization but not here.** For a player, the scoreboard measures how this player plays a hero relative to its norm. The draft model has no other source for that. For a hero, the profile is a fixed function of hero, map, tier and date, and the base WP already conditions on everything but the date.

## Recommendation

- **Base WP:** do not add profiles. Paper 1's split shows no lift (-0.10, CI -0.34 to 0.12), and composition families add nothing in either design.
- **Production refresh:** no change. The post-snapshot gain is staleness, and the monthly refresh already retrains on 90-day decayed statistics. That is a stronger version of the as-of win rate that beat profiles here.
- **Retraining the WP with profile inputs (stretch):** not warranted, because the offset model shows no real lift. If it were ever wanted, the work would be:
  - the 22 profile dimensions per hero slot (or the 31 team terms) built out-of-fold and as-of for training rows, with the same week-strict windows;
  - a refit of the three-seed WP (a short GPU job);
  - the same test and NODRIFT evaluation;
  - a kernel change so MCTS can read per-hero profile tables.
- **A possible non-predictive use:** profiles describe drafts in game terms ("no real healer", "two heavy-CC heroes"). That could label site drafts, but it should not be presented as a win-rate effect.

## Limits

- POST's test has 7 week blocks, so its bootstrap CIs are rough; P1 has 233.
- CC covers stun, root and silence only.
- Tiers are the site scheme from the live store; the base WP was trained on the same labels.
- The P1 offset fit has only 38,100 games, which is why the rich set cannot be fit there. The small-set result is still a clean null within ±0.3.
