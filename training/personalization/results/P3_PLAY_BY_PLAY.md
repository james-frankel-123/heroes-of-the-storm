# P3: what the play-by-play adds beyond the scoreboard (2026-10-24)

Max asked, while the replay download catches up: can the richer play-by-play in full replays predict a player's future results beyond what the scoreboard-based player model already captures? This is exploratory, for knowledge, and nothing here goes into the definitive model.

All numbers are log loss gains x1000 on whole games (team 0 wins) unless marked "pp" (points of win probability). SEs are game-level; day-block bootstrap SEs are in `results/pbp/holdout.json` and agree within about 10%.

## Headline

- **Replay features add +1.7 log-loss points (SE 0.5, 95% CI +0.7 to +2.7) on 21,308 held-out games**, over the closest runnable version of the chosen model plus draft WP. That baseline already gains +14.0 (SE 1.2) from its scoreboard channel, so the replay gain is about an eighth of what the scoreboard gives. Accuracy moves from 61.87% to 62.00%.
- **About 60% of what replays could add is already in the scoreboard.** Over a baseline without any scoreboard channel, the same features add +4.1 (SE 0.7). With the scoreboard channel, +1.7. With every DB scoreboard extra as well, +1.3 (SE 0.5).
- **In edge units:** players in the top third of the replay score beat the baseline's expectation by +1.6 pp and the bottom third fall short by 0.7 pp, a gap of **2.4 pp (SE 0.3)**. The replay score moves a player's predicted win probability with sd 1.4 pp, and the realized slope is 0.78, so the numbers are about right in size.
- **Where it comes from:** mechanics (+0.9, SE 0.3), fight positioning (+0.6, SE 0.3) and macro/soak (+0.6, SE 0.3). Death context adds nothing (+0.2, SE 0.2) once the scoreboard is in, and dropping it from the full set changes nothing (+0.05).
- **It needs dense history in the lobby.** In games where 7 to 10 players have 5+ past replays the replay features add +3.2 to +4.9. In games where at most 3 do, they add about 0. Most of the gain is available from a player's single most recent replay (+2.1 with 1 replay, +3.0 with all, on lobbies with 3+ heavy players).
- **A cheaper scoreboard-only finding of similar size:** each player's share of the team's hero damage, damage taken, healing, siege, XP and takedowns adds **+1.3 (SE 0.3)**. It needs no replay file and works on every game.
- **Recommendation:** not worth building into the next model version as a replay channel yet. Add the team-share features to the scoreboard channel (same size, every game, no new data). Revisit replay mechanics when the panel has a few more weeks of dense lobbies.

---

## 1. Data

- **Replays.** 41,996 non-sealed replays, each re-parsed into an extended timeline (`training/replays/extract_pbp.py`): the normal timeline plus per-player ability casts by ability, per-second command counts, camera position and right-click target every 5 s, lane-minion deaths with killer and position, XP-globe pickups, merc-unit deaths and hero respawns.
  - Cohorts: stratified_b and surprise_v1 (15,508) and p2_panel_v1 (26,488, including today's catch-up batch).
  - Games from 2026-09-28 to 10-23, 2.57 only.
  - Parsed on the 3080 (cores 4-15, before the 22:40 parse window, 0 errors) and on this box (cores 56-59, nice 19).
- **Sealed holdout.** Every replay id was checked against `sealed_ids.txt`, with zero overlap, asserted in the extractor, the feature builder and the evaluation. Nothing under `hots-replays/sealed` was opened. The 4,999 sealed games in the DB were dropped by id before any row was read.
- **Targets.** Every 2.57 Storm League game in the local research DB with ten player rows: 74,143 games, 54% of them with a replay file. Outcome: team 0 wins.
- **History rule.** Every feature for a game on UTC day D uses rows from days before D only. A rewrite test (`leakcheck`) replaces every replay and DB row from 10-11 on with noise; no feature of an earlier game changes (max difference 3e-13, float rounding).

## 2. Features

Per player-game, then normalized as the scoreboard is (per minute where it is a count, asinh, z within hero x map x outcome, shrunk to hero x outcome), with statistics from rows before 2026-10-16. A player's feature is the shrunk mean over earlier replays, k = (1 − r)/r pseudo-games.

| family | features |
|---|---|
| mechanics | commands, retargets, key presses and camera moves per minute; ability casts and right-clicks per minute; ability share of commands; idle share while alive (seconds in command gaps of 5 s or more); burstiness (CV of 10 s action counts); ability-mix entropy; ability-mix deviation from the hero's norm; talent pick latency; camera away from the hero |
| death context | isolated deaths; unpaid deaths (no enemy hero died within 10 s); traded share; deaths in the 45 s before a map objective; deaths after 16:00; first blood; how far forward deaths happen; outnumbered exposure; survival when outnumbered |
| fight and position | presence at team fights; takedown share in fights; died-in-fight share; distance to the team centroid; front or back of the team on the core-to-core axis, overall and in fights; base time; distance moved |
| macro | XP globes; minion last hits; enemy minion deaths nearby before and after 10:00 (soak proxy); merc-unit kill share; merc presence; presence at enemy structure kills; objective presence; structure last hits |
| learned embedding | a 16-dimension per-game "style" vector from a small 1D CNN over 10 s bins of the first 20 minutes (12 channels: actions, casts, alive, axis position, centroid distance, enemies and allies near, died, takedown, soak, camera away), trained on 5090 to recognize the same player across games (no outcomes used), rows before 10-16 only |
| DB scoreboard extras (every game, no replay needed) | stun, root, silence, teamfight damage, taken and healing, minion and creep damage, time dead; share of the team's hero damage, damage taken, healing, siege, XP and takedowns; game-to-game consistency (sd of the main metrics); talent choices (how unusual, and the as-of payoff of the chosen talents) |

Not built: rotation timing and camp-capture timing (no clean lane assignment across maps from 15 s positions), damage timing (the replay has no per-time damage), reaction times to specific events.

## 3. Reliability: how much of one game is a stable trait

Split-half reliability of one game's value (odd vs even games, rows before 10-16), 95% CI from 200 player bootstraps. "Redundant" is the share of the stable player-level part predicted by the same games' DB scoreboard (5-fold ridge). The win residual itself is 0.004 per game.

| feature | player x hero (95% CI) | player | redundant with scoreboard |
|---|---|---|---|
| ability share of commands | 0.98 (0.90, 1.05) | 0.88 | 0.01 |
| right-clicks / camera moves / retargets per min | 0.95 / 0.94 / 0.93 | 0.87 / 0.93 / 0.89 | 0.01 / 0.00 / 0.02 |
| ability-mix deviation from the hero | 0.93 (0.87, 0.98) | 0.71 | 0.00 |
| commands / ability casts / key presses per min | 0.91 / 0.90 / 0.88 | 0.83 / 0.81 / 0.75 | 0.01 / 0.02 / 0.02 |
| learned embedding, 16 dims | 0.70 to 0.92 | 0.65 to 0.87 | 0.00 to 0.05 |
| ability-mix entropy | 0.80 | 0.58 | 0.01 |
| burstiness | 0.51 | 0.42 | 0.05 |
| talent pick latency | 0.47 | 0.48 | 0.02 |
| camera away from the hero | 0.45 | 0.40 | 0.00 |
| minion last hits | 0.31 | 0.16 | 0.51 |
| idle share while alive | 0.26 | 0.23 | 0.06 |
| distance to the team centroid | 0.16 | 0.09 | 0.25 |
| merc-unit kill share | 0.14 | 0.07 | 0.37 |
| distance moved / front-back position | 0.13 / 0.13 | 0.08 / 0.08 | 0.06 / 0.28 |
| unpaid deaths / soak before 10:00 | 0.11 / 0.12 | 0.09 / 0.07 | 0.51 / 0.32 |
| merc presence / isolated deaths / XP globes | 0.10 / 0.09 / 0.08 | 0.06 / 0.07 / 0.05 | 0.37 / 0.43 / 0.20 |
| outnumbered exposure / died-in-fight share | 0.08 / 0.07 | 0.05 / 0.05 | 0.43 / 0.42 |
| late deaths, soak after 10:00, fight presence, deaths before objectives, structure and objective presence, front in fights, base time | 0.02 to 0.06 | 0.01 to 0.04 | 0.13 to 0.48 |
| first blood, traded share, survival when outnumbered, structure last hits | 0.01 to 0.02 | 0.01 | |

Reading:
- **Mechanics are a fingerprint.** Every mechanics measure is 0.45 to 0.98 stable in one game and almost none of it is in the scoreboard. The learned embedding is the same kind of fingerprint (it was trained to be one).
- **Everything about deaths, fights, positioning and macro is 0.01 to 0.16 stable**, like kills and assists on the scoreboard, and a quarter to a half of that is already in the scoreboard.

## 4. Design of the test

- **Windows**, fixed before any result was seen:
  - fit: target games 10-01 to 10-10;
  - tune: 10-11 to 10-15 (every choice: L2 strength, which features and families to keep);
  - holdout: 10-16 to 10-24, 21,308 games, run once, with coefficients refit on fit + tune (46,448 games).
- **Model.** Logistic regression on team differences with a fixed L2 penalty (λ = 300, chosen on tune from 3 to 1000).
- **Baselines**, all as of the game's day:

| arm | columns |
|---|---|
| A0 | draft WP (population ensemble, logit; 84% of games, the rest fall back to hero win rates: Haunted Mines and Xal'atath are unknown to it) and the heroes' 2.57 win rates |
| A1 | A0 + the main model (full_ms posterior + experience offset, frozen 2026-09-28) + the player's 2.57 record |
| **A2** | A1 + the direct scoreboard channel: 15 metric means on the hero and over all heroes from the full history to 09-28, and the same from 2.57 DB games. **This is the baseline**: the closest runnable version of the chosen model (LK2_full2 without its MMR term and calibration, which cannot be rebuilt past 09-28) |
| A3 | A2 + every DB scoreboard extra (section 2) |

- **Multiple comparisons.** About 50 replay features. Two selection rules were frozen on the tune window: single features with a tune gain above 2 SE (kept: survival when outnumbered, structure presence), and families with a gain above 1 SE (kept: mechanics, fight, macro, embedding). Only the holdout numbers below are claims. Tune-window numbers are in `results/pbp/tune_frozen.json`.

## 5. Results on the holdout

Baseline steps (all games):

| step | gain (SE) |
|---|---|
| A1 over A0 (main model and record) | +11.2 (1.0) |
| A2 over A1 (scoreboard channel) | +14.0 (1.2) |
| A3 over A2 (DB scoreboard extras) | +1.4 (0.4) |

Replay features (gain over the named arm, SE). "Deploy" is the realistic case: the held-out game has no replay file, only the players' past replays (12,296 games with 3+ players with history).

| arm | all games (21,308) | deploy (12,296) | game has a replay file (8,754) |
|---|---|---|---|
| all hand-built replay features over A1 (no scoreboard channel) | +4.07 (0.72) | +2.85 (0.85) | +6.00 (1.28) |
| **all hand-built replay features over A2** | **+1.69 (0.51)** | **+1.04 (0.61)** | +2.69 (0.89) |
| all hand-built replay features over A3 | +1.30 (0.48) | +0.82 (0.58) | +2.04 (0.84) |
| frozen family selection over A2 (mechanics, fight, macro, embedding) | +1.69 (0.52) | +0.98 (0.64) | |
| frozen single-feature selection over A2 (2 features) | +0.04 (0.16) | +0.02 (0.18) | |
| embedding added to all hand-built features | −0.10 (0.18) | −0.15 (0.24) | |
| hand-built features as player x hero means instead of player means | +0.13 (0.45) | −0.53 (0.49) | |

Per family over A2, and the cost of dropping each family from the full set:

| family | add to A2 | add to A1 | change from dropping it from the full set | deploy, add to A2 |
|---|---|---|---|---|
| mechanics | +0.92 (0.29) | +2.07 (0.41) | −0.68 (0.27) | +0.10 (0.37) |
| fight and position | +0.61 (0.28) | +2.25 (0.47) | −0.19 (0.23) | +0.60 (0.32) |
| macro | +0.58 (0.27) | +0.03 (0.36) | −0.44 (0.26) | +0.57 (0.32) |
| death context | +0.16 (0.24) | +1.36 (0.43) | +0.05 (0.15), a gain | +0.11 (0.26) |
| learned embedding | +0.68 (0.31) | +1.74 (0.42) | | −0.07 (0.40) |

DB scoreboard extras over A2 (every game, no replays):

| extra | gain (SE) |
|---|---|
| **share of the team's damage, damage taken, healing, siege, XP, takedowns** | **+1.27 (0.28)** |
| stun, root, silence, teamfight, minion, creep, time dead | +0.13 (0.28) |
| talent choices (unusual picks, as-of payoff) | +0.11 (0.09) |
| consistency across games | +0.01 (0.11) |

Reading:
- **Replays carry a real but small signal.** +1.7 over the baseline is 3.3 SE from zero, about an eighth of the scoreboard channel's +14.0.
- **Death context is a scoreboard echo.** It adds +1.4 before the scoreboard and nothing after. The same holds for most of fight and position: +2.3 before, +0.6 after.
- **Mechanics are the only family that is mostly new information**, as the redundancy table predicted, and it is the costliest family to drop (−0.7).
- **The single-feature rule failed.** The two features it kept add +0.04. Signal is spread thinly across many weak features, so family-level use works and picking winners on a 5-day window does not.
- **The learned embedding is a strong fingerprint and a weak predictor.** Alone it adds +0.7, close to mechanics, which it largely re-encodes; on top of the hand-built features it adds −0.1. Hand-built features are the better use of the same replays here.
- **No per-hero signal.** Player x hero replay means add +0.1 (SE 0.5) against +1.7 for player means. Three weeks of replays are too thin per hero, and the stable parts (mechanics) are player traits.
- **Team shares are the cheapest win.** A player's share of the team's damage, damage taken and so on adds +1.3 (SE 0.3) over the scoreboard channel, on every game, with no replay file. It is in the DB scoreboard already; the model just does not use it in this form.

### 5.1 Why the deploy number is lower

Mechanics and the embedding add +2.1 and +1.8 on games that have a replay file, and +0.1 and −0.1 on deploy games. History density explains the gap:
- games with a replay file have on average 7.7 players with 5+ past replays, deploy games 3.5;
- by the number of players with 5+ past replays, deploy and replay-file games agree: mechanics +2.3 (1.1) vs +2.4 (0.5) at 7 to 10 such players; all replay features +4.9 (2.1) vs +3.2 (1.1); and about 0 at 0 to 3 such players in both;
- replay-file games also skew to higher leagues (the panel rule picks lobbies of heavy players).

So the deploy number is what to expect at today's coverage of typical lobbies. Lobbies full of players with replay history get two to three times the average gain.

### 5.2 Edge units

Holdout slots with replay history (166,385). Each player's replay score is their own contribution to the team logit in the A2 + all-replay model; the residual is the team result minus the A2 probability.

| players by replay score | realized residual |
|---|---|
| top third | +1.6 pp |
| bottom third | −0.7 pp |
| **top minus bottom** | **+2.4 pp (SE 0.3, player bootstrap)** |
| sd of the predicted shift | 1.4 pp |
| slope of realized residual on predicted shift | 0.78 |

By the player's number of past replays: 1 replay +2.6 (0.8), 2-3 +1.1 (0.7), 4-7 +3.3 (0.7), 8-15 +1.5 (0.7), 16+ +2.7 (0.5) pp top minus bottom. The predicted shift grows with history (sd 0.8 pp at 1 replay, 1.7 pp at 16+), as the shrinkage intends.

### 5.3 How lift grows with replays per player

Holdout games with 3+ players with 16+ past replays (10,056 games). Each arm uses only each player's most recent k replays, refit on fit + tune:

| replays used per player | all hand-built features over A2 |
|---|---|
| 1 | +2.14 (0.59) |
| 2 | +1.85 (0.64) |
| 4 | +2.31 (0.64) |
| 8 | +3.09 (0.71) |
| 16 | +2.53 (0.81) |
| all | +3.02 (0.85) |

Most of the gain is there with a single replay, which fits the mechanics fingerprint (0.9 stable in one game). The low-reliability families would need dozens of replays per player to pay, and the curve gives no sign that they start to.

## 6. Coverage

Share of DB 2.57 games on each holdout day with enough past replays, counting every replay downloaded so far (10-16 to 10-24, stable across days):

| players with ... past replays | mean per game | games with 3+ such players | games with 6+ |
|---|---|---|---|
| 1+ | 7.8 | 0.99 | 0.84 |
| 5+ | 5.3 | 0.76 | 0.49 |
| 10+ | 3.9 | 0.59 | 0.33 |
| 20+ | 2.6 | 0.41 | 0.17 |

Two cautions before reading this as "real lobbies":
- **We hold replays for 54% of the DB's 2.57 games.** The DB itself is about half of Heroes Profile's Storm League uploads (about 19K of 35-40K a week), and it over-represents players who upload. So these coverage figures describe the DB, and real lobbies would be lower.
- At 23K downloads a week the panel keeps pace with the DB's volume, so coverage of DB lobbies should hold or rise. For lobbies outside Heroes Profile there is no replay history at all.

At today's density the expected gain on a typical DB lobby is the deploy figure, about +1.0 (SE 0.6). Lobbies where 7+ players have 5+ replays (about a third of DB games) get +3 to +5.

## 7. Recommendation

| item | decision | why |
|---|---|---|
| team-share features (share of team damage, damage taken, healing, siege, XP, takedowns) | **add to the scoreboard channel in the next model version** | +1.3 (SE 0.3) over the baseline, on every game, from data we already have |
| replay mechanics (APM, ability casts and share, ability-mix deviation, burstiness, idle) | **wait**; rerun this test in about four weeks of panel data | the only replay family that is new information (+0.9, SE 0.3); needs dense lobbies; deploy gain +0.1 today |
| fight position and macro/soak (centroid distance, front/back, last hits, soak, structure presence) | wait with mechanics | +0.6 each over the baseline, about a third redundant with the scoreboard |
| death context (isolated, unpaid, pre-objective, late, outnumbered) | drop | +0.2 (SE 0.2) over the scoreboard and nothing when dropped from the full set |
| learned style embedding | drop for prediction | re-encodes mechanics (−0.1 on top of hand-built features); could serve as a player fingerprint for other uses |
| per-hero replay features | drop for now | +0.1 (SE 0.5) |
| talent choices, consistency, extra score-screen stats | drop | +0.0 to +0.1 |

If mechanics are built later, the cheap form is the one tested here: shrunk as-of player means, one replay already carries most of the value, team-difference entry.

## 8. Caveats

- **One patch, 27 days.** All targets are 2.57 games from 10-01 to 10-24; the holdout is 9 days.
- **The baseline is a stand-in.** The chosen model (LK2_full2) cannot be rebuilt past 2026-09-28, so A2 combines its frozen 09-28 state with a 2.57 scoreboard channel refit here. It has no MMR term, no state-space updates and no calibration layer. A stronger baseline would likely absorb some of the +1.7.
- **Positions are 15 s samples**, interpolated, so presence, soak and front/back features are coarse.
- **Two cohorts with rules.** Panel lobbies are picked for many panel players, cohort B is stratified. Neither rule reads outcomes, but the replay-file games are not a random draw of lobbies (section 5.1).
- **The embedding** was trained once with fixed settings (12 epochs, 16 dims); no tuning.

## Files

All in `training/`.

**Code**
- `replays/extract_pbp.py`: extended timelines (pbp block) per replay, parallel, sealed-id checks.
- `personalization/p3_pbp_targets.py`: DB targets, draft WP, full-history scoreboard sums.
- `personalization/p3_pbp_features.py`: per player-game features (families above).
- `personalization/p3_pbp_embed.py`: sequences and the contrastive embedding (trained on the 5090).
- `personalization/p3_pbp_eval.py`: normalization, as-of features, reliability, tune, holdout, scaling, coverage, leak check.

**Results** (`personalization/results/pbp/`): `reliability.json`, `tune_frozen.json` (the selection used), `holdout.json`, `scaling.json`, `coverage.json`.

**Gitignored, with player keys:** `personalization/cache/pbp/` (targets, rows, design, sequences, embedding). Extended timelines live on the RAID under `/home/max/hots-replays/pbp/` (about 2 GB, not in git; the 3080 copy is under `~/hots/pbp/`).

**Compute.** This box: cores 56-59, nice 19. 3080: cores 4-15 from 05:46 to 07:50, 12 workers capped at 1.6 GB virtual memory each. 5090: two embedding fits of about 3 minutes each, one process holding a 2 GB sequence array, before the definitive run started. Not used: the 3090, the 3080's other cores. No tennis processes or files touched.
