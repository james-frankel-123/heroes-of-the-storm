# Replay/Data quota allocation plan (2026-07-14)

Budget: ~215–225K Replay/Data calls/week realized on key 1 (key 2's pool is
small and consumed by the daemon). Goal: minimize weeks to (1) QM-judge
validation for papers 1+2 and (2) a publication-ready personalization
panel, with most quota going to (2).

## What each goal actually needs

**(1) QM judges.** 2021 (95.6K) and 2022 (128.2K) buckets are DONE — both
exceed judge-training volume (~100K). The missing piece that carries the
validation is the **recent bucket (games from 2025-07 onward, target
120K)**: it trains the era-matched out-of-family judge that (a) rescores
paper 1's tournament with no era caveat and (b) validly adjudicates paper
2's maintained-vs-stale contrast (the thing the 2021 QM judge could not
do). The middle years (2023, 2024, 2025H1) only fill out the QM row of the
vintage matrix — worth having, never blocking.

**(2) Personalization.** Cursor sits at April 2026 with ~300K ranked
replays covered. Two different needs:
- *Dense recent window* for #2/#5/#7 (nested lift, variance decomposition,
  smurf census): covering 2025-01 onward completely ≈ 950K games total,
  ~650K more from here.
- *Longitudinal depth* for #3 (learning curves) and the smurf
  first-N-games analysis: blind descending is the SLOW way to deepen any
  given player's history. The fast way: enumerate top panel players'
  histories via **Player/Replays (a separate 5K/wk quota pool — free
  parallelism)** and route those replay ids into a priority queue the
  refetch worker drains FIRST. Each queued replay contains at least one
  target player by construction, and usually several (high-count players
  cluster in games), so panel-depth-per-call is a multiple of blind
  scanning.

## Allocation schedule

| Phase | QM rate | Refetch rate | QM work | Weeks |
|---|---|---|---|---|
| A (now → recent bucket full) | 80/min | 100/min | recent bucket: 120K games from 2025-07+ builds | ~1 |
| B (buckets fill) | 40/min | 140/min | 2024 (90K) → 2023 (90K) → 2025H1 (60K) | ~2–5 |
| C (QM done) | 0 (idle poll) | 180/min | maintenance trickle only | 5+ |

(The two workers' client-side limits always sum to ≤180/min, key 1's rate
cap. The QM worker switches its own rate by bucket; bump the refetch rate
manually at the phase boundaries — one restart.)

**If the weekly quota moves (Zemill):** downside is already handled — QM
buckets are independently sized and era-uniform within themselves (see
strata below), so a cut just stretches the schedule without biasing any
partial state. Upside rule, decided now: a limit increase goes to
personalization (refetch) first. QM buckets have fixed targets; the dense
window is the long pole.

## Projected timelines

- **Goal (1), validate-ready: end of week 1.** Recent bucket done →
  train the 2026-era QM judge → rescore the paper-1 tournament and the
  paper-2 W6 contrast with an era-valid out-of-family judge. Full QM
  vintage row (all buckets): ~week 5.
- **Goal (2), publication-ready: ~week 5.** Dense window: ~650K more
  replays at 100→140→180K/wk lands the complete 2025+2026 coverage around
  week 4–5. Longitudinal: the Player/Replays enumerator starts now
  (separate quota), targets the top ~3–5K panel players (~15–25K calls,
  3–5 weeks at 5K+0.5K/wk), and its priority queue rides inside the
  refetch quota from week 1, so learning-curve and smurf panels deepen
  while the dense window fills rather than after.

Versus the current blind 50/50: goal (1) moves from ~never (the ascending
scan was 3+ months from reaching 2026 games) to 1 week; goal (2) moves
from ~7+ weeks to ~5, with the longitudinal panels arriving weeks earlier.

## Implementation (2026-07-14)

- `sync/fetch-qm.ts`: era-bucket priority mode. Buckets defined by game
  DATE windows; the worker resolves each bucket's build set and scan-start
  id from the ranked corpus at startup (version→date-range map), counts
  progress from qm_games.game_date (resume-safe), advances buckets at
  target, sets its own rate per phase (80 → 40/min). A build belongs to a
  bucket if its active date RANGE overlaps the window — matching on the
  build's first-seen date alone excluded every window-straddling build
  (caught at launch: the recent bucket skipped 240K listings with zero
  matches before the fix). Within each bucket the scan runs 12 STRATA
  round-robin (cursors at ranked-id quantiles of the window), so a
  partially filled bucket is an era-uniform sample, never a
  first-months-only prefix.
- `sync/refetch-players.ts`: rate 100/min (bump to 140 at phase B); batch
  selection drains `player_fetch_queue` first, then continues descending.
- NEW `sync/enqueue-player-histories.ts`: Player/Replays enumerator
  (chunked date windows to dodge the endpoint's large-history timeouts),
  targets panel players by game count, inserts uncovered replay ids into
  the queue. Runs on its own quota; safe to invoke weekly. Seed run =
  top-500 heaviest (right for queue value density). Expansion runs to
  3-5K use `--stratified`: equal counts per activity-tier quartile, so
  personalization models do not train disproportionately on the heavy
  tail (more engaged, better, flatter learning curves than the median
  player). Panel analyses (smurf census, learning curves) can weight or
  filter by tier at analysis time.
