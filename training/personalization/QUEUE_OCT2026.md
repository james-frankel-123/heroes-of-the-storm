# Personalization lane: October 2026 queue

Started 2026-10-05. Sources: Fable review (sections 1, 6, 7), Astra review (folded in when it lands), `results/P3_AUDIT_FIXES.md`.
Rules for every item: research stays on the 90-hero v1 set; snapshot data and builds up to 2.55.17.98025 only, and no game from a build released after 2026-09-27; no pre-registration; bugs are fixed and rerun, and the write-ups carry the new number.

## Compute and data

- Main box: git, editing, tiny checks, and the one-time read-only exports from the local Postgres (`DATABASE_URL_RESEARCH`, run under `nice -n 19`). Never Neon.
- 3090 (`max-windows-3090`): all CPU work, under `hotsjob --mem-max`, with memmon running. At most 8 threads in total and at most 48 GB in total for this lane, so the host stays under about 100 GB with paper 1's ~30 GB. Small GPU training jobs (GD and BC refits, kernel parity) only when `nvidia-smi` shows room, and each stays under 6 GB.
- 3080: only personalized MCTS search slots after drift (at most 1), and at most 30 GB.
- Data reaches the remotes as frozen files (`p3_data_manifest.sh` plus `sync.sh`). The remotes get no DB credentials.
- Every job is a hotsjob plain job whose stages write their outputs atomically and skip finished outputs, so a pause loses only the stage in progress. Waiters poll `hotsjob.py state` and write alerts to `training/personalization/logs/ALERT_p3`.

## Queue

| # | item | host | depends on | ETA |
|---|---|---|---|---|
| A1 | `p3_heroes.py`: NUM_HEROES from `shared.py`, v1 asserted, slot-table assertion; replace every hard-coded 90/89/128 hero size | main (code) | none | Oct 6 |
| A2 | `prepare()` default becomes `prepare_l1` (lag-1, upload-order-free counts) | main (code) | none | Oct 6 |
| A3 | `hero_level_causal`: v2 band strings normalized, latest stamp parsed before the game started; NULL means unknown (no more 99); used by `p3_x_adopt`, `p3_x_learn`, `p3_x_newacct` and the smurf detector | export: main; build: 3090 CPU | parse-time column export (`fetched_at`) | Oct 7 |
| A4 | Player key (region, blizz_id): a NULL region is resolved to the account's unique known region, and is otherwise kept as an explicit unknown region; `p3_nested_lift` keyed the same way | main (code), 3090 rebuild of `hs_slots` | none | Oct 7 |
| A5 | Drafter harness: team-level pick-to-player mapping (picker, not final owner, with trades handled by assignment); no real hero added to the candidate list; train-only GD (5 variants) and BC prior (fit on games before 2026-02-10) replace the 98%-random models everywhere (rollouts, opponent model, imitation feature, prior) | 3090 GPU (small) + CPU | snapshot on the 3090 | Oct 7-8 |
| A6 | Sampling clamp at hero 89 fixed (sample among available heroes only); lobby ids deterministic (sorted, not `imap_unordered` order); monkeypatched protocol replaced by explicit arguments | main (code) | none | Oct 6 |
| A7 | Rerun what the fixes touch: the headline and phase-2 tables, adoption, learning, new-account and smurf, nested lift, one-step drafter and ban model | 3090 CPU | A1-A6 | Oct 8-9 |
| B1 | Fable #1: joint combiner of every tested term (status table, main share and forced-off-main, fine role, EWMA-100, rust, party, causal hero MMR), fit on V1 with ridge, scored on V2 and OOT, with leave-one-out contributions and a skill-coefficient gate of 3.4 to 4.0 | 3090 CPU | A7 | Oct 9 |
| B2 | #6: calibration along the drafter's trade-off direction (V2 teams binned by high ΔS with low WP_pop, and the reverse) | 3090 CPU | A7 | Oct 9 |
| B3 | #11: MMR mean reversion for "form" (long-run minus current causal MMR, lobby causal MMR imbalance) | 3090 CPU | none | Oct 8 |
| B4 | #7: one-trick pooling with concentration-dependent prior variance (capped player level; main-share-scaled kernel), judged against the 8.1pp main-ban natural experiment | 3090 CPU | A7 | Oct 10 |
| B5 | #9: state-space fixes (sum-to-zero heavy fit, recency variance scale, jump variance a·\|ΔWR\|) | 3090 CPU | A7 | Oct 10 |
| B6 | #3: causal (build, hero, tier) demeaning of residuals before the GP and the state-space model | 3090 CPU | A7 | Oct 11 |
| C1 | Backfill watch: daily `refetch-players --status`; trigger at fewer than 5K remaining or Oct 16, whichever comes first | main (tiny check) | quota (resumes about Oct 8) | Oct 12-15 |
| C2 | Full-history export (read-only, local store, builds up to 2.55.17.98025, Storm League), shipped to the 3090; rebuild the slot tables | main export, 3090 build | C1 | +1 day |
| C3 | #2: lifetime-experience axis (hero_level_causal bins × window games, plus account status), then the headline rerun on full history | 3090 CPU | C2, A3 | +2 days (about Oct 15-18) |
| D1 | Port the comp-fallback fix (`COMP_FALLBACK_MID_HIGH_LOW`, `COMP_FALLBACK_ORDER`) into `cuda_personal`, `cuda_prior` and `cuda_pgd`; hosts pass `search_mode` explicitly (chance is the headline, rollfwd is an arm); guard that refuses to run unless the order is mid_high_low and search_mode is set | main (code) | none | Oct 6 |
| D2 | Build the three kernels on the 3090 (sm_86), then a parity test (kernel leaf against the float64 Python mirror, 1e-5 or better) and a CPU trace test that the tree reaches a later own turn | 3090 (small GPU) | D1, A5 models | Oct 7-8 |
| D3 | #10: team-level pick plus assignment in the drafter (the drafter chooses a hero and the end-of-draft assignment) | 3090 CPU (one-step), GPU via D5 | A5 | Oct 10 |
| D4 | #8: ban model in the drafter, calibrated to the natural experiment (personal GD pick probability for mains, cross-fit argmax, per-tier base rates, premade control) | 3090 CPU | A7 | Oct 10 |
| D5 | Personalized MCTS waiter: launches only after the drift waiter has launched all 30 agents (its log says so and every `v2_*` manifest exists) and slots are free (3090 at most 2 MCTS in total, 3080 at most 1, no host in setup or paused); queue: full s400, real s400, curve s25/100/1500, collapse, distilled prior, personal-GD R1-R3, ban-step 1,500-sims arm, rollfwd arm, team-level arm | waiter on main (ssh only); jobs on 3090/3080 GPU | D2, drift waiter done (about Oct 14) | armed Oct 8; runs Oct 14-18 |
| E1 | #4: versioned causal WP scoring chain (vintage archive, latest vintage before each game's build, rescore since the last vintage) as code with tests | main (code) | none | Oct 11 |
| E2 | #12: regression battery and deployment gates (golden V2/OOT sets, slope, ECE, coefficient, per-n slopes, visibility, hero-set assertions) | main (code), 3090 run | A7 | Oct 11 |
| E3 | 5.5: nightly `player_skill_state` table builder keyed (region, blizz_id, hero) with the WP vintage id; not deployed | main (code) | E1 | Oct 12 |

Out of scope by instruction: Fable 2.57 prospective test (5.6) and any pre-registration.

## Milestone reports

1. Fixes landed (A1-A6 committed, A7 numbers).
2. B results.
3. MCTS waiter armed (D1-D2 verified, D5 running).
4. Reruns: C3 and D5 results.

## Status log

- 2026-10-05: plan written. Backfill 331,178 replays left below the cursor. Paper 1 holds the 3090 (2 MCTS) and the 3080 (1 MCTS). The drift waiter is waiting for HANDOFF.
