# Cycle 2 preparation: explanation cards, mechanism queries, sessions w2-K-*

Date: 2026-10-15. Spec: `FRAMING.md`. Previous: `WAVE0_REPORT.md`. Code: `training/scilearn/`. Private state: `/home/max/scilearn-private/`. The three cycle-2 sessions are prepared and fitted. **None is launched** (section 7).

## 0. At a glance

| Item | Result |
|---|---|
| Wave-1 close-out | 3 brokers stopped; all ledger chains verify; w1-K-b c07 and c10 had finished after the session stopped polling (both rejected, ledger seq 64-67); closeout records written |
| "harness ERROR" on w1-K-blinded c03, c05 (and c13) | Root cause: two Q2 threads launched the same incumbent refit after an accept, and hotsjob refused the second launch. Fixed with a per-key lock and an idempotent `remote.submit`. Regression test 7/7; the old code fails 4 of 7 |
| Second bug, found on the way | Near-miss retries were never charged double. Fixed. No wave-1 card was retried, so no charge was wrong |
| New card type `explain` | A kit or role theory scored against an accepted table: explain, transfer, onset, kit permutation. Synthetic fixtures 15/15, including 0 of 40 null theories accepted |
| Explanation null canaries on the real wave-1 tables | **0 of 8 accepted** (one per table, both sessions; explain z from -2.60 to 0.48). A first run found and fixed a confirm-table bug (section 3.4) |
| Replay data (M) | 15,508 of 15,508 pushed files parsed. 14,414 M games: 9,526 M-test, 4,888 M-explore (3,993 cohort B, 895 surprise_v1). Usable now |
| Q3, Q5, mechanism tests | Wired per FRAMING 5.2 and 9.2, with the M-explore / M-test split. End-to-end mechanism test on 9,526 shuffled M-test games: not shown (z = -0.96), as it should be |
| Existing suites after the changes | ledger 5/5, planted-leak fixtures 17/17, leaf 9/9, sandbox escape 10/10 |
| Sessions | w2-K-b, w2-K-blinded, w2-K-shuffled prepared from their wave-1 predecessors; incumbent, explore and confirm fits done |

## 1. Wave-1 close-out

### 1.1 Final state of the three sessions

| Session | Cards | Accepted | Rejected | Near miss | Harness error | Dev bits charged (of 120) | Sandbox numbers (of 5,000) |
|---|---|---|---|---|---|---|---|
| w1-K-b | 9 | 5 | 3 | 1 (c07) | 0 | 13.5 | 180 |
| w1-K-blinded | 13 | 5 | 4 | 1 (c06) | 3 | 15.0 | 1,517 |
| w1-K-shuffled | 4 | 0 | 4 | 0 | 0 | 6.0 | 748 |

- **w1-K-b c07 and c10.** Both finished after the session's poller stopped at 23:44 UTC and are in the ledger:
  - c07_recent_practice (decay, 14-day half-life): z = 2.80 against 3.00, invariance failed, released gain 0.0000. Rejected, flagged near miss.
  - c10_doublesupport_high (rule): z = 0.98 against 4.02 (its group already held 4.5 bits), released gain 0.0000. Rejected.
  - The session's summary says 10.5 bits; the ledger says 13.5, because both tests were charged when they finished.
- **Brokers.** All three stopped with `session stop`. A `closeout` record at the end of each ledger lists every card's final status. The three error cards in w1-K-blinded got retroactive `error` records (0 bits).
- **Ledgers.** `python -m training.scilearn.ledger verify`: every chain verifies (w1-K-b 68 records, w1-K-blinded 922, w1-K-shuffled 309, harness 65).
- **Q2 canaries of cycle 1** (harness ledger seq 17-38): null canaries accepted 0 of 12 across the three sessions (z from -1.96 to 1.71). Known-true canaries: w1-K-b 2 of 2, w1-K-blinded 1 of 2 (the role multiset, z = 2.47, was redundant with the session's own accepted c01), w1-K-shuffled 0 of 2 (labels shuffled, so both are null there).

### 1.2 The "harness ERROR" on c03, c05 and c13

**What happened.** w1-K-blinded accepted c01 at 18:40:21. It then submitted c02 and c03 at 18:41 and c04 and c05 at 18:42. Each `hyp test` starts a thread that first calls `ensure_incumbent`. After the accept, the incumbent (K0 + c01) was not in the registry yet. So c02's thread and c03's thread both took one of the two remote slots and both launched the remote job `inc_0b03640706bb7e85`. `hotsjob launch` refused the second with "scilearn_inc_0b03640706bb7e85 is already running", the exception left `finish_q2`, and c03 ended in status `error`. c05 took the freed slot while c02's refit was still running and failed the same way. The same race hit c13 at 20:50 on `inc_0129806d306d28af`, after c10's accept. The traceback is in `w1-K-blinded/logs/broker.log`. w1-K-b escaped by timing: its tests had all started before its first accept.

**Fix** (commit 3b1880e):
- `harness.ensure_fit` holds an exclusive flock per incumbent key (`locks/inc_<key>.lock`) while it fits, and rereads the registry inside the lock. Waiters then find the finished fit. Registry writes are atomic under their own lock.
- `remote.submit` asks the worker for the job's state first. A running job is attached to, and a completed job with a result is reused, never relaunched.
- `finish_q2` works on a copy of the session state taken under the broker lock.
- A harness error is now written to the session ledger (kind `error`, 0 bits), and the card can be tested again with `hyp test`, free.

**Test.** `PYTHONPATH=. python -m training.scilearn.tests.test_incumbent_race` drives the real `ensure_incumbent` and `remote.submit` against a fake worker that refuses a second launch, as hotsjob does, from 6 threads at once. New code: 7/7 (one launch, no exception, every caller gets the same job, one ledger fit record, attach and reuse both work). Old code: 3/7, with the wave-1 "already running" error.

**Second bug.** `apply_gate` decided "retry of a near miss, charge double" from `card["status"]`, which is always `testing` at that point. Retries were charged 1.5 bits instead of 3, and a retry could be flagged near miss again. The gate now reads the status the card had before the test.

## 2. What changed in the harness

| Area | Change |
|---|---|
| `explain.py` (new) | Table cell extraction, theory features, nuisance projection, cross-fit, leave-heroes-out, permutation test, onset, gate |
| `replay_features.py` (new) | M membership from the frozen rules, 18 mediators per game and team, store `rf1` |
| `worker.py` | Job kinds `explain` and `mtest` |
| `harness.py` | Race fix; explore (S) and confirm (Dpre+G) table fits; explanation and mechanism dispatch and gates; explanation null canaries; known-true Q2 canaries scored against the starting backbone; small fits may run on the 5090 CPU |
| `broker.py` | Q3 queries, `inspect` (Q5), `table`, `explain` and `mechanism` cards, `mtest`, mediator budget, group bits carried over from a predecessor session, harness errors logged and retestable |
| `session.py` | `prepare --from <predecessor>`, `reprompt`, `explain-canaries`; `prefit` also fits the explore and confirm tables |
| `session_kit/` | `hyp table`, `hyp inspect`, `hyp mtest`; guide sections for explanation and mechanism cards; `prompts/cycle2.md` |
| crontab | `replay_features build` daily at 05:30 local, after the 22:40 and 23:50 parse cycles |

## 3. Explanation cards

### 3.1 What a card claims

An accepted lookup table (hero x map, hero x tier, ally pairs, hero-vs-hero counters) is the model's accurate layer. An explanation card names one table and proposes a small function of kit axes or roles that reproduces its pattern, with at most 30 parameters:

| Table | Features |
|---|---|
| hero x map | `{kit: axis, by: map | map_class | lanes2}`, `{role: R, by: ...}` |
| hero x tier | `{kit: axis, by: tier}`, `{role: R, by: tier}` |
| ally pairs | `{kit_pair: [a, b]}` (symmetric), `{kit_sum: axis}` (stacking), `{same_role: true}`, `{role_pair: true}` |
| counters | `{kit_pair: [a, b]}`: a_mine x b_enemy minus the reverse, `{role_pair: true}` |

Any feature can carry a declared sign, enforced by a bounded fit. The card adds no term to the incumbent. An accepted explanation is recorded in the session state with its coefficients. Using it as the table's shrinkage target in a refit is the natural next step, and is not built.

### 3.2 How a card is scored

**Cells.** For each table cell j, the incumbent fit gives a posterior mean beta_j, a diagonal Laplace variance v_j and the block's prior precision lam. The data-only estimate is b_j = beta_j (1/v_j) / (1/v_j - lam), with variance s_j = 1/(1/v_j - lam) and weight w_j = 1/s_j. Hero main effects, plus context main effects for hero x context tables, are projected out of the target and the features by weighted least squares, because hero strength lives in other terms. A theory is credited only for interaction structure.

**Score.** For a held-out cell, d_j = w_j [r_j^2 - (r_j - f_j)^2], where r is the projected target and f is the theory's out-of-fold prediction. Sum(d) is the drop in weighted residual sum of squares against "explains nothing". For a theory with no content it is at most 0 in expectation. The signal sum of squares is S = sum(w r^2) - (n - rank N), since weighted noise contributes 1 per cell. The explained share is sum(d) / S. With Gaussian cell noise, sum(d) / (2 ln 2) is the number of bits the theory saves on held-out cells.

**Data.** Sessions explore on an S-only fit of their incumbent (`hyp table`, one sandbox number per cell shown). Cards are scored on a **confirm fit on Dpre+G**, which is disjoint from S and from the dev tranche Dpost. Only the onset check reads Dpost.

| Check | Data | Pass rule |
|---|---|---|
| (a) explain | Confirm table; 10 keyed folds over cells; SE by a bootstrap over hero clusters | z >= 3.0 + 0.59 sqrt(I_g), with I_g the bits already on group `explain:<table>`; **and** compression: bits saved > 0.5 k log2(cells), the theory's description cost |
| (b) transfer | Confirm table; theory fit with no cell of 20 O3 heroes (>= 20K S games, keyed hash, 5 folds), scored on their cells | z >= 2.0 and sum(d) > 0 |
| (c) onset | Dpost O2 slices: first 10K games of each 2.55.17 build (40K games); the incumbent's table shrunk toward the theory instead of toward 0 | paired log-loss z >= -2.0 (must not hurt); the first-1K slices are reported too |
| (d) kit permutation | Kit features recomputed from 200 within-role permutations of the kit table, scored as in (a) | the real axes beat 95% (p <= 0.05). Skipped for role-only theories |

**Charge and release.** 1.5 dev bits per test, to the group `explain:<table>`. That raises the bar for later theories of the same table, as on the Q2 gate. Released: the decision, the explained share and the transfer share (rounded to 0.01, noise sd = SE), whether it compresses, the onset gain (to 0.0001 with noise), whether it beats the kit permutations, and the reasons for a rejection. Everything else stays in the private ledger record.

**Why (d).** Within-role permutation keeps each role's mean kit values, so a kit theory whose real content is role-level (for example "ranged assassins do better on three-lane maps") would pass (a) and (b) on role information alone. (d) sends that credit to the honest role theory.

### 3.3 Fixtures

`PYTHONPATH=. python -m training.scilearn.tests.test_explain`: **15/15**. Synthetic world: 72 heroes in 6 roles, 12 random kit axes, 15 maps, cell precisions drawn like real tables.

| Fixture | Result |
|---|---|
| Hero x map, planted waveclear x lanes (planted share about 0.5) | Accepted: share 0.51, z 5.8, transfer z 2.6, permutation p 0.010; coefficient has the planted sign |
| Same table: unrelated axis (siege); waveclear permuted across heroes; kit permuted within role (the canary) | All rejected (shares -0.007, -0.001, -0.002) |
| Role-level truth, kit axis carrying only role means | Kit theory rejected by (d) (share 0.30, p 0.079); the role theory accepted (share 0.39) |
| Counters, planted hard_cc x mobility | Accepted: share 0.31 against planted 0.30, z 5.7, transfer z 3.7, p 0.010 |
| Same table: unrelated pair; within-role permuted kit; wrong declared sign | All rejected |
| Onset, thin cells (20 to 200 games of precision) | True theory prior: gain 0.00078, z 3.4. Null theory prior: z 1.0 |
| Null calibration: 40 single-axis theories on a table with no kit structure | 0 of 40 accepted |

### 3.4 Null canaries on the real wave-1 tables

`session explain-canaries <sid>` ran one null canary per accepted table in w2-K-b and w2-K-blinded, 8 in all. Each canary is a fixed kit theory for its table type, computed on a kit permuted within role under a canary key, and scored by the full gate against the session's real table. Results are in the harness ledger, seq 58-65, gate threshold 3.0 (no bits on the group):

| Session | Table | Cells | Theory (permuted kit) | Explained share | z | Bits saved / cost | Transfer z | Onset z | Perm. p | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| w2-K-b | c02 hero x tier | 360 | early_vs_late x tier | 0.022 | 0.48 | 7.3 / 12.7 | 1.24 | -0.76 | 0.095 | reject |
| w2-K-b | c04 hero x map | 1,170 | waveclear, global presence x map class | -0.005 | -1.79 | -6.6 / 10.2 | -2.19 | 0.55 | 0.677 | reject |
| w2-K-b | c05 ally pairs | 3,901 | hard_cc x burst, healing stacking | -0.003 | -1.20 | -3.1 / 11.9 | -0.69 | -1.28 | 0.701 | reject |
| w2-K-b | c06 counters | 3,997 | hard_cc x mobility, burst x shields | -0.002 | -1.29 | -3.4 / 12.0 | -1.29 | -1.74 | 0.433 | reject |
| w2-K-blinded | c04 hero x tier | 360 | early_vs_late x tier | 0.016 | 0.43 | 5.7 / 12.7 | 1.78 | -1.32 | 0.104 | reject |
| w2-K-blinded | c10 ally pairs | 1,595 | hard_cc x burst, healing stacking | -0.009 | -2.60 | -4.5 / 10.6 | -2.17 | 1.64 | 0.841 | reject |
| w2-K-blinded | c11 counters | 2,029 | hard_cc x mobility, burst x shields | 0.000 | 0.04 | 0.2 / 11.0 | 0.24 | 0.91 | 0.134 | reject |
| w2-K-blinded | c12 hero x map | 1,170 | waveclear, global presence x map class | -0.002 | -0.42 | -3.3 / 10.2 | -1.17 | -0.80 | 0.453 | reject |

- **All 8 fail (a) and the compression charge.** None comes near z = 3; the largest is 0.48. Each check was run on real tables of 360 to 3,997 cells. Pooled with the 40 synthetic null theories and the 12 cycle-1 Q2 null canaries, that is 0 accepts in 60 null trials.
- The hero x map confirm tables keep 1,170 of the incumbent's 1,260 cells; the 90 with no Dpre+G data are dropped by the precision filter.
- **No real-kit theory was scored, on purpose.** The harness ran only permuted-kit canaries, so that no real-kit result could leak into the prompts I wrote. Whether any kit theory explains these tables is the sessions' question.

**Bug found by the first canary run.** The first run (seq 52-55) used confirm tables fit with each family's `min_games` vocabulary recomputed on Dpre+G alone. On 330K games the blinded ally-pair table (min_games 10,000) kept 18 cells instead of 1,595. The canaries still all rejected, but explanations would have been scored on a different, tiny table. Fix (commit 1202c4e): the confirm fit now freezes every family's cell vocabulary to the incumbent fit's, through a harness-only `_vocab` key, which sessions cannot set because the broker strips `_` keys. The confirm tables were refit (inc_f28677e767901486, inc_e64b055e0ee64bb2, 3 minutes each on the 5090) and the canaries rerun (seq 58-65, above). Planted-leak fixtures after the library change: 17/17.

## 4. Replay mechanism queries (Q3, Q5, mechanism tests)

### 4.1 Data

- **Parsed:** the 22:40 parse cron on Oct 13 pushed all 15,000 cohort-B files and parsed them on 3090 cores 24-35. With surprise_v1, 15,508 of 15,508 files parsed, 0 failures (G7 parse success 100%).
- **M membership** comes from the frozen rules and reads no outcome. It covers 2.57 games only. Sealed IDs are skipped before any file is opened. Cohort-B games outside the 10% pool are split by the keyed `m_split` hash into M-explore (30%) and M-test (70%). surprise_v1 games come from the pool and go to M-explore only. C, F and audit games are never opened.

| | Members | Parsed, in store `rf1` |
|---|---|---|
| Cohort B, M-test | 9,526 | 9,526 |
| Cohort B, M-explore | 3,993 | 3,993 |
| surprise_v1, M-explore | 895 | 895 |

- **When M became usable:** on Oct 13 only 390 M-test games existed. The feature store rebuilt by cron at 09:30 UTC on Oct 15 holds all 9,526. M grows with each week's cohort-B downloads (cap 8,300 a week from week 2).
- **Caution for the C batch.** Snapshot `snap_20261013` was cut before any cohort-B download, so it lists today's B games under C. By the tranche rule (`tranches.assign` checks cohort B before C) they are M. The C batch runner must recompute membership at unlock and drop them, or C scores games the mechanism tests have used.

### 4.2 Feature store (FRAMING 9.1, item 12)

`python -m training.scilearn.replay_features build` writes one row per game and team to `scilearn-private/replay_features/rf1/`: covariates, mediators, a separate label file, and a manifest with code hash and counts. It is deterministic and versioned. It reads 15K timelines in under a minute on one core.

18 mediators, all with fixed early windows. End-of-game totals and game length are never mediators.
- **Levels and XP:** level lead at 5, 7, 10 and 15 minutes; first to level 10; XP lead at 10 minutes; minion, merc-camp and hero-kill XP shares at 10 minutes.
- **Deaths:** deaths before 10 minutes; isolated deaths (no other death within 15 s and 30 map units); first blood.
- **Fights:** chains of 2 or more deaths with gaps of at most 15 s. Fights won before 10 minutes, and from 10 to 20 minutes.
- **Structures and mercs:** towers lost by 15 minutes; forts and keeps lost by 15 minutes; first fort or keep lost; merc camps by 10 minutes.

rho (mediator-to-win correlation, which sets the Q3 charge) is measured on cohort-B M-explore games only, because surprise_v1 was selected with outcomes. Measured values run from 0.09 (merc-camp XP share) to 0.64 (level lead at 15 minutes); level lead at 10 minutes is 0.52.

Not done: hand validation of lane assignment and fight detection on 50 games (FRAMING risk 7), the per-map objective dictionary, and the day-2 SNR check (FRAMING 9.2). The SNR check can run now that M exists.

### 4.3 What sessions can do

| Class | Command | Data | Release and charge |
|---|---|---|---|
| Q3 mediator cell | `hyp query '{"class":"Q3","mediator":...}'` | M-test | Cells of at least 200 team rows, at most 20 cells, noise sd = 2 SE; 0.16 rho^2 mediator bits per number, 8 bits per group, 60 per cycle |
| Q5 game record | `hyp inspect [--hero] [--map] [--interesting|--random]` | M-explore | One full record, without player IDs or action rates; the game is spent; 300 per cycle |
| Mechanism test | a card's `mechanism` block, or `type: mechanism, of: <card>`, then `hyp mtest <id>` | M-test | The term's fitted contribution must predict the declared mediator in the declared direction, z >= 3.0 + 0.59 sqrt(I) on group `mech:<mediator>`, with cluster-robust SE by day. The negative control must have \|z\| < 2. 1.5 mediator bits, once per card. A pass replaces `unexplained` with `mechanism_ok`. Opens at 2,000 M-test games |

By arm:
- **Shuffled:** whole M records (trajectory plus outcome) are permuted within map x tier x build blocks under the session's shuffle key, for Q3, Q5 and mechanism tests. An inspected record shows this game's draft with another game's trajectory, and its talent names are removed.
- **Blinded:** H and M labels; no build, talent names, map events or unit names.

**End-to-end check.** On the shuffled smoke session `smoke2-shuf`:
- a role-multiset card went through `hyp card submit`, `hyp test` (rejected, gain 0.0000) and `hyp mtest` with mediator level_lead_10 and negative control minion_xp_share_10;
- the mechanism test ran remotely on 9,526 shuffled M-test games and returned "mechanism not shown" (z = -0.96 in the declared direction, standardized effect -0.02, negative control quiet);
- 1.5 mediator bits were charged.

The same smoke sessions exercised Q3, `inspect`, explanation and mechanism card submission, refusals, and the id-collision guard in all three arms.

## 5. Session prompts for cycle 2

`session prepare <sid> --arm <arm> --cycle 2 --from <w1 sid>` makes each w2 session a continuation of its wave-1 session:

- **Incumbent:** K0 plus the predecessor's accepted terms. w2-K-b and w2-K-blinded each hold a role multiset, hero x tier, hero x map, ally pairs and counters. w2-K-shuffled is K0.
- **Keys:** the shuffle key and the blinding key carry over, so the shuffled world and the H-labels are the same as in wave 1.
- **Bits:** bits already charged to each Dpost cell group carry over as `prior_group_bits`. The data is the same, so the gate keeps paying for earlier looks. Session-level budgets reset: 120 dev bits, 60 mediator bits, 40 cards, 300 inspections.
- **Card ids:** ids used in wave 1 are refused, so no term name can collide.
- **Prompt:** `standard.md`, plus `cycle2.md`, plus a harness-written digest of the previous cycle (every card, its term and its released result, in the session's own labels), plus the arm's extra (blinded). The predecessor's own `summary.md` is in the view as `previous_cycle_summary.md`.

What `cycle2.md` tells the sessions:
- Accepted tables are in the incumbent and are the accurate layer. A table spans every kit term of its shape, which was the main lesson of wave 1.
- The main job is explanation cards and mechanism cards. Predictive cards are still welcome for forms no table spans.
- Plan for at least 25 cards, and stop earlier only if the budget runs out. Wave-1 sessions stopped at about 10% of their budgets.
- Do not retry a theory with small variations; each test raises its group's bar.
- The summary must report each explanation's share, transfer and onset results, and each mechanism result.

The shuffled arm gets the same prompt and is still not told that its labels are shuffled.

## 6. Fits for cycle 2

| Session | Fit | Job | Rows | Parameters | Time |
|---|---|---|---|---|---|
| w2-K-b | incumbent, S+Dpre+G | inc_05e1a6e6fcae9609 | 2,275,052 | 18,874 | 122 min (3090) |
| w2-K-b | explore, S | inc_566980a9ddb645d4 | 1,946,097 | 18,830 | 96 min (3090) |
| w2-K-b | confirm, Dpre+G, cells frozen | inc_f28677e767901486 | 328,955 | 18,874 | 3 min (5090 CPU) |
| w2-K-blinded | incumbent | inc_5287edf42562e12b | 2,275,052 | 14,557 | 117 min (3090) |
| w2-K-blinded | explore | inc_a555a4dee8a22559 | 1,946,097 | 13,996 | 106 min (3090) |
| w2-K-blinded | confirm, cells frozen | inc_e64b055e0ee64bb2 | 328,955 | 14,557 | 3 min (5090 CPU) |
| w2-K-shuffled | incumbent (K0, shuffled labels) | inc_a3d8db7a258a8df1 (wave 1) | | 9,157 | cached |

The w2-K-b incumbent sits near the 20K parameter cap (18,874), because its counter and pair tables use min_games 500. A large new family on top of it will be refused at compile time.

## 7. Launch commands (not run)

```bash
cd /home/max/heroes-of-the-storm
export PYTHONPATH=. PY=/home/max/venvs/scilearn/bin/python

# 0. already done: prepare and prefit (prefit is cached, so rerunning it is instant)
#    $PY -m training.scilearn.session prepare w2-K-b        --arm b        --cycle 2 --from w1-K-b
#    $PY -m training.scilearn.session prepare w2-K-blinded  --arm blinded  --cycle 2 --from w1-K-blinded
#    $PY -m training.scilearn.session prepare w2-K-shuffled --arm shuffled --cycle 2 --from w1-K-shuffled
for s in w2-K-b w2-K-blinded w2-K-shuffled; do $PY -m training.scilearn.session prefit $s; done

# 1. brokers (one per session; the socket is the session's identity)
for s in w2-K-b w2-K-blinded w2-K-shuffled; do $PY -m training.scilearn.session broker $s; done

# 2. proposer sessions: claude2 account, claude -p --model claude-opus-5-5, in the sandbox
for s in w2-K-b w2-K-blinded w2-K-shuffled; do
  nohup $PY -m training.scilearn.session launch $s > /home/max/scilearn-sessions/$s/logs/launch.out 2>&1 &
done

# 3. once per session per cycle: the 6 Q2 canaries (harness ledger only). Known-true
#    canaries are scored against the K0 backbone.
for s in w2-K-b w2-K-blinded w2-K-shuffled; do $PY -m training.scilearn.session canaries $s; done
#    explanation null canaries were run before launch (section 3.4)

# monitoring and teardown
$PY -m training.scilearn.ledger verify
tail -f /home/max/scilearn-sessions/w2-K-b/logs/broker.log
$PY -m training.scilearn.replay_features status
for s in w2-K-b w2-K-blinded w2-K-shuffled; do $PY -m training.scilearn.session stop $s; done
```

- `session launch <sid> --dry-run` prints the exact `docker run` line.
- If the guide or prompts change before launch, `session reprompt <sid>` rebuilds that session's view and prompt. It refuses once the session has been launched.
- Smoke sessions `smoke2-b`, `smoke2-blinded` and `smoke2-shuf` exist from testing. They have their own ledgers and are not part of the study.

## 8. Compute used

- **3090** (cores 24-35, nice 19, two slots):
  - 4 cycle-2 fits, 96 to 122 minutes each, at most 2 at once, about 11 GB each;
  - 8 explanation canaries;
  - the mechanism-test check.
  - The nightly parse also ran there, from cron.
- **5090** (CPU only, under 11 GB): 4 confirm fits of 2 to 3 minutes; 2 of them superseded by the cell-freeze fix.
- **3080 and 3090 cores 0-23:** not touched.
- **This box:** fixtures, broker smoke tests, the escape suite and the feature-store build, at nice 19 on cores 60-63.

## 9. Open items

1. **C batch membership.** Recompute M at unlock (section 4.1).
2. **Explanations as shrinkage targets.** An accepted explanation is recorded but does not yet change the table's prior. Building that is a fitter change (prior mean per cell) for cycle 3.
3. **Mediator validation.** Hand-check fights and isolated deaths on 50 M-explore games, and run the SNR check, before family 3 and 5 mechanism results count.
4. **Parameter cap.** w2-K-b is at 18,874 of 20,000 parameters.
5. **Process note.** Twice, `pkill -f` and `pgrep -f` matched my own shell, as the memory note warns. Once this killed the first prefit driver; the race fix made the restart safe, because it attached to the jobs that were still running.
