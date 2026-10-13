# Wave 0 report: the scientific-learning harness

Date: 2026-10-13. Spec: `FRAMING.md`. Brief: `WAVE0_BRIEF.md`. Code: `training/scilearn/`. Private data, keys, ledgers and fit outputs: `/home/max/scilearn-private/` (never in git, never mounted into a sandbox). Session dirs: `/home/max/scilearn-sessions/`.

## 0. Acceptance at a glance

| Check | Result |
|---|---|
| Sandbox escape tests | **10/10 blocked** (`.env`, Postgres on 4 routes, label files, SSH to both workers, proxy abuse, read-only views, docker socket, host processes, other transcripts; broker reachable as the positive control) |
| Planted-leak fixtures | **17/17**: 8 planted leaks caught, 2 legitimate builders pass all 3 invariance checks, swap antisymmetry exact (0.0) on all 12 term types |
| Ledger tamper tests | **5/5** (edit, delete, reorder, rehashed edit all detected) |
| Dry-run hypothesis, real labels | Card `synergy2` (ally-pair family) went sandbox -> `hyp` -> broker -> remote fit -> gate -> ledger and was **accepted**: z = 8.95 vs threshold 3.72, sign held in 6/6 recent S builds, safety probes within tolerance |
| Shuffled-outcome dry run | The same card on shuffled labels: **rejected**, z = 0.14. The no-healer card: rejected, z = 0.02. The shuffled incumbent scores 50.0% (slope 0.01) |
| Canaries (12 trials over the two dry-run sessions) | 0/8 null canaries accepted; 1/2 known-true canaries accepted in the real session (the other was redundant with the accepted card); 0/6 accepted in the shuffled session |
| Baselines | A0, K0, E fit remotely and scored with N on Dpre, G, Dpost, O2, O5 (`results/baselines.md`) |
| O6 (small local run) | A0: 0% degenerate at MCTS-1000, five-tank 0.278 (bar 0.25), optimism vs J +0.15. K0: five-tank 0.379, optimism +0.03 (section 7) |
| Wave-1 launch commands | Section 9. Not launched. |

## 1. What was built

| Brief item | Where | State |
|---|---|---|
| 1. Data layer, tranches, quarantine | `snapshot.py`, `tranches.py`, `data.py` | Done. Snapshot `snap_20261013` |
| 2. Broker, budget, ledger | `broker.py`, `ledger.py`, `session_kit/bin/hyp` | Done for Q0, Q1, Q2. Q3, Q5 refuse (no parsed replays yet); Q4, Q6 refuse (section 8) |
| 3. Enforcement | `sandbox/` (Docker) | Done, 10/10 |
| 4. Model language, fitters, as-of contract, fixtures | `terms.py`, `fit.py`, `features.py`, `fixtures.py` | Done |
| 5. Evaluator, gate, controls, arms | `evaluator.py`, `harness.py`, `broker.py` (arms a, b, c, blinded, shuffled) | Done; O3 backtests and the C/F batch runner not yet |
| 6. Remote dispatch | `remote.py`, `worker.py` (on `training/remote_workers` hotsjob) | Done |
| 7. Baselines | `baselines/` and `results/baselines.md` | Done (A0, K0, E, N, P1) |
| 8. MCTS-leaf adapter and O6 | `leaf/` | Done, Python leaf in the CPU reference search; CUDA additive kernel not built |
| 9. Kit table | `kit/kit_2.57.json` | Done, 91 heroes, 12 axes |
| 10. Session launcher | `session.py`, `session_kit/` | Done |

## 2. Data and tranches

Snapshot `snap_20261013` from the local research role (`DATABASE_URL_RESEARCH`, read-only session; Neon never touched). Counts match FRAMING 4.1 exactly:

| Tranche | Games |
|---|---|
| S | 1,946,097 |
| Dpre | 173,764 |
| G | 155,191 |
| Dpost | 173,394 |
| C (so far) | 29,334 |
| pool (keyed-hash 10% of non-sealed 2.57) | 3,302, held apart in `pool.npz` |
| R sealed rows in the DB | 4,387, **dropped at read time, never written anywhere** |
| M | 0 (no cohort-B timelines parsed yet; only sealed files have been downloaded) |

- **Labels apart.** Outcomes, game length and avg MMR are in `labels.npz`; covariates in `games.npz`. Files are mode 0400 in a 0700 directory.
- **Private accounts honoured.** 80,141 player slots belonging to `player_privacy` private accounts have no player key, so they feed no player history (Zemill's condition).
- **First-seen times.** Historical rows use game date + 1 day (retrospective). A daily diff (`snapshot.py first-seen`, cron 04:15) now records true first-seen times, seeded today with 2,485,469 rows.
- **Keys** for the pool, M split, audit, canaries, blinding and noise are in `/home/max/scilearn-private/keys.json` (0600).
- `snapshot.py verify` re-hashes the files and confirms 0 sealed IDs anywhere.

## 3. Enforcement: sandbox escape tests

bwrap cannot isolate the network here: `kernel.apparmor_restrict_unprivileged_userns = 1` makes `bwrap --unshare-net` fail. Proposer sessions therefore run in Docker (`scilearn-sandbox:1`) with `--network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --read-only`, 8 GB, 2 CPUs. Mounts: the per-arm view (read-only), the session work dir, the session's broker socket, a proxy socket, and the account-2 claude config with every other session's files masked. Model API egress goes through a host-side CONNECT proxy on a Unix socket that allows port 443 to Anthropic hosts only and refuses every private, loopback, link-local, CGNAT (Tailscale) and IP-literal destination.

`python3 training/scilearn/sandbox/test_escape.py` (rerun 2026-10-13, exit 0):

```
PASS  proxy policy (host-side, stub resolver)
PASS  read .env                        (path absent; filesystem walk finds no .env; no DB env vars)
PASS  connect to postgres              (127.0.0.1:5433/5432, docker gateway, host IPs, Tailscale IP, the pg container: all refused; no sockets)
PASS  read label files                 (planted canary label file absent; no *labels* / *.npz / canary bytes anywhere)
PASS  ssh to workers                   (no ssh binary, no keys, TCP to both workers' Tailscale IPs :22/:2222 fails)
PASS  proxy refuses private/non-allowlisted  (CONNECT to 127.0.0.1:5433, 172.17.0.1:5433, Tailscale :22, example.com: 403)
PASS  read-only /opt/hyp and root
PASS  docker.sock, host processes, caps (no docker socket, 3 container pids, CapEff=0, NoNewPrivs=1, only lo)
PASS  other sessions' transcripts       (projects/, history.jsonl and the rest empty inside)
PASS  broker socket reachable (positive control)
10/10 passed
```

One real `claude -p` smoke call (haiku, "reply ok") went through the proxy and returned `ok`.

## 4. Planted-leak tests and the as-of contract

`PYTHONPATH=. python -m training.scilearn.fixtures` (exit 0):

```
PASS  legit asof_hero_winrate / delete_game, late_upload, own_label            (3 lines)
PASS  legit player_hero_experience_90d / delete_game, late_upload, own_label   (3 lines)
PASS  planted P1_self_inclusive_wr: caught by ['delete_game', 'late_upload', 'own_label']
PASS  planted P2_game_time_wr: caught by ['delete_game', 'late_upload', 'own_label']
PASS  planted P3_future_wr: caught by ['delete_game', 'late_upload', 'own_label']
PASS  planted P4_future_scoreboard_length: rejected: field 'game_length' is not visible at draft time
PASS  planted P5_parse_time_mmr: rejected: field 'avg_mmr' is not visible at draft time
PASS  planted P5b_mmr_in_predicate: rejected: term.predicates[1]: field 'avg_mmr' ...
PASS  planted P6_talents_scoreboard: rejected: term.keys[1]: field 'talents' ...
PASS  planted P6b_scoreboard_field: rejected: term.predicates[0]: field 'scoreboard' ...
PASS  legit library swap antisymmetry: 12 term types, max error 0.0e+00
PASS  planted P7_side_biased_term: swap error 2.00 > 1e-8, rejected
PASS  planted P8_fit_score_overlap: refused: refusing to score 1 rows that a fit read
17/17 passed
```

- The invariance checks run on 6,000 real Dpre games with player histories.
- Every card is compiled and swap-checked when submitted; a card that fails is rejected and logged (seen live: the `game_length` card in the smoke session).
- Every fit job writes its read set (`read_ids.npy`) and the worker refuses to score any row in it.
- Ledger tamper tests (`python -m training.scilearn.tests.test_ledger`): 5/5.

## 5. Dry runs, end to end

Two sessions, opened exactly as wave-1 sessions are: `dry-b` (arm b, real labels, dev = Dpost) and `dry-shuf` (arm shuffled: whole records permuted within map x tier x build blocks on every tranche the session touches). Incumbent: K0, fit remotely on S+Dpre+G under each session's labels. All commands ran inside the sandbox through `hyp` (scripts in `dryrun.py`); the logs are in each session's `logs/`.

**dry-b ledger** (13 records, chain verifies, head `099623f18c`):

| seq | kind | card | detail |
|---|---|---|---|
| 2 | query | | Q1 on S, no-healer teams by tier: 43.2 / 41.7 / 38.1 / 40.9%, 4 sandbox numbers |
| 3 | query | | Q1 on Dpost, no-healer pooled: noisy, 0.161 bits to `structure:roles` |
| 4 | refusal | | Q1 on C: "tranche 'C' has no query class" |
| 5-7 | card, test, reject | nohealer1 | z = -0.14 vs 3.24 (the group already held 0.161 bits): released gain 0.0000 |
| 8-10 | card, test, reject | synergy1 | z = 8.95, wrongly rejected by a gate bug, fixed (section 8) |
| 11-13 | card, test, accept | synergy2 | z = 8.95 vs 3.72, invariance 6/6 builds positive (Cochran p = 0.41), five-tank probe 0.419 -> 0.429, released gain +0.0011, flag `unexplained` |

- **The no-healer card adds nothing over K0, legitimately.** On S, no-healer teams win about 42%, but K0 already holds a free-shape function of the team's healing_throughput sum, which absorbs the effect (refit coefficient -0.08, dev gain -0.000002). The kit backbone does its job.
- **The synergy card is a real, large effect.** Ally pairs with at least 2,000 games (3,375 pairs) gain 0.00147 on held-out Dpost (SE 0.00016) over K0.
- **Each Q2 test takes 2 to 6 minutes on the 3090** (term-only refit against the cached incumbent's logit, 12 cores).

**dry-shuf** (13 records, chain verifies): the same queries and cards, each rejected: nohealer1 z = 0.02, synergy1 z = 0.14, synergy2 z = 0.14 (the same fit; only the threshold differs). The shuffled incumbent scores 0.69335 and 50.02% on its shuffled Dpost (calibration slope 0.01), so the shuffle removes every signal, as intended. **The planted null was not accepted.**

**Canaries.** Six per session, tested on the session's own contract against its current incumbent, logged only in the harness ledger, never charged or shown to the session:

| Session | Null canaries (hash x2, permuted kit axis x2) | Known-true canaries |
|---|---|---|
| dry-b | 0 of 4 accepted (z = 0.62, -0.14, -0.26, 2.74) | role multiset accepted (z = 3.59); ally pairs rejected (z = -2.9) because the session had already accepted an ally-pair family |
| dry-shuf | 0 of 4 accepted | 0 of 2 accepted (labels shuffled, so both are null here) |

So 0 false accepts in 10 null trials. Lesson: known-true canaries must be structure the session's current incumbent lacks; the harness should draw them against the starting backbone (todo below).

## 6. Baselines

Full tables: `results/baselines.md` (and `results/baselines_neural.md` for N's provenance). Headline, Dpost (173,394 games):

| Model | Fit on | Log loss | Acc. | vs N (paired, SE) | Slope |
|---|---|---|---|---|---|
| A0, state-space additive logit | S+Dpre+G | 0.67253 | 58.14% | -0.0064 (0.0004) | 0.88 |
| A0 | S only | 0.67435 | 57.93% | -0.0046 (0.0004) | 0.85 |
| **N, drift-aware MLP (frozen)** | to 2.55.14 | **0.67896** | **56.82%** | ref | 0.88 |
| E, EBM (InterpretML, 30 pairs) | S+Dpre+G, last 1M | 0.68058 | 56.39% | +0.0016 (0.0004) | 0.94 |
| K0, kit-anchored backbone | S+Dpre+G | 0.68086 | 56.37% | +0.0019 (0.0005) | 0.86 |
| P1 site enriched MLP (frozen) | paper-1 snapshot | 0.68166 | 56.21% | +0.0027 (0.0003) | 0.85 |

How to read it:
- **A0's lead over N is mostly personal information.** A0 and K0 carry an as-of player-hero experience term worth about 0.005 log loss; N uses no player data. Draft-only A0 is near 0.6777, about 0.001 better than N. The fair personal comparison is N with P3 (59.70%), not scored here.
- **K0 trails A0 by 0.008.** K0 has no hero-by-map, counter-pair or ally-pair terms, which are what the wave-1 sessions are meant to discover. Those terms are worth about 0.0024, 0.0035 and 0.0017 in A0, and the dry run's ally-pair card recovered 0.0015 of it.
- **O5 rare compositions** (12,544 Dpost games): A0 0.6616, N 0.6668, K0 0.6715 (slope 0.99, the best calibrated), E 0.6762.
- **O2 onset:** A0 beats N in all 8 first-1K and first-10K slices of the 2.55.17 builds.
- **D1 criterion** "A0 and K0 within 0.5pp of E": A0 is 1.75pp above E and K0 is level with E (-0.02pp).

## 7. MCTS leaf and O6

- **Adapter.** `leaf/export.py` turns a fitted model into a leafpack (lookup tables plus kit shapes, W, rules and variances). The leaf matches the fitted model to 8e-16 logit on random complete drafts across every map and tier, for K0, A0 and J. The player-experience term is dropped (it has no meaning in draft search).
- **Search.** `leaf/search.py` plugs the leaf into `cuda_mcts/ref_search.py`'s `RefSearch`, unchanged, with paper-1 GD opponents. `leaf/o6.py` runs the O6 probes. Tests: 9/9.
- **Judge.** J is a K0-form model fit on Dpre+G only (scores 0.6861 on Dpost), time-disjoint from W = K0 or A0 fit on S.

Small-count local runs: 12 configurations per drafter, paper-1 GD opponents, no mask, Cursed Hollow / mid for the probes. Full tables are in `results/o6_wave0.md`.

| | A0 (fit S) | K0 (fit S) | G3 bar |
|---|---|---|---|
| Degenerate rate, greedy | 0% | 25% | (paper-1 neural WP: 47.5%) |
| Degenerate rate, MCTS-400 / MCTS-1000 | 8% / 0% | 8% / 0 to 8% | <= 5% at MCTS-1000 |
| Five-tank probe P(win vs standard) | 0.278 | 0.379 | <= 0.25 |
| No-healer probe | 0.478 | 0.480 | |
| Optimism p_W - p_J, MCTS-1000 | +0.15 | +0.03 to +0.04 | <= half of N's |
| Split-half pick agreement, MCTS-1000 | 25% | 33% | >= 40% |

How to read it:
- **A0 drafts healthy teams** (0% degenerate at MCTS-1000) but is optimistic against J by 0.15: search finds its pair and counter terms. That is the exploitability that FRAMING 3 row 12 predicted for an additive leaf.
- **K0 is much less optimistic** (+0.03) but greedy-drafts degenerate teams 25% of the time: with no role-structure terms, it relies on kit sums alone.
- **Neither passes the five-tank bar.** Note that J itself rates five tanks at 0.43, because it has no role structure either. A judge with role terms (A0-form, on a disjoint block) is the better O6 judge for wave 1.
- **The split-half failure is mostly search noise.** Top candidates sit within 0.01 to 0.02 of each other, against rollout noise of about 0.05. Within identical draft histories, agreement is 50 to 56%. N must be measured under the same protocol before this criterion means anything.
- **N's O(W) is not measured yet** (it needs the neural leaf in the same search), so "optimism at most half of N's" cannot be checked.

## 8. Deviations from the spec, and known gaps

**Design deviations (all deliberate):**
1. **No `hypagent` Unix user** (no sudo). Isolation is the Docker container instead, which is stronger on the network side. Residual risks: anyone in the docker group is effectively root on this box, and the account-2 OAuth credentials are readable inside the sandbox, since claude needs them.
2. **The gate decides family terms by the held-out paired z, not a k-df LRT.** Terms are fit on the fit tranches and scored on held-out dev games, so the paired z already pays for every parameter. A chi-square threshold with k = 3,375 df rejected a z = 8.95 family in the first dry run (`synergy1`, ledger seq 9-10). The LRT decision is still recorded in `alt_decisions`, next to the FWER (z >= 3.3) and full-margin rules and the one-sided p for FDR at cycle close.
3. **Fit data by arm.** Dpre-gated arms (a, c) fit on S. Dpost-gated arms (b, blinded, shuffled) fit on S+Dpre+G.
4. **Q2 screens the card's term against the frozen incumbent's logit (offset fit), not a joint refit.** Accepted terms join the incumbent and trigger a full refit before the next test.
5. **Snapshots are npz, not parquet** (no pyarrow on the workers).
6. **Known-true canaries are a role-multiset family and an ally-pair family, not no-healer variants**, because K0's healing shape absorbs the no-healer effect.
7. **The EBM fits the most recent 1M games of its fit tranches**, with a config frozen in code before any dev query.

**Gaps before or during wave 1:**
- **Q3 and Q5 refuse:** M is empty until cohort-B timelines are parsed. **Q4 and Q6 refuse:** they are not implemented yet. The mediator SNR check (FRAMING 9.2) and the replay feature store are not built.
- **NN-free personal block:** only player-hero experience. P3's Kalman skill, rust and the rest are not rebuilt.
- **Not run yet:** O3 (leave one hero out), the S rolling-origin backtests, the C/F batch runner and post-stratified C scoring. The evaluator has the pieces (`post_stratify_weights`, disjointness, metrics); C stays untouched.
- **No CUDA additive kernel.** O6 at full scale should run on the workers; today's O6 used small counts locally.
- **Canary selection** should be against the starting backbone, and canaries should be run once per session per cycle (`session canaries <sid>`).

**For Max:**
- **Spot-check 15 kit heroes** (`training/scilearn/kit/SPOTCHECK.md`).
- **Licensing:** `kit/sources_2.57.json` holds patch-note text mirrored from hotspatchnotes.com. I kept it out of git (`.gitignore`) until you decide whether it may go in the public repo.
- **Account-2 directory:** the sandbox work added empty mount-point entries to `/home/max/.claude-account-2` (`CLAUDE.md`, `settings.local.json` = `{}`, and dirs `agents, commands, output-styles, hooks, rules, todos, statsig, ide, teams, plans, tasks`).
- **Crontab:** I added two lines, the ledger verify-and-copy at 03:30 and the first-seen diff at 04:15.

## 9. Launching wave 1 (not launched)

Sessions K-b, K-blinded and K-shuffled, each with its own budget, ledger, broker socket and work dir:

```bash
cd /home/max/heroes-of-the-storm
export PYTHONPATH=. PY=/home/max/venvs/scilearn/bin/python

# 1. open the sessions (state, ledger, per-arm read-only view, prompt)
$PY -m training.scilearn.session prepare w1-K-b        --arm b
$PY -m training.scilearn.session prepare w1-K-blinded  --arm blinded
$PY -m training.scilearn.session prepare w1-K-shuffled --arm shuffled

# 2. fit incumbents on the 3090. K-b and K-blinded share the real-label K0 fit (cached,
#    instant). K-shuffled has its own shuffle key: one new fit, about 35 minutes.
$PY -m training.scilearn.session prefit w1-K-b
$PY -m training.scilearn.session prefit w1-K-blinded
$PY -m training.scilearn.session prefit w1-K-shuffled

# 3. brokers (one per session; the socket is the session's identity)
for s in w1-K-b w1-K-blinded w1-K-shuffled; do $PY -m training.scilearn.session broker $s; done

# 4. proposer sessions: claude2 account, claude -p --model claude-opus-5-5, inside the sandbox
for s in w1-K-b w1-K-blinded w1-K-shuffled; do
  nohup $PY -m training.scilearn.session launch $s > /home/max/scilearn-sessions/$s/logs/launch.out 2>&1 &
done

# 5. once per session per cycle: the 6 canaries (harness ledger only)
for s in w1-K-b w1-K-blinded w1-K-shuffled; do $PY -m training.scilearn.session canaries $s; done

# monitoring and teardown
$PY -m training.scilearn.ledger verify                   # every chain
tail -f /home/max/scilearn-sessions/w1-K-b/logs/broker.log
$PY -m training.scilearn.session stop w1-K-b             # stops the broker
```

`session launch <sid> --dry-run` prints the exact `docker run` line. The launcher sets `CLAUDE_CONFIG_DIR=/home/max/.claude-account-2`, runs `claude -p --model claude-opus-5-5` with the arm's prompt, logs stream-json to `<session>/logs/`, and keeps transcripts in `<session>/claude-projects/`. At most 2 remote test jobs run at once across all sessions, which keeps scilearn under its 60 GB share on the 3090.

**Re-identification probe** for the blinded arm at cycle end: `session_kit/prompts/reid_probe.md`.

## 10. Compute used

- **3090** (cores 24-35, nice 19, mem cap 56 GB per job; our total peaked near 49 GB):
  - A0 fits: 84 and 107 minutes;
  - K0 and incumbent fits: 23 to 34 minutes each;
  - Q2 tests: 2 to 6 minutes each;
  - neural N scoring: 2 minutes per tranche.
- **5090** (CPU only, under 11 GB): EBM fits 6.5 minutes each; judge fit 1 minute. One K0 fit and one EBM attempt hit the 11 GB cap there; the history pass now streams in chunks.
- **3080:** not used. No GPU used anywhere.
- **This box:** snapshot (one 5-minute DB read), broker aggregates, fixtures and the O6 probes on at most 4 cores at nice 19.
