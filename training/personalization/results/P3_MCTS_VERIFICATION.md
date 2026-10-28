# MCTS drafting: independent verification of the CPU and GPU search (paper 2)

Status: 2026-10-19. Independent check of the search code used by paper 1 and by the P3 tournament. I did not write the kernels. All tests are new code in `training/personalization/xverify/` and `training/cuda_mcts/test_engine_isolation.py`.

## Verdict

- The search does what we say it does. A CPU replica written from the kernel sources reproduces the GPU searches simulation by simulation, with every root visit count identical, in 756 of 768 tested searches (98.4%) through 400 simulations. Every exception starts at a floating-point near tie, a gap below 1e-5.
- The check found two real bugs and a few small ones. Both real bugs are fixed and covered by regression tests:
  1. **P3 kernels, skip-path race (high).** In decide-only mode, about 3% of searches gave a different result when rerun with the same seed, and 2% executed a different hero. Fixed in 2ad31c1. Every affected tournament arm is being rerun.
  2. **Paper-1 kernel, upload race after 48ab696 (high in code, no exposure).** Any engine after the first one in a process searched with stale weights on the Windows (WDDM) workers. Fixed in b995800. No recorded result used a build with this code, so no paper number changes.
- Results against the fixed builds (2ad31c1 P3 kernels, b995800 paper-1 kernel; sm_86, RTX 3090):
  - every rerun is byte-identical;
  - no illegal action appeared in any search or draft;
  - leaf values match the Python value function to 2e-6;
  - the value is antisymmetric in the teams to float precision.

## 1. Implementation (draft for the paper)

**Draft and state.** A Storm League draft has 16 steps in a fixed order:
- bans by team 0, 1, 0, 1;
- picks 0, 1, 1, 0, 0;
- bans 1, 0;
- picks 1, 1, 0, 0, 1.

Kernel team 0 is always the team that picks first. The state records each team's picks, the bans, the map and the skill tier. A hero can be used once per draft. In the player-aware drafters, a team may only pick heroes that at least one of its five players has played before the lobby day. If none of those is still free, any free hero is allowed. Bans are unrestricted.

**Search.** At each of its turns the drafter runs Monte Carlo tree search from the current state.
- The tree has two kinds of nodes:
  - At *our* nodes the drafter chooses. Each child is scored with PUCT:

    Q(a) + c · P(a) · √N / (1 + n(a))

    - Q(a) is the mean value of the child (0 if unvisited).
    - P(a) is the prior from the policy network over legal heroes.
    - N is the parent's visit count and n(a) the child's.
    - c = 2.
    - Ties go to the lowest hero index.
    - Children are created on first visit.
  - At *opponent* nodes the opponent's reply is drawn at random from an opponent model. This is the behavioural "generic draft" network, optionally combined with the opponents' own play history (below). The distribution is computed once per node and cached.
- Progressive widening keeps rare replies from absorbing the budget. An opponent node with N visits may hold at most ⌈√N⌉ distinct replies. At that limit, a new draw is taken from the replies already present, renormalised.
- A simulation descends until it reaches one of our nodes that has never been expanded, or the end of the draft. The node is expanded with the policy prior.
- The draft is then completed by a rollout: both teams pick from the opponent model.
- The finished draft is valued once. Its value is added to every node on the path, always from the searching team's side. Our nodes maximise and opponent nodes average, so no sign flips are needed.
- The tree is closed-loop: every node corresponds to one exact draft state. Search therefore looks through the opponent's replies to our later picks and bans.
- After 400 simulations, the drafter plays the most-visited hero allowed by paper 1's role-structure rule. Ties are broken by Q.

**Value of a finished draft.**
- *Population value:* the win-probability network evaluated in both team orders and averaged, so swapping the teams exactly flips it.
- *Player-aware value:* adds personal terms on the logit scale:

  V = σ(b₀ + b₁ · logit WP + b₂ (S_ours − S_theirs) + b₃ (O_ours − O_theirs))

  - S is a team's summed per-player skill on its heroes.
  - O is its count of off-role picks.
  - Heroes are assigned to the team's five players by the assignment that maximises b₂S + b₃O, the first maximum over the 120 permutations.
  - The real pick order (who picked what) is never used.
- *Learned value variant:* replaces the linear terms with a small network on per-team features. It is antisymmetric by construction.

**Opponent and prior variants (exploration study).**
- *Player-aware opponent:* reweights the generic-draft probabilities to softmax(w₀ log p + bias), where the bias comes from the opponents' play counts.
- *Prior bias:* reshapes our prior to softmax(α log p + bias) at our picks and, optionally, our bans. At bans the bias favours heroes the opponents play well.

**Parallelism and randomness.**
- One GPU thread block runs one complete search; 256 threads share each network evaluation.
- Simulations run one at a time inside a block. There is no virtual loss or batched selection, so a search is fully determined by its inputs and seed.
- Each block has its own random stream, used only by thread 0. After 2ad31c1 the P3 kernels seed it with a splitmix64 hash of (batch seed, episode index). The paper-1 kernel uses batch seed + episode index.

## 2. How we verified it (draft for the paper)

We wrote a second implementation of the search on the CPU, from the kernel source, and fed both implementations the same inputs. The CPU version reproduces the GPU's random stream exactly: we ported curand's XORWOW generator and checked it against real curand on 64 of 64 subsequence draws. Its networks are the PyTorch models with the Python featurisation.

Because the random draws are identical, the two searches make the same choices until floating-point rounding differs. That happens only when two PUCT scores, or a random draw and a probability boundary, are closer than about 1e-6. We ran the GPU search at 78 simulation budgets from 1 to 400. Since a k-simulation search is exactly the first k simulations of a longer one, this shows the first simulation at which the two disagree.

On 512 randomized and adversarial positions across the four P3 kernels:
- 504 searches matched visit for visit through 400 simulations;
- each of the other 8 first diverged within a simulation that contained a near tie (gap below 1e-5);
- root values agreed to 2e-6 while the trees agreed.

We also:
- compared leaf values to the Python value function;
- checked exact symmetries (swapping sides, swapping teams and players);
- checked that the player-aware leaf with its personal terms off reproduces the population search byte for byte;
- tested the opponent model's draws with chi-square tests;
- confirmed convergence to the exact optimum on toy drafts;
- reran every kernel twice on 1,024 searches and required byte-identical output.

The rerun test is what found the two bugs.

## 3. Inventory of search implementations

| implementation | where | used by | semantics notes |
|---|---|---|---|
| Paper-1 GPU kernel, v2 chance search | `cuda_mcts/search_v2.cuh` via `kernel_bindings.cpp` (`MCTSKernelEngine`) | paper-1 training (`train_mcts_worker`), drift lane agents, rerun2026 and rebench benchmarks, production h91 build | as section 1. Full episodes: the opponent's real moves are sampled from GD. Root noise optional (Dirichlet, training only). Executed action: argmax (T < 0.05) or visits^(1/T). Modes: 0 legacy (pre-X2, bit-exact), 1 chance (default), 2 open-loop roll-forward |
| Paper-1 ofit kernel | `overfit2026/cuda_ofit` | `paper1_revision/bench_mcts.py` (paper-1 bench, ensemble, reconciliation) | earlier guarded copy. Not modified here. Not covered by the stream change |
| P3 population reference | `personalization/cuda_personal/ref` (`pop_ref_kernel`) | identity check for the P3 kernels | copy of the paper-1 kernel. Hashed seed under `P3_HASHED_SEED` |
| P3 personal | `cuda_personal` | aMCTS/pMCTS -cur, -bc, drift arms | adds pools (slot or team-union mode), forced bans, decide-only prefix replay, self-play, personal leaf (slot or best-assignment), imitation opponent in the main loop |
| P3 prior | `cuda_prior` | aMCTS-pp, -sp | + prior bias softmax(α log p + bias[slot]) at picks |
| P3 pvnet | `cuda_pvnet` | aMCTS-vn, x-*-vn | + learned value net leaf (cfg[57]) |
| P3 explore | `cuda_explore` | x-* design arms (opp, ban_bias, rich at bans) | + prior bias at bans (cfg[56] = 2), player-aware opponent at every opponent step: chance nodes, rollouts, main loop (cfg[6] = 2) |
| P3 pgd | `cuda_pgd` | earlier personalized-GD study | same main loop (race fixed in 2ad31c1). Not separately tested here |
| CPU reference (toy) | `cuda_mcts/ref_search.py` | unit tests of the search algebra | same rules on small games, exact expectimax |
| CPU replica (this work) | `personalization/xverify/xv_ref.py` | equivalence tests | the kernel's search on real networks, kernel random stream, every P3 hook |
| One-step drafter | `p3_dr_drafter._decide` (`OneStep` in `p3t_systems`) | a1step / p1step arms | top 15 GD + top 15 personal candidates, 4 GD rollouts each, argmax of mean value under the structural mask. Not a tree search |

How the tournament harness drives the P3 kernels (`p3t_systems.act_mcts`, `act_mcts_x`):
- decide-only at every step, cfg[4] = k, with the real prefix replayed;
- assign mode;
- team-union pools;
- no forced bans;
- dir_eps = 0;
- 400 simulations (1,600 for x-combo3);
- c = 2 unless tuned;
- widening k = 1, α = 0.5;
- kernel seed = pair hash + 1000 k + 7919 k + batch offset + 104729 · prior seed;
- the structural mask and the argmax are applied on the host.

## 4. Test matrix

Unless noted, every result below is against the fixed builds: 2ad31c1 P3 kernels and the b995800 paper-1 kernel, sm_86 (RTX 3090). "Exact" means byte-identical, or for visit counts integer-identical.

### CPU and GPU equivalence

| test | size | tolerance | result |
|---|---|---|---|
| curand XORWOW replica vs device curand | 64 subsequences + 8,192 first draws | exact | pass |
| Decide-only replay, personal kernel | 128 positions × 78 budgets (1–400) | visits exact, root Q 2e-4 | 124 exact through 400. 4 diverge after a near tie (gap < 1e-5). 0 unexplained. Max root-Q gap while equal 2.1e-6 |
| Decide-only replay, prior kernel | same | same | 125 exact, 3 near tie, 0 unexplained. Root prior gap 1.6e-6 |
| Decide-only replay, pvnet kernel | same | same | 127 exact, 1 near tie (at sim 400), 0 unexplained |
| Decide-only replay, explore kernel | same | same | 128 exact |
| Replay with c = 1, c = 4, widening off, widening k = 2, α = 0.6 | 64 × 2 budgets each | same | 252 of 256 exact, 4 near tie, 0 unexplained |
| Paper-1 kernel, full 8-turn episodes (argmax and T = 1; 50 and 200 sims) | 256 episodes | visits exact every turn, terminal WP 1e-6 | 248 identical (WP gap ≤ 6.6e-7). 8 diverge after a margin < 1e-6 |
| Leaf WP vs Python featurisation + WP (512 drafts; 40% with one team stacked in one role, so rare compositions exercise the mid-high-low fallback) | 512 | 1e-4 | max 7.5e-7 |
| Personal leaf, best-assignment, pvnet leaf vs float64 mirror | 512 each | 1e-4 | max 2.0e-6 |
| Root prior incl. bias (prior, pvnet, explore) | 384 | 1e-4 | max 1.9e-6 |

The adversarial positions cover:
- every draft phase (early bans, mid picks, late bans, last pick);
- tiny pools (1–2 heroes);
- full pools;
- pools that are fully taken (fallback to any free hero);
- exactly one free pool hero;
- identical player tables;
- mirrored tables;
- extreme personal values (s ~ N(0, 3²));
- forced bans;
- slot mode;
- personal term off;
- pool rule off;
- ban-step bias rows;
- player-aware opponent rows.

### Correctness properties

| property | result |
|---|---|
| No visits on illegal heroes (taken, outside the pool rule) | 0 in 768 replays and 40,448 snapshots |
| No repeated hero in any draft | 0 in 24,576 forced-ban drafts (2ad31c1); 0 in 95,400 finished tournament drafts |
| Side flip: V(our = 0) + V(our = 1) = 1 | max deviation 1.2e-7 (GPU), 2e-16 (CPU) |
| Team and table swap: WP(t0) + WP(swapped) = 1 | 6e-8. Personal value: logit V + logit V' = 2 b₀ to 3e-3 in logit (float32 logits of extreme values) |
| Personal term off == population kernel | P3 personal kernel with population settings == pop_ref, full episodes, byte-identical (50 and 200 sims). Aware with zero tables and b = (0, 1, ·, ·) == blind, byte-identical. Blind search invariant to the tables, byte-identical |
| Best assignment == brute force; kernel rule == host mirror == judge rule | 400 trials, gap 7e-15. Permutations differ only on exact ties of the objective (14 cases, same objective) |
| Opponent model = softmax(w₀ log p_GD + bias), sums to 1, support = valid heroes with GD mass | CPU: exact. GPU main loop, 8,192 draws: chi-square p = 0.42 (plain GD), 0.11 (player-aware), 0 draws on zero-probability heroes |
| Search converges to the exact optimum (toy drafts, exact leaf values, CPU reference) | mean regret 0.181 → 0.125 → 0.044 → 0.024 → 0.0025 → 0.0022 at 10 → 10,240 sims. Optimal 33% → 91%. The rest are near ties (gap ≤ 0.004) |
| Monotone in simulations (toy) | non-increasing within 2 SE at every step (CPU). GPU last-pick toy with real priors: regret 0.239 → 0.222 → 0.193 → 0.148 at 25 → 1,600 sims |
| Q equals the exact value where the value is deterministic (GPU, step 15) | max \|Q − V\| 1.7e-5 |
| Determinism: same inputs and seed twice | P3, 4 kernels × 1,024 decide-only + 2 × 1,024 forced-ban drafts × 3 runs: 0 differ. Paper-1: 256 episodes and the isolation test: 0 differ |
| Engines in one process (paper-1) | fresh engine after delete, engine alongside another, engines in 3 threads: 0 of 64 differ |
| Concurrency (three engines in threads == serial) | byte-identical |
| test_search_v2.py with legacy_reference.npz | 11/11 on the fixed build and the deployed build (3090 and 3080), with P1_TIERS unset |
| Fixed paper-1 build == pre-48ab696 deployed build, single engine | byte-identical in chance, roll-forward and legacy modes, argmax and training settings. stream_test compare IDENTICAL |

### Tournament harness

| check | result |
|---|---|
| Every pair played from both seats | tournament 318 ordered pairs, tune 156: none missing a partner. Every schedule (core, spokes, all, tune, ablation) is symmetric with no self pairs |
| Legality of the finished drafts | 95,400 + 46,800 drafts: no repeated hero. Every non-Real pick obeys the pool rule (954,000 picks). Real replays real picks: 39 real picks fall outside the earlier-play pools |
| Harness legal mask == kernel pool rule; structural mask == brute force over role completions | 300 states, 3,471 candidates: exact |
| Judge antisymmetry (swap teams and player tables) | consensus 3.7e-8, J 3.9e-8 |
| Real drafts and drafter drafts scored by the same function | yes (`score_drafts`). Stored scores matched the recomputation to 9e-4. The gap was a stale `real.npz` written before the final judge fit, fixed in 2ad31c1 |
| Drafter vs itself | GD vs GD: P(seat 0 wins) 0.493 ± 0.007 (J). pPol-cur: 0.512 ± 0.008. Seat-averaged head-to-head is 0.5 by construction |
| No judge or test player tables reach a drafter | no system reads state `judge` or `splitj` (49 cur, 7 s4m, 9 s1y, 4 splitd). Drafter states use evidence up to the day before the lobby. Pools and play counts (`n`) are as of the day before |
| Kernel seeds within a pair | 4,800 searches per side, all distinct |

## 5. Discrepancies found

| # | what | severity | status |
|---|---|---|---|
| 1 | **P3 main-loop race.** After `if (s_skip) continue;` there was no barrier, so thread 0 rewrote `s_skip`, `s_team` and the draft state for the next step before slower warps had read them. Warps fell one barrier out of step. This happens only on skipped steps (decide-only prefix replay, forced bans). Old build: 34–52 of 1,024 decide-only searches changed between identical runs, and 18–29 of 1,024 changed the executed hero (1.8–2.8%). The corrupted runs are far off (median total variation of the visit distribution 0.54–0.73, not near ties). 4 of 24,576 forced-ban drafts repeated a hero. In every case that changed, at least one run matched the CPU replica, and so did the patched build | high: tournament arms | fixed 2ad31c1 (local copy, barrier, branch). Reruns byte-identical, CPU replay intact. Arms rerunning |
| 2 | **Paper-1 upload race.** Since 48ab696 every launch is on the engine's non-blocking stream, but the constructor uploaded weights with `cudaMemcpy` on the legacy stream. Nothing ordered the two. On the WSL workers, any engine after the first in a process ran on stale weights: 64/64 episodes differ, leaf values near 0.50, and the engine stays wrong on later runs and after a 1 s wait. Mechanism (most likely): under WDDM the uploads sat in the legacy stream's unflushed command batch, and syncs on the non-blocking stream never flushed it. Any device-wide synchronize or memset in between fixed it, and device read-back after a synchronize showed correct bytes | high in code; exposure nil (section 6) | fixed b995800: uploads on the engine stream, synchronized before the constructor returns. Byte-identical to the pre-48ab696 build |
| 3 | First uniforms of neighbouring seeds were correlated (`curand_init(base + i, 0, 0)`): the first two draws across consecutive seeds fail a 50-bin chi-square (p < 1e-8). First opponent draws in a batch were slightly off (8,192 draws, p = 4e-5, TV 0.046) | low | P3: hashed seeds in 2ad31c1 (now p = 0.42). Paper-1 kernel unchanged: its first draws still show the bias (p = 3.5e-5). Fix: `curand_init(seed, i, 0)` or a hash, at the cost of the recorded legacy reference |
| 4 | A legal hero whose prior underflowed to exactly 0 could not be revived by its bias (alpha = 0 or a strong bias) | low (no case among tested states) | fixed 2ad31c1 (floor 1e-30) |
| 5 | `rich` + `ban_bias` together: the ban bias overwrote the rich ban row | latent (no config used both) | fixed 2ad31c1 (added) |
| 6 | Stale `scores/real.npz` (written before the final judge fit): judge gap up to 9e-4 | low | fixed 2ad31c1 (score files record c₂ and are recomputed) |
| 7 | P3 main-loop samplers fall back to hero 0 on rounding overrun (`action = 0`), which could be a taken hero. Reachable only with r within ~1e-7 of the total mass, about 1e-7 per draw. Not used by decide-only searches. The v2 tree and rollouts use the safe sampler | very low | open (main loop only; full-draft and self-play modes) |
| 8 | `test_search_v2.py` GPU tests had been broken since c66e81a (placeholder run `x_s0` no longer resolves). `test_trace_tree` fails if `P1_TIERS=site` is set (the test's reference GD then differs) | test harness | x2_common fixed in b995800. Run the suite with P1_TIERS unset |

Design notes for the paper (not bugs):
- The structural role constraint is applied to the executed root action only, not inside the search.
- With the sharp F prior, the search rarely finds the personal optimum of the last pick even at 1,600 simulations: 15% optimal, regret 0.15 on random tables. That is the prior hiding the personal leaf, as the probe in P3_TOURNAMENT.md found. Q values themselves are exact.
- The player-aware opponent applies at opponent bans only where the host gives a row. The tournament gives rows only at picks.

## 6. Exposure audit of the paper-1 upload race (bug 2)

No deployed build ever contained 48ab696. The commit changed the source, but the default builds were not rebuilt after it.

| location | builds | contains the stream code |
|---|---|---|
| 3090 / 3080 / 5090 `~/hots/repo/training/cuda_mcts` | 10-03 / 10-04 / 10-09 00:42 | no |
| 3080 `cuda_mcts/h91`, archive | 10-02, 10-03 | no |
| main box `training/cuda_mcts`, `h91` (production) | 10-01, 10-02 | no |
| 5090 `~/hots/streams`, `~/hots/x2noflag` | 10-09, 10-12 | yes. Test-only copies (stream_test, test_search_v2 legacy check). No job manifest or log references them |

Every runner loads the build in `training/cuda_mcts` on its host, or `h91` for production:
- multi_mcts;
- train_mcts_worker;
- paper1_revision/train_mcts;
- verify_kernel_site;
- drift_rebuild/v2;
- drift2026 phase_w4;
- rerun2026 phase4 and the diversity diagnostic;
- rebench_*;
- validate_search_budget;
- production kernel_parity.

`MCTS_KERNEL_DIR` is never set in any manifest. Paper-1 bench, ensemble and judges use `cuda_ofit`. The P3 tournament (including the paper-1 reconciliation) uses only the P3 kernels.

**Verdict:** every result in paper 1, the drift lane, personalization and production is safe. No number changes. The Nov 1 production cron (`build_h91.py --if-stale`) would have built the faulty code; it now builds b995800.

## 7. Files and how to rerun

- `xverify/xv_ref.py`: CPU replica (curand XORWOW with subsequence and offset skipahead; kernel search; all P3 hooks; `XV_KREV=2ad31c1` for the current prior rule).
- `xverify/xv_cases.py`: randomized and adversarial cases.
- `xverify/xv_gpu.py`: GPU runner (stages grid, variants, paper1, identity2, leaf, toy, opp, stream, determinism, race, p1race and the diagnostics).
- `xverify/xv_compare.py`: CPU comparison. Use `XV_SEEDING=splitmix` for the hashed-seed builds.
- `xverify/xv_cpu_tests.py`: CPU properties and harness checks. Output in `results/p3t_xverify_cpu.json`.
- `xverify/xv_harness.py`: checks on the finished tournament outputs (legality, seats, judge symmetry, self-match).
- Regression tests:
  - `xverify/test_kernel_determinism.py` (P3 reruns byte-identical, no repeated hero; paper-1 reruns);
  - `cuda_mcts/test_engine_isolation.py` (paper-1 engines in one process).

Typical run, from `training/` on a worker, with GPU outputs copied back:

```
XV_BUILD=<kernel builds> XV_OUT=<dir> python personalization/xverify/xv_gpu.py
XV_SEEDING=splitmix XV_KREV=2ad31c1 python personalization/xverify/xv_compare.py --gpu-out <dir>
python personalization/xverify/test_kernel_determinism.py
python cuda_mcts/test_engine_isolation.py
```

Not covered:
- GPU runs on the 5090 (sm_120). The fixed paper-1 build there is built and staged at `~/hots/kernels/cuda_mcts_fix_20261019` but not run, because the tournament holds that GPU.
- The pgd kernel.
- Root Dirichlet noise in the replica. Tournament searches use none.
