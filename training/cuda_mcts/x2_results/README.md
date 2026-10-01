# X2 fix: the MCTS tree now searches through opponent turns

Audit item X2 (and P3-08), `audits/CONSOLIDATED_AUDIT_2026-10-01.md`. Fixed in 1db7df8. The baseline kernel is a415a83.

## What was wrong

The CUDA kernel expanded a node only when our team was to move. A child whose next step belonged to the opponent stayed a leaf for ever, and every visit ended in a GD rollout. The tree covered only the current own-pick block (1 or 2 picks). Every MCTS agent in paper 1, drift, production and P3 was trained and evaluated with this search.

## What the fix does

`search_v2.cuh`, `search_mode=1` (default everywhere):

- Opponent turns are chance nodes. The GD distribution at the node's exact state is computed once and cached in the node, as the old kernel cached it. Selection samples the reply and descends into the child keyed by that hero, creating it on demand.
- Progressive widening at chance nodes: at most `max(1, ceil(N^0.5))` children after N visits. Below the limit the reply is drawn from full GD; at the limit it is drawn from GD renormalized over existing children. The limit grows without bound, so every reply is eventually admitted and the node's value converges to the GD expectation. At a finite budget the likely replies are revisited often enough to plan the next own decision under them. Top-k was rejected because its truncation bias never vanishes.
- Backup adds the leaf value, P(our team wins), to every node on the path. Chance nodes therefore hold the sampled-outcome average; decision nodes take PUCT maxima. Values are all in our perspective, so there are no sign flips.
- Leaves are evaluated as before: a GD rollout to the full draft, then the symmetrized WP. Priors come from the policy head.
- Children are allocated lazily in per-episode arenas of `3*sims+8` nodes and slots. One simulation creates at most 3 of each, so the arena cannot overflow. Every allocation is still guarded and counted.
- `search_mode=2` is the web kernel's open-loop roll-forward. It is kept for comparison only.
- `search_mode=0` is the old kernel, kept verbatim. It reproduces the pre-fix build bit-for-bit.

## Verification

All 11 checks pass (`test_search_v2.py`, `test_report.json`):

- **Trace replay (48 trees).** Each node's state is rebuilt from the root by its action path, and its kind and legality are checked. GD probabilities at every chance node and policy priors at every decision node are recomputed in PyTorch on the rebuilt state. Max differences are 7e-7 (GD) and 3e-6 (policy). The visit and value identities hold at every node. Every tree reaches the first later own step, and all 41 trees with a later own ban ahead reach it.
- **Chance-node statistics.** Child visit counts are Multinomial(N, GD) without widening: 49 nodes, mean chi-square p = 0.54, none below 0.01.
- **CPU reference search.** It picks the exact expectimax action on toy drafts, and its Q converges to the exact value. Without widening, the chance-node average is unbiased (max |z| = 1.8).
- **Determinism.** The same seed gives identical outputs and trees.
- **Capacity guard.** A 64-node arena trips the guard and the drafts stay valid. The default arena at 1600 sims has 0 cap hits.
- **Legacy bit-exactness.** `search_mode=0` reproduces the recorded outputs of the pre-fix .so (`legacy_reference.npz`, sha256 ab30c2f3...). The ofit and all four P3 copies are bit-identical to their previous builds in legacy mode, including personal valuation, prior bias, personalized GD and self-play.
- **Harness check.** Legacy mode reproduces the stored paper-1 revision benchmark drafts exactly (1000/1000 for F_oof_s0 at 200 sims, T=0).

## Benchmark (search only, `x2_bench.py`, `bench_summary.json`)

**Setup.** 1,000 configurations (seed 20260929), the 5 GD models cycled per batch, mid tier, root argmax. The leaf is the paper-1 revision's leak-free enriched WP with its own deploy statistics. The judges are judges_v2 with structure correction, plus gold.StructRealizedIndex (RN_struct). Contrasts are paired by configuration; F_oof pools seeds s0 to s2.

Chance minus legacy, in pp (± SE):

| prior | sims | proxy | consensus_v2 | gN_v2 | RN_struct | degenerate |
|---|---|---|---|---|---|---|
| F_oof | 100 | +0.08 ± 0.17 | +0.01 ± 0.13 | -0.06 ± 0.14 | +0.02 ± 0.12 | -0.7 ± 0.5 |
| F_oof | 200 | +0.01 ± 0.18 | +0.05 ± 0.13 | -0.09 ± 0.14 | -0.07 ± 0.12 | +0.8 ± 0.4 |
| F_oof | 400 | +0.07 ± 0.17 | +0.04 ± 0.13 | +0.03 ± 0.13 | +0.03 ± 0.12 | +1.0 ± 0.4 |
| F_oof | 800 | +0.10 ± 0.18 | +0.17 ± 0.12 | +0.19 ± 0.13 | +0.10 ± 0.12 | -0.9 ± 0.4 |
| F_oof | 1600 | +0.33 ± 0.17 | +0.31 ± 0.13 | +0.44 ± 0.14 | +0.36 ± 0.12 | -0.2 ± 0.5 |
| bc | 400 | -0.06 ± 0.31 | +0.07 ± 0.26 | +0.03 ± 0.28 | +0.25 ± 0.24 | -0.4 ± 0.2 |
| bc | 800 | -0.01 ± 0.29 | +0.41 ± 0.25 | +0.37 ± 0.27 | +0.34 ± 0.22 | -0.3 ± 0.2 |
| bc | 1600 | -0.29 ± 0.27 | +0.36 ± 0.25 | +0.48 ± 0.26 | +0.37 ± 0.22 | -0.3 ± 0.4 |

Sim curve, value at s minus value at 400 sims, in pp:

| prior, search | 100 | 200 | 800 | 1600 |
|---|---|---|---|---|
| F_oof legacy, proxy | +0.01 | -0.04 | -0.04 | +0.10 |
| F_oof legacy, consensus_v2 | +0.14 | -0.09 | -0.08 | -0.05 |
| F_oof chance, proxy | +0.02 | -0.10 | -0.01 | +0.36 ± 0.16 |
| F_oof chance, consensus_v2 | +0.11 | -0.07 | +0.06 | +0.22 ± 0.12 |
| bc legacy, proxy | -7.54 | -3.17 | +2.43 | +4.98 |
| bc legacy, consensus_v2 | -4.38 | -1.85 | +1.27 | +2.63 |
| bc chance, proxy | -7.50 | -3.07 | +2.48 | +4.75 |
| bc chance, consensus_v2 | -4.52 | -1.82 | +1.61 | +2.91 |

Absolute values, F_oof at 400 sims:

| search | proxy | consensus_v2 | gN_v2 | RN_struct | degenerate | distinct teams per 1000 |
|---|---|---|---|---|---|---|
| legacy | .7219 | .6604 | .6681 | .6341 | 2.7% | 200 |
| chance | .7227 | .6608 | .6684 | .6344 | 3.8% | 209 |

Other cells:

- **Open-loop roll-forward** behaves like chance at 200 and 400 sims. At 800 sims it scores +0.26 ± 0.12 pp consensus over legacy.
- **No widening** (pw_k=0) at 400 sims is no different from widening on the judges, but it reaches about 30% less deep.
- **Sampled root** (T=1) at 400 sims: chance and legacy are equal.

**Reading.**

- With the trained prior, real lookahead changes nothing up to 400 sims. It adds about 0.2 to 0.4 pp on every independent judge at 1600 sims, for about 4x the compute of 400. So 400 sims stays a sound inference operating point.
- The extra lookahead does not add proxy-only gain. Its gain at 1600 sims shows on the judges as much as on the proxy.
- With the outcome-free bc prior, at 800 and 1600 sims the fixed search buys +0.36 to +0.41 pp consensus at equal proxy. The proxy-only share of the 400 to 1600 gain falls from 47% (legacy) to 39% (chance).
- The over-optimization finding about TRAINING sims needs the retrained agents (rebuild list below).

## Short self-play comparison (`x2_selfplay.py`, `selfplay_matched.json`)

**Setup.** The paper-1 revision config F pipeline: 400 sims, leak-free WP, own statistics, value pretraining. 50K episodes, seed 0, one run per search, so treat it as a hint. Each checkpoint is benchmarked as above under both searches.

**Throughput.** Legacy 96 episodes/s, chance 23 episodes/s, on the same GPU.

Matched comparison (trained and evaluated with the same search), chance minus legacy, paired by configuration:

| sims | proxy | consensus_v2 | gN_v2 | RN_struct | degenerate |
|---|---|---|---|---|---|
| 200 | +0.82 ± 0.31 | +0.02 ± 0.23 | +0.25 ± 0.25 | -0.08 ± 0.21 | -3.7 ± 1.1 |
| 400 | +1.32 ± 0.31 | +0.41 ± 0.22 | +0.61 ± 0.24 | +0.35 ± 0.21 | -2.9 ± 1.1 |

About two thirds of the training gain at 400 sims shows only on the proxy. That is the same pattern as the over-optimization finding for training sims. The judged gain is small but positive, and degenerate teams fall from 7.8% to 4.9%. The legacy-trained checkpoint also scores higher when searched with the fixed kernel: +0.51 ± 0.23 pp consensus at 200 sims.

**Drill.** The chance run was paused at episode 2688 through the MCTS_PAUSE_FILE protocol: it exited 75 and wrote a resume state with 3 pending batches. Resumed with MCTS_FRESH=0, it completed 50,000 episodes. This is the end-to-end check for production's worker path.

## Tree size and depth (chance, F_oof, per search; max over 1000 drafts)

| sims | max nodes used | arena cap (3*sims+8) | old-layout nodes needed | own decisions reached below root per sim | in-tree depth (steps) | 1000 drafts |
|---|---|---|---|---|---|---|
| 200 | 388 | 608 | 17,462 | 1.9 | 4.4 | 22-36 s |
| 400 | 774 | 1,208 | 34,673 | 2.0 | 4.6 | 43-67 s |
| 800 | 1,538 | 2,408 | 69,027 | 2.1 | 4.8 | 83-101 s |
| 1600 | 3,051 | 4,808 | 137,477 | 2.2 | 5.0 | 166-184 s |

- **Cap hits.** None in any v2 cell.
- **Old 4096-node layout.** It would have overflowed from about 50 sims with the fix in place.
- **The old kernel itself overflows** with the bc prior at 1600 sims: it crashes with an illegal memory access, and the guarded ofit copy reports 213 cap hits. That cell is reported as `legacyg`.
- **Cost.** At equal sims, chance runs at about 3x the old kernel's time, because most simulations now expand a new own decision with a policy forward.
- **Training throughput** at 400 sims: 14.4 episodes/s, measured on a shared GPU while another lane's job used it.

## Building the fixed kernels

From each directory: `PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 nice -n 19 taskset -c 48-63 python3 setup.py build_ext --inplace`.

- **Python versions.** Use the brew python3 (3.14) for `cuda_mcts` and `overfit2026/cuda_ofit`. Use /usr/bin/python3 (3.12) for the P3 copies.
- **Remote workers.** Use `training/remote_workers/build_ext.sh` (arch 8.6). It compiles cleanly.
- **New headers.** `tree_v2.h`, `search_v2.cuh` and `search_v2_util.cuh` are picked up by `#include`, and `code.filter` already syncs them.
- **Install without disturbing running jobs.** Build elsewhere, then `cp` to `<name>.so.tmp` and `mv -f` it over the old file, so running processes keep the old inode.
- **Check after building:** `python3 cuda_mcts/test_search_v2.py` (from training/).

## Using it

- **Worker.** `MCTS_SEARCH_MODE=chance` is the default. `legacy` reproduces old runs. `MCTS_PW_K` and `MCTS_PW_ALPHA` set the widening.
- **Provenance.** The worker writes `kernel_info.json` into the run directory. The search mode is saved in `resume_state.pt`, and a mismatched resume is refused.
- **Engine API.** `run_episodes(..., search_mode=1, pw_k=1.0, pw_alpha=0.5)` and `run_episodes_into_buffer(..., search_mode=...)`. `last_stats()` returns per-episode tree diagnostics, `debug_tree()` returns one search's tree, and `set_tree_capacity()` is for tests.
- **ofit.** `overfit2026/search.run(..., search_mode=None)` takes the mode from `OFIT_SEARCH_MODE`, default 1.
- **P3.** `run(..., search_mode=1)` and `last_stats()`.
- **Production.** `refresh.py` pins `REFRESH_MCTS_SEARCH=chance`.
