# MCTS leaf adapter and O6 search probes

This directory plugs a fitted scientific-learning WP (an additive glass-box logit, FRAMING 6.1) into the draft search as its leaf value, and runs the O6 search-behavior test (FRAMING 1, O6 and G3; 11 item 11).

| File | What it does |
|---|---|
| `leaf.py` | `LeafModel.load(dir)`, `evaluate(drafts[N,10], map, tier, build=None) -> (eta[N], sd[N], {term: [N]})`, `leaf_value` = sigmoid(eta - kappa * sd), `explain`, `export_check` |
| `search.py` | `HotsDraftGame`, the game adapter for `training/cuda_mcts/ref_search.py` `RefSearch`, plus `mcts_draft`, `greedy_draft` and a numpy GD opponent |
| `o6.py` | O6 probes: degenerate rate (greedy, MCTS-400, MCTS-1000), five-tank and no-healer probes, optimism O(W) against a judge J, split-half pick agreement. Writes `o6.json`, `o6.md`, `o6_drafts.json` |
| `make_synthetic.py` | Synthetic leafpacks: `benign` and `exploit` (5-tank role bonus +2.5 logit and a rule cliff +1.5 logit) |
| `test_leaf.py` | Plain-python tests |

## Commands

All local runs use nice 19, cores 60-63 and no GPU.

```bash
L=training/scilearn/leaf
# tests (about 10 s)
nice -n 19 taskset -c 60-63 python3 $L/test_leaf.py

# synthetic packs: W benign, W exploitable, independent judge J (other noise seed)
nice -n 19 taskset -c 60-63 python3 $L/make_synthetic.py --out /tmp/leaf/benign  --kind benign  --seed 1
nice -n 19 taskset -c 60-63 python3 $L/make_synthetic.py --out /tmp/leaf/exploit --kind exploit --seed 1
nice -n 19 taskset -c 60-63 python3 $L/make_synthetic.py --out /tmp/leaf/judge   --kind benign  --seed 2

# O6 smoke run: 12 configurations per drafter, about 75 s per pack on 4 cores
nice -n 19 taskset -c 60-63 python3 $L/o6.py --pack /tmp/leaf/exploit --judge /tmp/leaf/judge --out /tmp/leaf/o6_exploit

# real run (remote workers): more configurations, more cores, N's numbers for G3
python3 $L/o6.py --pack <W leafpack dir> --judge <J leafpack dir> --out <dir> \
    --n-drafts 500 --procs 12 --sims 400,1000 --split-sims 1000 \
    --ref-degen <N MCTS-1000 degenerate %> --ref-optimism <N optimism>
```

Useful `o6.py` flags: `--kappa` (default: the pack's json `kappa`), `--split-pack` (a half-B fit for split-half agreement), `--prior gd|uniform`, `--prior-eps`, `--opponent gd|uniform`, `--maps`, `--tiers`, `--seed`, `--no-greedy`, `--gd-bans` (our bans by GD argmax instead of search).

## Leafpack contract (what `leaf.py` reads)

`leafpack.npz` arrays: `hero_ctx[H,M,T]`, `pair_within[H,H]` (symmetric), `role_table[6**6]`, `kit[H,K]`, `shape_knots[K,n]`, `shape_vals[K,n]`, `map_axis[M,K]`, `cross[H,H]`, `W[K,K]`, and the variances `hero_ctx_var`, `pair_within_var`, `role_table_var` (same shapes; missing means zero). `leafpack.json`: `heroes`, `maps`, `tiers`, `roles`, `K`, `axes`, `rules`, `kappa`, `version`, `term_ids`, `card_hashes`, `model_id`.

Additions this adapter needs from the exporter:

1. **Hero to role map (required).** The role code needs each hero's role, which the format did not carry. Either npz `hero_role` (int `[H]`, indices into json `roles`) or json `hero_roles` (`[H]` role names). The loader refuses a pack without one.
2. **Optional, read if present:** npz `cross_var[H,H]` (added to sd), json `build` (`evaluate(..., build=b)` raises on a mismatch), and rule predicate field `"tier"` (same syntax as `"map"`). Map and tier predicate values may be names or indices. Rule predicate ops: `>=`, `<=`, `==`, `!=`.

Conventions:

- **Contributions.** Keys are `hero_ctx`, `pair_within`, `role_table`, `shape`, `map_axis`, `cross_hero`, `cross_kit` and one `rule:<id>` per rule. Each is that term's share of eta (T(A) part minus T(B) part, or X(A,B) - X(B,A)). They sum to eta exactly.
- **sd** is the sd of eta when every table entry is independent. A touched entry enters with coefficient (uses by A) - (uses by B). Hero and pair entries are always +1 or -1. If both teams share one role code, its variance cancels instead of counting twice. This is the only reading of "sum of the variances of every table entry touched by either team" that is swap-symmetric and exact.
- **Index order.** Hero and map indices follow the leafpack's sorted order. This is not the kernel's v2 order, where Xal'atath (90) and Haunted Mines (14) are appended. In sorted order Xal'atath is hero 84 and Haunted Mines is map 8. The Python adapter maps by name, so this does not matter here. A CUDA leaf must permute the tables at upload.

## How the adapter hooks into ref_search

`ref_search.py` is unchanged. `RefSearch(game)` accepts any object with `kind, legal, prior, chance, apply, value, n_actions`. `search.HotsDraftGame` is that object for the ranked draft. It follows the v2 kernel (`search_v2.cuh`, SEARCH_CHANCE):

- **Draft order.** The 16-step paper-1 `DRAFT_ORDER`. Our picks and bans are decision nodes. The opponent's turns are chance nodes with GD probabilities and progressive widening (pw_k 1.0, pw_alpha 0.5). PUCT uses c = 2.0.
- **Leaf.** A partial state is rolled out to a complete draft with GD sampling for both teams. Then the leaf is evaluated: sigmoid(eta - kappa * sd), with our team in slots 0-4. The kernel does the same with its WP network, so the leaf never sees a partial draft.
- **Played action.** The argmax of root visits (root_temp 0, the paper-1 operating point). Opponent moves in the played draft are sampled from GD with an RNG seeded only by the configuration, so drafts by different search seeds stay matched.
- **Greedy** follows `paper1_revision/greedy_evals.py` (task_wr). At each own pick, every legal hero is tried. The draft is completed by GD argmax (a random ensemble member per step) and scored by the leaf, and the best hero is played. Own bans and opponent moves use GD argmax.

Differences from the kernel:

- **Prior.** The kernel uses a trained policy head. Here the prior is the GD ensemble mean mixed with a uniform floor (`--prior-eps 0.1`), so heroes GD cannot pick stay reachable. `--prior uniform` is the no-prior variant.
- **GD models.** These are the five v1 models in `training/rerun2026/models` (90 heroes, 14 maps), averaged and matched to the leafpack by name. Opponents never pick or ban Xal'atath, and Haunted Mines gets an all-zero map one-hot. O6 configurations default to the 14 maps GD knows.

## O6 definitions

- **Degenerate** is `shared.is_degenerate` (paper 1): no healer, or no frontline, or 3 or more of Tank, Melee Assassin, Support or Healer. Uther counts as frontline beside a second healer. It is measured on our team over seeded configurations (`random.Random(seed)`: map, tier, side = i % 2).
- **Five-tank and no-healer probes** use the paper-1 `DEGEN_COMPS["5 tanks"]` and `["No healer high WR"]` against `STANDARD` on Cursed Hollow, mid tier. They are flagged above 0.25. The other four paper-1 probes are reported too. The probe teams share Muradin and Valla with the standard team, as in paper 1.
- **Optimism O(W)** is the mean over W's MCTS drafts of sigmoid(eta_W) - sigmoid(eta_J). It is reported per drafter, and G3 uses MCTS-1000.
- **Split-half agreement** uses paper 1's same-step agreement (`paper1_revision/crosseval.py`): the share of our pick steps, on the same configuration and the same step, where two drafters chose the same hero. By default the two drafters are MCTS-1000 on W with search seeds 0 and 1. If a half-B fit exists, pass it with `--split-pack`. The output also reports agreement restricted to steps whose whole draft history is still identical in both runs (`pick_same_state`). The paper-1 definition also counts steps after the drafts have diverged.

  Caveat from the synthetic runs: at 1,000 simulations, with one GD rollout per simulation, the top root candidates differ by 0.01 to 0.02 in Q while each child's rollout noise is about 0.05. Searched bans are near ties too. Seed-to-seed pick agreement is therefore 10% to 20% even for the benign pack, below the G3 bar of 40%. Read G3's split-half item together with the search budget and N's own value under the same protocol.

## Future work: CUDA additive leaf

The kernel calls `wp_eval_symmetrized` (`enriched_features.cuh`) at the leaf in `search_v2.cuh`. An additive leaf would be a device function beside `device_forward.cuh` taking the tables in global memory, permuted to kernel hero and map order. The tables are 91x15x4 + 2x91x91 + 6**6 + small arrays, about 0.4 MB in float32. The function would compute eta with 10 hero lookups, 20 pair lookups, 2 role-code lookups, K interpolations, 25x2 cross lookups and the rules. It is not built here: no CUDA work on this box. The Python leaf is the reference it would be verified against (swap error below 1e-8).
