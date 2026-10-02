# Expert study v6 pool: manifest for review (NOT SEEDED)

Written 2026-10-02. The pool below was generated and labeled; nothing was written to the
database and `data/rating-items.json` (the live v5 pool) was not touched. Seeding waits
for the coordinator's confirmation after independent verification.

All paths are relative to the repo root. Every hash is sha256 of the file as it sits on
the main box after the fetch from the 3090 (`sha256sum <path>`).

## 0. Files to verify

| file | sha256 |
|---|---|
| `training/paper1_revision/results/expert_v6/rating-items-v6.json` (labeled pool, the candidate for seeding) | `6ba2a03256e2ffad9fc1d8c75cc45014e2e54f301f004da5c9c5116947a3e7b4` |
| `training/paper1_revision/results/expert_v6/rating-items-v6.generated.json` (same pool straight from the generator, before the label step) | `a908b89abafe57b2f9cfd600c51d5272d70973f14ee24eb3058ee3e96fca60d6` |
| `training/paper1_revision/results/expert_v6/struct_correction.json` | `bddd826733e2b2ff96cfef509ed9a1547c231e153bf242d9df1f6d311f0b8eab` |
| `training/paper1_revision/results/expert_v6/check-rating-pool.txt` | output of §6.1 |
| `training/paper1_revision/results/expert_v6/rating-pool-stats.txt` | output of §6.2 |
| `training/paper1_revision/results/expert_v6/tier_audit.json` | output of §6.3 |

The two pool files differ only in the label fields: removing `wpTeam0Sym`,
`wpTeam0Sym_uncorrected` and `wpTeam0Sym_tournament` from every item's provenance makes
them identical, and the generator's `wpTeam0Sym` equals the labeled file's
`wpTeam0Sym_tournament` on all 280 machine pairs (checked; §6.4).

## 1. Snapshot

| item | value |
|---|---|
| file | `training/snapshots/replay_snapshot_2026-09-01_sitetiers_2309349.json` |
| sha256 | `441d72264fe43f6fead0ffe7881a2c0800054c68019a4122043f87b6497c2f6d` |
| meta | `training/snapshots/replay_snapshot_2026-09-01_sitetiers_2309349.meta.json` |
| cutoff | game_date < 2026-09-01 (max game_date 2026-08-31T23:59:43, max replay_id 65,611,374) |
| rows | 2,309,349 pulled; 2,301,683 after the pre-2.55 exclusion (`training/rerun2026/pre255_exclude_ids.json`) |
| split | train 2,255,650, test 46,033 (replay-level) |
| tiers | site scheme: low = Bronze+Silver 946,019; mid = Gold+Platinum 962,687; high = Diamond+Master 400,643; 'unknown' excluded; 0 rows disagree with the site rule |
| statistics | decayed90 (half-life 90 days, reference 2026-09-01), training rows only; out of fold (5 hash folds) for training features, deploy stats from the full train split: `training/rerun2026/ns/oct2026/feature_cache/stats/deploy.json` (`d764a7b73e69d6b0ca0dde69a9bf29683fc98d814e2d189b56891126ff4ce9d4`), compositions `deploy_compositions.json` (`09eb509bb10dd7ccbd348d8a0918a395b1a5357e825032c1e3cc1fa3238f8b37`) |

Real-game items (anchors, calibration, screener and catch reals) come from the live DB,
`replay_draft_data`, with `game_date >= 2026-09-01` and `game_version` in
`2.55.17.97771`, `2.55.17.98025` only. Nothing from builds released after 2026-09-27 was
read. The earliest real game in the pool is 2026-09-18 (min replay_id 65,184,997).

## 2. Roster and checkpoints

Namespace `oct2026` (`training/rerun2026/ns/oct2026/`). Ten strategies; synthetic
augmentation removed (no `enriched_aug`). All tournament actors play argmax at every step
(no sampling, no search at play time).

| strategy | checkpoint(s) | sha256 |
|---|---|---|
| mcts | `mcts_runs/F_400sim_s0/draft_policy.pt` | `43030adf2f266c1d57d789744527324409df91eaa6d691d3e4051c5c0e33237d` |
| constrained_mcts | same file as mcts (constrained_search `load_mcts_policy`, patched to F_400sim_s0 via `P1R_MCTS_CKPT`; the only MCTS run in the namespace on the 3090) | same |
| enriched (WP greedy) | `models/wp_enriched_256.pt` | `4ee91ddda386f8ca09692033d421c4bfdb2eb3ea853e56f790074e0cf86c407e` |
| constrained_greedy | `models/wp_enriched_256.pt` | same |
| gd | `models/generic_draft_{0..4}.pt` (gd strategy and the rollout/opponent GD set for every actor) | 0 `37ceb21ea9b35ea11b49983b2a83af0ee8a4029349ba1a40aacdba51dd5237f7`, 1 `912cce5d845c4cfada8a81e97ff1e83beae6a1dd743f4c0924adf4d528db21c2`, 2 `d65b213001e7389f45c132494a84e4729698da789018991038cd2f91a4092afc`, 3 `62fe263a794b36c7c2635dbe8852b3b5d511383ea35dd89dcac13c08ee3d9a62`, 4 `5f5ebeda449804d46ba14a932f8fa5a099382289c91a75dc94a8455ae14aa9fb` |
| cql_naive_a1.0 | `models/cql/_cql_temp_a1.0.pt` | `e37e70d99bfd8a2ba7262f1c0c74f711db17dd5cd8bc568faa5fe5343ea01cb1` |
| cql_enr_a2.0 | `models/cql/_cql_enriched_a2.0.pt` | `99ef7c66abb19edbbd0c037090e96339a50aae8e7cd74354e8704c37a891dd32` |
| mcq_t0.5 | `models/mcq/_mcq_temp_t0.5.pt` | `fa1d72a62614d5d61a19d6566734fe4d75db410fe3bf6eded75b65f73dd791a1` |
| gourdeau | `models/gourdeau_wp.pt` | `c996b491624680dc0b5d689abfea119f26b1cb787080bdfecc0df50269d3b102` |
| gourdeau_disc | `models/gourdeau_discriminator.pt` | `3f4af647ec073b6dac975aff299ade703556fe75414bae86ce83ed3000463f39` |

Training notes (meta files in `models/meta/`, logs in `logs/`):
- GD early stopping (patience 10): GD 0 stopped by hand at about epoch 36 on the main box
  (2026-10-01 18:21, compute move) before early stopping fired; its file is the
  best-test-loss checkpoint (about epoch 31; test loss had moved by under 1e-6 relative
  since epoch 10; `models/meta/gd_0.json` says so). GD 1 epoch 25 (main box). GD 2 epoch 55,
  GD 3 epoch 38, GD 4 epoch 24 (3090). Best test losses 29954.824 to 29954.838.
- cql_naive_a1.0: early stop at epoch 11, test loss 0.7937, action match 13.3%.
  mcq_t0.5: test loss 0.6946. Both trained on the 3090.
- MCTS F_400sim_s0: 300,000 episodes, 400 sims, root argmax, enriched WP leaf
  (`wp_enriched_256.pt`), GD 0 rollouts, value-head pretraining on training replays
  (2,255,650 after excluding test and pre-2.55 ids). 10.0 h at 8.3 episodes/s on the
  3090. `draft_policy.pt` is the best-eval checkpoint (avg_wp 0.7611 at episode 266,240,
  internal proxy). One pause/resume drill at episode 4,096 (clean resume).

### F_400sim_s0 kernel_info.json

`training/rerun2026/ns/oct2026/mcts_runs/F_400sim_s0/kernel_info.json`
(`f11e8beffd4ebdf475d579c95f57bb7f83e780a85a0d60e9df7112eaa0f72b26`):

```json
{"search_mode": "chance", "pw_k": 1.0, "pw_alpha": 0.5, "num_sims": 400,
 "kernel_so": "/home/max/hots/repo/training/cuda_mcts/cuda_mcts_kernel.cpython-314-x86_64-linux-gnu.so",
 "git_head": "", "time": "2026-10-01 19:13:10"}
```

`git_head` is empty because the 3090 copy of the repo is an rsync of the working tree,
not a git checkout. The kernel sources on the 3090 (`cuda_mcts/*.cu, *.cuh, *.h, *.cpp,
setup.py`) had the same md5 as the main box's (which include commit 1db7df8; checked 2026-10-01
18:22); the .so there is built for sm_86 (sha256
`c66b74cdd1b320dbaf3cb2e625745825ec65ba86dc7b8b0b95e9218f096616fa`; the main box's sm_120
build is `f1c21661228adfa2bf5e0dae049464ab226865f0b0e137269496ee19bc86930f`). The
remote smoke test passed before training (`remote_workers/smoke.py --gpu`).

## 3. Evaluators and structure correction

Evaluators (WP models used for pool labels; also the tournament's scoring set):

| evaluator | file | sha256 | test acc / slope (selected seed) |
|---|---|---|---|
| naive | `models/wp_naive.pt` | `108b2aefa571dd0aa925f448c9f9ed31d8e8940d2772303d26428ad372219a02` | 0.574 / 1.15 (seed 123) |
| herostrength | `models/wp_herostrength.pt` | `86d3e2d21fbcacc50866ad44e9d65b52a89ade0e372fab8ae2c712a1119f6f56` | 0.573 / 1.13 (seed 42) |
| enriched | `models/wp_enriched_256.pt` | `4ee91ddda386f8ca09692033d421c4bfdb2eb3ea853e56f790074e0cf86c407e` | see `models/meta/wp_enriched_256.json` |

Seeds were chosen on a validation subset of training rows, not the test set. Each
evaluator's P(team0 wins) is symmetrized over team order. Consensus = mean of the three.

Structure correction (`training/paper1_revision/oct2026_struct_correction.py fit`,
output `struct_correction.json`):

    p* = sigmoid( logit p_J + beta_J . (s(team0) - s(team1)) ),  s = [no_healer, no_frontline, stack]

fitted by logistic regression with the evaluator's own prediction as a fixed offset and
no intercept. Data: Storm League games 2026-09-01 to 2026-09-27, builds 2.55.17.97771 and
2.55.17.98025, site tiers, all 621 real games in the pool excluded: 62,161 games, split
by salted replay-id hash into a fit half (31,071) and a check half (31,090).

| J | beta [no_healer, no_frontline, stack] | SE | check half: degenerate teams (n=1,093) observed / predicted uncorrected / corrected | log-loss uncorr / corr |
|---|---|---|---|---|
| naive | -0.064, 0.029, -0.340 | 0.076, 0.105, 0.297 | 0.441 / 0.448 / 0.437 | 0.6787 / 0.6786 |
| herostrength | -0.050, 0.008, -0.329 | 0.076, 0.105, 0.298 | 0.441 / 0.449 / 0.439 | 0.6793 / 0.6793 |
| enriched | -0.011, 0.043, -0.291 | 0.076, 0.105, 0.296 | 0.441 / 0.438 / 0.436 | 0.6799 / 0.6799 |
| consensus | -0.042, 0.027, -0.319 | 0.076, 0.105, 0.297 | 0.441 / 0.445 / 0.437 | 0.6786 / 0.6785 |

Every coefficient is within about 1.1 SE of zero. On held-out games these evaluators
already price degenerate teams within 1 pp, so the correction changes little (mean
|corrected - uncorrected| consensus 0.0035, max 0.087). The fit file records the pool it
excluded: `pool_sha256 = a908b89a…` (the generated file; same items as the labeled one).

Labels written by `training/paper1_revision/oct2026_pool_judges.py` (two passes, recorded
in the pool's top-level `judges` object):
- `provenance.wpTeam0Sym_uncorrected`: ns:naive, ns:herostrength, ns:enriched; consensus = mean.
- `provenance.wpTeam0Sym`: naive_sc, herostrength_sc, enriched_sc (corrected), consensus =
  consensus_sc (the consensus corrected with its own beta). This is the field the
  preregistration's analyses read.
- `provenance.wpTeam0Sym_tournament`: the values the tournament wrote (uncorrected, 3090
  scoring). They match the main box's uncorrected recomputation to 2.4e-7 (max over 3
  evaluators x 280 items).

## 4. Pool

| item | value |
|---|---|
| generator | `scripts/generate-rating-items.ts` with `RATING_ITEMS_OUT=training/paper1_revision/results/expert_v6/rating-items-v6.json` |
| SEED / PAIR_SAMPLE_SEED | 20261001 / 20261001 |
| tournament namespace | oct2026 (`results/roundrobin/`, 56 ordered pairs; `results/constrained/roundrobin/`, 34 ordered pairs; 200 drafts each; e.g. mcts__gd: low 70, mid 70, high 60) |
| generatedAt | 2026-10-02T15:48:45.545Z |
| items | 901 |

By block and tier:

| block | n | low | mid | high | source |
|---|---|---|---|---|---|
| screener | 8 | 3 | 3 | 2 | real ladder draft vs constructed degenerate team |
| calibration | 40 | 14 | 13 | 13 | ladder |
| pairs | 280 | 93 | 94 | 93 | tournament |
| anchors | 570 | 190 | 190 | 190 | ladder |
| catch | 3 | 1 | 1 | 1 | real vs degenerate |

Pairs by stratum and tier:

| stratum | low | mid | high | total |
|---|---|---|---|---|
| constrained_mcts_vs_mcts | 14 | 14 | 14 | 42 (of 400 candidate records) |
| mcts_vs_enriched | 14 | 14 | 14 | 42 (of 297) |
| vs_anchored | 39 | 40 | 39 | 118 (of 8,950) |
| other_pairs | 26 | 26 | 26 | 78 (of 6,686) |

Strategy appearances across the 280 pairs (each pair counts both sides): mcts 116,
constrained_mcts 79, enriched 78, cql_enr_a2.0 42, gd 42, gourdeau_disc 42,
constrained_greedy 41, cql_naive_a1.0 40, mcq_t0.5 40, gourdeau 40. Bradley-Terry graph:
10 strategies, 45 of 45 pairings present, one connected component.

By map and block:

| map | screener | calibration | pairs | anchors | catch | all |
|---|---|---|---|---|---|---|
| Alterac Pass | 1 | 2 | 20 | 48 | 1 | 72 |
| Battlefield of Eternity | 1 | 3 | 19 | 48 | 0 | 71 |
| Blackheart's Bay | 0 | 0 | 20 | 0 | 0 | 20 |
| Braxis Holdout | 0 | 5 | 20 | 47 | 0 | 72 |
| Cursed Hollow | 1 | 4 | 21 | 46 | 0 | 72 |
| Dragon Shire | 1 | 3 | 19 | 46 | 1 | 70 |
| Garden of Terror | 0 | 3 | 20 | 49 | 0 | 72 |
| Hanamura Temple | 0 | 0 | 20 | 0 | 0 | 20 |
| Haunted Mines | 1 | 5 | 0 | 45 | 0 | 51 |
| Infernal Shrines | 1 | 2 | 20 | 48 | 1 | 72 |
| Sky Temple | 0 | 3 | 20 | 49 | 0 | 72 |
| Tomb of the Spider Queen | 0 | 4 | 21 | 50 | 0 | 75 |
| Towers of Doom | 0 | 3 | 19 | 48 | 0 | 70 |
| Volskaya Foundry | 2 | 3 | 21 | 46 | 0 | 72 |
| Warhead Junction | 0 | 0 | 20 | 0 | 0 | 20 |

Machine pairs follow the tournament's 14-map list (Blackheart's Bay, Hanamura Temple and
Warhead Junction are in it; Haunted Mines is not), as in v5. Real games are drawn from
whatever maps the 2.55.17 ladder games were played on; none of the sampled ones are on
Blackheart's Bay, Hanamura Temple or Warhead Junction.

## 5. Near-ties

Machine pairs with |consensus - 0.5| <= threshold are excluded from the confirmatory test.

| threshold | uncorrected (`wpTeam0Sym_uncorrected`) | corrected (`wpTeam0Sym`, used) | entered / left the band after correction | effective items (corrected) | v5 (uncorrected, 4 evaluators) |
|---|---|---|---|---|---|
| 0.01 | 12 | 10 | 0 / 2 | 270 | 18 |
| 0.02 | 31 | 30 | 0 / 1 | 250 | 30 |
| 0.05 | 86 | 85 | 3 / 4 | 195 | 65 |

Favored-side flips (consensus crosses 0.5) from uncorrected to corrected: 2 items, one of
which is outside the 0.02 band under both labels. Single-evaluator near-tie counts at 0.02
(corrected): naive 35, herostrength 31, enriched 36, consensus 30 (effective items 245 /
249 / 244 / 250). vs_anchored stratum: 118 items, 112 after the 0.02 exclusion.

## 6. Automated checks

### 6.1 `npx tsx scripts/check-rating-pool.ts training/paper1_revision/results/expert_v6/rating-items-v6.json`

Full output in `check-rating-pool.txt`. Uses the app's own slot-assignment code.

```
  ok  ids 1..901 unique
  ok  screener: 8 items / calibration: 40 / pairs: 280 / anchors: 570 / catch: 3
  ok  anchors low/mid/high: 190 each; calibration 14/13/13
  ok  every slot: 240 items, 48-item opener, catch at 121/181/231, 60 pairs + 43/43/43 anchors
  ok  every pair covered exactly 3x
  ok  every anchor covered 3-4x; exactly 32 per tier covered 4x
  ok  test smoke flow: 7 items
  ok  all 621 real replays distinct; all tournament records distinct
  ok  every real game dated >= 2026-09-01
  ok  every item has 5+5 distinct heroes
pool seed 20261001, 901 items: ALL PASS
```

### 6.2 `npx tsx scripts/rating-pool-stats.ts training/paper1_revision/results/expert_v6/rating-items-v6.json`

Full output in `rating-pool-stats.txt` (block/tier/stratum/map counts, near-ties at three
thresholds with uncorrected counterparts, per-evaluator effective items, flips,
Bradley-Terry connectivity). Numbers are those in §4 and §5.

### 6.3 Tier-banner audit: `python3 training/paper1_revision/expert_tier_audit_v6.py`

Output `tier_audit.json`. Real rank from the live DB's `league_tier` (stored one above the
rank; NULL with an MMR = Master), against the banner shown to raters (Low = Bronze-Silver,
Mid = Gold-Platinum, High = Diamond-Master).

| block | real items | outside the shown range |
|---|---|---|
| anchors | 570 | 0 |
| calibration | 40 | 0 |
| screener | 8 | 0 |
| catch | 3 | 0 |
| total | 621 | 0 (v5: 239 of 621) |

Item tier equals the DB's site-scheme `skill_tier` for all 621; none is missing from the
DB. Anchor ranks: low Bronze 103 / Silver 87; mid Gold 103 / Platinum 87; high Diamond 120
/ Master 70. Machine items: the label population behind each site label in the training
window (2.55 builds, before 2026-09-01) is 100% inside the shown range for low, mid and
high (v5's old labels: mid 43%, high 39%).

### 6.4 Other checks run for this manifest

- Uncorrected labels recomputed on the main box equal the tournament's values to 2.4e-7.
- Labeled pool = generated pool apart from the three label fields (script in §0).
- Remote GD caches rebuilt on the 3090 are byte-identical to the main box's on the first
  and last 200 MB of `gd_train/states.dat` and `actions.dat`; row counts equal
  (29,439,680 train / 600,976 test).
- Pretraining exclusion for MCTS reproduced the training split on the 3090
  (2,309,349 -> 2,255,650 replays).
- No `enriched_aug` strategy or `augmented` evaluator appears anywhere in the pool.

## 7. Differences from v5 (seed 20260918, namespace sept2026)

1. **Tiers.** Snapshot relabeled to the site scheme (low = Bronze+Silver, mid =
   Gold+Platinum, high = Diamond+Master), 'unknown' excluded. v5 used the old labels
   (low = Bronze, mid = Silver+Gold+Master, high = Platinum+Diamond); 239 of its 621 real
   items carried a banner outside the game's rank.
2. **Leak-free statistics.** decayed90 statistics from training rows only, out of fold (5
   hash folds) for training features; the own-corpus role-composition table replaces the
   external Heroes Profile table.
3. **Model selection.** WP evaluators: early stopping and seed choice on a validation
   subset of training rows (v5: test set). Enriched CQL transitions and the 20 ensemble
   members use out-of-fold features.
4. **MCTS policy.** F_400sim_s0 (400 sims, 300K episodes, seed 0) instead of J_800sim_s9.
5. **MCTS kernel.** Trained with the X2-fixed kernel (commit 1db7df8, chance nodes at
   opponent turns with progressive widening, pw_k 1.0, pw_alpha 0.5). v5 and every
   earlier MCTS agent used the pre-fix kernel that expanded only the current own-pick
   block. Tournament play is unchanged (policy argmax).
6. **Constrained pairs.** The 'mcts' side of constrained-search pairs now uses the same
   F_400sim_s0 checkpoint as constrained_mcts. In v5 it used a stale July checkpoint
   (L_800sim_4M_s0) because `--mcts-run` was not forwarded; this affected all 42 v5
   `constrained_mcts_vs_mcts` items.
7. **Roster.** Synthetic augmentation removed (Max 2026-10-01): no `enriched_aug`
   strategy, no `augmented` evaluator. 10 strategies, 3 evaluators (v5: 11 and 4).
8. **Labels.** `wpTeam0Sym` is now the structure-corrected 3-evaluator label with a
   separately corrected consensus; the uncorrected values are kept in
   `wpTeam0Sym_uncorrected` and the tournament's in `wpTeam0Sym_tournament`.
9. **Real-game window.** Builds 2.55.17.97771 and 2.55.17.98025 only (v5: any build after
   the cutoff date).
10. **Seed.** 20261001 for both item and pair sampling (v5: 20260918).
11. **Compute.** Stages up to the WP models, GD 0/1, Gourdeau, enriched CQL, the ensemble
    and the discriminator cache ran on the main box. From 2026-10-01 18:20 (Max: no HotS
    compute on the main box) GD 2-4, naive CQL, MCQ, the discriminator, MCTS and both
    tournaments ran on the remote RTX 3090; GD trainers there used 4 data-loader workers
    (`P1R_GD_WORKERS`, shuffle order unchanged) and epoch-level resume. GD 0 was cut short
    (see §2). `onnxscript 0.6.2` was added to the 3090 venv because the GD ONNX export
    needs it; the first GD 2-4 runs finished training and then failed at that export,
    and were completed by resuming from their stopped state.
12. **Unchanged.** Pool design (screener 8, calibration 40, pairs 280 in the same four
    strata and counts, anchors 570, catch 3; 14 slots of 240), generator logic apart from
    seed, builds and namespace, and the app's assignment code.

## 8. Not done (waiting for confirmation)

- Seeding (`scripts/seed-rating-items.ts`; would also delete the 7 `test-watson`
  is_test ratings, exported to `draft_ratings_test_session_2026-09-30.csv`, as Max
  approved).
- Copying the pool to `data/rating-items.json`.
- Browser e2e run (`scripts/e2e-rate.mjs`).
- Prereg v3 and instrument v1.3: no drafts exist yet. Latest existing versions are
  `oss-export/docs/prereg_expert_study_v2.md` (+ .pdf) and
  `oss-export/docs/irb_survey_instrument.pdf`.
