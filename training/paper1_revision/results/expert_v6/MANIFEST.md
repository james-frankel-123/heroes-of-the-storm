# Expert study v6.1 pool: manifest for review (NOT SEEDED)

Written 2026-10-02 after the first verification round (reports in the coordinator's
scratchpad: `xaudit/{claude,grok,codex}_v6/VERIFY.md`). v6.1 regenerates the v6 pool with a
cap on repeated machine teams and a closed real-game window, adds the OOD covariates, and
fixes the manifest's errors. The v6 manifest is kept as `MANIFEST_v6.md`. Nothing was
written to the database and `data/rating-items.json` (the live v5 pool, sha256
`3b870df1…`) is untouched.

Paths are relative to the repo root unless they start with `ns/` (=
`training/rerun2026/ns/oct2026/`) or sit in this directory
(`training/paper1_revision/results/expert_v6/`). Hashes are sha256 on the main box.

## 0. Files to verify

| file (this directory) | what | sha256 |
|---|---|---|
| `rating-items-v6.1.json` | **the candidate to seed**: generated pool + labels + OOD covariates | `d699edcf511c1eeeceeae5df05db05f88724cc094c7564a509d1d4b9502b6ef5` |
| `rating-items-v6.1.generated.json` | generator output before the label and OOD steps | `681850287d0145b2abf2f91f51cd94a2911c94e9589f08a199aa09d5bc249a35` |
| `rating-items-v6.1.generated.ladder-candidates.json` | the ladder query's 6,000 candidate replay ids per tier | `192b0f543a675098f221ba6314b6d218b8c7a73c0a67062e5bb6223d58218968` |
| `struct_correction.json` | structure-correction fit (v6.1) | `f8503c845301f61a3471d9eaecd669b255107380ffcb1250b9accca3fd658038` |
| `struct_fit_games.json` | the replay ids of the fit and check halves | `3ca043533417ef8e2d8c576472ff7131c225825c338adbf891b564af7bb23ea2` |
| `check-rating-pool-v6.1.txt` | §6.1 output | `a0e0408d0dc09ec6c3c61c0ab9be1276e83456a066c4d374b5a0a886a3ebca4b` |
| `rating-pool-stats-v6.1.txt` | §6.2 output | `8567d03dd1f44d8e543cf4d1d85ca5248cab43910a029c8a1f4eba99036c1054` |
| `tier_audit_v6.1.json` | §6.3 output | `7977613be01a72a6b0b5610f09f7ad43e112528f47afb75d62148ac3f1c178bc` |
| `gd_valid_eval.json` | §2.2 GD metrics on valid rows | `e3ffe7df2feccb3cb470e77b3e608db65fbb1bdea23b0578b8d7bdb60650f418` |

The v6 files (`rating-items-v6*.json`, `struct_correction_v6.json`, `tier_audit.json`,
`check-rating-pool.txt`, `rating-pool-stats.txt`) are kept for the record and are not
candidates.

Normalization between the two pool files: removing every provenance key that starts with
`wpTeam0Sym` or `ood_`, and the top-level `judges` object, makes
`rating-items-v6.1.json` equal to `rating-items-v6.1.generated.json` (checked). The
generator's `wpTeam0Sym` equals the candidate's `wpTeam0Sym_tournament` on all 280 pairs.
A re-run of the generator reproduces the generated file except the `generatedAt` line
(the clock), provided the DB returns the same candidate ids; the candidate file makes
that checkable after the DB grows.

## 1. Snapshot and real-game window

| item | value |
|---|---|
| snapshot | `training/snapshots/replay_snapshot_2026-09-01_sitetiers_2309349.json`, sha256 `441d72264fe43f6fead0ffe7881a2c0800054c68019a4122043f87b6497c2f6d` |
| cutoff | DB game_date < 2026-09-01 (max 2026-08-31T23:59:43, max replay_id 65,611,374) |
| rows | 2,309,349; 2,301,683 after the pre-2.55 exclusion (`training/rerun2026/pre255_exclude_ids.json`); 'unknown' tier excluded in the pull |
| split | train 2,255,650, test 46,033 (replay-level) |
| tiers | site scheme: low = Bronze+Silver 946,019; mid = Gold+Platinum 962,687; high = Diamond+Master 400,643; 0 rows disagree with the site rule |
| statistics | decayed90 (half-life 90 days, reference 2026-09-01), training rows only; out of fold (5 hash folds) for training features; deploy stats `ns/feature_cache/stats/deploy.json` (`d764a7b7…`), compositions `deploy_compositions.json` (`09eb509b…`) |

**Real-game window (v6.1).** `scripts/generate-rating-items.ts` reads real games from
`replay_draft_data` with `game_date >= '2026-09-01' AND game_date < '2026-09-28'` and
`game_version IN ('2.55.17.97771', '2.55.17.98025')`, newest 6,000 per tier
(`order by game_date desc, replay_id desc`). All 621 real games drawn are on build
2.55.17.98025 (the newest 6,000 per tier all fall after 97771's last game on 09-12).
DB game_date of the 621 real games: 2026-09-14 18:25:37 to 2026-09-27 23:36:33 (anchors
09-14 to 09-27, calibration 09-15 to 09-27, screener 09-18 to 09-27, catch 09-17 to
09-25); min replay_id 65,176,068.

**Clock.** `game_date` is a timezone-less DB timestamp. In v6.1 the generator selects it
as text (`to_char(game_date, 'YYYY-MM-DD"T"HH24:MI:SS')`) and stores that string
unchanged in `provenance.gameDate` (no `Z`), with the build in `provenance.gameVersion`.
In v6 the Node driver read the value as local EDT and serialized it as UTC, so stored
values were the DB value + 4 h. The DB value is the canonical clock everywhere in this
study (the structure fit, the window, the checks).

## 2. Roster and checkpoints

Namespace `oct2026`. Ten strategies; synthetic augmentation removed (`enriched_aug` not
played; `ns/models/wp_aug_v2_512.pt` was trained by the pipeline before the removal and
is used by nothing).

| strategy | checkpoint(s) | sha256 | play at each step |
|---|---|---|---|
| mcts | `ns/mcts_runs/F_400sim_s0/draft_policy.pt` | `43030adf2f266c1d57d789744527324409df91eaa6d691d3e4051c5c0e33237d` | policy argmax, no search |
| constrained_mcts | same file (via `P1R_MCTS_CKPT`; replays 12/12 with it, 0/12 with the July L_800sim_4M_s0) | same | policy argmax over role-legal heroes |
| enriched | `ns/models/wp_enriched_256.pt` | `4ee91ddda386f8ca09692033d421c4bfdb2eb3ea853e56f790074e0cf86c407e` | one-step lookahead: each legal hero is rolled out to a full draft with randomly chosen GD members, best terminal WP wins |
| constrained_greedy | `ns/models/wp_enriched_256.pt` | same | as enriched, over role-legal heroes |
| gourdeau | `ns/models/gourdeau_wp.pt` | `c996b491624680dc0b5d689abfea119f26b1cb787080bdfecc0df50269d3b102` | as enriched, Gourdeau WP at the terminal |
| gd | `ns/models/generic_draft_{0..4}.pt` | 0 `37ceb21e…`, 1 `912cce5d…`, 2 `d65b2130…`, 3 `62fe263a…`, 4 `5f5ebeda…` (full hashes in `MANIFEST_v6.md` §2, unchanged) | a random GD member each step, its argmax |
| cql_naive_a1.0 | `ns/models/cql/_cql_temp_a1.0.pt` | `e37e70d99bfd8a2ba7262f1c0c74f711db17dd5cd8bc568faa5fe5343ea01cb1` | Q argmax |
| cql_enr_a2.0 | `ns/models/cql/_cql_enriched_a2.0.pt` | `99ef7c66abb19edbbd0c037090e96339a50aae8e7cd74354e8704c37a891dd32` | Q argmax |
| mcq_t0.5 | `ns/models/mcq/_mcq_temp_t0.5.pt` | `fa1d72a62614d5d61a19d6566734fe4d75db410fe3bf6eded75b65f73dd791a1` | Q argmax |
| gourdeau_disc | `ns/models/gourdeau_discriminator.pt` | `3f4af647ec073b6dac975aff299ade703556fe75414bae86ce83ed3000463f39` | minimize P(generated) |

The tournament results (`ns/results/roundrobin/`, 56 files; `ns/results/constrained/
roundrobin/`, 34 files; 200 drafts each) are unchanged from v6; v6.1 only re-samples
from them.

### 2.1 Training notes and model selection

- WP evaluators (naive, herostrength, enriched): early stopping and seed choice on a
  validation subset of training rows; test rows never used for selection.
- GD, CQL and MCQ still select on test loss (GD: best-test-loss checkpoint, patience 10;
  CQL/MCQ: test-loss early stopping), as in September. Test games are never pool games.
  GD early stopping: GD 0 stopped by hand at about epoch 36 (main box, 2026-10-01 18:21)
  before early stopping fired, its best-test-loss checkpoint (about epoch 31) is used;
  GD 1 epoch 25; GD 2 epoch 55; GD 3 epoch 38; GD 4 epoch 24. cql_naive_a1.0 stopped at
  epoch 11 (test loss 0.7937); mcq_t0.5 test loss 0.6946.
- MCTS F_400sim_s0: 300,000 episodes at 400 simulations with the X2-fixed kernel, root
  argmax, enriched WP leaf, GD 0 rollouts, seed 0; 10.0 h on the RTX 3090; one clean
  pause/resume at episode 4,096. `draft_policy.pt` is the best-eval checkpoint (internal
  proxy avg_wp 0.7611 at episode 266,240). The value-head pretraining step ran but did
  nothing: its loss stayed at 0.2500 (the constant-0.5 MSE) for all 30 epochs, as in
  September's J_800sim_s9; self-play trains the value head.

### 2.2 GD held-out metrics on valid rows

The GD test loss in the logs (29954.8) is dominated by rows whose target hero is already
picked or banned: the mask sets that logit to -1e9, so each such row adds about 1e9 to the
summed loss. `paper1_revision/oct2026_gd_valid_eval.py` counts those rows and evaluates
GD 0-4 on the rest (`gd_valid_eval.json`):

| model | test CE, all rows (as logged) | test CE, valid rows | top-1, valid | top-5, valid |
|---|---|---|---|---|
| GD 0 | 29954.83 | 3.5476 | 0.1303 | 0.3912 |
| GD 1 | 29954.83 | 3.5486 | 0.1297 | 0.3909 |
| GD 2 | 29954.83 | 3.5460 | 0.1302 | 0.3920 |
| GD 3 | 29954.83 | 3.5465 | 0.1305 | 0.3915 |
| GD 4 | 29954.84 | 3.5598 | 0.1284 | 0.3882 |

Invalid rows: test 18 of 600,976 (indices in the json; the same 18 rows the codex
verifier listed); train 822 of 29,439,680 (0.0028%). In
the logged training loss each invalid row adds a 1e9-scale term; the paper-1 lane is handling
GD retraining separately. GD 0, stopped by hand before early stopping, scores within the
range of GD 1-3 on valid rows (CE 3.548 vs 3.546-3.549; top-1 0.130 vs 0.130-0.131); v6's
argument that its logged loss "moved by under 1e-6" was not evidence of convergence,
because that loss is dominated by the 18 invalid rows. `gd_valid_eval.json` sha256 is listed in §0.

GD was not retrained for the pool; the tournament records show how each actor actually
played.

### 2.3 F_400sim_s0 kernel_info.json

`ns/mcts_runs/F_400sim_s0/kernel_info.json` (`f11e8beffd4ebdf475d579c95f57bb7f83e780a85a0d60e9df7112eaa0f72b26`):

```json
{"search_mode": "chance", "pw_k": 1.0, "pw_alpha": 0.5, "num_sims": 400,
 "kernel_so": "/home/max/hots/repo/training/cuda_mcts/cuda_mcts_kernel.cpython-314-x86_64-linux-gnu.so",
 "git_head": "", "time": "2026-10-01 19:13:10"}
```

The file has no episode count; the episode count comes from the job command
(`MCTS_NUM_EPISODES=300000`) and the log (`ns/logs/remote_jobs/oct26_mcts_F400.log`:
"Config: episodes=300000" ... "Episode 300000"; "Complete. 295904 episodes" counts the
episodes after the resume at 4,096). `git_head` is empty because the 3090 copy is an
rsync of the working tree. The 3090 kernel sources had the same md5 as the main box's
(which include commit 1db7df8), the worker refuses to run in chance mode on a pre-fix
build, and the log prints "Search: chance (mode 1, pw_k=1.0, pw_alpha=0.5)". The 3090
`.so` (sm_86) hashes to `c66b74cd…`; this was read on the 3090.

### 2.4 Code

All code the tournament and the pool steps import is committed (v6.1 commit). Files that
were uncommitted during the tournament and are now committed as they stood (unchanged
since May-September; not another job's work in progress):
`training/experiment_rich_evaluation.py` (defines `make_cql_enriched_strategy`,
`make_gourdeau_greedy_strategy`), `training/experiment_synthetic_augmentation.py`
(`ENRICHED_GROUPS`), `training/experiment_mcq_draft.py`, `training/retrain_frozen_stats.py`,
`training/rerun2026/{train_jobs,phase0_features,phase1_models,phase3_benchmarks,
train_gourdeau_discriminator,ensemble_uncertainty}.py` and
`training/rerun2026/pre255_exclude_ids.json`.

After the tournament, two guards were added (no change to what any actor does):
`rerun2026/common.require_ns_hooks()` makes every `common.setup()` entry point fail when
`RERUN_NS=oct2026` runs without `oct2026_site/sitecustomize.py`, `P1R_COMP_PATH` and
`P1R_MCTS_CKPT`; `phase3b._find_mcts_checkpoint` refuses to fall back to
`training/mcts_runs/` for an explicitly requested run in a namespace (and never falls
back in oct2026). `paper1_revision/oct2026_env.sh` runs any command in the oct2026 env.

## 3. Evaluators and structure correction

| evaluator | file | sha256 |
|---|---|---|
| naive | `ns/models/wp_naive.pt` | `108b2aefa571dd0aa925f448c9f9ed31d8e8940d2772303d26428ad372219a02` |
| herostrength | `ns/models/wp_herostrength.pt` | `86d3e2d21fbcacc50866ad44e9d65b52a89ade0e372fab8ae2c712a1119f6f56` |
| enriched | `ns/models/wp_enriched_256.pt` | `4ee91ddda386f8ca09692033d421c4bfdb2eb3ea853e56f790074e0cf86c407e` |

    p* = sigmoid( logit p_J + beta_J . (s(team0) - s(team1)) ),  s = [no_healer, no_frontline, stack]

Offset logistic regression, no intercept. Data: Storm League games with DB game_date
2026-09-01 to 2026-09-27, builds 2.55.17.97771/98025, site tiers, with all 621 real games
of the v6.1 pool removed (all 621 fall inside the window): 62,355 games (the DB gained
late uploads since the v6 fit, which used 62,161 with 567 pool games removed); fit half
31,172, check half 31,183. The game ids are frozen in `struct_fit_games.json`.

| J | beta [no_healer, no_frontline, stack] | SE | check half: degenerate teams (n=1,099) observed / predicted uncorrected / corrected | log-loss uncorr / corr |
|---|---|---|---|---|
| naive | -0.057, 0.017, -0.339 | 0.076, 0.105, 0.297 | 0.442 / 0.449 / 0.438 | 0.6785 / 0.6785 |
| herostrength | -0.044, -0.004, -0.327 | 0.076, 0.105, 0.298 | 0.442 / 0.449 / 0.438 | 0.6792 / 0.6791 |
| enriched | -0.005, 0.032, -0.290 | 0.076, 0.105, 0.296 | 0.442 / 0.437 / 0.436 | 0.6798 / 0.6798 |
| consensus | -0.035, 0.015, -0.317 | 0.076, 0.105, 0.297 | 0.442 / 0.445 / 0.437 | 0.6784 / 0.6784 |

All coefficients are within about 1.1 SE of zero. The consensus is the mean of the three
uncorrected evaluators, corrected with its own beta; it is not the mean of the corrected
evaluators (they differ by up to about 0.003). Mean |corrected - uncorrected| consensus
0.0030, max 0.083.

Labels (two runs of `paper1_revision/oct2026_pool_judges.py`, recorded in the pool's
top-level `judges`): `wpTeam0Sym_uncorrected` (ns:naive, ns:herostrength, ns:enriched,
consensus = mean); `wpTeam0Sym` (naive_sc, herostrength_sc, enriched_sc, consensus =
consensus_sc), the field the preregistration v3 reads; `wpTeam0Sym_tournament` (the
tournament's values; they equal the main box's uncorrected recomputation to 2.7e-7).

**This is a change to the registered endpoint.** Prereg v2 registered the uncorrected
mean of four evaluators. v3 registers the corrected three-evaluator consensus for H1, the
near-tie rule, the per-evaluator sensitivities, S2 and S5 (prereg v3 §0).

## 4. Pool

| item | value |
|---|---|
| generator | `scripts/generate-rating-items.ts`, `RATING_ITEMS_OUT=training/paper1_revision/results/expert_v6/rating-items-v6.1.generated.json` |
| SEED / PAIR_SAMPLE_SEED | 20261001 / 20261001 (unchanged from v6) |
| generatedAt | 2026-10-02T16:54:25.082Z |
| team cap | `MAX_TEAM_APPEARANCES = 7`: an identical sorted five-hero team may appear in at most 7 of the 280 pairs, counted across strata (v5's maximum); recorded in the pool as `maxTeamAppearances` |
| window | `realGameEnd = 2026-09-28`, `realGameBuilds = [2.55.17.97771, 2.55.17.98025]`, recorded in the pool |
| label + OOD steps | §3; `RATING_ITEMS_OOD_PATH=<pool> bash paper1_revision/oct2026_env.sh python3 rerun2026/rating_items_ood.py` |

**Machine-team repetition.**

| | v5 | v6 | v6.1 |
|---|---|---|---|
| most frequent team, appearances in 280 pairs | 7 | 21 | 7 (7 teams at the cap) |
| per-rater maximum exposure to one team (slots 0-13) | 3 3 3 4 3 3 2 2 2 3 4 4 2 3 | 6 8 6 8 6 6 4 6 4 4 3 6 6 6 | 3 4 4 3 3 4 4 5 3 3 3 3 2 2 |

v6.1 has 451 distinct machine teams (appearance histogram: 1x 401, 2x 28, 3x 10, 4x 3,
6x 2, 7x 7). The per-rater maximum is 5 for one slot (v5: 4). The cap was feasible in
every stratum with the stratum counts and tier balance unchanged. Five matchups (same two
teams) appear twice in the pool on different maps or tiers (v6: three).

By block and tier: screener 8 (3/3/2), calibration 40 (14/13/13), pairs 280 (low 93 / mid
94 / high 93), anchors 570 (190 per tier), catch 3 (1/1/1).

Pairs by stratum and tier:

| stratum | low | mid | high | total (candidate records) |
|---|---|---|---|---|
| constrained_mcts_vs_mcts | 14 | 14 | 14 | 42 (400) |
| mcts_vs_enriched | 14 | 14 | 14 | 42 (297) |
| vs_anchored | 39 | 40 | 39 | 118 (8,950) |
| other_pairs | 26 | 26 | 26 | 78 (6,686) |

Strategy appearances across the 280 pairs: mcts 115, enriched 80, constrained_mcts 78,
cql_enr_a2.0 43, gourdeau_disc 42, gd 42, constrained_greedy 42, gourdeau 41,
cql_naive_a1.0 41, mcq_t0.5 36. Bradley-Terry: 10 strategies, 45 of 45 pairings, one
component.

By map and block:

| map | screener | calibration | pairs | anchors | catch | all |
|---|---|---|---|---|---|---|
| Alterac Pass | 1 | 2 | 20 | 47 | 1 | 71 |
| Battlefield of Eternity | 0 | 3 | 20 | 48 | 0 | 71 |
| Blackheart's Bay | 0 | 0 | 19 | 0 | 0 | 19 |
| Braxis Holdout | 0 | 4 | 19 | 47 | 1 | 71 |
| Cursed Hollow | 0 | 5 | 20 | 48 | 0 | 73 |
| Dragon Shire | 0 | 5 | 20 | 47 | 0 | 72 |
| Garden of Terror | 1 | 4 | 20 | 46 | 0 | 71 |
| Hanamura Temple | 0 | 0 | 20 | 0 | 0 | 20 |
| Haunted Mines | 0 | 4 | 0 | 48 | 0 | 52 |
| Infernal Shrines | 0 | 3 | 20 | 48 | 1 | 72 |
| Sky Temple | 1 | 3 | 20 | 49 | 0 | 73 |
| Tomb of the Spider Queen | 1 | 3 | 20 | 47 | 0 | 71 |
| Towers of Doom | 1 | 3 | 21 | 48 | 0 | 73 |
| Volskaya Foundry | 3 | 1 | 20 | 47 | 0 | 71 |
| Warhead Junction | 0 | 0 | 21 | 0 | 0 | 21 |

Machine pairs follow the tournament's 14-map list; real games come from whatever maps the
sampled ladder games were played on.

## 5. Near-ties and OOD covariates

| threshold | uncorrected | corrected (used) | entered / left after correction | effective items (judgments) |
|---|---|---|---|---|
| 0.01 | 9 | 8 | 1 / 2 | 272 (816) |
| 0.02 | 25 | 25 | 0 / 0 | 255 (765) |
| 0.05 | 76 | 76 | 2 / 2 | 204 (612) |

Favored-side flips from uncorrected to corrected: 2 (one outside the 0.02 band under both
labels). Single-evaluator effective items at 0.02 (corrected): naive 249, herostrength
254, enriched 245. vs_anchored: 118 items, 114 after the 0.02 exclusion (342 judgments).
Item-level worst-case z at true 0.60: 3.30 / 3.19 / 2.86.

OOD covariates (20 oct2026 ensemble members, deploy statistics, oct2026 env; deterministic:
an independent re-run gave the same file hash):

| covariate | mean | median | p90 | p95 | max |
|---|---|---|---|---|---|
| `ood_var_max` | 0.00121 | 0.00101 | 0.00215 | 0.00254 | 0.00487 |
| `ood_var_matchup` | 0.00105 | 0.00083 | 0.00182 | 0.00241 | 0.00601 |

(numpy percentiles; `rating-pool-stats.ts` uses a different percentile rule and prints
p90/p95 0.00221/0.00255 and 0.00186/0.00243.) Spearman(`ood_var_max`, `ood_var_matchup`)
= 0.384. Reference substitutions on 181 of 560 probe records.

## 5b. Patch 2.57 heroes

Patch 2.57 (first games 2026-09-28) added Xal'atath and changed six heroes (nerfed:
Garrosh, Whitemane, Abathur, Qhira; buffed: Mal'Ganis, Yrel). The pool predates it:
Xal'atath appears in no item. Items with any of the six on either team:

| block | items | affected | low / mid / high affected |
|---|---|---|---|
| screener | 8 | 5 | 1 / 2 / 2 |
| calibration | 40 | 18 | 4 / 6 / 8 |
| pairs | 280 | 60 | 16 / 19 / 25 |
| anchors | 570 | 316 | 93 / 106 / 117 |
| catch | 3 | 1 | 1 / 0 / 0 |

Pairs by stratum: constrained_mcts_vs_mcts 4 of 42, mcts_vs_enriched 12 of 42,
vs_anchored 22 of 118, other_pairs 22 of 78. Hero appearances across all items:
Whitemane 118, Garrosh 112, Abathur 85, Qhira 76, Yrel 75, Mal'Ganis 36.

Effective n after the exclusion (prereg v3 §4 sensitivity, descriptive): H1 at 0.02
255 -> 200 items (765 -> 600 judgments); at 0.01 272 -> 214; at 0.05 204 -> 160; S5
114 -> 93 items; S1 anchors per tier 190 -> 97 / 84 / 73 (low / mid / high).

## 6. Automated checks

### 6.1 `npx tsx scripts/check-rating-pool.ts training/paper1_revision/results/expert_v6/rating-items-v6.1.json`

Extended in v6.1 with the upper date bound, the build allow-list, the team cap (and the
per-rater exposure line), label presence and OOD presence and consistency. It prints which
requirements it does not test (label values, DB agreement, tier vs rank).

```
  ok  ids 1..901 unique
  ok  screener 8 / calibration 40 / pairs 280 / anchors 570 / catch 3; anchors 190 per tier; calibration 14/13/13
  ok  every slot: 240 items, 48-item opener, catch at 121/181/231, 60 pairs + 43/43/43 anchors
  ok  every pair covered exactly 3x; every anchor 3-4x, exactly 32 per tier 4x
  ok  test smoke flow: 7 items
  ok  all 621 real replays distinct; all tournament records distinct
  ok  every real game dated >= 2026-09-01
  ok  every item has 5+5 distinct heroes
  ok  every real game DB date < 2026-09-28
  ok  every real game on builds 2.55.17.97771, 2.55.17.98025
  ok  no machine team in more than 7 pairs (max 7)
  ..  per-rater max exposure to one machine team (slots 0-13): 3 4 4 3 3 4 4 5 3 3 3 3 2 2
  ok  every pair has finite wpTeam0Sym and wpTeam0Sym_uncorrected (naive, herostrength, enriched, consensus)
  ok  every pair has finite, consistent OOD covariates (team0/1, max, mean, matchup, refs)
pool seed 20261001, 901 items: ALL PASS
```

### 6.2 `npx tsx scripts/rating-pool-stats.ts training/paper1_revision/results/expert_v6/rating-items-v6.1.json`

Output in `rating-pool-stats-v6.1.txt`; the numbers are those in §4 and §5.

### 6.3 Tier-banner audit: `TIER_AUDIT_OUT=… python3 training/paper1_revision/expert_tier_audit_v6.py <pool>`

Real rank from the live DB's `league_tier` (stored one above the rank; NULL with an MMR =
Master): 0 of 621 real items outside the banner; item tier equals DB `skill_tier` for all
621; none missing. Anchor ranks: low Bronze 102 / Silver 88; mid Gold 110 / Platinum 80;
high Diamond 122 / Master 68. Each site label's training-window population is 100%
inside its shown range.

### 6.4 Other checks

- Uncorrected labels recomputed on the main box equal the tournament's to 2.7e-7.
- Normalization in §0 holds; the OOD step is deterministic.
- The struct fit excluded all 621 pool games (`pool_replays_in_fit_window_excluded: 621`;
  v6's file called the id-list size `excluded_pool_replays`, 621, while only 567 were in
  that window).
- No `enriched_aug` strategy or `augmented` evaluator anywhere in the pool.

## 7. Differences

### 7.1 v6.1 vs v6 (seed unchanged)

1. Team cap 7 across the 280 pairs (v6: 21). Pair sample therefore differs in the
   vs_anchored and other_pairs strata (the first two strata were already under the cap
   and are identical).
2. Real games bounded to DB game_date < 2026-09-28 (v6: 54 real items on 09-28); the
   ladder draw therefore differs. The candidate ids are saved.
3. `provenance.gameDate` stores the DB value; `provenance.gameVersion` added; the pool
   records `realGameEnd`, `realGameBuilds`, `maxTeamAppearances`.
4. OOD covariates present (v6: absent).
5. Structure correction refit on the v6.1 exclusion set and the DB as of today
   (coefficients move by at most 0.012; game list frozen).
6. Checks extended (§6.1); oct2026 env guard; namespaced MCTS checkpoint guard; all
   imported code committed.

### 7.2 v6.1 vs v5 (seed 20260918, namespace sept2026)

1. Tiers: site scheme (v5: old labels; 239 of 621 v5 real items outside their banner;
   v6.1: 0).
2. Leak-free statistics: training rows only, out of fold; own-corpus composition table.
3. WP evaluators select on a validation subset of training rows (v5: test set). GD, CQL
   and MCQ still select on test loss in both.
4. MCTS: F_400sim_s0 (400 sims, 300K episodes, seed 0, X2-fixed chance kernel) instead of
   J_800sim_s9 (pre-fix kernel).
5. Constrained pairs: the 'mcts' side uses F_400sim_s0 (v5: a stale July L_800sim_4M_s0
   for all 42 constrained_mcts_vs_mcts items).
6. Roster: 10 strategies and 3 evaluators (v5: 11 and 4; synthetic augmentation removed).
7. Tournament: 56 + 34 ordered pairs x 200 drafts, replayed 2026-10-02 (v5: 72 + the
   constrained pairs, 2026-09-19).
8. Labels: structure-corrected 3-evaluator consensus (v5: uncorrected 4-evaluator mean),
   a change to the registered endpoint (§3).
9. Real games: DB date 2026-09-01 to 2026-09-27, builds 2.55.17.97771/98025 (v5: any
   build on or after 2026-09-01).
10. Team cap 7 (v5 had no cap; its maximum happened to be 7).
11. Seed 20261001 (v5: 20260918).
12. Compute: from 2026-10-01 18:20 the GPU stages and both tournaments ran on the RTX
    3090 (Max: no HotS compute on the main box); GD there used 4 data-loader workers
    (shuffle order unchanged) and epoch resume; `onnxscript 0.6.2` was added to its venv
    for the GD ONNX export.
13. Unchanged: pool design (8 / 40 / 280 / 570 / 3, strata 42/42/118/78, 14 slots of
    240), assignment code, rater-facing text.

## 8. Not done (waiting for confirmation)

- Copying `rating-items-v6.1.json` to `data/rating-items.json` and seeding with
  `scripts/seed-rating-items.ts` (deletes the 7 `test-watson` is_test ratings, exported
  to `draft_ratings_test_session_2026-09-30.csv`, as Max approved).
- Browser e2e run (`scripts/e2e-rate.mjs`).
- Prereg v3 and instrument v1.3 drafts: `oss-export/docs/prereg_expert_study_v3.md`
  (+ .pdf) and `oss-export/docs/irb_survey_instrument_v1.3.pdf` (untracked in the
  separate oss-export repo, like v2).
