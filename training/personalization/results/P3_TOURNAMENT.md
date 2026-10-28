# P3 tournament: player-aware vs player-blind drafters, and drafters under drift (started 2026-10-19)

## Definitive results (run def1027, finished 2026-10-28)

**Run.** Build def1027 (full player-history pull, CAP_TS 2026-09-28), two passes by `remote/p3t_definitive.sh`:
- **Headline:** the personalization lane's chosen state model, as the drafters' and the judge's player states, with disjoint player halves.
- **Sensitivity:** the old full_ms model.

Each pass has its own drafts, scores, judge and combiners. Results are in `results/p3t_def1027_headline/` and `results/p3t_def1027_oldmodel/`, tables in each folder's `P3_TOURNAMENT_TABLES.md`.
- **Lobbies:** 300 fresh headline lobbies. They exclude the earlier tune and tournament lobbies by replay ID (`p3t_excluded_replays.json`). The tune set is the earlier tune set.
- **Pairs:** 566 ordered pairs, both seats.
- **Gates and code pins:** determinism and parity gates passed on both GPU hosts in both passes. Each pass pinned the code it ran on every host and re-verified it at the end.
- **Judge:** 144,414 judge-window games. c2 3.05 (SE 0.04), lift 23.2 ×1000 (SE 0.55); old-model pass: c2 4.62, lift 20.0.
- **J_cal (fitted this pass):** 1.18 logit p_cons + 1.00 personal term.
- **Attenuation λ:** 0.55 (old model 0.39). The new judge credits about 0.55 of a true personal difference.

Intervals are 95%, lobby bootstrap widened for seed variation, both seats. "Record" is the judge-free paired record: each player's later realized record on the hero each arm put him on, over the players covered under both arms, in pp per player.

### The selected design against the baselines (headline pass)

`x-combo1-dr-chance`, the design picked on the tune lobbies before the prototype tournament. Row = its win % (or its record advantage):

| vs | J | J_cal | J outcome-free | CONS | record (pp) | coverage both |
|---|---|---|---|---|---|---|
| paper 1's MCTS (`pMCTS-cur`) | **59.0** (58.1, 60.0) | 58.1 (57.0, 59.1) | 60.7 (59.7, 61.8) | 44.3 (43.5, 45.1) | **+4.8** (2.2, 7.3) | 37% |
| Real drafts | **63.0** (62.3, 63.7) | 64.4 (63.7, 65.2) | 63.5 (62.5, 64.5) | 58.9 (58.4, 59.4) | **+3.1** (1.0, 5.1) | 56% |
| GD | **66.0** (65.2, 66.8) | 67.4 (66.6, 68.1) | 66.8 (66.2, 67.4) | 58.7 (58.2, 59.2) | +3.6 (1.5, 5.8) | 48% |
| DraftRec | **58.1** (57.4, 58.7) | 59.3 (58.6, 60.0) | 57.9 (57.3, 58.6) | 57.2 (56.7, 57.7) | +2.1 (0.4, 3.8) | 64% |
| DraftRec-pv | **55.1** (54.5, 55.7) | 55.3 (54.7, 56.0) | 55.7 (54.9, 56.5) | 51.0 (50.6, 51.5) | +3.5 (1.6, 5.5) | 53% |
| `aMCTS-pp` (the design comparison fixed before the tournament) | 51.6 (51.1, 52.2) | 51.4 (50.9, 52.0) | 52.0 (51.5, 52.6) | 48.8 (48.4, 49.3) | +0.7 (−1.0, 2.3) | |

- **Reading the record column.** In `analysis.json` each row is from the first-named system's side. The `pMCTS-cur vs x-combo1-dr-chance` row has −4.77 (−7.34, −2.20), which is +4.8 (2.2, 7.3) for the selected design. Real, GD, DraftRec and DraftRec-pv read the same way: the selected design's players later win more on its heroes than on the other arm's heroes.
- **Not the first positive judge-free result.** The prototype also had a positive judge-free result: +3.7 (1.6, 5.8) vs paper 1's MCTS and +2.7 (1.0, 4.3) vs real drafts. It used the unpaired record, compared over different players. The definitive numbers are the first on paired coverage (the same players under both arms) and with seed SE included. All five are positive with CIs above 0.
- **Coverage.** Against paper 1's MCTS the record rests on the 37% of players covered under both arms. Paper 1's MCTS puts players on heroes they rarely play (its own coverage is 44%, the selected design's 77%), so most of its picks have no later record.
- **Outcome-free J.** It scores the judge with players placed on heroes by their earlier games, not by the best of 120 assignments. It is within about 1.7pp of J everywhere, and higher against paper 1's MCTS (60.7 vs 59.0). The best-assignment choice does not flatter the selected design.
- **CONS.** Under the population judges the selected design still loses to paper 1's MCTS (44.3%) and beats real drafts (58.9%). This is the prototype pattern: the gain is personal, bought with some population win probability.

**Against the prototype (2026-10-21, store full_1017, old states) and the old-model pass:**

| selected design vs | prototype J | old-model pass J | headline J | old-model pass record | headline record |
|---|---|---|---|---|---|
| paper 1's MCTS | 60.9 (60.3, 61.6) | 61.0 (59.8, 62.2) | 59.0 (58.1, 60.0) | +4.0 (1.5, 6.4) | +4.8 (2.2, 7.3) |
| Real | 62.1 (61.5, 62.8) | 62.5 (61.3, 63.7) | 63.0 (62.3, 63.7) | +3.1 (1.3, 4.9) | +3.1 (1.0, 5.1) |
| GD | 66.9 (66.3, 67.5) | 66.2 (65.6, 66.8) | 66.0 (65.2, 66.8) | +5.8 (3.6, 7.9) | +3.6 (1.5, 5.8) |

- **Wider intervals.** The definitive intervals are wider than the prototype's because the seed SE of the exploration searches now counts (it was 0 before the C14 fix).
- **J_cal.** It moves as in the prototype: 58.1 vs paper 1's MCTS (prototype 59.9), 64.4 vs real drafts (63.5).
- **Smoke run on dry1022 (24 lobbies, two pairs).** It only checked the chain and is not a comparison point. Its paper-1 MCTS vs GD (56.5 J) sits next to this run's 57.0.

### Other comparisons fixed before the tournament (headline; old-model pass in brackets)

- **Primary,** `aMCTS-cur` vs `pMCTS-cur`: 50.9% (50.5, 51.2) [50.9]; prototype 51.4. Adding the personal term to the leaf of the blind-trained search still does almost nothing. Record +0.3 (−0.7, 1.4).
- **Secondary,** `aMCTS-bc` vs `pMCTS-bc`: 56.3% (55.7, 56.9) [57.6]; prototype 57.6.
- **Distilled prior,** `aMCTS-pp` vs paper 1's MCTS: 58.2% (57.0, 59.3) [60.0], record +5.8 (3.6, 7.9); prototype 60.4.
- **Paper 1's MCTS vs real drafts:** J 53.3 (52.7, 54.0), CONS 63.4. The prototype had J 50.5, and the old-model pass has 50.9. The new state model credits paper 1's MCTS 2.4pp more against real drafts under J, while CONS is unchanged.

### Bradley–Terry (J, logit, GD = 0; headline)

| system | BT J | BT CONS | J standing % | note |
|---|---|---|---|---|
| `x-combo1-dr-drban` | 0.760 (0.733, 0.786) | 0.368 | | design-study variant, one opponent on the headline lobbies |
| `x-combo1-dr` | 0.736 (0.714, 0.756) | 0.387 | 61.0 | |
| `x-pp-ban3` | 0.732 (0.704, 0.760) | 0.376 | | ablation row, one opponent |
| **`x-combo1-dr-chance` (selected)** | **0.726 (0.705, 0.746)** | 0.385 | 60.8 | |
| `x-combo3-split` | 0.724 (0.701, 0.749) | 0.439 | 61.5 | |
| `x-combo3` | 0.723 (0.702, 0.743) | 0.394 | 60.3 | |
| `x-combo1` | 0.720 (0.699, 0.741) | 0.374 | 60.1 | |
| `aMCTS-pp` | 0.663 (0.644, 0.680) | 0.406 | 59.5 | |
| `DraftRec-pv` | 0.543 (0.522, 0.561) | 0.359 | 56.5 | |
| `DraftRec` | 0.428 (0.407, 0.448) | 0.120 | 53.7 | |
| `aMCTS-cur` | 0.361 (0.338, 0.382) | 0.602 | 52.0 | |
| paper 1's MCTS | 0.339 (0.316, 0.361) | 0.602 | 51.4 | |
| Real | 0.196 (0.174, 0.219) | 0.026 | 47.5 | |
| GD | 0 | 0 | 42.4 | |

The seat (first-pick) term is 0.049. The old-model pass orders the top the same way within noise: `x-combo3-split` 0.748, `x-combo1-dr-drban` 0.741, selected 0.729.

**Design study, and what "selected design" means.**
- **The design was fixed before the definitive run.** `x-combo1-dr-chance` was chosen by the selection rule (mean J against `pMCTS-cur` and `aMCTS-pp`) on the tune lobbies in the prototype. The definitive run does not reselect.
- **Rescored on the definitive states, the tune set does not separate the top designs.** The top eight score 55.26 to 55.49 (headline) and 55.98 to 56.65 (old model).
  - Headline: `x-combo1-dr-drban` 55.49, the selected design 55.29.
  - Old model: `x-combo3-split` 56.65, the selected design 55.98.
  - Run on the definitive scores, the same rule would pick `x-combo1-dr-drban` (headline) or `x-combo3-split` (old model), by 0.2 to 0.7pp.
- **The selected design ties the other versions directly.** `x-combo1` scores 49.6 (49.1, 50.2) against it and `x-combo3` 49.9 (49.5, 50.4).
- **`x-combo1-dr-drban` and `x-pp-ban3` do play the headline lobbies (flag 1), but only in the design-ablation schedule:** both seats against `aMCTS-pp`, no other opponent. Their BT values therefore rest on one opponent and are linked to the rest only through `aMCTS-pp`.
  - Against `aMCTS-pp`, `x-combo1-dr-drban` scores 52.4 (52.0, 52.9); the selected design scores 51.6 (51.1, 52.2).
  - There is no direct match between them.
  - The 0.8pp difference sits inside the two intervals and inside the tune-set spread.
- **Wording for the paper:** "the design chosen on the tune lobbies before the definitive run, within noise of the top design-study entries". No claim that it is the best design.

### Split-history judge

A judge built from even days outside the tournament window, scoring drafters that use only odd days of history: λ 0.60 (old model 0.73; prototype 0.71).
- Selected design vs paper 1's MCTS: 63.9% (62.9, 65.0) [old model 67.4].
- `x-combo3-split` vs paper 1's MCTS: 62.9% (62.2, 63.7) [66.6]; prototype 65.7.
- `x-pp-split`: 62.1% (61.2, 63.0).
- Odd-day drafters against their full-history twins: 49.2 to 49.1%.

### Drift (headline; old model in brackets)

- **Prediction** (60,500 tournament-window games, share of the log-loss lift over a coin flip kept):
  - Population WP: 91% at 4 months, 72% at 1 year (prototype 72% at a year).
  - Full value with the player state frozen: 37% at 4 months, 23% at 1 year [46%, 32%]; prototype 43% and 30%. The new state model's fresh lift is larger (0.052 vs 0.044 nats), so freezing it costs a larger share.
- **Population search** (current vs stale twin):
  - Paper 1's MCTS against its 1-year-stale twin: J 53.3 (51.8, 54.8), CONS 52.2.
    - Refreshing the stats tables recovers nothing: 53.2.
    - A 2 GPU-hour fine-tune leaves 52.2 (CONS 52.2).
  - 4 months: J 51.8, CONS 52.2.
  - Prototype: 3.4 (J) and 2.2 (CONS) at a year; fine-tune 1.3 and 1.7.
- **Frozen player state, distilled-prior drafter:**
  - 4 months: 51.5% (51.0, 52.0) for the fresh state.
  - 1 year: 53.7% (53.2, 54.2).
  - CONS 49.5 and 48.7: no population cost.
  - Prototype: 1.7 and 3.8pp.
  - Old model: 51.7, 53.8.
- **Frozen state, leaf-only drafter** (`aMCTS-cur`): 50.2 and 50.4. It barely uses the state.

### Backtests on real games (headline; old model in brackets)

- **Agreement** (20,000 OOT games, 200,000 picks, behavioral-prior search):
  - Teams whose pick matched the aware recommendation won +0.82pp (SE 0.36) with draft-only WP and causal MMR controlled [+0.98].
  - With the skill model's predicted gap added: −0.88 (0.35) [−0.44].
  - Prototype: +0.9 and −0.5.
  - Paper 1's own F-prior search agrees with its blind twin on 96% of picks, so its agreement test has no power: +2.3 (2.8) on the 0.15% of picks where they differ.
- **Forced switches** (1.19M pick slots, 37,752 players): when a player's best hero is banned, realized cost −2.18pp (0.19) against −1.38 predicted (residual −0.80, SE 0.18) [−2.12 vs −1.26]; prototype −2.1 vs −1.3. Forced switches still cost more than the model says.

### Simulations (tune set, mean J against the tune opponents)

| sims | selected | blind |
|---|---|---|
| 100 | 53.2 | 46.0 |
| 200 | 54.8 | 46.1 |
| 400 (as run) | 55.3 | 45.9 |
| 800 | 55.5 | 46.2 |
| 1,600 | 55.2 | 45.9 |

- The selected design gains 2.1pp from 100 to 400 simulations, then is flat. Prototype: +3pp, then flat.
- The blind search is flat at every budget.
- Old model: 53.6 at 100, 56.0 to 56.2 from 400 on.

### Flags

1. **Design-study entries top BT** (`x-combo1-dr-drban`, `x-pp-ban3`); see "Design study" above. They played the same 300 headline lobbies, but against `aMCTS-pp` only, and they are within noise of the selected design. Report it as the design chosen on the tune lobbies before the definitive run, within noise of the top design-study entries, not as the best design.
2. **Judge-free coverage against paper 1's MCTS is low.** 37% of players are covered under both arms. The +4.8 rests on about a third of the players. It is positive, but each arm's realized-record sample is thin there.
3. **The new state model shifts paper 1's MCTS vs real drafts by 2.4pp under J** (53.3 vs 50.9 in the old-model pass), with CONS unchanged. The paper's statement that paper 1's MCTS "gives back" its population edge holds less strongly under the new model.
4. **The first attempt of pass 1's data stage ran out of memory** (job `p3t_data_10270242`, 03:37 on Oct 27). `p3t_vnet.py data` reached the 20 GB per-process cap: peak RSS 17.3 GB, then a 1.5 GB allocation in `TD._walk` failed. The full-history store is larger than dry1022's. The stage was resumed with `P3T_CPU_MEM=30G` and the same code, and passed. Nothing else changed between the attempts. For a later rerun, the 3080 CPU lane needs 30 GB, not 20 GB.
5. **Distilled prior vs paper 1's MCTS** is 1.9pp lower in the headline pass than in the old-model pass (58.2 vs 60.0). The selected design's lead over paper 1's MCTS is lower by the same amount (59.0 vs 61.0). Both passes agree on the ordering.


## Prototype record (2026-10-19 to 10-24; superseded by the definitive results above)

Status: results on current data (store full_1017, CAP_TS 2026-09-28), all from the race-fixed kernels (2026-10-19, see "Kernel fix" below). The full-data rerun after the player-history pull (about Oct 26 to 27) is one command: `training/personalization/remote/p3t_pipeline.sh`. The pre-stated comparisons below were committed before the tournament ran (commits af7427c, 5194ad6, 7ffda4c).

**Where the numbers were computed (2026-10-21).** The 3090 showed single-bit RAM flips. Every head-to-head, Bradley–Terry and matrix number in this file is now computed off the 3090: all drafts that the 3090 had drafted (46 ordered pairs) were redrafted on the 5090, and every draft was rescored and reanalyzed on the 5090 (main, split-history and tune sets; judge refit there too). Cross-check against the 3090's results:
- judge fit (c2, c3, log losses): equal to 1e-8; the split-history judge to 1e-9;
- rescoring the same drafts: all 271 common head-to-heads identical to two decimals, under J, CONS and the judge-free columns;
- redrafting on the 5090: 96 of 10,200 drafts differ (0.9%, near-ties between GPU architectures; same-machine reruns are byte-identical); per-pair J moves by at most 0.11pp, head-to-heads by at most 0.06pp;
- refitting the opponent models (comfort, DraftRec) on the 5090: coefficients equal to 4e-5; the distilled prior refits to the same quality (KL 1.302 vs 1.302, top-1 28.1% vs 28.1%) though not bit-identical (CPU training);
- the real-outcome calibration reruns identically.
Player states (2026-10-22, `p3t_xcheck_states.py`). The drafter state (`state_cur`, 1.84M player-lobby rows) and the judge state (`state_judge`, 605,000 rows) were recomputed in full on the 3080 from the same input files and the code of that date.
- Games, play counts and off-role flags are identical everywhere, as is the judge's raw window record (the judge-free scores).
- The skill values differ for 29 players in the drafter state (3,878 rows, at most 0.002 skill units) and 5 players in the judge state (336 rows, at most 0.013). Every row of those players differs; every other player is identical.
- The 3080 result does not depend on the batch (the 300-lobby and full recomputes agree), the thread count, or the CPU kernels (OpenBLAS and numba forced to generic code give identical output). The most likely cause is the 3090's RAM fault during the state build.
- Effect on the tournament: rescoring every draft with the recomputed judge state moves every head-to-head by at most 0.01pp (no number changes at one decimal). Redrafting the three headline pairs with the recomputed drafter state changes 0 to 13 of 300 drafts per side and moves J by at most 0.12pp (aMCTS-pp vs paper 1's MCTS 60.36 to 60.24; the selected design vs paper 1's MCTS 60.92 to 60.94; the primary 51.4 unchanged). The published numbers stand; the full-data rerun rebuilds every state off the 3090.
Still from the 3090 and not rechecked: the judge-fit inputs, the combiners, the rosters and pre-tournament lobbies, the value nets, the rich priors, the backtests and the layer parts file (`parts_full_ms`); the rerun rebuilds all of these off the 3090. Not rebuilt by the rerun's pipeline (pinned population inputs, manifest role "tournament"; inventory 2026-10-24): paper 1's WP seeds, stats, GD pool, BC prior and F policies, and the stale vintages. Of these, trained on the 3090 (`paper1_revision/site/vint/compute.json`): paper 1's F seeds 1 and 2 (2026-10-04), the s4m and s1y stats, features and WP seeds, the s1y GD and the fine-tune WPs (2026-10-19), and the F runs s4m seed 0, s1y seed 1 and s1y_ft seed 0 (2026-10-19 to 20); the other F runs (paper 1 seeds 0, 3, 4; s4m seed 1; s1y seed 0; s4m_ft seed 0) ran on the 3080 but the stale ones used the 3090-built stats and WP seeds. Integrity check of these, off the 3090 (2026-10-24, coordinator's rule: recompute the training-time metrics and scan the weights; keep if all match; `p3t_integrity_3090.py`, record `results/p3t/integrity_3090.json`). All of them are kept, checked:
- **Stats and features:** rebuilt from the pinned snapshot on the 5090. The s4m and s1y tables (deploy, oof0 to oof4, and their composition tables, 12 files each) are byte-identical to the pinned files.
- **WP seeds:** recomputed on the 5090 on features built from those tables. The validation loss, and the test and NODRIFT accuracy, log loss and slope, match enriched.json to 6 decimals. That covers 3 seeds each for s4m and s1y (s1y seed 42: val loss 0.678765 logged and recomputed). It also covers the fine-tuned WPs s4m_ft seed 123 and s1y_ft seed 777, plus their post-cutoff val and test metrics. The base vintages' selected WPs reproduce their logged "before" metrics as well.
- **s1y GD:** both GD variants (v0 and v1) reproduce their logged post-cutoff next-pick top-1, top-5, pick and ban accuracy exactly (v0 top-1 10.512, v1 10.412). This ran on the 3080.
- **F policies:** each saved policy equals its checkpoint's state, and the checkpoint's best eval equals the log's maximum. Each policy's own-search evaluation (the training loop's EVAL, 512 drafts, 200 sims, 5090) lands within the spread of the 3080- and 5090-trained siblings, against the mean of the last 10 logged evals:
  - paper 1 seeds 1 and 2 (3090): 0.7186 and 0.7209, against logged 0.7244 and 0.7267;
  - their siblings, seeds 0, 3 and 4: 0.7239 to 0.7287;
  - differences from the log: -0.006 to +0.005 for the 3090 runs and -0.012 to +0.004 for the others, with an SE of 0.003 per run.
- **Weight scan:** no weight is non-finite. The largest |w| is 1.21 to 1.35 for the 3090 policies (the others: 1.11 to 1.32), 10.8 to 13.7 for the WPs (paper-1 WPs: 11.4 to 12.3) and 2.3 to 2.4 for the GDs. There is no exponent-flip-sized value anywhere.

No retraining is needed.

**Kernel fix (2026-10-19).** An independent verification of the search kernels (`P3_MCTS_VERIFICATION.md`) found a race in the main draft loop of all five P3 kernels (`cuda_personal`, `cuda_prior`, `cuda_pgd`, `cuda_pvnet`, `cuda_explore`): threads read the shared skip and team flags after thread 0 had rewritten them for the next step. It bites only when steps are skipped (prefix replay, forced bans). It is fixed (local copies behind a barrier, commit 2ad31c1), and every number in this file comes from a rerun on the fixed kernels: every tournament pair, the tune set, the probe, the distilled, rich and self-play priors, the agreement backtests, the paper-1 bench and the timings.
- Checks after the fix: identical reruns are byte-identical (decide-only searches at steps 2, 7 and 13 in all four tournament kernels); full drafts with two forced bans repeat no hero; the parity checks against the population reference pass (`p3t_determinism.py`, `p3t_verify.py`, `p3t_pvnet_verify.py`, `p3t_explore_verify.py`). The verifier reproduced these on its own build.
- Smaller fixes in the same commit: hashed per-episode random seeds (neighbouring seeds were correlated; the opponent draw is now uniform, chi² p 0.42), a floor on the prior before its log so a bias can revive an underflowed hero, the rich prior no longer overwrites the ban bias, and score files record the judge they were scored with so stale scores are recomputed.
- Open, very low severity: in full-draft sampling (self-play and forced-ban runs only, never in a tournament decision) a rounding overrun falls back to hero 0, about once per 10^7 draws (0.03 expected in the whole self-play run). TODO (tournament lane, before the Oct 26 to 27 rerun): switch those samplers to `v2_sample`, which falls back to the last hero with positive mass, then rerun the determinism and parity checks.
- The paper-1 kernel `cuda_mcts` had a separate stream-ordering bug in a build no job used (fixed in b995800); nothing here uses that kernel's engine.

## Primary comparison (stated before the tournament ran)

- **Contrast:** the player-aware search drafter `aMCTS-cur` against its player-blind twin `pMCTS-cur`, head to head on the same real lobbies, each seat played both ways.
  - Same kernel, prior policy seeds, leaf WP, GD opponent model, simulations (400) and kernel seeds.
  - They differ only in the personal terms of the leaf value.
- **Metric:** mean probability that `aMCTS-cur` wins, averaged over both seatings. The judge is the player-aware judge J:
  - consensus of paper 1's composition-corrected judges;
  - plus the personal term from the skill model refit on the other player half, with evidence from the judge window only (2026-07-01 to 09-11, after every lobby).
- **Interval:** 95% CI by lobby-cluster bootstrap, widened for seed-to-seed variation: lobby j gives each side prior seed j mod (its seed count), so with equal seed counts the design is paired (seed i meets seed i) and the seed SE is that of the per-seed-pair means; with unequal counts the cells are crossed and a two-way random-effects decomposition gives the SE (`p3t_analyze.seed_se`). The interval uses the larger of the bootstrap and seed SEs. Every system that reads the vintage's F policies counts its seeds (the exploration and player-feature-prior searches included, fixed 2026-10-24).
- **Reported next to it, not instead of it:**
  - the same contrast under the population-only consensus judge;
  - the judge-free scores from later real games (future hero win rates, synergy, counters; each player's realized record on the assigned hero). They are model-free summaries of the same July to September window the judge's personal coefficients are fit on, not an independent outcome sample. The record is reported on the players covered under both arms (paired coverage), with each arm's coverage;
  - the drafters' own valuations (model-internal).
  - Every number is labeled as model-judged, judge-free (realized later outcomes) or model-internal.

**Secondary comparison** (added before the tournament ran, after a decision probe; see "Search prior" below):
- Contrast: `aMCTS-bc` against `pMCTS-bc`, the same search over the behavioral (BC) prior.
- Metric, judge and interval: as for the primary comparison.

**Exploration design study** (stated before any design-study system played a tournament lobby):
- **Tuning.** Every knob is tuned on a separate set of 300 tune lobbies, never the tournament lobbies. These are tournament-window lobbies that meet the same eligibility rules.
  - Each configuration plays `pMCTS-cur` and `aMCTS-pp` on both seats.
  - The selection score is its mean J win probability over those four matchups.
- **Primary design comparison.** The selected design ("best design") against `aMCTS-pp` on the 300 tournament lobbies, with J and the same interval rule as above.
- **Ablation table.** Each component added to `aMCTS-pp` alone: c_puct, prior mixing weight per pick position, personal bias scale, optimism and pessimism from posterior sd, assignment-aware root bias, opponent personal model inside the search, ban denial bias, and the rich learned prior by training signal.
  - Each row reports judged win rate against `aMCTS-pp` with CI, compute per draft, and training GPU-hours.
- **Split-history judge.** Versions using a drafter state built from odd days only are also scored by the split-history judge (even days).

## Summary

- **How much better is player-aware drafting?** On 300 real lobbies (May 23 to June 30, 2026), the selected player-aware search drafter (`x-combo1-dr-chance`) beats paper 1's player-blind MCTS 60.9% of the time (60.3, 61.6) under the player-aware judge J. It beats the real human drafts of the same lobbies 62.1% (61.5, 62.8), and GD 66.9% (66.3, 67.5).
  - Two other versions of the same design tie with it: `x-combo1` (the comfort opponent instead of DraftRec; 60.8% and 61.8%) and `x-combo3` (1,600 simulations, selected before the kernel fix; 60.7% and 62.6%). Against each other they score 49.5 to 50.5%.
  - The direct match against GD is 54.8% for paper 1's MCTS and 55.2% for real drafts.
  - These are model-judged and conservative. J sees players only through 2.5 months of later games, so it credits about 0.42 of a true personal difference.
  - J_cal, which recalibrates J's two parts on held-out judge-window games (the personal part is already calibrated; the population part is scaled up by 1.17), gives 59.9% (59.2, 60.6) against paper 1's MCTS and 63.5% (62.8, 64.1) against real drafts. Weighting population WP more moves each result toward the side with the population edge: paper 1's MCTS gains a point, real drafts lose 1.4.
  - A judge built from years of disjoint history (split-history, attenuation 0.71) puts `x-combo3`, run on half of each player's history, at 65.7% (65.0, 66.4) against paper 1's MCTS.
  - The judge-free check agrees: the players the selected design puts on heroes later win more on those heroes than the players paper 1's MCTS assigns, +3.7pp per player (1.6, 5.8), and more than the real drafts' players, +2.7pp (1.0, 4.3).
  - Under paper 1's own judges (CONS), the selected design beats real drafts 59.1% and paper 1's MCTS beats them 63.5%, a figure that reproduces paper 1.
- **Adding the personal term to the leaf of a blind-trained search does almost nothing.** The pre-stated primary, `aMCTS-cur` vs `pMCTS-cur`, comes in at 51.4% (50.8, 52.0).
  - Paper 1's F policy is so sharp that the leaf moves 4% of picks.
  - The players have to shape the exploration (the prior), not only the scoring:
    - BC prior with a personal leaf: 57.6% (57.0, 58.2) over its blind twin (the pre-stated secondary);
    - player-feature prior distilled from aware search: 60.4% over paper-1 MCTS;
    - ten-player exploration on top (an opponent model inside the tree, denial bans, assignment-aware root): +1.6 to 1.9pp against that prior on the tournament lobbies (51.9%, CI 51.4 to 52.4, for the selected design).
- **Population vs personal.** Paper 1's blind MCTS gains 13pp of population win probability over real drafts (consensus 63.5%). It gives almost all of that back by putting players on heroes they are bad at: J has it at 50.5% against real drafts.
  - The best player-aware drafters keep a population edge over real drafts (consensus 55 to 60%) and add the personal edge.
- **Drift.**
  - Prediction: a 1-year-stale WP keeps 72% of its log-loss lift. A frozen player state is far costlier: the full value keeps 30% of its lift with player state frozen 1 year, and 43% at 4 months.
  - Drafters: a 1-year-stale paper-1 MCTS loses 2.2pp (CONS) and 3.4pp (J) to its retrained twin. Refreshing its statistics tables recovers nothing; a 2 GPU-hour fine-tune leaves 1.7 and 1.3pp. Freezing the player state costs the distilled-prior drafter 1.7pp (J) at 4 months and 3.8pp at a year, with no population cost, and updating the state is CPU-cheap.
  - The drafters lose a smaller share of their edge than the prediction models lose of their lift at the same staleness.
- **DraftRec**, rebuilt on our data as a reviewer baseline, is strong: 36% top-1 pick accuracy against 9% for GD. Its paper's p+v rule ranks just below the distilled-prior drafters under J (BT 0.56 vs 0.68 for `aMCTS-pp` and 0.74 for the selected design).
- **DraftRec as the opponent model inside the search** doubles the opponent's predicted top-1 in team mode (9.6% to 19.9%) but moves the drafts by at most 0.5pp of J, inside the noise. By the pre-stated rule it still becomes part of the selected design (tune J 56.27 vs 56.07).
- **Backtests on real games:**
  - Teams whose real pick matched the player-aware search's recommendation won 0.9pp more (SE 0.4), with draft-only WP and causal MMR controlled. That is what the skill model predicts: with its predicted gap added, the residual is −0.5 (0.4).
  - When a player's best hero is banned, the realized cost is −2.1pp (0.2) against −1.3pp predicted. Forced switches cost more than the model says.

All numbers below are labeled: **J** (player-aware judge, model-judged, primary), **CONS** (population consensus judges, model-judged), **judge-free** (realized outcomes of later games), **own** (the drafter's own value, model-internal).

## Reconciliation with paper 1 (sanity gate, `p3t_sanity.py`)

**Metric.**
- A head-to-head number is the mean judged probability that A's final draft beats B's in the same lobby, averaged over the 300 lobbies and both seatings. The judge gives P(team 0 wins), symmetrized over team order.
- This is paper 1's tournament metric. It is not a share of lobbies won.
- CONS is paper 1's composition-corrected consensus (gN, gN-naive, RN, QM2026). J adds the personal term.

**Paper 1's benchmark through this tournament's code** (1,000 configs, mid tier, random maps, F seeds pooled vs the GD pool sampled, no pools, no mask, 200 sims, root temperature 1):

| | gN | RN | QM26 | degenerate |
|---|---|---|---|---|
| paper 1 Table (F) | .662 | .641 | .669 | 4.5% |
| reproduced | .655 | .636 | .665 | 5.2% |
| this tournament's settings (400 sims, argmax, mask) | consensus .656 | | | 1.6% |

**On the 300 real lobbies, under paper 1's consensus judges:**
- blind paper-1 MCTS vs GD: 63.0% (gN 64.3, RN 62.7, QM26 62.1);
- vs the real drafts: 63.5% (gN 63.4, RN 61.6, QM26 64.8);
- real drafts vs GD: 50.2%.

Paper 1's headline 0.61 is a mean over a field of ten agents, and paper 1 made no comparison with real drafts.

**Why the margin over real drafts shrinks when players count.**
- Paper 1's MCTS puts players on heroes they are bad at. Its teams' judge skill sum is 11.4pp below the real teams' (14.4pp in the drafter's own state).
- In later real games, the players it assigns realize 3.9pp less per player on those heroes than the real teams' players (judge-free, CI −6.0 to −1.8).
- Under J it ties the real drafts (50.5%). The player-aware drafters keep most of the population edge and add the personal one: 58.7% (CONS) and 61.8% (J) for `x-combo1`, 59.5% and 62.6% for `x-combo3`.

**Checks that this is not an artifact.**
- **J's scale.** On all 60,500 real tournament-window drafts, J predicts the REAL outcomes better than CONS:

  | | log loss | accuracy | AUC | calibration slope |
  |---|---|---|---|---|
  | CONS | 0.6710 | 59.1% | 0.629 | 1.73 |
  | J | 0.6564 | 61.1% | 0.655 | 1.30 |

  A slope above 1 means underconfident. Fitting the two parts separately on the same games (logit P(win) = a logit p_cons + k (c2 dS + c3 dO)): k = 1.02 (SE 0.03), a = 1.73 (SE 0.03). The personal part is calibrated; the underconfidence is all in the population part. J_cal (a = 1.17, k = 1.00, fit on held-out judge-window games) has log loss 0.655 here and is reported next to J in the tables.
- **Pools.** In the tournament lobbies, 2.9% of real picks are heroes new to the player and 0.07% fall outside the team's pool union, whose median is 87 of 90 heroes. The pool rule does not handicap the drafters against real drafts.
- **Same terms.** Real drafts are scored by the same judges with the same best hero-to-player assignment, and their rule-degenerate teams (8.5%) go through the same composition correction.
- **Caveat.** gN and RN were trained on paper 1's post-snapshot no-drift games, which include window games on build 97039. CONS's accuracy on the real window drafts is therefore partly in-sample. This does not touch drafted teams or J's personal term, which uses later games only.

## Results

### Head to head (300 tournament lobbies, both seats, 95% CI)

| matchup | J | CONS | personal part, judge (pp of team skill sum) | judge-free: realized record diff per player (pp) | judge-free: future hero WR diff |
|---|---|---|---|---|---|
| **primary:** aMCTS-cur vs pMCTS-cur | **51.4 (50.8, 52.0)** | 49.9 (49.7, 50.2) | +1.3 (1.0, 1.6) | +0.3 (−1.1, 1.8) | +0.0 |
| **secondary:** aMCTS-bc vs pMCTS-bc | **57.6 (57.0, 58.2)** | 49.5 (49.1, 50.0) | +7.3 (6.8, 7.8) | +2.1 (0.1, 4.2) | +0.1 |
| a1step-cur vs p1step-cur | 55.4 (54.6, 56.1) | 48.8 (48.3, 49.2) | +6.0 (5.4, 6.5) | +1.8 (−0.4, 4.0) | −0.2 |
| aMCTS-pp vs pMCTS-cur | 60.4 (59.7, 61.0) | 44.2 (43.8, 44.6) | +14.3 (13.6, 14.9) | +2.4 (0.3, 4.5) | −1.0 |
| aMCTS-pp vs aMCTS-bc | 56.5 (55.9, 57.1) | 54.1 (53.7, 54.6) | +2.1 (1.7, 2.6) | +1.2 (−0.3, 2.7) | +1.1 |
| aMCTS-sp vs pMCTS-cur | 57.4 (56.0, 58.9) | 49.3 (48.3, 50.2) | +7.3 (6.7, 7.9) | +3.8 (1.9, 5.7) | +0.1 |
| aMCTS-sp vs aMCTS-pp | 44.9 (44.4, 45.5) | 55.1 (54.8, 55.5) | −9.0 | −0.9 | +1.0 |
| aMCTS-vn vs aMCTS-bc | 51.1 (50.6, 51.6) | 49.8 (49.4, 50.1) | +1.2 | +0.6 | −0.0 |
| aPol-pp vs pPol-cur (no search) | 57.7 (57.0, 58.4) | 42.3 (41.7, 42.8) | +13.6 | +4.2 (2.3, 6.0) | −1.1 |
| **x-combo1-dr-chance vs pMCTS-cur (selected design)** | **60.9 (60.3, 61.6)** | 43.8 (43.3, 44.3) | +15.2 | +3.7 (1.6, 5.8) | −1.0 |
| x-combo1-dr-chance vs Real | 62.1 (61.5, 62.8) | 59.1 (58.6, 59.6) | +3.0 | +2.7 (1.0, 4.3) | +1.5 |
| x-combo1 vs pMCTS-cur | 60.8 (60.2, 61.5) | 43.8 (43.1, 44.4) | +15.2 | +4.7 (2.7, 6.6) | −1.0 |
| x-combo1 vs Real | 61.8 (61.1, 62.5) | 58.7 (58.2, 59.2) | +3.0 | +0.9 (−0.6, 2.4) | +1.5 |
| x-combo3 vs pMCTS-cur | 60.7 (60.1, 61.3) | 44.3 (43.8, 44.8) | +14.6 | +4.4 (2.4, 6.5) | −1.1 |
| x-combo3 vs Real | 62.6 (62.0, 63.2) | 59.5 (59.0, 60.0) | +3.2 | +1.3 (−0.3, 2.9) | +1.5 |
| pMCTS-cur vs pPol-cur (paper 1, search vs none) | 49.7 (49.3, 50.2) | 49.6 (49.4, 49.8) | +0.2 | −1.6 | −0.1 |
| pMCTS-cur vs Real | 50.5 (49.6, 51.3) | 63.5 (62.9, 64.0) | −11.4 | −3.9 (−6.0, −1.8) | +2.4 |
| aMCTS-pp vs Real | 59.8 (59.1, 60.5) | 59.1 (58.6, 59.6) | +0.8 | +0.4 | +1.6 |
| a1step-cur vs Real | 56.3 (55.5, 57.0) | 55.6 (55.1, 56.1) | +0.7 | +0.1 | +0.8 |
| aMCTS-bc vs Real | 55.0 (54.2, 55.7) | 55.3 (54.8, 55.8) | −0.1 | −1.3 | +0.5 |
| Real vs GD | 55.2 (54.4, 56.0) | 50.2 (49.7, 50.7) | +4.7 | +1.9 (0.2, 3.6) | +0.4 |
| DraftRec vs Real | 54.9 (54.2, 55.5) | 51.8 (51.3, 52.2) | +2.8 | +1.2 | +0.4 |
| DraftRec vs a1step-cur | 50.4 (49.7, 51.0) | 47.4 (47.0, 47.9) | +2.5 | +0.2 | −0.3 |

- The personal part is c2 × the judge's personal-sum difference. The judge-free record is each assigned player's mean residual (y − WP) on that hero in his later games, averaged over the covered players; hero-to-player assignment is by earlier play counts, which is outcome-free.
- CIs come from a lobby bootstrap, widened for prior-seed effects with the drift paper's crossed random effects, taking the larger SE. The seed SE is 0.0 to 1.0pp; the paired-seed designs are at the low end.

### Bradley–Terry strengths and safety (all systems, all pairs)

| system | BT under J (logit, GD = 0) | BT under CONS | degenerate % | off-role players per team | record coverage % |
|---|---|---|---|---|---|
| x-combo1-dr | 0.75 (0.72, 0.77) | 0.38 (0.37, 0.40) | 0.0 | 0.60 | 77 |
| x-combo1-dr-chance (selected) | 0.74 (0.72, 0.76) | 0.38 (0.37, 0.40) | 0.0 | 0.60 | 77 |
| x-combo3 | 0.74 (0.72, 0.76) | 0.39 (0.38, 0.41) | 0.0 | 0.66 | 76 |
| x-combo1 | 0.74 (0.71, 0.76) | 0.37 (0.36, 0.39) | 0.0 | 0.60 | 77 |
| x-pp-ban3 | 0.72 (0.69, 0.74) | 0.36 (0.34, 0.38) | 0.0 | 0.59 | 77 |
| x-pp-dr | 0.69 (0.67, 0.71) | 0.41 (0.40, 0.43) | 0.0 | 0.66 | 75 |
| aMCTS-pp | 0.68 (0.66, 0.69) | 0.40 (0.38, 0.41) | 0.0 | 0.62 | 76 |
| aPol-pp | 0.59 (0.57, 0.61) | 0.32 (0.30, 0.34) | 0.0 | 0.60 | 75 |
| DraftRec-pv | 0.56 (0.55, 0.58) | 0.37 (0.35, 0.38) | 0.4 | 0.80 | 72 |
| DraftRec | 0.49 (0.47, 0.52) | 0.14 (0.13, 0.16) | 0.4 | 0.53 | 85 |
| a1step-cur | 0.49 (0.47, 0.51) | 0.27 (0.26, 0.27) | 0.0 | 0.79 | 71 |
| aMCTS-sp | 0.48 (0.47, 0.51) | 0.58 (0.57, 0.60) | 0.0 | 1.16 | 57 |
| aMCTS-vn | 0.47 (0.45, 0.49) | 0.23 (0.22, 0.24) | 0.0 | 0.76 | 72 |
| aMCTS-bc | 0.43 (0.42, 0.45) | 0.23 (0.22, 0.24) | 0.0 | 0.82 | 70 |
| aMCTS-cur | 0.31 (0.29, 0.33) | 0.61 (0.60, 0.62) | 0.0 | 1.48 | 48 |
| pPol-cur | 0.28 (0.26, 0.31) | 0.61 (0.60, 0.63) | 0.0 | 1.44 | 47 |
| pMCTS-cur | 0.27 (0.25, 0.29) | 0.61 (0.59, 0.62) | 0.0 | 1.52 | 46 |
| p1step-cur | 0.26 (0.24, 0.27) | 0.32 (0.32, 0.33) | 0.0 | 1.15 | 56 |
| Real | 0.25 (0.22, 0.27) | 0.03 (0.02, 0.05) | 8.4 | 0.82 | 77 |
| pMCTS-bc | 0.15 (0.14, 0.16) | 0.26 (0.25, 0.26) | 0.0 | 1.13 | 55 |
| GD | 0.00 (0.00, 0.00) | 0.00 (0.00, 0.00) | 2.5 | 0.97 | 61 |

All 64 systems, with standings: `P3_TOURNAMENT_TABLES.md`. Ablation arms that played only `aMCTS-pp` (such as `x-combo1-dr-drban`, BT 0.77) have wide, schedule-dependent strengths and are left out here.

- Seat (first pick) term: about +0.02 logit; the seat does not matter here.
- Every search drafter fields 0% degenerate teams (structural mask). `Real` is degenerate by paper 1's rule in 8.4% of its tournament drafts (8.5% of the lobbies' own drafts; replaying against other drafters forces some GD fallbacks), GD 2.5%, DraftRec 0.4%.
- Off-role players per team: 1.5 for paper 1's blind searches, 0.8 for real teams, 0.55 to 0.62 for the distilled-prior drafters.


### Direct matches against GD (no Bradley–Terry conversion)

Every main system played GD on the 300 lobbies, both seats.

| system | J vs GD | J_cal vs GD | CONS vs GD | judge-free: realized record diff per player (pp) | judge-free: future hero WR diff |
|---|---|---|---|---|---|
| x-combo1-dr-chance (selected) | 66.9 (66.3, 67.5) | 68.1 (67.5, 68.8) | 59.0 (58.6, 59.5) | +4.2 (2.2, 6.2) | +2.1 |
| x-combo1 | 66.9 (66.3, 67.5) | 68.2 (67.5, 68.8) | 58.8 (58.4, 59.3) | +3.1 (1.1, 5.1) | +2.1 |
| x-combo3 | 67.5 (66.9, 68.2) | 68.9 (68.2, 69.5) | 59.7 (59.2, 60.1) | +2.7 (0.6, 4.7) | +2.1 |
| x-combo1-dr | 67.2 (66.6, 67.7) | 68.4 (67.8, 69.0) | 58.9 (58.4, 59.4) | +4.1 (2.2, 6.0) | +2.1 |
| x-pp-dr | 65.7 (65.1, 66.4) | 67.1 (66.4, 67.8) | 59.4 (59.0, 59.9) | +1.6 (−0.3, 3.5) | +2.1 |
| aMCTS-pp | 65.9 (65.3, 66.6) | 67.3 (66.6, 67.9) | 59.4 (58.9, 59.9) | +3.6 (1.7, 5.5) | +2.2 |
| aMCTS-cur | 55.7 (53.8, 57.6) | 57.8 (55.8, 59.8) | 63.1 (61.9, 64.3) | −1.4 (−3.7, 1.0) | +3.0 |
| pMCTS-cur (paper 1's MCTS) | 54.8 (52.9, 56.6) | 56.9 (54.9, 58.9) | 63.0 (61.7, 64.2) | −1.9 (−4.3, 0.5) | +2.9 |
| a1step-cur | 63.6 (63.0, 64.3) | 64.8 (64.1, 65.5) | 58.0 (57.6, 58.5) | +0.9 (−1.3, 3.1) | +1.7 |
| p1step-cur | 56.6 (55.9, 57.4) | 58.0 (57.2, 58.8) | 58.6 (58.1, 59.0) | −0.5 (−2.8, 1.8) | +1.8 |
| DraftRec | 63.7 (63.1, 64.4) | 64.5 (63.8, 65.2) | 55.1 (54.6, 55.6) | +1.7 (0.0, 3.4) | +1.5 |
| DraftRec-pv | 64.4 (63.7, 65.0) | 65.8 (65.2, 66.5) | 60.0 (59.6, 60.4) | +0.3 (−1.5, 2.0) | +2.5 |
| Real | 55.2 (54.4, 56.0) | 55.2 (54.4, 56.0) | 50.2 (49.7, 50.7) | +1.9 (0.2, 3.6) | +0.4 |

- Under J, the distilled-prior drafters beat GD by 16 to 17.5pp; paper 1's MCTS beats it by 4.8pp, and the real human drafts by 5.2pp.
- Under CONS the order flips at the top: paper 1's MCTS beats GD by 13pp and the player-aware designs by 9 to 10pp. Real drafts tie GD (50.2%), as paper 1 found.

### The head-to-head matrix

Every ordered pair among the 14 main systems was played (`P3_TOURNAMENT_TABLES.md` has the full 14 × 14 matrices under J and CONS with 95% CIs). Ten of them, row's win % against column, both seats:

Under J:

| row vs column | sel | xc1 | xc3 | pp | aMC | pMC | a1s | DRpv | Real | GD |
|---|---|---|---|---|---|---|---|---|---|---|
| sel (x-combo1-dr-chance) |  | 50.5 | 49.9 | 51.9 | 60.4 | 60.9 | 56.7 | 55.4 | 62.1 | 66.9 |
| xc1 (x-combo1) | 49.5 |  | 50.0 | 51.6 | 60.5 | 60.8 | 56.5 | 54.5 | 61.8 | 66.9 |
| xc3 (x-combo3) | 50.1 | 50.0 |  | 51.6 | 60.2 | 60.7 | 56.9 | 55.3 | 62.6 | 67.5 |
| pp (aMCTS-pp) | 48.1 | 48.4 | 48.4 |  | 59.3 | 60.4 | 54.2 | 53.1 | 59.8 | 65.9 |
| aMC (aMCTS-cur) | 39.6 | 39.5 | 39.8 | 40.7 |  | 51.4 | 45.8 | 43.3 | 51.1 | 55.7 |
| pMC (pMCTS-cur) | 39.1 | 39.2 | 39.3 | 39.6 | 48.6 |  | 45.1 | 42.6 | 50.5 | 54.8 |
| a1s (a1step-cur) | 43.3 | 43.5 | 43.1 | 45.8 | 54.2 | 54.9 |  | 48.1 | 56.3 | 63.6 |
| DRpv (DraftRec-pv) | 44.6 | 45.5 | 44.7 | 46.9 | 56.7 | 57.4 | 51.9 |  | 57.9 | 64.4 |
| Real (Real) | 37.9 | 38.2 | 37.4 | 40.2 | 48.9 | 49.5 | 43.7 | 42.1 |  | 55.2 |
| GD (GD) | 33.1 | 33.1 | 32.5 | 34.1 | 44.3 | 45.2 | 36.4 | 35.6 | 44.8 |  |

Under CONS:

| row vs column | sel | xc1 | xc3 | pp | aMC | pMC | a1s | DRpv | Real | GD |
|---|---|---|---|---|---|---|---|---|---|---|
| sel (x-combo1-dr-chance) |  | 50.3 | 49.7 | 49.0 | 43.8 | 43.8 | 53.4 | 51.1 | 59.1 | 59.0 |
| xc1 (x-combo1) | 49.7 |  | 49.7 | 49.1 | 43.9 | 43.8 | 53.2 | 50.5 | 58.7 | 58.8 |
| xc3 (x-combo3) | 50.3 | 50.3 |  | 49.4 | 44.4 | 44.3 | 54.0 | 51.3 | 59.5 | 59.7 |
| pp (aMCTS-pp) | 51.0 | 50.9 | 50.6 |  | 44.1 | 44.2 | 53.3 | 51.5 | 59.1 | 59.4 |
| aMC (aMCTS-cur) | 56.2 | 56.1 | 55.6 | 55.9 |  | 49.9 | 59.0 | 55.9 | 63.5 | 63.1 |
| pMC (pMCTS-cur) | 56.2 | 56.2 | 55.7 | 55.8 | 50.1 |  | 58.5 | 55.8 | 63.5 | 63.0 |
| a1s (a1step-cur) | 46.6 | 46.8 | 46.0 | 46.7 | 41.0 | 41.5 |  | 47.4 | 55.6 | 58.0 |
| DRpv (DraftRec-pv) | 48.9 | 49.5 | 48.7 | 48.5 | 44.1 | 44.2 | 52.6 |  | 58.5 | 60.0 |
| Real (Real) | 40.9 | 41.3 | 40.5 | 40.9 | 36.5 | 36.5 | 44.4 | 41.5 |  | 50.2 |
| GD (GD) | 41.0 | 41.2 | 40.3 | 40.6 | 36.9 | 37.0 | 42.0 | 40.0 | 49.8 |  |

Under J_cal (sensitivity: the consensus part rescaled by 1.17, the personal part unchanged):

| row vs column | sel | xc1 | xc3 | pp | aMC | pMC | a1s | DRpv | Real | GD |
|---|---|---|---|---|---|---|---|---|---|---|
| sel (x-combo1-dr-chance) |  | 50.6 | 49.9 | 51.7 | 59.4 | 59.9 | 57.3 | 55.5 | 63.5 | 68.1 |
| xc1 (x-combo1) | 49.4 |  | 50.0 | 51.4 | 59.5 | 59.8 | 57.0 | 54.5 | 63.1 | 68.2 |
| xc3 (x-combo3) | 50.1 | 50.0 |  | 51.5 | 59.2 | 59.8 | 57.5 | 55.5 | 64.0 | 68.9 |
| pp (aMCTS-pp) | 48.3 | 48.6 | 48.5 |  | 58.4 | 59.4 | 54.7 | 53.3 | 61.2 | 67.3 |
| aMC (aMCTS-cur) | 40.6 | 40.5 | 40.8 | 41.6 |  | 51.4 | 47.2 | 44.3 | 53.3 | 57.8 |
| pMC (pMCTS-cur) | 40.1 | 40.2 | 40.2 | 40.6 | 48.6 |  | 46.5 | 43.6 | 52.7 | 56.9 |
| a1s (a1step-cur) | 42.7 | 43.0 | 42.5 | 45.3 | 52.8 | 53.5 |  | 47.7 | 57.1 | 64.8 |
| DRpv (DraftRec-pv) | 44.5 | 45.5 | 44.5 | 46.7 | 55.7 | 56.4 | 52.3 |  | 59.2 | 65.8 |
| Real (Real) | 36.5 | 36.9 | 36.0 | 38.8 | 46.7 | 47.3 | 42.9 | 40.8 |  | 55.2 |
| GD (GD) | 31.9 | 31.9 | 31.1 | 32.7 | 42.2 | 43.1 | 35.2 | 34.2 | 44.8 |  |

- The matrix is consistent with the Bradley–Terry order: under J the three versions of the selected design tie with each other (49.5 to 50.5) and beat every other system; `aMCTS-pp` is next, then DraftRec-pv and the aware one-step drafter; paper 1's MCTS, real drafts and GD are last.
- No three systems form a cycle in which each beats the next with its CI above 50%, under either judge.

### Why the judged gain is below the drafter's own estimate (diagnosis)

The one-step drafter's own value said +7.2pp vs GD (the swag). Judged, the aware-vs-blind gains are 5 to 8pp, with 14pp of personal sum against paper 1's MCTS. Max expected more, so before reporting we checked four explanations.

1. **The judge's attenuation, which is the main reason.**
   - On real teams, the judge's personal sum moves 0.42 per unit of the drafter's (correlation 0.54). The judge sees 2.5 months of later games; the drafter sees the player's whole history.
   - A calibrated judge with less information credits only about that fraction of a true personal difference (paper 1 found the same attenuation for its population judges, 0.59).
   - The aware drafters' judged personal gains over their blind counterparts are 0.54 to 0.66 of their own estimates. That is above 0.42, so there is no sign of overfitting to the personal term.
   - Corrected for attenuation (a model-based sensitivity, labeled "J corrected" in the tables):
     - aMCTS-bc vs pMCTS-bc: 65.8%;
     - aMCTS-pp vs pMCTS-cur: 75.3%.
2. **A less attenuated judge.**
   - The split-history judge uses even-day evidence, years deep, outside the tournament window, and the other player half's fit. The drafters for this check use odd-day evidence only.
   - Its attenuation slope is 0.71.
   - Judged this way, aMCTS-bc@split vs pMCTS-bc is 59.5% (58.8, 60.2) and a1step@split vs p1step-cur 57.7% (56.9, 58.5), against 57.8% and 56.5% under the window judge. The drafter here has only half of each player's history.
3. **Judge-free.**
   - Each player's realized later record on the hero he is assigned is noisy but unshrunk. Against paper 1's blind MCTS it favors the aware drafters whose prior or candidates carry the players by 2.4 to 4.7pp per player, and each CI excludes zero; the value-net arm is +1.6 (−0.7, 3.9).
   - For the one-step aware drafter against paper 1's MCTS it is about 70% of the drafter's own estimate (+3.0 against +4.2pp per player).
4. **Pools are not the limit.** A team's pool union holds a median of 87 of 90 heroes (10th percentile 76), and 99.9% of real picks are in it.

What does cost the aware drafters is population win probability.
- Against paper 1's blind MCTS, CONS says 41 to 44% for every strongly personal drafter.
- The selected design trades 6pp of population WP for 15pp of personal sum.
- The aware drafters that keep CONS near 50% (`aMCTS-sp`, per-position mixing) give up part of the personal edge.

### Putting the players into the search (paper section)

**Options (a) to (d).** "vs" is the blind twin with the same prior family.

| option | system | J vs paper-1 MCTS | J vs its blind twin | CONS vs paper-1 MCTS | GPU ms per draft (5090, 8 decisions, 400 sims) | training for the player part |
|---|---|---|---|---|---|---|
| (a) leaf only, blind-trained prior | aMCTS-cur | 51.4 | 51.4 | 49.9 | 35.8 (blind 34.2) | none |
| (a') leaf only, behavioral prior | aMCTS-bc | 53.8 | 57.6 | 41.0 | 31.1 (blind 30.1) | none |
| (b) player-feature prior (distilled) | aMCTS-pp | **60.4** | | 44.2 | 40.6 | 0.4 GPU-h targets + 1 CPU-min fit |
| (c) value net with player features | aMCTS-vn | 54.5 | 58.8 (vs pMCTS-bc) | 41.0 | 31.2 | CPU minutes |
| (d) self-play with real rosters | aMCTS-sp | 57.4 | | 49.3 | 38.2 | 1.1 GPU-h (4 iterations × 4,096 rosters, 800 sims) |
| (b) + ten-player exploration, DraftRec opponent at chance nodes (selected design) | x-combo1-dr-chance | **60.9** | | 43.8 | 41 + about 30 ms of host DraftRec inference | as (b), plus the DraftRec opponent fit (GPU seconds) |
| the same with the comfort opponent | x-combo1 | 60.8 | | 43.8 | 41.3 | as (b), plus the opponent model fit (CPU) |
| the same at 1,600 sims | x-combo3 | 60.7 | | 44.3 | 148 | as (b) |

- Computing the ten players' states costs about 0.1 ms per player-day (one walk for all players of a lobby).
- The F policy itself took 8.3 GPU-hours per seed (3080).

**Design study.** Tune lobbies first (300, never the tournament lobbies; selection score = mean J against pMCTS-cur and aMCTS-pp). The design is the distilled player-feature prior + denial bans (λ 3) + opponent comfort model inside the tree (×2) + assignment-aware root bias (×1).
- Before the kernel fix the rule picked it at 1,600 sims (`x-combo3`). On the rerun tune lobbies the rule picks the same design at 400 sims (`x-combo1`, tune J 56.07 against 55.84 for `x-combo3`). With the DraftRec opponent variants added on 2026-10-20 (rule stated before they played), `x-combo1-dr-chance` scores 56.27 and becomes the selected design. All three gaps are inside the noise, and on the tournament lobbies the three versions tie (49.5 to 50.5% against each other).
- A split-history version (`x-combo3-split`, drafter state from odd days) also played the tune lobbies (56.39); it is a robustness arm for the split judge, not a candidate.

**Pre-stated design comparison:** the selected design against `aMCTS-pp` on the tournament lobbies is **51.9% under J (51.4, 52.4)** for `x-combo1-dr-chance`, with CONS 49.0 (48.5, 49.4). `x-combo1`: 51.6% (51.0, 52.1), CONS 49.1; `x-combo3`: 51.6% (51.1, 52.1), CONS 49.4.

Each component added alone to `aMCTS-pp`, against `aMCTS-pp`:

| design | J vs aMCTS-pp (CI) | CONS vs aMCTS-pp (CI) | tune-set J | GPU ms per draft |
|---|---|---|---|---|
| x-combo1-dr-chance (selected) | 51.9 (51.4, 52.4) | 49.0 (48.5, 49.4) | 56.3 | about 70 |
| x-combo1 (comfort opponent) | 51.6 (51.0, 52.1) | 49.1 (48.7, 49.5) | 56.1 | 41 |
| x-combo3 (same, 1,600 sims) | 51.6 (51.1, 52.1) | 49.4 (49.0, 49.8) | 55.8 | 148 |
| denial bans (λ 3) | 51.0 (50.4, 51.5) | 49.1 (48.7, 49.5) | 55.5 | 41 |
| opponent comfort model (×2) | 50.4 (50.0, 50.8) | 50.2 (49.8, 50.5) | 54.9 | 41 |
| assignment-aware root bias | 49.9 (49.5, 50.3) | 49.4 (49.1, 49.8) | 54.5 | 41 |
| value net v2 at the leaf | 50.4 (50.0, 50.8) | 49.7 (49.4, 50.1) | 54.9 | about 41 |
| 1,600 sims | 50.0 (49.5, 50.5) | 50.1 (49.8, 50.5) | 55.0 | about 145 (not timed) |
| c_puct 4 | 49.6 (49.2, 50.1) | 49.8 (49.5, 50.2) | 54.4 | |
| c_puct 1 | 48.5 (48.0, 49.0) | 49.5 (49.0, 49.8) | 52.6 | |
| per-position mixing (F weight 0.5 early) | 49.0 (48.4, 49.5) | 52.1 (51.7, 52.5) | 54.7 | |
| personal bias ×0.5 | 49.1 (48.7, 49.6) | 52.0 (51.6, 52.3) | 54.1 | |
| personal bias ×2 | 49.3 (48.9, 49.7) | 46.6 (46.3, 47.0) | 53.4 | |
| α 0.3 (more F) | 49.0 (48.5, 49.5) | 53.1 (52.7, 53.5) | 54.4 | |
| α 0 (no F prior) | 48.6 (48.1, 49.1) | 46.0 (45.6, 46.5) | 52.5 | |
| optimism 0.5 sd | 49.4 (49.0, 49.8) | 50.1 (49.7, 50.4) | 54.4 | |
| optimism 1 sd | 49.1 (48.7, 49.6) | 49.8 (49.4, 50.1) | 53.4 | |
| rich prior, search-distilled | 47.3 (46.8, 47.8) | 47.8 (47.4, 48.2) | 52.6 | |
| rich prior, outcome-weighted BC | 48.9 (48.4, 49.5) | 46.8 (46.4, 47.3) | 52.9 | |
| rich prior, one-step-distilled | 48.7 (48.1, 49.2) | 48.0 (47.6, 48.5) | 53.2 | |
| rich prior alone, no search (best: one-step) | 47.2 (46.6, 47.8) | 46.1 (45.7, 46.6) | 51.4 | |
| (reference) `x-pp`, the distilled prior in `cuda_explore` with every mode off | 50.4 (50.0, 50.9) | 50.1 (49.8, 50.5) | 55.1 | 40 |

- `x-pp` is `aMCTS-pp`'s design run through the design-study code path; it differs only in its kernel seeds (seeds are per pair, lobby and step). Its distance from 50% is the run-to-run noise of the search itself (measured directly from repeated runs: SD up to 0.35pp per head-to-head), which the lobby-bootstrap CIs do not include. Rows within about 0.5pp of `x-pp` are ties.


**Reading.**
- The distilled player-feature prior is close to a local optimum.
- Denial bans are the one single component that helps on both lobby sets: +1.0pp on the tournament lobbies, with the CI excluding 50, and +0.4 over `x-pp` on the tune lobbies.
- The ten-player combination adds +1.6pp, at a cost of about 0.9pp of population WP.
- Knobs that shift weight between the population prior and the personal prior trade along a frontier:
  - per-position mixing, smaller personal bias and α 0.3 gain 2 to 3pp of CONS and lose about 1pp of J;
  - α 0 and bias ×2 lose on both (CONS 3.4 to 4pp, J 0.7 to 1.4pp).
- The rich learned priors (all ten players' skill vectors, assignment gains, comfort, roles and step) predict their targets well. The search-distilled one matches 38% of target argmaxes against 20% for the F prior. Yet each drafts worse than the simple distilled prior, alone or inside the search, under both judges.
- So capacity is not the bottleneck. What matters is a prior that keeps paper 1's population knowledge (α about 0.16 on log p_F) and adds a personal term.
- Training signals for the rich prior, best to worst: one-step distillation ≈ outcome-weighted BC > search distillation.

**Value side.** On 112K pre-tournament games (paper 2's causal WP as offset, each team assigned by the drafter's own rule; last 15% of days held out), a richer value net predicts outcomes better than the linear personal sum:

| value model | gain over linear, x1000 log loss (SE) |
|---|---|
| the kernel's value net (assigned roster, 12 features) | +5.9 (0.7) |
| per-player set net | +7.7 (0.9) |
| plus the full 90-hero skill vectors | +10.7 (1.2) |

- The first value net, fit on 42,640 games the paper-1 WP never trained on, gained +0.8 (0.6) and moved the tournament by +1.1pp (`aMCTS-vn` vs `aMCTS-bc`, 51.1%).
- Refit on the 112K games (`x-pp-vn2`), it moves J by −0.2pp on the tune lobbies and 0.0 on the tournament lobbies (against `x-pp`).
- So a better leaf predicts better but does not draft better. In this search the prior and the exploration decide what gets evaluated, and the linear term is close to sufficient for choosing among those candidates.

### DraftRec baseline

- Rebuilt from the paper and its released code (player transformer over the last 50 games with scoreboard stats; draft transformer; pick and win heads, λ 0.1). Trained on 862,604 drafts through 2026-05-22 (1.2 GPU-hours).
- OOT accuracy: pick top-1 36.0%, top-5 68.5%; post-draft win AUC 0.654.
- Tournament results:
  - BT under J: the pick head 0.50 and the paper's p+v rule 0.56, against 0.68 for `aMCTS-pp`, 0.74 for the selected design, 0.49 for `a1step-cur` and 0.27 for paper-1 MCTS.
  - Head to head: DraftRec vs real drafts 54.9% (J), vs a1step 50.4%.
  - Under CONS the pick head is weak (BT 0.14): it imitates comfortable picks.

### DraftRec as the opponent inside the search (added 2026-10-20)

The search's opponent (chance nodes and rollouts) was GD, or GD times the opponents' comfort. DraftRec predicts real picks far better, so a distilled version of its pick head became a third opponent model (`p3t_oppdr.py`).
- **How it enters the kernel.** A transformer cannot run inside the CUDA search, so DraftRec runs once on the host at the decision state. Each later opponent step gets one static bias row in the kernel's opponent tensor:
  - opponent pick: softmax(w0 log p_GD(in-tree state) + lp log q), q = DraftRec's mixture over the opponent's not-yet-assigned players for that pick turn, each read from his own history;
  - opponent ban: the same with lb log r, r = DraftRec's mixture for our team's next pick (bans aimed at what we would pick).
  - Fitted on 3,000 tournament-window lobbies outside the tournament, probe and tune sets (real actions only, no outcomes, nothing from the judge window): w0 0.52, lp 0.79, lb 0.35.
  - `cfg[6] = 3` (new, kernel gate) keeps the model at chance nodes and gives rollouts plain GD. Determinism and parity gates pass with it (byte-identical reruns; identity rows reproduce the existing path; real rows change the search).
- **Prediction quality** on 300 held-out fit lobbies (team mode: who picks is unknown, the tournament's rule), next real action:

| opponent model | picks top-1 | picks NLL | bans top-1 | bans NLL |
|---|---|---|---|---|
| GD | 9.6% | 3.72 | 16.7% | 3.46 |
| GD × comfort (the opponent of `x-combo1`, fitted scale) | 12.5% | 3.51 | (GD) | (GD) |
| DraftRec opponent, as used in the kernel (bias read at the root) | **19.9%** | **3.18** | 16.8% | 3.42 |
| DraftRec mixture alone, read at the root | 20.2% | 3.20 | 8.5% | 3.98 |
| DraftRec mixture alone, exact state | 22.3% | 3.08 | | |
| GD + DraftRec, exact state (not usable in the kernel) | 22.3% | 3.07 | | |

  - DraftRec's 36% top-1 is in player mode (it knows which player picks). In team mode it is 21 to 22%. Reading it once at the decision state costs about 2 points of top-1.
  - DraftRec says little about bans: the ban model is barely better than GD.
- **Variants** (stated before they played, `p3t_explore.json`): `x-pp-dr` (`aMCTS-pp`'s design with the DraftRec opponent), `x-combo1-dr` (in place of the comfort opponent), each with a chance-node-only version (`-chance`), and `x-combo1-dr-drban` (denial bans aimed by DraftRec in place of the skill-based ones).

**Results** (J and CONS on the 300 tournament lobbies, 95% CI; tune J = the selection score on the tune lobbies):

| design | J vs aMCTS-pp | CONS vs aMCTS-pp | J vs its comfort-opponent twin | tune J | GPU+CPU s per draft |
|---|---|---|---|---|---|
| `x-pp-dr` (DraftRec opponent, chance nodes and rollouts) | 50.1 (49.7, 50.5) | 50.0 (49.7, 50.4) | vs `aMCTS-pp`: 50.1 | 55.08 | 0.11 |
| `x-pp-dr-chance` (chance nodes only) | 50.7 (50.3, 51.1) | 50.4 (50.0, 50.7) | | 55.09 | 0.10 |
| `x-combo1-dr` | 51.8 (51.3, 52.3) | 49.0 (48.6, 49.3) | vs `x-combo1`: 50.0 (49.5, 50.5) | 56.12 | 0.10 |
| `x-combo1-dr-chance` (**selected**) | 51.9 (51.4, 52.4) | 49.0 (48.5, 49.4) | vs `x-combo1`: 50.5 (50.0, 51.0) | **56.27** | 0.11 |
| `x-combo1-dr-drban` (denial bans aimed by DraftRec) | 52.3 (51.8, 52.8) | 49.4 (49.1, 49.8) | | 56.25 | 0.11 |
| `x-combo1` (comfort opponent, for reference) | 51.6 (51.0, 52.1) | 49.1 (48.7, 49.5) | | 56.07 | 0.08 |

- **A much better opponent model barely changes the drafts.** DraftRec doubles the opponent's predicted top-1 (9.6% to 19.9%), yet swapping it in moves J by 0.0 to 0.5pp against the same design with the old opponent, inside the search's own run-to-run noise (about 0.4pp). Three reasons fit the data:
  - the bias is read once at the root, so deeper chance nodes see a stale prediction;
  - the decide-only search spends 400 simulations over a few dozen of our candidates, and the opponent's reply mostly shifts all of them alike;
  - the judges score the final teams, and a better guess of the opponent's next pick rarely changes which hero is best for us.
- Chance nodes only and DraftRec-aimed denial bans are each a few tenths of a point better than the full version, again within noise.
- **Selection.** By the pre-stated rule, `x-combo1-dr-chance` (tune J 56.27) replaces `x-combo1` (56.07) as the selected design. The margin is 0.2pp, smaller than the noise; the honest reading is that the three versions of this design tie, and DraftRec in the chance nodes costs about 30 ms of host inference per draft.

### Drift

Each row is the current twin against the stale version, both seats, in pp above 50 (the stale drafter's loss), 95% CI with seed effects. "Edge lost" is the BT gap as a share of the current drafter's BT edge over GD. Stale vintages have 2 F seeds (fine-tune: 1), so the J intervals at 1 year are wide; the crossed seed effect dominates them.

**Population search (`pMCTS`), the upkeep arms.**

| arm | 4 months: CONS | 4 months: J | 1 year: CONS | 1 year: J | upkeep cost |
|---|---|---|---|---|---|
| (a) stale weights and statistics | 2.0 (0.9, 3.1) | 1.2 (0.1, 2.2) | 2.2 (1.6, 2.8) | 3.4 (−0.3, 7.0) | none |
| (b) stale weights, refreshed statistics (`-rs`) | 2.0 (0.8, 3.1) | 1.2 (−0.1, 2.4) | 2.2 (1.5, 2.9) | 3.6 (−0.1, 7.4) | CPU minutes |
| (c) fine-tune (`-ft`, seed 0) | 1.4 (1.0, 1.9) | 1.1 (−0.4, 2.7) | 1.7 (1.0, 2.4) | 1.3 (0.4, 2.2) | 1.6 and 2.1 GPU-h, plus under 1 GPU-minute for the WP |
| (d) full retrain (`pMCTS-cur`) | 0 | 0 | 0 | 0 | 8.3 GPU-h per seed (3080), plus 0.3 h of statistics and WP |

- Edge lost under CONS: 11% at 4 months and 12% at a year for (a); 6% and 11% for (c).
- Refreshing the statistics tables in the kernel changes nothing (`-rs` vs stale: 50.0% and 50.1%). The stale WP and policy, not the tables, carry the loss.
- The fine-tune recovers about a third of the population loss at 4 months and a quarter at a year (CONS), and most of the J loss at a year.

**Player-aware drafters.**

| drafter | what is stale | 4 months: CONS | 4 months: J | 1 year: CONS | 1 year: J | edge lost (J), 4m / 1y |
|---|---|---|---|---|---|---|
| `aMCTS` (leaf only) | everything, player state frozen at the cutoff | 2.1 (0.9, 3.3) | 1.3 (0.0, 2.6) | 2.1 (1.6, 2.7) | 3.7 (−0.2, 7.6) | 9% / 41% |
| `aMCTS-cur@v` | player state only | 0.0 (−0.2, 0.2) | 0.1 (−0.2, 0.5) | 0.1 (−0.2, 0.3) | 0.5 (0.2, 0.9) | 6% / 10% |
| `aMCTS-pp@v` | player state only | −0.4 (−0.8, −0.0) | 1.7 (1.2, 2.2) | −0.5 (−0.8, −0.1) | 3.8 (3.3, 4.2) | 11% / 23% |
| `pPol` (policy, no search) | everything | 2.0 (1.1, 3.0) | 1.5 (0.5, 2.5) | 2.3 (1.3, 3.3) | 3.8 (0.1, 7.6) | 11% / 45% |
| `p1step` | everything | | | 1.3 (0.8, 1.7) | 3.0 (2.2, 3.8) | – / 46% |
| `a1step` | everything | | | 1.2 (0.8, 1.6) | 4.9 (4.2, 5.6) | – / 37% |

- A frozen player state costs a drafter in proportion to how much its choices depend on the players. The leaf-only drafter barely uses them and barely notices; the distilled-prior drafter loses 1.7pp at 4 months and 3.8pp at a year under J, about a quarter of its edge, while its population side is unchanged.
- Keeping the player state current is the cheap part of upkeep: one walk over the players' games, about 0.1 ms per player-day on CPU.

**Drafters vs the prediction model at the same staleness** (prediction: share of the log-loss lift over a coin flip lost on tournament-window games).

| staleness | WP only | WP + player state | paper-1 MCTS, edge lost (CONS) | `aMCTS-pp`, player state only, edge lost (J) |
|---|---|---|---|---|
| 4 months | 9% | 56% | 11% | 11% |
| 1 year | 28% | 70% | 12% | 23% |

- The drafters lose less than the models they are built on. A search on a stale WP still ranks most of the same drafts on top, and a stale player state still knows most players' best heroes.

### Real-game backtests

**(a) Natural agreement** (20,000 OOT games, 200,000 picks). The drafter is the behavioral-prior search, aware and blind (`aMCTS-bc`, `pMCTS-bc`), at every real pick state.

| | value |
|---|---|
| real pick equals the aware recommendation | 9.3% |
| real pick equals the blind recommendation | 6.4% |
| aware and blind recommend the same hero | 40.7% |

Picking team's win, linear probability model, pp, SE clustered by player and game:

| model | matched aware | other terms |
|---|---|---|
| raw | +3.4 (0.4) | |
| + draft-only WP + causal MMR | **+0.9 (0.4)** | |
| matched only the aware pick (aware ≠ blind) + controls | +1.8 (0.5) | matched blind −0.4 (0.4) |
| + the skill model's predicted gap | −0.5 (0.4) | gap coefficient 1.02 (0.03) per unit |

- Agreeing with the player-aware drafter predicts winning, and the skill model's own gap accounts for all of it. The gap is calibrated on real outcomes (slope 1.02).
- This is evidence for the value the drafter uses, not for choices beyond it.
- With paper 1's F-prior drafter, matches are too rare to say anything (2.5% of picks).
- The slot table's MMR is the post-game rating and leaks the result, so the controls use the causal MMR (P3_MMR_AT_GAME).

**(b) Forced switches** (OOT, 1.05M pick slots of players with a 30+ game hero). Within player: realized r = y − WP in games where the hero was taken before his pick, against his other games, compared with the model's predicted r for the hero he played.

| situation | share of slots | realized pp (SE) | predicted pp | realized − predicted (SE) |
|---|---|---|---|---|
| best hero banned | 9.0% | −2.13 (0.18) | −1.26 | −0.86 (0.17) |
| best hero taken by a teammate | 2.5% | −1.44 (0.31) | −0.95 | −0.48 (0.31) |
| best hero taken by an opponent | 3.8% | −0.79 (0.26) | −1.00 | +0.21 (0.26) |
| main banned | 9.8% | −1.85 (0.17) | −1.10 | −0.75 (0.17) |
| main taken by a teammate | 2.8% | −1.74 (0.30) | −0.83 | −0.91 (0.30) |
| main taken by an opponent | 4.3% | −1.34 (0.23) | −0.84 | −0.50 (0.23) |
| best hero taken, any way | 15.4% | −1.90 (0.14) | −1.33 | −0.58 (0.14) |

- The main-ban effect replicates P3_TAILS (−1.76 there, −1.85 here on OOT games).
- The model predicts about 60% of the realized cost of a forced switch. Bans and teammates that take a player's hero cost more than the skill gap between his heroes.
- This is the case for denial bans in the drafter: the ban-denial prior helps on both lobby sets (+0.4pp on the tune lobbies, +1.0pp on the tournament lobbies).
- Caveats:
  - This is observational. Bans may target players in form.
  - When an opponent takes the hero, the cost matches the prediction, which hints that ban-driven selection explains part of the gap.

### Compute

**Training** (GPU-hours on one GPU each; "3090" means its WSL GPU; CPU work at nice 19):

| item | where | cost |
|---|---|---|
| paper-1 F policies (5 seeds, existing) | 3080 / 3090 | 8.3 GPU-h per seed alone (21 h when two shared a 3090) |
| stale vintages: statistics, features, 3 WP seeds | 3090 | 0.31 h (s4m) and 0.24 h (s1y) wall, of which 0.08 GPU-h |
| stale GD (s1y era, 2 variants) | 3090 | 1.34 GPU-h (s4m reuses the drift paper's cutoff GD) |
| stale F policies (2 seeds per vintage) | 3090, 3080 | 7.9 to 8.6 GPU-h per seed (33 GPU-h for the four) |
| fine-tune upkeep: WP | 3090 | under 1 GPU-minute per vintage |
| fine-tune upkeep: F continuation (60K episodes) | 3080 / 3090 | 1.6 GPU-h (s4m, 3080) and 2.1 GPU-h (s1y, 3090) |
| judge skill fit (other player half) | 5090 | 147 s |
| player states, combiners, judge fits | 3090 CPU | minutes each (one walk: 0.1 ms per player-day) |
| (b) distilled player-feature prior | 5090 + CPU | 0.4 GPU-h targets (shared GPU), 1 CPU-minute fit |
| (c) value net | CPU | minutes (12-feature MLP) |
| (d) self-play prior | 5090 | 1.1 GPU-h (4 iterations) |
| rich priors (3 signals) | 3090 CPU | 3 to 15 minutes each |
| opponent comfort model | CPU | 1 minute |
| DraftRec | 5090 | 1.2 GPU-h (plus 0.4 in aborted attempts) |
| kernels (`cuda_pvnet`, `cuda_explore`) | all hosts | CPU builds, about 10 minutes each per host |

**Per draft** (one side, 8 decisions, 5090):
- paper-1 MCTS 34 ms;
- aware leaf 36 ms (+5%);
- distilled prior 41 ms (+19%, mostly the CPU bias);
- selected design 41 ms (`x-combo1`, 400 sims); 148 ms for `x-combo3` (1,600 sims);
- one-step about 40 ms of CPU across 12 processes;
- policy-only drafters about 2 ms.

**Tournament compute.** The 300-lobby round robin, spokes, design ablation and the full matrix of main systems are 566 ordered pairs, about 6.2 hours of drafting time (5090 and 3090 searches plus the one-step CPU pool); the tune set is 184 pairs, 2.3 hours. Scoring is CPU minutes. The DraftRec opponent fit took seconds of GPU time after 3 CPU-minutes of history tensors.

## Validation against paper 1's checklist (added 2026-10-21)

Each check paper 1 used for its blind MCTS, and whether the player-aware tournament has it.

| paper 1 check | here | how, or what is missing |
|---|---|---|
| Multi-judge consensus, each judge reported (gN, gN-naive, RN, QM2026) | yes | CONS is the same composition-corrected consensus; every head-to-head also stores each judge (`analysis.json`). J adds the personal term. |
| Quick Match as a separate test population | yes | QM2026 inside CONS for the population part; the personal term checked on later Quick Match outcomes (addition 1 below: c2 2.55, SE 0.39, against 4.53 on SL). |
| Judges outside the drafters' data | yes, and stricter | The drafters stop at 2026-05-22; lobbies are 05-23 to 06-30; the judge window is 07-01 to 09-11; the judge's skill model is fit on the other half of the players; the split-history judge uses disjoint days. |
| Calibration against real outcomes | yes | 60,500 real tournament-window drafts: J log loss 0.656 vs CONS 0.671. Fitting the two parts separately, the personal term's slope is 1.02 (SE 0.03): it is calibrated. The consensus part's slope is 1.73 (SE 0.03): it is underconfident, so J understates population differences, not personal ones. J_cal is reported next to J in the tables; from the definitive run on, its two slopes are fitted per pass on the judge window with a cross-fitted personal term and stored in judge.json (the prototype used 1.17 and 1.00). J under the outcome-free assignment (players on heroes by their earlier games) is reported next to the best-assignment J. |
| Bradley–Terry | yes | Strengths plus a seat (first-pick) term, fitted on fractional outcomes per ordered pair; CIs from 1,000 lobby-bootstrap refits. The BT fit has no seed terms; seed variation enters the head-to-head intervals through the seed SE (paired or crossed, see Interval). |
| Every ordered pair, both seats | yes | All 91 pairs of the 14 main systems, both seats, 300 lobbies each. |
| Degenerate and safety rates (healer, frontline, ranged, role stacking) | yes | Per system, with paper 1's structural mask on every search drafter (0% degenerate); off-role rate added. |
| Leak controls (out-of-fold statistics, sealed holdout, test split) | yes | Paper 1's leak-free WP with out-of-fold statistics; the sealed build and the 5,000 sealed replays are never read; the DraftRec opponent, the distilled prior and the opponent comfort model are fit on lobbies outside the tournament, probe and tune sets. |
| Search-budget scaling | yes | 100 to 1,600 sims for the selected design and the blind search (addition 2 below). |
| Ensemble / LCB diagnostics on the value model | yes | Every draft rescored with the three enriched-WP seeds (addition 3 below). |
| OOD checks (manifold discriminator, hand-crafted probes, degenerate probes) | yes, except the manifold discriminator | Pool coverage, the structural mask, GD likelihood of every system's picks and hand-built personal probes (additions 4 and 5 below). |
| Interaction metrics (counter, synergy, resilience) and behavior (diversity, GD similarity) | yes | Future hero WR, synergy and counter (judge-free); resilience, diversity and GD similarity per system (addition 5 below). |
| Independent baseline | yes | DraftRec rebuilt from its paper and code, as a drafter and as the opponent model. |
| Self-grading check (cross-evaluation) | yes | Each drafter's own value against the judge's (diagnosis section); judge-free realized records. |
| Many seeds | partly | Paper 1 used 15 seeds per configuration. Here the prior has 5 seeds (2 for stale vintages, 1 for fine-tunes), and the search's own run-to-run noise is measured directly from four runs of the main pairs (SD 0.18pp median, 0.35 at most; addition 6 below). |

**Additions run on 2026-10-21** (all on the 5090; tables in `P3_TOURNAMENT_TABLES.md`):

1. **Quick Match check of the personal term** (`p3t_qm.py`). Quick Match has no draft, so each player's hero is known. The judge's Storm League skill state (as of 2026-09-12, from judge-window SL games only) is matched to Quick Match games from 2026-09-12 to 10-21 (local research DB, per-player rows; strictly later than every state). Population part: paper 1's QM2026 judge (trained on games up to 2026-06-18).
   - Coverage is the limit: 35,011 QM games have ten per-player rows and a known map, tier and hero (games with Xal'atath, a hero newer than the 90-hero set, are dropped), but only 8.8% of their player rows belong to a player with an SL state. 13,669 games have at least one such player (1.7 per game); 2,629 have one on each team; 25 have three or more per team.
   - Fit on the 13,669 games (unknown players count as the population, s = 0): logit P(win) = 0.84 logit p_QM2026 + **2.55 (SE 0.39)** dS + 0.20 (0.03) dO. On the 2,629 games with a known player on each team, c2 = 2.93 (0.72).
   - So SL skill predicts QM outcomes with the expected sign, far beyond chance (6.6 SE), at 56 to 65% of the SL judge's coefficient (4.53). Some shrinkage is expected: a different mode, other players' skill unknown, a month later. The off-role term flips sign in QM (players who queue off their SL role win slightly more), so it does not transfer; in J it is tiny (−0.03 logit per off-role player).
   - With more coverage (more QM history for SL players, or SL states for QM regulars) the test would tighten; the data that would help most is per-player QM rows for 2026-07 to 09 (we have 8K QM games for June and 14K for September with per-player rows).
2. **Search-budget curve** (tune lobbies, against `pMCTS-cur` at 400 sims), J: the selected design at 100, 200, 400, 800 and 1,600 sims scores 57.7, 59.1, 60.6, 60.9 and 59.9%; the blind search at the same budgets 50.1, 49.9, 49.8, 50.2 and 49.8%. The player-aware design gains about 3pp from 100 to 400 sims and is flat after; the blind search does not gain at all, because paper 1's F prior already concentrates almost all visits on one or two heroes (paper 1's sims gains were in training, not inference).
3. **WP-seed disagreement.** Rescoring every tournament draft with the three seeds of paper 1's enriched WP: each system's mean WP moves by at most 0.6pp across seeds (0.15 to 0.35pp for the player-aware designs, 0.5 to 0.6pp for paper 1's MCTS and the real drafts), and the per-draft spread is 1.0 to 1.2pp for every system. No drafter wins by exploiting one WP seed's blind spots.
4. **Personal probes** on the 300 real drafts (team 0's most comfortable hero swapped for a same-role hero with a near-identical population win rate):
   - to a hero none of the five has played (214 lobbies): J drops 7.7pp and ranks the probe below the real draft in 95% of lobbies; CONS moves +1.0pp, below in 37%;
   - to the hero the team is worst at (300 lobbies): J drops 9.4pp, below in 99%; CONS 0.0pp, below in 50%;
   - all five heroes to never-played same-role heroes (24 lobbies): J drops 30.5pp, below in 100%; CONS +0.5pp.
   J sees exactly what it should, and CONS, blind to players, does not.
5. **Behavior.** The player-aware designs use all 90 heroes (entropy 5.6 to 5.9 bits, top-10 share 0.40 to 0.47); paper 1's MCTS uses 45 (4.2 bits, top-10 share 0.77). GD similarity of their picks is 4 to 5%, against 16% for the real drafts and 25% for the BC-prior searches, and their picks are less GD-likely than real picks (mean log p_GD −4.6 vs −4.0): they leave the behavioral manifold, as paper 1's MCTS does (−5.2). Resilience (judge-window statistics) is −0.26 for the selected design, −0.31 for paper 1's MCTS and −0.22 for the real drafts: no system picks heroes the opponent cannot answer.
6. **Repeated runs** (search noise). The 20 head-to-heads among the main search drafters, Real and GD were replayed three more times with shifted kernel seeds (same lobbies, same prior seeds). The SD of the four J values is 0.01 to 0.35pp (median 0.18); for CONS 0.27 at most. Matchups with GD or Real are steadiest (≤ 0.24); the closest contests between player-aware designs vary most (0.33 to 0.35pp), about the size of their lobby-bootstrap SE (0.24 to 0.27). For those comparisons the honest SE is about 0.4pp (both combined), which is why the three versions of the selected design are called a tie: `x-combo1` vs the selected design scored 49.5, 50.3, 49.9 and 50.0% in the four runs.

## Design (fixed before the tournament)

**Windows.**

| what | dates |
|---|---|
| drafters' population data (current vintage) | pinned snapshot, through 2026-05-22 |
| tournament lobbies | real OOT games 2026-05-23 to 06-30 |
| judge window (personal evidence, truth statistics, judge combiner) | 2026-07-01 to 09-11 |
| backtest games | OOT 2026-05-23 to 09-11 |

- The sealed personalization build 2.55.17.98025 (from 2026-09-12) is never read.
- The personalization store keeps CAP_TS 2026-09-28.
- The 5,000 sealed replays are not touched.

**Lobbies.** Real games with a full 16-step draft whose ten picks join to known players:
- all ten players have 50+ earlier games;
- each has 10+ games in the judge window;
- 300 lobbies drawn with a fixed seed. Every ordered pair plays the same lobbies, with seat A as the real first-pick team.

Lobby rules, the same for every system:
- a team's picks come from the union of its players' pools (heroes each played before the lobby day);
- the team picks a hero, and personal values use each team's best assignment of heroes to players. The real pick order (who picked what) is never used.

**Player state** (the drafter's s(p, h) for all 90 heroes). From the main skill model: `met2_hs2` structure, full scoreboard export, `a_shrink`, merged full_1017 history.
- s = experience offset + posterior mean + temporal layer (drift and new-hero learning), with state through the day before.
- The state reproduces the `full_ms_layer` arm's slot predictions to 5e-6.

**Systems.**

| name | what |
|---|---|
| `pMCTS-cur` | paper 1's best MCTS (config F, 5 seeds, leak-free enriched WP, deploy statistics, chance-node search, mid_high_low composition fallback), searching at decision time (400 sims) with the population leaf |
| `aMCTS-cur` | the same search with the personal leaf V = sigmoid(b0 + b1 logit WP + b2 (S_ours − S_theirs) + b3 (O_ours − O_theirs)). The term sits inside the CUDA kernel (`cuda_personal`, assign mode): a per-lobby leaf adjustment, with no retraining |
| `pMCTS-bc`, `aMCTS-bc` | the same search with the outcome-free behavioral prior (train-window BC policy) in place of the F policy |
| `pPol-cur` | the F policy argmax, no search (paper 1's tournament agent) |
| `a1step-cur`, `p1step-cur` | the one-step drafter (top 15 GD plus top 15 personal candidates, 4 GD rollouts each), aware and blind, with the same WP |
| `GD` | behavioral cloning, argmax |
| `Real` | the real lobby's actions, replayed when still legal |

**Vintages for drift.**
- `s4m`: population data through build 2.55.14.95918 (2026-02-10), player state frozen at that date.
- `s1y`: through 2.55.10.94189 (2025-06-02).

Upkeep arms of the search drafter:

| arm | what |
|---|---|
| stale | stale weights and statistics |
| `-rs` | stale weights with the current statistics tables in the kernel |
| `-ft` | fine-tune from the stale weights on recent data |
| `pMCTS-cur` | the full retrain |

Player-state upkeep: `aMCTS-cur@s4m` and `aMCTS-cur@s1y` use the current population side with player state frozen at the vintage cutoff.

**Judges** (none shares data with a drafter):
- **Population:** paper 1's consensus (gN, gN-naive, RN, QM2026, composition-corrected). These are trained on post-snapshot or Quick Match games.
- **Personal term:** a skill model fit on the OTHER phase-1 half of players (`full_ms_flip`), with evidence from the judge window only. It is combined with the consensus as a fixed offset; the coefficients are fit on judge-window games.

**Statistics** (as paper 1 and the drift paper):
- pair means with lobby-cluster bootstrap CIs;
- standings over opponents;
- Bradley–Terry strengths on all pairs with bootstrap CIs;
- crossed random effects for the two sides' seeds;
- degenerate-team, healer and off-role rates.


## The drafters, exactly

**Shared by every drafter.**
- Draft order: Storm League (bans 0,1,0,1; picks 0,1,1,0,0; bans 1,0; picks 1,1,0,0,1). The real first-pick team is kernel team 0.
- Hero set: the 90-hero v1 set.
- Picks are from the union of the team's pools, falling back to any free hero.
- The team picks. Personal terms use each team's best hero-to-player assignment over all 120 permutations (assign mode).
- Every search drafter picks under paper 1's structural constraint. A hero is kept only if the team can still reach ≥ 1 healer, ≥ 1 frontline, ≥ 1 ranged and ≤ 2 of each stacking role from its legal heroes. With search, the constraint applies at the root: the drafter takes the most-visited allowed hero. GD and Real are unconstrained.
- The constraint is required here. Without it, the unconstrained first tournament pair fielded 19% and 43% degenerate teams: the first-pick search took two healers, and the second team's pools then had none left.

**Population side (vintage cur), paper 1's site-tier rebuild.**
- **Snapshot:** 1,956,753 replays through 2026-05-22. The split is the paper's test set (38,869), a 2% validation hash, and the rest for training.
- **Leaf WP:** `enriched` in `paper1_revision/train_wp.py`.
  - Inputs: 283 (197 base + 86 enriched features).
  - MLP 256 → 128, dropout 0.3, AdamW lr 5e-4, wd 5e-3, cosine, batch 4,096, ≤ 200 epochs, patience 25.
  - Training rows use out-of-fold statistics (5 hash folds).
  - Seed 42 of 3 was selected on validation. Symmetrized over team order.
- **Kernel tables:** the deploy statistics (all training games), with the own-corpus role-composition table (cells of ≥ 50 games). The composition fallback order is mid, high, low.
- **GD:** paper 1's five `generic_draft_{0..4}` models.
  - Architecture: 289 → 256 → 128 → 90, ReLU, dropout 0.2 / 0.1.
  - Data: the p1site rebuild, next-action cloning on snapshot drafts.
  - In search, GD is the chance-node opponent and the rollout policy. Draft i uses GD model i mod 5.
- **F policy (prior of `*MCTS-cur`, the `pPol` agents):** AlphaZeroDraftNet, base size, 4.4M parameters.
  - Residual backbone 289 → 768, 3 blocks, → 256; linear policy head; perspective-aware value head.
  - Bootstrapped from GD.
  - Self-play: 300K episodes, 400 sims per move, chance-node search (progressive widening k 1.0, α 0.5), c_puct 2.0.
  - Opponent: GD model 0. Buffer 150K, train batch 512, Adam lr 1e-3 with cosine T_max 150K.
  - Value pretraining on the training split's replays.
  - Seeds 0 to 4; draft i uses seed i mod 5. About 8.3 GPU-hours per seed on a 3080.
- **BC prior (`*MCTS-bc`):** AlphaZeroDraftNet base, distilled by behavioral cloning from GD variant 0. Train-window states only (games before 2026-02-10), outcome-free.

**Player side.**
- **State s(p, h):** the main skill model, as of the day before the lobby, for all 90 heroes.
  - Model: `met2_hs2` structure (overall column + 2 factors, shared metric loadings, per-hero s²). Inputs: the full scoreboard export with `a_shrink`, and the merged full_1017 history.
  - s = experience offset + posterior mean + temporal layer (drift half-life 730 d, k_d 10; new-hero learning k_e 2).
- **Off-role:** fine role under 10% of 50+ earlier games.
- **Combiner:** V = sigmoid(b1 logit WP + b2 (S_ours − S_theirs) + b3 (O_ours − O_theirs)).
  - Fit on 5,397 games the WP never trained on (2026-02-11 to 05-22: paper-1 test and validation splits plus post-snapshot uploads).
  - Fitted values: b1 1.20, b2 4.54 (SE 0.31), b3 −0.015. b0 = 0.

**Search.**
- Decision-time decide-only search in `cuda_personal` (aware leaf) or `cuda_prior` / `cuda_pvnet` (variants below).
- 400 simulations, chance-node opponent, c_puct 2.0, no root noise, root argmax of visits under the constraint.
- Kernel seeds are fixed per pair, lobby and step.
- Aware and blind twins share the prior seed, GD model and kernel seed of every decision.

**Ways to put players into the search (the four options).**

| option | system | prior | leaf | training for the player part |
|---|---|---|---|---|
| (a) leaf only, blind-trained prior | `aMCTS-cur` | F (blind self-play) | WP + personal terms | none (combiner fit, CPU seconds) |
| (a') leaf only, behavioral prior | `aMCTS-bc` | BC | WP + personal terms | none |
| (b) player-conditioned prior | `aMCTS-pp` | α log p_F + log mean_team exp g(f(p, h)) | WP + personal terms | targets: aware BC search at 1,600 sims on 30,000 real pick states of 3,000 non-tournament lobbies (0.4 GPU-h); fit α and g by cross-entropy (CPU, 1 min) |
| (c) value net with player features | `aMCTS-vn` | BC | b1 logit WP + h(F_ours) − h(F_theirs), h an MLP on the assigned roster (12 features) | 42,640 held-out games, CPU |
| (d) self-play with sampled real rosters | `aMCTS-sp` | as (b), refit each iteration | WP + personal terms | 4 iterations × 4,096 rosters (2026-03-29 to 05-22), 800 sims, root Dirichlet ε 0.25, temperature 1 |

- g is an MLP, 6 → 32 → 32 → 1. Its features are b2 s(p, h), off-role, log(1 + games on h), never played, the share of p's games on h, and log of p's fine-role share.
- The prior-only versions of (b) and (d) are `aPol-pp` and `aPol-sp`.

**One-step (`a1step`, `p1step`).**
- At each pick, candidates are the top 15 by GD plus the top 15 by the team's best personal term.
- Each candidate gets 4 rollouts completing the draft with the vintage's GD; bans in the rollouts are sampled from GD.
- Finished drafts are valued by the vintage WP (`p1step`) or V (`a1step`) on the same rollouts. Own bans: GD argmax.

**Imitation and the human reference.** `GD` is the 5-model GD average, argmax. `Real` replays the lobby's real actions; the first illegal action falls back to GD argmax (11.4% of its steps).

**Vintages for drift.** These are built by the same recipe on training games at or before the cutoff build, with that era's GD.

| vintage | cutoff | train games | WP test acc / slope | GD |
|---|---|---|---|---|
| s4m | 2.55.14.95918 (2026-02-10) | 1,731,780 | 0.5753 / 1.12 | drift `gd_cutoff` 0, 1 |
| s1y | 2.55.10.94189 (2025-06-02) | 1,262,321 | 0.5725 / 1.10 | era GD 2.55.10.94189 0, 1 |

- Each vintage has F seeds 0 and 1, and its player state is frozen at the cutoff.
- Upkeep arms:
  - `-rs`: the stale F policies, WP and GD with the current statistics tables.
  - `-ft`: the WP fine-tuned on post-cutoff training games (early stop on post-cutoff validation), plus the F policy continued 60K episodes from seed 0 against it, with current statistics and GD.
  - `aMCTS-cur@v`: the current population side with the vintage's frozen player state.

**Time rule for the learned player-side pieces.**
- The distilled prior's targets, the opponent comfort model and the search-distilled rich prior come from tournament-window lobbies (2026-05-23 to 06-30) that are not tournament, probe or tune lobbies.
  - They use no outcome, only real picks or search visits, and the players' states as of each lobby.
  - Their parameters are a small MLP over hero-agnostic features, so no player-specific information carries over.
- Everything else on the population side ends on 2026-05-22. The self-play rosters, the outcome-weighted BC and one-step targets, and DraftRec's training games come from before 2026-05-23.

**Exploration design study (systems `x-*`, `p3t_explore.json`).**
- All are `aMCTS-pp`-like decide-only searches in `cuda_explore`, a copy of `cuda_pvnet` with two gated modes. Parity:
  - byte-identical to `cuda_pvnet` when the new modes are off;
  - population path identical to the reference.
- The knobs:
  - **c_puct**: 1, 2 (default) or 4.
  - **α**: the weight on log p_F in the prior. Fixed at 0, 0.3 or 0.6; the fitted value (0.160); or per pick position (0.5 for the first three picks, 0.3 for picks 4 and 5, 0.156 after).
  - **Bias scale**: ×0.5 or ×2 on the distilled personal bias.
  - **Optimism**: the leaf uses s + κ sd for our players and s − κ sd for theirs (κ 0.5 or 1).
  - **Assignment bias**: at the decision state, each hero's best-assignment gain for our team (b2 units) is added to our pick prior (×1 or ×3).
  - **Opponent model** (`cfg[6]=2`): opponent picks inside the tree and in rollouts follow softmax(0.79 log p_GD + scale × log mean_j exp(0.64 log1p n_jh − 1.26 [n_jh = 0])), fit on 30,000 real picks (scale 1, 2 or 3).
  - **Denial bans** (`cfg[56]=2`): our ban prior gets λ × b2 × the opponents' best-player skill on each hero (λ 1, 3 or 5).
  - **Value net v2**: the leaf's value net refit on 112K pre-tournament games.
  - **Rich prior**: at the decision state the prior becomes softmax(rich logits). It is applied as a static bias for our later in-tree picks.
  - **Sims**: 400 or 1,600.
- **Selected design** (pre-stated rule): `x-combo1-dr-chance`, the distilled prior + denial bans λ 3 + the DraftRec opponent at chance nodes (rollouts keep GD) + assignment bias ×1, at 400 sims. `x-combo1` is the same with the comfort opponent (×2) at chance nodes and in rollouts; `x-combo3` is `x-combo1` at 1,600 sims (the rule's pick before the kernel fix).

## Search prior: a player-blind trained prior hides the personal leaf (probe, before the tournament)

Probe setup (`p3t_probe.py`): 500 tournament-window lobbies outside the tournament set, all 10 real pick states each (5,000 decisions). At each state two decide-only searches run with the same seeds, personal leaf on and off. The table is model-internal.

| search prior | sims | picks where aware and blind differ | aware search's own Q gain when they differ (pp) | root visit entropy, blind (nats) |
|---|---|---|---|---|
| paper-1 F policy | 400 | 4.4% | +5.2 | 0.44 |
| paper-1 F policy | 1,600 | 8.3% | +5.0 | 0.43 |
| behavioral (BC) | 400 | 59.0% | +6.4 | 2.69 |
| behavioral (BC) | 1,600 | 65.1% | +6.7 | 2.78 |

- Search cost per decision (5090, uncontended, 400 sims): 4.3 ms with the F prior, 3.8 ms with the BC prior; about 4 times that at 1,600 sims.
- The F policy was trained by self-play against a player-blind value. It is sharp: the blind search puts almost all visits on one or two heroes. A personal term added only at the leaf moves 4 to 8% of picks.
- Over a flat behavioral prior the same leaf moves about 60% of picks.
- So "train player-blind, add the personal term at the leaf" works only when the prior leaves room for search.
- The tournament therefore carries both priors. A third arm, a prior trained with player features (`aMCTS-pp`), follows when its training finishes.

## Files and rerun

In `training/personalization/`:

| file | role |
|---|---|
| `p3t_data.py` | lobbies, player states (cur, stale, judge, split), judge fit data, rosters, pre-tournament lobbies |
| `p3t_value.py` | combiners and prediction drift |
| `p3t_systems.py` | every drafting system and the structural mask |
| `p3t_tourney.py` | lobby selection, schedules, the runner |
| `p3t_judge.py` | judges, truth statistics, scoring |
| `p3t_analyze.py` | head-to-heads, Bradley–Terry, seed effects, standings |
| `p3t_diag.py` | attenuation diagnostics |
| `p3t_report.py` | `P3_TOURNAMENT_TABLES.md` |
| `p3t_backtest.py` | backtests (a) and (b) |
| `p3t_sanity.py` | the paper-1 reconciliation |
| `p3t_probe.py` | decision probe and timing |
| `p3t_pprior.py`, `p3t_selfplay.py`, `p3t_vnet.py`, `p3t_vstudy.py`, `p3t_rprior.py`, `p3t_oppmodel.py` | design options and the design study |
| `p3t_draftrec.py` | the DraftRec baseline |
| `p3t_oppdr.py` | DraftRec as the opponent inside the search (fit lobbies, histories, fit, kernel rows) |
| `p3t_verify.py`, `p3t_pvnet_verify.py`, `p3t_explore_verify.py`, `p3t_determinism.py` | kernel parity and determinism checks |
| `p3t_vintages.json`, `p3t_explore.json` | the drafter configurations |

- Kernels: `cuda_pvnet/` and `cuda_explore/` (new), plus `cuda_personal/` and `cuda_prior/`; all race-fixed on 2026-10-19 (`build_p3_kernels.sh`, `remote/p3t_build_all.sh`).
- Stale vintages: `paper1_revision/vintage.py`, `vintage_ft.py` and `vintage_launch.py` (`P1_VINTAGE=s4m|s1y|s4m_ft|s1y_ft`).
- Results: `results/p3t/*.json` (small JSON, no player keys) and `results/P3_TOURNAMENT_TABLES.md` (generated). Caches stay in `cache_full/p3t/` on the workers.
- **Rerun:** `nohup nice -n 19 taskset -c 48-51 training/personalization/remote/p3t_pipeline.sh > training/personalization/logs/p3t_pipeline.log 2>&1 &`, after the personalization rerun has refit `full_ms` on the new store. It refits the judge on the other player half and runs every stage above in order.
  - The pipeline first rebuilds every P3 kernel and stops unless the determinism and parity checks pass.
  - The exploration design is the one the rule picked on the tune lobbies (`x-combo1-dr-chance`, with `x-combo1` and `x-combo3` reported next to it). The full-data rerun re-reports the tune table and keeps that choice.
