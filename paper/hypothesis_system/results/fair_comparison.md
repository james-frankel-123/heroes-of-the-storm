# Fair comparison: readable models against the neural WP, confounds removed

Date: 2026-10-16. Code: `training/scilearn/compare/`. Numbers: `fair_comparison.json` (same directory). Per-game predictions and fits: `/home/max/scilearn-private/fair/`, `/home/max/scilearn-private/jobs/fair_*` (private). No C, F, R, pool or sealed row was read.

The old headline compared readable models at 0.6725 and 58.1% with N at 0.6790 and 56.8% on Dpost. That gap had two confounds:
1. The readable models used player data and N did not.
2. N stopped at 2.55.14, while the readable models were fit through 2.55.16.

This note removes both.

## 1. Headline

Log loss on three sets. Each cell gives the log loss, then in brackets the paired difference against the neural model in the same column, with its SE. A positive difference means worse than the neural model.
- The draft-only columns are measured against N matched.
- The player-data columns are measured against P3 on N matched.

| Model | Draft-only: dev | Draft-only: rare | Draft-only: onset | Player data: dev | Player data: rare | Player data: onset |
|---|---|---|---|---|---|---|
| A0 | 0.67772 (-0.00052, 0.00022) | 0.66711 (+0.00067, 0.00103) | 0.67711 (-0.00048, 0.00046) | 0.67253 (+0.00866, 0.00047) | 0.66157 (+0.00814, 0.00170) | 0.67119 (+0.00834, 0.00091) |
| K0 | 0.68605 (+0.00781, 0.00035) | 0.67739 (+0.01094, 0.00151) | 0.68589 (+0.00830, 0.00073) | 0.68086 (+0.01698, 0.00054) | 0.67146 (+0.01803, 0.00201) | 0.67983 (+0.01698, 0.00109) |
| w2-K-b final incumbent | 0.67767 (-0.00057, 0.00022) | 0.66665 (+0.00021, 0.00105) | 0.67702 (-0.00058, 0.00047) | 0.67223 (+0.00836, 0.00047) | 0.66090 (+0.00748, 0.00171) | 0.67089 (+0.00804, 0.00093) |
| w2-K-blinded final incumbent | 0.67849 (+0.00025, 0.00024) | 0.66910 (+0.00266, 0.00118) | 0.67803 (+0.00044, 0.00051) | 0.67303 (+0.00916, 0.00046) | 0.66309 (+0.00967, 0.00178) | 0.67187 (+0.00902, 0.00093) |
| **N matched / P3 on N matched** | **0.67824** (ref) | **0.66644** (ref) | **0.67759** (ref) | **0.66387** (ref) | **0.65342** (ref) | **0.66285** (ref) |
| N frozen / P3 on N frozen | 0.67896 (+0.00072, 0.00011) | 0.66681 (+0.00036, 0.00043) | 0.67857 (+0.00098, 0.00023) | 0.66449 (+0.00062, 0.00011) | 0.65375 (+0.00032, 0.00042) | 0.66373 (+0.00088, 0.00022) |

The three sets:
- **Dev** is Dpost, 173,394 games.
- **Rare** is O5, 12,544 Dpost games where either team's role multiset holds at most 0.2% of S teams.
- **Onset** is O2: the first 10K games of each 2.55.17 build, 39,473 games, pooled.

On dev, accuracy, Brier score and calibration slope are:

| Model | Draft-only: acc. | Draft-only: Brier | Draft-only: slope | Player data: acc. | Player data: Brier | Player data: slope |
|---|---|---|---|---|---|---|
| A0 | 57.16% | 0.24239 | 0.862 | 58.14% | 0.23988 | 0.881 |
| K0 | 54.82% | 0.24647 | 0.815 | 56.37% | 0.24391 | 0.861 |
| w2-K-b | 57.17% | 0.24236 | 0.861 | 58.24% | 0.23974 | 0.882 |
| w2-K-blinded | 57.02% | 0.24276 | 0.864 | 58.11% | 0.24012 | 0.885 |
| N matched / P3 on N matched | 56.92% | 0.24264 | 0.895 | 59.65% | 0.23578 | 0.941 |
| N frozen / P3 on N frozen | 56.83% | 0.24299 | 0.882 | 59.58% | 0.23607 | 0.935 |

## 2. What the readable models beat, match or lose to

**Without player data, at the same cutoff:**
- **A0 and the w2-K-b incumbent beat N, by a small margin.**
  - Dev: 0.0005 to 0.0006 better (about 2.5 SE) and 0.25pp more accurate.
  - Onset: about 0.0005 better, which is 1 SE and not separable.
  - Rare compositions: they match N (+0.0002 and +0.0007, each under 1 SE).
  - Calibration: N is better (slope 0.895 against 0.86).
- **The w2-K-blinded incumbent matches N on dev** (+0.0003, 1 SE) and loses on rare compositions (+0.0027, 2.3 SE).
- **K0 loses to N everywhere**, by 0.008 to 0.011 and 2.1pp of accuracy.

**With player data allowed, every readable model loses to paper 2's P3.**
- The margin is 0.008 to 0.010 for A0 and both incumbents, and 0.017 for K0.
- It holds on dev, rare and onset alike, at 4.4 SE or more on every set.
- P3 is also 1.4pp more accurate than the best readable model, and better calibrated (slope 0.94 against 0.88).
- Refit without it, the readable models' player term is worth 0.0052 to 0.0055. P3's personal layer is worth 0.0144.

**The old headline gap of 0.0064 (A0 against frozen N on dev) breaks down as:**
- about 0.0052 from player data;
- 0.0007 from recency (N matched against N frozen);
- 0.0005 left over.

**Dev is the w2 incumbents' own dev tranche.** Their accepted terms were selected on it, so it favours them. A0 is LLM-free and was not selected on Dpost.

**At the S cutoff the small draft-only edge flips sign** (section 4). Draft-only A0 fit on S beats frozen N on Dpre and G by 0.0005 to 0.0007. On Dpost, three builds further out, it loses to frozen N by 0.0006.

So, without player data, the readable model is at parity with the neural WP, give or take 0.0006 depending on the horizon.

## 3. Onset detail

Pooled first 1K games of each 2.55.17 build (4,000 games). Paired difference against N matched or P3 on N matched (SE):

| Model | Draft-only | Player data |
|---|---|---|
| A0 | 0.67702 (-0.00017, 0.00145) | 0.66875 (+0.01024, 0.00289) |
| K0 | 0.68675 (+0.00956, 0.00250) | 0.67830 (+0.01979, 0.00369) |
| w2-K-b | 0.67713 (-0.00007, 0.00147) | 0.66897 (+0.01046, 0.00290) |
| w2-K-blinded | 0.67899 (+0.00180, 0.00160) | 0.67089 (+0.01238, 0.00294) |
| N matched / P3 matched | 0.67719 (ref) | 0.65851 (ref) |
| N frozen / P3 frozen | 0.67824 (+0.00105, 0.00072) | 0.65931 (+0.00080, 0.00070) |

By build, draft-only A0 minus N matched, first 10K and first 1K:
- 97605: -0.0013 and -0.0017;
- 97650: -0.0013 and +0.0002;
- 97771: -0.0005 and -0.0004;
- 98025: +0.0011 and +0.0012.

A0 leads on the first three builds and trails on the last. Every 1K slice is within about 1 SE. All 8 slices, for every model, are in the JSON.

## 4. S-only fits (cutoff 2.55.14, the frozen N's)

Paired against frozen N (draft-only) or P3 on frozen N (player data). N's Dpre and G scores use its build-clock statistics (see section 6).

| Model | Dpre | G | Dpost | Rare (Dpost) | Onset 10K |
|---|---|---|---|---|---|
| A0 draft-only, fit S | 0.67603 (-0.00068, 0.00023) | 0.67628 (-0.00052, 0.00024) | 0.67960 (+0.00064, 0.00024) | 0.66943 (+0.00263, 0.00106) | 0.67966 (+0.00109, 0.00049) |
| K0 draft-only, fit S | 0.68519 (+0.00847, 0.00035) | 0.68507 (+0.00827, 0.00040) | 0.68737 (+0.00841, 0.00036) | 0.67862 (+0.01181, 0.00152) | 0.68798 (+0.00941, 0.00075) |
| N frozen | 0.67672 | 0.67680 | 0.67896 | 0.66681 | 0.67857 |
| A0 with player term, fit S | 0.67235 | 0.67176 | 0.67435 (+0.00986, 0.00045 vs P3) | 0.66376 | 0.67353 |
| K0 with player term, fit S | 0.68140 | 0.68048 | 0.68216 (+0.01767, 0.00053 vs P3) | 0.67269 | 0.68176 |

P3 is not scored on Dpre or G, because its three combiner weights are fit there.

## 5. What each model is

**Draft-only readable models.** Each model was refit from scratch:
- A0, K0 and the two w2 real-arm final incumbents (inc_be8151b00daa906a, inc_1a768a53b59ba3dc) were refit on the same tranches as the originals, with every player-history term removed:
  - the 180-day player-hero experience term in all four;
  - w2-K-b's 5-day practice term (p01);
  - w2-K-blinded's 4-day practice term (p02).
- The other terms re-absorb what they can. The refit loss for A0 is 0.00519 (SE 0.00025). The zeroing estimate in `baselines.md` was 0.0052.
- The fits ran through `remote.submit` like the harness's own, as jobs `fair_A0d_SDG`, `fair_K0d_SDG`, `fair_wKbd_SDG`, `fair_wKbld_SDG`, `fair_A0d_S` and `fair_K0d_S`. Disjointness was checked by the worker.

**N matched.** This is paper 2's d2c_cumprev, retrained with the same architecture and recipe (`drift2026/train_drift_wp.py`, regime "all", features "cumulative_prev"):
- Architecture: 283 to 256 to 128.
- Training: dropout 0.3; AdamW at 5e-4, weight decay 5e-3; cosine schedule, T_max 100; batch 4096; patience 25.
- Early stopping on a replay-level 2% validation slice drawn with RandomState(42).
- Seeds 42, 123 and 777, swap-symmetrized, averaged.
- Training set: every S, Dpre and G game on a 2.55 build. That is 2,267,386 games, exactly A0's fit rows minus the 7,666 S games on builds before 2.55. The recipe's features exist only on 2.55 builds.
- Features: 1,949,087 games reuse drift2026's cached rows. The other 318,299 (late uploads of S and Dpre builds, and G after 05-22) get the same features from `neural_n`'s code.
  - On 3,000 cached games the recomputed features match the cache exactly.
  - Run through the same pipeline, the frozen N reproduces paper 2's `wp_drift.npz` to 2e-7 and our earlier Dpre/G scores to 6e-8.
- Training result: validation accuracy 56.9% to 57.1%, 49 to 58 epochs, 1.2 to 1.4 minutes per seed on the 5090. The frozen seeds ran 54 to 61 epochs.

**N frozen** is unchanged: `d2c_cumprev_s{42,123,777}`, trained through 2.55.14.95918.

**P3** is paper 2's personal layer, "+CF rank 2", the configuration behind 59.70%:
- Each player-hero value is the lag-1 experience offset plus the GP posterior mean over the player's per-hero residual sums.
- A game's logit is b0 + b1 logit(wp) + b2 (team-0 sum minus team-1 sum).
- Kernels and the experience table are paper 2's, fit on window E (to 2.55.14) and not refit; we use `fix_hs_kernels_l1.npz`, the lag-1 refit.
- The combiner was refit on Dpre+G with frozen N: b = (0.0001, 1.003, 3.92). Paper 2's was (0.001, 1.000, 3.62), fit on V1.
- P3 on N matched reuses those weights. Fitting them on its in-sample WP would favour it.
- The personal state is built from N's residuals. For P3 on N matched, the residuals of S, Dpre and G games are in sample, as paper 2's were in window E.
- Private players get 0.
- Reproduction: P3 adds 0.0145 log loss to frozen N on Dpost. Paper 2 reported 0.0149 on its window, which overlaps Dpost.

## 6. The as-of contract

**P3** (`compare/p3_asof.py`). Paper 2 built state from every game dated on an earlier UTC day, including games uploaded after the scored game (audit P3-04). Here:
- A game enters the state and experience counts of game g only if its first_seen_ts is before g's game_ts. This is the harness's contract, from the snapshot's first-seen times.
- A game's own residual offset uses counts as of its own start.

fixtures.py's three invariance checks (delete a game, upload it late, flip its own label) pass on this builder, 5/5:
- the legitimate builder passes all three;
- a planted leak that enters each game at its start time is caught by all three;
- a non-triviality check also passes.

The library fixtures (`python -m training.scilearn.fixtures`) pass 17/17.

**N's statistics.**
- **The standard convention** (`neural_n.predict_drift`) gives each game of build b the statistics of all games of earlier builds. That includes games of those builds that started after the scored game, or were uploaded later.
- **The Dpost scores above are strict instead** (`n_matched.py score_strict`). A game on UTC day d uses the snapshot counts through 96881, plus only those games of builds strictly between 96881 and b that were first seen before 00:00 of day d.
  - The two conventions differ by 0.00000 to 0.00007 in log loss on every set (the `build-clock` rows in the JSON).
  - The mean absolute change in p is 7.6e-5.
- **The WP inputs to P3's state** are:
  - in-sample or build-clock N for S, Dpre and G games;
  - strict N for Dpost games.
- **Dpre and G scores of N** (section 4 only) use the build-clock convention, as in `baselines_neural.md`.

## 7. Caveats

- **P3 is not NN-free.** Its state, kernels and combiner are all built on N's residuals. It is the right contrast when player data is allowed. FRAMING 10.2 asks the readable models for an NN-free personal block, which P3 is not.
- **The SEs are paired block-bootstrap SEs** (week by player cluster, as in `baselines.md`). Rare and onset sets are small: a 1K onset slice cannot separate models closer than about 0.003.
- **The readable draft-only models have weaker calibration** (slope 0.86) than N matched (0.90). Their log-loss edge is not a calibration artifact, because their Brier score is lower too (0.24236 against 0.24264 for w2-K-b).

## 8. Compute

- **3090** (cores 24 to 35, nice 19, at most two jobs of ours at a time, under 30 GB together):
  - four refits of about 1 to 2 hours each;
  - feature building (78 s);
  - N matched scoring;
  - strict N scoring (about 25 min);
  - two P3 sweeps (2.5 min each).
- **5090 GPU:** N matched training (three seeds, about 4 minutes).
- **5090 CPU:** the two K0 draft-only refits, under 11 GB.
- **This box:** one local DB query on the research role, for 133,025 late-upload S rows. Everything else here was file assembly and the final scoring of saved predictions, under nice 19 on 4 cores.
