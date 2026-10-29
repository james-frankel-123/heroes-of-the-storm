# P3 numbers for the paper 2 draft (def1027 chosen model, 2026-10-29)

All computed on the main box, single-threaded BLAS, from the pinned def1027 arm files and fresh state-writer answers. Raw numbers are in `results/p3_paper2_extras.json`; the script is `p3_paper2_extras.py`. Gains are log-loss points (x1000) over population WP unless stated.

## 1. Draft-only accuracy on the post-freeze set

The post-freeze set is 83,955 complete 2.57 games (2026-09-28 to 2026-10-26), scored once with no refit (P3_POSTFREEZE.md).

| model | accuracy | edge over a coin flip | log-loss gain over a coin flip x1000 |
|---|---|---|---|
| draft only (population WP) | 57.17% | +7.17pp | +16.68 |
| today's kernel | 60.08% | +10.08pp | +32.95 |
| previous main model | 61.54% | +11.54pp | +42.31 |
| chosen model | 62.59% | +12.59pp | +50.15 |

On this set, the players add 5.42pp of accuracy to the draft alone (+33.47 log-loss points over population WP).

## 2-3. Stale and lagged player states (OOT, 286,016 games)

Each slot's player state is taken as of L days before the game, instead of the day before. Everything else is the model as run: the draft and population WP, the at-game MMR term, and the chosen arm's combiner. With L = 1 the run reproduces the arm, max difference 3e-08. Each lagged state is a fresh state-writer answer, so it carries the counts, layer and scoreboard channel of that earlier day.

| player state | gain over population WP | cost vs current (game SE) | week-block 95% CI | accuracy |
|---|---|---|---|---|
| current (day before the game) | +37.38 | | | 62.85% |
| 1 week old | +27.93 | 9.45 (0.22) | 8.84 to 10.08 | 61.69% |
| 1 month old | +17.36 | 20.02 (0.34) | 19.19 to 20.89 | 60.47% |
| 4 months old | +9.18 | 28.20 (0.42) | 27.20 to 29.12 | 59.33% |
| 1 year old | +3.89 | 33.49 (0.47) | 32.63 to 34.27 | 58.49% |

- A one-year-old state keeps about a tenth of the gain.
- Freshness matters a lot. Even a one-week-old state loses a quarter of the gain, and a one-month-old state falls below today's kernel run on a current state (+18.49).
- Much of the loss comes from players and heroes the old state has never seen; the breakdown below shows how much.

**One-year breakdown.** The cost when only one group of slots is given its one-year-old state, the rest current:

| what the one-year-old state is missing | slot share | cost x1000 (SE) |
|---|---|---|
| new player (no games at all a year earlier) | 23.4% | 17.68 (0.33) |
| new hero (player known, hero never played a year earlier) | 15.9% | 7.33 (0.24) |
| known pair (player had played the hero) | 60.7% | 9.48 (0.26) |

The three parts sum to 34.50 against 33.49 for all slots at once, because the costs interact a little through the game's team difference. New players carry about half of the one-year cost from under a quarter of the slots. Known player-hero pairs still lose 9.5 points over a year.

## 3. Gains by region and by tier

Chosen model vs previous main model and today's kernel (gain over population WP x1000; accuracy of the chosen model and of the draft alone):

| window | group | games | chosen | previous | today | chosen − previous (SE) | acc chosen | acc draft only |
|---|---|---|---|---|---|---|---|---|
| V2 | Americas | 22,332 | +41.6 | +32.0 | +20.9 | +9.7 (0.8) | 64.0% | 57.4% |
| V2 | Europe | 46,820 | +34.8 | +28.3 | +18.9 | +6.6 (0.6) | 62.8% | 57.1% |
| V2 | Asia | 3,322 | +26.4 | +24.3 | +13.6 | +2.1 (2.3) | 61.7% | 56.6% |
| OOT | Americas | 91,555 | +41.8 | +30.5 | +20.3 | +11.3 (0.4) | 63.2% | 56.8% |
| OOT | Europe | 180,669 | +36.4 | +28.1 | +18.4 | +8.3 (0.3) | 62.8% | 57.0% |
| OOT | Asia | 13,792 | +21.0 | +16.7 | +8.0 | +4.4 (1.1) | 61.2% | 57.5% |

| window | tier | games | chosen | previous | today | chosen − previous (SE) | acc chosen | acc draft only |
|---|---|---|---|---|---|---|---|---|
| V2 | Bronze | 13,140 | +30.1 | +20.1 | +8.8 | +10.0 (1.2) | 62.8% | 58.0% |
| V2 | Silver | 15,489 | +29.1 | +25.1 | +14.2 | +4.0 (1.0) | 62.4% | 57.3% |
| V2 | Gold | 14,771 | +39.9 | +33.7 | +21.5 | +6.3 (1.0) | 63.4% | 57.0% |
| V2 | Platinum | 13,409 | +43.1 | +35.1 | +24.9 | +7.9 (1.0) | 63.4% | 56.9% |
| V2 | Diamond | 9,188 | +44.7 | +35.4 | +28.3 | +9.3 (1.2) | 64.2% | 57.2% |
| V2 | Master | 6,477 | +34.4 | +26.5 | +22.8 | +7.9 (1.5) | 62.7% | 56.3% |
| OOT | Bronze | 52,738 | +32.7 | +21.9 | +11.5 | +10.8 (0.6) | 62.4% | 56.9% |
| OOT | Silver | 59,442 | +30.3 | +23.6 | +14.7 | +6.8 (0.5) | 61.9% | 56.6% |
| OOT | Gold | 59,666 | +38.6 | +30.4 | +18.7 | +8.2 (0.5) | 62.9% | 57.1% |
| OOT | Platinum | 51,135 | +44.3 | +34.8 | +23.9 | +9.5 (0.5) | 63.7% | 57.1% |
| OOT | Diamond | 33,192 | +43.0 | +32.4 | +24.3 | +10.6 (0.6) | 63.4% | 57.2% |
| OOT | Master | 29,843 | +39.2 | +29.4 | +22.3 | +9.8 (0.7) | 63.4% | 56.7% |

- Region is the game's server region: Americas, Europe, Asia. No Chinese-server games are in the corpus windows.
- Tier is the game's real league tier (`p3_c_tests.real_tier`). Games without a known tier are left out of the tier table.
- The chosen model beats the previous one in every region and tier.
  - The gain is smallest in Asia (fewest games, thinnest histories) and in Silver.
  - The players' value is largest in Platinum and Diamond, at +43 to +44 points over population WP.

## 4. New accounts

**No: the def1027 chosen model has no new-account starting estimate.**

- **How a new account is scored.** An account with no history gets what any player gets at zero games:
  - the experience offset for (0 games seen, 0 on the hero), from the E-fit experience table (`p3_hs_core.exp_bins` and `fit_experience`; built by `p3_c_merged.merged_predict`; stored as `table` in `cache_def_def1027/rcf/aug_merged.npz`);
  - then the shrinkage function's offset term (`beta_offset` in the arm's tail meta);
  - then the direct channel at zero evidence, and the smooth calibration.
- **What is not in the chain.** The account-status prior is not used anywhere in the def1027 chain: `p3_hero_level_causal.account_status`, which separates genuinely new accounts from old accounts new to the corpus, and the status-split experience table. Those exist only as the P3 extensions study (`p3_x_newacct.py`, P3_EXTENSIONS.md section 4). There they gained +0.0024 log loss per game on V2 and +0.0021 out of time, almost all from the 18% of games with a genuinely new account. The same file lists it as an extension, not part of the paper's model.
- **On the site.** Accounts the def1027 model has never seen (only post-cap games) get no card at all: 24,278 accounts dropped in the 2026-10-28 publish.
