# P3 tails: combined main model on full_1017, tail priors, known-bad heroes (2026-10-18)

This is the round after `P3_RICH_CF.md`. Everything here runs on the current panel:
- store `full_1017`: 49.1M rows, 42,249 pulled players;
- the A1 slot table and merged rows of the full-history merged chain: 22.4M slots, 25.5M Heroes Profile rows.

The comparison arm is A2S from `p3_c_tests_full`. My win-only rerun of today's kernel reproduces A2S exactly (+17.68 on V2, +17.44 on OOT), so every difference below is on the same games, history and offset.

Numbers are in `results/rcf_full/`:
- `p3_c_tests_main.*` (main model against A2S);
- `p3_c_tests_tails.*` (tail cards);
- `knownbad_full_ms*.json`;
- the cover and Cassia JSONs.

## Headline

- **The combined main model beats A2S by +11.0 (SE 0.5) log-loss points (x1000) on V2 and +10.4 (SE 0.26) on OOT.** Accuracy rises from 60.35% to 62.00% on V2 and from 60.14% to 61.66% on OOT.
  - Model: `met2_hs2` structure, full scoreboard export, `a_shrink` normalization, per-player temporal layer, no role or melee features.
  - It wins in every subset: all-10 pulled +4.4 / +5.4, 1+ proven +7.8 / +6.6, Diamond +8.4 / +8.9, Master +5.2 / +7.9 (V2 / OOT).
- **Display scale with the layer:** 2.18·s² + (2.27pp)². V2 80% coverage goes from 0.783 to 0.806.
- **The tails are not a prior-shape problem.**
  - None of the six tail cards improves lift: −0.03 to −0.34 on V2 and OOT.
  - Heavier tails (Student-t, mixture) make the top-tail over-credit worse.
  - The play-volume prior is the only card that improves calibration by record: slope 0.86 → 0.91 on 50+ game cells, at a cost of −0.3 (SE 0.16) on V2. It shows two real effects:
    - specialists with 100+ games are better than the model credits (+0.9 prior sd);
    - for never-played heroes the model's prior spread is about 3 times too wide in sd.
- **About the tail pattern you quoted:** losing records realizing 1.0 to 1.7pp worse than predicted is the A1 arm, the win-only CF on corpus history. A2S itself is fine at the bottom (100+ games below −5pp: −0.29 [−0.70, +0.17]) and over-credits the top (−1.60 [−2.18, −1.11]). The main model halves the top-tail over-credit (−0.28) but carries a +0.5pp level offset at slot level, which the game-level combiner absorbs.
- **Known-bad heroes.**
  - When a player has a hero they are 3pp+ worse on than their own average (30+ games), they queue it in 4.3% of games. Those games realize −4.7pp (SE 0.34) against the population WP; the same players' other games realize −0.0pp.
  - Banning a player's main raises the chance of a known-bad pick by +1.1pp (SE 0.14) within player.
  - Banning the main is itself worth −1.76pp (SE 0.27) on that player's realized residual, within player. That is the drafting edge. It comes almost entirely from losing the comfort pick, not from the known-bad picks it forces.

## 1. Combined main model on full_1017

**Model.**
- Overall-skill column plus 2 learned factors, metric loadings shared across heroes, per-hero hero-specific variance, predictive objective.
- Fit on the full_1017 fit half (5090).
- Inputs: the full scoreboard export (`rcf1017`, normalization `a_shrink`), which reaches 1.03M Heroes Profile history rows beyond the slots.
- Plus the temporal layer. V1 chose:
  - drift half-life 730 days, k_d 10;
  - new-hero learning k_e 2.

The layer adds +1.10 (0.18) on V2 and +1.32 (0.09) on OOT on this panel: new-hero learning alone gives +1.04 / +1.24, drift alone +0.77 / +0.95.

Log-loss gain over population WP (x1000). "vs A2S" is paired by game, with SE. "Proven" is A2S's own definition: 50+ prior games on the hero and A2S posterior above +5pp.

| set | games | A2S | main, no layer | **main** | main vs A2S (SE) | `imp2` | accuracy A2S → main |
|---|---|---|---|---|---|---|---|
| V2 all | 72,474 | +17.68 | +27.55 | **+28.65** | **+10.96 (0.51)** | +19.77 | 60.35 → 62.00% |
| V2 all-10 pulled | 6,562 | +22.54 | +26.97 | +26.97 | +4.43 (1.44) | +23.05 | 60.93 → 61.08% |
| V2 1+ proven | 9,456 | +31.96 | +39.06 | +39.80 | +7.84 (1.36) | +33.30 | 61.75 → 62.10% |
| V2 3+ proven | 322 | +49.83 | +56.82 | +54.13 | +4.30 (6.57) | +47.41 | |
| V2 Diamond | 9,188 | +26.51 | +33.28 | +34.87 | +8.35 (1.37) | +28.71 | 61.81 → 62.98% |
| V2 Master | 6,477 | +20.76 | +25.56 | +25.97 | +5.21 (1.65) | +19.73 | 60.41 → 61.19% |
| OOT all | 286,016 | +17.44 | +26.48 | **+27.80** | **+10.36 (0.26)** | +19.63 | 60.14 → 61.66% |
| OOT all-10 pulled | 9,839 | +22.20 | +27.09 | +27.58 | +5.38 (1.17) | +24.33 | 60.88 → 61.90% |
| OOT 1+ proven | 32,404 | +29.90 | +34.38 | +36.48 | +6.58 (0.74) | +32.35 | 61.45 → 62.48% |
| OOT 3+ proven | 932 | +26.68 | +17.07 | +20.12 | −6.56 (4.26) | +31.57 | |
| OOT Diamond | 33,192 | +22.77 | +29.25 | +31.70 | +8.93 (0.71) | +26.16 | 60.76 → 62.08% |
| OOT Master | 29,843 | +21.38 | +28.47 | +29.27 | +7.90 (0.78) | +22.93 | 60.76 → 62.01% |

The 3+ proven sets hold a few hundred games, and their sign flips between V2 and OOT, as it did for A2S.

**Display scale (V1 fit, V2 test, 8,678 cells).** Display variance is c·s² + k0², fit as of the end of E against V1 cells. With the layer, the layer's shift as of each date is added to the cell mean before fitting.

| model | c | k0 | V2 cover80 raw → scaled | by games at the split (0 / 1-5 / 6-20 / 21-100 / 101+), scaled |
|---|---|---|---|---|
| **main (with layer)** | 2.18 | 2.27pp | 0.783 → **0.806** | 0.78 / 0.78 / 0.81 / 0.81 / 0.82 |
| main, no layer | 2.23 | 2.29pp | 0.782 → 0.808 | 0.78 / 0.79 / 0.81 / 0.81 / 0.82 |
| `imp2` | 2.19 | 0.43pp | 0.784 → 0.805 | 0.75 / 0.79 / 0.82 / 0.81 / 0.82 |

**Cold start (Cassia R²).** Main 0.30, `imp2` 0.46. By role, tank / bruiser / healer / ranged / melee / support:
- main: +0.81 / 0.11 / 0.25 / 0.29 / 0.22 / −0.02;
- `imp2`: +0.70 / 0.77 / 0.29 / 0.45 / 0.27 / 0.34.

Tanks are positive for the first time; the full-panel cell set has far more tank history.

## 2. Tails

**Method** (`p3_rcf_tails.py`).
- For each slot the walk returns, exactly, the prior for θ at the query hero from everything except that hero's own win record: N(m0, v0). It also returns the record itself, Sh / Ph.
- The Gaussian model is their product. Test R7 checks the recombination, and that the Gaussian card on the quadrature grid equals it.
- A card replaces the Gaussian prior shape on x = (θ − m0)/√v0. The posterior is computed by quadrature on 401 points.
- Each card is fit by the posterior predictive likelihood of the scored game's own residual, on 250K V1 slots. V2 and OOT are the test.
- The temporal layer is refit on top of each card, with the same grid and the same V1 choice.

| card | prior shape on x | V1 log-lik gain over Gaussian | V2 vs main (SE) | OOT vs main (SE) | V2 all-10 vs main | OOT 1+ proven vs main | R1 slope, 50+ cells | 100+ gap, < −5pp | 100+ gap, > +10pp |
|---|---|---|---|---|---|---|---|---|---|
| A2S (win-only, reference) | | | −10.96 | −10.36 | | | 0.849 | −0.29 [−0.70, +0.17] | −1.60 [−2.18, −1.11] |
| **main (Gaussian)** | N(0, 1) | 0 | 0 | 0 | 0 | 0 | 0.857 | +0.87 [+0.46, +1.35] | −0.28 [−0.85, +0.23] |
| scale (control) | N(0.18, 1.06²) | +20.0 | −0.20 (0.07) | −0.24 (0.04) | −0.15 (0.19) | −0.05 (0.10) | 0.835 | +0.64 | −0.86 |
| (a) asym, split normal | mode 0.16, sd 1.04 below / 1.07 above | +20.0 | −0.20 (0.07) | −0.24 (0.04) | −0.16 (0.19) | −0.07 (0.10) | 0.834 | +0.64 | −0.89 |
| (a) mixture | the second component collapses (π < 0.01) | +8.8 | +0.01 (0.07) | −0.03 (0.05) | +0.04 (0.26) | −0.49 (0.33) | 0.781 | +1.06 | −1.32 |
| (b) Student-t | location 0.19, scale 0.89, 7.7 df | +23.4 | −0.19 (0.09) | −0.28 (0.05) | −0.32 (0.29) | −0.54 (0.23) | 0.819 | +0.53 | −1.59 |
| **(c) play volume** | N(μ_b, s_b²) by games × specialist | **+39.2** | −0.34 (0.16) | −0.23 (0.08) | −0.50 (0.54) | +0.14 (0.27) | **0.907** | +0.76 [+0.36, +1.24] | −0.49 [−1.04, +0.01] |
| (c) play volume, t shape | t per cell | +47.7 | −0.21 (0.18) | −0.20 (0.09) | −0.48 (0.63) | −0.30 (0.37) | 0.913 | +0.76 | −0.77 |

Notes on the table:
- The R1 slope is the regression of realized on predicted over the ten 50-99 and 100+ record bins, weighted by slots. 1 means calibrated across records; below 1 means records are trusted too much.
- Gaps are realized minus predicted, in pp, with week-block CIs.
- With 2 to 20 free parameters, a pure-noise V1 gain would be about +2 to +8 (smoke test on Gaussian data). Every gain here is real, and none is large.

**R1 calibration by record, before and after** (V2+OOT slots, bins by A2S's own count and record; realized | predicted | gap [CI], pp):

| games | record | slots | realized | A2S: pred / gap | main: pred / gap | (c) volume: pred / gap | (b) t: pred / gap |
|---|---|---|---|---|---|---|---|
| 50-99 | < −5pp | 67,213 | −2.16 | −2.04 / −0.12 [−0.53, +0.28] | −3.29 / +1.13 [+0.71, +1.53] | −2.11 / −0.05 [−0.47, +0.34] | −2.86 / +0.70 |
| 50-99 | −5 to 0 | 85,259 | −0.20 | −0.05 / −0.15 | −1.32 / +1.12 | −0.37 / +0.18 | −0.87 / +0.67 |
| 50-99 | 0 to +5 | 98,729 | +0.51 | +1.43 / −0.92 | +0.11 / +0.41 | +0.85 / −0.34 | +0.51 / +0.00 |
| 50-99 | +5 to +10 | 68,569 | +1.98 | +2.89 / −0.90 | +1.46 / +0.53 | +1.98 / −0.00 | +1.87 / +0.11 |
| 50-99 | > +10 | 44,019 | +4.31 | +5.56 / −1.26 [−1.60, −0.94] | +3.81 / +0.50 [+0.14, +0.85] | +3.90 / +0.41 [+0.04, +0.74] | +4.61 / −0.31 |
| 100+ | < −5pp | 47,146 | −2.79 | −2.50 / −0.29 [−0.70, +0.17] | −3.66 / +0.87 [+0.46, +1.35] | −3.55 / +0.76 [+0.36, +1.24] | −3.32 / +0.53 |
| 100+ | −5 to 0 | 158,404 | −0.35 | +0.04 / −0.39 | −1.09 / +0.75 | −0.95 / +0.60 | −0.79 / +0.44 |
| 100+ | 0 to +5 | 282,400 | +1.94 | +2.46 / −0.52 | +1.39 / +0.55 | +1.58 / +0.36 | +1.62 / +0.32 |
| 100+ | +5 to +10 | 139,581 | +3.66 | +4.96 / −1.30 | +3.84 / −0.18 | +4.00 / −0.33 | +4.29 / −0.63 |
| 100+ | > +10 | 33,050 | +6.37 | +7.96 / −1.60 [−2.18, −1.11] | +6.65 / −0.28 [−0.85, +0.23] | +6.85 / −0.49 [−1.04, +0.01] | +7.96 / −1.59 |

Full tables, including 1-49 games and per window: `results/rcf_full/p3_c_tests_tails.txt`.

**Reading.**
- **Most of the main model's bottom-tail gap is a level offset.** Slot predictions run about 0.5pp low overall (+0.54 across 50+ cells). The game-level combiner absorbs that, because it only sees team differences. Net of the level, the main model's tilt is +0.3 at the bottom and −0.8 at the top of 100+ cells. The direction is the opposite of A1's: today it trusts strong records slightly too much in both directions.
- **(b) Heavier tails make it worse.** The Student-t trusts extreme records more, so the top 100+ cell goes back to A2S's −1.59. A second mixture component finds no support (π under 0.01), and the asymmetric prior is the same as the scale control (sd below 1.04, above 1.07). The prior is not skewed.
- **(c) Volume works on calibration, not on lift.** The fitted cells (x units, games on the hero × specialist, where specialist means the hero is 20%+ of the player's games):

  | cell | prior mean | prior sd | reading |
  |---|---|---|---|
  | never played | +0.15 | 0.35 | the spread for a never-played hero is a third of what the model claims (the first-game result of P3_HERO_STRENGTH 5b) |
  | 1-9 games | +0.16 | 1.16 | wider |
  | 1-9 games, specialist | +0.25 | 1.57 | wider |
  | 10-29 games | +0.08 | 1.21 | wider |
  | 10-29 games, specialist | +0.31 | 1.51 | wider |
  | 30-99 games | +0.36 | 0.89 | narrower |
  | 30-99 games, specialist | +0.43 | 0.76 | narrower |
  | 100+ games | −0.12 | 0.92 | narrower |
  | 100+ games, specialist | **+0.91** | 1.00 | persistence of specialists |

  Thin cells need more spread, and heavy cells need more shrinkage. A 100+ game specialist is almost a full prior sd better than the factor model's m0: someone who keeps playing a hero hundreds of times is better on it than their other heroes imply. A 100+ game non-specialist sits slightly below m0, which is the known-bad persistence direction. With volume the 50-99 row is calibrated to within ±0.4pp and the slope rises to 0.91. But game-level lift drops by 0.2 to 0.3, because the volume terms take over what the drift layer did (drift-only ablation 0.0 with volume).

**Recommendation for the tails.**
- Keep the Gaussian main model for prediction.
- Use the volume card for the displayed per-hero number, where calibration across records matters more than 0.3 log-loss points. Its never-played spread (a third of the model's) also fixes the over-wide never-played band.
- Heavy-tailed or skewed priors: no.

## 3. Known-bad heroes (descriptive, as-of)

**Setup** (`p3_knownbad.py`).
- Slots: every V1 and V2 slot with the pick order and bans recorded, 1,436,108 slots from 95,084 players. The as-of estimate for all 90 heroes comes from the main model without the layer (offset plus posterior, state through the day before).
- "Main": the player's most-played hero with 30+ games.
- "Available": not banned before the player's pick and not picked earlier in the game.
- Standard errors: player-cluster bootstrap.

**Two definitions:**
- **Absolute:** 30+ games and estimate below −3pp. This includes the player's overall level: weak players have many such heroes.
- **Hero-specific:** 30+ games and 3pp+ below the player's own mean over their 30+ game heroes. Only players with 3+ such heroes count.

| | absolute | hero-specific |
|---|---|---|
| share of all slots that are a known-bad pick | 4.39% (0.08) | 1.46% (0.03) |
| share of slots whose player has 1+ known-bad hero | 37.2% | 34.0% |
| known-bad heroes per such player | 5.6 | 2.8 |
| known-bad pick rate when the player has one | 11.8% (0.15) | 4.29% (0.07) |
| realized r = y − WP on known-bad picks | −3.81pp (0.20) | −4.69pp (0.34) |
| model's prediction on those picks | −4.65pp | −5.43pp |
| the same players' other picks | −0.16pp | −0.01pp |
| best available 30+ game alternative (model) | +2.91pp | +3.96pp |
| expected cost against that alternative | 7.6pp | 9.4pp |
| share with a better available 30+ game hero | 96% | 99.7% |

**Draft situations.** Known-bad pick rate when the player has a known-bad hero, with versus without the situation:

| situation | absolute: with / without | hero-specific: with / without | share of hero-specific known-bad picks in this situation |
|---|---|---|---|
| main banned before the pick | 14.3% (0.4) / 11.5% | 4.86% (0.16) / 4.23% | 11.3% |
| main banned by the enemy | 15.0% (0.4) / 11.6% | 4.96% (0.20) / 4.25% | 7.3% |
| main picked earlier by another player | 14.2% (0.3) / 11.6% | 5.17% (0.13) / 4.22% | 9.0% |
| three most-played heroes all unavailable | 14.6% (0.7) / 11.8% | 5.07% (0.41) / 4.29% | 0.8% |
| team's last pick | 12.6% (0.2) / 11.6% | 5.06% (0.11) / 4.09% | 24.7% |
| role fill (picked role under 10% of the player's games) | 15.8% (0.5) / 11.4% | 5.65% (0.24) / 4.16% | 11.7% |

By role, hero-specific: support 8.9%, melee 7.0%, ranged 6.7%, tank 3.0%, bruiser 2.5%, healer 2.0%. By pick position, absolute: 11.1% to 11.4% for picks 1 to 4, rising to 12.5% to 12.7% for picks 7, 9 and 10.

**What drives it.**
- Bans, lost mains, role fill and last pick each raise the rate by 15% to 35% relative.
- But known-bad picks are only slightly concentrated in these situations. Each situation's share of known-bad picks sits close to its share of eligible slots:

  | situation | share of eligible slots | share of known-bad picks |
  |---|---|---|
  | main banned | 10.0% | 11.3% |
  | role fill | 8.9% | 11.7% |
  | last pick | 20.9% | 24.7% |

- So most known-bad picks are choices, not forced: 96% to 99.7% had a better available hero by their own history.

**Main bans, within player.** Players with a known-bad hero and a main, and games both with and without the main banned before their pick, weighted per player:
- absolute definition: 7,524 players. Known-bad pick rate 14.3% with the main banned against 11.3% without, a difference of **+3.95pp (SE 0.28)**;
- hero-specific definition: 6,573 players. 4.85% against 4.04%, **+1.08pp (SE 0.14)**;
- realized residual in games where the main was banned, against the same player's other games: **−1.73pp (SE 0.27)** and −1.76pp (SE 0.27).

Note that r is measured against the population WP. That WP sees the draft, but not whose main was banned.

**Is it a drafting edge?** Yes, but not through known-bad picks.
- Banning an opponent's main costs that player about 1.7pp of realized win probability in that game, beyond what the population model sees.
- The known-bad channel explains only about 0.05pp of it: a +1.1pp rate change times about 4.7pp cost per known-bad pick.
- The rest is losing the comfort pick.
- Caveats:
  - This is observational, within player. Bans may land more in games where the opponent team reads the player as dangerous, or at times when the player is in form. Both would bias the effect toward zero or flip it, but time-varying selection is not ruled out.
  - The personalized drafter already prices this through the experience offset and the per-hero estimates (P3_DRAFTER). The number to check there is whether its ban recommendations target opponents' mains at about this value.

## Leak checks

`tests/test_rcf.py` passes 7 of 7. New in this round, R7:
- the leave-own-hero-out parts recombine to the full posterior exactly;
- the Gaussian card on the tail quadrature grid equals that posterior;
- planting same-day and later rows changes none of a query's parts.

The tail and volume cards read only those parts, plus the as-of counts (games on the hero, total games). `test_merged` and `test_full_history` are unchanged and pass. The known-bad analysis reads the same lag-1 walk, slot-row counts on earlier days, and each game's own pick order and bans, which are known at pick time.

## Files

In `training/personalization/`:
- `p3_rcf_tails.py`: the tail cards.
- `p3_knownbad.py`: section 3.
- `p3_rcf.py`, with these additions:
  - walk parts (`--parts`) and all-hero posteriors;
  - tail cards (`--tail`);
  - the record and prior variance in arm files;
  - the layer-aware display scale (`cover --layer-arm`).
- `p3_c_tests.py`: tolerates arms without a kernel field.
- `remote/run_rcf_full*.sh`: the 3090 runners. They use `P3_CACHE=cache_full`, the full_1017 merged rows and the A1 kernels.
- `results/rcf_full/`: report JSON and text, cover, Cassia and known-bad JSONs, fit metadata.
- `results/p3_c_tests_full.{txt,json}`: the merged chain's report, copied from the 3090.

**Compute.**
- 3090: cores 24-35, under 60G. The coordinator released them from paper 3.
- 5090 GPU: fits, under 11G host RAM.
- Untouched: the 3080 and the 3090's cores 0-23.
