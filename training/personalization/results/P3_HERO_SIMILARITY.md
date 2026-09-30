# Hero similarity: skill vs play preference (2026-09-30)

"Skill similarity" asks whether a player who is good on hero A tends to be good on hero B. "Preference similarity" asks whether players who play A also play B (P3_PREF_SIMILARITY.md).

**Data.** Snapshot Storm League games from 2024-04 to 2026-05-22. Skill is the detrended residual y − WP_drift − experience offset, averaged per player×hero cell. Each cell's games are split into odd and even games in play order (game_date, then replay_id). Everything ran on 4 cores, no GPU.

## Summary

- **Pair-level skill similarity is not measurable with this data, at any threshold.**
  - Requiring 30+ games on both heroes leaves 71 of 4,005 pairs with 100+ qualifying players. The bootstrap sd per pair is 0.57 to 0.75. The whole matrix correlates with itself at r = 0.03 (total skill) and 0.22 (skill beyond general level) across two random halves of the players.
  - A pooled estimator using everyone with 4+ games on each hero covers 98% of pairs (median 733 players per pair). The sd per pair is still about 0.6, and split-half stability is 0.05.
  - The reason is noise. Hero-specific skill has sd of about 2 to 4pp, while one game's outcome has sd of about 49pp, so a cell with 30 games has reliability of roughly 0.05. The preference matrix reaches 0.98 because play counts are nearly noise-free.
- **Pooled over groups of pairs, skill similarity is measurable and follows preference.**
  - Pooled skill correlation rises steadily with preference similarity: 0.07 in the bottom preference decile, 0.33 in the top. The slope is 1.00 (0.78, 1.25).
  - Skill beyond general level shows the same climb: −0.20 to −0.03, slope 0.65 (0.43, 0.91).
  - Heroes of the same Blizzard role share more skill than cross-role pairs (0.29 vs 0.15), and the same holds for heroes in the same preference cluster (0.28 vs 0.17).
- **The learned skill embedding is honestly validated but only moderately stable.**
  - On held-out players, pooled skill correlation rises from −0.01 in the embedding's bottom decile to 0.36 in its top (slope 0.42, CI 0.31 to 0.53).
  - Refit on the other half of the players, the embedding's pair structure correlates 0.63 with the original; per-axis loadings correlate 0.74 and 0.75.
- **Skill vs preference.**
  - The skill embedding and the preference matrix correlate 0.23 to 0.27 over all 4,005 pairs.
  - Pair by pair, observed skill correlations correlate 0.16 (inverse-variance weighted) with preference, or about 0.55 after correcting for the skill matrix's low reliability.
  - Pairs high on both are clearly skill-linked on held-out players (0.42).
  - Where the two disagree, pairs the embedding calls similar but people do not co-play (0.21) share a little more held-out skill than pairs people co-play but the embedding calls dissimilar (0.10). The difference, 0.11 (−0.005, 0.28), is borderline.
- **Most interesting divergences:**
  - **Probius.** Players co-play him with other niche kits (Lost Vikings, Gall, Cho, Sgt. Hammer, Murky), but the skill embedding places his skill next to mainstream ranged and frontline heroes (Thrall, Johanna, Muradin, Valla, Falstad).
  - **The Butcher.** He is the one hero whose skill transfers significantly less to his preference neighbors than to other heroes (−0.64, CI −1.17 to −0.07).
  - **Heroes whose skill transfers more to their preference neighbors** (17 of 90 with CIs above zero): Medivh, Tyrael, Zarya, Chen, Uther, Genji, Hanzo, Maiev, Alarak, Li-Ming and others.

## 1. Pair-level skill correlation (`p3_hs_similarity.py`)

**Estimator.** For heroes h and k, among players qualifying on both, the cross-covariance is the mean of (x_hA x_kB + x_hB x_kA)/2 minus mean products, where A and B are the odd and even halves. Each hero's true-skill variance is the split-half cross product x_hA x_hB. Halves share no games, so game noise drops out and ρ = cross / √(var_h var_k) is the correlation of true skill, disattenuated without a noise model.

Two versions:

- **total:** the cell mean.
- **specific:** the half mean minus the player's same-half mean over all other heroes, i.e. skill relative to the player's general level. The centering induces a negative average correlation, about −0.1 here, so only differences between pairs carry meaning.

**Pooled estimator.** Players with 4+ games on each hero. Pair products are weighted by a_h a_k with a = min(games, 200), the inverse-variance weight for a noise-dominated product. Per-hero variances come from all of that hero's players.

| | 30+ games each, total | 30+ games each, specific | 4+ games (pooled), total | 4+ games (pooled), specific |
|---|---|---|---|---|
| players with a qualifying cell | 19,812 | 18,232 | 126,170 | 67,132 |
| pairs with 100+ qualifying players | 71 (1.8%) | 71 | 3,924 (98%) | 3,656 (91%) |
| median players per covered pair | 129 | 129 | 733 | 724 |
| median bootstrap sd of ρ | 0.75 | 0.57 | 0.65 | 0.62 |
| pairs whose 95% CI excludes 0 | 9.9% | 1.4% | 9.6% | 7.1% |
| split-half stability of the matrix (r over pairs) | 0.03 | 0.22 | 0.05 | −0.02 |

Pairs with 30+ games on both heroes, by number of players: 757 pairs have 30+ players, 349 have 50+, 71 have 100+ and 7 have 200+. No single pair's skill correlation can be reported with a useful CI. The pair-level heatmap in `fig_hero_skill_vs_pref.png` is shown for completeness and is mostly noise.

## 2. Learned embedding, validated on held-out players

The embeddings are the phase-1 kernels (overall skill + role blocks + hero + rank-r factors) and the pure rank-2 factorization from P3_SKILL_DRIFT.md. All were fit on the fit half of players, E window only, with rank chosen by held-out likelihood (rank 2 won in both families).

**Pair-level validation** (held-out players' pair ρ vs model-implied correlation, weighted r) is dominated by noise. The empirical fit-half vs held-out ceiling is 0.02. Still, the CF rank-2 kernel scores the highest of the candidates on total skill:

| model | player+role | rank 1 | rank 2 | rank 3 | rank 4 | rank 8 | pure rank 2 | preference matrix |
|---|---|---|---|---|---|---|---|---|
| weighted r | 0.08 | 0.14 | 0.17 | 0.16 | 0.15 | 0.15 | 0.15 | 0.12 |

**Pooled validation.** Pairs are grouped into deciles of the rank-2 kernel's implied correlation, and skill correlation is pooled on held-out players only:

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| model-implied corr (mean) | −0.23 | 0.00 | 0.14 | 0.24 | 0.32 | 0.39 | 0.45 | 0.51 | 0.58 | 0.70 |
| held-out pooled ρ, total | −0.01 | 0.05 | 0.06 | 0.11 | 0.10 | 0.19 | 0.20 | 0.22 | 0.35 | 0.36 |
| held-out pooled ρ, specific | −0.20 | −0.17 | −0.18 | −0.14 | −0.16 | −0.14 | −0.12 | −0.11 | −0.02 | −0.05 |

- Top minus bottom decile: total +0.37 (0.25, 0.48), specific +0.15 (0.06, 0.22).
- Slope 0.42 (total): the embedding ranks pairs correctly but overstates the spread by a bit more than 2x. Use it for ordering, not for sizes.

**Stability.** Refit on the other half of players (E window), the rank-2 structure V Vᵀ correlates 0.63 with the original. After rotation, the per-axis loadings correlate 0.74 and 0.75, and factor sds (2.1 and 2.5pp) match. The two axes and their reading (execution demand and ladder level vs straightforward kits; mainstream vs niche) replicate. Individual heroes' positions are only moderately precise.

## 3. Skill vs preference

**Preference matrix.** Recomputed with the P3_PREF_SIMILARITY definition (volume-centered log(1 + games), players with 100+ games, region-aware player key: 26,122 players). Split-half r is 0.98.

**Correlation of the two matrices** over pairs:

| comparison | r |
|---|---|
| skill embedding (rank 2, fit half) vs preference | 0.23 |
| skill embedding (held-out half refit) vs preference | 0.27 |
| CF rank-2 kernel (beyond general skill) vs preference | 0.25 |
| pair-level observed ρ (total) vs preference, inverse-variance weighted | 0.16 |
| same, corrected for the skill matrix's reliability (0.09 full-sample) | 0.55 |

**Do players who like the same heroes share skill across them?** Yes, moderately. Pooled skill correlation by decile of preference similarity (95% CIs from 200 player bootstraps):

| preference decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean preference r | −0.12 | −0.08 | −0.06 | −0.04 | −0.03 | −0.01 | 0.01 | 0.03 | 0.07 | 0.13 |
| pooled skill ρ, total | 0.07 | 0.13 | 0.13 | 0.13 | 0.14 | 0.17 | 0.16 | 0.23 | 0.25 | **0.33** (0.29, 0.37) |
| pooled skill ρ, specific | −0.20 | −0.17 | −0.15 | −0.17 | −0.15 | −0.13 | −0.13 | −0.10 | −0.08 | −0.03 |

Top minus bottom decile: total +0.26 (0.18, 0.34), specific +0.17 (0.09, 0.26).

| group | pooled skill ρ, total | pooled skill ρ, specific |
|---|---|---|
| all pairs | 0.19 (0.17, 0.22) | −0.12 |
| same Blizzard role | 0.29 (0.25, 0.34) | −0.06 |
| cross role | 0.15 (0.13, 0.18) | −0.14 |
| same fine role | 0.30 (0.27, 0.36) | −0.04 |
| same preference cluster | 0.28 (0.25, 0.34) | −0.05 |
| different preference cluster | 0.17 (0.15, 0.20) | −0.13 |

**Within the preference clusters (total skill).**

- **Tanks + Uther/Kharazim:** 0.35.
- **Hanzo / Falstad / Greymane:** 0.58, with a wide CI (0.31 to 1.0).
- **Bruisers:** 0.19.
- **Healers:** 0.31.
- **Big niche/mechanical cluster** (Genji, Zeratul, Cho, Gall, Probius, Butcher and 19 others): 0.19.
- **Mages:** 0.35.
- **Classic ranged assassins:** 0.34.
- **Pushers and odd kits:** 0.16.

The two clusters that mix the most unusual kits share the least skill.

**Where they diverge.** Groups are defined from the skill embedding (stable, pooled) and the preference matrix, then scored on held-out players:

| group of pairs | pairs | held-out pooled skill ρ, total | held-out pooled skill ρ, specific |
|---|---|---|---|
| high on both (top 20% each) | 321 | 0.42 (0.36, 0.51) | 0.03 |
| skill-similar, rarely co-played (embedding top 20%, preference bottom half) | 236 | 0.21 (0.12, 0.32) | −0.15 |
| co-played, not skill-similar (preference top 20%, embedding bottom half) | 254 | 0.10 (0.01, 0.20) | −0.13 |
| neither (bottom half both) | 1,159 | 0.04 (−0.03, 0.11) | −0.20 |

- **Co-played but skill does not transfer.** The largest gaps are all Probius and the "odd kit" heroes: Probius + Lost Vikings / Gall / Valeera / Sgt. Hammer / Cho / Murky / D.Va / Medivh, Murky + Butcher, Lost Vikings + Samuro, Butcher + Cho, Abathur + Zagara.
  - The same players pick these heroes, plausibly because they like unusual heroes.
  - Being good at one says little about being good at another: held-out pooled ρ is 0.10 against 0.42 for pairs high on both.
- **Skill transfers but rarely co-played.** Examples include Probius + Thrall / Johanna / Muradin / Falstad / Valla / Tychus / Diablo / E.T.C., Valla + Chen, Li-Ming + Chen / Rexxar, Valla / Falstad + Zarya, Muradin + Orphea, and Hanzo + Cho.
  - Held-out ρ for this group is 0.21, twice the co-played-only group. The difference, +0.11 (−0.005, 0.28), is suggestive, not established.
  - Read literally: a Probius player's skill looks more like a mainstream-hero skill set than a niche-kit one, even though Probius players rarely play mainstream heroes.
- **Per hero** (pooled total skill, the hero's 5 nearest preference neighbors vs its other heroes):
  - 17 of 90 heroes transfer significantly more skill to their preference neighbors: Medivh +0.63, Tyrael +0.59, Zarya +0.72, Chen +0.68, Uther +0.73, plus Genji, Hanzo, Maiev, Alarak, Li-Ming, Valla, Sylvanas, Gul'dan, Tychus, Falstad, Stitches and D.Va.
  - One transfers significantly less: **The Butcher** (−0.64, CI −1.17 to −0.07). His preference neighbors (Nova, Murky, Valeera and other stealth or niche heroes) share little skill with him.
  - Skill beyond general level shows the same pattern (13 positive; Qhira the only negative).

## 4. Style similarity (talents, scoreboard)

Not attempted. Pulling scoreboard and talent JSON for 10.8M player rows is a large DB read, and the question above was the priority. It would be the natural third matrix: for example, per-player hero-relative damage vs healing vs siege shares, correlated across heroes the same way.

## Takeaways for the paper and product

1. Hero×hero skill similarity cannot be read pair by pair from ranked data. Pair correlations need thousands of players with hundreds of games on both heroes. Only group-level or model-based (embedding) statements are supported.
2. Preference is a cheap, stable proxy for where skill transfers. Moving from the bottom to the top preference decile doubles to quadruples the pooled skill correlation. The proxy fails for niche-kit heroes (Probius, The Butcher, Lost Vikings, Gall, Murky).
3. For cold-start recommendations ("you might be good at X"), the validated skill embedding is the right tool and preference is a reasonable prior. Present the result as a direction ("heroes like the ones you are good at"), not as a number for a specific hero.

## Files (training/personalization/)

- `p3_hs_similarity.py`: pair-level estimator at 30+ and 4+ games, bootstrap CIs, coverage, split-half stability, embedding validation (pair level), skill vs preference pair comparison, figure `fig_hero_skill_vs_pref.png` (preference heatmap | skill heatmap | per-pair scatter, same hero order). Outputs `results/p3_hs_similarity.json/.txt`, `cache/hs_similarity.npz`.
- `p3_hs_similarity_pool.py`: pooled group estimates (role, preference deciles, embedding deciles on held-out players, preference clusters, per-hero neighbors), embedding refit on the held-out half, figure `fig_hero_skill_vs_pref_pooled.png`. Outputs `results/p3_hs_similarity_pool.json/.txt`.
- `p3_hs_similarity_diverge.py`: divergence groups scored on held-out players, with example pairs. Outputs `results/p3_hs_similarity_diverge.json/.txt`.
