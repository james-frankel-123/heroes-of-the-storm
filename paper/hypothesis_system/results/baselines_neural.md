# N contrast: frozen neural WP scored on Dpre, G and Dpost

Date: 2026-10-13. Machine-readable numbers: `baselines_neural.json` (same directory). Code: `training/scilearn/baselines/neural_n.py`. Per-game predictions: `/home/max/scilearn-private/fits/neural_n/` (private, out of git). No 2.57 row and no sealed replay ID was read.

## Headline (Dpost, 173,394 games, out of time for both models)

| Model | Log loss (95% CI) | Accuracy | Brier | Cal. slope | Cal. intercept |
|---|---|---|---|---|---|
| n_drift (drift-aware MLP, N) | 0.67896 (0.67808, 0.67987) | 56.82% | 0.24299 | 0.882 | +0.005 |
| p1_site_enriched (paper-1 leak-free enriched) | 0.68166 (0.68082, 0.68249) | 56.21% | 0.24431 | 0.847 | +0.005 |

n_drift beats p1_site_enriched by 0.0027 log loss (paired bootstrap CI 0.0022 to 0.0032). Both models are overconfident on 2.55.17 (slope below 0.9). The gate metric for G1 is the n_drift row.

## All tranches

| Tranche | n | Model | Log loss (95% CI) | Acc. | Brier | Slope | Intercept | Status |
|---|---|---|---|---|---|---|---|---|
| Dpre | 173,764 | n_drift | 0.67672 (0.67588, 0.67756) | 57.29% | 0.24192 | 0.986 | -0.001 | out of time |
| Dpre | 173,764 | p1_site_enriched | 0.67072 (0.66990, 0.67151) | 58.59% | 0.23903 | 1.221 | -0.001 | **in sample** |
| G | 155,191 | n_drift | 0.67680 (0.67590, 0.67771) | 57.11% | 0.24197 | 0.978 | +0.001 | out of time |
| G | 155,191 | p1_site_enriched | 0.67777 (0.67691, 0.67861) | 56.88% | 0.24245 | 0.980 | +0.001 | 12,337 games in sample |
| Dpost | 173,394 | n_drift | 0.67896 (0.67808, 0.67987) | 56.82% | 0.24299 | 0.882 | +0.005 | out of time |
| Dpost | 173,394 | p1_site_enriched | 0.68166 (0.68082, 0.68249) | 56.21% | 0.24431 | 0.847 | +0.005 | out of time |

**In-sample warning.** p1_site_enriched was trained on the paper-1 snapshot (replay_id up to 63,653,039, played through 2026-05-22). That covers 131,359 of the 173,764 Dpre games (126,287 in its training split, the rest in its test and validation splits) and 12,337 G games. Its frozen statistics also contain those games. Its Dpre numbers (58.59%, slope 1.22) are in sample and must not be used. On the 42,405 Dpre games uploaded after the snapshot it scores 0.67818 and 57.21%, against n_drift's 0.67730 and 57.30%. Those games come from the same era as its training data, so they are a held-out check, not an out-of-time one. On the 142,854 G games outside the snapshot it scores 0.67846 and 56.75%, against n_drift's 0.67677 and 57.13%.

n_drift's weights stop at build 2.55.14.95918 (2026-02-10), so all three tranches are out of time for it. Its statistics for each build use only earlier builds. One caveat: d2c_cumprev was chosen as paper 2's champion partly by its accuracy on 2.55.15 and 2.55.16, which overlap Dpre and early G. That is a mild selection effect on those two tranches, not a leak into the weights.

## Per build (accuracy / log loss)

| Build | n | n_drift | p1_site_enriched |
|---|---|---|---|
| 2.55.15.96370 | 18,707 | 56.83% / 0.67861 | 58.69% / 0.67097 (in sample) |
| 2.55.15.96477 | 105,959 | 57.35% / 0.67627 | 58.65% / 0.67059 (in sample) |
| 2.55.16.96881 | 45,008 | 57.37% / 0.67651 | 58.45% / 0.67060 (in sample) |
| 2.55.16.97039 (G) | 155,191 | 57.11% / 0.67680 | 56.88% / 0.67777 |
| 2.55.17.97605 | 9,473 | 56.95% / 0.67982 | 55.74% / 0.68333 |
| 2.55.17.97650 | 48,007 | 56.77% / 0.67957 | 56.57% / 0.68131 |
| 2.55.17.97771 | 70,628 | 56.85% / 0.67888 | 55.96% / 0.68207 |
| 2.55.17.98025 | 45,286 | 56.81% / 0.67824 | 56.34% / 0.68103 |

Three small Dpre builds (96443, 96846, 96870; 4,090 games in total) are in the JSON.

## OOD slices on Dpost

**O5, rare compositions.** A game counts as rare if either team's role multiset holds at most 0.2% of S teams. Roles come from `src/lib/data/hero-roles.ts`. S has 3,892,194 teams, and 220 of its 245 multisets are rare; together they make up 3.4% of S teams. 12,544 Dpost games (7.2%) qualify.

| Model | Log loss (95% CI) | Acc. | Slope | Intercept |
|---|---|---|---|---|
| n_drift | 0.66682 (0.66239, 0.67103) | 59.22% | 0.940 | -0.002 |
| p1_site_enriched | 0.66878 (0.66463, 0.67272) | 58.75% | 0.928 | -0.001 |

The pair-count part of O5 (a within-team pair with fewer than 100 training games) was not computed.

**O2, patch onset.** The first 1K and 10K games of each 2.55.17 build, ordered by game date. Both models were frozen before 2.55.17. Build 97605 has only 9,473 games, so its 10K slice is the whole build.

| Build | Slice | n_drift LL / acc / slope | p1_site_enriched LL / acc / slope |
|---|---|---|---|
| 97605 | first 1K | 0.68522 / 56.5% / 0.72 | 0.68797 / 54.8% / 0.66 |
| 97605 | first 10K | 0.67982 / 57.0% / 0.87 | 0.68333 / 55.7% / 0.80 |
| 97650 | first 1K | 0.67518 / 59.3% / 0.95 | 0.67792 / 59.1% / 0.94 |
| 97650 | first 10K | 0.67853 / 57.0% / 0.89 | 0.68027 / 56.7% / 0.88 |
| 97771 | first 1K | 0.67840 / 55.2% / 0.86 | 0.68279 / 55.6% / 0.79 |
| 97771 | first 10K | 0.67825 / 56.4% / 0.89 | 0.68249 / 55.8% / 0.81 |
| 98025 | first 1K | 0.67392 / 57.5% / 1.05 | 0.67767 / 56.2% / 0.98 |
| 98025 | first 10K | 0.67775 / 57.0% / 0.91 | 0.68132 / 56.5% / 0.87 |

1K slices carry a log-loss CI about ±0.012 wide, so they cannot separate the models. Log-loss CIs and the paired differences for every slice are in the JSON.

## What N is here, and what it is not

- **n_drift** = `training/drift2026/models/d2c_cumprev_s{42,123,777}.pt`, the mean of three seeds, each swap-symmetrized.
  - Features are cumulative statistics through the previous build, on the legacy research tiers.
  - For builds up to 2.55.16.96881 they come from `drift2026/patch_stats/cumulative/<previous build>.json.gz`, which holds snapshot games only.
  - For later builds they are the snapshot counts through 96881 plus every DB game of the builds strictly between 96881 and the scored build. This is the `personalization/p3_x_wp_post.py` convention behind the 56.94% figure.
  - Check: on 2,000 G games the code reproduces `personalization/cache/x_wp_post.npz` to 9e-8. On 2.55.17 it differs by 0.001 on average, because more games of the earlier builds have been uploaded since Sept 30.
- **p1_site_enriched** = `training/paper1_revision/site/models/enriched_s42.pt`, the selected seed.
  - It uses frozen `site/cache/stats/deploy.json` and `deploy_compositions.json` (site tiers).
  - Features are re-extracted for each team order, as in `paper1_revision/core.featurize`.
  - Check: it reproduces predictions from the cached `T17.npz` features to 6e-8.
- **Tiers.** Both schemes are derived from league_tier and avg_mmr.
  - Legacy: 2 is low, 3 and 4 are mid, 5 and 6 are high, missing is mid. This matches the 2026-09-30 backup table on every 2.55.1x row.
  - Site scheme: games with neither rank nor MMR are "unknown". There are 6,918, 14,453 and 4,234 of them in Dpre, G and Dpost. p1_site_enriched never saw this tier, so it scores them as mid. Dropping them changes no conclusion; see `site_tier_known` in the JSON.
- **The P3 personal layer is not included.** FRAMING cites 59.70% for it, but P3 needs per-player residual state from the numba pipeline in `training/personalization` (GB-scale slot caches). It does not cache per-game outputs, and FRAMING section 2 requires it to be rebuilt NN-free. The numbers above are N's population MLP only.
- **Comparison with FRAMING section 1.** FRAMING quotes 56.94% and slope 0.93 for June to Sept 2026. Here n_drift scores 56.82% and slope 0.88 on Dpost alone, and 57.11% and 0.98 on G. This fits P3_EXTENSIONS, where the population WP drifts to a slope of 0.89 on 2.55.17.

## Compute

Rows were extracted locally (one light query, nice 19). Scoring ran on max-windows-3090 as jobs `scilearn_n_{Dpre,G,Dpost}_r2`: CPU only, cores 24 to 35, nice 19, at most 18 GB each. Each job took 1.5 to 2.5 minutes.
