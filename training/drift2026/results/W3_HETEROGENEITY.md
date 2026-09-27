# W3(f) — Drift and recovery heterogeneity by skill tier and map

Cross-build stability (r_adj, disattenuated; raw r in parens where reliability is too low) at build-lags 1/3/8 over the 28 sizable builds, plus within-build games to 1pp hero-WR recovery (vs final, median across builds; day-boundary resolution).

Reading note: recovery is measured in SLICE-local games against the slice's own final values, over the slice's own eligible key set — compare recovery numbers within a table, not against the pooled W3_RECOVERY row (fewer keys pass the games minimum in a slice, and a fixed game count is a larger fraction of a slice, which flatters the vs-final metric mechanically).

## By skill tier

| tier | games/build (med) | hero WR @1/3/8 | pick rate @1/3/8 | comp WR @1/3/8 | pair-with @1/3/8 | recovery to 1pp (games) |
|---|---|---|---|---|---|---|
| low | 14,013 | 0.923 / 0.897 / 0.827 | 0.997 / 0.988 / 0.974 | 0.968 / 0.959 / 0.980 | 0.906 / 0.870 / 0.741 | 6,460 (28/28) |
| mid | 33,738 | 0.940 / 0.871 / 0.795 | 0.986 / 0.967 / 0.942 | 0.979 / 0.963 / 0.969 | 0.946 / 0.870 / 0.795 | 8,668 (28/28) |
| high | 22,219 | 0.930 / 0.867 / 0.763 | 0.981 / 0.959 / 0.924 | 0.957 / 0.886 / 0.904 | 0.919 / 0.853 / 0.738 | 7,753 (28/28) |
| (pooled) | - | 0.941 / 0.878 / 0.783 | 0.984 / 0.965 / 0.939 | 0.983 / 0.944 / 0.952 | - | - |

## By map (hero-per-map WR)

| map | slice games/build (med) | map-hero WR @1/3/8 | recovery to 1pp (games) |
|---|---|---|---|
| Alterac Pass | 6,153 | 0.871 / 0.813 / 0.830 | 4,249 (28/28) |
| Battlefield of Eternity | 6,161 | 0.908 / 0.817 / 0.749 | 3,938 (28/28) |
| Blackheart's Bay | 6,378 | 1.000 / - / - | 3,465 (2/2) |
| Braxis Holdout | 6,126 | 0.908 / 0.897 / 0.855 | 4,127 (25/25) |
| Cursed Hollow | 6,231 | 0.931 / 0.883 / 0.846 | 3,821 (28/28) |
| Dragon Shire | 6,185 | 0.934 / 0.902 / 0.887 | 4,102 (28/28) |
| Garden of Terror | 6,131 | 0.855 / 0.824 / 0.730 | 3,885 (28/28) |
| Hanamura Temple | 6,347 | 0.933 / 0.849 / - | 4,182 (9/9) |
| Infernal Shrines | 6,170 | 0.912 / 0.848 / 0.805 | 4,091 (28/28) |
| Sky Temple | 6,201 | 0.946 / 0.905 / 0.863 | 4,037 (28/28) |
| Tomb of the Spider Queen | 6,168 | 0.924 / 0.875 / 0.882 | 3,872 (28/28) |
| Towers of Doom | 6,113 | 0.887 / 0.882 / 0.799 | 4,022 (28/28) |
| Volskaya Foundry | 5,157 | 0.919 / 0.883 / 0.754 | 3,376 (19/19) |
| Warhead Junction | 6,324 | 0.978 / 0.971 / - | 4,150 (10/10) |
| (pooled hero WR ref) | - | 0.941 / 0.878 / 0.783 | see W3_RECOVERY |
