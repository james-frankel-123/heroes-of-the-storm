# W3(b) — Hero-level changepoint detection at build boundaries

27 boundaries between consecutive sizable builds (hotfix builds folded in). Per hero: two-proportion tests on WR (>= 200 games/side), pick rate and ban rate; BH FDR within boundary x family. Detection latency = games into the new build until the sequential test flags (z at day boundaries; SPRT with |shift|=3pp alternative, ~95% threshold).

## Detected changed-hero counts per boundary

| boundary (new build) | games | wr_p05 | wr_p05_eff | wr_bh | pickban_bh | pickban_eff2 | any_bh | any_bh_eff | truth | P / R (any_bh_eff) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.55.1.87306 | 73,338 | 11 | 11 | 6 | 76 | 13 | 77 | 60 | 17 | 0.283 / 1.0 |
| 2.55.2.87774 | 39,783 | 8 | 8 | 3 | 73 | 10 | 73 | 55 | 13 | 0.236 / 1.0 |
| 2.55.2.88122 | 77,973 | 8 | 8 | 3 | 59 | 5 | 60 | 48 | 3 | 0.042 / 0.667 |
| 2.55.3.88481 | 94,940 | 6 | 2 | 0 | 65 | 5 | 65 | 36 | 0 | 0.0 / - |
| 2.55.3.88936 | 80,862 | 6 | 3 | 0 | 56 | 3 | 56 | 33 | 0 | 0.0 / - |
| 2.55.3.89566 | 25,475 | 4 | 4 | 0 | 32 | 0 | 32 | 26 | 0 | 0.0 / - |
| 2.55.3.89754 | 145,966 | 9 | 8 | 0 | 56 | 5 | 56 | 51 | 0 | 0.0 / - |
| 2.55.3.90670 | 46,683 | 8 | 6 | 0 | 49 | 2 | 49 | 32 | 0 | 0.0 / - |
| 2.55.3.91093 | 47,258 | 4 | 4 | 0 | 39 | 1 | 39 | 31 | 20 | 0.29 / 0.45 |
| 2.55.4.91418 | 71,133 | 4 | 4 | 0 | 50 | 3 | 50 | 36 | 88 | 0.972 / 0.398 |
| 2.55.4.91769 | 98,243 | 3 | 1 | 0 | 59 | 3 | 59 | 32 | 64 | 0.781 / 0.391 |
| 2.55.5.92264 | 75,379 | 11 | 9 | 4 | 54 | 7 | 54 | 36 | 77 | 0.944 / 0.442 |
| 2.55.6.92665 | 57,139 | 7 | 5 | 2 | 58 | 8 | 58 | 40 | 67 | 0.8 / 0.478 |
| 2.55.7.93054 | 23,514 | 5 | 5 | 3 | 46 | 6 | 46 | 44 | 32 | 0.409 / 0.562 |
| 2.55.7.93151 | 24,649 | 4 | 4 | 0 | 37 | 2 | 37 | 36 | 38 | 0.444 / 0.421 |
| 2.55.8.93382 | 47,961 | 7 | 7 | 1 | 47 | 3 | 47 | 45 | 40 | 0.467 / 0.525 |
| 2.55.9.93640 | 45,355 | 12 | 10 | 0 | 67 | 9 | 67 | 53 | 35 | 0.396 / 0.6 |
| 2.55.10.93810 | 85,037 | 9 | 7 | 1 | 44 | 5 | 44 | 35 | 25 | 0.343 / 0.48 |
| 2.55.10.94189 | 26,625 | 8 | 7 | 5 | 56 | 7 | 58 | 53 | 39 | 0.491 / 0.667 |
| 2.55.10.94387 | 21,923 | 5 | 5 | 1 | 64 | 7 | 64 | 61 | 7 | 0.082 / 0.714 |
| 2.55.10.94470 | 68,922 | 7 | 7 | 1 | 65 | 6 | 65 | 62 | 0 | 0.0 / - |
| 2.55.12.94786 | 95,113 | 15 | 11 | 5 | 61 | 8 | 61 | 40 | 51 | 0.675 / 0.529 |
| 2.55.13.95301 | 134,197 | 8 | 4 | 3 | 61 | 6 | 61 | 31 | 56 | 0.677 / 0.375 |
| 2.55.14.95817 | 24,727 | 10 | 10 | 3 | 84 | 19 | 84 | 77 | 18 | 0.234 / 1.0 |
| 2.55.14.95918 | 115,564 | 6 | 6 | 3 | 63 | 9 | 63 | 58 | 5 | 0.069 / 0.8 |
| 2.55.15.96477 | 80,954 | 19 | 15 | 11 | 78 | 16 | 78 | 55 | 18 | 0.236 / 0.722 |
| 2.55.16.96881 | 31,750 | 11 | 11 | 3 | 66 | 10 | 66 | 58 | 20 | 0.276 / 0.8 |

## Detection latency (WR-flagged heroes, games into the new build)

| method | median build-games to flag | p75 | n heroes |
|---|---|---|---|
| z_naive | 12,922 | 27,804 | 215 |
| z_bonf | 19,698 | 38,526 | 69 |
| sprt | 17,614 | 29,061 | 179 |

## Validation vs official patch notes

Ground truth: source: Official Blizzard patch notes via news.blizzard.com (index: https://news.blizzard.com/en-us/api/feed/heroes-of...; fetch: Plain HTTP GET, stdlib urllib. Blizzard: JSON feed API paginated with ?offset=N, then each article's HTML pars...; 59 patches; 45 corpus builds with known truth (19 with zero balance changes); bugfix-only/ARAM-only hero mentions excluded from truth

27 validatable boundaries (every folded build has known truth; zero-change maintenance builds count as known-empty). Micro-averaged:

| variant | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| wr_p05 | 98 | 117 | 635 | 0.456 | 0.134 | 0.207 |
| wr_p05_eff | 83 | 99 | 650 | 0.456 | 0.113 | 0.181 |
| wr_bh | 46 | 12 | 687 | 0.793 | 0.063 | 0.116 |
| pickban_bh | 501 | 1064 | 232 | 0.32 | 0.683 | 0.436 |
| pickban_eff2 | 104 | 74 | 629 | 0.584 | 0.142 | 0.228 |
| any_bh | 501 | 1068 | 232 | 0.319 | 0.683 | 0.435 |
| any_bh_eff | 385 | 839 | 348 | 0.315 | 0.525 | 0.393 |

- Median latency to flag a patch-note-listed hero (naive z): 9,391 build games (n=98).

Caveats: patch notes are not a perfect oracle — indirect effects (nerfing a counter shifts a hero's WR without any note) create principled false positives, and small note-level tweaks (e.g. a talent no one picks) are principled false negatives. Precision here is therefore a lower bound on detector quality.
