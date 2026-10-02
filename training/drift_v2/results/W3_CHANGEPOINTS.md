# W3(b) — Hero-level changepoint detection at build boundaries

27 boundaries between consecutive sizable builds (hotfix builds folded in). Per hero: two-proportion tests on WR (>= 200 games/side), pick rate and ban rate; BH FDR within boundary x family. Detection latency = games into the new build until the sequential test flags (z at day boundaries; SPRT with |shift|=3pp alternative, ~95% threshold).

## Detected changed-hero counts per boundary

| boundary (new build) | games | wr_p05 | wr_p05_eff | wr_bh | pickban_bh | pickban_eff2 | any_bh | any_bh_eff | truth | P / R (any_bh_eff) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.55.1.87306 | 73,337 | 11 | 11 | 6 | 76 | 13 | 77 | 60 | 17 | 0.283 / 1.0 |
| 2.55.2.87774 | 39,771 | 8 | 8 | 3 | 73 | 10 | 73 | 55 | 13 | 0.236 / 1.0 |
| 2.55.2.88122 | 77,960 | 8 | 8 | 3 | 60 | 5 | 61 | 49 | 3 | 0.041 / 0.667 |
| 2.55.3.88481 | 94,918 | 6 | 2 | 0 | 65 | 5 | 65 | 36 | 0 | 0.0 / - |
| 2.55.3.88936 | 80,854 | 6 | 3 | 0 | 56 | 3 | 56 | 33 | 0 | 0.0 / - |
| 2.55.3.89566 | 25,447 | 4 | 4 | 0 | 32 | 0 | 32 | 27 | 0 | 0.0 / - |
| 2.55.3.89754 | 145,863 | 10 | 9 | 0 | 56 | 5 | 56 | 51 | 0 | 0.0 / - |
| 2.55.3.90670 | 46,635 | 8 | 6 | 0 | 48 | 2 | 48 | 32 | 0 | 0.0 / - |
| 2.55.3.91093 | 47,254 | 4 | 4 | 0 | 39 | 1 | 39 | 31 | 20 | 0.258 / 0.4 |
| 2.55.4.91418 | 70,988 | 4 | 4 | 0 | 50 | 3 | 50 | 36 | 88 | 0.972 / 0.398 |
| 2.55.4.91769 | 98,171 | 3 | 1 | 0 | 60 | 3 | 60 | 31 | 64 | 0.774 / 0.375 |
| 2.55.5.92264 | 75,313 | 11 | 9 | 4 | 54 | 7 | 54 | 36 | 77 | 0.944 / 0.442 |
| 2.55.6.92665 | 57,075 | 7 | 5 | 2 | 58 | 8 | 58 | 41 | 67 | 0.78 / 0.478 |
| 2.55.7.93054 | 23,480 | 4 | 4 | 3 | 45 | 6 | 45 | 43 | 32 | 0.395 / 0.531 |
| 2.55.7.93151 | 24,623 | 4 | 4 | 0 | 40 | 2 | 40 | 39 | 38 | 0.513 / 0.526 |
| 2.55.8.93382 | 47,861 | 7 | 7 | 1 | 47 | 3 | 47 | 46 | 40 | 0.457 / 0.525 |
| 2.55.9.93640 | 45,181 | 12 | 10 | 0 | 66 | 9 | 66 | 53 | 35 | 0.396 / 0.6 |
| 2.55.10.93810 | 85,019 | 9 | 8 | 1 | 45 | 5 | 45 | 35 | 25 | 0.343 / 0.48 |
| 2.55.10.94189 | 26,566 | 8 | 8 | 5 | 56 | 7 | 58 | 54 | 39 | 0.5 / 0.692 |
| 2.55.10.94387 | 21,877 | 5 | 5 | 1 | 65 | 7 | 65 | 62 | 7 | 0.081 / 0.714 |
| 2.55.10.94470 | 68,859 | 9 | 9 | 1 | 66 | 6 | 66 | 62 | 0 | 0.0 / - |
| 2.55.12.94786 | 94,962 | 15 | 11 | 5 | 61 | 8 | 61 | 40 | 51 | 0.675 / 0.529 |
| 2.55.13.95301 | 133,928 | 8 | 5 | 3 | 61 | 6 | 61 | 34 | 56 | 0.706 / 0.429 |
| 2.55.14.95817 | 24,690 | 10 | 10 | 3 | 83 | 19 | 83 | 77 | 18 | 0.234 / 1.0 |
| 2.55.14.95918 | 115,134 | 6 | 6 | 3 | 63 | 10 | 63 | 58 | 5 | 0.069 / 0.8 |
| 2.55.15.96477 | 79,296 | 17 | 14 | 11 | 78 | 16 | 78 | 54 | 18 | 0.241 / 0.722 |
| 2.55.16.96881 | 30,267 | 11 | 11 | 4 | 64 | 9 | 64 | 57 | 20 | 0.263 / 0.75 |

## Detection latency (WR-flagged heroes, games into the new build)

| method | median build-games to flag | p75 | n heroes |
|---|---|---|---|
| z_naive | 13,757 | 28,349 | 215 |
| z_bonf | 20,383 | 38,857 | 68 |
| sprt | 17,638 | 29,960 | 182 |

## Validation vs official patch notes

Ground truth: source: Official Blizzard patch notes via news.blizzard.com (index: https://news.blizzard.com/en-us/api/feed/heroes-of...; fetch: Plain HTTP GET, stdlib urllib. Blizzard: JSON feed API paginated with ?offset=N, then each article's HTML pars...; 59 patches; 45 corpus builds with known truth (19 with zero balance changes); bugfix-only/ARAM-only hero mentions excluded from truth

27 validatable boundaries (every folded build has known truth; zero-change maintenance builds count as known-empty). Micro-averaged:

| variant | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| wr_p05 | 99 | 116 | 634 | 0.46 | 0.135 | 0.209 |
| wr_p05_eff | 86 | 100 | 647 | 0.462 | 0.117 | 0.187 |
| wr_bh | 46 | 13 | 687 | 0.78 | 0.063 | 0.116 |
| pickban_bh | 502 | 1065 | 231 | 0.32 | 0.685 | 0.437 |
| pickban_eff2 | 105 | 73 | 628 | 0.59 | 0.143 | 0.231 |
| any_bh | 502 | 1069 | 231 | 0.32 | 0.685 | 0.436 |
| any_bh_eff | 389 | 843 | 344 | 0.316 | 0.531 | 0.396 |

- Median latency to flag a patch-note-listed hero (naive z): 9,575 build games (n=99).

Caveats: patch notes are not a perfect oracle — indirect effects (nerfing a counter shifts a hero's WR without any note) create principled false positives, and small note-level tweaks (e.g. a talent no one picks) are principled false negatives. Precision here is therefore a lower bound on detector quality.
