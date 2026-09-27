# W3(b) — Hero-level changepoint detection at build boundaries

4 boundaries between consecutive sizable builds (hotfix builds folded in). Per hero: two-proportion tests on WR (>= 200 games/side), pick rate and ban rate; BH FDR within boundary x family. Detection latency = games into the new build until the sequential test flags (z at day boundaries; SPRT with |shift|=3pp alternative, ~95% threshold).

## Detected changed-hero counts per boundary

| boundary (new build) | games | wr_p05 | wr_p05_eff | wr_bh | pickban_bh | pickban_eff2 | any_bh | any_bh_eff | truth | P / R (any_bh_eff) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.55.14.95817 | 27,178 | 12 | 12 | 3 | 84 | 17 | 84 | 75 | 18 | 0.227 / 0.944 |
| 2.55.14.95918 | 129,647 | 6 | 6 | 2 | 66 | 10 | 66 | 57 | 5 | 0.07 / 0.8 |
| 2.55.15.96477 | 102,708 | 22 | 12 | 11 | 78 | 17 | 78 | 55 | 18 | 0.236 / 0.722 |
| 2.55.16.96881 | 43,492 | 10 | 9 | 4 | 69 | 10 | 69 | 57 | 20 | 0.246 / 0.7 |

## Detection latency (WR-flagged heroes, games into the new build)

| method | median build-games to flag | p75 | n heroes |
|---|---|---|---|
| z_naive | 16,553 | 28,443 | 50 |
| z_bonf | 19,501 | 36,320 | 21 |
| sprt | 19,653 | 31,198 | 36 |

## Validation vs official patch notes

Ground truth: source: Official Blizzard patch notes via news.blizzard.com (index: https://news.blizzard.com/en-us/api/feed/heroes-of...; fetch: Plain HTTP GET, stdlib urllib. Blizzard: JSON feed API paginated with ?offset=N, then each article's HTML pars...; 59 patches; 11 corpus builds with known truth (5 with zero balance changes); bugfix-only/ARAM-only hero mentions excluded from truth

4 validatable boundaries (every folded build has known truth; zero-change maintenance builds count as known-empty). Micro-averaged:

| variant | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| wr_p05 | 21 | 29 | 40 | 0.42 | 0.344 | 0.378 |
| wr_p05_eff | 18 | 21 | 43 | 0.462 | 0.295 | 0.36 |
| wr_bh | 14 | 6 | 47 | 0.7 | 0.23 | 0.346 |
| pickban_bh | 55 | 242 | 6 | 0.185 | 0.902 | 0.307 |
| pickban_eff2 | 34 | 20 | 27 | 0.63 | 0.557 | 0.591 |
| any_bh | 55 | 242 | 6 | 0.185 | 0.902 | 0.307 |
| any_bh_eff | 48 | 196 | 13 | 0.197 | 0.787 | 0.315 |

- Median latency to flag a patch-note-listed hero (naive z): 11,144 build games (n=21).

Caveats: patch notes are not a perfect oracle — indirect effects (nerfing a counter shifts a hero's WR without any note) create principled false positives, and small note-level tweaks (e.g. a talent no one picks) are principled false negatives. Precision here is therefore a lower bound on detector quality.
