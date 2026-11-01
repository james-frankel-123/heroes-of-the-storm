# Weekly refresh of the player cards (2026-11-01)

The site's skill estimates were the def1027 states as of 2026-09-27. The weekly refresh keeps them current without refitting: the def1027 chosen model, every coefficient frozen, walked over all games through yesterday.

## What runs (`run_weekly_refresh.sh`, main box, nice 19, cores 44-47, single-threaded BLAS)

| stage | what | runtime | peak RAM |
|---|---|---|---|
| fetch | post-cap 2.57 games from the local DB, appended to def1027's pinned post-snapshot files | 50 s | 2.5 GB |
| wp | population WP for them (frozen d2c models; pinned games keep their WP exactly) | 62 s | 4.3 GB |
| export | def1027's exports byte for byte plus the post-cap rows | 45 s | 0.1 GB |
| store | merged history store, cap = today | 249 s | 14.1 GB |
| a1 | slot table, MMR at game, merged rows, cards, aug caches | 1,204 s | 15.1 GB |
| score | today's kernel and the chosen model, frozen (`P3_RCF_FROZEN`) | 1,443 s | 17.7 GB |
| checks | regression (arm and states), join and changed-account lists | ~280 s | 18.9 GB |
| cards | site build, selfcheck, states, encode, check | 200 s | 19.2 GB |
| refit check | reports only | 2 s | 0.1 GB |

**Total about 59 minutes; peak RAM 19 GB.** No GPU is needed.

Every post-cap path excludes the 5,000 sealed replays (`p3_sealed.py`). Each stage runs once per build; a rerun resumes after the last finished stage.

## First build: wk261101 (cap 2026-11-01, games through 2026-10-31)

**Regression against def1027 on pre-cap days:**
- **Arm:** slot predictions match within 2.5e-08 and game probabilities within 2.9e-08 on 330,166 games and 4.17M slots. The frozen coefficients are identical.
- **Published site states as of 2026-09-27:** reproduced for 164,889 accounts. Counts, base sd and band are exact; estimates agree within 1.5e-08 (float32 rounding).
- **Left out of the comparison:**
  - 606 accounts whose def1027 scoreboard rows were joined to another game (bug below; the weekly build uses the corrected join);
  - accounts with new Heroes Profile history (none this week);
  - 918 games with a duplicated MMR stamp key, reported separately.

**State-writer selfcheck:** max difference from the arm 1.5e-08. The card check passed: band against the anchored_v1 formula 1.5e-08; card values within their quantization.

**Cards:**

| | def1027 cards (published) | wk261101 (staged) |
|---|---|---|
| estimates as of | 2026-09-27 | 2026-10-31 |
| accounts carded | 165,433 | 188,806 |
| published (3+ recent games) | 123,965 | 136,667 |
| accounts with games but no card | 24,278 | 2,893 |

Of the 24,278 accounts without a card in the def1027 publish, 21,546 now get one.

**Post-freeze sanity (games 2026-10-04 to 10-31, 78,855 games, frozen coefficients):**
- chosen model +33.64 over population WP and +17.07 over today's kernel (OOT: +37.38 and +18.89);
- accuracy 62.65%, against 57.17% for the draft alone;
- game calibration slope 0.956 (week-block 95% CI 0.916 to 0.983).

## Refit rule (`p3_weekly.py refit-check`, reports only)

Refit the coefficients (a new definitive build) when any of these holds on the last 28 days of post-cap games:

| rule | trigger | this week |
|---|---|---|
| R1 | the game-level calibration slope's 95% CI lies entirely outside [0.9, 1.1] | 0.956 [0.916, 0.983]: no |
| R2 | the gain over today's kernel falls below half its OOT value (+18.89) | +17.07: no |
| R3 | the coefficients are older than 92 days (quarterly) | 5 days: no |

Verdict: keep frozen coefficients.

- **R1** catches a patch or meta shift that miscalibrates the probabilities. The current slope already sits below 1, so a second patch could trip it.
- **R2** catches the player model going stale relative to the simplest kernel.
- **R3** bounds drift that neither of the others sees.

One known gap: Xal'atath and Haunted Mines are not in the 90-hero v1 set, so games with Xal'atath stay out of the slot table until a refit moves to the v2 hero set.

## New accounts (`sync/fetch-mmr-history.ts --add-recent`)

Once a week the MMR-history pull appends a panel tranche (`panelN.tsv`): accounts from the last 7 days of corpus games that are in no panel yet, most games first, as many as fit in the week's remaining budget. The budget is 95% of the weekly MMR budget, minus the rolling week's calls and any unfetched panel rows, at about 1 call per account. Sealed replays are excluded at the query.

Each run is logged to `cache/mmr_history/recent_adds.jsonl` and the run log. Heroes Profile is called offline only, never from the site.

A preview on 2026-11-01 found 41,151 accounts in the last 7 days. Of these, 18,509 are in no panel, and all would fit (room 31,329).

The new accounts' Heroes Profile histories enter the next weekly store. The regression checks then leave those accounts out (`changed-keys`).

## Bugs found and fixed on the way

- **Scoreboard join key overflow** (`p3_rcf.aug_metrics`). The join key `(replay_id << 40) | blizz_id` overflows int64 for replay ids of 2^23 and above. So a history row could take the scoreboard of the same player's game 2^24 replay ids away.
  - In def1027, 715 of 58.7M rows (606 accounts) carried another game's scoreboard.
  - Effect on def1027's headline numbers is nil to four digits: chosen OOT +37.381 before and after.
  - The exact pair join is behind `P3_METRIC_JOIN_FIX=1`. It is used by the weekly refresh; the default reproduces def1027.
- **Sealed replays in post-cap paths.** These are now excluded everywhere (P3_POSTFREEZE.md, correction).

## Card format

Card version 3 adds the estimates' as-of day. The site decodes versions 2 and 3, and both API routes now take the as-of day from the published cards (`estimatesAsOf()`), falling back to the artifact.

The publish emits version 3 only with `SITE_CARD_VERSION=3`. That needs the site change deployed first, so until then the cards stay version 2 and the page's "games through" date comes from the artifact (2026-09-27).
