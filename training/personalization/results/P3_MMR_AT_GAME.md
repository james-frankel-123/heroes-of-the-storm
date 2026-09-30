# P3: MMR at game time

Date: 2026-09-30. Code: `p3_fetch_mmr_stamps.py`, `p3_mmr_at_game.py`, `p3_mmr_panel.py`, `sync/fetch-mmr-history.ts`. Raw numbers: `p3_mmr_at_game.txt` / `.json`.

## What the stored MMR is

Every `replay_players` row carries HP's rating AFTER that game, computed in HP's processing (parse) order, and `raw_extras.mmr_date_parsed` is when HP processed it, in US Eastern local time (`game_date` is UTC, the game end). Checks:

- Parse order is the chain: the sign of the MMR step matches the game's result for 90% of consecutive pairs in parse order, 68% in play order. The other 10% fit games we do not store and same-second ties (6% of rows share a parse time).
- Timezone: after converting Eastern to UTC, 22% of window stamps land 0 to 60 s after the game end. 0.59% sit before the end (median 42 s, up to about 3 h); these are clamped to the game end.
- Many stamps are late: in the 2024-04 .. 2026-05 window only 22% are parsed within a minute, 37% more than 30 days later. Some window games were stamped as late as September 2026.

So a row's MMR includes its own result, and it is information only from its stamp time on. This is why the lagged MMR leaked.

## Definitions (`cache/mmr_at_game.npz`, one row per stored SL player-game, 17.66M rows)

- `*_mmr_causal`: post-game rating of the player's latest stamp parsed strictly before this game's start (start = end - game_length). Clean by construction. Player, hero (same hero) and role (same Blizzard role).
- `*_mmr_fresh`: HP's own pre-game rating (post-game value of this game's parse-order predecessor), used only when every game HP had processed up to that predecessor ended before this game started; otherwise equals causal. It is used for 49% of snapshot slots; it differs from causal (by more than 0.5 MMR, or where causal is missing) in 7% of window rows (p90 |diff| 5 MMR where used).
- `prev_stamp_mmr`: stamped MMR of the previous game in play order (the old lagged value), with `prev_stamp_parsed_before_start`.
- `own_stamp_mmr`: this game's own stamp (certain leak, positive control).
- `*_causal_age_s`: seconds from the source stamp to the game start.

## Coverage (P3 snapshot, 10.82M slots, all matched)

| | all slots | panel (>=100 games) |
|---|---|---|
| player causal | 96.5% | 99.5% |
| hero causal | 73.9% | 86.8% |
| role causal | 90.9% | 97.9% |
| previous stamp parsed before start | 36.4% | 36.5% |
| days since causal point, median (p90) | 1.2 (15.9) | 0.9 (7.0) |

Only about a third of previous-game stamps were known when the next game started.

## Leak check

Residual r = y - WP(draft), regressed on the player's value minus the lobby mean of causal MMR (so only the tested player's own value changes), 815K full lobbies. Log-loss gain x1000:

| stratum | causal | fresh | prev stamp | own stamp |
|---|---|---|---|---|
| all | 0.039 | 0.046 | 0.676 | 3.345 |
| prev parsed before start (3.15M slots) | 0.099 | 0.101 | 0.094 | 3.540 |
| prev parsed after start (5.01M slots) | 0.016 | 0.023 | 1.528 | 3.227 |
| lobbies with all 10 prev stamps clean (7.8K) | 0.185 | 0.189 | 0.187 | 3.187 |

Where the previous stamp was known before the game, at-game MMR ties it (0.099 vs 0.094). Where it was not, the previous stamp predicts the game 95x better than at-game MMR: that is the leak, and it covers 64% of slots. At-game player MMR on its own is a weak predictor once the draft is known (the lobby is matched on it). Hero MMR (causal) gives 0.75 on its own full lobbies, role MMR 0.16.

## HP MMR-history pull

`GET /api/external/v1/players/mmr/history` (battletag, region, game_type) gives one row per match with player, hero and role ratings, their changes, and `mmr_date_parsed`, so one call per player covers all three. It fills games we do not store and gives HP's exact change per game. It is blocked until v1 live data is activated (activation kills the old key that the full-corpus refetch still uses). Panel: 29,636 accounts. At 5 calls/min and a 45K rolling-week cap it takes about 4.6 days of live pulling. Hourly cron at :20 does nothing until activation. Rerun `p3_mmr_at_game.py` after the pull; it merges the shards (API values win) and reports the API game_date offset.
