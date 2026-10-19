# Personalization on HotS Fever: design (2026-10-19)

Status: built on branch `personal-site`, not merged or deployed. Launch is after the Oct 28 rerun: the coordinator creates the Neon table and runs the first publish. This version replaces the 2026-10-18 design, which looked players up live through the Heroes Profile (HP) API. Max's decisions of 2026-10-19 are in: 2.57 games on the site (sealed replays excluded), name-only search, the 3+ games threshold, and the off-role refit.

## 0. What changed and why

**The website never calls Heroes Profile.** HP's terms allow offline use only. So the HP key stays off Vercel, and the site has no HP client, no quota guard and no route that calls HP. Max asked why we would need to pull players in real time at all; we don't.

**Every player number is computed offline.** A weekly job on the main box builds the numbers from our research store: our corpus plus the HP history pull. It publishes them to Neon as one compact table, and the site only reads that table.

What ships:
- **`/players`:** lookup by full BattleTag, or by visible name alone, from the published data. If an account isn't in our data, the page says so plainly.
- **In-game mode on `/draft`:** type the visible names of the other players, without the #number. We match each name against the published index using the party's region and MMR. A clear winner is selected; otherwise 2 to 4 candidates are shown. Matched players' skill feeds the draft suggestions through the P3_DRAFTER value term.

## 1. Data and the publish job

**Inputs**, all local:
- the merged history store `training/personalization/cache/full_history/<tag>/history.npz` (corpus rows plus the HP per-match history pull, deduplicated), which stops at the research cap (no game from 2026-09-28 on);
- **site only:** the corpus games from 2026-09-28 on (the 2.57 builds), read from the local DB at build time;
- the corpus scoreboard card `cache/rcf/sb_a_shrink.npz`;
- names, discriminators and privacy states from the local research DB (`replay_players`, `player_privacy`, read-only role).

**2.57 games and the sealed replays.** Max decided on 2026-10-19 to include 2.57 games so the cards stay fresh; paper 3 is cancelled, so the C, F and pool tranches no longer matter.
- The 5,000 sealed replays (`training/replays/cohorts/sealed_ids.txt`, rule in `sealed_rule.json`) are always excluded. The post-cap query filters them in SQL (`NOT (replay_id = ANY(sealed))`), the store rows and the name query are filtered against the same list, and the job asserts none got through.
- This is a site-only addition in `p3_site_publish.py`. `p3_full_history.py` and its `CAP_TS` are unchanged, because the Oct 28 personalization rerun is defined with them. `--research-cap` builds without post-cap rows.
- **Post-cap rows have no scoreboard.** The `a_shrink` card covers the capped export only, so post-cap games enter the win channel and the counts, not the metric channel.
- `replay_draft_data.game_date` is UTC since the 2026-10-16 timezone fix (`sync/docs/timezone-fix-2026-10.md`).
- Build of 2026-10-19: 600,115 post-cap player rows; data through 2026-10-19.

**Who gets a card** (build of 2026-10-19):
- any account with at least one game in the 180 days before the latest game and a visible name: 182,400 accounts;
- of those, 128,357 have 3 or more games in the window, and they are the default load (`--min-recent 3`, Max's decision);
- 712 private accounts in `player_privacy` are dropped;
- 15,603 accounts have no corpus scoreboard rows (mostly new since the cap) and get `imp2` cards.

**Model, per account:**
- **Displayed per-hero number:** the main model (`full_ms`, the `met2_hs2` structure with the `a_shrink` scoreboard card) through the play-volume card (P3_TAILS), plus the experience offset. It applies wherever the account has corpus scoreboard rows: 166,797 of 182,400 accounts. The rest get `imp2` (wins and losses only).
- **Band:** ±1.2816·√(c·var + k0²), with the no-layer main arm's V1 scale (c 2.23, k0 2.29 pp; 80% coverage on V2: 0.808) or `imp2`'s (c 2.19, k0 0.43 pp; 0.805).
- **Draft term s(p, h):** the main model's Gaussian posterior mean plus the offset. This is the quantity its V1 combiner was fit on.
- **Never-played suggestions:** the top 5 by `imp2`.
- **No temporal layer in the cards.** The layer needs a per-game walk with as-of scoreboard evidence. It is worth +1.1 log-loss points of the main model's +11 over A2S, and it moves a card by about 0.1 pp. Section 6 has the details.

**Card format.** One bytea per account, 201 B on average and 590 B at the 95th percentile:
- a header with the model, the game count and the 3-number factor mean â of the draft model;
- per played hero, 10 B: hero, games, last played, display mean, display sd and draft s;
- 5 suggestions.

For a never-played hero the page computes s = offset(games, 0) + U_h·â from the bundled artifact. That is exact, because with no own games the posterior mean at h is U_h·â.

**Name index columns:** region, visible name (no #), discriminator, MMR (HP post-game MMR of the last game in the store), last played, games in our data, games in the window, top 3 heroes. They share the row with the card: table `personal_accounts`, primary key (region, blizz_id), index on lower(name). blizz_id never leaves the server.

**Code:**
- `training/personalization/p3_site_publish.py`, with stages:
  - `build`: about 3 minutes on 4 cores, numpy only;
  - `check`;
  - `load`;
  - `simulate`;
  - `fixture`;
- `training/personalization/run_site_publish.sh`, the weekly job: `build` → `check` → `load` → `sync/publish-site.ts`.
- `personal_accounts` is in `SITE_TABLES`, so the existing publish-site flow pushes it to Neon:
  - it is pushed only when its content hash changes;
  - it is gated against a drop of over 20% in row count;
  - it swaps in one transaction.

  publish-site now skips, with a warning, a site table that does not exist on Neon yet, so the merge cannot break the other tables' publish.
- **Privacy:** `sync/privacy-sync.ts` now also deletes a newly private account from `personal_accounts`. The publish after it carries the delete to Neon. The build also drops accounts already private.

**Schedule.** Weekly. With 2.57 games the table changes every run, and publish-site replaces it whole (about 48 MB written to Neon, plus dead rows until autovacuum), so weekly keeps Neon churn low. Daily works too if fresher cards are worth the writes. Cron line for the coordinator (not installed):

```
15 23 * * 0 flock -n /home/max/heroes-of-the-storm/sync/.site-publish.lock sh -c 'cd /home/max/heroes-of-the-storm && set -a && . ./.env && set +a && nice -n 19 ionice -c3 taskset -c 48-51 training/personalization/run_site_publish.sh' >> /home/max/heroes-of-the-storm/sync/logs/site-publish.log 2>&1
```

The scoreboard enrichment from the previous design is folded into `build`: scoreboard evidence is computed for every account in one pass. There is no separate enrich job and no queue.

**Store refresh.** Point `SITE_STORE` at a newer store tag when research builds one, for example after the Oct 28 rerun. Also refresh the artifact then: `p3_site_export.py tables` on the 3090, then `artifact`. The post-cap rows keep the cards fresh between store refreshes.

**The date the pages show** is the latest `last_played` in the table, cached for an hour per server instance. There is no env setting.

## 2. Sizes

Measured on the local store after `VACUUM FULL`; Neon will be close.

**Default, build of 2026-10-19 with 2.57 games:** 128,357 accounts, **47.7 MB** (table plus indexes). Max approved about +44 MB at the 3+ threshold; the 2.57 games add 14,352 accounts and about 4 MB.

The thresholds as measured on the capped build of the same day (the lobby share comes from the simulation):

| `--min-recent` (games in the 180-day window) | accounts | table + indexes | lobby players findable (simulation) |
|---|---|---|---|
| 1 | 159,814 | 54.5 MB | 78% |
| **3 (default)** | **114,005** | **43.6 MB** | **74%** |
| 5 | 92,953 | 38.0 MB | 71% |
| 10 | 67,529 | 30.3 MB | 66% |

Neon holds about 40 MB today, so the default roughly doubles it. publish-site replaces rows with DELETE plus INSERT. Until autovacuum reuses the space, a changed publish can briefly hold two copies, but with a capped store that happens only on a store refresh.

**Egress:**
- a lookup reads at most a few rows of about 350 B;
- a name match reads the small columns of the same-name rows, then the cards of at most 4;
- responses carry a 1-hour CDN cache (`s-maxage=3600`), so repeats skip Neon;
- publish-site computes its fingerprint on Neon and sends back one row.

There are no bulk reads.

## 3. The players page (`/players`)

- **Lookup by full BattleTag,** `Name#1234`. The page reads every account with that name and discriminator: one per region it exists in. If there are several, the visitor picks the region.
- **Search by name alone** (Max's decision; same open privacy stance as in-game). `Name` lists every published account with that visible name, up to 50, most recently active first, each with region, discriminator, MMR, last played, games and top heroes. Picking one opens its profile.
- **Not in our data.** "Name#1234 is not in our data. We have cards for accounts with at least 3 Storm League games in our replay data in the 180 days up to <date>."
- **Profile:**
  - per-hero cards: estimate, n games, last played, 80% range, verdict;
  - "Heroes you might be good at";
  - a plain "How far to trust this" section quoting the coverage (0.81 main, 0.80 `imp2`) and the held-out accuracy: 61.8% of winners called with scoreboard stats (main, no layer) and 60.8% with wins and losses only, against 60.35% for the old A2S;
  - "Data provided by Heroes Profile".

## 4. In-game mode on the draft page

**Flow:**
1. Party members come from the existing tracked-battletag slots.
2. The "In-game players" panel takes up to 4 teammate names (5 minus the party) and 5 enemy names, plus an optional region override.
3. "Match players" calls `GET /api/personal/match?names=…&party=…`.

**Matching:**
1. **Candidates:** every published account with the same visible name, ignoring case.
2. **Context:** the party's region (the most common among party accounts found by full tag) and the party's mean MMR.
3. **Score:** 10 for the same region, minus |MMR − party MMR|/300 (1 if either MMR is unknown), plus exp(−days since last played / 30) as the recency tiebreak.
4. **Selected:** the top candidate when its score is at least 9.5 and it leads the next by 1.5. Otherwise the panel shows up to 4 candidates (region, MMR, last played, games, top heroes) to pick from, or "leave as average".
5. **Unmatched or unpicked players** add nothing, so they count at population values.

**Accuracy (simulation).** `p3_site_publish.py simulate`, in `results/site_match_sim.json`. It runs on the research store only, so it predates the 2.57 games:
- **Setup:**
  - the index was built as of 2026-09-13 from the 180 days before;
  - the test lobbies are 20,000 corpus games from Sep 13 to 27;
  - in each lobby one random player is the party, and the other nine are matched by visible name only.
- **Tuning:** thresholds were tuned on half of the lobbies; the other half is reported, 89,883 players.

| default index (3+ games) | share of lobby players |
|---|---|
| true account in the index | 73.8% |
| …of those, ranked first / in the top 3 | 93.9% / 98.5% |
| selected automatically | 54.7% (correct 53.6%, wrong 1.1%; precision 98.0%) |
| shown as 2 to 4 choices | 23.5% (true account among them 80.8%) |
| no account with that name | 21.4% |

Most wrong automatic picks are players missing from the index who share a name with someone of similar MMR in the same region. With 2.57 games the published index is at most a week old at each run, closer to this 2-week test than the capped design was.

**Personal value** (P3_DRAFTER), V = sigmoid(b0 + b1·logit Q + Σ_ours b2·s(p, h) − Σ_theirs b2·s(p, h) + b3·(O_ours − O_theirs)):
- **Inputs:**
  - Q is the page's own search value for the row;
  - b0, b1 and b2 are each model's V1 combiner, refit with the off-role count: main 0.0011, 0.982, 4.69; `imp2` players use b2 = 3.74;
  - b3 is the off-role weight per off-role player, refit on the 3090 (`p3_site_offrole.py`, `results/rcf_full/site/offrole_refit.json`). Off-role means the hero's fine role is under 10% of the player's 50 or more earlier games; it holds on 10.2% of slots. Each player carries their own model's b3.

    | model | b3 (was −0.055) | log-loss gain from the term, x1000, V1 / V2 / OOT (SE) |
    |---|---|---|
    | main | −0.020 | +0.05 (0.04) / +0.09 (0.04) / +0.02 (0.02) |
    | `imp2` | −0.043 | +0.22 (0.08) / +0.37 (0.08) / +0.18 (0.04) |

    With the main model the off-role term is nearly absorbed: its skill estimates already price heroes a player rarely plays. It stays in, small. On the site, off-role uses the card's hero counts (all games); the refit used the slot table's counts, as P3_DRAFTER did.
- **Who plays which hero.** That is unknown during a draft, so each team's term is its best assignment of picked heroes to its matched players, with unmatched players at 0. A pick the user has assigned to a tracked player stays pinned.
- **Our picks:** each row's value is recomputed with the candidate added, and the row says who would play it ("Name: +3.1 pp on Ana").
- **Our bans:** the row gains the drop in an unassigned matched enemy's best available option ("Takes away Name's best option").
- **Extra candidates.** Each unassigned matched player's top heroes are added as rows. They were not searched, so the current value stands in for Q, and the panel says so.
- **The header.** It keeps the search value. The panel adds one line: "With player skill for N matched players: x% now (search alone y%)".
- **Scope.** This is a re-ranking on top of the search, not inside it. Putting the personal term in the MCTS leaf evaluation is the follow-up, since it is additive in logit and needs no retraining.

## 5. Tests

- **`p3_site_publish.py check`** (in the weekly job): on 40 random active accounts, the batched card build equals the research reference within 1.2e-16 in win-probability units. The reference is `p3_site_export.reference`, which calls `p3_rcf.posterior` and `p3_rcf_tails.post`. It checks the draft term, display mean, display sd and `imp2`. Output: `results/site_publish_check.json`.
- **`src/lib/personal/__tests__/personal.test.ts`**, 13 tests:
  - card decoding and the all-hero draft term, on synthetic cards encoded by the Python job: no real accounts in the repo;
  - match scores equal to the Python rule;
  - the clear-winner rule;
  - the best assignment, with pins and off-role;
  - "no matched players equals the population value";
  - the ban gain.
- Type-check, lint, the full Vitest suite and `next build` pass.

## 6. Known differences from the research setting

1. **Every row uses the hero-by-quarter win-rate proxy**, including corpus games, which have a draft. Research used the draft-aware WP for corpus rows. Expect slightly less sharp estimates than the validation numbers. Using the stored WP for corpus rows is a possible follow-up, at the cost of reading the slot WP caches in the job.
2. **Experience counts use all games in the store**, corpus and HP. Research offsets counted corpus slot rows only.
3. **No temporal layer** (section 1).
4. **The band under the volume card.** The displayed mean is the volume card's; the band is the Gaussian variance with the validated scale. No coverage test was run for the volume card itself.
5. **Post-cap games have no scoreboard,** so for players active mostly since 2026-09-28 the metric channel is older than their win record. Exporting the scoreboard for 2.57 games through the `a_shrink` card is a possible follow-up.

## 7. Before launch (coordinator)

1. Create the table on Neon: `drizzle/0001_personal_accounts.sql`. It is additive. Max approved creating it.
2. Merge `personal-site`.
3. Run `training/personalization/run_site_publish.sh` once by hand: about 4 minutes, and it pushes the table with publish-site. Then install the cron line from section 1.
4. No new Vercel env is needed.
5. `/rate` and the study tables are untouched.

## 8. Decisions (2026-10-19) and what is still open

Decided by Max:
- +44 MB on Neon is fine at the 3+ games threshold. The build of 2026-10-19 measures 47.7 MB with the 2.57 games.
- Name-only search on `/players`, with the same open privacy stance as in-game.
- 2.57 games on the site, with the sealed replays always excluded.
- The off-role weight is refit with the main model.

Still open:
1. Wrong automatic matches are 1.1% of players in the simulation. Keep the current bar, or raise it and show more choice lists?
2. Should the 2.57 scoreboard be exported so post-cap games also feed the metric channel?
