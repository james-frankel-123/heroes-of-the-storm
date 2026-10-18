# Personalization on HotS Fever: design (2026-10-18)

Status: phase 1 is built (commit on `main`, not pushed or deployed). Phase 2, the draft page, is a plan below.

This page covers what we put on the site from paper 2's personalization work (`training/personalization/`, results in `results/P3_RICH_CF.md`, `P3_TAILS.md` and `P3_DRAFTER.md`), how it runs, and what it costs. Max asked for lookups of any player, not just the tracked party.

## 1. What a visitor sees

### Player lookup (`/players`, new nav item "Players")

1. **Search.** Heroes Profile (HP) has no search by name: `/players` needs the full BattleTag and a region, and a name alone returns 404 or 422 (checked live on 2026-10-18). So the box asks for `Name#1234`.
   - With a `#number`, we try all four regions (Americas, Europe, Asia, China) in parallel. A miss is a 404 and seems not to count against the allowance (the remaining count did not move). A hit costs one call from the Player pool.
   - Without a `#number`, we search only accounts someone has looked up here before (our cache, by name). The page says to add the number to find anyone else.
2. **Disambiguation.** Every match is a card showing BattleTag, region, Storm League games and win rate, rank, last played date, and the three most played heroes with game counts. All of it comes from one `/players` call per account. Cards are sorted by last played. Regions where the account is private on HP are named, and nothing else is shown for them.
3. **Skill cards.** One card per hero played, sorted by games. Each card shows:
   - the estimate, for example "+3.1 pp" on Ana: how many percentage points more games we expect this player to win on the hero than an average player would in the same draft;
   - "29 games, 55% won, last 2026-09-22";
   - the 80% range, for example "−3.4 to +9.6";
   - a verdict: "Above average" if the whole range is above 0, "Below average" if it is all below 0, otherwise "Not clearly different from average".

   The first 18 cards show, with a button for the rest.
4. **Heroes you might be good at.** The top 5 heroes the player has never played, ranked by the `imp2` model's estimate, each with its 80% range and the words "once learned". The copy says first games on a new hero usually run about 3 points worse (realized −3.19 pp in P3_RICH_CF).
5. **How far to trust this.** A short plain section on the page:
   - one game is mostly noise;
   - the ranges are 80% ranges, and on games the model never saw they held the true value about 81% of the time (main model, 0.806) or 80% (`imp2`, 0.805), so about one range in five misses;
   - combined with the draft, estimates for all ten players called the winner of 62.0% of held-out games with scoreboard stats (main model), and 60.8% with wins and losses only (`imp2`). The old win-only model, A2S, called 60.35%. These are real gains, and a reminder that the draft and the players leave most of a game unexplained;
   - private accounts are never shown;
   - "Data provided by Heroes Profile".

### As-of note

The research numbers are strictly as-of: each prediction used only games from earlier days, and the scored game never fed its own prediction. Live use is prospective by construction: we estimate from past games for games not yet played, so there is no leakage question. The live card includes games played earlier today, since it predicts the next game, not one of today's.

What the UI must not overstate:
- **Accuracy (main):** +11.0 log-loss points (x1000) over A2S on V2; 60.35% to 62.00% correct on V2 and 60.14% to 61.66% on OOT.
- **Accuracy (`imp2`):** 60.79% on V2 and 60.44% on OOT.
- **Band calibration:** 0.806 coverage at nominal 80% for the main model, 0.805 for `imp2`.

Both are for the research setting, and live inputs differ a little (section 6), so the copy says "about".

## 2. Models

| use | model | source | display scale (V1 fit) | validated |
|---|---|---|---|---|
| per-hero cards, when we hold corpus scoreboard stats for the player | main: `full_ms` (`met2_hs2` structure, full scoreboard export, `a_shrink` normalization), shown through the play-volume card plus the volume arm's temporal layer | P3_TAILS §1-2 | 2.18·var + (2.27 pp)² | V2 cover80 0.806; V2 +28.65 lift; accuracy 62.00% V2, 61.66% OOT |
| per-hero cards without scoreboard stats; never-played ranking | `imp2` (`full_imp2`: rank 2 plus play preference) | P3_RICH_CF variant 3, P3_TAILS §1 | 2.19·var + (0.43 pp)² | V2 cover80 0.805; V2 +19.77 lift; accuracy 60.79% V2, 60.44% OOT; Cassia R² 0.46 |
| draft value (phase 2) | V = sigmoid(b0 + b1·logit WP_pop + b2·(S0 − S1) + b3·(O0 − O1)) | P3_DRAFTER | | |

The displayed number for a played hero is:
- **with scoreboard stats:** experience offset + volume-card posterior mean + layer shift;
- **without them:** experience offset + `imp2` posterior mean.

The band is ±1.2816·√(c·var + k0²):
- with scoreboard stats, var is the main model's Gaussian posterior variance, the quantity the 0.806 coverage was measured on;
- without them, var is the `imp2` posterior variance.

Volume card and layer constants:
- volume card: 10 cells of (prior mean, log sd), from `full_ms_tail_vol`;
- layer: half-life 45 days, k_d 160, k_e 5, drift scale −0.077, learn scale 0.127, from the volume arm's V1 choice.

Why `imp2` and not the main model without scoreboard stats:
- `full_ms` was fit with the scoreboard channel pinning down general level. Its overall-skill prior sd is 5.6 pp, against 1.7 pp for win-only fits.
- Run on wins alone, that prior trusts win records too much, and nothing validated it.
- `imp2` is a complete win-only model with its own validated band.

## 3. Architecture

### Where the math runs: a TypeScript port, on Vercel, from static artifacts

| option | for | against | verdict |
|---|---|---|---|
| **TS port of the posterior** | the math is small: K = 3, so every posterior is a 3×3 Cholesky; the volume card is a 401-point quadrature; the layer is one pass over the player's games. A 2,400-game player computes in milliseconds in a serverless function. No new service to run. Testable against the Python functions. | the scoreboard channel needs corpus data that Vercel cannot read | **chosen** for everything the site computes |
| small Python service | runs the research code itself | the box is not reachable from Vercel; it would need a tunnel, uptime and auth. Nothing in the math needs Python. | no |
| precompute for every player | no live compute | only covers corpus players (620,893 accounts, 185,467 with 20+ games). Per-player rows on Neon would be tens of MB. Stale between refreshes. | no, except for the 9-number scoreboard summary below, which is precomputed only for looked-up accounts |

The one piece that needs the corpus is scoreboard evidence. Metrics enter the posterior only through two sums: Λ_m (3×3) and η_m (3). That holds because the metric loadings act on the player factor a_p, never on the hero-specific term. So the corpus part of a player is exactly 6 + 3 numbers for a fixed model. A local job computes them for accounts the site has looked up and writes one small row to Neon: the enrichment job (section 3, Enrichment).

### Static artifacts (exported from training)

`src/lib/personal/model-artifact.json` (60 KB, in the server bundle only), written by `training/personalization/p3_site_export.py artifact`:
- 90 hero names in model order (HP spells them the same way);
- main model U (90×3) and s² (90);
- `imp2` U, s², W (3×91), z mean and sd;
- the experience-offset table (8×8 bins);
- the hero × quarter win-rate proxy (90 × 18 quarters, prior 200 games, clipped to 0.2-0.8);
- the volume-card cells and the layer constants;
- the display scales;
- the V1 combiner, kept for phase 2;
- the validation numbers the UI quotes.

The offset and proxy tables come from the full_1017 panel on the 3090 (`p3_site_export.py tables`). They were checked against the main arm's own per-slot offsets: max error 1.5e-9 over 4.3M slots.

Refresh: after each research refit (the Oct 28 rerun), rerun `tables` on the 3090, then `artifact` and `fixture` locally. Then bump the artifact `version`. That invalidates cached cards, and the enrichment job republishes evidence under the new version.

### Request flow and API routes

- `GET /api/personal/search?q=Name#1234`:
  1. cached summary if it is under 24 h old;
  2. else `/players` per region.

  It returns the candidates, the private regions, and an unavailable message if the guard refused.
- `GET /api/personal/player?tag=Name#1234&region=1`:
  1. a cached card under 24 h old is served, unless enrichment has since arrived;
  2. otherwise refresh the summary if it is stale (`/players`);
  3. fetch the history:
     - `/players/mmr/history` when its budget is above 0: full history, normally one call;
     - otherwise `/players/matches`: 100 games a page, 3 pages, so the 300 most recent games;
  4. compute the profile with any enrichment evidence;
  5. store the profile and set `wants_metric` when no evidence exists.

  A private account (403 `player_unavailable`) deletes our cache rows for it and shows a private message. When HP or the guard refuses, a stale cached card is served with a note.
- Per-IP soft limit: 20 HP-touching requests per 10 minutes per serverless instance. The real protection is the global guard.

Code:
- `src/lib/personal/model.ts`: the port;
- `hp.ts`: HP calls and the quota guard;
- `store.ts`: the Neon cache;
- `lookup.ts`: the flow;
- `src/app/api/personal/*`;
- `src/app/players/*`.

### Enrichment (corpus scoreboard evidence)

`training/personalization/p3_site_enrich.py` runs on the main box, after privacy-sync:
1. It reads up to 200 accounts with `wants_metric` from Neon.
2. It skips any account private in the local `player_privacy` table.
3. It reads the account's corpus games from the local store: `replay_players` by (blizz_id, region), through the existing index.
4. It joins them to the `a_shrink` scoreboard rows through a blizz-sorted memory-mapped copy, built once in 6 s, in the gitignored cache.
5. It reduces them to Λ_m and η_m and upserts one `personal_metric_evidence` row.
6. It clears `wants_metric`, also when the account has no corpus rows, so it is not retried.

The next view of that player then upgrades from `imp2` to the main model. Measured: a dry run for one account took 6.5 s including the one-time index build.

Example, tracked account Django#1458 (dry run): 1,244 corpus games, 661 with a scoreboard row in the export.

It is not in cron yet. Proposed line, after privacy-sync, with the same flock style:
`45 */6 * * *` … `&& /home/linuxbrew/.linuxbrew/bin/python3 training/personalization/p3_site_enrich.py`

Running it on the existing 6-hourly Neon touch means no extra Neon wakeups. Copy promises "within a day".

### Quota guard

Pools seen in live response headers on 2026-10-18:

| pool | endpoint | allowance per week | other users of the pool | site budget per week (env) | hourly cap | floor on HP's remaining count |
|---|---|---|---|---|---|---|
| Player | `/players` | 50,000 | none (49,999 left) | 3,000 (`PERSONAL_HP_PLAYER_WEEKLY`) | 120 | 5,000 |
| Player/Match/History | `/players/matches` | 5,000 | tracked-player sync, about 1,100/wk (3,911 left mid-window) | 1,500 (`PERSONAL_HP_MATCHES_WEEKLY`) | 60 | 1,500 |
| Player/MMR | `/players/mmr/history` | 50,000 | tranche-2 pull, at its ledger cap of 49,800/wk, 57,166 accounts left, about 8 days | **0 until Max raises it** (`PERSONAL_HP_MMR_WEEKLY`) | 60 | 2,000 |

How the guard works:
- Every site call is counted in `personal_api_usage`, by pool and hour, with HP's `x-hp-quota-remaining`.
- A call is refused when either holds:
  - the site's own 7-day sum or this hour's sum would pass the budget;
  - HP's last reported remaining count, minus the call, is below the floor.

  The floor protects the sync jobs that share the key even if the site's count drifts.
- On refusal the page says new lookups are paused, and players looked up recently still work.
- 429s and HP outages map to the same state.

What the budgets buy:
- **While the MMR pool is off (until about Oct 26):** a new lookup costs up to 4 Player calls for the search (misses look free), 1 Player call for the summary, and up to 3 match pages. That is about 500 new players a week, capped by the 1,500 match-page budget.
- **After tranche 2, with MMR at for example 5,000/week:** a lookup is 1 MMR call for the full history, so about 3,000 new players a week, capped by the Player budget.
- **Repeat views:** within 24 h they are free.

### Neon footprint

| table | row | cap | worst case |
|---|---|---|---|
| `personal_profiles` | summary JSON about 350 B plus profile JSON 1.1 to 4.4 KB (measured on the fixture players; skill values rounded to 1e-5; 0.4 to 1.3 KB deflated) | 2,000 rows (`PERSONAL_MAX_PROFILES`), oldest summaries evicted on insert | about 6 MB |
| `personal_metric_evidence` | 9 reals plus keys, about 100 B | follows profiles (evicted with them) | 0.2 MB |
| `personal_api_usage` | about 50 B | 3 pools × 336 hours (rows over 14 days deleted) | 0.05 MB |

The total stays under about 6.5 MB on top of today's 40 MB, and is probably 1 to 3 MB in practice.

Egress:
- one lookup reads a few KB;
- privacy-sync reads at most 2,000 (battletag, region) pairs from Neon every 6 h, about 40 KB.

No bulk table, no match rows and no history are stored on Neon. Only the computed cards are.

Tables are created by `drizzle/0001_personal_lookup.sql`. It is additive and creates no index on the existing tables. They are listed in `sync/table-groups.ts` as `PERSONAL_TABLES`, so `drizzle.config.ts` sees them and `publish-site.ts` never touches them. `verify_store.py` skips them.

### Privacy and terms

- **HP is the gate.** Any 403 `player_unavailable` drops our rows for that account and shows nothing.
- **privacy-sync** now also deletes cached lookup rows for any account that is private in `player_privacy`, every 6 hours (`purgeLookupCache` in `sync/privacy-sync.ts`).
- **The enrichment job** skips private accounts.
- **Public data only.** Lookups show only what HP's public data shows:
  - counts, dates and heroes from HP;
  - model estimates from HP data plus corpus games that are themselves HP public replay data.

  Nothing from the research store is displayed raw.
- **Public repo.** No battletags are committed. The test fixture has no ids at all: three real Heroes Profile histories, the last 600 games each, days shifted by a random offset, plus synthetic cases.
- **Attribution.** "Data provided by Heroes Profile" is in the site footer and on the lookup page. The Terms page already binds users to the licence terms.
- **The expert study is untouched.** Its tables, routes and `/rate` are unchanged.

## 4. Tests

`src/lib/personal/__tests__/model.test.ts` checks 8 reference cases, against outputs from the Python functions themselves (`p3_rcf.posterior`, `p3_rcf_tails.post`, and the layer formula of `temporal_layer`):
- 3 de-identified real players: 40, 600 and 600 games, with 21 to 549 corpus scoreboard rows;
- 5 synthetic cases:
  - empty;
  - one game;
  - a 220-game one-trick;
  - a same-day burst;
  - games older than the proxy table.

What is checked:
- counts, offsets and the `imp2` posterior, for every case;
- the main Gaussian posterior, the leave-own-hero-out prior, the volume card and the layer terms, for cases with scoreboard evidence;
- all to 1e-7 in win-probability units (0.00001 pp), and 1e-9 on variances.

The TS side takes the 9-number evidence while the Python side takes the raw metric sums, so the test also proves the reduction exact.

Plus 5 behaviour tests:
- the band is the 80% interval;
- the main model is used only with evidence;
- games on the request day are dropped, as are unknown heroes;
- offsets and proxies follow their bins.

## 5. Phase 2: the draft page (plan only)

Goal: in `/draft`, add teammates and opponents beyond the tracked party, and let personal terms change the suggestions.

1. **Adding players.**
   - Each of the 10 player slots gets a "link player" control: it opens the same search and disambiguation, and stores (region, battletag) for the slot.
   - The draft client fetches each linked player's cached profile from `/api/personal/player`; a new lookup costs the same quota.
   - Tracked players keep their current one-click entries.
2. **Personal tables.**
   - For each linked player the API returns s(p, h) for all 90 heroes, not only played ones: offset plus posterior mean plus layer shift, under the main model when evidence exists, else `imp2`.
   - It also returns the off-role flag O(p, h): fine role under 10% of 50 or more games.
   - That is 90 × 2 numbers per player. Add it to the cached profile as one compact array, about 1 KB.
3. **Value.** V = sigmoid(b0 + b1·logit WP_pop + b2·(S_ours − S_theirs) + b3·(O_ours − O_theirs)).
   - With the main model use its own V1 combiner: b0 0.0015, b1 0.985, b2 4.58, from `full_ms_layer`. The layer is already folded into s.
   - The off-role term b3 −0.055 was fit with the phase-1 kernel. Refit it with the main model before launch; that is one logistic fit on V1 on the 3090.
   - Unlinked slots have s = 0 and O = 0, so a draft with nobody linked is the population draft.
4. **Search.** The browser MCTS and greedy suggestion code calls the ONNX WP. Add the personal term as a post-WP logit shift in the leaf evaluation; it is additive, so no retraining.
   - Candidates for a linked player: the union of the top 15 by population prior and the top 15 by s(p, h), as P3_DRAFTER did.
   - Bans: show "banning their main costs them about 1.7 points" only as a note (P3_TAILS §3). The number is observational.
5. **UI.**
   - A small "+x pp for this player" tag on suggested heroes.
   - A per-player chip with their top 3 heroes.
   - One line of copy: personal terms move about half of the recommendations in simulation (P3_DRAFTER: 51 to 52% of picks differ, +3.4 to +3.6 pp in V when they differ). That is model-internal, not a measured win-rate gain.
6. **Tests.** Port the combiner and check it against `p3_dr_core` on a few lobbies, from a fixture exported as in phase 1.

## 6. Known differences from the research setting

1. **Every live game uses the win-rate proxy.** In research, corpus games used the draft-aware WP and only extra HP rows used the hero-by-quarter proxy. Live history from HP has no draft, so every game uses the proxy, and quarters after the table's last one (2026 Q1) reuse it. The residuals are noisier, so expect slightly less sharp estimates than the validation numbers. The band includes the model's own variance, not this extra noise.
2. **Experience counts use all games.** Research offsets used corpus-game counts only (`exp_counts="slots"`). Live offsets count every HP game the player has. For regular players both are in the top bins (300+ games, 50+ on the hero).
3. **The layer walk uses the current metric summary.** The research layer computed each past residual with the scoreboard evidence as of that day. Live, the walk uses the current summary throughout. The layer terms are small: the drift scale −0.077 times a shrunk mean residual moves cards by about 0.1 pp. The new-hero term moves heroes with under 5 games by up to about 1 pp.
4. **The band under the volume card.** The displayed mean comes from the volume card, but the band uses the Gaussian variance with the V1 scale. That band is validated at 0.806; no coverage test was run for the volume card itself. P3_TAILS found the volume prior's never-played spread is about a third of the model's, so on thin heroes our band is, if anything, wide.
5. **Truncated history while the MMR pool is off.** `/players/matches` serves the 300 most recent games. The page says so ("these estimates use the 300 most recent").
6. **No search by name only.** HP has none, and we do not publish a name index.

## 7. Deploy checklist (coordinator)

1. Create the Neon tables: run `drizzle/0001_personal_lookup.sql` on Neon. It is additive only.
2. Set `HEROES_PROFILE_V2_API_KEY` in Vercel (server env only). Optionally set the `PERSONAL_*` budgets; the defaults are in `hp.ts`, `store.ts` and `lookup.ts`.
3. Add the enrichment cron line (section 3, Enrichment) after Max agrees.
4. After about Oct 26, when tranche 2 is done, set `PERSONAL_HP_MMR_WEEKLY`, for example to 5,000.
5. After the Oct 28 rerun, refresh the artifact (section 3, Static artifacts).

## 8. Open questions for Max

1. **MMR budget after tranche 2.** How much of the 50K Player/MMR pool goes to the site? 5,000 a week gives full histories for about 3,000 new players a week. Until then lookups use the 300 most recent games.
2. **Name-only search.** Should we publish a small name index, so "Name" without the number finds people? For example, active public accounts with 50+ corpus games in the last 6 months: about 30K rows, about 2 to 3 MB on Neon. HP has no name search, and this would list players who never asked to be listed.
3. **Who sees what.** Any visitor can look up any public account. HP's own site does the same, but our cards rate skill per hero, which is more pointed. Fine as is, or add a "hide me" request link?
4. **Enrichment cron.** OK to run `p3_site_enrich.py` every 6 hours after privacy-sync? It reads the research store, read-only, and writes 9 numbers per looked-up account to Neon.
5. **The band under the volume card.** Keep the validated Gaussian band (current), or run a coverage test of the volume card's own posterior on the 3090 and switch if it holds?
6. **Phase 2 off-role refit.** Refit b3 with the main model on V1 before the draft integration?
