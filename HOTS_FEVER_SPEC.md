# HotS Fever — Product & Technical Specification

## Overall

This site is a one-stop shop for Heroes of the Storm players to gain insights and navigate the significant data the game provides.

All data only looks at storm league, no ARAM, QM, or other data.

Development is all done via Claude Code via the Opus model. In addition to source code, we maintain a parallel document for each page/major feature documenting design choices and product specifications. If ever any changes need to be made from requested/specified functionality due to challenges in implementation this must be flagged in these documents in a VERY OBVIOUS MANNER. We maintain a high bar for testing, including loading the website (potentially having Claude take screenshots and viewing them via a visual model), simulating clicks, etc. Given we want to also test on real data, we primarily test with battletags Django#1458, AzmoDonTrump#1139, and SirWatsonII#1400. We know that, respectively, their Ana + Falstad, Nazeebo + Azmodan, and Malthael + Dehaka tend to be their very strong heroes, and expect to see those crop up regularly in recommendations. We commit one feature at a time, and only once it's been tested and in a working state. We do not develop multiple features on the same branch/at the same time before committing.

We are heavily inspired by heroesprofile.com and use them for data, but believe we can build a superior product, especially on ease of use and responsiveness as a draft assistant.

The main pages/functionality:

1. Draft Insight
2. Heroes Insight
3. Map Insight

For each of these, we pull both personal data for the player and overall community data. For community data, we bucket into three broad categories:

1. Low rank (Bronze + Silver)
2. Mid rank (Gold + Plat)
3. High rank (Diamond + Master)

We aggregate hero, map, and talent level insights by these buckets. This allows us to precompute and cache many things, in addition to simplifying user experience.

---

## Dashboard

Acts as a home page showing high level insights, trends, and aggregation. Includes personal information (how particular heroes are improving or falling) as well as broad trends (heroes that are particularly strong or weak in the current patch, pairings that are strong together, counters, etc).

### Meta Madness

In the current major patch (or minor if sufficient data), split by the three skill groupings: heroes dramatically outperforming/underperforming overall, outperforming/underperforming per map, and pairwise heroes that are strongest/weakest together and strongest/weakest against one another. Only considers storm league games.

### Personal Insights

Accounts can subscribe to up to 10 battletags and have a section on personal insights for each. Personal insights cover heroes the player is underperforming/overperforming with overall across all patches, and also provides a momentum-adjusted win % (MAWP) that weighs recent performance more heavily.

#### Momentum-Adjusted Win % (MAWP) Formula

Weight for game *i*:

    w(i) = w_games(i) × w_time(i)

Game count factor — full weight for last 30 games, then exponential decay:

    w_games(i) = 1.0                          if game is within last 30
    w_games(i) = exp(-λ_g × (rank - 30))      otherwise
    λ_g = ln(2) / 30   (half-life of 30 additional games, so game #60 has weight 0.5)

Where `rank` is the game's position sorted newest-first (1 = most recent).

Time factor — full weight for last 6 months, then exponential decay:

    w_time(i) = 1.0                           if game is within last 180 days
    w_time(i) = exp(-λ_t × (days - 180))      otherwise
    λ_t = ln(2) / 90   (half-life of 90 days past the cliff, so 9 months ago ≈ 0.5)

Final calculation:

    MAWP = Σ(w(i) × outcome(i)) / Σ(w(i))
    where outcome(i) = 1 for win, 0 for loss

Only considers storm league games. Not split by skill level groupings (it's personal data).

---

## Draft Insights

Offers a draft assistant. Latency is of the utmost importance, as the draft is timed. The user enters the map, whether their team is banning first or not, level range (low/mid/high), and enters bans and picks as they happen.

Optionally, the user can enter battletags for players on their own team, though battletags must correspond to registered HotS Fever users (otherwise generic stats are used). Opponent battletags are not supported — only display names are visible during draft, and disambiguation is not feasible.

### Draft Sequence

Team A bans 1, Team B bans 1, Team A bans 1, Team B bans 1. Team A picks 1, Team B picks 2, Team A picks 2, Team B bans 1, Team A bans 1, Team B picks 2, Team A picks 2, Team B picks 1.

### Recommendations

The system showcases both broad draft recommendations (aggregate data for selected skill tier) alongside personalized insights for registered users. Personalized integration uses simple heuristics layered on aggregate data:

- Is any player on the team experienced with and decent at a hero that has favorable stats on this map?
- Does a hero counter a selected opponent pick?
- Does a hero pair well with another hero already drafted or likely to be drafted?
- Is a player a likely smurf or one-trick?

Uses only individual and pairwise stats — e.g. Lunara's win rate against Uther, not Lunara + Ana vs Uther. This limits computation branching.

---

## Hero Insights

Page with all heroes showing aggregate averages (KDA, hero damage, etc).

Details view per hero with:
- Global and personal stats per tracked battletag
- Game history
- Heroes this hero is good/bad against and with
- Talent statistics
- Win rate trends

Heroes Profile does a good job with this; we can link to it or emulate it.

---

## Map Insights

Aggregate view per map with personal insights on win rate. Details view per map showing personal and aggregate (split by skill level) heroes with winning and losing records.

---

## Hosting, Database & Data Processing

### Hosting

- **Frontend + API:** Vercel (Next.js). Auto-deploys from GitHub main branch. Preview deploys on PRs.
- **Database:** Vercel Postgres (Neon-backed). Connection string via Vercel env vars.
- **Data sync (cron):** Runs on a separate Debian server (always-on, massively overspec'd — not a constraint). Connects to the same Neon Postgres instance via connection string.

### Database

- **Engine:** PostgreSQL (via Vercel Postgres / Neon).
- **ORM:** Drizzle or Prisma — pick one early, stay consistent.
- **Schema design principles:**
  - Precomputed aggregate tables bucketed by skill tier (low/mid/high) for heroes, maps, talents, and pairwise hero synergy/counter stats.
  - Per-user tables for match history, hero performance, and momentum-adjusted stats.
  - All draft-relevant data (hero win rates by map, pairwise stats) must be queryable with no external API calls. The draft assistant reads only from the local DB.
- **Scale assumption:** Handful of users. Single Neon instance, free or low-paid tier. No read replicas, no sharding, no connection pooling concerns for now.

### Data Sync (Cron Job)

- **Runs on:** Debian server via systemd timer or cron.
- **Frequency:** Every few hours (configurable, start with every 4h).
- **Language:** Python or Node — match whatever is easier to maintain alongside the Next.js frontend.
- **What it does:**
  1. Pull aggregate data from Heroes Profile API: hero stats, map stats, talent stats, pairwise hero data. Bucket by skill tier (Bronze+Silver / Gold+Plat / Diamond+Master).
  2. Pull personal match history for every registered user's tracked battletags (up to 10 per account).
  3. Compute derived stats: momentum-adjusted win %, hero trends, power picks (hero+map combos ≥65% WR), pairwise synergy/counter scores.
  4. Write everything to Neon Postgres, replacing stale data.
- **Error handling:** Log failures, retry transient errors, don't clobber good data on partial failure. Idempotent writes (upserts).
- **Connection:** Standard Postgres connection string. Store in a `.env` file on the Debian server (same connection string as Vercel uses, obtained from Vercel dashboard or `vercel env pull`).

### Draft Assistant — Data Strategy

- All draft recommendations come from precomputed DB data. Zero external API calls during draft.
- For registered users on their team: Full personalized recommendations (hero pool, per-map performance, synergy with teammates' picks, counter-pick data). Data is already in the DB from the cron sync.
- For non-registered players: No battletag-specific data. Recommendations use only aggregate stats for the selected skill tier. The app does NOT attempt to look up or disambiguate players by name during draft.
- Latency target: Draft recommendations should render in <200ms. All local DB reads of precomputed data.

### Environment & Secrets

- **Vercel:** Env vars managed via Vercel dashboard. Includes `DATABASE_URL`, Heroes Profile API key, any LLM API keys if needed.
- **Debian server:** `.env` file with `DATABASE_URL` and Heroes Profile API key. Not committed to git.

### What NOT to Do

- Don't put the cron job on Vercel (timeout limits, wrong tool for the job).
- Don't make real-time Heroes Profile API calls from the frontend or during draft.
- Don't try to precompute battletag-specific data for all possible players — only for registered users' tracked battletags.
- Don't over-engineer the DB. Single Postgres instance, straightforward relational schema. Revisit if usage grows past ~50 active users.

---

## Future Features (Not MVP — Do Not Build Yet)

### Teams Analytics

Analyzes performance in parties (duo, trio, 4-stack, 5-stack). For each tracked battletag, group matches by party composition. Show win rates and total games per party combination, best heroes per player within each party, and which maps the party performs best on. Depends on Heroes Profile API exposing party/group data — verify before speccing further.
