# HotS Fever — Product & Technical Specification

## Overall

This site is a one-stop shop for Heroes of the Storm players to gain insights and navigate the significant data the game provides.

All data only looks at storm league, no ARAM, QM, or other data.

Development is all done via Claude Code via the Opus model. In addition to source code, we maintain a parallel document for each page/major feature documenting design choices and product specifications. If ever any changes need to be made from requested/specified functionality due to challenges in implementation this must be flagged in these documents in a VERY OBVIOUS MANNER. We maintain a high bar for testing, including loading the website (potentially having Claude take screenshots and viewing them via a visual model), simulating clicks, etc. Given we want to also test on real data, we primarily test with battletags Django#1458, AzmoDonTrump#1139, SirWatsonII#1400, notoriusPIG#11231, and Mestupidum#1183. We know that, respectively, their Ana + Falstad, Nazeebo + Azmodan, and Malthael + Dehaka tend to be their very strong heroes, and expect to see those crop up regularly in recommendations. We commit one feature at a time, and only once it's been tested and in a working state. We do not develop multiple features on the same branch/at the same time before committing.

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

Estimates a player's current win probability for a hero via three mechanisms:

**1. Game-count weighting** — recent games matter more:

    w_games(i) = 1.0                          if rank ≤ 30
    w_games(i) = exp(-λ_g × (rank - 30))      otherwise
    λ_g = ln(2) / 30   (half-life of 30 additional games, so game #60 has weight 0.5)

Where `rank` is the game's position sorted newest-first (1 = most recent).

**2. Time-decay blending** — old game outcomes blend toward 50%:

    w_time(i) = 1.0                           if game is within last 180 days
    w_time(i) = exp(-λ_t × (days - 180))      otherwise
    λ_t = ln(2) / 90   (half-life of 90 days past the cliff, so 9 months ago ≈ 0.5)

    effectiveOutcome(i) = outcome(i) × w_time(i) + 0.5 × (1 - w_time(i))

This ensures old games contribute roughly 50% (unknown) rather than vanishing
entirely. A game played 2 years ago is treated almost as a coin flip regardless
of whether it was a win or loss.

**3. Bayesian padding** — low game counts shrink toward 50%:

If a player has fewer than 30 games on a hero, pad with (30 - games) phantom
50% observations at full weight. This prevents extreme MAWP values when sample
size is small (e.g. 8 games at 62.5% WR → ~53% MAWP, not 62.5%).

**Final calculation:**

    MAWP = (Σ(w_games(i) × effectiveOutcome(i)) + phantomPadding)
         / (Σ(w_games(i)) + phantomCount)

    where:
      outcome(i)          = 1 for win, 0 for loss
      effectiveOutcome(i) = outcome(i) × w_time(i) + 0.5 × (1 - w_time(i))
      phantomCount         = max(0, 30 - games)
      phantomPadding       = phantomCount × 0.5

Returns a value in [0, 1]. Multiply by 100 for percentage display.
Empty input returns 0.5 (the Bayesian prior — no data means unknown).

Only considers storm league games. Not split by skill level groupings (it's personal data).

---

## Draft Insights

Offers a draft assistant. Latency is of the utmost importance, as the draft is timed. The user enters the map, whether their team is banning first or not, level range (low/mid/high), and enters bans and picks as they happen.

Optionally, the user can enter battletags for players on their own team, though battletags must correspond to registered HotS Fever users (otherwise generic stats are used). Opponent battletags are not supported — only display names are visible during draft, and disambiguation is not feasible.

### Draft Sequence

Team A bans 1, Team B bans 1, Team A bans 1, Team B bans 1. Team A picks 1, Team B picks 2, Team A picks 2, Team B bans 1, Team A bans 1, Team B picks 2, Team A picks 2, Team B picks 1.

### Scoring Model

All scores are expressed as net win-rate deltas from a 50% baseline (e.g. "+3.2%" means "picking this hero shifts expected win probability by +3.2 percentage points"). Hero base WR prefers map-specific data when available (≥50 games on the selected map), falling back to overall WR. All pairwise data (synergies and counters) is fetched and stored per skill tier.

#### Pick Recommendations (Our Turn)

Five factors contribute to the displayed net delta:

1. **Hero base WR**: `(heroWR - 50)`. Prefers map-specific data.
2. **Counter matchups vs enemy picks**: Average of normalized pairwise deltas against each enemy hero. Normalized by subtracting the expected WR given both heroes' base rates to avoid double-counting individual hero strength.
3. **Synergies with ally picks**: Average of normalized pairwise deltas with each ally. Same normalization as counters.
4. **Player strength**: If a registered battletag is available and has ≥10 games on the hero, uses their confidence-adjusted MAWP. The best available player's delta replaces (not adds to) the hero base WR contribution for that slot.
5. **Composition WR**: Data-driven scoring based on achievable 5-role team compositions from Heroes Profile. Finds the best composition (by confidence-adjusted WR) that the current picks + candidate can achieve, and scores the delta from a popularity-weighted baseline. Scales from 0 impact at the start of draft to full impact at the last pick. Penalizes heroes that lead to no known viable composition.

#### Ban Recommendations

Three factors determine ban priority:

1. **Hero base WR**: How strong the hero is overall (or on the selected map). A strong hero is a good ban target.
2. **Counter strength vs our picks**: If the hero is strong against heroes we've already picked, banning protects our team. Only counts matchups that exceed the expected WR by ≥3 percentage points.
3. **Synergy with opponent's picks**: If the hero synergizes well with what the opponent has already picked, banning denies them a strong combination. Only counts synergies that exceed the expected pair WR by ≥2 percentage points.

#### Enemy Pick Predictions

Shows what the opponent is likely to pick, scored by hero base WR, normalized counter strength against our picks, and composition fit with their existing picks.

### Running Win % Estimate

A live team win percentage updates as each pick is made, giving both teams a running score throughout the draft. Starting from 50%, it accumulates:

1. **Hero base WR deltas** (sum per hero, map-specific when available)
2. **Intra-team synergies** (average across all ally pairs, normalized)
3. **Cross-team counters** (average across all matchups, normalized)
4. **Player adjustments** (confidence-adjusted MAWP replaces hero base WR for assigned players)
5. **Composition WR** (data-driven boost/penalty based on team roles, scaled by picks made)

Both teams' win estimates are normalized to sum to 100% when both have picks. Result clamped to [1%, 99%]. Color-coded green (>53%), yellow (47-53%), red (<47%).

A "Draft Domination" celebration triggers when the draft completes with our team at ≥60% estimated win rate.

### Normalization of Pairwise Deltas

To avoid double-counting hero strength in pairwise stats, all synergy and counter deltas are normalized against hero base win rates:

- **Counters**: `pairwiseVsWR - (heroWR + (100 - enemyWR) - 50)` — isolates the matchup-specific advantage beyond what you'd expect from each hero's individual strength.
- **Synergies**: `pairwiseWithWR - (50 + (heroA_WR - 50) + (heroB_WR - 50))` — isolates the synergy beyond both heroes being individually strong.

### Cho'gall Handling

Cho and Gall are always picked/banned together. If either is selected, the other auto-fills the next available pick slot for that team. Both are excluded from recommendations if the team has fewer than 2 consecutive pick slots remaining in the current turn.

### Other Draft UI Features

- Skip ban button for intentional missed bans
- Undo button to step back through selections
- Player assignment dropdowns to map battletags to specific pick slots
- Typeahead hero search clears on selection

---

## Draft Insights Hyper Pro Max

We’re experimenting with AI-powered draft insights. These use a small AI model that can run inference in browser to suggest the best characters to pick and ban, as well as the expected impact on win probability of making those picks.

This is effectively the same net result as Draft Insights, though powered by an AI model rather than statistics. The exception is player specific hero skill; the model will not intrinsically incorporate this data, so it will be combined by top level hero-win momentum adjusted win rate adjustments per character in picks, and have no impact on bans.

The UI will allow a toggle in draft insights to switch to Hyper Max Pro mode, which switches how we rank.

The complexity lies in building the datasets and training pipeline.

This involves three distinct steps, all sharing a common data collection foundation.

### Data Collection

We collect recent Storm League replay data via a continuous daemon (`sync/replay-daemon.ts`, managed by systemd).

**Discovery phase:** The `/Replay/Max` endpoint returns the latest replay ID as plain text (not JSON). Replay IDs are sequential integers. We scan backwards from max using `/Replay/Min_id`, which returns up to 1,000 replay metadata entries per call (1M calls/week/key). We filter for Storm League matches and enqueue them into `replay_fetch_queue`. Crucially, the Min_id response includes `league_tier` and `avg_mmr` metadata that is NOT available from Replay/Data.

**Fetch phase:** The `/Replay/Data` endpoint returns full replay JSON including `draft_order` (an array of 16 steps with hero, type (ban/pick), pick_number, and player_slot), plus per-player data with hero, team, and winner fields. This gives us everything needed for both the Generic Draft and Win Probability models — no `.StormReplay` file downloads are needed. Replay/Data supports 25,000 calls/account/week (50,000 total across both keys). We randomize fetch order for tier spread across low/mid/high skill tiers.

We use two API keys (`HEROES_PROFILE_API_KEY` and `HEROES_PROFILE_API_KEY2`) via a `MultiKeyApi` round-robin wrapper that alternates calls equally between them. The daemon alternates discovery batches (200 calls) and fetch batches (300 calls) continuously, with state persisted in `replay_sync_state` for resumability. Target: 100,000–200,000 games.

### The “Generic Draft” Model

This model predicts the next ban or pick based on map, skill level tier of the players (low/medium/high as specified through this document based on average of all players), current draft state, and predicts the next pick/ban.

Input: team0_picks (90) + team1_picks (90) + bans (90) + map (14) + tier (3) + step (1) + type (1) = 289 features. Output: softmax over 90 heroes, masked to valid/available heroes. Architecture: 289 → 256 → 128 → 90. Each replay produces 16 training samples (one per draft step). Trained with cross-entropy loss and early stopping. Exported to ONNX for browser inference.

At inference time, the model samples weighted randomly from the softmax distribution (with temperature scaling) rather than always picking the argmax. This stochastic noise makes the Generic Draft a more realistic and diverse opponent for training the Draft Policy model.

Training: `python training/train_generic_draft.py` (requires `DATABASE_URL` env var).

### “Win Probability” Model

This model takes the heroes drafted by both teams, map, and skill tier, then outputs the probability that team 0 wins.

Input: team0_heroes (90) + team1_heroes (90) + map (14) + tier (3) = 197 features. Output: sigmoid probability. Architecture: 197 → 128 → 64 → 1. Trained with BCE loss and early stopping. Exported to ONNX for browser inference.

Both the Generic Draft and Win Probability models train from the same `replay_draft_data` table populated by the fetch phase above.

Training: `python training/train_win_probability.py` (requires `DATABASE_URL` env var).

### “Draft Policy” Model

Optimizes for win probability going up against a Generic Draft. Outputs incremental win probability of each “move” (hero pick/ban option).

This will take into account likely counter picks, future drafts, composition, synergy, and more implicitly. We will still need to layer in player skill with specific heroes via a per move recommended probability bump based on that hero’s MAWP-50, both in that move and overall win probability for the match.

Uses Q-Learning (DQN with target network and experience replay buffer). The policy plays as team 0 against the Generic Draft model (team 1), simulating full 16-step drafts across random maps and tiers. The reward is only the final win probability as computed by the Win Probability model. There is no discount factor (gamma = 1.0). Epsilon-greedy exploration decays from 0.3 to 0.05. Evaluated against a held-out test set (2% of data).

Requires pre-trained Win Probability and Generic Draft models. Exported to ONNX for browser inference.

Training: `python training/train_draft_policy.py` (requires `DATABASE_URL` env var and pre-trained models).


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
- **ORM:** Drizzle ORM.
- **Schema design principles:**
  - Precomputed aggregate tables bucketed by skill tier (low/mid/high) for heroes, maps, talents, and pairwise hero synergy/counter stats.
  - Per-user tables for match history, hero performance, and momentum-adjusted stats.
  - All draft-relevant data (hero win rates by map, pairwise stats) must be queryable with no external API calls. The draft assistant reads only from the local DB.
- **Scale assumption:** Handful of users. Single Neon instance, free or low-paid tier. No read replicas, no sharding, no connection pooling concerns for now.

### Data Sync (Cron Job)

- **Runs on:** Debian server via systemd timer or cron.
- **Frequency:** Every few hours (configurable, start with every 4h).
- **Language:** TypeScript (Node), same codebase as the Next.js frontend.
- **What it does:**
  1. Pull aggregate data from Heroes Profile API: hero stats, map stats, talent stats, pairwise hero synergy/counter data. All bucketed by skill tier (Bronze+Silver / Gold+Plat / Diamond+Master) — including pairwise matchup data, which is fetched per tier per hero.
  2. Pull personal match history for every registered user's tracked battletags (up to 10 per account).
  3. Compute derived stats: momentum-adjusted win %, hero trends, power picks (hero+map combos ≥65% WR), pairwise synergy/counter scores.
  4. Scrape composition win rate data from Heroes Profile (role-based 5-hero team compositions with win rates and popularity, per skill tier).
  5. Write everything to Neon Postgres, replacing stale data.
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
