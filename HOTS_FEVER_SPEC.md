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

### Step 1: Data Collection

Target: **500k–1M games** for the Win Probability model, **200k+** for the Generic Draft model.

We collect recent Storm League replay data via a continuous daemon (`sync/replay-daemon.ts`, managed by systemd).

**Discovery phase:** The `/Replay/Max` endpoint returns the latest replay ID as plain text (not JSON). Replay IDs are sequential integers. We scan backwards from max using `/Replay/Min_id`, which returns up to 1,000 replay metadata entries per call (1M calls/week/key). We filter for Storm League matches and enqueue them into `replay_fetch_queue`. Crucially, the Min_id response includes `league_tier` and `avg_mmr` metadata that is NOT available from Replay/Data.

**Fetch phase:** The `/Replay/Data` endpoint returns full replay JSON including `draft_order` (an array of 16 steps with hero, type (ban/pick), pick_number, and player_slot), plus per-player data with hero, team, and winner fields. This gives us everything needed for both the Generic Draft and Win Probability models — no `.StormReplay` file downloads are needed. Replay/Data supports 25,000 calls/account/week (50,000 total across both keys). We randomize fetch order for tier spread across low/mid/high skill tiers.

We use two API keys (`HEROES_PROFILE_API_KEY` and `HEROES_PROFILE_API_KEY2`) via a `MultiKeyApi` round-robin wrapper that alternates calls equally between them. The daemon alternates discovery batches (200 calls) and fetch batches (300 calls) continuously, with state persisted in `replay_sync_state` for resumability.

### Step 2: Train Win Probability Model

This model is the game simulator and the quality ceiling for everything downstream. Every percentage point of noise here propagates into the policy.

Input: team0_heroes (90) + team1_heroes (90) + map (14) + tier (3) = 197 features. Output: sigmoid probability of team 0 winning. Architecture: 197 → 256 → 128 → 1 (wider than initially planned given the importance of this model). Trained with BCE loss and early stopping on a held-out test set (2% of data). Do not proceed to later steps until satisfied with accuracy. Exported to ONNX for browser inference.

Training: `python training/train_win_probability.py` (requires `DATABASE_URL` env var).

### Step 3: Train 3–5 Generic Draft Models

This model predicts the next ban or pick based on map, skill level tier of the players (low/medium/high as specified through this document based on average of all players), current draft state, and predicts the next pick/ban.

Input: team0_picks (90) + team1_picks (90) + bans (90) + map (14) + tier (3) + step (1) + type (1) = 289 features. Output: softmax over 90 heroes, masked to valid/available heroes. Architecture: 289 → 256 → 128 → 90. Each replay produces 16 training samples (one per draft step). Trained with cross-entropy loss and early stopping.

We train **3–5 models** with different random seeds and slight hyperparameter variation (learning rate, dropout rate). These serve as the opponent pool for AlphaZero training. Each model is evaluated on held-out data to ensure they’re all reasonable — we want diversity, not broken models. All exported to ONNX (one is used for browser inference during opponent simulation).

At inference time, the model samples weighted randomly from the softmax distribution (with temperature scaling) rather than always picking the argmax. This stochastic noise makes the Generic Draft a more realistic and diverse opponent.

Training: `python training/train_generic_draft.py` (requires `DATABASE_URL` env var). Trains all variants in one run.

### Step 4: Initialize and Train the AlphaZero-Style Draft Policy Network

This replaces the previous DQN approach with AlphaZero-style MCTS + neural network, which is far more effective for this kind of sequential decision problem.

**Architecture:**

The network uses a shared residual backbone splitting into policy and value heads:

```
Input (289)
→ FC 512, BN, ReLU
→ Residual block (FC 512 → BN → ReLU → FC 512 → BN → skip connection → ReLU)
→ Residual block (FC 512 → BN → ReLU → FC 512 → BN → skip connection → ReLU)
→ FC 256, BN, ReLU
→ FC 128, BN, ReLU
├─ Policy head: 128 → 90 (softmax, masked to legal actions)
│  Outputs prior probability of each hero being the right pick/ban at this step.
└─ Value head: 128 → 64 → 1 (tanh, scaled to [0, 1])
   Outputs estimated win probability for team 0 from the current draft state.
```

Residual connections are critical — they’re what made AlphaZero’s network trainable and they help gradient flow during self-play training. 3–4 residual blocks of fully-connected layers is plenty for this input dimensionality. A single policy head with action type encoded as input handles both bans and picks.

**Initialization:** The policy head is bootstrapped from a trained Generic Draft model’s weights. The value head is briefly pre-trained on the same data the Win Probability model used.

**Training loop (MCTS vs Generic Draft opponents):**

For each training episode: select a random map, random tier, and randomly choose one of the 3–5 Generic Draft models as the opponent.

The network plays as team 0 using MCTS. At each of team 0’s decision points, run MCTS (200–400 simulations) using the policy head as the prior and value head for leaf evaluation. At each of team 1’s decision points, sample from the selected Generic Draft model with variable temperature (randomly chosen from [0.5, 1.0, 1.5] per episode to simulate varying opponent skill/predictability).

When MCTS reaches a terminal state (draft complete), evaluate using the Win Probability model to get the outcome. After each completed draft, generate training targets: at each of team 0’s steps, the MCTS visit count distribution becomes the policy target, and the final win probability becomes the value target. Train the network on accumulated self-play data using a replay buffer of recent games (last 50k–100k drafts).

Periodically evaluate: run the current network against each Generic Draft model over 1000 drafts with the Win Probability model as judge. Track win rate over time. Also run it against the statistical Draft Insights system’s greedy recommendations for comparison.

Training: `python training/train_draft_policy.py` (requires `DATABASE_URL` env var and pre-trained models).

**Player skill integration via MAWP adjustments:**

This architecture naturally supports player-specific hero skill. During MCTS search, at each node we evaluate candidates using the policy prior and value estimate. We inject player skill by modifying value estimates at leaf nodes:

1. MCTS completes a simulated draft to a leaf/terminal state
2. Get base win probability from the Win Probability model
3. For each pick slot that has a registered player assigned, compute their MAWP delta: `(MAWP - 50) / 100`
4. Adjust: `adjusted_wp = base_wp + Σ(weight × mawp_delta)` for our team’s players (no opponent battletags supported)
5. Weight calibrated empirically — start with 0.03–0.05 per player (meaning a player with 60% MAWP on a hero shifts the team’s win probability by +0.3 to +0.5 percentage points)

### Step 5: Export to ONNX

Export the trained AlphaZero network (both heads) to ONNX. Quantize to uint8 for browser delivery.

At inference time in the browser:
- Run MCTS with ~50–200 simulations per move (tunable based on latency budget, likely 100–300ms)
- Opponent moves during search are sampled from a Generic Draft model (also exported to ONNX and running in browser)
- For registered players with assigned pick slots, apply MAWP adjustments at leaf evaluations
- The policy head’s output (after MCTS) gives the recommendation ranking
- The value head gives the running win probability estimate

**Models running in browser:** three ONNX models — the AlphaZero network (policy + value heads, ~1M params), one Generic Draft model (for simulating opponent moves during search, ~100k params), and the Win Probability model (for leaf evaluation during search, ~100k params). Total footprint roughly 1.2M params, well under 5MB even before quantization.

### Retraining Cadence

Retrain all models every major patch. The Win Probability and Generic Draft models retrain on fresh data from the new patch. The AlphaZero network re-initializes from the new Generic Draft model and runs the training loop again. Between major patches, if the replay daemon has accumulated significant new data, consider refreshing the Win Probability and Generic Draft models (since these are faster to train than the full MCTS loop).


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
