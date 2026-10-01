# P3 phase 2: skill drift, momentum, uncertainty, hero similarity (2026-09-30)

Same data and windows as P3_HERO_STRENGTH.md:

- Snapshot only: Storm League games from 2024-04-01 to 2026-05-22, all ten players labeled. No post-snapshot builds were used.
- **E window:** estimation, up to the WP training cutoff.
- **V1:** 2026-02-10 to 2026-03-28, 70,974 games. Used to fit combiners.
- **V2:** 2026-03-29 to 2026-05-22, 72,474 games. The test set. The split is the median game day.

"Gain" is held-out log-loss improvement over the population WP on V2 (M0: 57.19% accuracy, log loss 0.67647). CIs are 200 replay bootstraps. Players recur across games, so these CIs are too narrow by a design effect of about 1.7 in variance: the headline (0.0131, 0.0152) becomes about (0.0127, 0.0157).

Revised 2026-10-01 after the consolidated audit (experience counts from earlier days only, offset table refit on them; drift-rate statements profiled); changes are logged in `P3_AUDIT_FIXES.md`. Results: `results/fix/p3_sd_eval.json/.txt`, `p3_sd_order.txt`, `p3_fix_cf.json`.

## Summary

- **Lagged MMR leaks at every lag, not only at 1 day.** When the lagged source game was uploaded after the game being predicted, its MMR is as predictive as the next game's MMR, our positive control. The leaky share is 42% of player-MMR sources at a 1-day lag, 35% at 7 days and 25% at 30 days. An upload-order-safe version requires the source replay_id to be at least about 2 days of uploads below the current game's. It keeps +0.0017 (G = 7), against +0.0048 naive, so about two thirds of the naive lagged-MMR value was leak. On top of the skill model, safe MMR adds +0.0006 to +0.0008. With causal at-game MMR (P3_MMR_AT_GAME) the picture is the same; see P3_HERO_STRENGTH section 6.
- **Skill drift, fitted by a Kalman filter:**
  - For the typical player, overall skill has a permanent part (sd 1.3pp), a 25-day part (1.1pp) and a 3-day "form" part (2.8pp).
  - Hero-specific drift is not detected: its rate sits at the lower bound. A profile of the rate gives a 95% interval for the correlation half-life of 2 years or longer.
  - The cross-hero factors (the CF part) drift slowly. The best profile fit is a half-life of 8 years, with a 95% interval from 3 years to no decay. The earlier point fits (2,500 and 1,620 days) had no SE.
  - For veterans (300+ games), overall skill beyond matchmaking is about zero. What remains is static hero-specific skill (2.3pp) and slow factors. There is no within-day session effect once same-day games are ordered by play time (1b).
  - Drift rates do not differ reliably by role.
- **Ties to the drift paper.** Over the 16-build lag (about 410 days), a hero's meta win rate keeps correlation 0.62 with itself. A player's hero-specific skill keeps 0.90 to 0.93 at the point estimates. Meta information halves in about 290 days. For a player's hero-specific information we can only bound the rate: no decay is detected, and the CF part's half-life is at least about 3 years. The game drifts much faster than a player's relative hero skill.
- **Head-to-head at a 1-day lag.**
  - The state-space model ties the phase-1 static model: +0.01422 vs +0.01423, paired −0.00001 (CI −0.00017, +0.00017). The 365-day decay variant is +0.01410.
  - Coverage of the state-space model's raw 80% intervals is 0.78 to 0.84 in every days-since-last-played bin for heroes already played. The realized/claimed variance ratio is 1.11 overall (static model: 1.76), but that average hides the bins: 0.62, 0.37, 0.68, 0.89 and 1.11 from 1-7 days to 181-365 days since the hero was last played, and 3.14 on never-played heroes. The state-space intervals are too wide for recently played heroes and about right after half a year. They are not uniformly calibrated.
  - When predictions may use the player's earlier games today, the drift model beats the static model by +0.00037 (0.00018, 0.00057) under the strict ordering rule (section 1b). Using same-day games at all is worth +0.0011 over the 1-day protocol.
- **Momentum and volume.**
  - Volume adds nothing beyond the skill model (+0.00000 overall, +0.00002 per hero).
  - Short-horizon form (last 10 or 20 games) adds nothing either.
  - Long-horizon EWMAs add a little: residual with a 100-game half-life +0.00071 (0.00045, 0.00098); win rate +0.00053.
  - All arms together add +0.0025 (0.0020, 0.0029).
  - A Max-style model (volume plus EWMA win rates, without the WP residual) reaches +0.0108, against +0.0142 for the skill model.
- **Uncertainty in the win estimate.** The posterior mean is the right point estimate.
  - Adding the posterior sd adds 0.00000.
  - A lower-bound (m − 1.28 s) adjustment loses 0.0012 when used alone and adds nothing on top of the mean.
  - Estimate × reliability adds +0.00012 (0.00004, 0.00019), which is negligible.
  - Calibration within uncertainty quintiles follows the population WP's own pattern.
- **Hero similarity.**
  - Beyond general skill, skill structure is mostly a two-factor space. Official roles explain little: role, fine-role and melee blocks together carry 12% of a cell's variance, against 53% for two learned factors.
  - Axis 1 runs from low-ladder, straightforward heroes (Leoric, Qhira, Malthael, Kharazim, Li Li) to high-ladder, high-execution heroes (Genji, Hanzo, Zeratul, Kerrigan, Medivh, Cho, Yrel). It correlates 0.61 with the mean MMR of each hero's players.
  - Axis 2 runs from popular mainstream picks to niche, unusual kits (Gall, Lost Vikings, Valeera, Abathur, Medivh, Cho).
  - Clusters agree with official roles no better than chance: ARI 0.009 for Blizzard roles and 0.029 for fine roles, below the 95% permutation value of 0.040.

## 0. MMR leak test (`p3_sd_mmrleak.py`)

For each slot I took the lagged source game: the player's latest game at least G days earlier. The test regresses the slot residual r = y − wp on lobby-centered source MMR, comparing sources uploaded after the current game (higher replay_id) with those uploaded before, at the same day gap. Slopes are per 100 MMR.

| player MMR | share uploaded after | slope, source after | slope, source before | before, closest id tertile | before, farthest tertile |
|---|---|---|---|---|---|
| next game (positive control) | 77% | 0.029 | | | |
| G = 1, gap 1 day | 42% | 0.032 | 0.008 | 0.011 | 0.007 |
| G = 7, gap 7 days | 35% | 0.033 | 0.007 | 0.009 | 0.005 |
| G = 30, gap 30 days | 25% | 0.032 | 0.008 | 0.011 | 0.007 |

Hero MMR gives the same picture: slope 0.020 to 0.028 when the source was uploaded after, 0.006 to 0.007 when before, and 0.019 for the positive control. The shares uploaded after are 27%, 20% and 13% at G = 1, 7 and 30.

Reading:

- Heroes Profile stamps MMR as of parse time. Replays uploaded late carry ratings that already include later results.
- Standard errors are 0.000 to 0.001, so every "after vs before" gap is many SEs wide.
- Close upload order leaks a little even among "before" sources, which is why the safe version adds a margin. I required the source replay_id to be below the current id minus 21,202, about 2 days of uploads.

Game level (player + role + hero lagged MMR, combiner fit on V1):

| lag | naive alone | safe alone | naive on top of skill model | safe on top of skill model |
|---|---|---|---|---|
| G = 1 | +0.0086 | +0.0019 | +0.0057 | +0.0008 |
| G = 7 | +0.0048 | +0.0017 | +0.0031 | +0.0008 |
| G = 30 | +0.0022 | +0.0013 | +0.0013 | +0.0006 |

- The phase-1 lagged-MMR rows (in P3_HERO_STRENGTH.md and P3_NESTED_LIFT.md) used the naive definition and overstate MMR.
- The recency advantage of fresh MMR is mostly leak: safe MMR at 1 day and at 7 days are the same.
- Recommendation: headline safe MMR only, and drop the naive 1-day row.

## 1. Skill drift (`p3_sd_kalman.py`, `p3_sd_eval.py`, `p3_sd_session.py`)

**Model.** For each player, a 202-dimensional state holds:

- overall skill in slow, medium and fast parts;
- Blizzard-role, fine-role and melee effects;
- two CF factors (loadings from the phase-1 rank-2 fit);
- hero-specific skill in slow and fast parts (90 each).

Each component is an Ornstein-Uhlenbeck process. Over short gaps it is a random walk with variance 2λs² per day, and over long gaps it forgets toward the population. Each game is one noisy observation (noise wp(1 − wp)) of the loading-weighted sum. Filtering is causal.

**Checks.** With all rates at zero, the filter reproduces the phase-1 GP posterior to machine precision (max difference 4e-17). Rates and variances were fit by maximum likelihood on E-window games, on two fit-half player samples:

- a random 40,000 players (1.27M games);
- 6,000 heavy players with 300+ games.

A finite-difference optimizer failed on the flat surface, so the fits use coordinate-wise bounded 1-D searches.

**Fitted components.** sd is in pp. "Half-life" is the correlation half-life ln 2/λ. "Min" means the rate hit the lower bound: no detectable drift, half-life above about 190 years.

| component | random 40k: sd | random 40k: half-life | heavy 6k: sd | heavy 6k: half-life |
|---|---|---|---|---|
| overall, permanent | 1.32 | min | 0.01 | |
| overall, medium | 1.05 | 25 days | 0.01 | |
| overall, fast ("form") | 2.78 | 3.4 days | 0.98 (not robust, see 1b) | 0.65 days |
| role / fine role / melee | 1.05 / 1.07 / 0.72 | static (not fit) | same | |
| CF factors (per-cell sd 3.5) | | 2,500 days (profile: best 8 y, 95% 3 y to none) | | 1,620 days |
| hero-specific, slow | 1.98 | min (profile: 95% ≥ 2 y) | 2.31 | min |
| hero-specific, fast | 0.00 | | 0.93 | 3,390 days |
| log-lik gain vs static | +30.5 | | +79.2 | |

Findings:

- **Clock.** A calendar-day clock beats a game-count clock: +30.5 vs +21.6 on the random sample and +79.2 vs +77.3 on heavy players. Skill decays with time, not with games played.
- **Role rates.** Letting the hero rates differ by Blizzard role adds +1.5 and +0.6. The role that seems to drift differs between samples (bruisers in one, tanks in the other), so there is no reliable role difference.
- **Overall-skill spread shrinks with volume.** The noise-corrected sd of a player's overall residual is 5.3pp at 20 to 50 games, 3.9 at 50 to 100, 2.7 at 100 to 300, 1.5 at 300 to 1,000 and 1.1 at 1,000+. On heavy players, forcing the permanent overall part to 1pp costs 17.5 log-lik units. Matchmaking prices in general skill once a player has played enough; per-hero skill is what it cannot see.

**How fast information is lost.** Skill correlation at lag D is the share of today's cell variance knowable D days ago (`p3_sd_autocorr.txt`):

| lag | random 40k: whole cell | random 40k: hero-specific part | heavy: whole cell | heavy: hero-specific part |
|---|---|---|---|---|
| 1 day | 0.95 | 1.00 | 0.97 | 1.00 |
| 7 days | 0.79 | 1.00 | 0.96 | 1.00 |
| 30 days | 0.71 | 0.99 | 0.95 | 0.99 |
| 90 days | 0.69 | 0.98 | 0.94 | 0.98 |
| 365 days | 0.66 | 0.94 | 0.87 | 0.91 |
| 410 days (about 16 builds) | 0.66 | 0.93 | 0.86 | 0.90 |
| 730 days | 0.62 | 0.88 | 0.80 | 0.84 |

**Half-lives of information** (squared correlation falling to 0.5):

- The hero-specific part takes about 6 years (random sample) or 4 years (heavy players) at the point estimates. These come from the CF rates, which the profile leaves wide open (3 years to no decay), so read them as "years, not months".
- The whole cell for a typical player takes about a month, because a quarter of its variance is short-lived form.
- The whole cell for a veteran takes about 3.5 years.

**Comparison with the drift paper.** Disattenuated hero win-rate correlation falls from 0.94 at 1 build to 0.62 at 16 builds. Since 2024, 16 builds span about 410 days (median 18 days per build). Fitting the slow part gives a time constant of 940 days, and the meta's squared correlation reaches 0.5 at about 290 days. At the same 410-day lag:

- a hero's strength in the meta: 0.62;
- a player's strength on that hero relative to their other heroes: 0.90 to 0.93;
- a veteran's whole cell: 0.86.

For the merged paper: the meta moves on a scale of months. A player's hero-specific skill moves on a scale of years, and their overall level mostly on a scale of days to weeks. The paper-2 practice of refreshing hero statistics every build is right for the meta. Per-player hero skill can be estimated from years of history without decay, provided residuals are taken against a drift-aware WP.

**Head-to-head** (V2 games; predictions use games through the previous day unless noted):

| model | acc % | log loss | gain vs M0 (95% CI) | vs state-space |
|---|---|---|---|---|
| experience offset only | 59.00 | 0.66815 | +0.0083 (0.0074, 0.0092) | −0.0059 |
| EB static kernel, online (phase-1 winner) | 60.02 | 0.66224 | +0.01423 (0.01312, 0.01519) | +0.00001 (−0.00017, 0.00017) |
| EB exponential decay, half-life 365 days | 59.85 | 0.66237 | +0.01410 (0.01300, 0.01512) | −0.00012 (−0.00030, 0.00006) |
| **state-space, calendar clock** | 59.92 | 0.66225 | +0.01422 (0.01307, 0.01523) | |
| state-space, hero rates by role | 59.89 | 0.66231 | +0.01416 | −0.00006 (−0.00009, −0.00003) |
| state-space, game-count clock | 59.93 | 0.66232 | +0.01415 | −0.00007 (−0.00020, 0.00004) |

These rows match phase 1's +0.0142. Same-day rows are in 1b.

### 1b. Same-day information and the ordering rule (`p3_sd_order.py`, `results/fix/p3_sd_order.txt`)

Experience counts always use earlier days only. This section asks what a player's earlier games from the same day add to the residual state, and how they must be ordered. `game_date` is the game's end time: consecutive games of one player overlap in 0.2% of pairs under that reading and 7% if it were the start. So start = game_date − game_length. Upload order (replay_id) reverses play order in 29.5% of a player's consecutive same-day game pairs.

Three orderings of a player's games within a day:

- **rid:** replay_id order (upload order).
- **time:** game_date order, replay_id as tiebreak.
- **strict:** time order, and an earlier same-day game counts only if it ended before the current game started and has a lower replay_id. This is the only rule that uses information available at draft time.

The fast components were refit under each rule, on the same samples, with other parameters fixed at the drift fit.

| | rid | time | strict |
|---|---|---|---|
| random 40k: fast player sd / half-life | 2.91pp / 3.4 days | 2.91pp / 3.4 days | 2.74pp / 3.6 days |
| random 40k: log-lik from the fast part | +26.8 | +26.8 | +19.1 |
| heavy 6k: fast player sd / half-life | 1.14pp / 15h | 1.14pp / 15h | 0.01pp |
| heavy 6k: log-lik from the fast part | +0.5 | +0.5 | 0.0 |
| same-day, static vs lag 1 day (game-level gain) | +0.00091 (0.00078, 0.00102) | +0.00091 (0.00081, 0.00103) | +0.00077 (0.00067, 0.00087) |
| same-day, drift vs static | +0.00048 (0.00025, 0.00070) | +0.00046 (0.00025, 0.00069) | +0.00037 (0.00018, 0.00057) |
| same-day drift vs lag 1 day | +0.00139 (0.00109, 0.00165) | +0.00137 (0.00108, 0.00165) | +0.00113 (0.00088, 0.00138) |

Game level under the strict rule: static with same-day games +0.01500 (60.09%), drift with same-day games +0.01536 (60.16%), against +0.01423 for the static model a day ahead.

Reading:

- **Time ordering changes nothing.** With no time step inside a day, the joint likelihood of a day's games does not depend on their order. Games used are also nearly the same.
- **The strict rule is what bites.** It removes games that overlapped in time or were uploaded later, and cuts the same-day value by about a sixth.
- **The typical player's fast component survives:** 2.7pp, half-life about 3.6 days, with a smaller log-likelihood contribution.
- **The veteran "session effect" does not.** Once the fast hero part is free, the fast player part adds +0.5 log-lik units under replay_id order and nothing under the strict rule.
- **Headline:** same-day information is worth +0.0011, and the drift model's same-day edge over the static model is +0.0004, both under the strict rule.

Per-slot gain a day ahead (×1000 per slot, `p3_sd_session.py` on the lag-1 contract, `results/fix/p3_sd_session.txt`) grows with position in the day's session: the static model scores 1.11 on the first game of the day, 1.45 on games 2-3 and 2.43 on game 4+ (state-space 1.13, 1.47, 2.46). Players in long sessions are better predicted even without today's games.

**Interval coverage by recency.** The posterior is frozen at the V1/V2 split and compared with each cell's V2 mean residual (cells with ≥ 10 V2 games). Values are coverage of the 80% interval / realized-to-claimed variance ratio.

| days since hero last played | cells | EB static (raw) | state-space (raw) | state-space claimed sd pp |
|---|---|---|---|---|
| all | 8,678 | 0.79 / 1.76 | **0.80 / 1.11** | 4.6 |
| never played | 1,407 | 0.73 / 4.22 | 0.73 / 3.14 | 5.2 |
| 1-7 | 3,736 | 0.80 / 1.08 | 0.82 / 0.62 | 4.4 |
| 8-30 | 1,786 | 0.81 / 0.59 | 0.82 / 0.37 | 4.5 |
| 31-90 | 929 | 0.79 / 1.11 | 0.79 / 0.68 | 4.6 |
| 91-180 | 365 | 0.80 / 1.43 | 0.81 / 0.89 | 4.8 |
| 181-365 | 321 | 0.77 / 1.65 | 0.78 / 1.11 | 4.9 |
| 366+ | 134 | 0.83 / (noisy) | 0.84 / (noisy) | 5.0 |

- The static model's error variance grows with time since the hero was last played, from 1.08 to 1.65 times its claim. The state-space model widens its band with recency (4.4 to 5.0pp) and stays at 0.37 to 1.11 of its claim, so it over-covers recently played heroes.
- The cells here are those with 10+ games in V2, which selects on continued play. On never-played heroes that selection matters; see P3_HERO_STRENGTH section 5 for the unselected test.
- Per-slot variance tests are not usable. Single-game noise (0.24) is about 150 times the skill variance, so small WP calibration errors dominate e² − v.

## 2. Momentum, volume and form (`p3_sd_eval.py`)

These are proxies for Max's signals until his formula arrives. All are causal (games before the current day).

- EWMA arms are computed over the player's own games, with half-lives in games.
- Win-rate EWMAs use y − 0.5. Residual EWMAs use y − wp, the "surprise-weighted" version.
- Volume is log(1 + games in the previous 365 days).

| arm | alone: gain vs M0 | added to state-space: extra gain (95% CI) |
|---|---|---|
| games in last 365 days, all heroes | +0.00002 | +0.00000 |
| games in last 365 days, this hero | +0.0039 | +0.00004 (−0.00008, 0.00016) |
| EWMA win rate, 10 / 30 / 100 games | +0.0018 / +0.0027 / +0.0032 | +0.00004 / +0.00025 / +0.00053 (0.00030, 0.00080) |
| EWMA residual, 10 / 30 / 100 games | +0.0021 / +0.0031 / +0.0037 | +0.00009 / +0.00038 / **+0.00071 (0.00045, 0.00098)** |
| EWMA hero win rate, 10 games on hero | +0.0026 | +0.00022 (0.00007, 0.00037) |
| form: mean residual of last 10 / 20 games | +0.0012 / +0.0018 | +0.00001 / +0.00006 |
| Max-style: volume (all + hero) + EWMA WR 10/30/100 | +0.0108 | |
| all arms together | | +0.0025 (0.0020, 0.0029) |
| all arms + safe lagged MMR (G = 7) | | +0.0030 (0.0025, 0.0035) |

Reading:

- Volume carries no information beyond the skill model and experience offset, which already use game counts.
- Short-term form measured as the last 10 to 20 games adds nothing. The state-space fast component takes it up, and it only helps within the day.
- What survives is a long-horizon (100-game) recent level, worth about 0.0005 to 0.0007 on its own and 0.0025 with everything together. The skill model shrinks overall skill hard for heavy players; a long EWMA lets the combiner put some of it back.
- The surprise-weighted residual beats plain win rate at every horizon, alone and on top of the model.
- Max's style of signal (volume plus momentum win rate) gets 76% of the skill model's gain (+0.0108 vs +0.0142) without any WP residual. Most of that comes from per-hero volume, the same thing the experience offset captures.

## 3. Uncertainty in the win estimate (`p3_sd_eval.py`)

| added to logit(WP) | EB static | state-space |
|---|---|---|
| posterior mean (reference) | +0.01423 | +0.01422 |
| mean + team sd sum | −0.00001 | −0.00001 |
| lower bound m − 1.28 s alone | −0.0017 | −0.0012 |
| mean + lower bound | −0.00001 | −0.00001 |
| estimate × reliability alone | −0.0089 | −0.0092 |
| mean + estimate × reliability | +0.00010 (n.s.) | +0.00012 (0.00004, 0.00019) |

Calibration was checked within quintiles of total uncertainty (the sum of the ten slots' posterior variance; mean slot sd 4.4 to 5.1pp). For the state-space combiner the calibration slopes are 0.99, 0.96, 1.04, 1.02, 1.12 and ECE is 0.005 to 0.011. The population WP alone shows the same pattern: slopes 0.96, 0.89, 1.05, 1.02, 1.11. The drift in slope comes from which games have uncertain lobbies (newer accounts). Personalization adds none.

The posterior mean already is the uncertainty-weighted estimate. Uncertainty belongs in the display, not in the win probability.

## 4. Hero similarity through common strengths (`p3_sd_similarity.py`, `p3_sd_similarity_viz.py`)

**Variance of a player×hero cell** (phase-1 kernel, 4.8pp total sd):

| component | general skill | Blizzard role | fine role | melee/ranged | 2 learned factors | hero-specific |
|---|---|---|---|---|---|---|
| share | 13% | 5% | 5% | 2% | 53% | 21% |

**Embedding dimension** (pure factorization, no side features). Held-out log likelihood gain over player+hero:

| rank | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| gain | +178 | **+248** | +228 | +145 | +123 | +80 | +83 |

Rank 2 is chosen. Adding role, fine-role and melee side features on top raises it further (phase 1: +694 vs +461 over player-only for side features alone), so side features and factors are complementary.

**Axes** (`fig_hero_embedding.png`):

- **Axis 1** (sd 2.6pp):
  - One end: Leoric, Qhira, Malthael, Kharazim, Mei, Li Li, The Butcher.
  - Other end: Yrel, Medivh, Cho, Genji, Kerrigan, Hanzo, Zeratul, Alarak.
  - Correlates 0.61 with the mean MMR of the hero's players and 0.30 with a hand-labeled "high mechanics" flag.
  - Label: "low-ladder, straightforward heroes vs high-ladder, high-execution heroes". Players who are relatively strong on one end are relatively weak on the other.
- **Axis 2** (sd 2.3pp):
  - One end: Probius, Jaina, Thrall, Lunara, Imperius, Chen.
  - Other end: Gall, The Lost Vikings, Valeera, Cho, Medivh, Abathur.
  - Correlates −0.53 with hero popularity and +0.43 with the mechanics flag.
  - Label: "mainstream picks vs niche, unusual kits". Two-player and split-control heroes (Cho'Gall, Lost Vikings, Abathur, Medivh) sit at the top.

**Clusters** (Ward on embedding coordinates, k = 6). ARI vs Blizzard role is 0.009 and vs fine role 0.029, below the permutation 95% value of 0.040 at every k from 4 to 8.

| cluster | heroes (role) |
|---|---|
| straightforward / low-ladder (24) | Leoric (B), Kharazim (H), Qhira (M), Mei (T), Butcher (M), Li Li (H), Samuro (M), Malthael (B), Anduin (H), Stitches (T), Nazeebo, Stukov (H), Kael'thas, Ragnaros (B), Tassadar, Lt. Morales (H), Azmodan, Nova, Artanis (B), Deckard (H), Gazlowe (B), Xul (B), Brightwing (H), Mal'Ganis (T) |
| mainstream core (36) | Probius, Thrall, Jaina, Falstad, Imperius, Valla, Muradin, Orphea, Rexxar, Illidan, Lunara, Chromie, Gul'dan, Garrosh, Diablo, Dehaka, Auriel, Johanna, Zagara, Sonya, Tyrande, Blaze, Zul'jin, Malfurion, E.T.C., Cassia, Rehgar, Raynor, Uther, Lúcio, Anub'arak, Tychus, Arthas, Tyrael, Whitemane, Varian (all six roles) |
| mechanical carries (10) | Kerrigan (M), Alarak (M), Zeratul (M), Zarya (S), Chen (B), Hogger (B), Li-Ming, Sylvanas, Tracer, Greymane |
| high-execution showpieces (6) | Cho (T), Medivh (S), Yrel (B), Genji, Hanzo, Maiev (M) |
| two-body heroes (2) | Gall, The Lost Vikings |
| niche kits (12) | Valeera (M), Mephisto, Murky, Kel'Thuzad, Abathur (S), Ana (H), Sgt. Hammer, Junkrat, D.Va (B), Fenix, Deathwing (B), Alexstrasza (H) |

Unmarked heroes are ranged assassins.

**Why heroes cross role lines.** Skill beyond general skill travels with execution demand, ladder level and kit oddity, not with the role a hero fills:

- The same players are relatively good on Genji, Hanzo and Maiev and on Cho, Medivh and Yrel. These are high-execution picks from five different roles.
- Players relatively good on Leoric, Butcher, Li Li, Kharazim and Mei tend to be relatively weak on those.
- The auto-attack vs mage split, which the fine roles encode, is not a strength axis. Ranged AA heroes are spread across the mainstream core and the carry cluster.
- Global heroes (Dehaka, Falstad, Brightwing, Abathur, Zagara) also do not group.

**Top cross-role pairs** by shared factor covariance, among heroes with sizable loadings (all are corr ≥ 0.8 in the factor space):

| pair | shared factor cov (pp²) | held-out empirical cov (pp²) | held-out players with both |
|---|---|---|---|
| Cho (T) + Gall (RA) | 55 | +52 | 30 |
| Gall (RA) + Medivh (S) | 53 | +96 | 22 |
| Cho (T) + Medivh (S) | 50 | −2 | 38 |
| Cho (T) + Yrel (B) | 46 | −67 | 25 |
| Medivh (S) + Yrel (B) | 46 | −37 | 48 |
| Genji (RA) + Cho (T) | 43 | +46 | 52 |
| Genji (RA) + Medivh (S) | 43 | +23 | 213 |
| Genji (RA) + Yrel (B) | 42 | −12 | 110 |
| Cho (T) + Maiev (M) | 40 | +18 | 44 |
| Hanzo (RA) + Yrel (B) | 40 | +4 | 151 |
| Medivh (S) + Maiev (M) | 39 | +28 | 158 |
| Hanzo (RA) + Cho (T) | 38 | +58 | 77 |
| Hanzo (RA) + Medivh (S) | 38 | +25 | 282 |
| Zeratul (M) + Yrel (B) | 36 | +10 | 81 |

- Cho+Gall is one hero played by two people; a strong Cho usually plays in a coordinated duo, so its pairing with Gall is expected.
- Pair-level empirical values are noisy: the median split-half reliability of a single cell is 0.10.

**Held-out validation.** The embedding is fit on half the players; the check uses the other half. Empirical excess covariance rises across deciles of the model's predicted covariance: −3.6, −1.3, −2.3, −0.8, −1.1, −0.4, 0.0, +0.9, +1.7, +5.1 pp². The weighted slope is 0.35: the model direction is right, but it overstates the spread by about 3x, as expected when ranking on noisy fits.

The pure 2-factor model understates role structure. Held-out same-role pairs covary more than cross-role pairs (5.2 vs 3.1 pp²), while the model has 3.3 vs 2.8. This is why the production kernel keeps role blocks alongside the factors.

**Figures.**

- `fig_hero_skill_corr.png`: two panels. Left is the model's correlation of player skill across heroes, including general skill. Right is the held-out empirical disattenuated correlation (noisy). Both are ordered by cluster, and label colors mark Blizzard roles.
- `fig_hero_embedding.png`: the 2-D hero map.

## Follow-ups

Two follow-up hypotheses are tested in P3_PATCH_AND_ROLE.md:

- **No patch-specific reshuffle of which players are good on a hero is detected**, in or out of sample, but jumps up to about 1.25pp sd cannot be ruled out.
- **Off-role play is associated with about 0.5 to 0.8pp lower residual per off-role slot beyond the skill model** in the post-cutoff window (SE about 0.2). Role terms add +0.0005 in log loss.

## Product and paper implications

- **Model to run.** Use the state-space model: a calendar clock, the random-sample fit, and today's earlier games included.
  - It ties the static model a day ahead and wins +0.0004 when it can see today's games (strict play-time rule, 1b).
  - Its intervals need no hand calibration on average, but they are too wide for heroes played in the last month (variance ratio 0.4 to 0.7) and too narrow for never-played heroes.
  - Display: strength ± band, games played, and "last played N days ago". At the point estimate the band widens by about 0.15pp per 100 days unplayed, from the slow factor drift; the drift rate itself is poorly determined.
- **MMR.** Use only the upload-order-safe lagged MMR. The naive lagged MMR used in phase 1 and in P3_NESTED_LIFT overstated MMR's value by about 3x.
- **Momentum.** A long-horizon (100-game) EWMA of residuals is the one momentum arm worth keeping in the combiner. Volume and short form are not.
- **Framing for the merged paper.** Per-player hero skill is stable on the scale of years, while hero strength in the meta turns over within a year. The drift-aware WP moves quickly so that the personal layer can move slowly.

## Caveats

- The drift fits use 1.27M (random) and about 1.2M (heavy) E-window games.
- Parameter uncertainty is not bootstrapped. Differences under about 2 log-lik units (the role splits) should be read as zero.
- The heavy-player fit selects on volume.
- Experience counts use earlier days only (lag-1 contract). Within-day order matters only for the same-day rows, which use the strict play-time rule in 1b.
- "Earlier days" means every game eventually uploaded with an earlier date, not what was uploaded by draft time (audit P3-04). The audit measured the cost of that difference at about 0.0004 on the headline. Production should filter on ingestion time.
- Variance-component fits assume game noise wp(1 − wp). In-sample residuals are about 2 to 4% less noisy (the jump test in P3_PATCH_AND_ROLE profiles the scale at 0.96), so latent variances and drift rates are somewhat overstated.
- The meta comparison uses the drift paper's disattenuated correlations converted to days with the median build length since 2024 (18 days).
- The coordinate-wise optimizer can stop at a local optimum. The random and heavy fits agree on the static hero-specific result and on a fast player part.

## Files (training/personalization/)

- `p3_sd_mmrleak.py`: MMR leak test (`results/p3_sd_mmrleak.json/.txt`).
- `p3_sd_kalman.py`: state-space model and fits (`cache/sd_params.json`, `results/p3_sd_fit.json/.txt`). `results/p3_sd_heavy_check.txt` holds the volume check and `results/p3_sd_autocorr.json/.txt` the lag curves.
- `p3_sd_eval.py`: head-to-head, momentum arms, uncertainty tests, safe vs naive MMR, and coverage by recency (`results/fix/p3_sd_eval.json/.txt`).
- `p3_fix_counts.py`: the lag-1 count contract and `run_fixed`, through which eval, session and order were run. `p3_fix_arms.py cf`: drift-rate profiles (`results/fix/p3_fix_cf.json`).
- `p3_sd_session.py`: same-day information test (`results/fix/p3_sd_session.json/.txt`).
- `p3_fetch_gametime.py`, `p3_sd_order.py`: play-time ordering check (`cache/gametime_2024q2.npz`, `results/fix/p3_sd_order.txt`).
- `p3_sd_similarity.py` and `p3_sd_similarity_viz.py`: correlation matrix, embedding selection, axes, clusters, held-out validation and figures (`results/p3_sd_similarity*.json/.txt`, `fig_hero_skill_corr.png`, `fig_hero_embedding.png`).

To reproduce, run from `training/` with `/usr/bin/python3` after the phase-1 pipeline: kalman → eval → session → similarity → similarity_viz. The mmrleak step is independent. The full run takes about 2 hours on CPU; no GPU was used.
