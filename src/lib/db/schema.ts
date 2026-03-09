import {
  pgTable,
  pgEnum,
  serial,
  varchar,
  integer,
  real,
  boolean,
  timestamp,
  text,
  uniqueIndex,
  index,
  jsonb,
} from 'drizzle-orm/pg-core'

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

export const skillTierEnum = pgEnum('skill_tier', ['low', 'mid', 'high'])

// ---------------------------------------------------------------------------
// Users & Battletags
// ---------------------------------------------------------------------------

export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  email: varchar('email', { length: 255 }).notNull().unique(),
  displayName: varchar('display_name', { length: 100 }),
  createdAt: timestamp('created_at').defaultNow().notNull(),
})

export const trackedBattletags = pgTable(
  'tracked_battletags',
  {
    id: serial('id').primaryKey(),
    userId: integer('user_id')
      .notNull()
      .references(() => users.id, { onDelete: 'cascade' }),
    battletag: varchar('battletag', { length: 100 }).notNull(),
    region: integer('region').default(1), // 1=US, 2=EU, 3=KR
    addedAt: timestamp('added_at').defaultNow().notNull(),
    lastSynced: timestamp('last_synced'),
  },
  (t) => ({
    userBattletagIdx: uniqueIndex('user_battletag_idx').on(t.userId, t.battletag),
  })
)

// ---------------------------------------------------------------------------
// Aggregate stats — precomputed by cron, bucketed by skill tier
// ---------------------------------------------------------------------------

/** Per-hero aggregate stats for a skill tier (current patch) */
export const heroStatsAggregate = pgTable(
  'hero_stats_aggregate',
  {
    id: serial('id').primaryKey(),
    hero: varchar('hero', { length: 80 }).notNull(),
    skillTier: skillTierEnum('skill_tier').notNull(),
    games: integer('games').notNull().default(0),
    wins: integer('wins').notNull().default(0),
    winRate: real('win_rate').notNull().default(0),
    banRate: real('ban_rate').default(0),
    pickRate: real('pick_rate').default(0),
    avgKills: real('avg_kills').default(0),
    avgDeaths: real('avg_deaths').default(0),
    avgAssists: real('avg_assists').default(0),
    avgHeroDamage: real('avg_hero_damage').default(0),
    avgSiegeDamage: real('avg_siege_damage').default(0),
    avgHealing: real('avg_healing').default(0),
    avgExperience: real('avg_experience').default(0),
    // NOTE: avg_damage_soaked, avg_merc_captures, avg_self_healing, avg_time_dead
    // are defined in the app-layer types but NOT yet migrated to the actual DB.
    // Omitted here to prevent "column does not exist" errors.
    patchTag: varchar('patch_tag', { length: 40 }),
    updatedAt: timestamp('updated_at').defaultNow().notNull(),
  },
  (t) => ({
    heroTierIdx: uniqueIndex('hero_tier_idx').on(t.hero, t.skillTier),
  })
)

/** Per-map aggregate stats for a skill tier */
export const mapStatsAggregate = pgTable(
  'map_stats_aggregate',
  {
    id: serial('id').primaryKey(),
    map: varchar('map', { length: 80 }).notNull(),
    skillTier: skillTierEnum('skill_tier').notNull(),
    games: integer('games').notNull().default(0),
    // Could track side win rates etc. in future
    updatedAt: timestamp('updated_at').defaultNow().notNull(),
  },
  (t) => ({
    mapTierIdx: uniqueIndex('map_tier_idx').on(t.map, t.skillTier),
  })
)

/** Hero performance per map per skill tier */
export const heroMapStatsAggregate = pgTable(
  'hero_map_stats_aggregate',
  {
    id: serial('id').primaryKey(),
    hero: varchar('hero', { length: 80 }).notNull(),
    map: varchar('map', { length: 80 }).notNull(),
    skillTier: skillTierEnum('skill_tier').notNull(),
    games: integer('games').notNull().default(0),
    wins: integer('wins').notNull().default(0),
    winRate: real('win_rate').notNull().default(0),
    updatedAt: timestamp('updated_at').defaultNow().notNull(),
  },
  (t) => ({
    heroMapTierIdx: uniqueIndex('hero_map_tier_idx').on(t.hero, t.map, t.skillTier),
  })
)

/** Talent stats per hero per tier */
export const heroTalentStats = pgTable(
  'hero_talent_stats',
  {
    id: serial('id').primaryKey(),
    hero: varchar('hero', { length: 80 }).notNull(),
    skillTier: skillTierEnum('skill_tier').notNull(),
    talentTier: integer('talent_tier').notNull(), // 1, 4, 7, 10, 13, 16, 20
    talentName: varchar('talent_name', { length: 120 }).notNull(),
    games: integer('games').notNull().default(0),
    wins: integer('wins').notNull().default(0),
    winRate: real('win_rate').notNull().default(0),
    pickRate: real('pick_rate').default(0),
    updatedAt: timestamp('updated_at').defaultNow().notNull(),
  },
  (t) => ({
    heroTalentIdx: uniqueIndex('hero_talent_idx').on(
      t.hero,
      t.skillTier,
      t.talentTier,
      t.talentName
    ),
  })
)

/**
 * Pairwise hero stats: hero A with/against hero B per skill tier.
 * "with" = same team synergy, "against" = counter/matchup.
 */
export const heroPairwiseStats = pgTable(
  'hero_pairwise_stats',
  {
    id: serial('id').primaryKey(),
    heroA: varchar('hero_a', { length: 80 }).notNull(),
    heroB: varchar('hero_b', { length: 80 }).notNull(),
    relationship: varchar('relationship', { length: 10 }).notNull(), // 'with' | 'against'
    skillTier: skillTierEnum('skill_tier').notNull(),
    games: integer('games').notNull().default(0),
    wins: integer('wins').notNull().default(0),
    winRate: real('win_rate').notNull().default(0),
    updatedAt: timestamp('updated_at').defaultNow().notNull(),
  },
  (t) => ({
    pairwiseIdx: uniqueIndex('pairwise_idx').on(
      t.heroA,
      t.heroB,
      t.relationship,
      t.skillTier
    ),
    // Reverse lookup: find all heroes that pair well with heroB
    reversePairIdx: index('reverse_pair_idx').on(t.heroB, t.relationship, t.skillTier),
  })
)

// ---------------------------------------------------------------------------
// Personal player data — synced for tracked battletags only
// ---------------------------------------------------------------------------

/** Individual match records for tracked battletags */
export const playerMatchHistory = pgTable(
  'player_match_history',
  {
    id: serial('id').primaryKey(),
    battletag: varchar('battletag', { length: 100 }).notNull(),
    replayId: varchar('replay_id', { length: 100 }).notNull(),
    hero: varchar('hero', { length: 80 }).notNull(),
    map: varchar('map', { length: 80 }).notNull(),
    win: boolean('win').notNull(),
    gameDate: timestamp('game_date').notNull(),
    gameLength: integer('game_length'), // seconds
    kills: integer('kills').default(0),
    deaths: integer('deaths').default(0),
    assists: integer('assists').default(0),
    heroDamage: integer('hero_damage').default(0),
    siegeDamage: integer('siege_damage').default(0),
    healing: integer('healing').default(0),
    experience: integer('experience').default(0),
    // Store full talent build as JSON array
    talents: jsonb('talents'),
    gameMode: varchar('game_mode', { length: 40 }),
    rank: varchar('rank', { length: 40 }),
    createdAt: timestamp('created_at').defaultNow().notNull(),
  },
  (t) => ({
    playerReplayIdx: uniqueIndex('player_replay_idx').on(t.battletag, t.replayId),
    playerHeroIdx: index('player_hero_idx').on(t.battletag, t.hero),
    playerDateIdx: index('player_date_idx').on(t.battletag, t.gameDate),
  })
)

/** Pre-aggregated per-hero stats for a tracked battletag, including MAWP */
export const playerHeroStats = pgTable(
  'player_hero_stats',
  {
    id: serial('id').primaryKey(),
    battletag: varchar('battletag', { length: 100 }).notNull(),
    hero: varchar('hero', { length: 80 }).notNull(),
    games: integer('games').notNull().default(0),
    wins: integer('wins').notNull().default(0),
    winRate: real('win_rate').notNull().default(0),
    mawp: real('mawp'), // Momentum-adjusted win percentage
    avgKills: real('avg_kills').default(0),
    avgDeaths: real('avg_deaths').default(0),
    avgAssists: real('avg_assists').default(0),
    // Trend: is this hero improving or declining?
    recentWinRate: real('recent_win_rate'), // last 20 games
    trend: real('trend'), // recentWinRate - winRate
    updatedAt: timestamp('updated_at').defaultNow().notNull(),
  },
  (t) => ({
    playerHeroStatsIdx: uniqueIndex('player_hero_stats_idx').on(t.battletag, t.hero),
  })
)

/** Per-hero per-map stats for a tracked battletag */
export const playerHeroMapStats = pgTable(
  'player_hero_map_stats',
  {
    id: serial('id').primaryKey(),
    battletag: varchar('battletag', { length: 100 }).notNull(),
    hero: varchar('hero', { length: 80 }).notNull(),
    map: varchar('map', { length: 80 }).notNull(),
    games: integer('games').notNull().default(0),
    wins: integer('wins').notNull().default(0),
    winRate: real('win_rate').notNull().default(0),
    updatedAt: timestamp('updated_at').defaultNow().notNull(),
  },
  (t) => ({
    playerHeroMapIdx: uniqueIndex('player_hero_map_idx').on(t.battletag, t.hero, t.map),
  })
)

// ---------------------------------------------------------------------------
// Sync tracking
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Replay data — for AI model training (Draft Insights Hyper Pro Max)
// ---------------------------------------------------------------------------

/**
 * Replay draft data extracted from Replay/Data endpoint.
 * Contains everything needed for Win Probability and Generic Draft models:
 * - Full draft order (bans + picks)
 * - Team compositions with winner
 * - Map, tier, MMR
 */
export const replayDraftData = pgTable(
  'replay_draft_data',
  {
    replayId: integer('replay_id').primaryKey(),
    region: integer('region').notNull(),
    gameMap: varchar('game_map', { length: 80 }).notNull(),
    gameDate: timestamp('game_date').notNull(),
    gameLength: integer('game_length'), // seconds
    gameVersion: varchar('game_version', { length: 40 }).notNull(),
    avgMmr: real('avg_mmr'),
    leagueTier: integer('league_tier'), // 1-6 (Bronze-Master)
    // Draft order: array of { pick_number, type (0=ban,1=pick), player_slot, hero }
    draftOrder: jsonb('draft_order').notNull(),
    // Team compositions
    team0Heroes: jsonb('team0_heroes').notNull(), // string[] of 5 heroes
    team1Heroes: jsonb('team1_heroes').notNull(), // string[] of 5 heroes
    team0Bans: jsonb('team0_bans').notNull(),     // string[] of banned heroes
    team1Bans: jsonb('team1_bans').notNull(),     // string[] of banned heroes
    winner: integer('winner').notNull(), // 0 or 1
    // Skill tier bucket (low/mid/high) derived from league_tier
    skillTier: varchar('skill_tier', { length: 10 }).notNull(),
    fetchedAt: timestamp('fetched_at').defaultNow().notNull(),
  },
  (t) => ({
    mapIdx: index('replay_draft_map_idx').on(t.gameMap),
    tierIdx: index('replay_draft_tier_idx').on(t.skillTier),
    dateIdx: index('replay_draft_date_idx').on(t.gameDate),
  })
)

/**
 * Tracks the replay sync cursor so the daemon can resume.
 * Stores the current scan position and high-water mark.
 */
export const replaySyncState = pgTable('replay_sync_state', {
  id: serial('id').primaryKey(),
  // Discovery cursor: next min_id to query Replay/Min_id with
  discoveryCursor: integer('discovery_cursor').notNull().default(0),
  // Highest known replay ID (from Replay/Max)
  maxKnownId: integer('max_known_id').notNull().default(0),
  // How many replays we've discovered (passed filtering)
  discoveredCount: integer('discovered_count').notNull().default(0),
  // How many we've fetched full data for
  fetchedCount: integer('fetched_count').notNull().default(0),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
})

/**
 * Queue of replay IDs discovered via Min_id that still need
 * full data fetched via Replay/Data.
 */
export const replayFetchQueue = pgTable(
  'replay_fetch_queue',
  {
    replayId: integer('replay_id').primaryKey(),
    gameMap: varchar('game_map', { length: 80 }),
    leagueTier: integer('league_tier'),
    avgMmr: real('avg_mmr'),
    gameVersion: varchar('game_version', { length: 40 }),
    discoveredAt: timestamp('discovered_at').defaultNow().notNull(),
    fetched: boolean('fetched').notNull().default(false),
  },
  (t) => ({
    fetchedIdx: index('replay_queue_fetched_idx').on(t.fetched),
  })
)

// ---------------------------------------------------------------------------
// Sync tracking
// ---------------------------------------------------------------------------

export const syncLog = pgTable('sync_log', {
  id: serial('id').primaryKey(),
  syncType: varchar('sync_type', { length: 40 }).notNull(), // 'aggregate' | 'player'
  battletag: varchar('battletag', { length: 100 }), // null for aggregate syncs
  status: varchar('status', { length: 20 }).notNull(), // 'running' | 'success' | 'error'
  matchesProcessed: integer('matches_processed').default(0),
  errorMessage: text('error_message'),
  startedAt: timestamp('started_at').defaultNow().notNull(),
  completedAt: timestamp('completed_at'),
})
