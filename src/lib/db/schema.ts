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
