// Which tables live where (sync/docs/data-architecture-plan.md).
// Python mirror: sync/backup_local.py (keep the two in step).

/** Read by the deployed site; pushed local -> Neon by sync/publish-site.ts. */
export const SITE_TABLES = [
  'hero_stats_aggregate',
  'map_stats_aggregate',
  'hero_map_stats_aggregate',
  'hero_talent_stats',
  'hero_pairwise_stats',
  'player_match_history',
  'player_hero_stats',
  'player_hero_map_stats',
  'tracked_battletags',
] as const

/** Live expert study: Neon is the only home. Never published over, never dropped. */
export const STUDY_TABLES = ['rating_items', 'draft_ratings'] as const

/** Every table Neon keeps. drizzle.config.ts (Neon) only ever sees these. */
export const NEON_TABLES = [...SITE_TABLES, ...STUDY_TABLES] as const
