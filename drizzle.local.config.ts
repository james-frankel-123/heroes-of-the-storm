import 'dotenv/config'
import { defineConfig } from 'drizzle-kit'

// Local primary store (all sync writers, research, production refresh).
// Tables managed outside schema.ts (player_privacy, privacy_feed_state,
// player_history_marks, *_backup_*) are left alone by the filter below.
const SCHEMA_TABLES = [
  'users', 'tracked_battletags', 'hero_stats_aggregate', 'map_stats_aggregate',
  'hero_map_stats_aggregate', 'hero_talent_stats', 'hero_pairwise_stats',
  'player_match_history', 'player_hero_stats', 'player_hero_map_stats',
  'replay_draft_data', 'replay_players', 'replay_extras', 'player_refetch_state',
  'qm_games', 'player_fetch_queue', 'qm_fetch_state', 'replay_sync_state',
  'replay_fetch_queue', 'rating_items', 'draft_ratings', 'sync_log',
]

export default defineConfig({
  schema: './src/lib/db/schema.ts',
  out: './drizzle',
  dialect: 'postgresql',
  dbCredentials: {
    url: process.env.DATABASE_URL!,
  },
  tablesFilter: SCHEMA_TABLES.filter(t => t !== 'rating_items' && t !== 'draft_ratings'),
})
