import 'dotenv/config'
import { defineConfig } from 'drizzle-kit'
import { NEON_TABLES } from './sync/table-groups'

// Neon (serves the site). Only the site and study tables exist there; the
// filter stops push/generate from recreating or touching anything else.
// The full corpus lives in the local store: drizzle.local.config.ts.
export default defineConfig({
  schema: './src/lib/db/schema.ts',
  out: './drizzle',
  dialect: 'postgresql',
  dbCredentials: {
    url: process.env.NEON_DATABASE_URL!,
  },
  tablesFilter: [...NEON_TABLES],
})
