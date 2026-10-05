-- Local primary store additions (not on Neon). Idempotent.
-- See sync/docs/data-architecture-plan.md.

-- updated_at: bumped on every insert/update; the incremental-backup watermark.
-- clock_timestamp(), not now(): a long transaction must not stamp rows earlier
-- than a watermark taken while it ran (backup_local.py also re-reads an overlap).
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at := clock_timestamp();
  RETURN NEW;
END $$;

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['replay_players', 'replay_draft_data', 'replay_extras', 'qm_games'] LOOP
    EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now()', t);
    EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I', t || '_touch', t);
    EXECUTE format('CREATE TRIGGER %I BEFORE INSERT OR UPDATE ON %I FOR EACH ROW EXECUTE FUNCTION touch_updated_at()', t || '_touch', t);
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (updated_at)', t || '_updated_at_idx', t);
  END LOOP;
END $$;

-- Read-only role for research and paper work (password set out of band).
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'hots_research') THEN
    CREATE ROLE hots_research LOGIN;
  END IF;
END $$;
GRANT CONNECT ON DATABASE hots TO hots_research;
GRANT USAGE ON SCHEMA public TO hots_research;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO hots_research;
ALTER DEFAULT PRIVILEGES FOR ROLE hots IN SCHEMA public GRANT SELECT ON TABLES TO hots_research;
ALTER ROLE hots_research SET default_transaction_read_only = on;
