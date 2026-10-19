-- Player lookup cards and name index (docs/personalization-site-design.md).
-- Neon and the local store. Filled by training/personalization/p3_site_publish.py
-- (local) and sync/publish-site.ts (Neon). Additive only.
CREATE TABLE IF NOT EXISTS "personal_accounts" (
	"region" smallint NOT NULL,
	"blizz_id" integer NOT NULL,
	"name" varchar(24) NOT NULL,
	"discriminator" integer NOT NULL,
	"mmr" smallint,
	"last_played" date NOT NULL,
	"games" integer NOT NULL,
	"recent_games" integer NOT NULL,
	"top_heroes" bytea NOT NULL,
	"card" bytea NOT NULL,
	CONSTRAINT "personal_accounts_region_blizz_id_pk" PRIMARY KEY("region","blizz_id")
);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "personal_accounts_name_idx" ON "personal_accounts" USING btree (lower("name"));
