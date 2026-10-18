-- Player lookup cache (docs/personalization-site-design.md). Neon only.
-- Hand-written to match src/lib/db/schema.ts (drizzle/ has no meta snapshots,
-- so drizzle-kit generate would re-emit every table). Additive only.
CREATE TABLE IF NOT EXISTS "personal_profiles" (
	"region" smallint NOT NULL,
	"blizz_id" integer NOT NULL,
	"battletag" varchar(64) NOT NULL,
	"name_lower" varchar(32) NOT NULL,
	"summary" jsonb NOT NULL,
	"summary_at" timestamp with time zone NOT NULL,
	"profile" jsonb,
	"profile_at" timestamp with time zone,
	"history_source" varchar(16),
	"wants_metric" boolean DEFAULT false NOT NULL,
	CONSTRAINT "personal_profiles_region_blizz_id_pk" PRIMARY KEY("region","blizz_id")
);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "personal_profiles_name_idx" ON "personal_profiles" USING btree ("name_lower");
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "personal_metric_evidence" (
	"region" smallint NOT NULL,
	"blizz_id" integer NOT NULL,
	"lam" real[] NOT NULL,
	"eta" real[] NOT NULL,
	"n_rows" integer NOT NULL,
	"model_version" varchar(40) NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "personal_metric_evidence_region_blizz_id_pk" PRIMARY KEY("region","blizz_id")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "personal_api_usage" (
	"pool" varchar(16) NOT NULL,
	"hour" timestamp with time zone NOT NULL,
	"calls" integer DEFAULT 0 NOT NULL,
	"last_remaining" integer,
	CONSTRAINT "personal_api_usage_pool_hour_pk" PRIMARY KEY("pool","hour")
);
