CREATE TYPE "public"."skill_tier" AS ENUM('low', 'mid', 'high');--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "hero_map_stats_aggregate" (
	"id" serial PRIMARY KEY NOT NULL,
	"hero" varchar(80) NOT NULL,
	"map" varchar(80) NOT NULL,
	"skill_tier" "skill_tier" NOT NULL,
	"games" integer DEFAULT 0 NOT NULL,
	"wins" integer DEFAULT 0 NOT NULL,
	"win_rate" real DEFAULT 0 NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "hero_pairwise_stats" (
	"id" serial PRIMARY KEY NOT NULL,
	"hero_a" varchar(80) NOT NULL,
	"hero_b" varchar(80) NOT NULL,
	"relationship" varchar(10) NOT NULL,
	"skill_tier" "skill_tier" NOT NULL,
	"games" integer DEFAULT 0 NOT NULL,
	"wins" integer DEFAULT 0 NOT NULL,
	"win_rate" real DEFAULT 0 NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "hero_stats_aggregate" (
	"id" serial PRIMARY KEY NOT NULL,
	"hero" varchar(80) NOT NULL,
	"skill_tier" "skill_tier" NOT NULL,
	"games" integer DEFAULT 0 NOT NULL,
	"wins" integer DEFAULT 0 NOT NULL,
	"win_rate" real DEFAULT 0 NOT NULL,
	"ban_rate" real DEFAULT 0,
	"pick_rate" real DEFAULT 0,
	"avg_kills" real DEFAULT 0,
	"avg_deaths" real DEFAULT 0,
	"avg_assists" real DEFAULT 0,
	"avg_hero_damage" real DEFAULT 0,
	"avg_siege_damage" real DEFAULT 0,
	"avg_healing" real DEFAULT 0,
	"avg_experience" real DEFAULT 0,
	"patch_tag" varchar(40),
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "hero_talent_stats" (
	"id" serial PRIMARY KEY NOT NULL,
	"hero" varchar(80) NOT NULL,
	"skill_tier" "skill_tier" NOT NULL,
	"talent_tier" integer NOT NULL,
	"talent_name" varchar(120) NOT NULL,
	"games" integer DEFAULT 0 NOT NULL,
	"wins" integer DEFAULT 0 NOT NULL,
	"win_rate" real DEFAULT 0 NOT NULL,
	"pick_rate" real DEFAULT 0,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "map_stats_aggregate" (
	"id" serial PRIMARY KEY NOT NULL,
	"map" varchar(80) NOT NULL,
	"skill_tier" "skill_tier" NOT NULL,
	"games" integer DEFAULT 0 NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "player_hero_map_stats" (
	"id" serial PRIMARY KEY NOT NULL,
	"battletag" varchar(100) NOT NULL,
	"hero" varchar(80) NOT NULL,
	"map" varchar(80) NOT NULL,
	"games" integer DEFAULT 0 NOT NULL,
	"wins" integer DEFAULT 0 NOT NULL,
	"win_rate" real DEFAULT 0 NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "player_hero_stats" (
	"id" serial PRIMARY KEY NOT NULL,
	"battletag" varchar(100) NOT NULL,
	"hero" varchar(80) NOT NULL,
	"games" integer DEFAULT 0 NOT NULL,
	"wins" integer DEFAULT 0 NOT NULL,
	"win_rate" real DEFAULT 0 NOT NULL,
	"mawp" real,
	"avg_kills" real DEFAULT 0,
	"avg_deaths" real DEFAULT 0,
	"avg_assists" real DEFAULT 0,
	"recent_win_rate" real,
	"trend" real,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "player_match_history" (
	"id" serial PRIMARY KEY NOT NULL,
	"battletag" varchar(100) NOT NULL,
	"replay_id" varchar(100) NOT NULL,
	"hero" varchar(80) NOT NULL,
	"map" varchar(80) NOT NULL,
	"win" boolean NOT NULL,
	"game_date" timestamp NOT NULL,
	"game_length" integer,
	"kills" integer DEFAULT 0,
	"deaths" integer DEFAULT 0,
	"assists" integer DEFAULT 0,
	"hero_damage" integer DEFAULT 0,
	"siege_damage" integer DEFAULT 0,
	"healing" integer DEFAULT 0,
	"experience" integer DEFAULT 0,
	"talents" jsonb,
	"game_mode" varchar(40),
	"rank" varchar(40),
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "sync_log" (
	"id" serial PRIMARY KEY NOT NULL,
	"sync_type" varchar(40) NOT NULL,
	"battletag" varchar(100),
	"status" varchar(20) NOT NULL,
	"matches_processed" integer DEFAULT 0,
	"error_message" text,
	"started_at" timestamp DEFAULT now() NOT NULL,
	"completed_at" timestamp
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "tracked_battletags" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"battletag" varchar(100) NOT NULL,
	"region" integer DEFAULT 1,
	"added_at" timestamp DEFAULT now() NOT NULL,
	"last_synced" timestamp
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "users" (
	"id" serial PRIMARY KEY NOT NULL,
	"email" varchar(255) NOT NULL,
	"display_name" varchar(100),
	"created_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "users_email_unique" UNIQUE("email")
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "tracked_battletags" ADD CONSTRAINT "tracked_battletags_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "hero_map_tier_idx" ON "hero_map_stats_aggregate" USING btree ("hero","map","skill_tier");--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "pairwise_idx" ON "hero_pairwise_stats" USING btree ("hero_a","hero_b","relationship","skill_tier");--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "reverse_pair_idx" ON "hero_pairwise_stats" USING btree ("hero_b","relationship","skill_tier");--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "hero_tier_idx" ON "hero_stats_aggregate" USING btree ("hero","skill_tier");--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "hero_talent_idx" ON "hero_talent_stats" USING btree ("hero","skill_tier","talent_tier","talent_name");--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "map_tier_idx" ON "map_stats_aggregate" USING btree ("map","skill_tier");--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "player_hero_map_idx" ON "player_hero_map_stats" USING btree ("battletag","hero","map");--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "player_hero_stats_idx" ON "player_hero_stats" USING btree ("battletag","hero");--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "player_replay_idx" ON "player_match_history" USING btree ("battletag","replay_id");--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "player_hero_idx" ON "player_match_history" USING btree ("battletag","hero");--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "player_date_idx" ON "player_match_history" USING btree ("battletag","game_date");--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "user_battletag_idx" ON "tracked_battletags" USING btree ("user_id","battletag");