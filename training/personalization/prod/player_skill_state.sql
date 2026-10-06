-- player_skill_state: nightly per-player x hero skill state for the P3 personal layer.
-- Written by training/personalization/prod/nightly_state.py (table__* arrays of its npz).
-- NOT applied anywhere yet. Intended home: the local store, published to the site
-- through sync/publish-site.ts. No battletags: the site resolves a battletag to
-- candidate (region, blizz_id) accounts at serve time (nightly_state.lookup).
--
-- Invariants:
--   * every counted game had fetched_at < as_of and ended before as_of;
--   * a snapshot may serve a game only if as_of <= the game's start time;
--   * s and p are sums of r/v and 1/v with r = y - wp from the causal WP chain
--     (training/personalization/prod/wp_chain.py), v = wp (1 - wp);
--   * region 0 is the explicit unknown region, never a negative code.

CREATE TABLE IF NOT EXISTS player_skill_state (
  region           smallint         NOT NULL,           -- 1 US, 2 EU, 3 KR, 5 CN, 0 unknown
  blizz_id         bigint           NOT NULL,
  hero             smallint         NOT NULL,           -- index into the 90-hero v1 list (training/shared.py)
  s                double precision NOT NULL,           -- sum r / v
  p                double precision NOT NULL,           -- sum 1 / v
  n_games          integer          NOT NULL,
  last_played_day  integer          NOT NULL,           -- UTC days since 1970-01-01 of game end
  first_seen_day   integer          NOT NULL,           -- player's earliest counted game, any hero
  account_status   smallint         NOT NULL CHECK (account_status BETWEEN 0 AND 3),
  wp_vintage_id    varchar(64)      NOT NULL,           -- newest WP vintage among the row's games
  max_fetched_at   timestamptz      NOT NULL,           -- latest source fetched_at in the row
  as_of            timestamptz      NOT NULL,           -- snapshot time
  PRIMARY KEY (region, blizz_id, hero),
  CHECK (max_fetched_at < as_of)
);

CREATE INDEX IF NOT EXISTS player_skill_state_player_idx ON player_skill_state (region, blizz_id);
