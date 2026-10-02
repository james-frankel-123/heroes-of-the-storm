/**
 * Encoding sizes shared by every kernel source. Defaults are the v1 encoding
 * (90 heroes, 14 maps) that all research builds use; the production v2 build
 * (setup_h91.py, training/shared.py HOTS_HERO_SET=v2) passes
 * -DNUM_HEROES=91 -DNUM_MAPS=15. New heroes and maps are appended, so v1
 * indices keep their meaning.
 */
#pragma once

#ifndef NUM_HEROES
#define NUM_HEROES 90
#endif
#ifndef NUM_MAPS
#define NUM_MAPS 14
#endif
#ifndef NUM_TIERS
#define NUM_TIERS 3
#endif

// hero bitsets are uint32_t[3]
#if NUM_HEROES > 96
#error "hero bitsets hold at most 96 heroes"
#endif

// policy state: t0 + t1 + bans + map + tier + step + is_pick + our_team
#define STATE_DIM (NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS + 3)
#define GD_STATE_DIM (STATE_DIM - 1)
// WP input: t0 + t1 + map + tier [+ 86 enriched] [+ 8 step embedding]
#define WP_BASE_DIM (NUM_HEROES * 2 + NUM_MAPS + NUM_TIERS)
#define ENRICHED_DIM 86
#define WP_FULL_DIM (WP_BASE_DIM + ENRICHED_DIM)
#define STEP_EMBED_DIM 8
#define WP_INPUT_DIM (WP_FULL_DIM + STEP_EMBED_DIM)
// shared state buffer holds either a policy state or a WP input
#define STATE_BUF_SIZE (STATE_DIM > WP_INPUT_DIM ? STATE_DIM : WP_INPUT_DIM)
