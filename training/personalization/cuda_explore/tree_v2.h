/**
 * Plain data layout shared by the v2 search kernel (search_v2.cuh) and the
 * host bindings. Included by both nvcc and the host C++ compiler, so it holds
 * only POD structs and constants.
 *
 * v2 = the X2 fix (audits/CONSOLIDATED_AUDIT_2026-10-01.md): opponent turns
 * are explicit chance nodes in the tree, so the search reaches later own
 * picks and bans instead of stopping at the current own-pick block.
 */
#pragma once

#ifndef NUM_HEROES
#define NUM_HEROES 90
#endif
#ifndef STATE_DIM
#define STATE_DIM 290
#endif
#ifndef MAX_OUR_TURNS
#define MAX_OUR_TURNS 8
#endif

// search modes
#define SEARCH_LEGACY 0    // pre-fix kernel, bit-for-bit (separate legacy kernel)
#define SEARCH_CHANCE 1    // closed loop: opponent chance nodes (default)
#define SEARCH_ROLLFWD 2   // open loop: tree over our actions only, opponent steps
                           // re-sampled with GD on every visit (web kernel style)

// node kinds
#define KIND_DECISION 0
#define KIND_CHANCE 1
#define KIND_TERMINAL 2

// a child slot entry that is not a legal action at the node's state
#define CHILD_NONE (-2)
// a legal action whose node has not been created yet
#define CHILD_LAZY (-1)

struct TreeNodeV2 {
    int parent;
    int action;       // action that led here (-1 at the root)
    int visits;
    float value_sum;  // sum of backed-up values, P(our team wins)
    int slot;         // TreeSlotV2 index once expanded (decision) / GD cached (chance), else -1
    int n_kids;       // children created so far
    int step;         // draft step of the state at this node
    int kind;         // KIND_*
};

// Per expanded node: action distribution and child index by action.
//   decision node: p = PUCT priors (masked softmax of the policy head, root
//                  Dirichlet noise applied); child = CHILD_NONE for illegal
//   chance node:   p = GD opponent distribution at the node's state (cached
//                  once; the state is exact because the tree is closed loop)
struct TreeSlotV2 {
    float p[NUM_HEROES];
    int child[NUM_HEROES];
};

// per-episode diagnostics (int counters, accumulated over all our-turn searches)
#define ST_SEARCHES 0
#define ST_SIMS 1
#define ST_MAX_NODES 2      // max nodes used by one search
#define ST_MAX_SLOTS 3      // max slots used by one search
#define ST_CAP_HITS 4       // node/slot allocations refused by the capacity guard
#define ST_MAX_EAGER 5      // max node count the legacy (eager-children) layout would need
#define ST_SUM_LEAF_STEPS 6 // sum over sims of (leaf step - root step): in-tree depth
#define ST_SUM_OWN_AHEAD 7  // sum over sims of own decision points reached below the root
#define ST_REACHED_MASK 8   // bit s set: some sim reached our decision at draft step s in-tree
#define ST_GD_TREE 9        // GD forwards for chance-node caches / open-loop steps
#define ST_POLICY_FWD 10    // policy forwards for expansions (root included)
#define ST_MAX_DEPTH 11     // max (leaf step - root step) over sims
#define ST_HIST0 12         // ST_HIST0 + k: sims that reached k own decisions below the root (k<=7)
#define NUM_STATS_V2 20

// Tree arena + search settings passed to kernels that keep their own
// episode layout (overfit/P3 copies).
struct V2Args {
    TreeNodeV2* nodes;
    TreeSlotV2* slots;
    int node_cap;
    int slot_cap;
    int mode;        // SEARCH_*; SEARCH_LEGACY selects the kernel's legacy code
    float pw_k;
    float pw_alpha;
};

struct EpisodeOutV2 {
    float out_states[MAX_OUR_TURNS][STATE_DIM];
    float out_policies[MAX_OUR_TURNS][NUM_HEROES];
    float out_masks[MAX_OUR_TURNS][NUM_HEROES];
    int num_our_turns;
    float win_prob;
    float terminal_state[STATE_DIM];
    int our_team;
    int num_nodes;    // of the most recent search (debug_tree reads the arena)
    int num_slots;
    int root_step;
    int stats[NUM_STATS_V2];
};
