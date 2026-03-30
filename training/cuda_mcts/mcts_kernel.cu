/**
 * Full MCTS episode kernel: one thread block = one complete draft episode.
 * 256 threads cooperate on matrix multiplications. Thread 0 drives all
 * sequential logic (tree traversal, UCB selection, action sampling).
 * Zero kernel launch overhead per forward pass.
 */
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include "device_forward.cuh"
#include "enriched_features.cuh"

#define NUM_HEROES 90
#define NUM_MAPS 14
#define NUM_TIERS 3
#define STATE_DIM 290
#define GD_STATE_DIM 289
#define MAX_NODES 4096
#define MAX_CHILD_INDICES 81920
#define MAX_OUR_TURNS 8
#define DRAFT_STEPS 16
#define MAX_PATH_DEPTH 64

__constant__ int c_draft_team[16] = {0,1,0,1, 0,1,1,0,0, 1,0, 1,1,0,0,1};
__constant__ int c_draft_is_pick[16] = {0,0,0,0, 1,1,1,1,1, 0,0, 1,1,1,1,1};

// ── GPU-side data structures ───────────────────────────────────────

struct DraftStateGPU {
    uint32_t t0_picks[3];
    uint32_t t1_picks[3];
    uint32_t bans[3];
    uint32_t taken[3];
    int step;
    int map_idx;
    int tier_idx;
    int our_team;

    __device__ void init(int m, int t, int ot) {
        memset(this, 0, sizeof(DraftStateGPU));
        map_idx = m; tier_idx = t; our_team = ot;
    }

    __device__ void copy_from(const DraftStateGPU& o) {
        memcpy(this, &o, sizeof(DraftStateGPU));
    }

    __device__ void apply_action(int hero_idx, int team, int is_pick) {
        int w = hero_idx / 32;
        uint32_t bit = 1u << (hero_idx % 32);
        taken[w] |= bit;
        if (!is_pick) bans[w] |= bit;
        else if (team == 0) t0_picks[w] |= bit;
        else t1_picks[w] |= bit;
        step++;
    }

    __device__ bool is_taken(int hero_idx) const {
        return (taken[hero_idx / 32] >> (hero_idx % 32)) & 1;
    }

    __device__ bool is_terminal() const { return step >= 16; }
    __device__ int current_team() const { return c_draft_team[step]; }
    __device__ int current_is_pick() const { return c_draft_is_pick[step]; }

    __device__ void to_float_array(float* out) const {
        int tid = threadIdx.x;
        for (int i = tid; i < STATE_DIM; i += blockDim.x) out[i] = 0.0f;
        __syncthreads();
        if (tid == 0) {
            for (int i = 0; i < NUM_HEROES; i++) {
                int w = i / 32, b = i % 32;
                if ((t0_picks[w] >> b) & 1) out[i] = 1.0f;
                if ((t1_picks[w] >> b) & 1) out[NUM_HEROES + i] = 1.0f;
                if ((bans[w] >> b) & 1) out[2 * NUM_HEROES + i] = 1.0f;
            }
            out[3 * NUM_HEROES + map_idx] = 1.0f;
            out[3 * NUM_HEROES + NUM_MAPS + tier_idx] = 1.0f;
            if (step < 16) {
                out[STATE_DIM - 3] = step / 15.0f;
                out[STATE_DIM - 2] = c_draft_is_pick[step] ? 1.0f : 0.0f;
            } else {
                out[STATE_DIM - 3] = 1.0f;
                out[STATE_DIM - 2] = 1.0f;
            }
            out[STATE_DIM - 1] = (float)our_team;
        }
        __syncthreads();
    }

    __device__ void valid_mask(float* out) const {
        int tid = threadIdx.x;
        for (int i = tid; i < NUM_HEROES; i += blockDim.x)
            out[i] = is_taken(i) ? 0.0f : 1.0f;
        __syncthreads();
    }
};

struct MCTSNodeGPU {
    int parent_idx;
    int action;
    float prior;
    int visit_count;
    float value_sum;
    int children_start;
    int num_children;
    int is_expanded;
    int has_cached_opp;
    float cached_opp_probs[NUM_HEROES];

    __device__ float q_value() const {
        return visit_count == 0 ? 0.0f : value_sum / (float)visit_count;
    }
};

struct EpisodeMemory {
    MCTSNodeGPU nodes[MAX_NODES];
    int child_indices[MAX_CHILD_INDICES];
    int num_nodes;
    int num_children_allocated;
    float out_states[MAX_OUR_TURNS][STATE_DIM];
    float out_policies[MAX_OUR_TURNS][NUM_HEROES];
    float out_masks[MAX_OUR_TURNS][NUM_HEROES];
    int num_our_turns;
    float win_prob;
    float terminal_state[STATE_DIM];  // for WP model evaluation on host
    int our_team;                      // needed for WP perspective
};


// ── Main MCTS Episode Kernel ───────────────────────────────────────

extern "C" __global__ void mcts_episodes_kernel(
    const float* __restrict__ W_policy,
    const float* __restrict__ W_gd,
    const float* __restrict__ W_wp,
    PolicyNetOffsets policy_off,
    GDNetOffsets gd_off,
    WPNetOffsets wp_off,
    const WPLookupTables* __restrict__ lut,
    const int* __restrict__ episode_configs,  // (num_episodes, 3)
    EpisodeMemory* __restrict__ episodes,
    int num_simulations,
    float c_puct,
    unsigned long long base_seed
) {
    int ep_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Shared memory layout (dynamic based on policy network size):
    // state_buf:     290
    // mask_buf:       90
    // priors_buf:     90
    // buf_e:         edim (backbone output, persists across head calls)
    // workspace:     hdim*3 + cdim (backbone scratch, reused for WP forward)
    // enriched_buf:   86
    extern __shared__ float smem[];
    int edim = policy_off.edim;
    int ws_size = policy_off.hdim * 3 + policy_off.cdim;
    float* state_buf = smem;
    float* mask_buf = smem + STATE_DIM;
    float* priors_buf = smem + STATE_DIM + NUM_HEROES;
    float* buf_e = smem + STATE_DIM + NUM_HEROES + NUM_HEROES;
    float* workspace = smem + STATE_DIM + NUM_HEROES + NUM_HEROES + edim;
    float* enriched_buf = smem + STATE_DIM + NUM_HEROES + NUM_HEROES + edim + ws_size;

    EpisodeMemory* ep = &episodes[ep_idx];

    // Thread-local RNG (only thread 0 uses it)
    curandState rng;
    if (tid == 0) curand_init(base_seed + ep_idx, 0, 0, &rng);

    // Init draft state
    __shared__ DraftStateGPU main_state;
    if (tid == 0) {
        int cfg = ep_idx * 3;
        main_state.init(episode_configs[cfg], episode_configs[cfg+1], episode_configs[cfg+2]);
        ep->num_our_turns = 0;
    }
    __syncthreads();

    // ── DRAFT LOOP ──
    for (int draft_step = 0; draft_step < DRAFT_STEPS; draft_step++) {
        __shared__ int s_team, s_is_pick, s_our_team;
        if (tid == 0) {
            if (main_state.is_terminal()) { s_team = -1; }
            else {
                s_team = main_state.current_team();
                s_is_pick = main_state.current_is_pick();
                s_our_team = main_state.our_team;
            }
        }
        __syncthreads();
        if (s_team < 0) break;

        if (s_team == s_our_team) {
            // ═══ OUR TURN: MCTS SEARCH ═══

            // Save state for training
            main_state.to_float_array(state_buf);
            main_state.valid_mask(mask_buf);
            if (tid == 0) {
                int t = ep->num_our_turns;
                memcpy(ep->out_states[t], state_buf, STATE_DIM * sizeof(float));
                memcpy(ep->out_masks[t], mask_buf, NUM_HEROES * sizeof(float));
            }
            __syncthreads();

            // Init fresh tree
            if (tid == 0) {
                ep->num_nodes = 1;
                ep->num_children_allocated = 0;
                ep->nodes[0].parent_idx = -1;
                ep->nodes[0].action = -1;
                ep->nodes[0].visit_count = 0;
                ep->nodes[0].value_sum = 0.0f;
                ep->nodes[0].is_expanded = 0;
                ep->nodes[0].num_children = 0;
                ep->nodes[0].has_cached_opp = 0;
            }
            __syncthreads();

            // Expand root: backbone + policy head
            d_policy_backbone(state_buf, W_policy, policy_off, buf_e, workspace);
            d_policy_head(buf_e, mask_buf, W_policy, policy_off, priors_buf, workspace, main_state.step);

            if (tid == 0) {
                ep->nodes[0].is_expanded = 1;
                int nv = 0;
                for (int i = 0; i < NUM_HEROES; i++) if (mask_buf[i] > 0.5f) nv++;
                ep->nodes[0].children_start = 0;
                ep->nodes[0].num_children = nv;
                ep->num_children_allocated = nv;
                int ci = 0;
                for (int i = 0; i < NUM_HEROES; i++) {
                    if (mask_buf[i] < 0.5f) continue;
                    int ch = ep->num_nodes++;
                    ep->child_indices[ci] = ch;
                    ep->nodes[ch].parent_idx = 0;
                    ep->nodes[ch].action = i;
                    ep->nodes[ch].prior = priors_buf[i];
                    ep->nodes[ch].visit_count = 0;
                    ep->nodes[ch].value_sum = 0.0f;
                    ep->nodes[ch].is_expanded = 0;
                    ep->nodes[ch].num_children = 0;
                    ep->nodes[ch].has_cached_opp = 0;
                    ci++;
                }
            }
            __syncthreads();

            // ── Run simulations ──
            for (int sim = 0; sim < num_simulations; sim++) {
                __shared__ DraftStateGPU scratch;
                __shared__ int s_path[MAX_PATH_DEPTH];
                __shared__ int s_path_len;
                __shared__ int s_leaf_idx;
                __shared__ int s_leaf_needs_expand;
                __shared__ int s_needs_gd;
                __shared__ int s_gd_node_idx;
                __shared__ int s_select_done;

                if (tid == 0) {
                    scratch.copy_from(main_state);
                    s_path[0] = 0;
                    s_path_len = 1;
                    s_select_done = 0;
                    s_needs_gd = 0;
                    s_leaf_idx = 0;
                    s_leaf_needs_expand = 0;
                }
                __syncthreads();

                // SELECT with GD cache miss handling
                while (true) {
                    if (s_needs_gd) {
                        // All threads cooperate on GD forward pass
                        scratch.to_float_array(state_buf);
                        scratch.valid_mask(mask_buf);
                        d_gd_forward(state_buf, mask_buf, W_gd, priors_buf, gd_off, workspace);

                        if (tid == 0) {
                            MCTSNodeGPU& node = ep->nodes[s_gd_node_idx];
                            float mx = -1e30f;
                            for (int i = 0; i < NUM_HEROES; i++) mx = fmaxf(mx, priors_buf[i]);
                            float sm = 0;
                            for (int i = 0; i < NUM_HEROES; i++) {
                                node.cached_opp_probs[i] = expf(priors_buf[i] - mx);
                                sm += node.cached_opp_probs[i];
                            }
                            if (sm > 0) for (int i = 0; i < NUM_HEROES; i++) node.cached_opp_probs[i] /= sm;
                            node.has_cached_opp = 1;
                            s_needs_gd = 0;
                        }
                        __syncthreads();
                    }

                    if (s_select_done) break;

                    // Thread 0 continues SELECT
                    if (tid == 0) {
                        int node_idx = s_path[s_path_len - 1];
                        // Continue traversal until leaf, terminal, or GD miss
                        while (ep->nodes[node_idx].is_expanded && !scratch.is_terminal()) {
                            if (scratch.current_team() == scratch.our_team) {
                                MCTSNodeGPU& node = ep->nodes[node_idx];
                                float best_score = -1e30f;
                                int best_slot = -1;
                                for (int c = 0; c < node.num_children; c++) {
                                    int ci = ep->child_indices[node.children_start + c];
                                    MCTSNodeGPU& child = ep->nodes[ci];
                                    float q = child.q_value();
                                    float u = c_puct * child.prior *
                                              sqrtf((float)node.visit_count) / (1.0f + child.visit_count);
                                    if (q + u > best_score) { best_score = q + u; best_slot = c; }
                                }
                                if (best_slot < 0) break;
                                int chosen = ep->child_indices[node.children_start + best_slot];
                                scratch.apply_action(ep->nodes[chosen].action,
                                    scratch.current_team(), scratch.current_is_pick());
                                node_idx = chosen;
                                if (s_path_len < MAX_PATH_DEPTH) s_path[s_path_len++] = node_idx;
                            } else {
                                MCTSNodeGPU& node = ep->nodes[node_idx];
                                if (!node.has_cached_opp) {
                                    s_needs_gd = 1;
                                    s_gd_node_idx = node_idx;
                                    break;
                                }
                                float r = curand_uniform(&rng);
                                float cum = 0;
                                int opp_action = 0;
                                for (int j = 0; j < NUM_HEROES; j++) {
                                    cum += node.cached_opp_probs[j];
                                    if (cum > r) { opp_action = j; break; }
                                }
                                scratch.apply_action(opp_action,
                                    scratch.current_team(), scratch.current_is_pick());
                            }
                        }
                        if (!s_needs_gd) {
                            s_leaf_idx = node_idx;
                            s_select_done = 1;
                            s_leaf_needs_expand = (!scratch.is_terminal() &&
                                                    !ep->nodes[node_idx].is_expanded &&
                                                    scratch.current_team() == scratch.our_team) ? 1 : 0;
                        }
                    }
                    __syncthreads();
                }

                // EVALUATE leaf: policy head for priors (if expanding) + WP model for value

                __shared__ float s_value;

                if (s_leaf_needs_expand) {
                    // Need backbone + policy head for expansion priors
                    scratch.to_float_array(state_buf);
                    scratch.valid_mask(mask_buf);
                    d_policy_backbone(state_buf, W_policy, policy_off, buf_e, workspace);
                    d_policy_head(buf_e, mask_buf, W_policy, policy_off, priors_buf, workspace, scratch.step);

                    if (tid == 0) {
                        MCTSNodeGPU& leaf = ep->nodes[s_leaf_idx];
                        int nv = 0;
                        for (int i = 0; i < NUM_HEROES; i++) if (mask_buf[i] > 0.5f) nv++;
                        leaf.is_expanded = 1;
                        leaf.children_start = ep->num_children_allocated;
                        leaf.num_children = nv;
                        ep->num_children_allocated += nv;
                        int ci = 0;
                        for (int i = 0; i < NUM_HEROES; i++) {
                            if (mask_buf[i] < 0.5f) continue;
                            int ch = ep->num_nodes++;
                            ep->child_indices[leaf.children_start + ci] = ch;
                            ep->nodes[ch].parent_idx = s_leaf_idx;
                            ep->nodes[ch].action = i;
                            ep->nodes[ch].prior = priors_buf[i];
                            ep->nodes[ch].visit_count = 0;
                            ep->nodes[ch].value_sum = 0.0f;
                            ep->nodes[ch].is_expanded = 0;
                            ep->nodes[ch].num_children = 0;
                            ep->nodes[ch].has_cached_opp = 0;
                            ci++;
                        }
                    }
                    __syncthreads();
                }

                // LEAF EVALUATION: rollout to terminal with GD, then evaluate with WP
                // This ensures the WP model always sees complete 5v5 states (in-distribution)
                // instead of partial mid-draft states (out-of-distribution)

                // Rollout remaining steps using GD for both teams
                while (!scratch.is_terminal()) {
                    scratch.to_float_array(state_buf);
                    scratch.valid_mask(mask_buf);
                    d_gd_forward(state_buf, mask_buf, W_gd, priors_buf, gd_off, workspace);
                    if (tid == 0) {
                        // Softmax + sample
                        float mx = -1e30f;
                        for (int i = 0; i < NUM_HEROES; i++) mx = fmaxf(mx, priors_buf[i]);
                        float sm = 0;
                        for (int i = 0; i < NUM_HEROES; i++) {
                            priors_buf[i] = expf(priors_buf[i] - mx);
                            sm += priors_buf[i];
                        }
                        if (sm > 0) for (int i = 0; i < NUM_HEROES; i++) priors_buf[i] /= sm;
                        float r = curand_uniform(&rng);
                        float cum = 0;
                        int action = 0;
                        for (int i = 0; i < NUM_HEROES; i++) {
                            cum += priors_buf[i];
                            if (cum > r) { action = i; break; }
                        }
                        scratch.apply_action(action, scratch.current_team(), scratch.current_is_pick());
                    }
                    __syncthreads();
                }

                // Now scratch is a complete draft — evaluate with WP (in-distribution)
                __shared__ int s_wp_t0h[5], s_wp_t1h[5], s_wp_n0, s_wp_n1;
                if (tid == 0) {
                    s_wp_n0 = 0; s_wp_n1 = 0;
                    for (int i = 0; i < NUM_HEROES; i++) {
                        int w = i / 32, b = i % 32;
                        if ((scratch.t0_picks[w] >> b) & 1 && s_wp_n0 < 5)
                            s_wp_t0h[s_wp_n0++] = i;
                        if ((scratch.t1_picks[w] >> b) & 1 && s_wp_n1 < 5)
                            s_wp_t1h[s_wp_n1++] = i;
                    }
                }
                __syncthreads();

                {
                    float val = wp_eval_symmetrized(
                        s_wp_t0h, s_wp_n0, s_wp_t1h, s_wp_n1,
                        scratch.map_idx, scratch.tier_idx, scratch.our_team,
                        lut, W_wp, wp_off, state_buf, enriched_buf, workspace);
                    if (tid == 0) s_value = val;
                }
                __syncthreads();

                // BACKPROP
                if (tid == 0) {
                    for (int p = 0; p < s_path_len; p++) {
                        ep->nodes[s_path[p]].visit_count++;
                        ep->nodes[s_path[p]].value_sum += s_value;
                    }
                }
                __syncthreads();
            }

            // Extract visit distribution and choose action
            if (tid == 0) {
                int t = ep->num_our_turns;
                float vsum = 0;
                for (int i = 0; i < NUM_HEROES; i++) ep->out_policies[t][i] = 0;
                for (int c = 0; c < ep->nodes[0].num_children; c++) {
                    int ci = ep->child_indices[ep->nodes[0].children_start + c];
                    float v = (float)ep->nodes[ci].visit_count;
                    ep->out_policies[t][ep->nodes[ci].action] = v;
                    vsum += v;
                }
                if (vsum > 0) for (int i = 0; i < NUM_HEROES; i++) ep->out_policies[t][i] /= vsum;

                float r = curand_uniform(&rng);
                float cum = 0;
                int chosen = 0;
                for (int i = 0; i < NUM_HEROES; i++) {
                    cum += ep->out_policies[t][i];
                    if (cum > r) { chosen = i; break; }
                }
                main_state.apply_action(chosen, s_team, s_is_pick);
                ep->num_our_turns++;
            }
            __syncthreads();

        } else {
            // ═══ OPPONENT TURN: GD ═══
            main_state.to_float_array(state_buf);
            main_state.valid_mask(mask_buf);
            d_gd_forward(state_buf, mask_buf, W_gd, priors_buf, gd_off, workspace);

            if (tid == 0) {
                float mx = -1e30f;
                for (int i = 0; i < NUM_HEROES; i++) mx = fmaxf(mx, priors_buf[i]);
                float sm = 0;
                for (int i = 0; i < NUM_HEROES; i++) {
                    priors_buf[i] = expf(priors_buf[i] - mx);
                    sm += priors_buf[i];
                }
                if (sm > 0) for (int i = 0; i < NUM_HEROES; i++) priors_buf[i] /= sm;

                float r = curand_uniform(&rng);
                float cum = 0;
                int action = 0;
                for (int i = 0; i < NUM_HEROES; i++) {
                    cum += priors_buf[i];
                    if (cum > r) { action = i; break; }
                }
                main_state.apply_action(action, s_team, s_is_pick);
            }
            __syncthreads();
        }
    }

    // Terminal: compute WP in-kernel using enriched WP model (symmetrized)
    __shared__ int s_term_t0h[5], s_term_t1h[5], s_term_n0, s_term_n1;
    if (tid == 0) {
        s_term_n0 = 0; s_term_n1 = 0;
        for (int i = 0; i < NUM_HEROES; i++) {
            int w = i / 32, b = i % 32;
            if ((main_state.t0_picks[w] >> b) & 1 && s_term_n0 < 5)
                s_term_t0h[s_term_n0++] = i;
            if ((main_state.t1_picks[w] >> b) & 1 && s_term_n1 < 5)
                s_term_t1h[s_term_n1++] = i;
        }
    }
    __syncthreads();

    float term_wp = wp_eval_symmetrized(
        s_term_t0h, s_term_n0, s_term_t1h, s_term_n1,
        main_state.map_idx, main_state.tier_idx, main_state.our_team,
        lut, W_wp, wp_off, state_buf, enriched_buf, workspace);

    // Write terminal state (for debugging/logging) and WP to global memory
    main_state.to_float_array(state_buf);
    for (int i = tid; i < STATE_DIM; i += blockDim.x)
        ep->terminal_state[i] = state_buf[i];
    if (tid == 0) {
        ep->our_team = main_state.our_team;
        ep->win_prob = term_wp;
    }
}
