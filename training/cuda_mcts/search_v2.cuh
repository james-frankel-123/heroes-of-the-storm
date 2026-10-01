/**
 * v2 MCTS episode kernel: the X2 fix.
 *
 * The legacy kernel (mcts_episodes_kernel, kept verbatim above this include)
 * expands a node only when our team is to move. A child whose next step
 * belongs to the opponent therefore stays a leaf for ever, every visit ends in
 * a GD rollout, and the tree covers only the current own-pick block.
 *
 * SEARCH_CHANCE (default). Closed-loop expectimax-style MCTS:
 *   - decision nodes (our team to move): PUCT over children keyed by our
 *     action, priors from the policy head, children created lazily (an
 *     untried action has Q = 0 and N = 0, exactly as the legacy eager
 *     children did);
 *   - chance nodes (opponent to move): the GD distribution at the node's
 *     exact state is computed once and cached in the node's slot (the same
 *     cache the legacy kernel kept per node). Selection samples the
 *     opponent's action from it and descends into the child keyed by that
 *     action, creating it on demand.
 *   - progressive widening at chance nodes (pw_k > 0): a chance node with N
 *     visits may hold at most max(1, ceil(pw_k * N^pw_alpha - 1e-3)) children. Below
 *     the limit the outcome is drawn from the full GD distribution; at the
 *     limit it is drawn from GD renormalized over the existing children
 *     (drawing from the full distribution and resampling on a miss gives
 *     the same law). The limit grows without bound, so every outcome with
 *     positive probability is eventually admitted and the chance node's
 *     value converges to the GD expectation; at a finite budget the search
 *     revisits the likely replies often enough to plan our next decision
 *     under them instead of spending single visits on the long tail.
 *     pw_k <= 0 disables widening (plain sampled chance nodes).
 *   - backup: every node on the path, chance nodes included, gets
 *     visits += 1 and value_sum += v, with v = P(our team wins) from the
 *     leaf rollout + symmetrized WP. All values are in our team's
 *     perspective (decision nodes maximize, chance nodes average), so no
 *     sign flips. A chance node's Q is the sampled-outcome average.
 *   - selection stops at an unexpanded decision node (expanded with the
 *     policy head, then evaluated as before: GD rollout to the terminal
 *     draft, symmetrized WP) or at a terminal state (WP directly, which is
 *     what the legacy rollout loop does when there is nothing to roll).
 *   So the tree reaches later own picks and bans through the opponent's
 *   replies.
 *
 * SEARCH_ROLLFWD. Open loop, like the web search (src/lib/draft/mcts-search.ts):
 *   the tree is keyed by our actions only; after each of our actions the
 *   opponent steps are re-sampled from GD on every visit, so a node
 *   aggregates over opponent replies. Children already taken in the current
 *   sample are skipped at selection. Nodes do not correspond to one state.
 *
 * Capacity: nodes and slots live in per-episode global arenas sized by the
 * host (node_cap, slot_cap). One simulation creates at most 3 nodes and 3
 * slots (our action, up to two opponent replies, the next own decision), so
 * 3 * sims + 8 never overflows. Every allocation is still guarded: when an
 * arena is full the simulation stops at the current node, evaluates it by
 * rollout, and ST_CAP_HITS counts the event.
 *
 * Randomness: one curand stream per episode, used by thread 0 only, so a
 * fixed seed gives identical results run to run.
 */
#pragma once
#include "tree_v2.h"

#include "search_v2_util.cuh"

extern "C" __global__ void mcts_episodes_kernel_v2(
    const float* __restrict__ W_policy,
    const float* __restrict__ W_gd,
    const float* __restrict__ W_wp,
    PolicyNetOffsets policy_off,
    GDNetOffsets gd_off,
    WPNetOffsets wp_off,
    const WPLookupTables* __restrict__ lut,
    const int* __restrict__ episode_configs,  // (num_episodes, 3)
    EpisodeOutV2* __restrict__ outs,
    TreeNodeV2* __restrict__ nodes_all,
    TreeSlotV2* __restrict__ slots_all,
    int node_cap,
    int slot_cap,
    int num_simulations,
    float c_puct,
    unsigned long long base_seed,
    float root_temp,
    float dir_alpha,
    float dir_eps,
    int mode,
    float pw_k,
    float pw_alpha,
    int stop_after_turn       // >= 0: stop after that our-turn search (debug_tree)
) {
    int ep_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Shared memory layout identical to the legacy kernel.
    extern __shared__ float smem[];
    int edim = policy_off.edim;
    int ws_size = policy_off.hdim * 3 + policy_off.cdim;
    const int state_buf_size = 291;
    float* state_buf = smem;
    float* mask_buf = smem + state_buf_size;
    float* priors_buf = smem + state_buf_size + NUM_HEROES;
    float* buf_e = smem + state_buf_size + NUM_HEROES + NUM_HEROES;
    float* workspace = smem + state_buf_size + NUM_HEROES + NUM_HEROES + edim;
    float* enriched_buf = smem + state_buf_size + NUM_HEROES + NUM_HEROES + edim + ws_size;

    EpisodeOutV2* ep = &outs[ep_idx];
    TreeNodeV2* nodes = nodes_all + (size_t)ep_idx * node_cap;
    TreeSlotV2* slots = slots_all + (size_t)ep_idx * slot_cap;

    curandState rng;
    if (tid == 0) curand_init(base_seed + ep_idx, 0, 0, &rng);

    __shared__ DraftStateGPU main_state;
    __shared__ DraftStateGPU scratch;
    __shared__ int s_team, s_is_pick, s_our_team, s_stop;
    __shared__ int s_path[MAX_PATH_DEPTH];
    __shared__ int s_path_len, s_req, s_req_node, s_select_done, s_leaf_expand;
    __shared__ int s_puct_node;
    __shared__ int s_n_nodes, s_n_slots, s_root_step;
    __shared__ float s_value;
    __shared__ int s_wp_t0h[5], s_wp_t1h[5], s_wp_n0, s_wp_n1;

    if (tid == 0) {
        int cfg = ep_idx * 3;
        main_state.init(episode_configs[cfg], episode_configs[cfg+1], episode_configs[cfg+2]);
        ep->num_our_turns = 0;
        ep->num_nodes = 0;
        ep->num_slots = 0;
        ep->root_step = -1;
        for (int i = 0; i < NUM_STATS_V2; i++) ep->stats[i] = 0;
        s_stop = 0;
    }
    __syncthreads();

    for (int draft_step = 0; draft_step < DRAFT_STEPS; draft_step++) {
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
            // ═══ OUR TURN: search ═══
            main_state.to_float_array(state_buf);
            main_state.valid_mask(mask_buf);
            if (tid == 0) {
                int t = ep->num_our_turns;
                memcpy(ep->out_states[t], state_buf, STATE_DIM * sizeof(float));
                memcpy(ep->out_masks[t], mask_buf, NUM_HEROES * sizeof(float));
            }
            __syncthreads();

            d_policy_backbone(state_buf, W_policy, policy_off, buf_e, workspace);
            d_policy_head(buf_e, mask_buf, W_policy, policy_off, priors_buf, workspace, main_state.step);

            for (int i = tid; i < NUM_HEROES; i += blockDim.x) {
                bool ok = mask_buf[i] > 0.5f;
                slots[0].p[i] = ok ? priors_buf[i] : 0.0f;
                slots[0].child[i] = ok ? CHILD_LAZY : CHILD_NONE;
            }
            __syncthreads();
            if (tid == 0) {
                s_root_step = main_state.step;
                s_n_nodes = 1;
                s_n_slots = 1;
                v2_init_node(nodes[0], -1, -1, main_state.step, KIND_DECISION);
                nodes[0].slot = 0;
                TreeSlotV2& rs = slots[0];
                int nv = 0;
                for (int i = 0; i < NUM_HEROES; i++) nv += mask_buf[i] > 0.5f;
                ep->stats[ST_POLICY_FWD]++;
                ep->stats[ST_SEARCHES]++;
                // Dirichlet root noise, same draw order as the legacy kernel
                if (dir_eps > 0.0f) {
                    float g[NUM_HEROES];
                    float gsum = 0.0f;
                    for (int c = 0; c < nv; c++) {
                        g[c] = d_sample_gamma(&rng, dir_alpha);
                        gsum += g[c];
                    }
                    if (gsum > 1e-12f) {
                        int c = 0;
                        for (int i = 0; i < NUM_HEROES; i++) {
                            if (rs.child[i] == CHILD_NONE) continue;
                            rs.p[i] = (1.0f - dir_eps) * rs.p[i] + dir_eps * (g[c] / gsum);
                            c++;
                        }
                    }
                }
            }
            __syncthreads();

            // node count the legacy layout (all children of an expanded
            // decision node allocated at once) would need; thread 0 only
            int eager_nodes = 0;
            if (tid == 0) {
                int nv = 0;
                for (int i = 0; i < NUM_HEROES; i++) nv += (slots[0].child[i] != CHILD_NONE);
                eager_nodes = 1 + nv;
            }

            for (int sim = 0; sim < num_simulations; sim++) {
                if (tid == 0) {
                    scratch.copy_from(main_state);
                    s_path[0] = 0;
                    s_path_len = 1;
                    s_select_done = 0;
                    s_req = REQ_NONE;
                    s_leaf_expand = 0;
                    s_puct_node = -1;
                }
                __syncthreads();

                // ── SELECT (thread 0), with cooperative GD forwards on request ──
                int own_ahead = 0;   // thread 0 only
                while (true) {
                    // Snapshot the request flags, then a barrier: thread 0 only
                    // rewrites them after it, so no warp can read a flag from
                    // the next round (which would mismatch the barriers below).
                    int req = s_req;
                    int done = s_select_done;
                    __syncthreads();
                    if (req == REQ_PUCT) {
                        // PUCT argmax by warp 0: per-lane first max over
                        // ascending actions, then lowest action among equal
                        // maxima = the sequential first-max rule
                        if (tid < 32) {
                            const TreeNodeV2& nd = nodes[s_req_node];
                            const TreeSlotV2& sl = slots[nd.slot];
                            float sqrt_n = sqrtf((float)nd.visits);
                            float best = -1e30f;
                            int ba = -1;
                            for (int a = tid; a < NUM_HEROES; a += 32) {
                                int ci = sl.child[a];
                                if (ci == CHILD_NONE) continue;
                                if (mode == SEARCH_ROLLFWD && scratch.is_taken(a)) continue;
                                int n = 0;
                                float q = 0.0f;
                                if (ci >= 0) {
                                    n = nodes[ci].visits;
                                    q = n == 0 ? 0.0f : nodes[ci].value_sum / (float)n;
                                }
                                float u = c_puct * sl.p[a] * sqrt_n / (1.0f + n);
                                float sc = q + u;
                                if (sc > best) { best = sc; ba = a; }
                            }
                            for (int off = 16; off > 0; off >>= 1) {
                                float ob = __shfl_down_sync(0xffffffffu, best, off);
                                int oa = __shfl_down_sync(0xffffffffu, ba, off);
                                if (oa >= 0 && (ba < 0 || ob > best || (ob == best && oa < ba))) {
                                    best = ob;
                                    ba = oa;
                                }
                            }
                            if (tid == 0) { s_puct_node = s_req_node; s_req_node = ba; }
                        }
                    } else if (req != REQ_NONE) {
                        scratch.to_float_array(state_buf);
                        scratch.valid_mask(mask_buf);
                        d_gd_forward(state_buf, mask_buf, W_gd, priors_buf, gd_off, workspace);
                        if (tid == 0) {
                            v2_softmax90(priors_buf);
                            ep->stats[ST_GD_TREE]++;
                            if (req == REQ_GD_CACHE) {
                                s_n_slots++;   // reserve; filled cooperatively below
                            } else {
                                float r = curand_uniform(&rng);
                                int a = v2_sample(priors_buf, r);
                                scratch.apply_action(a, scratch.current_team(), scratch.current_is_pick());
                            }
                        }
                        __syncthreads();
                        if (req == REQ_GD_CACHE) {
                            int sl = s_n_slots - 1;
                            for (int i = tid; i < NUM_HEROES; i += blockDim.x) {
                                slots[sl].p[i] = priors_buf[i];
                                slots[sl].child[i] = priors_buf[i] > 0.0f ? CHILD_LAZY : CHILD_NONE;
                            }
                            if (tid == 0) nodes[s_req_node].slot = sl;
                            __syncthreads();
                        }
                    }
                    if (done) break;

                    if (tid == 0) {
                        s_req = REQ_NONE;
                        int node_idx = s_path[s_path_len - 1];
                        while (true) {
                            if (scratch.is_terminal()) { s_select_done = 1; break; }
                            int team = scratch.current_team();
                            TreeNodeV2& nd = nodes[node_idx];
                            if (team == scratch.our_team) {
                                if (s_puct_node != node_idx) {
                                    // first arrival at this own decision in this sim
                                    if (node_idx != 0) {
                                        own_ahead++;
                                        ep->stats[ST_REACHED_MASK] |= (1 << scratch.step);
                                    }
                                    if (nd.slot < 0) {
                                        // unexpanded own decision: the leaf
                                        if (s_n_slots >= slot_cap) ep->stats[ST_CAP_HITS]++;
                                        else s_leaf_expand = 1;
                                        s_select_done = 1;
                                        break;
                                    }
                                    s_req = REQ_PUCT;
                                    s_req_node = node_idx;
                                    break;
                                }
                                int best_a = s_req_node;   // PUCT result (warp 0)
                                s_puct_node = -1;
                                TreeSlotV2& sl = slots[nd.slot];
                                if (best_a < 0) { s_select_done = 1; break; }
                                int ci = sl.child[best_a];
                                if (ci < 0) {
                                    if (s_n_nodes >= node_cap) {
                                        ep->stats[ST_CAP_HITS]++;
                                        s_select_done = 1;
                                        break;
                                    }
                                    ci = s_n_nodes++;
                                    sl.child[best_a] = ci;
                                    nd.n_kids++;
                                    v2_init_node(nodes[ci], node_idx, best_a, scratch.step + 1, KIND_DECISION);
                                }
                                scratch.apply_action(best_a, team, scratch.current_is_pick());
                                nodes[ci].kind = v2_kind_of(scratch, mode);
                                node_idx = ci;
                                if (s_path_len < MAX_PATH_DEPTH) s_path[s_path_len++] = node_idx;
                            } else if (mode == SEARCH_ROLLFWD) {
                                s_req = REQ_GD_STEP;
                                break;
                            } else {
                                // chance node: opponent to move
                                if (nd.slot < 0) {
                                    if (s_n_slots >= slot_cap) {
                                        ep->stats[ST_CAP_HITS]++;
                                        s_select_done = 1;
                                        break;
                                    }
                                    s_req = REQ_GD_CACHE;
                                    s_req_node = node_idx;
                                    break;
                                }
                                TreeSlotV2& sl = slots[nd.slot];
                                int limit = 1 << 30;
                                if (pw_k > 0.0f)   // epsilon: fast-math powf(196, .5) > 14
                                    limit = max(1, (int)ceilf(pw_k * powf((float)nd.visits, pw_alpha) - 1e-3f));
                                float r = curand_uniform(&rng);
                                int a = nd.n_kids < limit ? v2_sample(sl.p, r) : v2_sample_existing(sl, r);
                                if (a < 0) { s_select_done = 1; break; }
                                int ci = sl.child[a];
                                if (ci < 0) {
                                    if (s_n_nodes >= node_cap) {
                                        ep->stats[ST_CAP_HITS]++;
                                        s_select_done = 1;
                                        break;
                                    }
                                    ci = s_n_nodes++;
                                    sl.child[a] = ci;
                                    nd.n_kids++;
                                    v2_init_node(nodes[ci], node_idx, a, scratch.step + 1, KIND_DECISION);
                                    eager_nodes++;
                                }
                                scratch.apply_action(a, team, scratch.current_is_pick());
                                nodes[ci].kind = v2_kind_of(scratch, mode);
                                node_idx = ci;
                                if (s_path_len < MAX_PATH_DEPTH) s_path[s_path_len++] = node_idx;
                            }
                        }
                    }
                    __syncthreads();
                }

                // ── EXPAND the leaf decision node: policy priors ──
                if (s_leaf_expand) {
                    scratch.to_float_array(state_buf);
                    scratch.valid_mask(mask_buf);
                    d_policy_backbone(state_buf, W_policy, policy_off, buf_e, workspace);
                    d_policy_head(buf_e, mask_buf, W_policy, policy_off, priors_buf, workspace, scratch.step);
                    {
                        int sl = s_n_slots;
                        for (int i = tid; i < NUM_HEROES; i += blockDim.x) {
                            bool ok = mask_buf[i] > 0.5f;
                            slots[sl].p[i] = ok ? priors_buf[i] : 0.0f;
                            slots[sl].child[i] = ok ? CHILD_LAZY : CHILD_NONE;
                        }
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int leaf = s_path[s_path_len - 1];
                        nodes[leaf].slot = s_n_slots++;
                        int nv = 0;
                        for (int i = 0; i < NUM_HEROES; i++) nv += mask_buf[i] > 0.5f;
                        eager_nodes += nv;
                        ep->stats[ST_POLICY_FWD]++;
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    int depth = scratch.step - s_root_step;
                    ep->stats[ST_SIMS]++;
                    ep->stats[ST_SUM_LEAF_STEPS] += depth;
                    ep->stats[ST_MAX_DEPTH] = max(ep->stats[ST_MAX_DEPTH], depth);
                    ep->stats[ST_SUM_OWN_AHEAD] += own_ahead;
                    ep->stats[ST_HIST0 + min(own_ahead, 7)]++;
                }

                // ── ROLLOUT to the terminal draft with GD for both teams ──
                while (!scratch.is_terminal()) {
                    scratch.to_float_array(state_buf);
                    scratch.valid_mask(mask_buf);
                    d_gd_forward(state_buf, mask_buf, W_gd, priors_buf, gd_off, workspace);
                    if (tid == 0) {
                        v2_softmax90(priors_buf);
                        float r = curand_uniform(&rng);
                        int a = v2_sample(priors_buf, r);
                        scratch.apply_action(a, scratch.current_team(), scratch.current_is_pick());
                    }
                    __syncthreads();
                }

                // ── EVALUATE the complete draft with the symmetrized WP ──
                if (tid == 0) {
                    s_wp_n0 = 0; s_wp_n1 = 0;
                    for (int i = 0; i < NUM_HEROES; i++) {
                        int w = i / 32, b = i % 32;
                        if ((scratch.t0_picks[w] >> b) & 1 && s_wp_n0 < 5) s_wp_t0h[s_wp_n0++] = i;
                        if ((scratch.t1_picks[w] >> b) & 1 && s_wp_n1 < 5) s_wp_t1h[s_wp_n1++] = i;
                    }
                }
                __syncthreads();
                {
                    float val = wp_eval_symmetrized(
                        s_wp_t0h, s_wp_n0, s_wp_t1h, s_wp_n1,
                        scratch.map_idx, scratch.tier_idx, scratch.our_team,
                        scratch.step,
                        lut, W_wp, wp_off, state_buf, enriched_buf, workspace);
                    if (tid == 0) s_value = val;
                }
                __syncthreads();

                // ── BACKUP: every node on the path, our perspective ──
                if (tid == 0) {
                    for (int p = 0; p < s_path_len; p++) {
                        nodes[s_path[p]].visits++;
                        nodes[s_path[p]].value_sum += s_value;
                    }
                }
                __syncthreads();
            }

            // root visit distribution and executed action (legacy rules)
            if (tid == 0) {
                ep->stats[ST_MAX_NODES] = max(ep->stats[ST_MAX_NODES], s_n_nodes);
                ep->stats[ST_MAX_SLOTS] = max(ep->stats[ST_MAX_SLOTS], s_n_slots);
                ep->stats[ST_MAX_EAGER] = max(ep->stats[ST_MAX_EAGER], eager_nodes);
                ep->num_nodes = s_n_nodes;
                ep->num_slots = s_n_slots;
                ep->root_step = s_root_step;

                int t = ep->num_our_turns;
                float vsum = 0;
                for (int i = 0; i < NUM_HEROES; i++) ep->out_policies[t][i] = 0;
                for (int i = 0; i < NUM_HEROES; i++) {
                    int ci = slots[0].child[i];
                    if (ci < 0) continue;
                    float v = (float)nodes[ci].visits;
                    ep->out_policies[t][i] = v;
                    vsum += v;
                }
                if (vsum > 0) for (int i = 0; i < NUM_HEROES; i++) ep->out_policies[t][i] /= vsum;

                float r = curand_uniform(&rng);
                int chosen = -1;
                if (root_temp < 0.05f) {
                    float best = -1.0f;
                    for (int i = 0; i < NUM_HEROES; i++) {
                        if (ep->out_policies[t][i] > best) { best = ep->out_policies[t][i]; chosen = i; }
                    }
                } else if (fabsf(root_temp - 1.0f) < 1e-6f) {
                    chosen = v2_sample(ep->out_policies[t], r);
                } else {
                    float w[NUM_HEROES];
                    float wsum = 0.0f;
                    float inv_t = 1.0f / root_temp;
                    for (int i = 0; i < NUM_HEROES; i++) {
                        w[i] = ep->out_policies[t][i] > 0.0f ? powf(ep->out_policies[t][i], inv_t) : 0.0f;
                        wsum += w[i];
                    }
                    if (wsum > 0.0f) for (int i = 0; i < NUM_HEROES; i++) w[i] /= wsum;
                    chosen = v2_sample(w, r);
                }
                if (chosen < 0 || main_state.is_taken(chosen)) {
                    // no visits recorded (num_simulations == 0): follow the prior
                    chosen = v2_sample(slots[0].p, r);
                }
                main_state.apply_action(chosen, s_team, s_is_pick);
                ep->num_our_turns++;
                if (stop_after_turn >= 0 && t == stop_after_turn) s_stop = 1;
            }
            __syncthreads();
            if (s_stop) return;

        } else {
            // ═══ OPPONENT TURN: GD ═══
            main_state.to_float_array(state_buf);
            main_state.valid_mask(mask_buf);
            d_gd_forward(state_buf, mask_buf, W_gd, priors_buf, gd_off, workspace);
            if (tid == 0) {
                v2_softmax90(priors_buf);
                float r = curand_uniform(&rng);
                int a = v2_sample(priors_buf, r);
                main_state.apply_action(a, s_team, s_is_pick);
            }
            __syncthreads();
        }
    }

    // Terminal: symmetrized WP of the finished draft
    __shared__ int s_term_t0h[5], s_term_t1h[5], s_term_n0, s_term_n1;
    if (tid == 0) {
        s_term_n0 = 0; s_term_n1 = 0;
        for (int i = 0; i < NUM_HEROES; i++) {
            int w = i / 32, b = i % 32;
            if ((main_state.t0_picks[w] >> b) & 1 && s_term_n0 < 5) s_term_t0h[s_term_n0++] = i;
            if ((main_state.t1_picks[w] >> b) & 1 && s_term_n1 < 5) s_term_t1h[s_term_n1++] = i;
        }
    }
    __syncthreads();
    float term_wp = wp_eval_symmetrized(
        s_term_t0h, s_term_n0, s_term_t1h, s_term_n1,
        main_state.map_idx, main_state.tier_idx, main_state.our_team,
        main_state.step,
        lut, W_wp, wp_off, state_buf, enriched_buf, workspace);
    main_state.to_float_array(state_buf);
    for (int i = tid; i < STATE_DIM; i += blockDim.x)
        ep->terminal_state[i] = state_buf[i];
    if (tid == 0) {
        ep->our_team = main_state.our_team;
        ep->win_prob = term_wp;
    }
}
