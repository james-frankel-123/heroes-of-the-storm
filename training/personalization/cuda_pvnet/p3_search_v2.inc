// ── P3 v2 search fragment (X2 fix; P3-08) ─────────────────────────────
// Included INSIDE the P3 kernels' our-turn branch when v2.mode != 0, in
// place of the legacy tree block. Same algorithm as
// training/cuda_mcts/search_v2.cuh (opponent chance nodes with progressive
// widening, closed-loop tree, backup of P(our side wins) on every path
// node), plus the P3 rules the legacy tree applied:
//   - pool masks (P3_MASK) at every expansion, GD cache and rollout step;
//   - forced bans (P3_FORCED) are deterministic transitions: applied right
//     after every tree action, in rollouts, never a node;
//   - leaf value = P3_VALUE(population WP) (personal valuation);
//   - priors / GD from P3_PRIORS / P3_GD (prior bias, personalized GD).
// The including kernel defines those macros and has in scope: state_buf,
// mask_buf, priors_buf, buf_e, workspace, enriched_buf, ep, main_state, rng,
// tid, num_simulations, c_puct, dir_alpha, dir_eps, v2, lut, W_*, *_off.
// Output: ep->out_policies[t], ep->out_q[t], ep->max_nodes_used,
// ep->n_capacity_hits, ep->v2_stats. state_buf / mask_buf hold the root on entry.
{
    TreeNodeV2* nodes = v2.nodes + (size_t)blockIdx.x * v2.node_cap;
    TreeSlotV2* slots = v2.slots + (size_t)blockIdx.x * v2.slot_cap;
    const int mode = v2.mode;
    __shared__ DraftStateGPU v2_scratch;
    __shared__ int v2_path[MAX_PATH_DEPTH];
    __shared__ int v2_path_len, v2_req, v2_req_node, v2_done, v2_leaf_expand, v2_puct_node;
    __shared__ int v2_n_nodes, v2_n_slots, v2_root_step, v2_rfa;
    __shared__ float v2_value;
    __shared__ int v2_t0h[5], v2_t1h[5], v2_n0, v2_n1;
    DraftStateGPU& scratch = v2_scratch;

    P3_PRIORS(main_state);
    for (int i = tid; i < NUM_HEROES; i += blockDim.x) {
        bool ok = mask_buf[i] > 0.5f;
        slots[0].p[i] = ok ? priors_buf[i] : 0.0f;
        slots[0].child[i] = ok ? CHILD_LAZY : CHILD_NONE;
    }
    __syncthreads();
    P3_ROOT_HOOK();
    int eager_nodes = 0;   // thread 0
    if (tid == 0) {
        v2_root_step = main_state.step;
        v2_n_nodes = 1;
        v2_n_slots = 1;
        v2_init_node(nodes[0], -1, -1, main_state.step, KIND_DECISION);
        nodes[0].slot = 0;
        int nv = 0;
        for (int i = 0; i < NUM_HEROES; i++) nv += mask_buf[i] > 0.5f;
        eager_nodes = 1 + nv;
        ep->v2_stats[ST_POLICY_FWD]++;
        ep->v2_stats[ST_SEARCHES]++;
        if (dir_eps > 0.0f) {
            float g[NUM_HEROES];
            float gsum = 0.0f;
            for (int c = 0; c < nv; c++) { g[c] = d_sample_gamma(&rng, dir_alpha); gsum += g[c]; }
            if (gsum > 1e-12f) {
                int c = 0;
                for (int i = 0; i < NUM_HEROES; i++) {
                    if (slots[0].child[i] == CHILD_NONE) continue;
                    slots[0].p[i] = (1.0f - dir_eps) * slots[0].p[i] + dir_eps * (g[c] / gsum);
                    c++;
                }
            }
        }
    }
    __syncthreads();

    for (int sim = 0; sim < num_simulations; sim++) {
        if (tid == 0) {
            scratch.copy_from(main_state);
            v2_path[0] = 0;
            v2_path_len = 1;
            v2_done = 0;
            v2_req = REQ_NONE;
            v2_leaf_expand = 0;
            v2_puct_node = -1;
        }
        __syncthreads();
        int own_ahead = 0;   // thread 0
        while (true) {
            int req = v2_req;
            int done = v2_done;
            __syncthreads();
            if (req == REQ_PUCT) {
                if (tid < 32) {
                    const TreeNodeV2& nd = nodes[v2_req_node];
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
                        float sc = q + c_puct * sl.p[a] * sqrt_n / (1.0f + n);
                        if (sc > best) { best = sc; ba = a; }
                    }
                    for (int off = 16; off > 0; off >>= 1) {
                        float ob = __shfl_down_sync(0xffffffffu, best, off);
                        int oa = __shfl_down_sync(0xffffffffu, ba, off);
                        if (oa >= 0 && (ba < 0 || ob > best || (ob == best && oa < ba))) { best = ob; ba = oa; }
                    }
                    if (tid == 0) { v2_puct_node = v2_req_node; v2_req_node = ba; }
                }
            } else if (req != REQ_NONE) {
                scratch.to_float_array(state_buf);
                P3_MASK(scratch);
                P3_GD(scratch);
                if (tid == 0) {
                    v2_softmax90(priors_buf);
                    ep->v2_stats[ST_GD_TREE]++;
                    if (req == REQ_GD_CACHE) {
                        v2_n_slots++;
                    } else {
                        int a = v2_sample(priors_buf, curand_uniform(&rng));
                        scratch.apply_action(a, scratch.current_team(), scratch.current_is_pick());
                        int fa;
                        while (!scratch.is_terminal() && (fa = P3_FORCED(scratch)) >= 0)
                            scratch.apply_action(fa, scratch.current_team(), 0);
                    }
                }
                __syncthreads();
                if (req == REQ_GD_CACHE) {
                    int sl = v2_n_slots - 1;
                    for (int i = tid; i < NUM_HEROES; i += blockDim.x) {
                        slots[sl].p[i] = priors_buf[i];
                        slots[sl].child[i] = priors_buf[i] > 0.0f ? CHILD_LAZY : CHILD_NONE;
                    }
                    if (tid == 0) nodes[v2_req_node].slot = sl;
                    __syncthreads();
                }
            }
            if (done) break;

            if (tid == 0) {
                v2_req = REQ_NONE;
                int node_idx = v2_path[v2_path_len - 1];
                while (true) {
                    if (scratch.is_terminal()) { v2_done = 1; break; }
                    int team = scratch.current_team();
                    TreeNodeV2& nd = nodes[node_idx];
                    if (team == scratch.our_team) {
                        if (v2_puct_node != node_idx) {
                            if (node_idx != 0) {
                                own_ahead++;
                                ep->v2_stats[ST_REACHED_MASK] |= (1 << scratch.step);
                            }
                            if (nd.slot < 0) {
                                if (v2_n_slots >= v2.slot_cap) ep->v2_stats[ST_CAP_HITS]++;
                                else v2_leaf_expand = 1;
                                v2_done = 1;
                                break;
                            }
                            v2_req = REQ_PUCT;
                            v2_req_node = node_idx;
                            break;
                        }
                        int best_a = v2_req_node;
                        v2_puct_node = -1;
                        TreeSlotV2& sl = slots[nd.slot];
                        if (best_a < 0) { v2_done = 1; break; }
                        int ci = sl.child[best_a];
                        if (ci < 0) {
                            if (v2_n_nodes >= v2.node_cap) { ep->v2_stats[ST_CAP_HITS]++; v2_done = 1; break; }
                            ci = v2_n_nodes++;
                            sl.child[best_a] = ci;
                            nd.n_kids++;
                            v2_init_node(nodes[ci], node_idx, best_a, scratch.step + 1, KIND_DECISION);
                        }
                        scratch.apply_action(best_a, team, scratch.current_is_pick());
                        int fa;
                        while (!scratch.is_terminal() && (fa = P3_FORCED(scratch)) >= 0)
                            scratch.apply_action(fa, scratch.current_team(), 0);
                        nodes[ci].kind = v2_kind_of(scratch, mode);
                        node_idx = ci;
                        if (v2_path_len < MAX_PATH_DEPTH) v2_path[v2_path_len++] = node_idx;
                    } else if (mode == SEARCH_ROLLFWD) {
                        v2_req = REQ_GD_STEP;
                        break;
                    } else {
                        if (nd.slot < 0) {
                            if (v2_n_slots >= v2.slot_cap) { ep->v2_stats[ST_CAP_HITS]++; v2_done = 1; break; }
                            v2_req = REQ_GD_CACHE;
                            v2_req_node = node_idx;
                            break;
                        }
                        TreeSlotV2& sl = slots[nd.slot];
                        int limit = 1 << 30;
                        if (v2.pw_k > 0.0f)
                            limit = max(1, (int)ceilf(v2.pw_k * powf((float)nd.visits, v2.pw_alpha) - 1e-3f));
                        float r = curand_uniform(&rng);
                        int a = nd.n_kids < limit ? v2_sample(sl.p, r) : v2_sample_existing(sl, r);
                        if (a < 0) { v2_done = 1; break; }
                        int ci = sl.child[a];
                        if (ci < 0) {
                            if (v2_n_nodes >= v2.node_cap) { ep->v2_stats[ST_CAP_HITS]++; v2_done = 1; break; }
                            ci = v2_n_nodes++;
                            sl.child[a] = ci;
                            nd.n_kids++;
                            v2_init_node(nodes[ci], node_idx, a, scratch.step + 1, KIND_DECISION);
                            eager_nodes++;
                        }
                        scratch.apply_action(a, team, scratch.current_is_pick());
                        int fa;
                        while (!scratch.is_terminal() && (fa = P3_FORCED(scratch)) >= 0)
                            scratch.apply_action(fa, scratch.current_team(), 0);
                        nodes[ci].kind = v2_kind_of(scratch, mode);
                        node_idx = ci;
                        if (v2_path_len < MAX_PATH_DEPTH) v2_path[v2_path_len++] = node_idx;
                    }
                }
            }
            __syncthreads();
        }

        if (v2_leaf_expand) {
            scratch.to_float_array(state_buf);
            P3_MASK(scratch);
            P3_PRIORS(scratch);
            {
                int sl = v2_n_slots;
                for (int i = tid; i < NUM_HEROES; i += blockDim.x) {
                    bool ok = mask_buf[i] > 0.5f;
                    slots[sl].p[i] = ok ? priors_buf[i] : 0.0f;
                    slots[sl].child[i] = ok ? CHILD_LAZY : CHILD_NONE;
                }
            }
            __syncthreads();
            if (tid == 0) {
                nodes[v2_path[v2_path_len - 1]].slot = v2_n_slots++;
                int nv = 0;
                for (int i = 0; i < NUM_HEROES; i++) nv += mask_buf[i] > 0.5f;
                eager_nodes += nv;
                ep->v2_stats[ST_POLICY_FWD]++;
            }
            __syncthreads();
        }
        if (tid == 0) {
            int depth = scratch.step - v2_root_step;
            ep->v2_stats[ST_SIMS]++;
            ep->v2_stats[ST_SUM_LEAF_STEPS] += depth;
            ep->v2_stats[ST_MAX_DEPTH] = max(ep->v2_stats[ST_MAX_DEPTH], depth);
            ep->v2_stats[ST_SUM_OWN_AHEAD] += own_ahead;
            ep->v2_stats[ST_HIST0 + min(own_ahead, 7)]++;
        }

        // rollout: forced bans, pool masks, GD for both teams
        while (!scratch.is_terminal()) {
            if (tid == 0) v2_rfa = P3_FORCED(scratch);
            __syncthreads();
            if (v2_rfa >= 0) {
                if (tid == 0) scratch.apply_action(v2_rfa, scratch.current_team(), 0);
                __syncthreads();
                continue;
            }
            scratch.to_float_array(state_buf);
            P3_MASK(scratch);
            P3_GD(scratch);
            if (tid == 0) {
                v2_softmax90(priors_buf);
                int a = v2_sample(priors_buf, curand_uniform(&rng));
                scratch.apply_action(a, scratch.current_team(), scratch.current_is_pick());
            }
            __syncthreads();
        }

        if (tid == 0) {
            v2_n0 = 0; v2_n1 = 0;
            for (int i = 0; i < NUM_HEROES; i++) {
                int w = i / 32, b = i % 32;
                if ((scratch.t0_picks[w] >> b) & 1 && v2_n0 < 5) v2_t0h[v2_n0++] = i;
                if ((scratch.t1_picks[w] >> b) & 1 && v2_n1 < 5) v2_t1h[v2_n1++] = i;
            }
        }
        __syncthreads();
        {
            float val = wp_eval_symmetrized(
                v2_t0h, v2_n0, v2_t1h, v2_n1,
                scratch.map_idx, scratch.tier_idx, scratch.our_team, scratch.step,
                lut, W_wp, wp_off, state_buf, enriched_buf, workspace);
            if (tid == 0) v2_value = P3_VALUE(val, scratch);
        }
        __syncthreads();
        if (tid == 0) {
            for (int p = 0; p < v2_path_len; p++) {
                nodes[v2_path[p]].visits++;
                nodes[v2_path[p]].value_sum += v2_value;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        ep->v2_stats[ST_MAX_NODES] = max(ep->v2_stats[ST_MAX_NODES], v2_n_nodes);
        ep->v2_stats[ST_MAX_SLOTS] = max(ep->v2_stats[ST_MAX_SLOTS], v2_n_slots);
        ep->v2_stats[ST_MAX_EAGER] = max(ep->v2_stats[ST_MAX_EAGER], eager_nodes);
        if (v2_n_nodes > ep->max_nodes_used) ep->max_nodes_used = v2_n_nodes;
        ep->n_capacity_hits = ep->v2_stats[ST_CAP_HITS];
        int t = ep->num_our_turns;
        float vsum = 0;
        for (int i = 0; i < NUM_HEROES; i++) { ep->out_policies[t][i] = 0; ep->out_q[t][i] = 0.0f; }
        for (int i = 0; i < NUM_HEROES; i++) {
            int ci = slots[0].child[i];
            if (ci < 0) continue;
            float v = (float)nodes[ci].visits;
            ep->out_policies[t][i] = v;
            ep->out_q[t][i] = nodes[ci].visits == 0 ? 0.0f : nodes[ci].value_sum / (float)nodes[ci].visits;
            vsum += v;
        }
        if (vsum > 0) for (int i = 0; i < NUM_HEROES; i++) ep->out_policies[t][i] /= vsum;
    }
    __syncthreads();
}
