/**
 * GPU-side enriched feature computation for WP model evaluation.
 *
 * Computes the 86 enriched features from draft state + static lookup tables.
 * All lookup tables are uploaded to GPU global memory once at startup.
 *
 * Feature groups (86 total):
 *   role_counts:        18 (9 fine roles × 2 teams)
 *   team_avg_wr:         2 (avg hero WR per team)
 *   map_delta:           2 (sum of hero map WR - overall WR per team)
 *   pairwise_counters:   2 (avg counter delta per team)
 *   pairwise_synergies:  2 (avg synergy delta per team)
 *   counter_detail:     50 (5×5 cross-team hero matchup deltas, both perspectives)
 *   meta_strength:       4 (avg pick_rate + avg ban_rate per team)
 *   draft_diversity:     2 (std dev of hero WRs per team)
 *   comp_wr:             4 (composition WR + log_games per team)
 */
#pragma once
#include <cuda_runtime.h>
#include <cmath>

#ifndef NUM_HEROES
#define NUM_HEROES 90
#endif
#ifndef NUM_MAPS
#define NUM_MAPS 14
#endif
#ifndef NUM_TIERS
#define NUM_TIERS 3
#endif
#define NUM_FINE_ROLES 9
#define NUM_BLIZZ_ROLES 6
#define MAX_COMP_ENTRIES 512  // max compositions per tier
#define ENRICHED_DIM 86
#define WP_BASE_DIM 197
#define WP_FULL_DIM 283  // 197 + 86

// ── Lookup table structures (uploaded to GPU once) ──

struct WPLookupTables {
    // hero_wr[tier][hero] = win rate (50.0 default)
    float hero_wr[NUM_TIERS][NUM_HEROES];

    // hero_map_wr[tier][map][hero] = map-specific win rate (0 = use hero_wr)
    float hero_map_wr[NUM_TIERS][NUM_MAPS][NUM_HEROES];

    // pairwise_counter[tier][heroA][heroB] = WR of A vs B (0 = no data)
    // Only store for pairs with >= 30 games; 0 means missing
    float pairwise_counter[NUM_TIERS][NUM_HEROES][NUM_HEROES];

    // pairwise_synergy[tier][heroA][heroB] = WR of A with B (0 = no data)
    float pairwise_synergy[NUM_TIERS][NUM_HEROES][NUM_HEROES];

    // hero_meta[tier][hero][0=pick_rate, 1=ban_rate]
    float hero_meta[NUM_TIERS][NUM_HEROES][2];

    // hero_fine_role[hero] = fine role index (0-8)
    int hero_fine_role[NUM_HEROES];

    // hero_blizz_role[hero] = blizzard role index (0-5)
    int hero_blizz_role[NUM_HEROES];

    // comp_wr: sorted 5-role tuple -> (wr, games)
    // Encoded as a hash table: key = role tuple hash, value = (wr, games)
    // Simple encoding: 5 blizz roles sorted, packed into 20 bits (4 bits each)
    int comp_keys[NUM_TIERS][MAX_COMP_ENTRIES];      // packed role tuple
    float comp_wr[NUM_TIERS][MAX_COMP_ENTRIES];       // win rate
    float comp_games[NUM_TIERS][MAX_COMP_ENTRIES];     // game count
    int comp_count[NUM_TIERS];                         // entries per tier
};


// ── Device functions for enriched feature computation ──

// Pack 5 sorted blizzard role indices into one int for comp lookup
__device__ int pack_comp_key(int roles[5]) {
    // Sort roles (5 elements, simple bubble sort)
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 5; j++)
            if (roles[j] < roles[i]) { int t = roles[i]; roles[i] = roles[j]; roles[j] = t; }
    return (roles[0]) | (roles[1] << 4) | (roles[2] << 8) | (roles[3] << 12) | (roles[4] << 16);
}

// Look up composition WR from hash table
__device__ void lookup_comp_wr(
    const WPLookupTables* lut, int tier, int key,
    float* out_wr, float* out_games
) {
    // Linear scan (tables are small, ~164 entries max)
    for (int i = 0; i < lut->comp_count[tier]; i++) {
        if (lut->comp_keys[tier][i] == key) {
            *out_wr = lut->comp_wr[tier][i];
            *out_games = lut->comp_games[tier][i];
            return;
        }
    }
    // Fallback to all tiers
    for (int t = 0; t < NUM_TIERS; t++) {
        if (t == tier) continue;
        for (int i = 0; i < lut->comp_count[t]; i++) {
            if (lut->comp_keys[t][i] == key) {
                *out_wr = lut->comp_wr[t][i];
                *out_games = lut->comp_games[t][i];
                return;
            }
        }
    }
    // Unknown composition = pessimistic default
    *out_wr = 33.0f;
    *out_games = 0.0f;
}

// Compute all 86 enriched features from draft state picks
// Thread 0 only (sequential, but fast for small data)
// Output: enriched[86] floats
__device__ void compute_enriched_features(
    const int* t0_heroes, int n_t0,    // hero indices for team 0
    const int* t1_heroes, int n_t1,    // hero indices for team 1
    int map_idx, int tier_idx,
    const WPLookupTables* lut,
    float* enriched                     // output: 86 floats
) {
    int off = 0;

    // ── role_counts (18): 9 fine roles × 2 teams ──
    for (int r = 0; r < NUM_FINE_ROLES; r++) {
        float c0 = 0, c1 = 0;
        for (int i = 0; i < n_t0; i++)
            if (lut->hero_fine_role[t0_heroes[i]] == r) c0 += 1.0f;
        for (int i = 0; i < n_t1; i++)
            if (lut->hero_fine_role[t1_heroes[i]] == r) c1 += 1.0f;
        enriched[off++] = c0;
        enriched[off++] = c1;
    }

    // ── team_avg_wr (2) ──
    float t0_wr_sum = 0, t1_wr_sum = 0;
    for (int i = 0; i < n_t0; i++)
        t0_wr_sum += lut->hero_wr[tier_idx][t0_heroes[i]];
    for (int i = 0; i < n_t1; i++)
        t1_wr_sum += lut->hero_wr[tier_idx][t1_heroes[i]];
    enriched[off++] = n_t0 > 0 ? (t0_wr_sum / n_t0 - 50.0f) / 5.0f : 0.0f;
    enriched[off++] = n_t1 > 0 ? (t1_wr_sum / n_t1 - 50.0f) / 5.0f : 0.0f;

    // ── map_delta (2): sum of (hero_map_wr - hero_wr) per team ──
    float t0_map_delta = 0, t1_map_delta = 0;
    for (int i = 0; i < n_t0; i++) {
        float mwr = lut->hero_map_wr[tier_idx][map_idx][t0_heroes[i]];
        float hwr = lut->hero_wr[tier_idx][t0_heroes[i]];
        if (mwr > 0) t0_map_delta += (mwr - hwr);
    }
    for (int i = 0; i < n_t1; i++) {
        float mwr = lut->hero_map_wr[tier_idx][map_idx][t1_heroes[i]];
        float hwr = lut->hero_wr[tier_idx][t1_heroes[i]];
        if (mwr > 0) t1_map_delta += (mwr - hwr);
    }
    enriched[off++] = t0_map_delta / 5.0f;
    enriched[off++] = t1_map_delta / 5.0f;

    // ── pairwise_counters (2): avg normalized counter delta per team ──
    float t0_counter_sum = 0, t1_counter_sum = 0;
    int t0_counter_n = 0, t1_counter_n = 0;
    for (int i = 0; i < n_t0; i++) {
        float hwr = lut->hero_wr[tier_idx][t0_heroes[i]];
        for (int j = 0; j < n_t1; j++) {
            float raw = lut->pairwise_counter[tier_idx][t0_heroes[i]][t1_heroes[j]];
            if (raw > 0) {
                float owr = lut->hero_wr[tier_idx][t1_heroes[j]];
                float expected = hwr + (100.0f - owr) - 50.0f;
                t0_counter_sum += (raw - expected);
                t0_counter_n++;
            }
        }
    }
    for (int i = 0; i < n_t1; i++) {
        float hwr = lut->hero_wr[tier_idx][t1_heroes[i]];
        for (int j = 0; j < n_t0; j++) {
            float raw = lut->pairwise_counter[tier_idx][t1_heroes[i]][t0_heroes[j]];
            if (raw > 0) {
                float owr = lut->hero_wr[tier_idx][t0_heroes[j]];
                float expected = hwr + (100.0f - owr) - 50.0f;
                t1_counter_sum += (raw - expected);
                t1_counter_n++;
            }
        }
    }
    enriched[off++] = t0_counter_n > 0 ? t0_counter_sum / t0_counter_n / 10.0f : 0.0f;
    enriched[off++] = t1_counter_n > 0 ? t1_counter_sum / t1_counter_n / 10.0f : 0.0f;

    // ── pairwise_synergies (2): avg normalized synergy delta per team ──
    float t0_syn_sum = 0, t1_syn_sum = 0;
    int t0_syn_n = 0, t1_syn_n = 0;
    for (int i = 0; i < n_t0; i++) {
        float wr_i = lut->hero_wr[tier_idx][t0_heroes[i]];
        for (int j = i+1; j < n_t0; j++) {
            float raw = lut->pairwise_synergy[tier_idx][t0_heroes[i]][t0_heroes[j]];
            if (raw > 0) {
                float wr_j = lut->hero_wr[tier_idx][t0_heroes[j]];
                float expected = 50.0f + (wr_i - 50.0f) + (wr_j - 50.0f);
                t0_syn_sum += (raw - expected);
                t0_syn_n++;
            }
        }
    }
    for (int i = 0; i < n_t1; i++) {
        float wr_i = lut->hero_wr[tier_idx][t1_heroes[i]];
        for (int j = i+1; j < n_t1; j++) {
            float raw = lut->pairwise_synergy[tier_idx][t1_heroes[i]][t1_heroes[j]];
            if (raw > 0) {
                float wr_j = lut->hero_wr[tier_idx][t1_heroes[j]];
                float expected = 50.0f + (wr_i - 50.0f) + (wr_j - 50.0f);
                t1_syn_sum += (raw - expected);
                t1_syn_n++;
            }
        }
    }
    enriched[off++] = t0_syn_n > 0 ? t0_syn_sum / t0_syn_n / 10.0f : 0.0f;
    enriched[off++] = t1_syn_n > 0 ? t1_syn_sum / t1_syn_n / 10.0f : 0.0f;

    // ── counter_detail (50): all 25 cross-team matchup deltas × 2 ──
    // For each (t0_hero_i, t1_hero_j), compute normalized counter delta
    // Pad with 0 if fewer than 5 heroes per team
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (i < n_t0 && j < n_t1) {
                float raw = lut->pairwise_counter[tier_idx][t0_heroes[i]][t1_heroes[j]];
                if (raw > 0) {
                    float hwr = lut->hero_wr[tier_idx][t0_heroes[i]];
                    float owr = lut->hero_wr[tier_idx][t1_heroes[j]];
                    float expected = hwr + (100.0f - owr) - 50.0f;
                    enriched[off++] = (raw - expected) / 10.0f;
                } else {
                    enriched[off++] = 0.0f;
                }
            } else {
                enriched[off++] = 0.0f;
            }
        }
    }
    // Reverse: t1 vs t0
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (i < n_t1 && j < n_t0) {
                float raw = lut->pairwise_counter[tier_idx][t1_heroes[i]][t0_heroes[j]];
                if (raw > 0) {
                    float hwr = lut->hero_wr[tier_idx][t1_heroes[i]];
                    float owr = lut->hero_wr[tier_idx][t0_heroes[j]];
                    float expected = hwr + (100.0f - owr) - 50.0f;
                    enriched[off++] = (raw - expected) / 10.0f;
                } else {
                    enriched[off++] = 0.0f;
                }
            } else {
                enriched[off++] = 0.0f;
            }
        }
    }

    // ── meta_strength (4): avg pick_rate + avg ban_rate per team ──
    float t0_pick = 0, t0_ban = 0, t1_pick = 0, t1_ban = 0;
    for (int i = 0; i < n_t0; i++) {
        t0_pick += lut->hero_meta[tier_idx][t0_heroes[i]][0];
        t0_ban += lut->hero_meta[tier_idx][t0_heroes[i]][1];
    }
    for (int i = 0; i < n_t1; i++) {
        t1_pick += lut->hero_meta[tier_idx][t1_heroes[i]][0];
        t1_ban += lut->hero_meta[tier_idx][t1_heroes[i]][1];
    }
    enriched[off++] = n_t0 > 0 ? t0_pick / n_t0 : 0.0f;
    enriched[off++] = n_t0 > 0 ? t0_ban / n_t0 : 0.0f;
    enriched[off++] = n_t1 > 0 ? t1_pick / n_t1 : 0.0f;
    enriched[off++] = n_t1 > 0 ? t1_ban / n_t1 : 0.0f;

    // ── draft_diversity (2): std dev of hero WRs per team ──
    float t0_mean = n_t0 > 0 ? t0_wr_sum / n_t0 : 50.0f;
    float t1_mean = n_t1 > 0 ? t1_wr_sum / n_t1 : 50.0f;
    float t0_var = 0, t1_var = 0;
    for (int i = 0; i < n_t0; i++) {
        float d = lut->hero_wr[tier_idx][t0_heroes[i]] - t0_mean;
        t0_var += d * d;
    }
    for (int i = 0; i < n_t1; i++) {
        float d = lut->hero_wr[tier_idx][t1_heroes[i]] - t1_mean;
        t1_var += d * d;
    }
    enriched[off++] = n_t0 > 1 ? sqrtf(t0_var / n_t0) / 5.0f : 0.0f;
    enriched[off++] = n_t1 > 1 ? sqrtf(t1_var / n_t1) / 5.0f : 0.0f;

    // ── comp_wr (4): composition WR + log_games per team ──
    if (n_t0 == 5) {
        int roles[5];
        for (int i = 0; i < 5; i++) roles[i] = lut->hero_blizz_role[t0_heroes[i]];
        int key = pack_comp_key(roles);
        float wr, games;
        lookup_comp_wr(lut, tier_idx, key, &wr, &games);
        enriched[off++] = (wr - 50.0f) / 10.0f;
        enriched[off++] = log1pf(games) / 15.0f;
    } else {
        enriched[off++] = (33.0f - 50.0f) / 10.0f;  // unknown = pessimistic
        enriched[off++] = 0.0f;
    }
    if (n_t1 == 5) {
        int roles[5];
        for (int i = 0; i < 5; i++) roles[i] = lut->hero_blizz_role[t1_heroes[i]];
        int key = pack_comp_key(roles);
        float wr, games;
        lookup_comp_wr(lut, tier_idx, key, &wr, &games);
        enriched[off++] = (wr - 50.0f) / 10.0f;
        enriched[off++] = log1pf(games) / 15.0f;
    } else {
        enriched[off++] = (33.0f - 50.0f) / 10.0f;
        enriched[off++] = 0.0f;
    }
}


// ── WP model forward pass (enriched, 283→MLP→1) ──
// Architecture varies by variant:
//   base: 197→1024(BN+ReLU)→512(BN+ReLU)→512(BN+ReLU)→128(ReLU)→1(Sigmoid)
//   enriched: 283→256→128→1 (simple MLP, dropout=0.3 off in eval)
//   augmented: 283→512→256→128→1

struct WPNetOffsets {
    int num_layers;        // number of linear layers
    int layer_in[6];       // input dim per layer
    int layer_out[6];      // output dim per layer
    int weight_off[6];     // offset into weight array for each layer's weight
    int bias_off[6];       // offset for each layer's bias
    int has_bn[6];         // 1 if BatchNorm follows this layer
    int bn_w_off[6];       // BN gamma offset
    int bn_b_off[6];       // BN beta offset
    int bn_m_off[6];       // BN running_mean offset
    int bn_v_off[6];       // BN running_var offset
    int use_relu[6];       // 1 if ReLU after this layer (+ BN if present)
    int use_sigmoid;       // 1 if final activation is sigmoid (vs none)
    int input_dim;         // total input dimension (197 or 283)
};

// Generic WP forward pass: handles any MLP with optional BN layers
// All 256 threads cooperate on linear layers
// workspace needs max(layer_out) floats × 2 for ping-pong
__device__ float wp_forward_device(
    const float* __restrict__ input,    // (input_dim,) in shared memory
    const float* __restrict__ W,        // WP weights in global memory
    WPNetOffsets off,
    float* workspace                    // scratch: 2 × max_hidden floats
) {
    float* buf_a = workspace;
    float* buf_b = workspace + 1024;  // max hidden = 1024

    // First layer: input → buf_a
    d_linear_layer(input, off.layer_in[0],
                   W + off.weight_off[0], W + off.bias_off[0],
                   off.layer_out[0], buf_a);
    if (off.has_bn[0])
        d_batchnorm_relu(buf_a, off.layer_out[0],
                         W + off.bn_w_off[0], W + off.bn_b_off[0],
                         W + off.bn_m_off[0], W + off.bn_v_off[0]);
    else if (off.use_relu[0])
        d_relu_inplace(buf_a, off.layer_out[0]);

    // Remaining layers: ping-pong between buf_a and buf_b
    float* src = buf_a;
    float* dst = buf_b;
    for (int l = 1; l < off.num_layers; l++) {
        d_linear_layer(src, off.layer_in[l],
                       W + off.weight_off[l], W + off.bias_off[l],
                       off.layer_out[l], dst);
        if (off.has_bn[l])
            d_batchnorm_relu(dst, off.layer_out[l],
                             W + off.bn_w_off[l], W + off.bn_b_off[l],
                             W + off.bn_m_off[l], W + off.bn_v_off[l]);
        else if (off.use_relu[l])
            d_relu_inplace(dst, off.layer_out[l]);

        // Swap
        float* tmp = src; src = dst; dst = tmp;
    }

    // Final output is in src[0]
    float result = src[0];
    if (off.use_sigmoid) result = 1.0f / (1.0f + expf(-result));
    return result;
}

// Symmetrized WP evaluation using enriched features
// Runs WP model twice (normal + team-swapped) and averages
__device__ float wp_eval_symmetrized(
    const int* t0_heroes, int n_t0,
    const int* t1_heroes, int n_t1,
    int map_idx, int tier_idx, int our_team,
    const WPLookupTables* lut,
    const float* W_wp,
    WPNetOffsets wp_off,
    float* state_buf,     // shared: WP_FULL_DIM floats for building WP input
    float* enriched_buf,  // shared: ENRICHED_DIM floats
    float* workspace      // shared: 2048 floats for MLP forward
) {
    int tid = threadIdx.x;

    // ── Normal perspective: t0 as team0, t1 as team1 ──
    // Build base features (197 dims): multi-hot heroes + map + tier
    for (int i = tid; i < WP_FULL_DIM; i += blockDim.x) state_buf[i] = 0.0f;
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < n_t0; i++) state_buf[t0_heroes[i]] = 1.0f;
        for (int i = 0; i < n_t1; i++) state_buf[NUM_HEROES + t1_heroes[i]] = 1.0f;
        state_buf[2 * NUM_HEROES + map_idx] = 1.0f;  // Wait -- base WP has no bans slot
        // Base WP: t0(90) + t1(90) + map(14) + tier(3) = 197
        // We're putting map at position 180 (90+90), tier at 194 (90+90+14)
        // Actually need to recheck the base feature layout...
        // The sweep_enriched_wp extract_features does:
        //   base = concat(t0_multi_hot, t1_multi_hot, map_one_hot, tier_one_hot) = 197
        // So: positions 0-89 = t0, 90-179 = t1, 180-193 = map, 194-196 = tier
        // Clear and redo:
        for (int i = 0; i < WP_FULL_DIM; i++) state_buf[i] = 0.0f;
        for (int i = 0; i < n_t0; i++) state_buf[t0_heroes[i]] = 1.0f;
        for (int i = 0; i < n_t1; i++) state_buf[90 + t1_heroes[i]] = 1.0f;
        state_buf[180 + map_idx] = 1.0f;
        state_buf[194 + tier_idx] = 1.0f;

        // Compute enriched features
        compute_enriched_features(t0_heroes, n_t0, t1_heroes, n_t1,
                                  map_idx, tier_idx, lut, enriched_buf);
        // Append enriched to base
        for (int i = 0; i < ENRICHED_DIM; i++)
            state_buf[WP_BASE_DIM + i] = enriched_buf[i];
    }
    __syncthreads();

    float wp_normal = wp_forward_device(state_buf, W_wp, wp_off, workspace);
    __syncthreads();

    // ── Swapped perspective: t1 as team0, t0 as team1 ──
    for (int i = tid; i < WP_FULL_DIM; i += blockDim.x) state_buf[i] = 0.0f;
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < n_t1; i++) state_buf[t1_heroes[i]] = 1.0f;
        for (int i = 0; i < n_t0; i++) state_buf[90 + t0_heroes[i]] = 1.0f;
        state_buf[180 + map_idx] = 1.0f;
        state_buf[194 + tier_idx] = 1.0f;

        compute_enriched_features(t1_heroes, n_t1, t0_heroes, n_t0,
                                  map_idx, tier_idx, lut, enriched_buf);
        for (int i = 0; i < ENRICHED_DIM; i++)
            state_buf[WP_BASE_DIM + i] = enriched_buf[i];
    }
    __syncthreads();

    float wp_swapped = wp_forward_device(state_buf, W_wp, wp_off, workspace);
    __syncthreads();

    // Symmetrize and convert to our_team perspective
    float wp_t0 = (wp_normal + (1.0f - wp_swapped)) / 2.0f;
    return (our_team == 0) ? wp_t0 : (1.0f - wp_t0);
}
