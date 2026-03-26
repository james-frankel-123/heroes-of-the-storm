/**
 * C++ MCTS engine with fused CUDA inference.
 * Complete tree traversal + leaf evaluation without any Python calls.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace py = pybind11;

// ── Constants ──────────────────────────────────────────────────────

constexpr int NUM_HEROES = 90;
constexpr int NUM_MAPS = 14;
constexpr int NUM_TIERS = 3;
constexpr int STATE_DIM = 290;
constexpr int GD_STATE_DIM = 289;
constexpr int MAX_NODES = 4096;
constexpr int MAX_CHILDREN = MAX_NODES * 10;

constexpr int DRAFT_ORDER_TEAM[16] = {0,1,0,1, 0,1,1,0,0, 1,0, 1,1,0,0,1};
constexpr int DRAFT_ORDER_IS_PICK[16] = {0,0,0,0, 1,1,1,1,1, 0,0, 1,1,1,1,1};

// ── Weight offset structs (must match fused_forward.cu) ────────────

struct PolicyNetOffsets {
    int input_fc_w, input_fc_b;
    int input_bn_w, input_bn_b, input_bn_mean, input_bn_var;
    int res1_fc1_w, res1_fc1_b, res1_bn1_w, res1_bn1_b, res1_bn1_mean, res1_bn1_var;
    int res1_fc2_w, res1_fc2_b, res1_bn2_w, res1_bn2_b, res1_bn2_mean, res1_bn2_var;
    int res2_fc1_w, res2_fc1_b, res2_bn1_w, res2_bn1_b, res2_bn1_mean, res2_bn1_var;
    int res2_fc2_w, res2_fc2_b, res2_bn2_w, res2_bn2_b, res2_bn2_mean, res2_bn2_var;
    int res3_fc1_w, res3_fc1_b, res3_bn1_w, res3_bn1_b, res3_bn1_mean, res3_bn1_var;
    int res3_fc2_w, res3_fc2_b, res3_bn2_w, res3_bn2_b, res3_bn2_mean, res3_bn2_var;
    int compress1_w, compress1_b, compress1_bn_w, compress1_bn_b, compress1_bn_mean, compress1_bn_var;
    int compress2_w, compress2_b, compress2_bn_w, compress2_bn_b, compress2_bn_mean, compress2_bn_var;
    int policy_w, policy_b;
    int value_fc1_w, value_fc1_b;
    int value_fc2_w, value_fc2_b;
    int value_out_w, value_out_b;
};

struct GDNetOffsets {
    int fc1_w, fc1_b;
    int fc2_w, fc2_b;
    int fc3_w, fc3_b;
};

// External kernel declarations (defined in fused_forward.cu with extern "C")
extern "C" void fused_policy_forward(
    const float* state, const float* valid_mask, const float* weights,
    float* priors_out, float* value_out, PolicyNetOffsets offsets);
extern "C" void fused_gd_forward(
    const float* state, const float* valid_mask, const float* weights,
    float* logits_out, GDNetOffsets offsets);
extern "C" void fused_policy_forward_batched(
    const float* states, const float* valid_masks, const float* weights,
    float* priors_out, float* values_out, PolicyNetOffsets offsets, int K);

// ── DraftState ─────────────────────────────────────────────────────

struct DraftState {
    uint32_t t0_picks[3] = {};
    uint32_t t1_picks[3] = {};
    uint32_t bans[3] = {};
    uint32_t taken[3] = {};
    int step = 0;
    int map_idx = 0;
    int tier_idx = 0;
    int our_team = 0;

    void apply_action(int hero_idx, int team, bool is_pick) {
        int word = hero_idx / 32;
        uint32_t bit = 1u << (hero_idx % 32);
        taken[word] |= bit;
        if (!is_pick) bans[word] |= bit;
        else if (team == 0) t0_picks[word] |= bit;
        else t1_picks[word] |= bit;
        step++;
    }

    bool is_taken(int hero_idx) const {
        return (taken[hero_idx / 32] >> (hero_idx % 32)) & 1;
    }

    bool is_terminal() const { return step >= 16; }
    int current_team() const { return DRAFT_ORDER_TEAM[step]; }
    bool current_is_pick() const { return DRAFT_ORDER_IS_PICK[step]; }

    void to_float_array(float* out) const {
        std::memset(out, 0, STATE_DIM * sizeof(float));
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
            out[STATE_DIM - 2] = DRAFT_ORDER_IS_PICK[step] ? 1.0f : 0.0f;
        } else {
            out[STATE_DIM - 3] = 1.0f;
            out[STATE_DIM - 2] = 1.0f;
        }
        out[STATE_DIM - 1] = (float)our_team;
    }

    void valid_mask(float* out) const {
        for (int i = 0; i < NUM_HEROES; i++)
            out[i] = is_taken(i) ? 0.0f : 1.0f;
    }
};

// ── MCTSNode + Tree ────────────────────────────────────────────────

struct MCTSNode {
    int parent_idx = -1;
    int action = -1;
    float prior = 0.0f;
    int visit_count = 0;
    float value_sum = 0.0f;
    int children_start = -1;
    int num_children = 0;
    bool is_expanded = false;
    float cached_opp_probs[NUM_HEROES] = {};
    bool has_cached_opp = false;

    float q_value() const {
        return visit_count == 0 ? 0.0f : value_sum / visit_count;
    }
};

struct MCTSTree {
    MCTSNode nodes[MAX_NODES];
    int child_indices[MAX_CHILDREN];
    int num_nodes = 0;
    int num_children_allocated = 0;

    int alloc_node() { return num_nodes++; }
    int alloc_children(int count) {
        int start = num_children_allocated;
        num_children_allocated += count;
        return start;
    }
};

// ── CUDA Inference Engine ──────────────────────────────────────────

class CUDAInferenceEngine {
public:
    CUDAInferenceEngine(
        const float* policy_weights_host, int policy_weight_count,
        const float* gd_weights_host, int gd_weight_count,
        PolicyNetOffsets policy_offsets, GDNetOffsets gd_offsets,
        int device_id = 0
    ) : policy_offsets_(policy_offsets), gd_offsets_(gd_offsets) {
        cudaSetDevice(device_id);
        cudaStreamCreate(&stream_);

        // Copy weights to GPU
        cudaMalloc(&d_policy_weights_, policy_weight_count * sizeof(float));
        cudaMemcpy(d_policy_weights_, policy_weights_host,
                   policy_weight_count * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_gd_weights_, gd_weight_count * sizeof(float));
        cudaMemcpy(d_gd_weights_, gd_weights_host,
                   gd_weight_count * sizeof(float), cudaMemcpyHostToDevice);

        // Allocate pinned host + device I/O buffers
        cudaMallocHost(&h_state_, STATE_DIM * sizeof(float));
        cudaMallocHost(&h_mask_, NUM_HEROES * sizeof(float));
        cudaMallocHost(&h_priors_, NUM_HEROES * sizeof(float));
        cudaMallocHost(&h_value_, sizeof(float));
        cudaMallocHost(&h_gd_logits_, NUM_HEROES * sizeof(float));

        cudaMalloc(&d_state_, STATE_DIM * sizeof(float));
        cudaMalloc(&d_mask_, NUM_HEROES * sizeof(float));
        cudaMalloc(&d_priors_, NUM_HEROES * sizeof(float));
        cudaMalloc(&d_value_, sizeof(float));
        cudaMalloc(&d_gd_state_, GD_STATE_DIM * sizeof(float));
        cudaMalloc(&d_gd_logits_, NUM_HEROES * sizeof(float));

        // Batch buffers (for virtual-loss MCTS)
        constexpr int MAX_BATCH = 128;
        cudaMallocHost(&h_batch_states_, MAX_BATCH * STATE_DIM * sizeof(float));
        cudaMallocHost(&h_batch_masks_, MAX_BATCH * NUM_HEROES * sizeof(float));
        cudaMallocHost(&h_batch_priors_, MAX_BATCH * NUM_HEROES * sizeof(float));
        cudaMallocHost(&h_batch_values_, MAX_BATCH * sizeof(float));
        cudaMalloc(&d_batch_states_, MAX_BATCH * STATE_DIM * sizeof(float));
        cudaMalloc(&d_batch_masks_, MAX_BATCH * NUM_HEROES * sizeof(float));
        cudaMalloc(&d_batch_priors_, MAX_BATCH * NUM_HEROES * sizeof(float));
        cudaMalloc(&d_batch_values_, MAX_BATCH * sizeof(float));
    }

    // Policy+value inference (single sample)
    void predict_policy(const float* state_290, const float* mask_90,
                        float* priors_90, float* value_1) {
        cudaMemcpyAsync(d_state_, state_290, STATE_DIM * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_mask_, mask_90, NUM_HEROES * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);

        int shared_mem = (768 * 3 + 512 + 256 + 257) * sizeof(float);
        void* args[] = {&d_state_, &d_mask_, &d_policy_weights_,
                        &d_priors_, &d_value_, &policy_offsets_};
        cudaLaunchKernel((void*)fused_policy_forward, dim3(1), dim3(256),
                         args, shared_mem, stream_);

        cudaMemcpyAsync(priors_90, d_priors_, NUM_HEROES * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(value_1, d_value_, sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

    // GD inference (single sample)
    void predict_gd(const float* state_289, const float* mask_90, float* logits_90) {
        cudaMemcpyAsync(d_gd_state_, state_289, GD_STATE_DIM * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_mask_, mask_90, NUM_HEROES * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);

        int shared_mem = (256 + 128) * sizeof(float);
        void* args[] = {&d_gd_state_, &d_mask_, &d_gd_weights_,
                        &d_gd_logits_, &gd_offsets_};
        cudaLaunchKernel((void*)fused_gd_forward, dim3(1), dim3(128),
                         args, shared_mem, stream_);

        cudaMemcpyAsync(logits_90, d_gd_logits_, NUM_HEROES * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

    // Batched policy+value inference (for virtual-loss MCTS)
    void predict_policy_batch(const float* states, const float* masks,
                              float* priors, float* values, int K) {
        cudaMemcpyAsync(d_batch_states_, states, K * STATE_DIM * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_batch_masks_, masks, K * NUM_HEROES * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);

        int shared_mem = (768 * 3 + 512 + 256 + 257) * sizeof(float);
        void* args[] = {&d_batch_states_, &d_batch_masks_, &d_policy_weights_,
                        &d_batch_priors_, &d_batch_values_, &policy_offsets_, &K};

        cudaLaunchKernel((void*)fused_policy_forward_batched, dim3(K), dim3(256),
                         args, shared_mem, stream_);

        cudaMemcpyAsync(priors, d_batch_priors_, K * NUM_HEROES * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(values, d_batch_values_, K * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

    void update_policy_weights(const float* new_weights, int count) {
        cudaMemcpy(d_policy_weights_, new_weights, count * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    ~CUDAInferenceEngine() {
        cudaFree(d_policy_weights_); cudaFree(d_gd_weights_);
        cudaFree(d_state_); cudaFree(d_mask_); cudaFree(d_priors_); cudaFree(d_value_);
        cudaFree(d_gd_state_); cudaFree(d_gd_logits_);
        cudaFree(d_batch_states_); cudaFree(d_batch_masks_);
        cudaFree(d_batch_priors_); cudaFree(d_batch_values_);
        cudaFreeHost(h_state_); cudaFreeHost(h_mask_);
        cudaFreeHost(h_priors_); cudaFreeHost(h_value_);
        cudaFreeHost(h_gd_logits_);
        cudaFreeHost(h_batch_states_); cudaFreeHost(h_batch_masks_);
        cudaFreeHost(h_batch_priors_); cudaFreeHost(h_batch_values_);
        cudaStreamDestroy(stream_);
    }

private:
    float* d_policy_weights_;
    float* d_gd_weights_;
    PolicyNetOffsets policy_offsets_;
    GDNetOffsets gd_offsets_;
    cudaStream_t stream_;
    float *h_state_, *h_mask_, *h_priors_, *h_value_, *h_gd_logits_;
    float *d_state_, *d_mask_, *d_priors_, *d_value_, *d_gd_state_, *d_gd_logits_;
    float *h_batch_states_, *h_batch_masks_, *h_batch_priors_, *h_batch_values_;
    float *d_batch_states_, *d_batch_masks_, *d_batch_priors_, *d_batch_values_;
};

// ── Symmetrized prediction ─────────────────────────────────────────

void predict_symmetrized(CUDAInferenceEngine& engine, const DraftState& state,
                         float* priors, float* value) {
    float state_buf[STATE_DIM], mask_buf[NUM_HEROES];
    state.to_float_array(state_buf);
    state.valid_mask(mask_buf);

    float value1, value2;
    engine.predict_policy(state_buf, mask_buf, priors, &value1);

    // Flip our_team
    state_buf[STATE_DIM - 1] = 1.0f - state_buf[STATE_DIM - 1];
    float priors_ignored[NUM_HEROES];
    engine.predict_policy(state_buf, mask_buf, priors_ignored, &value2);

    *value = (value1 + (1.0f - value2)) / 2.0f;
}

// ── MCTS Search ────────────────────────────────────────────────────

void mcts_search(
    const DraftState& root_state,
    CUDAInferenceEngine& engine,
    int num_simulations, float c_puct,
    std::mt19937& rng,
    float* visit_dist_out  // (NUM_HEROES,)
) {
    MCTSTree tree;
    int root_idx = tree.alloc_node();
    int our_team = root_state.our_team;

    // Expand root
    float priors[NUM_HEROES], root_value;
    predict_symmetrized(engine, root_state, priors, &root_value);

    float mask[NUM_HEROES];
    root_state.valid_mask(mask);

    float prior_sum = 0;
    for (int i = 0; i < NUM_HEROES; i++) { priors[i] *= mask[i]; prior_sum += priors[i]; }
    if (prior_sum > 0) for (int i = 0; i < NUM_HEROES; i++) priors[i] /= prior_sum;

    int num_valid = 0;
    for (int i = 0; i < NUM_HEROES; i++) if (mask[i] > 0.5f) num_valid++;

    MCTSNode& root = tree.nodes[root_idx];
    root.is_expanded = true;
    root.children_start = tree.alloc_children(num_valid);
    root.num_children = num_valid;

    int ci = 0;
    for (int i = 0; i < NUM_HEROES; i++) {
        if (mask[i] < 0.5f) continue;
        int child_idx = tree.alloc_node();
        tree.child_indices[root.children_start + ci] = child_idx;
        tree.nodes[child_idx].parent_idx = root_idx;
        tree.nodes[child_idx].action = i;
        tree.nodes[child_idx].prior = priors[i];
        ci++;
    }

    // Simulations
    for (int sim = 0; sim < num_simulations; sim++) {
        int node_idx = root_idx;
        DraftState scratch = root_state;
        std::vector<int> path = {root_idx};

        // SELECT
        while (tree.nodes[node_idx].is_expanded && !scratch.is_terminal()) {
            if (scratch.current_team() == our_team) {
                MCTSNode& node = tree.nodes[node_idx];
                float best_score = -1e30f;
                int best_slot = -1;

                for (int c = 0; c < node.num_children; c++) {
                    int ci = tree.child_indices[node.children_start + c];
                    MCTSNode& child = tree.nodes[ci];
                    float q = child.q_value();
                    float u = c_puct * child.prior *
                              std::sqrt((float)node.visit_count) / (1.0f + child.visit_count);
                    if (q + u > best_score) {
                        best_score = q + u;
                        best_slot = c;
                    }
                }

                if (best_slot < 0) break;
                int chosen = tree.child_indices[node.children_start + best_slot];
                scratch.apply_action(tree.nodes[chosen].action,
                                     scratch.current_team(), scratch.current_is_pick());
                node_idx = chosen;
                path.push_back(node_idx);
            } else {
                // Opponent: cached GD distribution
                MCTSNode& node = tree.nodes[node_idx];
                if (!node.has_cached_opp) {
                    float full[STATE_DIM], gd_state[GD_STATE_DIM], gd_mask[NUM_HEROES], gd_logits[NUM_HEROES];
                    scratch.to_float_array(full);
                    std::memcpy(gd_state, full, GD_STATE_DIM * sizeof(float));
                    scratch.valid_mask(gd_mask);

                    engine.predict_gd(gd_state, gd_mask, gd_logits);

                    float max_l = -1e30f;
                    for (int i = 0; i < NUM_HEROES; i++) max_l = std::max(max_l, gd_logits[i]);
                    float sum = 0;
                    for (int i = 0; i < NUM_HEROES; i++) {
                        node.cached_opp_probs[i] = std::exp(gd_logits[i] - max_l);
                        sum += node.cached_opp_probs[i];
                    }
                    if (sum > 0) for (int i = 0; i < NUM_HEROES; i++) node.cached_opp_probs[i] /= sum;
                    node.has_cached_opp = true;
                }

                std::discrete_distribution<int> dist(
                    node.cached_opp_probs, node.cached_opp_probs + NUM_HEROES);
                int opp_action = dist(rng);
                scratch.apply_action(opp_action, scratch.current_team(), scratch.current_is_pick());
            }
        }

        // EVALUATE
        float value;
        if (scratch.is_terminal() || tree.nodes[node_idx].is_expanded) {
            float dummy_priors[NUM_HEROES];
            predict_symmetrized(engine, scratch, dummy_priors, &value);
        } else {
            // Expand leaf
            MCTSNode& leaf = tree.nodes[node_idx];
            if (scratch.current_team() == our_team) {
                float leaf_priors[NUM_HEROES];
                predict_symmetrized(engine, scratch, leaf_priors, &value);

                float leaf_mask[NUM_HEROES];
                scratch.valid_mask(leaf_mask);
                float psum = 0;
                for (int i = 0; i < NUM_HEROES; i++) { leaf_priors[i] *= leaf_mask[i]; psum += leaf_priors[i]; }
                if (psum > 0) for (int i = 0; i < NUM_HEROES; i++) leaf_priors[i] /= psum;

                int nv = 0;
                for (int i = 0; i < NUM_HEROES; i++) if (leaf_mask[i] > 0.5f) nv++;
                leaf.is_expanded = true;
                leaf.children_start = tree.alloc_children(nv);
                leaf.num_children = nv;
                int ci = 0;
                for (int i = 0; i < NUM_HEROES; i++) {
                    if (leaf_mask[i] < 0.5f) continue;
                    int ch = tree.alloc_node();
                    tree.child_indices[leaf.children_start + ci] = ch;
                    tree.nodes[ch].parent_idx = node_idx;
                    tree.nodes[ch].action = i;
                    tree.nodes[ch].prior = leaf_priors[i];
                    ci++;
                }
            } else {
                float dummy_priors[NUM_HEROES];
                predict_symmetrized(engine, scratch, dummy_priors, &value);
            }
        }

        // BACKPROP
        for (int idx : path) {
            tree.nodes[idx].visit_count++;
            tree.nodes[idx].value_sum += value;
        }
    }

    // Extract visit distribution
    std::memset(visit_dist_out, 0, NUM_HEROES * sizeof(float));
    float visit_sum = 0;
    for (int c = 0; c < tree.nodes[root_idx].num_children; c++) {
        int ci = tree.child_indices[tree.nodes[root_idx].children_start + c];
        visit_dist_out[tree.nodes[ci].action] = (float)tree.nodes[ci].visit_count;
        visit_sum += tree.nodes[ci].visit_count;
    }
    if (visit_sum > 0) for (int i = 0; i < NUM_HEROES; i++) visit_dist_out[i] /= visit_sum;
}

// ── Full Episode ───────────────────────────────────────────────────

struct EpisodeResult {
    float win_prob;
    std::vector<std::array<float, STATE_DIM>> states;
    std::vector<std::array<float, NUM_HEROES>> policies;
    std::vector<std::array<float, NUM_HEROES>> masks;
};

EpisodeResult run_episode(
    CUDAInferenceEngine& engine,
    int map_idx, int tier_idx, int our_team,
    int num_simulations, float c_puct, int seed
) {
    std::mt19937 rng(seed);
    DraftState state;
    state.map_idx = map_idx;
    state.tier_idx = tier_idx;
    state.our_team = our_team;

    EpisodeResult result;

    float gd_temperature = 1.0f;
    // Random temperature from {0.5, 0.8, 1.0, 1.2, 1.5}
    float temps[] = {0.5f, 0.8f, 1.0f, 1.2f, 1.5f};
    gd_temperature = temps[rng() % 5];

    while (!state.is_terminal()) {
        int team = state.current_team();
        bool is_pick = state.current_is_pick();

        if (team == our_team) {
            // Our turn: MCTS
            std::array<float, STATE_DIM> state_features;
            state.to_float_array(state_features.data());

            std::array<float, NUM_HEROES> valid;
            state.valid_mask(valid.data());

            float visit_dist[NUM_HEROES];
            mcts_search(state, engine, num_simulations, c_puct, rng, visit_dist);

            std::array<float, NUM_HEROES> visit_arr;
            std::copy(visit_dist, visit_dist + NUM_HEROES, visit_arr.data());

            result.states.push_back(state_features);
            result.policies.push_back(visit_arr);
            result.masks.push_back(valid);

            // Sample action
            std::discrete_distribution<int> dist(visit_dist, visit_dist + NUM_HEROES);
            int action = dist(rng);
            state.apply_action(action, team, is_pick);
        } else {
            // Opponent: GD
            float full[STATE_DIM], gd_state[GD_STATE_DIM], gd_mask[NUM_HEROES], gd_logits[NUM_HEROES];
            state.to_float_array(full);
            std::memcpy(gd_state, full, GD_STATE_DIM * sizeof(float));
            state.valid_mask(gd_mask);

            engine.predict_gd(gd_state, gd_mask, gd_logits);

            // Softmax with temperature
            float max_l = -1e30f;
            for (int i = 0; i < NUM_HEROES; i++) max_l = std::max(max_l, gd_logits[i]);
            float probs[NUM_HEROES], sum = 0;
            for (int i = 0; i < NUM_HEROES; i++) {
                probs[i] = std::exp((gd_logits[i] - max_l) / gd_temperature);
                sum += probs[i];
            }
            if (sum > 0) for (int i = 0; i < NUM_HEROES; i++) probs[i] /= sum;

            std::discrete_distribution<int> dist(probs, probs + NUM_HEROES);
            int action = dist(rng);
            state.apply_action(action, team, is_pick);
        }
    }

    // Terminal evaluation (symmetrized)
    float dummy[NUM_HEROES], terminal_value;
    predict_symmetrized(engine, state, dummy, &terminal_value);
    result.win_prob = (our_team == 0) ? terminal_value : (1.0f - terminal_value);

    return result;
}


// ── pybind11 Bindings ──────────────────────────────────────────────

PolicyNetOffsets dict_to_policy_offsets(py::dict d) {
    PolicyNetOffsets o;
    o.input_fc_w = d["input_fc.weight"].cast<int>();
    o.input_fc_b = d["input_fc.bias"].cast<int>();
    o.input_bn_w = d["input_bn.weight"].cast<int>();
    o.input_bn_b = d["input_bn.bias"].cast<int>();
    o.input_bn_mean = d["input_bn.running_mean"].cast<int>();
    o.input_bn_var = d["input_bn.running_var"].cast<int>();

    auto rb = [&](const char* prefix, int& fc1_w, int& fc1_b,
                   int& bn1_w, int& bn1_b, int& bn1_m, int& bn1_v,
                   int& fc2_w, int& fc2_b,
                   int& bn2_w, int& bn2_b, int& bn2_m, int& bn2_v) {
        std::string p(prefix);
        fc1_w = d[(p + ".fc1.weight").c_str()].cast<int>();
        fc1_b = d[(p + ".fc1.bias").c_str()].cast<int>();
        bn1_w = d[(p + ".bn1.weight").c_str()].cast<int>();
        bn1_b = d[(p + ".bn1.bias").c_str()].cast<int>();
        bn1_m = d[(p + ".bn1.running_mean").c_str()].cast<int>();
        bn1_v = d[(p + ".bn1.running_var").c_str()].cast<int>();
        fc2_w = d[(p + ".fc2.weight").c_str()].cast<int>();
        fc2_b = d[(p + ".fc2.bias").c_str()].cast<int>();
        bn2_w = d[(p + ".bn2.weight").c_str()].cast<int>();
        bn2_b = d[(p + ".bn2.bias").c_str()].cast<int>();
        bn2_m = d[(p + ".bn2.running_mean").c_str()].cast<int>();
        bn2_v = d[(p + ".bn2.running_var").c_str()].cast<int>();
    };

    rb("res_block1", o.res1_fc1_w, o.res1_fc1_b, o.res1_bn1_w, o.res1_bn1_b,
       o.res1_bn1_mean, o.res1_bn1_var, o.res1_fc2_w, o.res1_fc2_b,
       o.res1_bn2_w, o.res1_bn2_b, o.res1_bn2_mean, o.res1_bn2_var);
    rb("res_block2", o.res2_fc1_w, o.res2_fc1_b, o.res2_bn1_w, o.res2_bn1_b,
       o.res2_bn1_mean, o.res2_bn1_var, o.res2_fc2_w, o.res2_fc2_b,
       o.res2_bn2_w, o.res2_bn2_b, o.res2_bn2_mean, o.res2_bn2_var);
    rb("res_block3", o.res3_fc1_w, o.res3_fc1_b, o.res3_bn1_w, o.res3_bn1_b,
       o.res3_bn1_mean, o.res3_bn1_var, o.res3_fc2_w, o.res3_fc2_b,
       o.res3_bn2_w, o.res3_bn2_b, o.res3_bn2_mean, o.res3_bn2_var);

    o.compress1_w = d["compress1.weight"].cast<int>();
    o.compress1_b = d["compress1.bias"].cast<int>();
    o.compress1_bn_w = d["compress1_bn.weight"].cast<int>();
    o.compress1_bn_b = d["compress1_bn.bias"].cast<int>();
    o.compress1_bn_mean = d["compress1_bn.running_mean"].cast<int>();
    o.compress1_bn_var = d["compress1_bn.running_var"].cast<int>();
    o.compress2_w = d["compress2.weight"].cast<int>();
    o.compress2_b = d["compress2.bias"].cast<int>();
    o.compress2_bn_w = d["compress2_bn.weight"].cast<int>();
    o.compress2_bn_b = d["compress2_bn.bias"].cast<int>();
    o.compress2_bn_mean = d["compress2_bn.running_mean"].cast<int>();
    o.compress2_bn_var = d["compress2_bn.running_var"].cast<int>();

    o.policy_w = d["policy_head.weight"].cast<int>();
    o.policy_b = d["policy_head.bias"].cast<int>();
    o.value_fc1_w = d["value_fc1.weight"].cast<int>();
    o.value_fc1_b = d["value_fc1.bias"].cast<int>();
    o.value_fc2_w = d["value_fc2.weight"].cast<int>();
    o.value_fc2_b = d["value_fc2.bias"].cast<int>();
    o.value_out_w = d["value_out.weight"].cast<int>();
    o.value_out_b = d["value_out.bias"].cast<int>();

    return o;
}

GDNetOffsets dict_to_gd_offsets(py::dict d) {
    GDNetOffsets o;
    o.fc1_w = d["net.0.weight"].cast<int>();
    o.fc1_b = d["net.0.bias"].cast<int>();
    o.fc2_w = d["net.3.weight"].cast<int>();
    o.fc2_b = d["net.3.bias"].cast<int>();
    o.fc3_w = d["net.6.weight"].cast<int>();
    o.fc3_b = d["net.6.bias"].cast<int>();
    return o;
}


PYBIND11_MODULE(cuda_mcts, m) {
    m.doc() = "C++ MCTS with fused CUDA inference kernels";

    py::class_<CUDAInferenceEngine>(m, "CUDAInferenceEngine")
        .def(py::init([](py::array_t<float> policy_w, py::array_t<float> gd_w,
                         py::dict policy_off, py::dict gd_off, int device_id) {
            auto pw = policy_w.unchecked<1>();
            auto gw = gd_w.unchecked<1>();
            return new CUDAInferenceEngine(
                pw.data(0), pw.shape(0),
                gw.data(0), gw.shape(0),
                dict_to_policy_offsets(policy_off),
                dict_to_gd_offsets(gd_off),
                device_id
            );
        }), py::arg("policy_weights"), py::arg("gd_weights"),
            py::arg("policy_offsets"), py::arg("gd_offsets"),
            py::arg("device_id") = 0)
        .def("update_policy_weights", [](CUDAInferenceEngine& e, py::array_t<float> w) {
            auto r = w.unchecked<1>();
            e.update_policy_weights(r.data(0), r.shape(0));
        });

    m.def("test_batch_forward", [](CUDAInferenceEngine& engine,
                                  py::array_t<float> states, py::array_t<float> masks) {
        auto s = states.unchecked<2>();
        auto m = masks.unchecked<2>();
        int K = s.shape(0);
        auto priors_np = py::array_t<float>({K, NUM_HEROES});
        auto values_np = py::array_t<float>(K);
        engine.predict_policy_batch(s.data(0, 0), m.data(0, 0),
            priors_np.mutable_data(), values_np.mutable_data(), K);
        return py::make_tuple(priors_np, values_np);
    });

    m.def("test_forward", [](CUDAInferenceEngine& engine,
                             py::array_t<float> state, py::array_t<float> mask) {
        auto s = state.unchecked<1>();
        auto m = mask.unchecked<1>();
        float priors[NUM_HEROES], value;
        engine.predict_policy(s.data(0), m.data(0), priors, &value);
        auto priors_np = py::array_t<float>(NUM_HEROES);
        std::memcpy(priors_np.mutable_data(), priors, NUM_HEROES * sizeof(float));
        return py::make_tuple(priors_np, value);
    });

    m.def("run_episode", [](CUDAInferenceEngine& engine,
                            int map_idx, int tier_idx, int our_team,
                            int num_simulations, float c_puct, int seed) {
        EpisodeResult result = run_episode(engine, map_idx, tier_idx, our_team,
                                           num_simulations, c_puct, seed);

        int n = result.states.size();
        py::list examples;
        for (int i = 0; i < n; i++) {
            auto s = py::array_t<float>(STATE_DIM, result.states[i].data());
            auto p = py::array_t<float>(NUM_HEROES, result.policies[i].data());
            auto v = py::array_t<float>(NUM_HEROES, result.masks[i].data());
            examples.append(py::make_tuple(s, p, v));
        }
        return py::make_tuple(result.win_prob, examples);
    }, py::arg("engine"), py::arg("map_idx"), py::arg("tier_idx"),
       py::arg("our_team"), py::arg("num_simulations"),
       py::arg("c_puct") = 2.0f, py::arg("seed") = 42);
}
