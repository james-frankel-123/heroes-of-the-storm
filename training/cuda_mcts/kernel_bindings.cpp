/**
 * pybind11 bindings for the full MCTS kernel.
 * Host-side code: allocate GPU memory, launch kernel, read results.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <algorithm>

namespace py = pybind11;

#include "hots_dims.h"
#define MAX_OUR_TURNS 8
#define MAX_NODES 4096
#define MAX_CHILD_INDICES 81920

// ── WP model structs (must match enriched_features.cuh) ──

#define NUM_FINE_ROLES 9
#define NUM_BLIZZ_ROLES 6
#define MAX_COMP_ENTRIES 512

struct WPLookupTables {
    float hero_wr[NUM_TIERS][NUM_HEROES];
    float hero_map_wr[NUM_TIERS][NUM_MAPS][NUM_HEROES];
    float pairwise_counter[NUM_TIERS][NUM_HEROES][NUM_HEROES];
    float pairwise_synergy[NUM_TIERS][NUM_HEROES][NUM_HEROES];
    float hero_meta[NUM_TIERS][NUM_HEROES][2];
    int hero_fine_role[NUM_HEROES];
    int hero_blizz_role[NUM_HEROES];
    int comp_keys[NUM_TIERS][MAX_COMP_ENTRIES];
    float comp_wr[NUM_TIERS][MAX_COMP_ENTRIES];
    float comp_games[NUM_TIERS][MAX_COMP_ENTRIES];
    int comp_count[NUM_TIERS];
    float step_embed[16][8];  // step embedding for partial WP model
};

struct WPNetOffsets {
    int use_enriched;
    int num_layers;
    int layer_in[6];
    int layer_out[6];
    int weight_off[6];
    int bias_off[6];
    int has_bn[6];
    int bn_w_off[6];
    int bn_b_off[6];
    int bn_m_off[6];
    int bn_v_off[6];
    int use_relu[6];
    int use_sigmoid;
    int input_dim;
};

// Must match device_forward.cuh
#define MAX_RES_BLOCKS 8

struct PolicyNetOffsets {
    int hdim, cdim, edim, n_blocks;
    int input_fc_w, input_fc_b;
    int input_bn_w, input_bn_b, input_bn_mean, input_bn_var;
    int res_fc1_w[MAX_RES_BLOCKS], res_fc1_b[MAX_RES_BLOCKS];
    int res_bn1_w[MAX_RES_BLOCKS], res_bn1_b[MAX_RES_BLOCKS];
    int res_bn1_mean[MAX_RES_BLOCKS], res_bn1_var[MAX_RES_BLOCKS];
    int res_fc2_w[MAX_RES_BLOCKS], res_fc2_b[MAX_RES_BLOCKS];
    int res_bn2_w[MAX_RES_BLOCKS], res_bn2_b[MAX_RES_BLOCKS];
    int res_bn2_mean[MAX_RES_BLOCKS], res_bn2_var[MAX_RES_BLOCKS];
    int compress1_w, compress1_b, compress1_bn_w, compress1_bn_b, compress1_bn_mean, compress1_bn_var;
    int compress2_w, compress2_b, compress2_bn_w, compress2_bn_b, compress2_bn_mean, compress2_bn_var;

    int policy_head_type;
    int policy_n_layers;
    int policy_layer_in[3];
    int policy_layer_out[3];
    int policy_layer_w[3];
    int policy_layer_b[3];
    int step_embed_w;

    int policy_w, policy_b;
    int value_fc1_w, value_fc1_b;
    int value_fc2_w, value_fc2_b;
    int value_out_w, value_out_b;
};

struct GDNetOffsets {
    int fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b;
};

// Must match mcts_kernel.cu
struct MCTSNodeGPU {
    int parent_idx, action;
    float prior;
    int visit_count;
    float value_sum;
    int children_start, num_children, is_expanded, has_cached_opp;
    float cached_opp_probs[NUM_HEROES];
    float q_value() const { return visit_count == 0 ? 0.0f : value_sum / (float)visit_count; }
};

struct EpisodeMemory {
    MCTSNodeGPU nodes[MAX_NODES];
    int child_indices[MAX_CHILD_INDICES];
    int num_nodes, num_children_allocated;
    float out_states[MAX_OUR_TURNS][STATE_DIM];
    float out_policies[MAX_OUR_TURNS][NUM_HEROES];
    float out_masks[MAX_OUR_TURNS][NUM_HEROES];
    int num_our_turns;
    float win_prob;
    float terminal_state[STATE_DIM];
    int our_team;
};

// Kernel declaration
extern "C" void mcts_episodes_kernel(
    const float*, const float*, const float*,
    PolicyNetOffsets, GDNetOffsets, WPNetOffsets,
    const WPLookupTables*,
    const int*, EpisodeMemory*, int, float, unsigned long long,
    float, float, float);

// v2 search (X2 fix): see search_v2.cuh
#include "tree_v2.h"
extern "C" void mcts_episodes_kernel_v2(
    const float*, const float*, const float*,
    PolicyNetOffsets, GDNetOffsets, WPNetOffsets,
    const WPLookupTables*,
    const int*, EpisodeOutV2*, TreeNodeV2*, TreeSlotV2*, int, int,
    int, float, unsigned long long,
    float, float, float,
    int, float, float, int);

// Default search for every entry point. 0 = legacy (pre-fix, bit-for-bit),
// 1 = opponent chance nodes (the fix), 2 = open-loop roll-forward.
static const int DEFAULT_SEARCH_MODE = SEARCH_CHANCE;
static const float DEFAULT_PW_K = 1.0f;
static const float DEFAULT_PW_ALPHA = 0.5f;

PolicyNetOffsets dict_to_policy_offsets(py::dict d);
GDNetOffsets dict_to_gd_offsets(py::dict d);

// Forward declarations of dict converters (reuse from mcts_engine.cpp)
PolicyNetOffsets dict_to_policy_offsets(py::dict d) {
    PolicyNetOffsets o;
    memset(&o, 0, sizeof(o));

    o.hdim = d["hdim"].cast<int>();
    o.cdim = d["cdim"].cast<int>();
    o.edim = d["edim"].cast<int>();
    o.n_blocks = d["n_blocks"].cast<int>();

    o.input_fc_w = d["input_fc.weight"].cast<int>();
    o.input_fc_b = d["input_fc.bias"].cast<int>();
    o.input_bn_w = d["input_bn.weight"].cast<int>();
    o.input_bn_b = d["input_bn.bias"].cast<int>();
    o.input_bn_mean = d["input_bn.running_mean"].cast<int>();
    o.input_bn_var = d["input_bn.running_var"].cast<int>();

    for (int r = 0; r < o.n_blocks; r++) {
        std::string p = "res_blocks." + std::to_string(r);
        o.res_fc1_w[r] = d[(p+".fc1.weight").c_str()].cast<int>();
        o.res_fc1_b[r] = d[(p+".fc1.bias").c_str()].cast<int>();
        o.res_bn1_w[r] = d[(p+".bn1.weight").c_str()].cast<int>();
        o.res_bn1_b[r] = d[(p+".bn1.bias").c_str()].cast<int>();
        o.res_bn1_mean[r] = d[(p+".bn1.running_mean").c_str()].cast<int>();
        o.res_bn1_var[r] = d[(p+".bn1.running_var").c_str()].cast<int>();
        o.res_fc2_w[r] = d[(p+".fc2.weight").c_str()].cast<int>();
        o.res_fc2_b[r] = d[(p+".fc2.bias").c_str()].cast<int>();
        o.res_bn2_w[r] = d[(p+".bn2.weight").c_str()].cast<int>();
        o.res_bn2_b[r] = d[(p+".bn2.bias").c_str()].cast<int>();
        o.res_bn2_mean[r] = d[(p+".bn2.running_mean").c_str()].cast<int>();
        o.res_bn2_var[r] = d[(p+".bn2.running_var").c_str()].cast<int>();
    }

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

    // Policy head config
    o.policy_head_type = d.contains("policy_head_type") ? d["policy_head_type"].cast<int>() : 0;
    o.policy_n_layers = d.contains("policy_n_layers") ? d["policy_n_layers"].cast<int>() : 1;
    o.step_embed_w = d.contains("step_embed_w") ? d["step_embed_w"].cast<int>() : -1;
    memset(o.policy_layer_in, 0, sizeof(o.policy_layer_in));
    memset(o.policy_layer_out, 0, sizeof(o.policy_layer_out));
    memset(o.policy_layer_w, 0, sizeof(o.policy_layer_w));
    memset(o.policy_layer_b, 0, sizeof(o.policy_layer_b));
    if (d.contains("policy_layers")) {
        auto pl = d["policy_layers"].cast<py::list>();
        for (int i = 0; i < std::min((int)pl.size(), 3); i++) {
            auto layer = pl[i].cast<py::dict>();
            o.policy_layer_in[i] = layer["in"].cast<int>();
            o.policy_layer_out[i] = layer["out"].cast<int>();
            o.policy_layer_w[i] = layer["w"].cast<int>();
            o.policy_layer_b[i] = layer["b"].cast<int>();
        }
    }

    // Legacy single-layer (always populated for backward compat)
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

WPNetOffsets dict_to_wp_offsets(py::dict d) {
    WPNetOffsets o;
    memset(&o, 0, sizeof(o));
    o.use_enriched = d.contains("use_enriched") ? d["use_enriched"].cast<int>() : 1;
    o.num_layers = d["num_layers"].cast<int>();
    o.input_dim = d["input_dim"].cast<int>();
    o.use_sigmoid = d["use_sigmoid"].cast<int>();
    auto li = d["layer_in"].cast<py::list>();
    auto lo = d["layer_out"].cast<py::list>();
    auto wo = d["weight_off"].cast<py::list>();
    auto bo = d["bias_off"].cast<py::list>();
    auto hb = d["has_bn"].cast<py::list>();
    auto bw = d["bn_w_off"].cast<py::list>();
    auto bb = d["bn_b_off"].cast<py::list>();
    auto bm = d["bn_m_off"].cast<py::list>();
    auto bv = d["bn_v_off"].cast<py::list>();
    auto ur = d["use_relu"].cast<py::list>();
    for (int i = 0; i < 6; i++) {
        o.layer_in[i] = li[i].cast<int>();
        o.layer_out[i] = lo[i].cast<int>();
        o.weight_off[i] = wo[i].cast<int>();
        o.bias_off[i] = bo[i].cast<int>();
        o.has_bn[i] = hb[i].cast<int>();
        o.bn_w_off[i] = bw[i].cast<int>();
        o.bn_b_off[i] = bb[i].cast<int>();
        o.bn_m_off[i] = bm[i].cast<int>();
        o.bn_v_off[i] = bv[i].cast<int>();
        o.use_relu[i] = ur[i].cast<int>();
    }
    return o;
}


class MCTSKernelEngine {
public:
    MCTSKernelEngine(
        py::array_t<float> policy_weights,
        py::array_t<float> gd_weights,
        py::array_t<float> wp_weights,
        py::dict policy_offsets_dict,
        py::dict gd_offsets_dict,
        py::dict wp_offsets_dict,
        py::array_t<uint8_t> lut_blob,
        int max_concurrent_episodes = 128,
        int device_id = 0
    ) : max_episodes_(max_concurrent_episodes),
        policy_off_(dict_to_policy_offsets(policy_offsets_dict)),
        gd_off_(dict_to_gd_offsets(gd_offsets_dict)),
        wp_off_(dict_to_wp_offsets(wp_offsets_dict))
    {
        cudaSetDevice(device_id);
        // One non-blocking stream per engine: every copy, launch and wait of this
        // engine is ordered on it, so engines in one process run concurrently and
        // never wait on each other or on torch's default stream.
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

        auto pw = policy_weights.unchecked<1>();
        auto gw = gd_weights.unchecked<1>();
        auto ww = wp_weights.unchecked<1>();
        policy_weight_count_ = pw.shape(0);

        cudaMalloc(&d_policy_weights_, pw.shape(0) * sizeof(float));
        cudaMemcpy(d_policy_weights_, pw.data(0), pw.shape(0) * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_gd_weights_, gw.shape(0) * sizeof(float));
        cudaMemcpy(d_gd_weights_, gw.data(0), gw.shape(0) * sizeof(float), cudaMemcpyHostToDevice);

        // Upload WP model weights
        cudaMalloc(&d_wp_weights_, ww.shape(0) * sizeof(float));
        cudaMemcpy(d_wp_weights_, ww.data(0), ww.shape(0) * sizeof(float), cudaMemcpyHostToDevice);

        // Upload lookup tables (one contiguous struct blob)
        auto lb = lut_blob.unchecked<1>();
        if (lb.shape(0) != sizeof(WPLookupTables)) {
            throw std::runtime_error(
                "LUT blob size mismatch: got " + std::to_string(lb.shape(0)) +
                " expected " + std::to_string(sizeof(WPLookupTables)));
        }
        cudaMalloc(&d_lut_, sizeof(WPLookupTables));
        cudaMemcpy(d_lut_, lb.data(0), sizeof(WPLookupTables), cudaMemcpyHostToDevice);

        cudaMalloc(&d_configs_, max_episodes_ * 3 * sizeof(int));
        // Legacy episode memory (~2 MB per episode) and the v2 arenas are
        // allocated on first use.
        d_episodes_ = nullptr;
        h_episodes_ = nullptr;
        cudaMalloc(&d_outs_, max_episodes_ * sizeof(EpisodeOutV2));
        cudaMallocHost(&h_outs_, max_episodes_ * sizeof(EpisodeOutV2));
        std::memset(h_outs_, 0, max_episodes_ * sizeof(EpisodeOutV2));
        last_n_ = 0;
        last_mode_ = -1;
    }

    int shared_mem_bytes() const {
        return (STATE_BUF_SIZE + NUM_HEROES + NUM_HEROES + policy_off_.edim
                + policy_off_.hdim * 3 + policy_off_.cdim + ENRICHED_DIM) * sizeof(float);
    }

    static void check_cuda(const char* what) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string(what) + ": CUDA error: " + cudaGetErrorString(err));
    }

    // Tree arena per episode for the v2 search. One simulation creates at
    // most 3 nodes and 3 slots, so 3 * sims + 8 cannot overflow; the kernel
    // guards every allocation anyway. set_tree_capacity() overrides (tests).
    void ensure_arena(int num_sims) {
        int need_n = 3 * std::max(num_sims, 1) + 8;
        int need_s = need_n;
        if (cap_override_nodes_ > 0) { need_n = cap_override_nodes_; need_s = cap_override_slots_; }
        if (d_nodes_ != nullptr && need_n <= node_cap_ && need_s <= slot_cap_) return;
        if (d_nodes_) cudaFree(d_nodes_);
        if (d_slots_) cudaFree(d_slots_);
        node_cap_ = need_n;
        slot_cap_ = need_s;
        cudaMalloc(&d_nodes_, (size_t)max_episodes_ * node_cap_ * sizeof(TreeNodeV2));
        cudaMalloc(&d_slots_, (size_t)max_episodes_ * slot_cap_ * sizeof(TreeSlotV2));
        check_cuda("arena alloc");
    }

    void set_tree_capacity(int node_cap, int slot_cap) {
        cap_override_nodes_ = node_cap;
        cap_override_slots_ = slot_cap;
        if (d_nodes_) { cudaFree(d_nodes_); d_nodes_ = nullptr; }
        if (d_slots_) { cudaFree(d_slots_); d_slots_ = nullptr; }
        node_cap_ = slot_cap_ = 0;
    }

    void ensure_legacy() {
        if (d_episodes_ != nullptr) return;
        cudaMalloc(&d_episodes_, max_episodes_ * sizeof(EpisodeMemory));
        cudaMallocHost(&h_episodes_, max_episodes_ * sizeof(EpisodeMemory));
        check_cuda("legacy alloc");
    }

    // Launch one batch. Results land in h_episodes_ (mode 0) or h_outs_ (v2).
    void launch(py::array_t<int> configs, int num_sims, float c_puct,
                unsigned long long seed, float root_temp, float dir_alpha,
                float dir_eps, int search_mode, float pw_k, float pw_alpha,
                int stop_after_turn) {
        auto cfg = configs.unchecked<2>();
        int n = cfg.shape(0);
        if (n > max_episodes_) throw std::runtime_error("Too many episodes");
        if (search_mode < 0 || search_mode > 2) throw std::runtime_error("search_mode must be 0, 1 or 2");
        std::vector<int> cfg_host(cfg.data(0, 0), cfg.data(0, 0) + (size_t)n * 3);
        int shared_mem = shared_mem_bytes();
        py::gil_scoped_release no_gil;  // other Python threads (other engines, training) run meanwhile
        cudaMemcpyAsync(d_configs_, cfg_host.data(), n * 3 * sizeof(int), cudaMemcpyHostToDevice, stream_);
        if (search_mode == SEARCH_LEGACY) {
            if (stop_after_turn >= 0) throw std::runtime_error("stop_after_turn needs a v2 search mode");
            ensure_legacy();
            void* args[] = {&d_policy_weights_, &d_gd_weights_, &d_wp_weights_,
                            &policy_off_, &gd_off_, &wp_off_, &d_lut_,
                            &d_configs_, &d_episodes_, &num_sims, &c_puct, &seed,
                            &root_temp, &dir_alpha, &dir_eps};
            cudaLaunchKernel((void*)mcts_episodes_kernel, dim3(n), dim3(256),
                             args, shared_mem, stream_);
            cudaMemcpyAsync(h_episodes_, d_episodes_, n * sizeof(EpisodeMemory), cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
            check_cuda("legacy kernel");
        } else {
            ensure_arena(num_sims);
            void* args[] = {&d_policy_weights_, &d_gd_weights_, &d_wp_weights_,
                            &policy_off_, &gd_off_, &wp_off_, &d_lut_,
                            &d_configs_, &d_outs_, &d_nodes_, &d_slots_, &node_cap_, &slot_cap_,
                            &num_sims, &c_puct, &seed,
                            &root_temp, &dir_alpha, &dir_eps,
                            &search_mode, &pw_k, &pw_alpha, &stop_after_turn};
            cudaLaunchKernel((void*)mcts_episodes_kernel_v2, dim3(n), dim3(256),
                             args, shared_mem, stream_);
            cudaMemcpyAsync(h_outs_, d_outs_, n * sizeof(EpisodeOutV2), cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
            check_cuda("v2 kernel");
        }
        last_n_ = n;
        last_mode_ = search_mode;
    }

    // Uniform view of one episode's outputs for both layouts.
    struct EpView {
        const float (*states)[STATE_DIM];
        const float (*policies)[NUM_HEROES];
        const float (*masks)[NUM_HEROES];
        int num_our_turns;
        float win_prob;
        const float* terminal_state;
        int our_team;
    };
    EpView view(int i) const {
        if (last_mode_ == SEARCH_LEGACY) {
            const EpisodeMemory& e = h_episodes_[i];
            return {e.out_states, e.out_policies, e.out_masks, e.num_our_turns,
                    e.win_prob, e.terminal_state, e.our_team};
        }
        const EpisodeOutV2& e = h_outs_[i];
        return {e.out_states, e.out_policies, e.out_masks, e.num_our_turns,
                e.win_prob, e.terminal_state, e.our_team};
    }

    py::list run_episodes(py::array_t<int> configs, int num_sims, float c_puct,
                          unsigned long long seed,
                          float root_temp = 1.0f, float dir_alpha = 0.3f,
                          float dir_eps = 0.0f, int search_mode = DEFAULT_SEARCH_MODE,
                          float pw_k = DEFAULT_PW_K, float pw_alpha = DEFAULT_PW_ALPHA) {
        launch(configs, num_sims, c_puct, seed, root_temp, dir_alpha, dir_eps,
               search_mode, pw_k, pw_alpha, -1);
        py::list results;
        for (int i = 0; i < last_n_; i++) {
            EpView ep = view(i);
            py::list examples;
            for (int t = 0; t < ep.num_our_turns; t++) {
                auto s = py::array_t<float>(STATE_DIM);
                auto p = py::array_t<float>(NUM_HEROES);
                auto m = py::array_t<float>(NUM_HEROES);
                std::memcpy(s.mutable_data(), ep.states[t], STATE_DIM * sizeof(float));
                std::memcpy(p.mutable_data(), ep.policies[t], NUM_HEROES * sizeof(float));
                std::memcpy(m.mutable_data(), ep.masks[t], NUM_HEROES * sizeof(float));
                examples.append(py::make_tuple(s, p, m));
            }
            auto ts = py::array_t<float>(STATE_DIM);
            std::memcpy(ts.mutable_data(), ep.terminal_state, STATE_DIM * sizeof(float));
            results.append(py::make_tuple(ep.win_prob, examples, ts, ep.our_team));
        }
        return results;
    }

    // Run episodes, write training data into the ring buffer, return
    // (n_written, wp_values). Training path: root_temp 1, no root noise.
    py::tuple run_episodes_into_buffer(
        py::array_t<int> configs, int num_sims, float c_puct, unsigned long long seed,
        py::array_t<float> buf_states,    // (BUFFER_SIZE, 290)
        py::array_t<float> buf_policies,  // (BUFFER_SIZE, 90)
        py::array_t<float> buf_masks,     // (BUFFER_SIZE, 90)
        py::array_t<float> buf_values,    // (BUFFER_SIZE,)
        int write_offset, int buffer_size,
        int search_mode = DEFAULT_SEARCH_MODE,
        float pw_k = DEFAULT_PW_K, float pw_alpha = DEFAULT_PW_ALPHA
    ) {
        launch(configs, num_sims, c_puct, seed, 1.0f, 0.3f, 0.0f,
               search_mode, pw_k, pw_alpha, -1);
        int n = last_n_;

        auto s_ptr = buf_states.mutable_unchecked<2>();
        auto p_ptr = buf_policies.mutable_unchecked<2>();
        auto m_ptr = buf_masks.mutable_unchecked<2>();
        auto v_ptr = buf_values.mutable_unchecked<1>();
        auto wp_values = py::array_t<float>(n);
        auto wp_ptr = wp_values.mutable_unchecked<1>();

        int write_pos = write_offset;
        int total_written = 0;
        for (int e = 0; e < n; e++) {
            EpView mem = view(e);
            float wp = mem.win_prob;  // kernel-computed symmetrized WP
            wp_ptr(e) = wp;
            for (int t = 0; t < mem.num_our_turns; t++) {
                int idx = write_pos % buffer_size;
                std::memcpy(s_ptr.mutable_data(idx, 0), mem.states[t], STATE_DIM * sizeof(float));
                std::memcpy(p_ptr.mutable_data(idx, 0), mem.policies[t], NUM_HEROES * sizeof(float));
                std::memcpy(m_ptr.mutable_data(idx, 0), mem.masks[t], NUM_HEROES * sizeof(float));
                v_ptr(idx) = wp;
                write_pos++;
                total_written++;
            }
        }
        return py::make_tuple(total_written, wp_values);
    }

    // Per-episode search diagnostics of the last call, (n, NUM_STATS_V2)
    // int32 (indices: tree_v2.h ST_*). Zeros after a legacy call except
    // ST_MAX_NODES = node count of the legacy tree's final search.
    py::array_t<int> last_stats() const {
        auto out = py::array_t<int>({last_n_, NUM_STATS_V2});
        auto o = out.mutable_unchecked<2>();
        for (int i = 0; i < last_n_; i++)
            for (int k = 0; k < NUM_STATS_V2; k++) {
                if (last_mode_ == SEARCH_LEGACY)
                    o(i, k) = (k == ST_MAX_NODES) ? h_episodes_[i].num_nodes : 0;
                else
                    o(i, k) = h_outs_[i].stats[k];
            }
        return out;
    }

    py::dict search_info() const {
        py::dict d;
        d["default_search_mode"] = DEFAULT_SEARCH_MODE;
        d["default_pw_k"] = DEFAULT_PW_K;
        d["default_pw_alpha"] = DEFAULT_PW_ALPHA;
        d["node_cap"] = node_cap_;
        d["slot_cap"] = slot_cap_;
        d["stat_names"] = py::make_tuple(
            "searches", "sims", "max_nodes", "max_slots", "cap_hits", "max_eager_nodes",
            "sum_leaf_steps", "sum_own_ahead", "reached_mask", "gd_tree_fwd", "policy_fwd",
            "max_depth", "hist0", "hist1", "hist2", "hist3", "hist4", "hist5", "hist6", "hist7");
        return d;
    }

    // Run the draft only up to (and including) our turn `turn`, then return
    // that search's tree for every episode: dict of numpy arrays.
    py::list debug_tree(py::array_t<int> configs, int num_sims, float c_puct,
                        unsigned long long seed, int turn,
                        int search_mode = DEFAULT_SEARCH_MODE,
                        float pw_k = DEFAULT_PW_K, float pw_alpha = DEFAULT_PW_ALPHA,
                        float root_temp = 1.0f, float dir_alpha = 0.3f, float dir_eps = 0.0f) {
        if (search_mode == SEARCH_LEGACY) throw std::runtime_error("debug_tree: v2 modes only");
        launch(configs, num_sims, c_puct, seed, root_temp, dir_alpha, dir_eps,
               search_mode, pw_k, pw_alpha, turn);
        int n = last_n_;
        std::vector<TreeNodeV2> hn((size_t)n * node_cap_);
        std::vector<TreeSlotV2> hs((size_t)n * slot_cap_);
        cudaMemcpyAsync(hn.data(), d_nodes_, hn.size() * sizeof(TreeNodeV2), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(hs.data(), d_slots_, hs.size() * sizeof(TreeSlotV2), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        check_cuda("debug_tree copy");
        py::list out;
        for (int e = 0; e < n; e++) {
            const EpisodeOutV2& ep = h_outs_[e];
            py::dict d;
            int t = ep.num_our_turns - 1;
            d["turn_reached"] = t;
            d["root_step"] = ep.root_step;
            int nn = ep.num_nodes, ns = ep.num_slots;
            auto ints = [&](int field) {
                auto a = py::array_t<int>(nn);
                int* q = a.mutable_data();
                for (int k = 0; k < nn; k++) {
                    const TreeNodeV2& x = hn[(size_t)e * node_cap_ + k];
                    const int vals[7] = {x.parent, x.action, x.visits, x.slot, x.n_kids, x.step, x.kind};
                    q[k] = vals[field];
                }
                return a;
            };
            d["parent"] = ints(0); d["action"] = ints(1); d["visits"] = ints(2);
            d["slot"] = ints(3); d["n_kids"] = ints(4); d["step"] = ints(5); d["kind"] = ints(6);
            auto vs = py::array_t<float>(nn);
            for (int k = 0; k < nn; k++) vs.mutable_data()[k] = hn[(size_t)e * node_cap_ + k].value_sum;
            d["value_sum"] = vs;
            auto sp = py::array_t<float>({ns, NUM_HEROES});
            auto sc = py::array_t<int>({ns, NUM_HEROES});
            for (int k = 0; k < ns; k++) {
                const TreeSlotV2& x = hs[(size_t)e * slot_cap_ + k];
                std::memcpy(sp.mutable_data(k, 0), x.p, NUM_HEROES * sizeof(float));
                std::memcpy(sc.mutable_data(k, 0), x.child, NUM_HEROES * sizeof(int));
            }
            d["slot_p"] = sp;
            d["slot_child"] = sc;
            auto rs = py::array_t<float>(STATE_DIM);
            std::memcpy(rs.mutable_data(), ep.out_states[t], STATE_DIM * sizeof(float));
            d["root_state"] = rs;
            auto rp = py::array_t<float>(NUM_HEROES);
            std::memcpy(rp.mutable_data(), ep.out_policies[t], NUM_HEROES * sizeof(float));
            d["root_policy"] = rp;
            auto st = py::array_t<int>(NUM_STATS_V2);
            std::memcpy(st.mutable_data(), ep.stats, NUM_STATS_V2 * sizeof(int));
            d["stats"] = st;
            out.append(d);
        }
        return out;
    }

    void update_weights(py::array_t<float> new_weights) {
        auto w = new_weights.unchecked<1>();
        // ordered after this engine's in-flight batch, as the old default-stream copy was
        const float* src = w.data(0);
        size_t bytes = w.shape(0) * sizeof(float);
        py::gil_scoped_release no_gil;
        cudaMemcpyAsync(d_policy_weights_, src, bytes, cudaMemcpyHostToDevice, stream_);
        cudaStreamSynchronize(stream_);
    }

    ~MCTSKernelEngine() {
        if (stream_) { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); }
        cudaFree(d_policy_weights_);
        cudaFree(d_gd_weights_);
        cudaFree(d_wp_weights_);
        cudaFree(d_lut_);
        if (d_episodes_) cudaFree(d_episodes_);
        if (h_episodes_) cudaFreeHost(h_episodes_);
        cudaFree(d_configs_);
        cudaFree(d_outs_);
        cudaFreeHost(h_outs_);
        if (d_nodes_) cudaFree(d_nodes_);
        if (d_slots_) cudaFree(d_slots_);
    }

private:
    int max_episodes_;
    cudaStream_t stream_ = nullptr;
    int policy_weight_count_;
    PolicyNetOffsets policy_off_;
    GDNetOffsets gd_off_;
    WPNetOffsets wp_off_;
    float *d_policy_weights_, *d_gd_weights_, *d_wp_weights_;
    WPLookupTables *d_lut_;
    EpisodeMemory *d_episodes_, *h_episodes_;
    int *d_configs_;
    EpisodeOutV2 *d_outs_, *h_outs_;
    TreeNodeV2 *d_nodes_ = nullptr;
    TreeSlotV2 *d_slots_ = nullptr;
    int node_cap_ = 0, slot_cap_ = 0;
    int cap_override_nodes_ = 0, cap_override_slots_ = 0;
    int last_n_, last_mode_;
};


PYBIND11_MODULE(cuda_mcts_kernel, m) {
    m.doc() = "Full MCTS kernel: one block = one episode, zero launch overhead";

    py::class_<MCTSKernelEngine>(m, "MCTSKernelEngine")
        .def(py::init<py::array_t<float>, py::array_t<float>, py::array_t<float>,
                       py::dict, py::dict, py::dict,
                       py::array_t<uint8_t>, int, int>(),
             py::arg("policy_weights"), py::arg("gd_weights"), py::arg("wp_weights"),
             py::arg("policy_offsets"), py::arg("gd_offsets"), py::arg("wp_offsets"),
             py::arg("lut_blob"),
             py::arg("max_concurrent") = 128, py::arg("device_id") = 0)
        .def("run_episodes", &MCTSKernelEngine::run_episodes,
             py::arg("configs"), py::arg("num_sims"), py::arg("c_puct"),
             py::arg("seed"), py::arg("root_temp") = 1.0f,
             py::arg("dir_alpha") = 0.3f, py::arg("dir_eps") = 0.0f,
             py::arg("search_mode") = DEFAULT_SEARCH_MODE,
             py::arg("pw_k") = DEFAULT_PW_K, py::arg("pw_alpha") = DEFAULT_PW_ALPHA)
        .def("run_episodes_into_buffer", &MCTSKernelEngine::run_episodes_into_buffer,
             py::arg("configs"), py::arg("num_sims"), py::arg("c_puct"), py::arg("seed"),
             py::arg("buf_states"), py::arg("buf_policies"), py::arg("buf_masks"),
             py::arg("buf_values"), py::arg("write_offset"), py::arg("buffer_size"),
             py::arg("search_mode") = DEFAULT_SEARCH_MODE,
             py::arg("pw_k") = DEFAULT_PW_K, py::arg("pw_alpha") = DEFAULT_PW_ALPHA)
        .def("last_stats", &MCTSKernelEngine::last_stats)
        .def("search_info", &MCTSKernelEngine::search_info)
        .def("set_tree_capacity", &MCTSKernelEngine::set_tree_capacity,
             py::arg("node_cap"), py::arg("slot_cap"))
        .def("debug_tree", &MCTSKernelEngine::debug_tree,
             py::arg("configs"), py::arg("num_sims"), py::arg("c_puct"), py::arg("seed"),
             py::arg("turn"), py::arg("search_mode") = DEFAULT_SEARCH_MODE,
             py::arg("pw_k") = DEFAULT_PW_K, py::arg("pw_alpha") = DEFAULT_PW_ALPHA,
             py::arg("root_temp") = 1.0f, py::arg("dir_alpha") = 0.3f,
             py::arg("dir_eps") = 0.0f)
        .def("update_weights", &MCTSKernelEngine::update_weights);
    m.attr("DEFAULT_SEARCH_MODE") = DEFAULT_SEARCH_MODE;
    m.attr("SEARCH_LEGACY") = SEARCH_LEGACY;
    m.attr("SEARCH_CHANCE") = SEARCH_CHANCE;
    m.attr("SEARCH_ROLLFWD") = SEARCH_ROLLFWD;
#ifdef COMP_FALLBACK_MID_HIGH_LOW
    m.attr("COMP_FALLBACK_ORDER") = "mid_high_low";
#else
    m.attr("COMP_FALLBACK_ORDER") = "low_mid_high";
#endif
}
