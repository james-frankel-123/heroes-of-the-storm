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

namespace py = pybind11;

#define NUM_HEROES 90
#define STATE_DIM 290
#define MAX_OUR_TURNS 16
#define CFG_LEN 58
#define NUM_SLOTS 10
#include "tree_v2.h"   // X2 fix: v2 search arena and stats layout
#define MAX_NODES 4096
#define MAX_CHILD_INDICES 81920

// ── WP model structs (must match enriched_features.cuh) ──

#define NUM_FINE_ROLES 9
#define NUM_BLIZZ_ROLES 6
#define MAX_COMP_ENTRIES 512
#define ENRICHED_DIM 86
#define WP_BASE_DIM 197
#define WP_FULL_DIM 283

#ifndef NUM_MAPS
#define NUM_MAPS 14
#endif
#ifndef NUM_TIERS
#define NUM_TIERS 3
#endif

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
    int n_out;
    float lcb_lambda;
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
    int max_nodes_used;
    int n_capacity_hits;
    float out_q[MAX_OUR_TURNS][NUM_HEROES];
    int out_step[MAX_OUR_TURNS];
    int out_team[MAX_OUR_TURNS];
    int terminal_actions[16];
    float wp_t0;
    float v_t0;
    int decided;
    int v2_stats[NUM_STATS_V2];
};

struct PersonalArgs {
    const int* cfg;
    const float* pers_s;
    const float* pers_off;
    const float* imit_bias;
    const unsigned int* pool;
    float b0, b1, b2, b3, imit_w0;
    const float* pgd_bias;
    const float* gd_b1;
    float pgd_alpha_pick, pgd_alpha_ban;
};

// Kernel declaration

extern "C" void mcts_episodes_kernel(
    const float*, const float*, const float*,
    PolicyNetOffsets, GDNetOffsets, WPNetOffsets,
    const WPLookupTables*,
    const int*, EpisodeMemory*, int, float, unsigned long long,
    float, float, float, int, PersonalArgs, V2Args);
extern "C" void leaf_eval_kernel(const float*, WPNetOffsets, const WPLookupTables*,
                                 PersonalArgs, float*);
extern "C" void pgd_eval_kernel(const float*, GDNetOffsets, PersonalArgs, float*);

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
    o.n_out = d.contains("n_out") ? d["n_out"].cast<int>() : 1;
    o.lcb_lambda = d.contains("lcb_lambda") ? d["lcb_lambda"].cast<float>() : 0.0f;
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



static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

class PersonalEngine {
public:
    PersonalEngine(py::array_t<float> policy_weights, py::array_t<float> gd_weights,
                   py::array_t<float> wp_weights, py::dict policy_offsets_dict,
                   py::dict gd_offsets_dict, py::dict wp_offsets_dict,
                   py::array_t<uint8_t> lut_blob, int max_concurrent = 128, int device_id = 0)
        : max_(max_concurrent), policy_off_(dict_to_policy_offsets(policy_offsets_dict)),
          gd_off_(dict_to_gd_offsets(gd_offsets_dict)), wp_off_(dict_to_wp_offsets(wp_offsets_dict)) {
        check(cudaSetDevice(device_id), "set device");
        auto pw = policy_weights.unchecked<1>();
        auto gw = gd_weights.unchecked<1>();
        auto ww = wp_weights.unchecked<1>();
        check(cudaMalloc(&d_pw_, pw.shape(0) * sizeof(float)), "malloc pw");
        cudaMemcpy(d_pw_, pw.data(0), pw.shape(0) * sizeof(float), cudaMemcpyHostToDevice);
        check(cudaMalloc(&d_gw_, gw.shape(0) * sizeof(float)), "malloc gw");
        cudaMemcpy(d_gw_, gw.data(0), gw.shape(0) * sizeof(float), cudaMemcpyHostToDevice);
        check(cudaMalloc(&d_ww_, ww.shape(0) * sizeof(float)), "malloc ww");
        cudaMemcpy(d_ww_, ww.data(0), ww.shape(0) * sizeof(float), cudaMemcpyHostToDevice);
        auto lb = lut_blob.unchecked<1>();
        if (lb.shape(0) != (ssize_t)sizeof(WPLookupTables))
            throw std::runtime_error("LUT blob size mismatch");
        check(cudaMalloc(&d_lut_, sizeof(WPLookupTables)), "malloc lut");
        cudaMemcpy(d_lut_, lb.data(0), sizeof(WPLookupTables), cudaMemcpyHostToDevice);
        check(cudaMalloc(&d_ep_, (size_t)max_ * sizeof(EpisodeMemory)), "malloc episodes");
        check(cudaMalloc(&d_cfg_, (size_t)max_ * CFG_LEN * sizeof(int)), "malloc cfg");
        check(cudaMalloc(&d_s_, (size_t)max_ * NUM_SLOTS * NUM_HEROES * sizeof(float)), "malloc s");
        check(cudaMalloc(&d_off_, (size_t)max_ * NUM_SLOTS * NUM_HEROES * sizeof(float)), "malloc off");
        check(cudaMalloc(&d_imit_, (size_t)max_ * NUM_SLOTS * NUM_HEROES * sizeof(float)), "malloc imit");
        check(cudaMalloc(&d_pool_, (size_t)max_ * NUM_SLOTS * 3 * sizeof(unsigned int)), "malloc pool");
        check(cudaMalloc(&d_out_, (size_t)max_ * 90 * sizeof(float)), "malloc out");
        check(cudaMalloc(&d_pgd_, (size_t)max_ * 14 * NUM_HEROES * sizeof(float)), "malloc pgd");
        check(cudaMalloc(&d_b1_, (size_t)max_ * 256 * sizeof(float)), "malloc b1");
        tail_off_ = offsetof(EpisodeMemory, num_nodes);
        tail_len_ = sizeof(EpisodeMemory) - tail_off_;
        h_tail_ = (char*)malloc((size_t)max_ * tail_len_);
    }
    ~PersonalEngine() {
        cudaFree(d_pw_); cudaFree(d_gw_); cudaFree(d_ww_); cudaFree(d_lut_); cudaFree(d_ep_);
        cudaFree(d_cfg_); cudaFree(d_s_); cudaFree(d_off_); cudaFree(d_imit_); cudaFree(d_pool_);
        cudaFree(d_out_); cudaFree(d_pgd_); cudaFree(d_b1_); free(h_tail_);
        if (d_nodes_) cudaFree(d_nodes_);
        if (d_slots_) cudaFree(d_slots_);
    }

    PersonalArgs upload(py::array_t<int> cfg, py::array_t<float> s, py::array_t<float> off,
                        py::array_t<float> imit, py::array_t<unsigned int> pool,
                        py::array_t<float> coefs, py::array_t<float> pgd, py::array_t<float> b1, int& n) {
        auto c = cfg.unchecked<2>();
        n = c.shape(0);
        if (n > max_) throw std::runtime_error("too many episodes");
        if (c.shape(1) != CFG_LEN) throw std::runtime_error("cfg width");
        cudaMemcpy(d_cfg_, c.data(0, 0), (size_t)n * CFG_LEN * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_s_, s.data(), (size_t)n * NUM_SLOTS * NUM_HEROES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_off_, off.data(), (size_t)n * NUM_SLOTS * NUM_HEROES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_imit_, imit.data(), (size_t)n * NUM_SLOTS * NUM_HEROES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pool_, pool.data(), (size_t)n * NUM_SLOTS * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
        auto k = coefs.unchecked<1>();
        PersonalArgs pa;
        pa.cfg = d_cfg_; pa.pers_s = d_s_; pa.pers_off = d_off_; pa.imit_bias = d_imit_; pa.pool = d_pool_;
        pa.b0 = k(0); pa.b1 = k(1); pa.b2 = k(2); pa.b3 = k(3); pa.imit_w0 = k(4);
        cudaMemcpy(d_pgd_, pgd.data(), (size_t)n * 14 * NUM_HEROES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1_, b1.data(), (size_t)n * 256 * sizeof(float), cudaMemcpyHostToDevice);
        pa.pgd_bias = d_pgd_; pa.gd_b1 = d_b1_;
        pa.pgd_alpha_pick = k(5); pa.pgd_alpha_ban = k(6);
        return pa;
    }

    py::tuple run(py::array_t<int> cfg, py::array_t<float> s, py::array_t<float> off,
                  py::array_t<float> imit, py::array_t<unsigned int> pool, py::array_t<float> coefs,
                  py::array_t<float> pgd, py::array_t<float> b1, int num_sims, float c_puct, unsigned long long seed, float root_temp,
                  float dir_alpha, float dir_eps, int guard,
                  int search_mode = SEARCH_CHANCE, float pw_k = 1.0f, float pw_alpha = 0.5f) {
        int n;
        PersonalArgs pa = upload(cfg, s, off, imit, pool, coefs, pgd, b1, n);
        int shared_mem = (291 + NUM_HEROES + NUM_HEROES + policy_off_.edim
                          + policy_off_.hdim * 3 + policy_off_.cdim + ENRICHED_DIM) * sizeof(float);
        const int* cptr = d_cfg_;
        if (search_mode < 0 || search_mode > 2) throw std::runtime_error("search_mode must be 0, 1 or 2");
        V2Args v2;
        v2.mode = search_mode; v2.pw_k = pw_k; v2.pw_alpha = pw_alpha;
        v2.nodes = nullptr; v2.slots = nullptr; v2.node_cap = 0; v2.slot_cap = 0;
        if (search_mode != SEARCH_LEGACY) {
            ensure_arena(num_sims);
            v2.nodes = d_nodes_; v2.slots = d_slots_; v2.node_cap = node_cap_; v2.slot_cap = slot_cap_;
        }
        void* args[] = {&d_pw_, &d_gw_, &d_ww_, &policy_off_, &gd_off_, &wp_off_, &d_lut_,
                        &cptr, &d_ep_, &num_sims, &c_puct, &seed, &root_temp, &dir_alpha,
                        &dir_eps, &guard, &pa, &v2};
        check(cudaLaunchKernel((void*)mcts_episodes_kernel, dim3(n), dim3(256), args, shared_mem, 0), "launch");
        check(cudaDeviceSynchronize(), "sync");
        check(cudaGetLastError(), "kernel");
        check(cudaMemcpy2D(h_tail_, tail_len_, (char*)d_ep_ + tail_off_, sizeof(EpisodeMemory),
                           tail_len_, n, cudaMemcpyDeviceToHost), "copy tail");
        py::array_t<float> win({n}), wpt0({n}), vt0({n});
        py::array_t<int> turns({n}), maxn({n}), caps({n}), dec({n}), ours({n});
        py::array_t<int> term({n, 16}), ostep({n, MAX_OUR_TURNS}), oteam({n, MAX_OUR_TURNS});
        py::array_t<float> opol({n, MAX_OUR_TURNS, NUM_HEROES}), oq({n, MAX_OUR_TURNS, NUM_HEROES});
        last_stats_.assign((size_t)n * NUM_STATS_V2, 0);
        last_n_ = n;
        for (int i = 0; i < n; i++) {
            EpisodeMemory* e = (EpisodeMemory*)(h_tail_ + (size_t)i * tail_len_ - tail_off_);
            win.mutable_data()[i] = e->win_prob;
            wpt0.mutable_data()[i] = e->wp_t0;
            vt0.mutable_data()[i] = e->v_t0;
            turns.mutable_data()[i] = e->num_our_turns;
            maxn.mutable_data()[i] = e->max_nodes_used;
            std::memcpy(last_stats_.data() + (size_t)i * NUM_STATS_V2, e->v2_stats, NUM_STATS_V2 * sizeof(int));
            caps.mutable_data()[i] = e->n_capacity_hits;
            dec.mutable_data()[i] = e->decided;
            ours.mutable_data()[i] = e->our_team;
            std::memcpy(term.mutable_data() + i * 16, e->terminal_actions, 16 * sizeof(int));
            std::memcpy(ostep.mutable_data() + i * MAX_OUR_TURNS, e->out_step, MAX_OUR_TURNS * sizeof(int));
            std::memcpy(oteam.mutable_data() + i * MAX_OUR_TURNS, e->out_team, MAX_OUR_TURNS * sizeof(int));
            std::memcpy(opol.mutable_data() + (size_t)i * MAX_OUR_TURNS * NUM_HEROES, e->out_policies,
                        MAX_OUR_TURNS * NUM_HEROES * sizeof(float));
            std::memcpy(oq.mutable_data() + (size_t)i * MAX_OUR_TURNS * NUM_HEROES, e->out_q,
                        MAX_OUR_TURNS * NUM_HEROES * sizeof(float));
        }
        return py::make_tuple(win, wpt0, vt0, term, turns, opol, oq, ostep, oteam, maxn, caps, dec, ours);
    }

    py::array_t<float> leaf_eval(py::array_t<int> cfg, py::array_t<float> s, py::array_t<float> off,
                                 py::array_t<float> imit, py::array_t<unsigned int> pool,
                                 py::array_t<float> coefs, py::array_t<float> pgd, py::array_t<float> b1) {
        int n;
        PersonalArgs pa = upload(cfg, s, off, imit, pool, coefs, pgd, b1, n);
        int shared_mem = (291 + 96 + 4096) * sizeof(float);
        void* args[] = {&d_ww_, &wp_off_, &d_lut_, &pa, &d_out_};
        check(cudaLaunchKernel((void*)leaf_eval_kernel, dim3(n), dim3(256), args, shared_mem, 0), "launch leaf");
        check(cudaDeviceSynchronize(), "sync leaf");
        py::array_t<float> out({n, 2});
        cudaMemcpy(out.mutable_data(), d_out_, (size_t)n * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        return out;
    }

    py::array_t<float> pgd_eval(py::array_t<int> cfg, py::array_t<float> s, py::array_t<float> off,
                                py::array_t<float> imit, py::array_t<unsigned int> pool,
                                py::array_t<float> coefs, py::array_t<float> pgd, py::array_t<float> b1) {
        int n;
        PersonalArgs pa = upload(cfg, s, off, imit, pool, coefs, pgd, b1, n);
        void* args[] = {&d_gw_, &gd_off_, &pa, &d_out_};
        check(cudaLaunchKernel((void*)pgd_eval_kernel, dim3(n), dim3(256), args, 0, 0), "launch pgd");
        check(cudaDeviceSynchronize(), "sync pgd");
        py::array_t<float> out({n, 90});
        cudaMemcpy(out.mutable_data(), d_out_, (size_t)n * 90 * sizeof(float), cudaMemcpyDeviceToHost);
        return out;
    }

private:
    int max_;
    PolicyNetOffsets policy_off_;
    GDNetOffsets gd_off_;
    WPNetOffsets wp_off_;
    float *d_pw_, *d_gw_, *d_ww_, *d_s_, *d_off_, *d_imit_, *d_out_, *d_pgd_, *d_b1_;
    WPLookupTables* d_lut_;
    EpisodeMemory* d_ep_;
    // X2 fix: v2 tree arena (3 * sims + 8 nodes and slots per episode, never overflows)
    TreeNodeV2* d_nodes_ = nullptr;
    TreeSlotV2* d_slots_ = nullptr;
    int node_cap_ = 0, slot_cap_ = 0, last_n_ = 0;
    std::vector<int> last_stats_;
    void ensure_arena(int num_sims) {
        int need = 3 * std::max(num_sims, 1) + 8;
        if (d_nodes_ != nullptr && need <= node_cap_) return;
        if (d_nodes_) cudaFree(d_nodes_);
        if (d_slots_) cudaFree(d_slots_);
        node_cap_ = slot_cap_ = need;
        check(cudaMalloc(&d_nodes_, (size_t)max_ * node_cap_ * sizeof(TreeNodeV2)), "malloc v2 nodes");
        check(cudaMalloc(&d_slots_, (size_t)max_ * slot_cap_ * sizeof(TreeSlotV2)), "malloc v2 slots");
    }
public:
    // v2 search diagnostics of the last run, (n, NUM_STATS_V2) (tree_v2.h ST_*)
    py::array_t<int> last_stats() const {
        py::array_t<int> out({last_n_, NUM_STATS_V2});
        if (last_n_) std::memcpy(out.mutable_data(), last_stats_.data(), (size_t)last_n_ * NUM_STATS_V2 * sizeof(int));
        return out;
    }
private:
    int* d_cfg_;
    unsigned int* d_pool_;
    size_t tail_off_, tail_len_;
    char* h_tail_;
};

PYBIND11_MODULE(pgd_kernel, m) {
    m.doc() = "P3 personalized MCTS kernel with a personalized GD opponent model";
    py::class_<PersonalEngine>(m, "PgdEngine", py::module_local())
        .def(py::init<py::array_t<float>, py::array_t<float>, py::array_t<float>,
                      py::dict, py::dict, py::dict, py::array_t<uint8_t>, int, int>(),
             py::arg("policy_weights"), py::arg("gd_weights"), py::arg("wp_weights"),
             py::arg("policy_offsets"), py::arg("gd_offsets"), py::arg("wp_offsets"),
             py::arg("lut_blob"), py::arg("max_concurrent") = 128, py::arg("device_id") = 0)
        .def("run", &PersonalEngine::run, py::arg("cfg"), py::arg("s"), py::arg("off"), py::arg("imit"), py::arg("pool"), py::arg("coefs"), py::arg("pgd"), py::arg("b1"),
             py::arg("num_sims"), py::arg("c_puct"), py::arg("seed"), py::arg("root_temp"),
             py::arg("dir_alpha"), py::arg("dir_eps"), py::arg("guard"),
             py::arg("search_mode") = SEARCH_CHANCE, py::arg("pw_k") = 1.0f, py::arg("pw_alpha") = 0.5f)
        .def("last_stats", &PersonalEngine::last_stats)
        .def("leaf_eval", &PersonalEngine::leaf_eval)
        .def("pgd_eval", &PersonalEngine::pgd_eval);
    m.attr("CFG_LEN") = CFG_LEN;
    m.attr("MAX_OUR_TURNS") = MAX_OUR_TURNS;
}
