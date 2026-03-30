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
#define MAX_OUR_TURNS 8
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
};

struct WPNetOffsets {
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
    const int*, EpisodeMemory*, int, float, unsigned long long);

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

        // Allocate episode memory on GPU
        cudaMalloc(&d_episodes_, max_episodes_ * sizeof(EpisodeMemory));
        cudaMalloc(&d_configs_, max_episodes_ * 3 * sizeof(int));

        // Pinned host memory for results
        cudaMallocHost(&h_episodes_, max_episodes_ * sizeof(EpisodeMemory));
    }

    py::list run_episodes(py::array_t<int> configs, int num_sims, float c_puct,
                          unsigned long long seed) {
        auto cfg = configs.unchecked<2>();
        int n = cfg.shape(0);
        if (n > max_episodes_) throw std::runtime_error("Too many episodes");

        // Copy configs to GPU
        cudaMemcpy(d_configs_, cfg.data(0, 0), n * 3 * sizeof(int), cudaMemcpyHostToDevice);

        // Shared memory: dynamic based on policy net size
        int shared_mem = (STATE_DIM + NUM_HEROES + NUM_HEROES + policy_off_.edim
                         + policy_off_.hdim * 3 + policy_off_.cdim + ENRICHED_DIM) * sizeof(float);

        // Launch kernel: one block per episode
        void* args[] = {&d_policy_weights_, &d_gd_weights_, &d_wp_weights_,
                        &policy_off_, &gd_off_, &wp_off_, &d_lut_,
                        &d_configs_, &d_episodes_, &num_sims, &c_puct, &seed};
        cudaLaunchKernel((void*)mcts_episodes_kernel, dim3(n), dim3(256),
                         args, shared_mem, 0);
        cudaDeviceSynchronize();

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
        }

        // Copy results back
        cudaMemcpy(h_episodes_, d_episodes_, n * sizeof(EpisodeMemory), cudaMemcpyDeviceToHost);

        // Package results
        py::list results;
        for (int i = 0; i < n; i++) {
            EpisodeMemory& ep = h_episodes_[i];
            py::list examples;
            for (int t = 0; t < ep.num_our_turns; t++) {
                auto s = py::array_t<float>(STATE_DIM);
                auto p = py::array_t<float>(NUM_HEROES);
                auto m = py::array_t<float>(NUM_HEROES);
                std::memcpy(s.mutable_data(), ep.out_states[t], STATE_DIM * sizeof(float));
                std::memcpy(p.mutable_data(), ep.out_policies[t], NUM_HEROES * sizeof(float));
                std::memcpy(m.mutable_data(), ep.out_masks[t], NUM_HEROES * sizeof(float));
                examples.append(py::make_tuple(s, p, m));
            }
            auto ts = py::array_t<float>(STATE_DIM);
            std::memcpy(ts.mutable_data(), ep.terminal_state, STATE_DIM * sizeof(float));
            results.append(py::make_tuple(ep.win_prob, examples, ts, ep.our_team));
        }
        return results;
    }

    // Run episodes and write results directly into pre-allocated numpy ring buffer.
    // Returns number of training examples written.
    // Run episodes, write training data into ring buffer, output terminal states for WP eval.
    // Returns (n_written, terminal_states, our_teams)
    py::tuple run_episodes_into_buffer(
        py::array_t<int> configs, int num_sims, float c_puct, unsigned long long seed,
        py::array_t<float> buf_states,    // (BUFFER_SIZE, 290)
        py::array_t<float> buf_policies,  // (BUFFER_SIZE, 90)
        py::array_t<float> buf_masks,     // (BUFFER_SIZE, 90)
        py::array_t<float> buf_values,    // (BUFFER_SIZE,)
        int write_offset, int buffer_size
    ) {
        auto cfg = configs.unchecked<2>();
        int n = cfg.shape(0);
        if (n > max_episodes_) throw std::runtime_error("Too many episodes");

        // Launch kernel
        cudaMemcpy(d_configs_, cfg.data(0, 0), n * 3 * sizeof(int), cudaMemcpyHostToDevice);
        int shared_mem = (STATE_DIM + NUM_HEROES + NUM_HEROES + 256 + 768*3 + 512 + ENRICHED_DIM) * sizeof(float);
        void* args[] = {&d_policy_weights_, &d_gd_weights_, &d_wp_weights_,
                        &policy_off_, &gd_off_, &wp_off_, &d_lut_,
                        &d_configs_, &d_episodes_, &num_sims, &c_puct, &seed};
        cudaLaunchKernel((void*)mcts_episodes_kernel, dim3(n), dim3(256),
                         args, shared_mem, 0);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));

        // Copy results from GPU
        cudaMemcpy(h_episodes_, d_episodes_, n * sizeof(EpisodeMemory), cudaMemcpyDeviceToHost);

        // Write directly into pre-allocated numpy buffers (zero allocation)
        auto s_ptr = buf_states.mutable_unchecked<2>();
        auto p_ptr = buf_policies.mutable_unchecked<2>();
        auto m_ptr = buf_masks.mutable_unchecked<2>();
        auto v_ptr = buf_values.mutable_unchecked<1>();

        // WP values from kernel (for logging)
        auto wp_values = py::array_t<float>(n);
        auto wp_ptr = wp_values.mutable_unchecked<1>();

        int write_pos = write_offset;
        int total_written = 0;

        for (int ep = 0; ep < n; ep++) {
            EpisodeMemory& mem = h_episodes_[ep];
            float wp = mem.win_prob;  // kernel-computed symmetrized WP
            wp_ptr(ep) = wp;

            for (int t = 0; t < mem.num_our_turns; t++) {
                int idx = write_pos % buffer_size;
                std::memcpy(s_ptr.mutable_data(idx, 0), mem.out_states[t], STATE_DIM * sizeof(float));
                std::memcpy(p_ptr.mutable_data(idx, 0), mem.out_policies[t], NUM_HEROES * sizeof(float));
                std::memcpy(m_ptr.mutable_data(idx, 0), mem.out_masks[t], NUM_HEROES * sizeof(float));
                v_ptr(idx) = wp;  // WP from kernel's symmetrized enriched WP model
                write_pos++;
                total_written++;
            }
        }
        return py::make_tuple(total_written, wp_values);
    }

    void update_weights(py::array_t<float> new_weights) {
        auto w = new_weights.unchecked<1>();
        cudaMemcpy(d_policy_weights_, w.data(0), w.shape(0) * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~MCTSKernelEngine() {
        cudaFree(d_policy_weights_);
        cudaFree(d_gd_weights_);
        cudaFree(d_wp_weights_);
        cudaFree(d_lut_);
        cudaFree(d_episodes_);
        cudaFree(d_configs_);
        cudaFreeHost(h_episodes_);
    }

private:
    int max_episodes_;
    int policy_weight_count_;
    PolicyNetOffsets policy_off_;
    GDNetOffsets gd_off_;
    WPNetOffsets wp_off_;
    float *d_policy_weights_, *d_gd_weights_, *d_wp_weights_;
    WPLookupTables *d_lut_;
    EpisodeMemory *d_episodes_, *h_episodes_;
    int *d_configs_;
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
        .def("run_episodes", &MCTSKernelEngine::run_episodes)
        .def("run_episodes_into_buffer", &MCTSKernelEngine::run_episodes_into_buffer)
        .def("update_weights", &MCTSKernelEngine::update_weights);
}
