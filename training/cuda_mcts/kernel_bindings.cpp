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
#define MAX_NODES 2048
#define MAX_CHILD_INDICES 40960

// Must match device_forward.cuh
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
    const float*, const float*, PolicyNetOffsets, GDNetOffsets,
    const int*, EpisodeMemory*, int, float, unsigned long long);

PolicyNetOffsets dict_to_policy_offsets(py::dict d);
GDNetOffsets dict_to_gd_offsets(py::dict d);

// Forward declarations of dict converters (reuse from mcts_engine.cpp)
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
        fc1_w = d[(p+".fc1.weight").c_str()].cast<int>();
        fc1_b = d[(p+".fc1.bias").c_str()].cast<int>();
        bn1_w = d[(p+".bn1.weight").c_str()].cast<int>();
        bn1_b = d[(p+".bn1.bias").c_str()].cast<int>();
        bn1_m = d[(p+".bn1.running_mean").c_str()].cast<int>();
        bn1_v = d[(p+".bn1.running_var").c_str()].cast<int>();
        fc2_w = d[(p+".fc2.weight").c_str()].cast<int>();
        fc2_b = d[(p+".fc2.bias").c_str()].cast<int>();
        bn2_w = d[(p+".bn2.weight").c_str()].cast<int>();
        bn2_b = d[(p+".bn2.bias").c_str()].cast<int>();
        bn2_m = d[(p+".bn2.running_mean").c_str()].cast<int>();
        bn2_v = d[(p+".bn2.running_var").c_str()].cast<int>();
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


class MCTSKernelEngine {
public:
    MCTSKernelEngine(
        py::array_t<float> policy_weights,
        py::array_t<float> gd_weights,
        py::dict policy_offsets_dict,
        py::dict gd_offsets_dict,
        int max_concurrent_episodes = 128,
        int device_id = 0
    ) : max_episodes_(max_concurrent_episodes),
        policy_off_(dict_to_policy_offsets(policy_offsets_dict)),
        gd_off_(dict_to_gd_offsets(gd_offsets_dict))
    {
        cudaSetDevice(device_id);

        auto pw = policy_weights.unchecked<1>();
        auto gw = gd_weights.unchecked<1>();
        policy_weight_count_ = pw.shape(0);

        cudaMalloc(&d_policy_weights_, pw.shape(0) * sizeof(float));
        cudaMemcpy(d_policy_weights_, pw.data(0), pw.shape(0) * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_gd_weights_, gw.shape(0) * sizeof(float));
        cudaMemcpy(d_gd_weights_, gw.data(0), gw.shape(0) * sizeof(float), cudaMemcpyHostToDevice);

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

        // Shared memory: 3542 floats = 14.2 KB
        int shared_mem = (STATE_DIM + NUM_HEROES + NUM_HEROES + 256 + 768*3 + 512) * sizeof(float);

        // Launch kernel: one block per episode
        void* args[] = {&d_policy_weights_, &d_gd_weights_, &policy_off_, &gd_off_,
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
            results.append(py::make_tuple(ep.win_prob, examples));
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
        int shared_mem = (STATE_DIM + NUM_HEROES + NUM_HEROES + 256 + 768*3 + 512) * sizeof(float);
        void* args[] = {&d_policy_weights_, &d_gd_weights_, &policy_off_, &gd_off_,
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

        // Build terminal state output arrays
        auto term_states = py::array_t<float>({n, STATE_DIM});
        auto term_teams = py::array_t<int>(n);
        auto ts_ptr = term_states.mutable_unchecked<2>();
        auto tt_ptr = term_teams.mutable_unchecked<1>();

        // Track which buffer indices belong to which episode (for WP writeback)
        auto ep_start_idx = py::array_t<int>(n);   // buffer index of first example per episode
        auto ep_num_turns = py::array_t<int>(n);    // num turns per episode
        auto esi_ptr = ep_start_idx.mutable_unchecked<1>();
        auto ent_ptr = ep_num_turns.mutable_unchecked<1>();

        int write_pos = write_offset;
        int total_written = 0;

        for (int ep = 0; ep < n; ep++) {
            EpisodeMemory& mem = h_episodes_[ep];
            // Copy terminal state for WP eval
            std::memcpy(ts_ptr.mutable_data(ep, 0), mem.terminal_state, STATE_DIM * sizeof(float));
            tt_ptr(ep) = mem.our_team;
            esi_ptr(ep) = write_pos % buffer_size;
            ent_ptr(ep) = mem.num_our_turns;

            for (int t = 0; t < mem.num_our_turns; t++) {
                int idx = write_pos % buffer_size;
                std::memcpy(s_ptr.mutable_data(idx, 0), mem.out_states[t], STATE_DIM * sizeof(float));
                std::memcpy(p_ptr.mutable_data(idx, 0), mem.out_policies[t], NUM_HEROES * sizeof(float));
                std::memcpy(m_ptr.mutable_data(idx, 0), mem.out_masks[t], NUM_HEROES * sizeof(float));
                v_ptr(idx) = 0.5f;  // placeholder, host will overwrite with WP eval
                write_pos++;
                total_written++;
            }
        }
        return py::make_tuple(total_written, term_states, term_teams, ep_start_idx, ep_num_turns);
    }

    void update_weights(py::array_t<float> new_weights) {
        auto w = new_weights.unchecked<1>();
        cudaMemcpy(d_policy_weights_, w.data(0), w.shape(0) * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~MCTSKernelEngine() {
        cudaFree(d_policy_weights_);
        cudaFree(d_gd_weights_);
        cudaFree(d_episodes_);
        cudaFree(d_configs_);
        cudaFreeHost(h_episodes_);
    }

private:
    int max_episodes_;
    int policy_weight_count_;
    PolicyNetOffsets policy_off_;
    GDNetOffsets gd_off_;
    float *d_policy_weights_, *d_gd_weights_;
    EpisodeMemory *d_episodes_, *h_episodes_;
    int *d_configs_;
};


PYBIND11_MODULE(cuda_mcts_kernel, m) {
    m.doc() = "Full MCTS kernel: one block = one episode, zero launch overhead";

    py::class_<MCTSKernelEngine>(m, "MCTSKernelEngine")
        .def(py::init<py::array_t<float>, py::array_t<float>,
                       py::dict, py::dict, int, int>(),
             py::arg("policy_weights"), py::arg("gd_weights"),
             py::arg("policy_offsets"), py::arg("gd_offsets"),
             py::arg("max_concurrent") = 128, py::arg("device_id") = 0)
        .def("run_episodes", &MCTSKernelEngine::run_episodes)
        .def("run_episodes_into_buffer", &MCTSKernelEngine::run_episodes_into_buffer)
        .def("update_weights", &MCTSKernelEngine::update_weights);
}
