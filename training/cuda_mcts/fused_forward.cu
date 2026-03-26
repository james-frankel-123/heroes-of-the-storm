/**
 * Fused CUDA kernels for AlphaZeroDraftNet and GenericDraftModel inference.
 * Single-sample forward pass in one kernel launch, no PyTorch dependency.
 *
 * Policy network: 289 → 768 (BN+ReLU) → 3×ResBlock(768) → 512 (BN+ReLU) → 256 (BN+ReLU)
 *   Policy head: 256 → 90 (softmax)
 *   Value head: 257 → 128 (ReLU) → 64 (ReLU) → 1 (tanh→[0,1])
 *
 * GD network: 289 → 256 (ReLU) → 128 (ReLU) → 90
 */
#include <cuda_runtime.h>
#include <cmath>

// ── Device helper functions ────────────────────────────────────────

// Linear layer: output[o] = bias[o] + dot(weight[o,:], input)
// Each thread handles multiple output elements
__device__ void linear_layer(
    const float* __restrict__ input, int in_dim,
    const float* __restrict__ weight,
    const float* __restrict__ bias, int out_dim,
    float* __restrict__ output
) {
    int tid = threadIdx.x;
    for (int o = tid; o < out_dim; o += blockDim.x) {
        float sum = bias[o];
        const float* w_row = weight + o * in_dim;
        for (int i = 0; i < in_dim; i++) {
            sum += w_row[i] * input[i];
        }
        output[o] = sum;
    }
    __syncthreads();
}

// BatchNorm (eval mode) + ReLU, in-place
__device__ void batchnorm_relu(
    float* __restrict__ data, int dim,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    const float* __restrict__ mean, const float* __restrict__ var
) {
    int tid = threadIdx.x;
    for (int i = tid; i < dim; i += blockDim.x) {
        float norm = (data[i] - mean[i]) * rsqrtf(var[i] + 1e-5f);
        float val = gamma[i] * norm + beta[i];
        data[i] = fmaxf(val, 0.0f);
    }
    __syncthreads();
}

// BatchNorm (eval mode) without ReLU, in-place
__device__ void batchnorm_only(
    float* __restrict__ data, int dim,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    const float* __restrict__ mean, const float* __restrict__ var
) {
    int tid = threadIdx.x;
    for (int i = tid; i < dim; i += blockDim.x) {
        float norm = (data[i] - mean[i]) * rsqrtf(var[i] + 1e-5f);
        data[i] = gamma[i] * norm + beta[i];
    }
    __syncthreads();
}

__device__ void relu_inplace(float* data, int dim) {
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        data[i] = fmaxf(data[i], 0.0f);
    __syncthreads();
}

__device__ void add_inplace(float* a, const float* b, int dim) {
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        a[i] += b[i];
    __syncthreads();
}

__device__ void copy_buf(const float* src, float* dst, int dim) {
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        dst[i] = src[i];
    __syncthreads();
}

// Residual block: residual=x; fc1→bn1→relu→fc2→bn2→add(residual)→relu
// Uses 3 buffers: buf_a (input/output), buf_b (residual scratch), buf_c (scratch)
__device__ void residual_block(
    float* buf_a, float* buf_b, float* buf_c, int dim,
    const float* fc1_w, const float* fc1_b,
    const float* bn1_g, const float* bn1_b, const float* bn1_m, const float* bn1_v,
    const float* fc2_w, const float* fc2_b,
    const float* bn2_g, const float* bn2_b, const float* bn2_m, const float* bn2_v
) {
    copy_buf(buf_a, buf_b, dim);                                  // buf_b = residual
    linear_layer(buf_a, dim, fc1_w, fc1_b, dim, buf_c);          // buf_c = fc1(x)
    batchnorm_relu(buf_c, dim, bn1_g, bn1_b, bn1_m, bn1_v);     // buf_c = relu(bn1(fc1(x)))
    linear_layer(buf_c, dim, fc2_w, fc2_b, dim, buf_a);          // buf_a = fc2(...)
    batchnorm_only(buf_a, dim, bn2_g, bn2_b, bn2_m, bn2_v);     // buf_a = bn2(fc2(...))
    add_inplace(buf_a, buf_b, dim);                               // buf_a += residual
    relu_inplace(buf_a, dim);                                     // buf_a = relu(...)
}


// ── Weight offset struct ───────────────────────────────────────────

struct PolicyNetOffsets {
    int input_fc_w, input_fc_b;
    int input_bn_w, input_bn_b, input_bn_mean, input_bn_var;
    // ResBlock 1
    int res1_fc1_w, res1_fc1_b, res1_bn1_w, res1_bn1_b, res1_bn1_mean, res1_bn1_var;
    int res1_fc2_w, res1_fc2_b, res1_bn2_w, res1_bn2_b, res1_bn2_mean, res1_bn2_var;
    // ResBlock 2
    int res2_fc1_w, res2_fc1_b, res2_bn1_w, res2_bn1_b, res2_bn1_mean, res2_bn1_var;
    int res2_fc2_w, res2_fc2_b, res2_bn2_w, res2_bn2_b, res2_bn2_mean, res2_bn2_var;
    // ResBlock 3
    int res3_fc1_w, res3_fc1_b, res3_bn1_w, res3_bn1_b, res3_bn1_mean, res3_bn1_var;
    int res3_fc2_w, res3_fc2_b, res3_bn2_w, res3_bn2_b, res3_bn2_mean, res3_bn2_var;
    // Compress
    int compress1_w, compress1_b, compress1_bn_w, compress1_bn_b, compress1_bn_mean, compress1_bn_var;
    int compress2_w, compress2_b, compress2_bn_w, compress2_bn_b, compress2_bn_mean, compress2_bn_var;
    // Heads
    int policy_w, policy_b;
    int value_fc1_w, value_fc1_b;
    int value_fc2_w, value_fc2_b;
    int value_out_w, value_out_b;
};

struct GDNetOffsets {
    int fc1_w, fc1_b;  // 289→256
    int fc2_w, fc2_b;  // 256→128
    int fc3_w, fc3_b;  // 128→90
};


// ── Main fused policy forward pass ─────────────────────────────────
// One block, 256 threads. Shared memory: 768*3 + 512 + 256 + 128 + 257 = 3441 floats ≈ 13.8KB

extern "C" __global__ void fused_policy_forward(
    const float* __restrict__ state,       // (290,)
    const float* __restrict__ valid_mask,  // (90,)
    const float* __restrict__ W,           // all weights flattened
    float* __restrict__ priors_out,        // (90,)
    float* __restrict__ value_out,         // (1,)
    PolicyNetOffsets off
) {
    extern __shared__ float smem[];
    float* buf_a = smem;                          // 768
    float* buf_b = smem + 768;                    // 768
    float* buf_c = smem + 768 * 2;                // 768
    float* buf_d = smem + 768 * 3;                // 512
    float* buf_e = smem + 768 * 3 + 512;          // 256
    float* buf_vh = smem + 768 * 3 + 512 + 256;   // 257 (value head input)

    int tid = threadIdx.x;

    // ── Backbone (input: first 289 of state) ──
    linear_layer(state, 289, W + off.input_fc_w, W + off.input_fc_b, 768, buf_a);
    batchnorm_relu(buf_a, 768, W + off.input_bn_w, W + off.input_bn_b,
                   W + off.input_bn_mean, W + off.input_bn_var);

    // ResBlock 1
    residual_block(buf_a, buf_b, buf_c, 768,
        W + off.res1_fc1_w, W + off.res1_fc1_b,
        W + off.res1_bn1_w, W + off.res1_bn1_b, W + off.res1_bn1_mean, W + off.res1_bn1_var,
        W + off.res1_fc2_w, W + off.res1_fc2_b,
        W + off.res1_bn2_w, W + off.res1_bn2_b, W + off.res1_bn2_mean, W + off.res1_bn2_var);

    // ResBlock 2
    residual_block(buf_a, buf_b, buf_c, 768,
        W + off.res2_fc1_w, W + off.res2_fc1_b,
        W + off.res2_bn1_w, W + off.res2_bn1_b, W + off.res2_bn1_mean, W + off.res2_bn1_var,
        W + off.res2_fc2_w, W + off.res2_fc2_b,
        W + off.res2_bn2_w, W + off.res2_bn2_b, W + off.res2_bn2_mean, W + off.res2_bn2_var);

    // ResBlock 3
    residual_block(buf_a, buf_b, buf_c, 768,
        W + off.res3_fc1_w, W + off.res3_fc1_b,
        W + off.res3_bn1_w, W + off.res3_bn1_b, W + off.res3_bn1_mean, W + off.res3_bn1_var,
        W + off.res3_fc2_w, W + off.res3_fc2_b,
        W + off.res3_bn2_w, W + off.res3_bn2_b, W + off.res3_bn2_mean, W + off.res3_bn2_var);

    // compress1: 768 → 512
    linear_layer(buf_a, 768, W + off.compress1_w, W + off.compress1_b, 512, buf_d);
    batchnorm_relu(buf_d, 512, W + off.compress1_bn_w, W + off.compress1_bn_b,
                   W + off.compress1_bn_mean, W + off.compress1_bn_var);

    // compress2: 512 → 256
    linear_layer(buf_d, 512, W + off.compress2_w, W + off.compress2_b, 256, buf_e);
    batchnorm_relu(buf_e, 256, W + off.compress2_bn_w, W + off.compress2_bn_b,
                   W + off.compress2_bn_mean, W + off.compress2_bn_var);

    // ── Policy head: 256 → 90 ──
    // Reuse buf_a for policy logits (only need 90 floats)
    linear_layer(buf_e, 256, W + off.policy_w, W + off.policy_b, 90, buf_a);

    // Apply mask
    for (int i = tid; i < 90; i += blockDim.x) {
        if (valid_mask[i] < 0.5f) buf_a[i] = -1e9f;
    }
    __syncthreads();

    // Softmax (single thread, 90 elements is tiny)
    if (tid == 0) {
        float max_val = -1e30f;
        for (int i = 0; i < 90; i++) max_val = fmaxf(max_val, buf_a[i]);
        float sum = 0.0f;
        for (int i = 0; i < 90; i++) {
            buf_a[i] = expf(buf_a[i] - max_val);
            sum += buf_a[i];
        }
        if (sum > 0.0f) {
            for (int i = 0; i < 90; i++) buf_a[i] /= sum;
        }
    }
    __syncthreads();

    // Write priors
    for (int i = tid; i < 90; i += blockDim.x) {
        priors_out[i] = buf_a[i];
    }

    // ── Value head: cat(buf_e[256], our_team) → 257 → 128 → 64 → 1 ──
    copy_buf(buf_e, buf_vh, 256);
    if (tid == 0) buf_vh[256] = state[289];  // our_team
    __syncthreads();

    // fc1: 257 → 128 + ReLU (reuse buf_b for 128 output)
    linear_layer(buf_vh, 257, W + off.value_fc1_w, W + off.value_fc1_b, 128, buf_b);
    relu_inplace(buf_b, 128);

    // fc2: 128 → 64 + ReLU (reuse buf_c)
    linear_layer(buf_b, 128, W + off.value_fc2_w, W + off.value_fc2_b, 64, buf_c);
    relu_inplace(buf_c, 64);

    // out: 64 → 1 (tanh, scale to [0,1])
    if (tid == 0) {
        float sum = W[off.value_out_b];
        for (int i = 0; i < 64; i++) sum += W[off.value_out_w + i] * buf_c[i];
        *value_out = tanhf(sum) * 0.5f + 0.5f;
    }
    __syncthreads();
}


// ── Batched policy forward (K blocks, one per sample) ──────────────

extern "C" __global__ void fused_policy_forward_batched(
    const float* __restrict__ states,       // (K, 290)
    const float* __restrict__ valid_masks,  // (K, 90)
    const float* __restrict__ W,
    float* __restrict__ priors_out,         // (K, 90)
    float* __restrict__ values_out,         // (K,)
    PolicyNetOffsets off,
    int K
) {
    int sample = blockIdx.x;
    if (sample >= K) return;

    const float* state = states + sample * 290;
    const float* valid_mask = valid_masks + sample * 90;
    float* my_priors = priors_out + sample * 90;
    float* my_value = values_out + sample;

    extern __shared__ float smem[];
    float* buf_a = smem;
    float* buf_b = smem + 768;
    float* buf_c = smem + 768 * 2;
    float* buf_d = smem + 768 * 3;
    float* buf_e = smem + 768 * 3 + 512;
    float* buf_vh = smem + 768 * 3 + 512 + 256;

    int tid = threadIdx.x;

    linear_layer(state, 289, W + off.input_fc_w, W + off.input_fc_b, 768, buf_a);
    batchnorm_relu(buf_a, 768, W + off.input_bn_w, W + off.input_bn_b,
                   W + off.input_bn_mean, W + off.input_bn_var);

    residual_block(buf_a, buf_b, buf_c, 768,
        W + off.res1_fc1_w, W + off.res1_fc1_b,
        W + off.res1_bn1_w, W + off.res1_bn1_b, W + off.res1_bn1_mean, W + off.res1_bn1_var,
        W + off.res1_fc2_w, W + off.res1_fc2_b,
        W + off.res1_bn2_w, W + off.res1_bn2_b, W + off.res1_bn2_mean, W + off.res1_bn2_var);
    residual_block(buf_a, buf_b, buf_c, 768,
        W + off.res2_fc1_w, W + off.res2_fc1_b,
        W + off.res2_bn1_w, W + off.res2_bn1_b, W + off.res2_bn1_mean, W + off.res2_bn1_var,
        W + off.res2_fc2_w, W + off.res2_fc2_b,
        W + off.res2_bn2_w, W + off.res2_bn2_b, W + off.res2_bn2_mean, W + off.res2_bn2_var);
    residual_block(buf_a, buf_b, buf_c, 768,
        W + off.res3_fc1_w, W + off.res3_fc1_b,
        W + off.res3_bn1_w, W + off.res3_bn1_b, W + off.res3_bn1_mean, W + off.res3_bn1_var,
        W + off.res3_fc2_w, W + off.res3_fc2_b,
        W + off.res3_bn2_w, W + off.res3_bn2_b, W + off.res3_bn2_mean, W + off.res3_bn2_var);

    linear_layer(buf_a, 768, W + off.compress1_w, W + off.compress1_b, 512, buf_d);
    batchnorm_relu(buf_d, 512, W + off.compress1_bn_w, W + off.compress1_bn_b,
                   W + off.compress1_bn_mean, W + off.compress1_bn_var);
    linear_layer(buf_d, 512, W + off.compress2_w, W + off.compress2_b, 256, buf_e);
    batchnorm_relu(buf_e, 256, W + off.compress2_bn_w, W + off.compress2_bn_b,
                   W + off.compress2_bn_mean, W + off.compress2_bn_var);

    linear_layer(buf_e, 256, W + off.policy_w, W + off.policy_b, 90, buf_a);
    for (int i = tid; i < 90; i += blockDim.x)
        if (valid_mask[i] < 0.5f) buf_a[i] = -1e9f;
    __syncthreads();

    if (tid == 0) {
        float max_val = -1e30f;
        for (int i = 0; i < 90; i++) max_val = fmaxf(max_val, buf_a[i]);
        float sum = 0.0f;
        for (int i = 0; i < 90; i++) { buf_a[i] = expf(buf_a[i] - max_val); sum += buf_a[i]; }
        if (sum > 0.0f) for (int i = 0; i < 90; i++) buf_a[i] /= sum;
    }
    __syncthreads();

    for (int i = tid; i < 90; i += blockDim.x) my_priors[i] = buf_a[i];

    copy_buf(buf_e, buf_vh, 256);
    if (tid == 0) buf_vh[256] = state[289];
    __syncthreads();

    linear_layer(buf_vh, 257, W + off.value_fc1_w, W + off.value_fc1_b, 128, buf_b);
    relu_inplace(buf_b, 128);
    linear_layer(buf_b, 128, W + off.value_fc2_w, W + off.value_fc2_b, 64, buf_c);
    relu_inplace(buf_c, 64);

    if (tid == 0) {
        float sum = W[off.value_out_b];
        for (int i = 0; i < 64; i++) sum += W[off.value_out_w + i] * buf_c[i];
        *my_value = tanhf(sum) * 0.5f + 0.5f;
    }
    __syncthreads();
}


// ── Fused GD forward pass ──────────────────────────────────────────
// 289 → 256 (ReLU) → 128 (ReLU) → 90

extern "C" __global__ void fused_gd_forward(
    const float* __restrict__ state,       // (289,)
    const float* __restrict__ valid_mask,  // (90,)
    const float* __restrict__ W,
    float* __restrict__ logits_out,        // (90,)
    GDNetOffsets off
) {
    extern __shared__ float smem[];
    float* buf_a = smem;        // 256
    float* buf_b = smem + 256;  // 128

    // fc1: 289 → 256 + ReLU
    linear_layer(state, 289, W + off.fc1_w, W + off.fc1_b, 256, buf_a);
    relu_inplace(buf_a, 256);

    // fc2: 256 → 128 + ReLU
    linear_layer(buf_a, 256, W + off.fc2_w, W + off.fc2_b, 128, buf_b);
    relu_inplace(buf_b, 128);

    // fc3: 128 → 90
    linear_layer(buf_b, 128, W + off.fc3_w, W + off.fc3_b, 90, buf_a);

    // Apply mask and write output
    int tid = threadIdx.x;
    for (int i = tid; i < 90; i += blockDim.x) {
        logits_out[i] = (valid_mask[i] > 0.5f) ? buf_a[i] : -1e9f;
    }
    __syncthreads();
}
