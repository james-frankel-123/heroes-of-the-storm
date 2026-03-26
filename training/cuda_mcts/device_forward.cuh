/**
 * Device-callable forward pass functions for use inside the MCTS kernel.
 * Converted from the __global__ versions in fused_forward.cu.
 * The math is identical -- just __device__ instead of __global__.
 *
 * Split into: backbone (expensive, run once) + policy head + value head
 * This allows value symmetrization without re-running the backbone.
 */
#pragma once
#include <cuda_runtime.h>

// ── Primitive ops (same as fused_forward.cu) ───────────────────────

__device__ void d_linear_layer(
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

__device__ void d_batchnorm_relu(
    float* __restrict__ data, int dim,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    const float* __restrict__ mean, const float* __restrict__ var
) {
    int tid = threadIdx.x;
    for (int i = tid; i < dim; i += blockDim.x) {
        float norm = (data[i] - mean[i]) * rsqrtf(var[i] + 1e-5f);
        data[i] = fmaxf(gamma[i] * norm + beta[i], 0.0f);
    }
    __syncthreads();
}

__device__ void d_batchnorm_only(
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

__device__ void d_relu_inplace(float* data, int dim) {
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        data[i] = fmaxf(data[i], 0.0f);
    __syncthreads();
}

__device__ void d_add_inplace(float* a, const float* b, int dim) {
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        a[i] += b[i];
    __syncthreads();
}

__device__ void d_copy_buf(const float* src, float* dst, int dim) {
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        dst[i] = src[i];
    __syncthreads();
}

// Residual block: residual=x; fc1→bn1→relu→fc2→bn2→add(residual)→relu
__device__ void d_residual_block(
    float* buf_a, float* buf_b, float* buf_c, int dim,
    const float* fc1_w, const float* fc1_b,
    const float* bn1_g, const float* bn1_b, const float* bn1_m, const float* bn1_v,
    const float* fc2_w, const float* fc2_b,
    const float* bn2_g, const float* bn2_b, const float* bn2_m, const float* bn2_v
) {
    d_copy_buf(buf_a, buf_b, dim);
    d_linear_layer(buf_a, dim, fc1_w, fc1_b, dim, buf_c);
    d_batchnorm_relu(buf_c, dim, bn1_g, bn1_b, bn1_m, bn1_v);
    d_linear_layer(buf_c, dim, fc2_w, fc2_b, dim, buf_a);
    d_batchnorm_only(buf_a, dim, bn2_g, bn2_b, bn2_m, bn2_v);
    d_add_inplace(buf_a, buf_b, dim);
    d_relu_inplace(buf_a, dim);
}


// ── Offset structs ─────────────────────────────────────────────────

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


// ── Split forward: backbone (expensive) ────────────────────────────

// Backbone: 289 → 768 → 3×ResBlock → 512 → 256
// Output: 256-dim embedding in buf_e_out
// Workspace: needs 768*3 + 512 floats
__device__ void d_policy_backbone(
    const float* __restrict__ state_289,
    const float* __restrict__ W,
    PolicyNetOffsets off,
    float* buf_e_out,     // 256 floats output
    float* workspace      // 768*3 + 512 = 2816 floats
) {
    float* buf_a = workspace;
    float* buf_b = workspace + 768;
    float* buf_c = workspace + 768 * 2;
    float* buf_d = workspace + 768 * 3;

    d_linear_layer(state_289, 289, W + off.input_fc_w, W + off.input_fc_b, 768, buf_a);
    d_batchnorm_relu(buf_a, 768, W + off.input_bn_w, W + off.input_bn_b,
                     W + off.input_bn_mean, W + off.input_bn_var);

    d_residual_block(buf_a, buf_b, buf_c, 768,
        W+off.res1_fc1_w, W+off.res1_fc1_b, W+off.res1_bn1_w, W+off.res1_bn1_b,
        W+off.res1_bn1_mean, W+off.res1_bn1_var, W+off.res1_fc2_w, W+off.res1_fc2_b,
        W+off.res1_bn2_w, W+off.res1_bn2_b, W+off.res1_bn2_mean, W+off.res1_bn2_var);
    d_residual_block(buf_a, buf_b, buf_c, 768,
        W+off.res2_fc1_w, W+off.res2_fc1_b, W+off.res2_bn1_w, W+off.res2_bn1_b,
        W+off.res2_bn1_mean, W+off.res2_bn1_var, W+off.res2_fc2_w, W+off.res2_fc2_b,
        W+off.res2_bn2_w, W+off.res2_bn2_b, W+off.res2_bn2_mean, W+off.res2_bn2_var);
    d_residual_block(buf_a, buf_b, buf_c, 768,
        W+off.res3_fc1_w, W+off.res3_fc1_b, W+off.res3_bn1_w, W+off.res3_bn1_b,
        W+off.res3_bn1_mean, W+off.res3_bn1_var, W+off.res3_fc2_w, W+off.res3_fc2_b,
        W+off.res3_bn2_w, W+off.res3_bn2_b, W+off.res3_bn2_mean, W+off.res3_bn2_var);

    d_linear_layer(buf_a, 768, W+off.compress1_w, W+off.compress1_b, 512, buf_d);
    d_batchnorm_relu(buf_d, 512, W+off.compress1_bn_w, W+off.compress1_bn_b,
                     W+off.compress1_bn_mean, W+off.compress1_bn_var);
    d_linear_layer(buf_d, 512, W+off.compress2_w, W+off.compress2_b, 256, buf_e_out);
    d_batchnorm_relu(buf_e_out, 256, W+off.compress2_bn_w, W+off.compress2_bn_b,
                     W+off.compress2_bn_mean, W+off.compress2_bn_var);
}


// ── Split forward: policy head (256 → 90 softmax) ─────────────────

__device__ void d_policy_head(
    const float* buf_e,
    const float* valid_mask,
    const float* W,
    PolicyNetOffsets off,
    float* priors_out     // 90 floats
) {
    d_linear_layer(buf_e, 256, W + off.policy_w, W + off.policy_b, 90, priors_out);
    int tid = threadIdx.x;
    for (int i = tid; i < 90; i += blockDim.x)
        if (valid_mask[i] < 0.5f) priors_out[i] = -1e9f;
    __syncthreads();

    if (tid == 0) {
        float mx = -1e30f;
        for (int i = 0; i < 90; i++) mx = fmaxf(mx, priors_out[i]);
        float sm = 0.0f;
        for (int i = 0; i < 90; i++) { priors_out[i] = expf(priors_out[i] - mx); sm += priors_out[i]; }
        if (sm > 0.0f) for (int i = 0; i < 90; i++) priors_out[i] /= sm;
    }
    __syncthreads();
}


// ── Split forward: value head with symmetrization ──────────────────
// Runs value head twice (original + flipped our_team), averages.
// Only the value head is run twice, not the backbone (since backbone
// doesn't see our_team). This halves the symmetrization cost.
// Workspace: 257 + 128 + 64 = 449 floats

__device__ float d_value_head_symmetrized(
    const float* buf_e,    // 256 backbone output (shared memory)
    float our_team,
    const float* W,
    PolicyNetOffsets off,
    float* workspace       // 449 floats
) {
    float* buf_vh = workspace;        // 257
    float* buf_b = workspace + 257;   // 128
    float* buf_c = workspace + 257 + 128; // 64
    int tid = threadIdx.x;

    // Pass 1: original perspective
    d_copy_buf(buf_e, buf_vh, 256);
    if (tid == 0) buf_vh[256] = our_team;
    __syncthreads();
    d_linear_layer(buf_vh, 257, W + off.value_fc1_w, W + off.value_fc1_b, 128, buf_b);
    d_relu_inplace(buf_b, 128);
    d_linear_layer(buf_b, 128, W + off.value_fc2_w, W + off.value_fc2_b, 64, buf_c);
    d_relu_inplace(buf_c, 64);
    float v1 = 0.0f;
    if (tid == 0) {
        float s = W[off.value_out_b];
        for (int i = 0; i < 64; i++) s += W[off.value_out_w + i] * buf_c[i];
        v1 = tanhf(s) * 0.5f + 0.5f;
    }
    __syncthreads();
    // Broadcast v1
    if (tid == 0) buf_vh[0] = v1;
    __syncthreads();
    v1 = buf_vh[0];

    // Pass 2: flipped perspective
    d_copy_buf(buf_e, buf_vh, 256);
    if (tid == 0) buf_vh[256] = 1.0f - our_team;
    __syncthreads();
    d_linear_layer(buf_vh, 257, W + off.value_fc1_w, W + off.value_fc1_b, 128, buf_b);
    d_relu_inplace(buf_b, 128);
    d_linear_layer(buf_b, 128, W + off.value_fc2_w, W + off.value_fc2_b, 64, buf_c);
    d_relu_inplace(buf_c, 64);
    float v2 = 0.0f;
    if (tid == 0) {
        float s = W[off.value_out_b];
        for (int i = 0; i < 64; i++) s += W[off.value_out_w + i] * buf_c[i];
        v2 = tanhf(s) * 0.5f + 0.5f;
    }
    __syncthreads();

    return (v1 + (1.0f - v2)) / 2.0f;
}


// ── GD forward (device version) ────────────────────────────────────
// Workspace: 256 + 128 = 384 floats

__device__ void d_gd_forward(
    const float* __restrict__ state_289,
    const float* __restrict__ valid_mask,
    const float* __restrict__ W,
    float* __restrict__ logits_out,
    GDNetOffsets off,
    float* workspace
) {
    float* buf_a = workspace;
    float* buf_b = workspace + 256;

    d_linear_layer(state_289, 289, W + off.fc1_w, W + off.fc1_b, 256, buf_a);
    d_relu_inplace(buf_a, 256);
    d_linear_layer(buf_a, 256, W + off.fc2_w, W + off.fc2_b, 128, buf_b);
    d_relu_inplace(buf_b, 128);
    d_linear_layer(buf_b, 128, W + off.fc3_w, W + off.fc3_b, 90, logits_out);

    int tid = threadIdx.x;
    for (int i = tid; i < 90; i += blockDim.x)
        if (valid_mask[i] < 0.5f) logits_out[i] = -1e9f;
    __syncthreads();
}
