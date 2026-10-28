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

#define MAX_RES_BLOCKS 8

struct PolicyNetOffsets {
    // Architecture dimensions
    int hdim;      // backbone hidden dim (768, 1536, 2048)
    int cdim;      // compress intermediate dim (512, 768, 1024)
    int edim;      // embedding output dim (256, 512)
    int n_blocks;  // number of residual blocks (3, 6, 8)

    int input_fc_w, input_fc_b;
    int input_bn_w, input_bn_b, input_bn_mean, input_bn_var;

    // Res blocks: up to MAX_RES_BLOCKS
    int res_fc1_w[MAX_RES_BLOCKS], res_fc1_b[MAX_RES_BLOCKS];
    int res_bn1_w[MAX_RES_BLOCKS], res_bn1_b[MAX_RES_BLOCKS];
    int res_bn1_mean[MAX_RES_BLOCKS], res_bn1_var[MAX_RES_BLOCKS];
    int res_fc2_w[MAX_RES_BLOCKS], res_fc2_b[MAX_RES_BLOCKS];
    int res_bn2_w[MAX_RES_BLOCKS], res_bn2_b[MAX_RES_BLOCKS];
    int res_bn2_mean[MAX_RES_BLOCKS], res_bn2_var[MAX_RES_BLOCKS];

    int compress1_w, compress1_b, compress1_bn_w, compress1_bn_b, compress1_bn_mean, compress1_bn_var;
    int compress2_w, compress2_b, compress2_bn_w, compress2_bn_b, compress2_bn_mean, compress2_bn_var;

    // Policy head: 0=linear(edim→90), 1=deep(edim→512→256→90),
    //              2=step(edim+16→256→128→90), 3=deep_step(edim+16→512→256→90)
    int policy_head_type;
    int policy_n_layers;              // number of linear layers in policy head
    int policy_layer_in[3];           // input dim per layer
    int policy_layer_out[3];          // output dim per layer
    int policy_layer_w[3];            // weight offset per layer
    int policy_layer_b[3];            // bias offset per layer
    int step_embed_w;                 // step embedding weight offset (16×16), -1 if none

    // Legacy single-layer (kept for backward compat, used when policy_head_type==0)
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

// Backbone: 289 → hdim → n_blocks×ResBlock → cdim → edim
// Output: edim-dim embedding in buf_e_out
// Workspace: needs hdim*3 + cdim floats
__device__ void d_policy_backbone(
    const float* __restrict__ state_289,
    const float* __restrict__ W,
    PolicyNetOffsets off,
    float* buf_e_out,     // edim floats output
    float* workspace      // hdim*3 + cdim floats
) {
    int hdim = off.hdim;
    int cdim = off.cdim;
    int edim = off.edim;
    float* buf_a = workspace;
    float* buf_b = workspace + hdim;
    float* buf_c = workspace + hdim * 2;
    float* buf_d = workspace + hdim * 3;

    d_linear_layer(state_289, 289, W + off.input_fc_w, W + off.input_fc_b, hdim, buf_a);
    d_batchnorm_relu(buf_a, hdim, W + off.input_bn_w, W + off.input_bn_b,
                     W + off.input_bn_mean, W + off.input_bn_var);

    for (int r = 0; r < off.n_blocks; r++) {
        d_residual_block(buf_a, buf_b, buf_c, hdim,
            W+off.res_fc1_w[r], W+off.res_fc1_b[r],
            W+off.res_bn1_w[r], W+off.res_bn1_b[r],
            W+off.res_bn1_mean[r], W+off.res_bn1_var[r],
            W+off.res_fc2_w[r], W+off.res_fc2_b[r],
            W+off.res_bn2_w[r], W+off.res_bn2_b[r],
            W+off.res_bn2_mean[r], W+off.res_bn2_var[r]);
    }

    d_linear_layer(buf_a, hdim, W+off.compress1_w, W+off.compress1_b, cdim, buf_d);
    d_batchnorm_relu(buf_d, cdim, W+off.compress1_bn_w, W+off.compress1_bn_b,
                     W+off.compress1_bn_mean, W+off.compress1_bn_var);
    d_linear_layer(buf_d, cdim, W+off.compress2_w, W+off.compress2_b, edim, buf_e_out);
    d_batchnorm_relu(buf_e_out, edim, W+off.compress2_bn_w, W+off.compress2_bn_b,
                     W+off.compress2_bn_mean, W+off.compress2_bn_var);
}


// ── Split forward: policy head (edim → 90 softmax) ─────────────────
// Supports: linear (type 0), deep MLP (type 1), step-conditioned (type 2,3)
// For step-conditioned: step_idx extracted from state_buf, embedding concatenated
// workspace: reuses backbone workspace (needs max 512 + 256 floats for intermediates)

__device__ void d_policy_head(
    const float* buf_e,        // edim backbone output
    const float* valid_mask,
    const float* W,
    PolicyNetOffsets off,
    float* priors_out,         // 90 floats output
    float* workspace,          // scratch for deep heads (512+ floats)
    int step_idx               // draft step (0-15), only used for type 2,3
) {
    int tid = threadIdx.x;

    if (off.policy_head_type == 0) {
        // Original: single linear layer
        d_linear_layer(buf_e, off.edim, W + off.policy_w, W + off.policy_b, 90, priors_out);
    } else {
        // Deep / step-conditioned heads
        // Build policy input: buf_e (+ step embedding for types 2,3)
        float* p_input = workspace;           // edim or edim+16
        float* p_buf_a = workspace + 544;     // max 512 output
        float* p_buf_b = workspace + 544+512; // max 256 output
        int p_input_dim = off.edim;

        // Copy backbone output to workspace
        d_copy_buf(buf_e, p_input, off.edim);

        if (off.policy_head_type >= 2 && off.step_embed_w >= 0) {
            // Append step embedding: W[step_embed_w + step_idx*16 ... +16]
            if (tid == 0) {
                int emb_off = off.step_embed_w + step_idx * 16;
                for (int i = 0; i < 16; i++)
                    p_input[off.edim + i] = W[emb_off + i];
            }
            __syncthreads();
            p_input_dim = off.edim + 16;
        }

        // Multi-layer forward: ping-pong through layers
        float* src = p_input;
        int src_dim = p_input_dim;
        float* dst = p_buf_a;
        for (int l = 0; l < off.policy_n_layers; l++) {
            int out_dim = off.policy_layer_out[l];
            float* out = (l == off.policy_n_layers - 1) ? priors_out : dst;
            d_linear_layer(src, src_dim, W + off.policy_layer_w[l], W + off.policy_layer_b[l],
                           out_dim, out);
            if (l < off.policy_n_layers - 1) {
                d_relu_inplace(out, out_dim);
            }
            src = out;
            src_dim = out_dim;
            dst = (dst == p_buf_a) ? p_buf_b : p_buf_a;
        }
    }

    // Mask invalid + softmax
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
// Workspace: (edim+1) + 128 + 64 floats

__device__ float d_value_head_symmetrized(
    const float* buf_e,    // edim backbone output (shared memory)
    float our_team,
    const float* W,
    PolicyNetOffsets off,
    float* workspace       // (edim+1) + 128 + 64 floats
) {
    int edim = off.edim;
    float* buf_vh = workspace;              // edim+1
    float* buf_b = workspace + edim + 1;    // 128
    float* buf_c = workspace + edim + 1 + 128; // 64
    int tid = threadIdx.x;

    // Pass 1: original perspective
    d_copy_buf(buf_e, buf_vh, edim);
    if (tid == 0) buf_vh[edim] = our_team;
    __syncthreads();
    d_linear_layer(buf_vh, edim + 1, W + off.value_fc1_w, W + off.value_fc1_b, 128, buf_b);
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
    if (tid == 0) buf_vh[0] = v1;
    __syncthreads();
    v1 = buf_vh[0];

    // Pass 2: flipped perspective
    d_copy_buf(buf_e, buf_vh, edim);
    if (tid == 0) buf_vh[edim] = 1.0f - our_team;
    __syncthreads();
    d_linear_layer(buf_vh, edim + 1, W + off.value_fc1_w, W + off.value_fc1_b, 128, buf_b);
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
