/**
 * Device helpers shared by the v2 search (search_v2.cuh) and the P3
 * personalization kernels' v2 fragment (personalization/cuda_{personal,prior,pgd}/p3_search_v2.inc).
 * Needs DraftStateGPU and tree_v2.h in scope.
 */
#pragma once
#include "tree_v2.h"

#define REQ_NONE 0
#define REQ_GD_CACHE 1   // chance node: compute and cache GD at scratch
#define REQ_GD_STEP 2    // open loop: compute GD at scratch, sample, apply
#define REQ_PUCT 3       // decision node: PUCT argmax over its children (warp 0)

// Thread 0. Sample from a probability vector; zero entries are never chosen.
// Falls back to the last positive entry when rounding leaves r above the
// accumulated mass (the legacy sampler fell back to hero 0, taken or not).
__device__ __forceinline__ int v2_sample(const float* p, float r) {
    float cum = 0.0f;
    int last = -1;
    for (int i = 0; i < NUM_HEROES; i++) {
        if (p[i] <= 0.0f) continue;
        last = i;
        cum += p[i];
        if (cum > r) return i;
    }
    return last;
}

// Thread 0. Sample from p restricted to existing children, renormalized.
__device__ __forceinline__ int v2_sample_existing(const TreeSlotV2& s, float r) {
    float tot = 0.0f;
    for (int i = 0; i < NUM_HEROES; i++)
        if (s.child[i] >= 0) tot += s.p[i];
    float x = r * tot, cum = 0.0f;
    int last = -1;
    for (int i = 0; i < NUM_HEROES; i++) {
        if (s.child[i] < 0) continue;
        last = i;
        cum += s.p[i];
        if (cum > x) return i;
    }
    return last;
}

// Thread 0. In-place masked softmax of GD logits (masked logits are -1e9 -> 0).
__device__ __forceinline__ void v2_softmax90(float* x) {
    float mx = -1e30f;
    for (int i = 0; i < NUM_HEROES; i++) mx = fmaxf(mx, x[i]);
    float sm = 0.0f;
    for (int i = 0; i < NUM_HEROES; i++) { x[i] = expf(x[i] - mx); sm += x[i]; }
    if (sm > 0.0f) for (int i = 0; i < NUM_HEROES; i++) x[i] /= sm;
}

__device__ __forceinline__ void v2_init_node(TreeNodeV2& n, int parent, int action, int step, int kind) {
    n.parent = parent; n.action = action; n.visits = 0; n.value_sum = 0.0f;
    n.slot = -1; n.n_kids = 0; n.step = step; n.kind = kind;
}

__device__ __forceinline__ int v2_kind_of(const DraftStateGPU& s, int mode) {
    if (s.is_terminal()) return KIND_TERMINAL;
    if (s.current_team() == s.our_team) return KIND_DECISION;
    return mode == SEARCH_CHANCE ? KIND_CHANCE : KIND_DECISION;
}

