/**
 * P3 explore variant (copy of cuda_pvnet/pvnet_kernel.cu). Two gated additions:
 *  - cfg[56] = 2 (BAN_BIAS_MODE): the per-slot prior term also applies at our
 *    BAN steps, using row ecfg[24 + step] when it is >= 0 (the host puts the
 *    acting team's ban-bias vector in a dedicated row). cfg[56] = 1 is the
 *    pvnet kernel's picks-only rule unchanged.
 *  - cfg[6] = 2 (OPP_PERSONAL_MODE): the opponent's distribution at every
 *    opponent step (picks and bans) becomes softmax over valid heroes of
 *    (imit_w0 log p_GD + imit[ecfg[24 + step]]) (plain GD when the row is
 *    -1), everywhere the opponent acts: main draft loop, search chance
 *    nodes (GD cache), open-loop GD steps (rollfwd) and rollouts. Our-side
 *    rollout steps keep plain GD. See opp_personalize. cfg[6] in {0, 1} is
 *    the pvnet kernel unchanged.
 *  - cfg[6] = 3 (2026-10-20): as 2 at chance nodes, open-loop steps and the main
 *    loop, but rollouts keep plain GD for the opponent (P3_OPP_ROLL).
 *  - party mode (2026-10-22): cfg[40 + step] = -2 on our steps = a stranger teammate's step,
 *    searched as a chance node (P3_STRANGER); see p3_search_v2.inc. v2 search only.
 *    A flag on the ROOT step (2026-10-23, the lock-or-wait arm) makes the root itself a chance
 *    node: out_policies = visit shares of the stranger's sampled heroes, root value = sum(pol * q).
 * Original header (cuda_pvnet):
 * P3 learned personal value net variant (copy of cuda_prior/prior_kernel.cu,
 * option (c): a value net with per-lobby player features). Per episode a
 * third (10, 90) tensor feat (host: log1p of the slot player's earlier games
 * on each hero) and one shared weight vector vnet (PVNET_LEN = 225 floats).
 * With cfg[57] = 1 (assign mode required, personal valuation cfg[7] = 1) the
 * leaf is V = sigmoid(b0s + b1 logit p_our + h(F_our) - h(F_opp)), where for
 * each team t the assignment is assign mode's best permutation (max
 * b2 S + b3 O) and
 *   F_t = [S_t, O_t, the 5 assigned s values sorted descending,
 *          the 5 assigned feat values sorted descending]   (12 dims),
 *   h(x) = w2 . relu(W1 x + c1) + c2, hidden 16;
 *   vnet = W1 (16 x 12 row-major), c1 (16), w2 (16), c2 (1).
 * A slot without a pick (incomplete draft; leaves are always complete)
 * contributes 0 to both sorted lists. With cfg[57] = 0 the kernel is the
 * prior kernel unchanged (and with cfg[56] = 0 the personalized kernel).
 * Assign mode (cfg[58] = 1; any other value = slot mode, the code path
 * below it unchanged): at a pick step the TEAM picks, candidates are the
 * heroes in the union of the acting team's 5 slot pools; the personal
 * terms S_t, O_t of each team come from the assignment of its picks to its
 * 5 slots maximizing b2 S_t + b3 O_t (120 permutations, first maximum in
 * lexicographic order, identity = slot mode's pick order first). Leaves are
 * always complete drafts (rollout to terminal), so no partial-draft rule is
 * needed. Per-slot PUCT / imitation / personalized-GD terms still use the
 * acting slot step_slot[step] (priors and opponent model only, not value).
 * P3 distilled-prior variant (copy of cuda_personal/personal_kernel.cu). Adds
 * a per-slot additive prior term: at pick steps with cfg[56] = 1 the PUCT
 * priors of the acting slot become softmax(alpha log p_policy + bias[slot]),
 * over the same valid heroes; root priors are reported in out_prior. With
 * cfg[56] = 0 the kernel is the personalized kernel unchanged.
 * P3 personalized MCTS kernel (copy of overfit2026/cuda_ofit/mcts_kernel.cu,
 * itself a guarded copy of training/cuda_mcts). Changes, all per episode
 * through PersonalArgs (one launch serves many lobbies):
 *  - leaf value: population WP (symmetrized, our side) plus personal terms
 *    on the logit scale: V = sigmoid(b0 + b1 logit WP + b2 (S_our - S_opp)
 *    + b3 (O_our - O_opp)); S = sum of per-slot per-hero skill terms, O =
 *    off-role count; slots follow the pick-step -> player map step_slot.
 *    With personal=0, or with a zero adjustment and b = (0, 1, *, *), the
 *    population value is returned unchanged (bit-identical to the ofit
 *    kernel).
 *  - pool masks: at pick steps the acting slot may only pick heroes in its
 *    pool (if the pool has no free hero, any free hero).
 *  - forced bans: a ban step with a given, still-free hero is applied
 *    deterministically everywhere (main draft, tree, rollouts).
 *  - prefix + decide-only mode: replay given actions up to decide_step,
 *    search once for the acting team there, report root visits / Q, stop.
 *  - self-play: every non-forced step is searched from the acting team's
 *    side.
 *  - imitation opponent (main draft only): opponent picks sampled from
 *    softmax(w0 log p_GD + bias[slot]).
 *  - tree-capacity guard (from cuda_ofit) kept; MAX_OUR_TURNS raised to 16.
 * Original header:
 * Full MCTS episode kernel: one thread block = one complete draft episode.
 * 256 threads cooperate on matrix multiplications. Thread 0 drives all
 * sequential logic (tree traversal, UCB selection, action sampling).
 * Zero kernel launch overhead per forward pass.
 */
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include "device_forward.cuh"
#include "enriched_features.cuh"

// P3 safe inverse-CDF draw (2026-10-24, plan review C24): entries p[i] / denom <= 0 (or NaN) are never
// chosen; when rounding leaves r above the accumulated mass the LAST positive entry is returned (the
// legacy loops fell back to hero 0, taken or not); no positive entry -> -1, and the caller applies
// P3_SAFE_FALLBACK (first hero not yet taken). Whenever the legacy loop found an entry the result is
// identical (zero entries add nothing to the cumulative sum).
__device__ __forceinline__ int p3_sample_cdf(const float* p, float r, float denom) {
    float cum = 0.0f;
    int last = -1;
    for (int i = 0; i < NUM_HEROES; i++) {
        float q = denom == 1.0f ? p[i] : p[i] / denom;
        if (!(q > 0.0f)) continue;
        last = i;
        cum += q;
        if (cum > r) return i;
    }
    return last;
}
#define P3_SAFE_FALLBACK(X, ST) do { if ((X) < 0) { (X) = 0; while ((X) < NUM_HEROES - 1 && (ST).is_taken(X)) (X)++; } } while (0)

#define NUM_HEROES 90
#define NUM_MAPS 14
#define NUM_TIERS 3
#define STATE_DIM 290
#define GD_STATE_DIM 289
#define MAX_NODES 4096
#define MAX_CHILD_INDICES 81920
#define MAX_OUR_TURNS 16
#define CFG_LEN 59
#define CFG_ASSIGN 58
#define NUM_SLOTS 10
#define DRAFT_STEPS 16
#define MAX_PATH_DEPTH 64
#include "tree_v2.h"   // X2 fix: v2 search arena and stats layout

__constant__ int c_draft_team[16] = {0,1,0,1, 0,1,1,0,0, 1,0, 1,1,0,0,1};
__constant__ int c_draft_is_pick[16] = {0,0,0,0, 1,1,1,1,1, 0,0, 1,1,1,1,1};

// ── GPU-side data structures ───────────────────────────────────────

struct DraftStateGPU {
    uint32_t t0_picks[3];
    uint32_t t1_picks[3];
    uint32_t bans[3];
    uint32_t taken[3];
    int step;
    int map_idx;
    int tier_idx;
    int our_team;
    short step_action[16];

    __device__ void init(int m, int t, int ot) {
        memset(this, 0, sizeof(DraftStateGPU));
        map_idx = m; tier_idx = t; our_team = ot;
    }

    __device__ void copy_from(const DraftStateGPU& o) {
        memcpy(this, &o, sizeof(DraftStateGPU));
    }

    __device__ void apply_action(int hero_idx, int team, int is_pick) {
        int w = hero_idx / 32;
        uint32_t bit = 1u << (hero_idx % 32);
        taken[w] |= bit;
        if (step < 16) step_action[step] = (short)hero_idx;
        if (!is_pick) bans[w] |= bit;
        else if (team == 0) t0_picks[w] |= bit;
        else t1_picks[w] |= bit;
        step++;
    }

    __device__ bool is_taken(int hero_idx) const {
        return (taken[hero_idx / 32] >> (hero_idx % 32)) & 1;
    }

    __device__ bool is_terminal() const { return step >= 16; }
    __device__ int current_team() const { return c_draft_team[step]; }
    __device__ int current_is_pick() const { return c_draft_is_pick[step]; }

    __device__ void to_float_array(float* out) const {
        int tid = threadIdx.x;
        for (int i = tid; i < STATE_DIM; i += blockDim.x) out[i] = 0.0f;
        __syncthreads();
        if (tid == 0) {
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
                out[STATE_DIM - 2] = c_draft_is_pick[step] ? 1.0f : 0.0f;
            } else {
                out[STATE_DIM - 3] = 1.0f;
                out[STATE_DIM - 2] = 1.0f;
            }
            out[STATE_DIM - 1] = (float)our_team;
        }
        __syncthreads();
    }

    __device__ void valid_mask(float* out) const {
        int tid = threadIdx.x;
        for (int i = tid; i < NUM_HEROES; i += blockDim.x)
            out[i] = is_taken(i) ? 0.0f : 1.0f;
        __syncthreads();
    }
};

struct MCTSNodeGPU {
    int parent_idx;
    int action;
    float prior;
    int visit_count;
    float value_sum;
    int children_start;
    int num_children;
    int is_expanded;
    int has_cached_opp;
    float cached_opp_probs[NUM_HEROES];

    __device__ float q_value() const {
        return visit_count == 0 ? 0.0f : value_sum / (float)visit_count;
    }
};

struct EpisodeMemory {
    MCTSNodeGPU nodes[MAX_NODES];
    int child_indices[MAX_CHILD_INDICES];
    int num_nodes;
    int num_children_allocated;
    float out_states[MAX_OUR_TURNS][STATE_DIM];
    float out_policies[MAX_OUR_TURNS][NUM_HEROES];
    float out_masks[MAX_OUR_TURNS][NUM_HEROES];
    int num_our_turns;
    float win_prob;
    float terminal_state[STATE_DIM];  // for WP model evaluation on host
    int our_team;                      // needed for WP perspective
    int max_nodes_used;                // overfit2026 diagnostics
    int n_capacity_hits;               // expansions that would overflow the tree arrays
    // P3 additions
    float out_q[MAX_OUR_TURNS][NUM_HEROES];   // root child Q (our side), 0 if unvisited
    int out_step[MAX_OUR_TURNS];
    int out_team[MAX_OUR_TURNS];
    int terminal_actions[16];
    float wp_t0;                        // population WP, team 0 side
    float v_t0;                         // personal value, team 0 side
    int decided;                        // decide-only mode finished
    float out_prior[MAX_OUR_TURNS][NUM_HEROES];  // root priors actually used
    int v2_stats[NUM_STATS_V2];         // X2 fix: v2 search diagnostics (tree_v2.h ST_*)
};

struct PersonalArgs {
    const int* cfg;            // (n, CFG_LEN): see layout below
    const float* pers_s;       // (n, 10, 90) per-slot per-hero skill term
    const float* pers_off;     // (n, 10, 90) per-slot per-hero off-role indicator
    const float* imit_bias;    // (n, 10, 90) imitation personal part
    const unsigned int* pool;  // (n, 10, 3) pool bitsets
    float b0, b1, b2, b3, imit_w0;
    const float* prior_bias;   // (n, 10, 90) distilled per-slot prior term
    float prior_alpha;
    const float* pers_feat;    // (n, 10, 90) pvnet: log1p(earlier games of the slot's player)
    const float* vnet;         // (PVNET_LEN) pvnet weights, shared by all episodes
};

#define PVNET_IN 12
#define PVNET_HID 16
#define PVNET_LEN (PVNET_HID * PVNET_IN + PVNET_HID + PVNET_HID + 1)

// Thread 0: priors <- softmax(alpha log priors + bias[slot]) over valid heroes.
// cfg[56] = 2 also applies it at ban steps (row ecfg[24 + step] when >= 0).
__device__ void apply_prior_bias(float* pri, const float* mask, int step, const int* ecfg,
                                 const float* eb, float alpha) {
    if (threadIdx.x == 0 && ecfg[56] && step < 16 && (c_draft_is_pick[step] || ecfg[56] == 2)) {
        int slot = ecfg[24 + step];
        if (slot >= 0) {
            float mx = -1e30f;
            for (int i = 0; i < NUM_HEROES; i++) {
                if (mask[i] > 0.5f) {
                    // underflow fix (2026-10-19): a valid hero whose base prior underflowed to 0
                    // can still be revived by the bias (floor 1e-30 before the log)
                    pri[i] = alpha * logf(fmaxf(pri[i], 1e-30f)) + eb[slot * NUM_HEROES + i];
                    mx = fmaxf(mx, pri[i]);
                } else {
                    pri[i] = -1e30f;
                }
            }
            float sm = 0.0f;
            for (int i = 0; i < NUM_HEROES; i++) {
                pri[i] = pri[i] > -1e29f ? expf(pri[i] - mx) : 0.0f;
                sm += pri[i];
            }
            if (sm > 0.0f) for (int i = 0; i < NUM_HEROES; i++) pri[i] /= sm;
        }
    }
    __syncthreads();
}

// Thread 0 only (no barrier). cfg[6] = 2: player-aware opponent. p holds the
// GD probabilities at state step `step` (team c_draft_team[step]); when that
// team is not `our` and row = ecfg[24 + step] >= 0, p <- softmax over heroes
// with p > 0 and a valid mask of (w0 log p + imit[row]). Picks and bans.
// With w0 = 1 and an all-zero row this is the identity, so p is left
// untouched (exact, avoids rounding from the log/exp round trip).
__device__ __forceinline__ void opp_personalize(float* p, const float* mask, int step, int our,
                                                const int* ecfg, const float* eimit, float w0) {
    if ((ecfg[6] != 2 && ecfg[6] != 3) || step < 0 || step >= 16) return;
    if (c_draft_team[step] == our && ecfg[40 + step] != -2) return;   // our steps: only stranger steps (party mode)
    int row = ecfg[24 + step];
    if (row < 0 || row >= NUM_SLOTS) return;
    const float* b = eimit + row * NUM_HEROES;
    if (w0 == 1.0f) {
        bool zero = true;
        for (int i = 0; i < NUM_HEROES; i++) if (b[i] != 0.0f) { zero = false; break; }
        if (zero) return;
    }
    float mu = -1e30f;
    for (int i = 0; i < NUM_HEROES; i++) {
        float u = (p[i] > 0.0f && mask[i] > 0.5f) ? w0 * logf(p[i]) + b[i] : -1e30f;
        p[i] = u;
        mu = fmaxf(mu, u);
    }
    float su = 0.0f;
    for (int i = 0; i < NUM_HEROES; i++) {
        p[i] = p[i] > -1e29f ? expf(p[i] - mu) : 0.0f;
        su += p[i];
    }
    if (su > 0.0f) for (int i = 0; i < NUM_HEROES; i++) p[i] /= su;
}
// cfg[6] = 2: player-aware opponent everywhere (opp_personalize); 1 = main loop picks only
// cfg[56] = 2: prior term at our picks and bans (apply_prior_bias)
// cfg[56]: use the distilled prior term (0/1)
// cfg layout: [0] map [1] tier [2] our_team [3] selfplay [4] decide_step (-1 = full draft)
// [5] use_pool [6] opp_model (0 GD, 1 imitation) [7] personal valuation (0/1)
// [8..23] prefix actions per step [24..39] step_slot per step (-1 = ban)
// [40..55] forced ban hero per step (-1 = none)
// [56] prior kernel: distilled prior term; pgd kernel: personalized GD; else unused
// [57] pgd kernel: personalized-GD PUCT priors; pvnet kernel: learned personal value net (0/1)
// [58] CFG_ASSIGN: 1 = team pick + assignment ("assign mode"), else slot mode.
//      All three P3 kernels share this 59-wide layout.

__device__ int forced_action(const DraftStateGPU& s, const int* ecfg) {
    int st = s.step;
    if (st >= 16 || c_draft_is_pick[st]) return -1;
    int h = ecfg[40 + st];
    if (h < 0 || s.is_taken(h)) return -1;
    return h;
}

__device__ void ep_valid_mask(const DraftStateGPU& s, float* out, const int* ecfg,
                              const unsigned int* epool) {
    s.valid_mask(out);
    if (threadIdx.x == 0) {
        int st = s.step;
        if (st < 16 && ecfg[5] && c_draft_is_pick[st] && ecfg[CFG_ASSIGN] == 1) {
            // assign mode: the team picks; pool = union of the acting team's 5 slot pools
            int base = 5 * c_draft_team[st];
            uint32_t u[3] = {0u, 0u, 0u};
            for (int k = 0; k < 5; k++)
                for (int w = 0; w < 3; w++) u[w] |= epool[(base + k) * 3 + w];
            int any = 0;
            for (int i = 0; i < NUM_HEROES; i++)
                if (out[i] > 0.5f && ((u[i / 32] >> (i % 32)) & 1u)) any = 1;
            if (any)
                for (int i = 0; i < NUM_HEROES; i++)
                    if (out[i] > 0.5f && !((u[i / 32] >> (i % 32)) & 1u))
                        out[i] = 0.0f;
        } else if (st < 16 && ecfg[5] && c_draft_is_pick[st]) {
            int slot = ecfg[24 + st];
            if (slot >= 0) {
                int any = 0;
                for (int i = 0; i < NUM_HEROES; i++)
                    if (out[i] > 0.5f && ((epool[slot * 3 + i / 32] >> (i % 32)) & 1u)) any = 1;
                if (any)
                    for (int i = 0; i < NUM_HEROES; i++)
                        if (out[i] > 0.5f && !((epool[slot * 3 + i / 32] >> (i % 32)) & 1u))
                            out[i] = 0.0f;
            }
        }
    }
    __syncthreads();
}

// Thread 0. Lexicographic next permutation of 5 ints (std::next_permutation).
__device__ bool next_perm5(int* a) {
    int i = 3;
    while (i >= 0 && a[i] >= a[i + 1]) i--;
    if (i < 0) return false;
    int j = 4;
    while (a[j] <= a[i]) j--;
    int t = a[i]; a[i] = a[j]; a[j] = t;
    for (int l = i + 1, r = 4; l < r; l++, r--) { t = a[l]; a[l] = a[r]; a[r] = t; }
    return true;
}

// Thread 0, assign mode. Team tm's picks h[0..n) in pick-step order; slot
// 5 tm + i plays pick perm[i] (none if perm[i] >= n). Over the 120
// permutations in lexicographic order (identity first), keep the first one
// maximizing b2 S + b3 O (same rule as p3_dr_drafter / p3_assign); return
// its S and O.
__device__ void best_assignment(int tm, const DraftStateGPU& s, const float* es, const float* eo,
                                const PersonalArgs& pa, float* S_out, float* O_out) {
    int h[5];
    int n = 0;
    for (int st = 0; st < 16 && st < s.step; st++)
        if (c_draft_is_pick[st] && c_draft_team[st] == tm && n < 5) h[n++] = s.step_action[st];
    int base = 5 * tm;
    int p[5] = {0, 1, 2, 3, 4};
    float best = -1e30f, bS = 0.0f, bO = 0.0f;
    do {
        float sS = 0.0f, sO = 0.0f;
        for (int i = 0; i < 5; i++) {
            if (p[i] >= n) continue;
            sS += es[(base + i) * NUM_HEROES + h[p[i]]];
            sO += eo[(base + i) * NUM_HEROES + h[p[i]]];
        }
        float obj = pa.b2 * sS + pa.b3 * sO;
        if (obj > best) {
            best = obj; bS = sS; bO = sO;
        }
    } while (next_perm5(p));
    *S_out = bS;
    *O_out = bO;
}

// Thread 0, pvnet. Same assignment rule as best_assignment (kept separate so
// the cfg[57] = 0 path is untouched); fills the 12 features F_t.
__device__ void pvnet_features(int tm, const DraftStateGPU& s, const float* es, const float* eo,
                               const float* ef, const PersonalArgs& pa, float* F) {
    int h[5];
    int n = 0;
    for (int st = 0; st < 16 && st < s.step; st++)
        if (c_draft_is_pick[st] && c_draft_team[st] == tm && n < 5) h[n++] = s.step_action[st];
    int base = 5 * tm;
    int p[5] = {0, 1, 2, 3, 4};
    int bp[5] = {0, 1, 2, 3, 4};
    float best = -1e30f, bS = 0.0f, bO = 0.0f;
    do {
        float sS = 0.0f, sO = 0.0f;
        for (int i = 0; i < 5; i++) {
            if (p[i] >= n) continue;
            sS += es[(base + i) * NUM_HEROES + h[p[i]]];
            sO += eo[(base + i) * NUM_HEROES + h[p[i]]];
        }
        float obj = pa.b2 * sS + pa.b3 * sO;
        if (obj > best) {
            best = obj; bS = sS; bO = sO;
            for (int i = 0; i < 5; i++) bp[i] = p[i];
        }
    } while (next_perm5(p));
    float sv[5], fv[5];
    for (int i = 0; i < 5; i++) {
        int j = bp[i];
        sv[i] = j < n ? es[(base + i) * NUM_HEROES + h[j]] : 0.0f;
        fv[i] = j < n ? ef[(base + i) * NUM_HEROES + h[j]] : 0.0f;
    }
    // insertion sort, descending
    for (int i = 1; i < 5; i++)
        for (int k = i; k > 0 && sv[k] > sv[k - 1]; k--) { float x = sv[k]; sv[k] = sv[k - 1]; sv[k - 1] = x; }
    for (int i = 1; i < 5; i++)
        for (int k = i; k > 0 && fv[k] > fv[k - 1]; k--) { float x = fv[k]; fv[k] = fv[k - 1]; fv[k - 1] = x; }
    F[0] = bS;
    F[1] = bO;
    for (int i = 0; i < 5; i++) { F[2 + i] = sv[i]; F[7 + i] = fv[i]; }
}

// Thread 0, pvnet. h(x) = w2 . relu(W1 x + c1) + c2.
__device__ float pvnet_h(const float* F, const float* w) {
    const float* W1 = w;
    const float* c1 = w + PVNET_HID * PVNET_IN;
    const float* w2 = c1 + PVNET_HID;
    float out = w2[PVNET_HID];   // c2
    for (int j = 0; j < PVNET_HID; j++) {
        float a = c1[j];
        for (int i = 0; i < PVNET_IN; i++) a += W1[j * PVNET_IN + i] * F[i];
        out += w2[j] * fmaxf(a, 0.0f);
    }
    return out;
}

// Thread 0 only. p_our: population WP from our_team's side. ef: the
// episode's feat rows (read only with cfg[57] = 1).
__device__ float personal_value(float p_our, const DraftStateGPU& s, int our,
                                const int* ecfg, const float* es, const float* eo,
                                const float* ef, const PersonalArgs& pa) {
    if (!ecfg[7]) return p_our;
    if (ecfg[57] == 1) {
        float F0[PVNET_IN], F1[PVNET_IN];
        pvnet_features(0, s, es, eo, ef, pa, F0);
        pvnet_features(1, s, es, eo, ef, pa, F1);
        float h0 = pvnet_h(F0, pa.vnet), h1 = pvnet_h(F1, pa.vnet);
        float dh = (our == 0) ? h0 - h1 : h1 - h0;
        float b0s = (our == 0) ? pa.b0 : -pa.b0;
        float pc = fminf(fmaxf(p_our, 1e-6f), 1.0f - 1e-6f);
        float z = b0s + pa.b1 * logf(pc / (1.0f - pc)) + dh;
        return 1.0f / (1.0f + expf(-z));
    }
    float S[2] = {0.0f, 0.0f}, O[2] = {0.0f, 0.0f};
    if (ecfg[CFG_ASSIGN] == 1) {
        // assign mode: each team's heroes go to its players by the best assignment
        best_assignment(0, s, es, eo, pa, &S[0], &O[0]);
        best_assignment(1, s, es, eo, pa, &S[1], &O[1]);
    } else {
        for (int st = 0; st < 16 && st < s.step; st++) {
            if (!c_draft_is_pick[st]) continue;
            int slot = ecfg[24 + st];
            if (slot < 0) continue;
            int h = s.step_action[st];
            int tm = slot < 5 ? 0 : 1;
            S[tm] += es[slot * NUM_HEROES + h];
            O[tm] += eo[slot * NUM_HEROES + h];
        }
    }
    float adj = pa.b2 * (S[our] - S[1 - our]) + pa.b3 * (O[our] - O[1 - our]);
    float b0s = (our == 0) ? pa.b0 : -pa.b0;
    if (adj == 0.0f && b0s == 0.0f && pa.b1 == 1.0f) return p_our;
    float pc = fminf(fmaxf(p_our, 1e-6f), 1.0f - 1e-6f);
    float z = b0s + pa.b1 * logf(pc / (1.0f - pc)) + adj;
    return 1.0f / (1.0f + expf(-z));
}


#include "search_v2_util.cuh"

// ── v2 search hooks (p3_search_v2.inc) ──
#define P3_MASK(ST) ep_valid_mask(ST, mask_buf, ecfg, epool)
#define P3_FORCED(ST) forced_action(ST, ecfg)
#define P3_GD(ST) d_gd_forward(state_buf, mask_buf, W_gd, priors_buf, gd_off, workspace)
#define P3_PRIORS(ST) do { \
        d_policy_backbone(state_buf, W_policy, policy_off, buf_e, workspace); \
        d_policy_head(buf_e, mask_buf, W_policy, policy_off, priors_buf, workspace, (ST).step); \
        apply_prior_bias(priors_buf, mask_buf, (ST).step, ecfg, ebias, pa.prior_alpha); \
    } while (0)
#define P3_ROOT_HOOK() do { \
        if (tid == 0) for (int i_ = 0; i_ < NUM_HEROES; i_++) \
            ep->out_prior[ep->num_our_turns][i_] = mask_buf[i_] > 0.5f ? priors_buf[i_] : 0.0f; \
    } while (0)
#define P3_VALUE(V, ST) personal_value(V, ST, (ST).our_team, ecfg, epers, eoff, efeat, pa)
// thread 0, after the GD softmax at an opponent step (chance cache, open-loop step, rollout)
#define P3_OPP(ST, P) opp_personalize(P, mask_buf, (ST).step, (ST).our_team, ecfg, eimit, pa.imit_w0)
// party mode (2026-10-22): cfg[40 + step] = -2 on one of OUR steps marks it as a stranger's
// step (a chance node like an opponent step, distribution GD x row ecfg[24 + step]); -1 / >= 0 keep
// their meaning (no forced ban / forced ban hero). Without -2 flags every path is unchanged.
#define P3_STRANGER(st) ((st) >= 0 && (st) < 16 && ecfg[40 + (st)] == -2)
// rollout steps: cfg[6] = 3 keeps the player-aware opponent at chance nodes and the main loop only
#define P3_OPP_ROLL(ST, P) do { if (ecfg[6] != 3) P3_OPP(ST, P); } while (0)

// ── Gamma / Dirichlet sampling (root exploration noise) ────────────
// Marsaglia-Tsang with the alpha<1 boost. Only thread 0 calls this.
__device__ float d_sample_gamma(curandState* rng, float alpha) {
    float boost = 1.0f;
    if (alpha < 1.0f) {
        float u = curand_uniform(rng);
        boost = powf(u, 1.0f / alpha);
        alpha += 1.0f;
    }
    float d = alpha - 1.0f / 3.0f;
    float c = rsqrtf(9.0f * d);
    for (int it = 0; it < 100; it++) {
        float x = curand_normal(rng);
        float v = 1.0f + c * x;
        if (v <= 0.0f) continue;
        v = v * v * v;
        float u = curand_uniform(rng);
        if (u < 1.0f - 0.0331f * x * x * x * x) return boost * d * v;
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) return boost * d * v;
    }
    return boost * d;
}

// ── Main MCTS Episode Kernel ───────────────────────────────────────
// root_temp: temperature over root visit counts for the executed action
//   (1.0 = historical behavior: sample proportional to visits; <0.05 = argmax).
// dir_alpha/dir_eps: AlphaZero-style Dirichlet noise mixed into ROOT priors
//   at every our-turn search (dir_eps <= 0 disables; historical behavior).
// out_policies (training targets / logged visit distribution) are NOT
// affected by root_temp — only the executed action selection is.

extern "C" __global__ void mcts_episodes_kernel(
    const float* __restrict__ W_policy,
    const float* __restrict__ W_gd,
    const float* __restrict__ W_wp,
    PolicyNetOffsets policy_off,
    GDNetOffsets gd_off,
    WPNetOffsets wp_off,
    const WPLookupTables* __restrict__ lut,
    const int* __restrict__ episode_configs,  // (num_episodes, 3)
    EpisodeMemory* __restrict__ episodes,
    int num_simulations,
    float c_puct,
    unsigned long long base_seed,
    float root_temp,
    float dir_alpha,
    float dir_eps,
    int guard,
    PersonalArgs pa,
    V2Args v2               // X2 fix: v2.mode 0 = legacy tree (bit-identical), else p3_search_v2.inc
) {
    int ep_idx = blockIdx.x;
    const int* ecfg = pa.cfg + ep_idx * CFG_LEN;
    const unsigned int* epool = pa.pool + ep_idx * NUM_SLOTS * 3;
    const float* epers = pa.pers_s + ep_idx * NUM_SLOTS * NUM_HEROES;
    const float* eoff = pa.pers_off + ep_idx * NUM_SLOTS * NUM_HEROES;
    const float* eimit = pa.imit_bias + ep_idx * NUM_SLOTS * NUM_HEROES;
    const float* ebias = pa.prior_bias + ep_idx * NUM_SLOTS * NUM_HEROES;
    const float* efeat = pa.pers_feat + ep_idx * NUM_SLOTS * NUM_HEROES;
    int tid = threadIdx.x;

    // Shared memory layout (dynamic based on policy network size):
    // state_buf:     max(STATE_DIM, WP_INPUT_DIM) = 291 (reused for policy 290d and WP 291d)
    // mask_buf:       90
    // priors_buf:     90
    // buf_e:         edim (backbone output, persists across head calls)
    // workspace:     hdim*3 + cdim (backbone scratch, reused for WP forward)
    // enriched_buf:   86
    extern __shared__ float smem[];
    int edim = policy_off.edim;
    int ws_size = policy_off.hdim * 3 + policy_off.cdim;
    const int state_buf_size = 291;  // max(STATE_DIM=290, WP_INPUT_DIM=291)
    float* state_buf = smem;
    float* mask_buf = smem + state_buf_size;
    float* priors_buf = smem + state_buf_size + NUM_HEROES;
    float* buf_e = smem + state_buf_size + NUM_HEROES + NUM_HEROES;
    float* workspace = smem + state_buf_size + NUM_HEROES + NUM_HEROES + edim;
    float* enriched_buf = smem + state_buf_size + NUM_HEROES + NUM_HEROES + edim + ws_size;

    EpisodeMemory* ep = &episodes[ep_idx];

    // Thread-local RNG (only thread 0 uses it)
    curandState rng;
    // RNG fix (2026-10-19): a hashed per-episode seed (splitmix64) instead of base_seed + ep_idx
    // with subsequence 0, whose neighbouring seeds give correlated first draws.
    if (tid == 0) {
        unsigned long long z = (unsigned long long)base_seed + 0x9E3779B97F4A7C15ULL * (unsigned long long)(ep_idx + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        curand_init(z, 0, 0, &rng);
    }

    // Init draft state
    __shared__ DraftStateGPU main_state;
    if (tid == 0) {
        main_state.init(ecfg[0], ecfg[1], ecfg[2]);
        ep->num_our_turns = 0;
        ep->max_nodes_used = 0;
        ep->n_capacity_hits = 0;
        ep->decided = 0;
        for (int i = 0; i < NUM_STATS_V2; i++) ep->v2_stats[i] = 0;
    }
    __syncthreads();

    // ── DRAFT LOOP ──
    for (int draft_step = 0; draft_step < DRAFT_STEPS; draft_step++) {
        __shared__ int s_team, s_is_pick, s_our_team, s_skip, s_stop;
        if (tid == 0) {
            s_skip = 0;
            s_stop = 0;
            if (!main_state.is_terminal()) {
                int st = main_state.step;
                int dstep = ecfg[4];
                if (dstep >= 0 && st > dstep) {
                    s_stop = 1;
                } else if (dstep >= 0 && st < dstep) {
                    main_state.apply_action(ecfg[8 + st], main_state.current_team(),
                                            main_state.current_is_pick());
                    s_skip = 1;
                } else {
                    int fa = forced_action(main_state, ecfg);
                    if (fa >= 0) {
                        main_state.apply_action(fa, main_state.current_team(), 0);
                        s_skip = 1;
                    }
                }
            }
            if (main_state.is_terminal() || s_stop) { s_team = -1; }
            else if (!s_skip) {
                s_team = main_state.current_team();
                s_is_pick = main_state.current_is_pick();
                if (ecfg[3] || ecfg[4] >= 0) main_state.our_team = s_team;
                s_our_team = main_state.our_team;
            }
        }
        __syncthreads();
        // Race fix (2026-10-19): copy the flags, then a barrier, before branching. Without it
        // thread 0 could rewrite s_skip/s_team for the next step (after a `continue`) before
        // slower warps read them, putting warps one __syncthreads() out of step.
        {
            int l_skip = s_skip, l_team = s_team;
            __syncthreads();
            if (l_skip) continue;
            if (l_team < 0) break;
        }

        if (s_team == s_our_team) {
            // ═══ OUR TURN: MCTS SEARCH ═══

            // Save state for training
            main_state.to_float_array(state_buf);
            ep_valid_mask(main_state, mask_buf, ecfg, epool);
            if (tid == 0) {
                int t = ep->num_our_turns;
                memcpy(ep->out_states[t], state_buf, STATE_DIM * sizeof(float));
                memcpy(ep->out_masks[t], mask_buf, NUM_HEROES * sizeof(float));
            }
            __syncthreads();

            if (v2.mode == SEARCH_LEGACY) {
            // Init fresh tree
            if (tid == 0) {
                ep->num_nodes = 1;
                ep->num_children_allocated = 0;
                ep->nodes[0].parent_idx = -1;
                ep->nodes[0].action = -1;
                ep->nodes[0].visit_count = 0;
                ep->nodes[0].value_sum = 0.0f;
                ep->nodes[0].is_expanded = 0;
                ep->nodes[0].num_children = 0;
                ep->nodes[0].has_cached_opp = 0;
            }
            __syncthreads();

            // Expand root: backbone + policy head
            d_policy_backbone(state_buf, W_policy, policy_off, buf_e, workspace);
            d_policy_head(buf_e, mask_buf, W_policy, policy_off, priors_buf, workspace, main_state.step);
            apply_prior_bias(priors_buf, mask_buf, main_state.step, ecfg, ebias, pa.prior_alpha);

            if (tid == 0) {
                for (int i = 0; i < NUM_HEROES; i++)
                    ep->out_prior[ep->num_our_turns][i] = mask_buf[i] > 0.5f ? priors_buf[i] : 0.0f;
                ep->nodes[0].is_expanded = 1;
                int nv = 0;
                for (int i = 0; i < NUM_HEROES; i++) if (mask_buf[i] > 0.5f) nv++;
                ep->nodes[0].children_start = 0;
                ep->nodes[0].num_children = nv;
                ep->num_children_allocated = nv;
                int ci = 0;
                for (int i = 0; i < NUM_HEROES; i++) {
                    if (mask_buf[i] < 0.5f) continue;
                    int ch = ep->num_nodes++;
                    ep->child_indices[ci] = ch;
                    ep->nodes[ch].parent_idx = 0;
                    ep->nodes[ch].action = i;
                    ep->nodes[ch].prior = priors_buf[i];
                    ep->nodes[ch].visit_count = 0;
                    ep->nodes[ch].value_sum = 0.0f;
                    ep->nodes[ch].is_expanded = 0;
                    ep->nodes[ch].num_children = 0;
                    ep->nodes[ch].has_cached_opp = 0;
                    ci++;
                }
                // Dirichlet root noise (inference-time exploration):
                // prior <- (1-eps)*prior + eps*Dir(alpha) over valid root actions
                if (dir_eps > 0.0f) {
                    float g[NUM_HEROES];
                    float gsum = 0.0f;
                    int nc = ep->nodes[0].num_children;
                    for (int c = 0; c < nc; c++) {
                        g[c] = d_sample_gamma(&rng, dir_alpha);
                        gsum += g[c];
                    }
                    if (gsum > 1e-12f) {
                        for (int c = 0; c < nc; c++) {
                            int chn = ep->child_indices[ep->nodes[0].children_start + c];
                            ep->nodes[chn].prior = (1.0f - dir_eps) * ep->nodes[chn].prior
                                                   + dir_eps * (g[c] / gsum);
                        }
                    }
                }
            }
            __syncthreads();

            // ── Run simulations ──
            for (int sim = 0; sim < num_simulations; sim++) {
                __shared__ DraftStateGPU scratch;
                __shared__ int s_path[MAX_PATH_DEPTH];
                __shared__ int s_path_len;
                __shared__ int s_leaf_idx;
                __shared__ int s_leaf_needs_expand;
                __shared__ int s_needs_gd;
                __shared__ int s_gd_node_idx;
                __shared__ int s_select_done;

                if (tid == 0) {
                    scratch.copy_from(main_state);
                    s_path[0] = 0;
                    s_path_len = 1;
                    s_select_done = 0;
                    s_needs_gd = 0;
                    s_leaf_idx = 0;
                    s_leaf_needs_expand = 0;
                }
                __syncthreads();

                // SELECT with GD cache miss handling
                while (true) {
                    if (s_needs_gd) {
                        // All threads cooperate on GD forward pass
                        scratch.to_float_array(state_buf);
                        ep_valid_mask(scratch, mask_buf, ecfg, epool);
                        d_gd_forward(state_buf, mask_buf, W_gd, priors_buf, gd_off, workspace);

                        if (tid == 0) {
                            MCTSNodeGPU& node = ep->nodes[s_gd_node_idx];
                            float mx = -1e30f;
                            for (int i = 0; i < NUM_HEROES; i++) mx = fmaxf(mx, priors_buf[i]);
                            float sm = 0;
                            for (int i = 0; i < NUM_HEROES; i++) {
                                node.cached_opp_probs[i] = expf(priors_buf[i] - mx);
                                sm += node.cached_opp_probs[i];
                            }
                            if (sm > 0) for (int i = 0; i < NUM_HEROES; i++) node.cached_opp_probs[i] /= sm;
                            P3_OPP(scratch, node.cached_opp_probs);
                            node.has_cached_opp = 1;
                            s_needs_gd = 0;
                        }
                        __syncthreads();
                    }

                    if (s_select_done) break;

                    // Thread 0 continues SELECT
                    if (tid == 0) {
                        int node_idx = s_path[s_path_len - 1];
                        // Continue traversal until leaf, terminal, or GD miss
                        while (ep->nodes[node_idx].is_expanded && !scratch.is_terminal()) {
                            {
                                int fa = forced_action(scratch, ecfg);
                                if (fa >= 0) {
                                    scratch.apply_action(fa, scratch.current_team(), 0);
                                    continue;
                                }
                            }
                            if (scratch.current_team() == scratch.our_team) {
                                MCTSNodeGPU& node = ep->nodes[node_idx];
                                float best_score = -1e30f;
                                int best_slot = -1;
                                for (int c = 0; c < node.num_children; c++) {
                                    int ci = ep->child_indices[node.children_start + c];
                                    MCTSNodeGPU& child = ep->nodes[ci];
                                    float q = child.q_value();
                                    float u = c_puct * child.prior *
                                              sqrtf((float)node.visit_count) / (1.0f + child.visit_count);
                                    if (q + u > best_score) { best_score = q + u; best_slot = c; }
                                }
                                if (best_slot < 0) break;
                                int chosen = ep->child_indices[node.children_start + best_slot];
                                scratch.apply_action(ep->nodes[chosen].action,
                                    scratch.current_team(), scratch.current_is_pick());
                                node_idx = chosen;
                                if (s_path_len < MAX_PATH_DEPTH) s_path[s_path_len++] = node_idx;
                            } else {
                                MCTSNodeGPU& node = ep->nodes[node_idx];
                                if (!node.has_cached_opp) {
                                    s_needs_gd = 1;
                                    s_gd_node_idx = node_idx;
                                    break;
                                }
                                float r = curand_uniform(&rng);
                                int opp_action = p3_sample_cdf(node.cached_opp_probs, r, 1.0f);
                                P3_SAFE_FALLBACK(opp_action, scratch);
                                scratch.apply_action(opp_action,
                                    scratch.current_team(), scratch.current_is_pick());
                            }
                        }
                        if (!s_needs_gd) {
                            int fa;
                            while (!scratch.is_terminal() &&
                                   (fa = forced_action(scratch, ecfg)) >= 0)
                                scratch.apply_action(fa, scratch.current_team(), 0);
                            s_leaf_idx = node_idx;
                            s_select_done = 1;
                            s_leaf_needs_expand = (!scratch.is_terminal() &&
                                                    !ep->nodes[node_idx].is_expanded &&
                                                    scratch.current_team() == scratch.our_team) ? 1 : 0;
                            if (s_leaf_needs_expand &&
                                (ep->num_nodes + NUM_HEROES > MAX_NODES ||
                                 ep->num_children_allocated + NUM_HEROES > MAX_CHILD_INDICES)) {
                                ep->n_capacity_hits++;
                                if (guard) s_leaf_needs_expand = 0;
                            }
                        }
                    }
                    __syncthreads();
                }

                // EVALUATE leaf: policy head for priors (if expanding) + WP model for value

                __shared__ float s_value;

                if (s_leaf_needs_expand) {
                    // Need backbone + policy head for expansion priors
                    scratch.to_float_array(state_buf);
                    ep_valid_mask(scratch, mask_buf, ecfg, epool);
                    d_policy_backbone(state_buf, W_policy, policy_off, buf_e, workspace);
                    d_policy_head(buf_e, mask_buf, W_policy, policy_off, priors_buf, workspace, scratch.step);
                    apply_prior_bias(priors_buf, mask_buf, scratch.step, ecfg, ebias, pa.prior_alpha);

                    if (tid == 0) {
                        MCTSNodeGPU& leaf = ep->nodes[s_leaf_idx];
                        int nv = 0;
                        for (int i = 0; i < NUM_HEROES; i++) if (mask_buf[i] > 0.5f) nv++;
                        leaf.is_expanded = 1;
                        leaf.children_start = ep->num_children_allocated;
                        leaf.num_children = nv;
                        ep->num_children_allocated += nv;
                        int ci = 0;
                        for (int i = 0; i < NUM_HEROES; i++) {
                            if (mask_buf[i] < 0.5f) continue;
                            int ch = ep->num_nodes++;
                            ep->child_indices[leaf.children_start + ci] = ch;
                            ep->nodes[ch].parent_idx = s_leaf_idx;
                            ep->nodes[ch].action = i;
                            ep->nodes[ch].prior = priors_buf[i];
                            ep->nodes[ch].visit_count = 0;
                            ep->nodes[ch].value_sum = 0.0f;
                            ep->nodes[ch].is_expanded = 0;
                            ep->nodes[ch].num_children = 0;
                            ep->nodes[ch].has_cached_opp = 0;
                            ci++;
                        }
                    }
                    __syncthreads();
                }

                // LEAF EVALUATION: rollout to terminal with GD, then evaluate with WP
                // This ensures the WP model always sees complete 5v5 states (in-distribution)
                // instead of partial mid-draft states (out-of-distribution)

                // Rollout remaining steps using GD for both teams
                while (!scratch.is_terminal()) {
                    __shared__ int s_rfa;
                    if (tid == 0) s_rfa = forced_action(scratch, ecfg);
                    __syncthreads();
                    if (s_rfa >= 0) {
                        if (tid == 0) scratch.apply_action(s_rfa, scratch.current_team(), 0);
                        __syncthreads();
                        continue;
                    }
                    scratch.to_float_array(state_buf);
                    ep_valid_mask(scratch, mask_buf, ecfg, epool);
                    d_gd_forward(state_buf, mask_buf, W_gd, priors_buf, gd_off, workspace);
                    if (tid == 0) {
                        // Softmax + sample
                        float mx = -1e30f;
                        for (int i = 0; i < NUM_HEROES; i++) mx = fmaxf(mx, priors_buf[i]);
                        float sm = 0;
                        for (int i = 0; i < NUM_HEROES; i++) {
                            priors_buf[i] = expf(priors_buf[i] - mx);
                            sm += priors_buf[i];
                        }
                        if (sm > 0) for (int i = 0; i < NUM_HEROES; i++) priors_buf[i] /= sm;
                        P3_OPP_ROLL(scratch, priors_buf);   // opponent rollout steps only
                        float r = curand_uniform(&rng);
                        int action = p3_sample_cdf(priors_buf, r, 1.0f);
                        P3_SAFE_FALLBACK(action, scratch);
                        scratch.apply_action(action, scratch.current_team(), scratch.current_is_pick());
                    }
                    __syncthreads();
                }

                // Now scratch is a complete draft — evaluate with WP (in-distribution)
                __shared__ int s_wp_t0h[5], s_wp_t1h[5], s_wp_n0, s_wp_n1;
                if (tid == 0) {
                    s_wp_n0 = 0; s_wp_n1 = 0;
                    for (int i = 0; i < NUM_HEROES; i++) {
                        int w = i / 32, b = i % 32;
                        if ((scratch.t0_picks[w] >> b) & 1 && s_wp_n0 < 5)
                            s_wp_t0h[s_wp_n0++] = i;
                        if ((scratch.t1_picks[w] >> b) & 1 && s_wp_n1 < 5)
                            s_wp_t1h[s_wp_n1++] = i;
                    }
                }
                __syncthreads();

                {
                    float val = wp_eval_symmetrized(
                        s_wp_t0h, s_wp_n0, s_wp_t1h, s_wp_n1,
                        scratch.map_idx, scratch.tier_idx, scratch.our_team,
                        scratch.step,
                        lut, W_wp, wp_off, state_buf, enriched_buf, workspace);
                    if (tid == 0) s_value = personal_value(val, scratch, scratch.our_team,
                                                           ecfg, epers, eoff, efeat, pa);
                }
                __syncthreads();

                // BACKPROP
                if (tid == 0) {
                    for (int p = 0; p < s_path_len; p++) {
                        ep->nodes[s_path[p]].visit_count++;
                        ep->nodes[s_path[p]].value_sum += s_value;
                    }
                }
                __syncthreads();
            }

            // legacy root extraction (unchanged)
            if (tid == 0) {
                if (ep->num_nodes > ep->max_nodes_used) ep->max_nodes_used = ep->num_nodes;
                int t = ep->num_our_turns;
                float vsum = 0;
                for (int i = 0; i < NUM_HEROES; i++) ep->out_policies[t][i] = 0;
                for (int c = 0; c < ep->nodes[0].num_children; c++) {
                    int ci = ep->child_indices[ep->nodes[0].children_start + c];
                    float v = (float)ep->nodes[ci].visit_count;
                    ep->out_policies[t][ep->nodes[ci].action] = v;
                    vsum += v;
                }
                if (vsum > 0) for (int i = 0; i < NUM_HEROES; i++) ep->out_policies[t][i] /= vsum;
                for (int i = 0; i < NUM_HEROES; i++) ep->out_q[t][i] = 0.0f;
                for (int c = 0; c < ep->nodes[0].num_children; c++) {
                    int ci = ep->child_indices[ep->nodes[0].children_start + c];
                    ep->out_q[t][ep->nodes[ci].action] = ep->nodes[ci].q_value();
                }
            }
            } else {
#include "p3_search_v2.inc"
            }

            // Extract visit distribution and choose action
            if (tid == 0) {
                int t = ep->num_our_turns;
                ep->out_step[t] = main_state.step;
                ep->out_team[t] = s_team;

                // Executed-action selection over the (normalized) visit
                // distribution. The r draw stays unconditional so the RNG
                // stream (and thus GD opponent behavior) is identical across
                // root_temp settings for a given seed.
                float r = curand_uniform(&rng);
                int chosen = -1;
                if (root_temp < 0.05f) {
                    // argmax (T -> 0)
                    float best = -1.0f;
                    for (int i = 0; i < NUM_HEROES; i++) {
                        if (ep->out_policies[t][i] > best && ep->out_policies[t][i] > 0.0f) {
                            best = ep->out_policies[t][i];
                            chosen = i;
                        }
                    }
                } else if (fabsf(root_temp - 1.0f) < 1e-6f) {
                    // T=1: exact historical code path (bit-identical results)
                    chosen = p3_sample_cdf(ep->out_policies[t], r, 1.0f);
                } else {
                    // General T: p_i ∝ visits_i^(1/T)
                    float w[NUM_HEROES];
                    float wsum = 0.0f;
                    float inv_t = 1.0f / root_temp;
                    for (int i = 0; i < NUM_HEROES; i++) {
                        w[i] = ep->out_policies[t][i] > 0.0f
                                 ? powf(ep->out_policies[t][i], inv_t) : 0.0f;
                        wsum += w[i];
                    }
                    chosen = wsum > 0.0f ? p3_sample_cdf(w, r, wsum) : -1;
                }
                P3_SAFE_FALLBACK(chosen, main_state);
                main_state.apply_action(chosen, s_team, s_is_pick);
                ep->num_our_turns++;
                if (ecfg[4] >= 0) ep->decided = 1;
            }
            __syncthreads();
            if (ep->decided) break;

        } else {
            // ═══ OPPONENT TURN: GD ═══
            main_state.to_float_array(state_buf);
            ep_valid_mask(main_state, mask_buf, ecfg, epool);
            d_gd_forward(state_buf, mask_buf, W_gd, priors_buf, gd_off, workspace);

            if (tid == 0) {
                float mx = -1e30f;
                for (int i = 0; i < NUM_HEROES; i++) mx = fmaxf(mx, priors_buf[i]);
                float sm = 0;
                for (int i = 0; i < NUM_HEROES; i++) {
                    priors_buf[i] = expf(priors_buf[i] - mx);
                    sm += priors_buf[i];
                }
                if (sm > 0) for (int i = 0; i < NUM_HEROES; i++) priors_buf[i] /= sm;
                // imitation opponent: softmax(w0 log p_GD + personal bias of the acting slot)
                int oslot = ecfg[24 + main_state.step];
                if (ecfg[6] == 1 && s_is_pick && oslot >= 0) {
                    float mu = -1e30f;
                    for (int i = 0; i < NUM_HEROES; i++) {
                        float u = (priors_buf[i] > 0.0f && mask_buf[i] > 0.5f)
                            ? pa.imit_w0 * logf(priors_buf[i]) + eimit[oslot * NUM_HEROES + i] : -1e30f;
                        priors_buf[i] = u;
                        mu = fmaxf(mu, u);
                    }
                    float su = 0.0f;
                    for (int i = 0; i < NUM_HEROES; i++) {
                        priors_buf[i] = priors_buf[i] > -1e29f ? expf(priors_buf[i] - mu) : 0.0f;
                        su += priors_buf[i];
                    }
                    if (su > 0) for (int i = 0; i < NUM_HEROES; i++) priors_buf[i] /= su;
                }
                // cfg[6] = 2: player-aware opponent at picks and bans
                opp_personalize(priors_buf, mask_buf, main_state.step, main_state.our_team, ecfg, eimit,
                                pa.imit_w0);

                float r = curand_uniform(&rng);
                int action = p3_sample_cdf(priors_buf, r, 1.0f);
                P3_SAFE_FALLBACK(action, main_state);
                main_state.apply_action(action, s_team, s_is_pick);
            }
            __syncthreads();
        }
    }

    // Terminal: compute WP in-kernel using enriched WP model (symmetrized)
    if (ep->decided) return;
    __shared__ int s_term_t0h[5], s_term_t1h[5], s_term_n0, s_term_n1;
    if (tid == 0) {
        s_term_n0 = 0; s_term_n1 = 0;
        for (int i = 0; i < NUM_HEROES; i++) {
            int w = i / 32, b = i % 32;
            if ((main_state.t0_picks[w] >> b) & 1 && s_term_n0 < 5)
                s_term_t0h[s_term_n0++] = i;
            if ((main_state.t1_picks[w] >> b) & 1 && s_term_n1 < 5)
                s_term_t1h[s_term_n1++] = i;
        }
    }
    __syncthreads();

    float term_wp = wp_eval_symmetrized(
        s_term_t0h, s_term_n0, s_term_t1h, s_term_n1,
        main_state.map_idx, main_state.tier_idx, main_state.our_team,
        main_state.step,  // terminal step (15 for complete drafts)
        lut, W_wp, wp_off, state_buf, enriched_buf, workspace);

    // Write terminal state (for debugging/logging) and WP to global memory
    main_state.to_float_array(state_buf);
    for (int i = tid; i < STATE_DIM; i += blockDim.x)
        ep->terminal_state[i] = state_buf[i];
    if (tid == 0) {
        ep->our_team = main_state.our_team;
        ep->win_prob = term_wp;
        float w0 = (main_state.our_team == 0) ? term_wp : 1.0f - term_wp;
        ep->wp_t0 = w0;
        ep->v_t0 = personal_value(w0, main_state, 0, ecfg, epers, eoff, efeat, pa);
        for (int i = 0; i < 16; i++) ep->terminal_actions[i] = main_state.step_action[i];
    }
}

// Verification: leaf value of complete drafts given as 16 prefix actions.
// out[2i] = population WP (our side), out[2i+1] = personal value (our side).
extern "C" __global__ void leaf_eval_kernel(
    const float* __restrict__ W_wp,
    WPNetOffsets wp_off,
    const WPLookupTables* __restrict__ lut,
    PersonalArgs pa,
    float* __restrict__ out
) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const int* ecfg = pa.cfg + idx * CFG_LEN;
    const float* epers = pa.pers_s + idx * NUM_SLOTS * NUM_HEROES;
    const float* eoff = pa.pers_off + idx * NUM_SLOTS * NUM_HEROES;
    const float* efeat = pa.pers_feat + idx * NUM_SLOTS * NUM_HEROES;
    extern __shared__ float smem[];
    float* state_buf = smem;
    float* enriched_buf = smem + 291;
    float* workspace = smem + 291 + 96;
    __shared__ DraftStateGPU st;
    __shared__ int t0h[5], t1h[5], n0, n1;
    if (tid == 0) {
        st.init(ecfg[0], ecfg[1], ecfg[2]);
        for (int k = 0; k < 16; k++)
            st.apply_action(ecfg[8 + k], c_draft_team[k], c_draft_is_pick[k]);
        n0 = 0; n1 = 0;
        for (int i = 0; i < NUM_HEROES; i++) {
            int w = i / 32, b = i % 32;
            if ((st.t0_picks[w] >> b) & 1 && n0 < 5) t0h[n0++] = i;
            if ((st.t1_picks[w] >> b) & 1 && n1 < 5) t1h[n1++] = i;
        }
    }
    __syncthreads();
    float p = wp_eval_symmetrized(t0h, n0, t1h, n1, st.map_idx, st.tier_idx, st.our_team,
                                  st.step, lut, W_wp, wp_off, state_buf, enriched_buf, workspace);
    if (tid == 0) {
        out[2 * idx] = p;
        out[2 * idx + 1] = personal_value(p, st, st.our_team, ecfg, epers, eoff, efeat, pa);
    }
}
