// Hidden-worlds truth leaf (training/worlds/tl_p2.py). With -DP3_TRUTH, wp_eval_symmetrized returns the
// world's true P(our team wins) instead of the WP network: the base world model of training/worlds/model.py
// (hero + hero x map + hero x tier, pair synergy, antisymmetric pair counter, CP triple term) for one
// posterior draw, plus family a's off-support dials (delta, sigma, attenuation a graded by g(min3)).
// The W_wp buffer carries the parameters instead of network weights (layout: worlds/tl_p2.py truth_params).
// Family a's triple layer is a dense per-world table (L3); the other layers are not used by family a.
#pragma once
#define TR_OFF_W 16
__device__ __forceinline__ unsigned long long tr_sm64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ float tr_team(const int* T, int n, int m, int t, const float* P) {
    const int r = (int)P[0];
    const float delta = P[1], sigma = P[2], att = P[3], F = P[4];
    const float* w = P + TR_OFF_W;
    const float* V = w + 90;
    const float* U = V + 14 * 90;
    const float* S = U + 3 * 90;
    const float* E = S + 2 * 8100;         // C sits between S and E
    const float* C3 = E + 90 * r;
    const float* L3 = C3 + 729000;          // family a's triple layer of this world (dense 90^3)
    int h[5];
    for (int i = 0; i < n; i++) h[i] = T[i];
    for (int i = 1; i < n; i++) { int v = h[i], j = i - 1; while (j >= 0 && h[j] > v) { h[j + 1] = h[j]; j--; } h[j + 1] = v; }
    float main = 0.0f, own = 0.0f;
    for (int i = 0; i < n; i++) main += w[h[i]] + V[m * 90 + h[i]] + U[t * 90 + h[i]];
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) own += S[h[i] * 90 + h[j]];
    for (int k = 0; k < r; k++) {
        float p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        for (int i = 0; i < n; i++) { float e = E[h[i] * r + k]; p1 += e; p2 += e * e; p3 += e * e * e; }
        own += (p1 * p1 * p1 - 3.0f * p1 * p2 + 2.0f * p3) / 6.0f;
    }
    float mn = 1e30f;
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) for (int c = b + 1; c < n; c++) {
        int key = (h[a] * 90 + h[b]) * 90 + h[c];
        mn = fminf(mn, C3[key]);
        own += L3[key];
    }
    if (n < 3) mn = 1e30f;
    float g = fminf(fmaxf(1.0f - log1pf(mn) / log1pf(F), 0.0f), 1.0f);
    float s = main + (1.0f + (att - 1.0f) * g) * own + 0.04f * delta * g;
    if (sigma != 0.0f && n == 5) {
        unsigned long long key = 0ULL;
        for (int i = 0; i < 5; i++) key = key * 90ULL + (unsigned long long)h[i];
        unsigned long long sd = tr_sm64((unsigned long long)P[5] * 1000003ULL + (unsigned long long)P[6]);
        unsigned long long hh = tr_sm64(key ^ sd);
        double u = ((double)(hh >> 11) + 0.5) * 1.1102230246251565e-16;
        s += 0.04f * sigma * g * (float)normcdfinv(u);
    }
    return s;
}

__device__ float truth_eval(const int* t0, int n0, const int* t1, int n1, int m, int t, int our, const float* P) {
    const float* Cm = P + TR_OFF_W + 90 + 14 * 90 + 3 * 90 + 8100;
    float z = tr_team(t0, n0, m, t, P) - tr_team(t1, n1, m, t, P);
    for (int i = 0; i < n0; i++) for (int j = 0; j < n1; j++) z += Cm[t0[i] * 90 + t1[j]];
    float p0 = 1.0f / (1.0f + expf(-z));
    return our == 0 ? p0 : 1.0f - p0;
}
