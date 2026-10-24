"""P3 kernels: safe outer action sampling (plan review C24, 2026-10-24).
  K1 every P3 kernel (explore, personal, pvnet, pgd, prior, the personal population reference)
     carries the same p3_sample_cdf helper and P3_SAFE_FALLBACK macro, and no legacy
     `int x = 0; ... if (cum > r)` loop is left
  K2 the helper, compiled from the kernel source on the GPU: mass only on the last legal hero with
     r = 1.0 (float sum below r) returns that hero, never hero 0; an all-zero vector returns -1 and
     the fallback skips taken heroes; a NaN entry is never chosen; the general-temperature path
     (p / denom) behaves the same
  K3 equivalence: on 20,000 random sparse distributions and draws the helper returns exactly what
     the legacy loop returns whenever the legacy loop found an entry
  K4 (P3T_TEST_KERNELS=1 and a rosters file in $P3_CACHE/p3t): full self-play drafts through the
     prior kernel at root temperature 1 (p3t_selfplay.selfplay_round) have 16 distinct heroes
Run (GPU host, from training/): python personalization/tests/test_p3t_kernel_sampling.py"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

KERNELS = ["cuda_explore/explore_kernel.cu", "cuda_personal/personal_kernel.cu", "cuda_pvnet/pvnet_kernel.cu",
           "cuda_pgd/pgd_kernel.cu", "cuda_prior/prior_kernel.cu", "cuda_personal/ref/mcts_kernel.cu"]
H = 90


def helper_source():
    srcs = []
    for f in KERNELS:
        s = open(os.path.join(HERE, f)).read()
        m = re.search(r"__device__ __forceinline__ int p3_sample_cdf\(.*?\n}\n#define P3_SAFE_FALLBACK[^\n]*\n", s, re.S)
        assert m, f"{f}: helper missing"
        srcs.append(m.group(0))
        assert not re.search(r"int \w+ = 0;\n\s*for \(int \w = 0; \w < NUM_HEROES; \w\+\+\) \{\n\s*cum \+=", s), f
        assert "int chosen = 0;" not in s, f
    assert all(x == srcs[0] for x in srcs), "kernels carry different helpers"
    return srcs[0]


def legacy(p, r, denom=1.0):
    cum = np.float32(0)
    for i in range(H):
        cum = np.float32(cum + (np.float32(p[i]) if denom == 1.0 else np.float32(p[i]) / np.float32(denom)))
        if cum > r:
            return i
    return None          # the legacy loop then kept its initial 0


def build():
    import torch
    from torch.utils.cpp_extension import load_inline
    cuda = r'''
#include <torch/extension.h>
#define NUM_HEROES 90
struct FakeState {
    unsigned int taken[3];
    __device__ bool is_taken(int h) const { return (taken[h / 32] >> (h % 32)) & 1u; }
};
''' + helper_source() + r'''
__global__ void k_sample(const float* p, const float* r, const float* denom, const int* taken, int* out, int n) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n) return;
    FakeState st;
    for (int w = 0; w < 3; w++) st.taken[w] = 0u;
    for (int h = 0; h < NUM_HEROES; h++) if (taken[b * NUM_HEROES + h]) st.taken[h / 32] |= 1u << (h % 32);
    int x = p3_sample_cdf(p + b * NUM_HEROES, r[b], denom[b]);
    out[2 * b] = x;
    P3_SAFE_FALLBACK(x, st);
    out[2 * b + 1] = x;
}
torch::Tensor sample(torch::Tensor p, torch::Tensor r, torch::Tensor denom, torch::Tensor taken) {
    int n = p.size(0);
    auto out = torch::zeros({n, 2}, torch::dtype(torch::kInt32).device(p.device()));
    k_sample<<<(n + 127) / 128, 128>>>(p.data_ptr<float>(), r.data_ptr<float>(), denom.data_ptr<float>(),
                                       taken.data_ptr<int>(), out.data_ptr<int>(), n);
    return out;
}
'''
    return load_inline("p3_sample_test", cpp_sources="torch::Tensor sample(torch::Tensor p, torch::Tensor r, torch::Tensor denom, torch::Tensor taken);",
                       cuda_sources=cuda, functions=["sample"], verbose=False)


def run(mod, P, R, D, T):
    import torch
    dev = "cuda"
    o = mod.sample(torch.tensor(np.asarray(P, np.float32), device=dev), torch.tensor(np.asarray(R, np.float32), device=dev),
                   torch.tensor(np.asarray(D, np.float32), device=dev), torch.tensor(np.asarray(T, np.int32), device=dev))
    return o.cpu().numpy()


def main():
    helper_source()
    print("K1 ok")
    mod = build()
    # K2 boundary cases
    P, R, D, T = [], [], [], []
    p = np.zeros(H, np.float32); p[89] = 0.9999999; P.append(p); R.append(1.0); D.append(1.0); T.append(np.zeros(H))
    p = np.zeros(H, np.float32); p[[10, 50, 89]] = [0.3, 0.3, 0.3999]; P.append(p); R.append(1.0); D.append(1.0); T.append(np.zeros(H))
    p = np.zeros(H, np.float32); P.append(p); R.append(0.5); D.append(1.0); t = np.zeros(H); t[[0, 1, 2]] = 1; T.append(t)
    p = np.zeros(H, np.float32); p[0] = np.nan; p[7] = 1.0; P.append(p); R.append(0.5); D.append(1.0); T.append(np.zeros(H))
    w = np.zeros(H, np.float32); w[[20, 89]] = [1.0, 2.0]; P.append(w); R.append(1.0); D.append(2.9999); T.append(np.zeros(H))
    o = run(mod, P, R, D, T)
    assert o[0].tolist() == [89, 89], o[0]
    assert o[1].tolist() == [89, 89], o[1]
    assert o[2].tolist() == [-1, 3], o[2]
    assert o[3].tolist() == [7, 7], o[3]
    assert o[4].tolist() == [89, 89], o[4]
    assert legacy(P[0], np.float32(1.0)) is None          # the legacy loop would have played hero 0
    print("K2 ok")
    # K3 equivalence with the legacy loop wherever it found an entry
    rng = np.random.RandomState(0)
    n = 20000
    P = np.zeros((n, H), np.float32)
    for i in range(n):
        k = rng.randint(1, 30)
        idx = rng.choice(H, k, replace=False)
        P[i, idx] = rng.dirichlet(np.ones(k)).astype(np.float32)
    R = rng.random_sample(n).astype(np.float32)
    R[:200] = 1.0
    D = np.ones(n, np.float32)
    D[n // 2:] = P[n // 2:].sum(1) * 2.0
    P[n // 2:] *= 2.0
    o = run(mod, P, R, D, np.zeros((n, H)))
    found = diff = 0
    for i in range(n):
        lg = legacy(P[i], R[i], float(D[i]))
        if lg is not None:
            found += 1
            diff += int(o[i, 0] != lg)
        else:
            assert P[i, o[i, 0]] > 0, i
    assert diff == 0, f"{diff} differences from the legacy loop"
    print(f"K3 ok ({found} of {n} draws where the legacy loop found an entry, all identical; {n - found} boundary draws safe)")
    if os.environ.get("P3T_TEST_KERNELS") == "1":
        import p3t_data as TD
        import p3t_selfplay as SP
        Z = dict(np.load(os.path.join(TD.OUT, "rosters.npz")))
        idx = np.arange(min(512, len(Z["map"])))
        Rr = {k: Z[k][idx] for k in ("s", "off", "n", "map", "tier")}
        acts, _ = SP.selfplay_round(Rr, None, 64, 4242)
        bad = sum(len(set(a.tolist())) != 16 for a in acts)
        assert bad == 0, f"{bad} self-play drafts repeat a hero"
        print(f"K4 ok ({len(acts)} temperature-1 self-play drafts, all 16 heroes distinct)")
    print("ALL OK")


if __name__ == "__main__":
    main()
