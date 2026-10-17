"""Rich CF (p3_rcf): exactness and as-of fixtures.
  R1 the factor-form posterior equals the phase-1 GP (gp_query) for
     K = U U^T + diag(s2), at played and unplayed heroes
  R2 today's "+CF rank 2" kernel in factor form equals the kernel file
  R3 online_walk: planted rows on the query's own day and later, a flipped
     own outcome, and changed own metrics change no prediction; a lag-0 walk
     is caught (it does change)
  R4 the predictive log likelihood equals a brute-force Gaussian computation
Run: python3 personalization/tests/test_rcf.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3_hs_core as C
import p3_rcf as R

H = 90


def rand_params(rng, K=5, M=3, implicit=True, metrics=True):
    U = 0.02 * rng.standard_normal((H, K))
    s2 = 0.0004 + 0.0002 * rng.random(H)
    kw = {}
    if implicit:
        kw.update(W=0.1 * rng.standard_normal((K, H + 1)), zmean=np.zeros(H + 1), zsd=np.ones(H + 1))
    if metrics:
        kw.update(G=0.2 * rng.standard_normal((M, H, K)), c=0.05 * rng.standard_normal((M, H)),
                  rho2=0.1 + rng.random(M) * 0.1, eps2=0.7 + rng.random(M) * 0.2)
    return R.Params(U, s2, **kw)


def stats(rng, n):
    P = np.where(rng.random((n, H)) < 0.2, rng.integers(1, 40, (n, H)) * 4.2, 0.0)
    S = P * 0.03 * rng.standard_normal((n, H))
    return S, P


def test_r1_equals_gp():
    rng = np.random.default_rng(0)
    par = rand_params(rng, implicit=False, metrics=False)
    S, P = stats(rng, 300)
    h = rng.integers(0, H, 300)
    m1, v1 = R.posterior(par, S, P, h)
    m2, v2 = C.gp_query(np.ascontiguousarray(S), np.ascontiguousarray(P), h, par.kernel())
    assert np.allclose(m1, m2, atol=1e-10), np.abs(m1 - m2).max()
    assert np.allclose(v1, v2, atol=1e-10), np.abs(v1 - v2).max()
    ma, va = R.posterior(par, S, P, h, want_all=True)
    assert np.allclose(ma[np.arange(300), h], m1) and np.allclose(va[np.arange(300), h], v1)


def test_r2_today_kernel():
    if not os.path.exists(R.KERN):
        print("  R2 skipped (no kernel file)")
        return
    import p3_x_common as X  # noqa
    from p3_hs_fit import KERNELS
    kz = np.load(R.KERN)
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    K0 = C.kernel_from([bases[b] for b in KERNELS["+CF rank 2"]], kz["theta_+CF rank 2"])
    par = R.Params.from_kernel_file(R.KERN)
    assert np.allclose(par.kernel(), K0, atol=1e-12), np.abs(par.kernel() - K0).max()


def walk_fixture(rng, n_pl=40, ns=4000, M=3):
    day = np.sort(1000 + rng.integers(0, 200, ns))
    pid = rng.integers(0, n_pl, ns)
    hero = rng.integers(0, 12, ns)
    wp = np.clip(0.5 + 0.1 * rng.standard_normal(ns), 0.2, 0.8)
    y = (rng.random(ns) < wp).astype(np.float64)
    aug = {"day": day, "pid": pid, "hero": hero, "v": wp * (1 - wp), "r_adj": y - wp,
           "slot": np.arange(ns), "n_players": np.int64(n_pl)}
    Z = rng.standard_normal((ns, M)).astype(np.float32)
    Z[rng.random((ns, M)) < 0.1] = np.nan
    return aug, Z, wp, y


def test_r3_asof():
    rng = np.random.default_rng(1)
    par = rand_params(rng)
    aug, Z, wp, y = walk_fixture(rng)
    ns = len(aug["day"])
    q = rng.choice(np.flatnonzero(aug["day"] > 1100), 50, replace=False)
    base = R.online_walk(aug, q, par, metrics=Z)
    caught = 0
    for i in q[:25]:
        # flip own outcome and change own metrics
        a2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in aug.items()}
        a2["r_adj"][i] = (1 - y[i]) - wp[i]
        Z2 = Z.copy()
        Z2[i] = 4.0
        # plant future rows: every later row of the same player on its hero gets a huge outcome
        later = (aug["day"] >= aug["day"][i]) & (aug["pid"] == aug["pid"][i])
        a2["r_adj"][later] = 0.9
        Z2[later] = -4.0
        r2 = R.online_walk(a2, np.array([i]), par, metrics=Z2)
        j = np.flatnonzero(q == i)[0]
        assert abs(r2["mean"][0] - base["mean"][j]) < 1e-12, (i, r2["mean"][0], base["mean"][j])
        leak = R.online_walk(a2, np.array([i]), par, metrics=Z2, lag=0)
        caught += abs(leak["mean"][0] - base["mean"][j]) > 1e-9
    assert caught == 25, caught


def test_r4_predictive_ll():
    rng = np.random.default_rng(2)
    par = rand_params(rng, K=3, implicit=False, metrics=False)
    S1, P1 = stats(rng, 20)
    S2, P2 = stats(rng, 20)
    ll = R.pred_ll_np(par, S1, P1, S2, P2)
    K = par.kernel()
    for i in range(20):
        o1 = np.flatnonzero(P1[i] > 0)
        o2 = np.flatnonzero(P2[i] > 0)
        # theta | E1 by the GP, then E2 cell means = theta + noise
        A = K[np.ix_(o1, o1)] + np.diag(1 / P1[i, o1])
        Kx = K[:, o1]
        mu = Kx @ np.linalg.solve(A, S1[i, o1] / P1[i, o1])
        Sig = K - Kx @ np.linalg.solve(A, Kx.T)
        Cv = Sig[np.ix_(o2, o2)] + np.diag(1 / P2[i, o2])
        r = S2[i, o2] / P2[i, o2] - mu[o2]
        ref = -0.5 * (r @ np.linalg.solve(Cv, r) + np.linalg.slogdet(Cv)[1] + len(o2) * np.log(2 * np.pi))
        assert abs(ref - ll[i]) < 1e-6 * max(1, abs(ref)), (i, ref, ll[i])


if __name__ == "__main__":
    for t in (test_r1_equals_gp, test_r2_today_kernel, test_r3_asof, test_r4_predictive_ll):
        t()
        print(f"{t.__name__}: pass")
