"""Warm-start and cold-to-convergence incumbent refits (Max, 2026-10-15; CYCLE3_PREP.md 1).

  PYTHONPATH=. python -m training.scilearn.tests.test_warm_cold

Fitter, on a synthetic world (real hero, map and kit tables; planted hero, hero x map and
ally-pair effects; 40,000 games):
  1. the legacy path is bit-identical to the cycle-2 fitter (commit fbb4858);
  2. a cold fit meets the convergence criterion (inner and EB);
  3. a warm fit from the start-of-cycle incumbent plus the new term's Q2 offset fit reaches the
     cold optimum within tolerance, in fewer iterations;
  4. the warm start copies coefficients by term spec and name, and new terms start at their
     Q2 values.
Harness, on a fake 3090 (as in test_refit_scheduling):
  5. legacy sessions keep their registry keys; a fit-policy session's start incumbent is cold,
     and its refit after an accept is warm from that incumbent and the card's Q2 job;
  6. the registry entry records the fit mode and the convergence status;
  7. with a warm refit and an end-of-cycle cold refit pending at once, at most one refit
     holds a slot and a short test still runs promptly.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from training.scilearn import config as C
from training.scilearn import fit as F
from training.scilearn import fixtures as FX
from training.scilearn import harness as HN
from training.scilearn import remote as R
from training.scilearn import terms as TM
from training.scilearn.ledger import canonical
from training.scilearn.tests import test_refit_scheduling as TRS

EXTRACT = C.PRIVATE / "extracts" / "snap_20261013_real_kited64c8e6"


def world(n=40000, seed=3):
    meta = json.loads(EXTRACT.with_suffix(".json").read_text())
    kit = np.load(EXTRACT.with_suffix(".npz"))["kit"]
    ctx = FX.synthetic_ctx(meta, kit, n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    H, M = len(meta["heroes"]), len(meta["maps"])
    th = rng.normal(0, 0.15, H)
    thm = rng.normal(0, 0.10, (H, M))
    pair = rng.normal(0, 0.25, (H, H))
    pair = np.triu(pair, 1) + np.triu(pair, 1).T
    A, B = ctx.heroes[:, :5], ctx.heroes[:, 5:]
    ia, ib = np.triu_indices(5, 1)
    eta = (th[A].sum(1) - th[B].sum(1) + thm[A, ctx.map[:, None]].sum(1) - thm[B, ctx.map[:, None]].sum(1)
           + pair[A[:, ia], A[:, ib]].sum(1) - pair[B[:, ia], B[:, ib]].sum(1))
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(np.float64)
    axes = meta["kit_axes"]
    start = [{"type": "lookup", "keys": ["hero"], "name": "h"},
             {"type": "lookup", "keys": ["hero", "map"], "name": "hm"},
             {"type": "attr_sum", "axis": axes[0], "shape": "concave_inc", "n_knots": 4, "name": "ax0"}]
    new = {"type": "pair_family", "min_games": 60, "name": "card_p01"}
    return ctx, y, start, new


def old_fitter():
    """The fitter as of the cycle-2 results commit, loaded from git."""
    src = subprocess.run(["git", "show", "fbb4858:training/scilearn/fit.py"], capture_output=True, text=True,
                         cwd=C.REPO, check=True).stdout
    spec = importlib.util.spec_from_loader("fit_cycle2", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "fit_cycle2.py", "exec"), mod.__dict__)
    return mod


def fitter_cases():
    out = []
    ctx, y, start, new = world()
    n = ctx.n
    tr = np.arange(n) < int(n * 0.8)
    ctx_tr, ctx_te = ctx.subset(np.flatnonzero(tr)), ctx.subset(np.flatnonzero(~tr))
    full = start + [new]

    # 1. legacy path unchanged
    ctx_tr.max_fit_build = int(ctx_tr.build.max())
    X, blocks, offs = TM.compile_model(full, ctx_tr)
    old = old_fitter()
    b0, v0, l0, h0 = old.fit_design(X, y[tr], blocks, offs)
    b1, v1, l1, h1 = F.fit_design(X, y[tr], blocks, offs)
    out.append(("legacy fit_design is bit-identical to the cycle-2 fitter",
                np.array_equal(b0, b1) and np.array_equal(v0, v1) and np.array_equal(l0, l1)
                and [h["nll"] for h in h0] == [h["nll"] for h in h1]))

    # 2. cold
    t0 = time.time()
    cold = F.fit(full, ctx_tr, y[tr], converge={"mode": "cold"})
    t_cold = time.time() - t0
    sc = cold.fit_info["fit_status"]
    out.append((f"cold fit converges (inner and EB): {sc['iterations']} iterations, {sc['eb_rounds']} EB rounds, "
                f"criterion {sc['final_crit']:.1e}", sc["converged"] and sc["inner_converged"] and sc["eb_converged"]))

    # 3. warm: start-of-cycle incumbent (cold) + the new term's Q2 offset fit
    inc0 = F.fit(start, ctx_tr, y[tr], converge={"mode": "cold"})
    inc0.tag = "start incumbent"
    q2 = F.fit([new], ctx_tr, y[tr], offset=inc0.logit(ctx_tr))      # legacy, as the Q2 job does
    q2.tag = "Q2 offset fit"
    t0 = time.time()
    warm = F.fit(full, ctx_tr, y[tr], converge={"mode": "warm"}, warm_from=[inc0, q2])
    t_warm = time.time() - t0
    sw = warm.fit_info["fit_status"]
    dz = np.abs(warm.beta - cold.beta) / np.sqrt(cold.var)
    dlam = np.abs(np.log10(warm.lam) - np.log10(cold.lam)).max()

    def ll(m):
        e = m.logit(ctx_te)
        return float(np.mean(np.logaddexp(0, -np.where(y[~tr] > 0, e, -e))))
    dll = abs(ll(warm) - ll(cold))
    out.append((f"warm fit converges: {sw['iterations']} iterations, {sw['eb_rounds']} EB rounds", sw["converged"]))
    out.append((f"warm reaches the cold optimum: max |d beta| {dz.max():.3f} posterior SD (median {np.median(dz):.4f}), "
                f"max |d log10 lam| {dlam:.3f}, held-out log loss differs by {dll:.1e}",
                dz.max() < 0.25 and np.median(dz) < 0.02 and dlam < 0.1 and dll < 2e-5))
    out.append((f"warm needs fewer iterations than cold ({sw['iterations']} vs {sc['iterations']}; "
                f"{t_warm:.1f} s vs {t_cold:.1f} s)", sw["iterations"] < sc["iterations"]))

    # 4. warm-start mapping
    ws = warm.fit_info["warm_start"]
    names = warm.names
    i_new = [i for i, nm in enumerate(names) if nm.startswith("card_p01[")]
    b_init = F.warm_start(names, TM.compile_model(full, ctx_tr)[1], full, [inc0, q2])[0]
    q2_by = dict(zip(q2.names, q2.beta))
    out.append((f"warm start copies {ws['copied']} coefficients ({ws['from'][0]['coefficients']} from the "
                f"incumbent, {ws['from'][1]['coefficients']} from the Q2 fit), {ws['zero']} at zero",
                ws["from"][0]["coefficients"] == inc0.beta.size and ws["from"][1]["coefficients"] == len(i_new)
                and all(b_init[i] == q2_by[names[i]] for i in i_new)))
    return out


# --------------------------------------------------------------------------- harness

class FakeHost(TRS.FakeHost):
    """The scheduling test's fake 3090, plus job outputs on the local side (spec, INC.npz)
    and a fit_status in the result, as the real worker writes them."""

    def __init__(self):
        super().__init__()
        self.specs = {}

    def wait(self, job_id, host=None, **kw):
        while self.jobs.get(f"scilearn_{job_id}") == "running":
            time.sleep(0.01)
        d = R.JOBS_LOCAL / job_id
        spec = json.loads((d / "spec.json").read_text())
        self.specs[job_id] = spec
        (d / "out").mkdir(parents=True, exist_ok=True)
        np.savez(d / "out" / "INC.npz", beta=np.zeros(1))
        fo = spec.get("fit_options") or {}
        st = {"mode": fo.get("mode", "legacy"), "converged": fo.get("mode") in ("cold", "warm"), "iterations": 7}
        return {"models": {"INC": {"eval": {"Dpost": {"logloss": 0.68}}, "fit_status": st,
                                   "warm_start": {"copied": 5} if fo.get("mode") == "warm" else None}},
                "paired": {}}


def legacy_key(state, dev):
    """incumbent_key as of the cycle-2 results commit."""
    specs = HN.incumbent_specs(state)
    return hashlib.sha256(canonical({"specs": specs, "fit": HN.fit_tranches(dev), "labels": HN.label_mode(state),
                                     "snap": state["snapshot"]}).encode()).hexdigest()[:16]


def harness_cases():
    out = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fake = FakeHost()
        saved = TRS._patch(tmp, fake, [])
        R.make_extract = lambda label_mode="real", **kw: "snap_test_real"
        try:
            # 5. keys
            leg = TRS.FakeBroker("L", [TRS.T("l_c01")])
            out.append(("a session without a fit policy keeps its cycle-2 registry key",
                        HN.incumbent_key(leg.state, "Dpost") == legacy_key(leg.state, "Dpost")
                        and HN.fit_mode(leg.state, "Dpost") == "legacy"))
            A = TRS.FakeBroker("A", [TRS.T("a_c01")])
            A.state["fit_policy"] = {"start_terms_key": HN._terms_key(A.state)}
            e0 = HN.ensure_incumbent(A.state, "Dpost")
            s0 = fake.specs[e0["job"]]
            out.append(("the start incumbent of a fit-policy session is a cold fit to convergence",
                        HN.fit_mode(A.state, "Dpost") == "cold" and s0["fit_options"]["mode"] == "cold"
                        and not s0.get("warm_from")))
            # an accepted card with a finished Q2 job
            A.add_card("p01")
            A.state["cards"]["p01"]["term"] = TRS.T("card_p01")
            (R.JOBS_LOCAL / "q2_A_p01" / "out").mkdir(parents=True)
            np.savez(R.JOBS_LOCAL / "q2_A_p01" / "out" / "term.npz", beta=np.zeros(1))
            A.accept("card_p01")
            e1 = HN.ensure_incumbent(A.state, "Dpost")
            s1 = fake.specs[e1["job"]]
            out.append(("after an accept the refit is warm, from the start incumbent and the card's Q2 fit",
                        s1["fit_options"]["mode"] == "warm"
                        and [w["job"] for w in s1["warm_from"]] == [e0["job"], "q2_A_p01"]))
            # 6. metadata
            reg = HN._registry()
            out.append(("registry entries record fit mode, convergence and warm sources",
                        reg[e0["job"][4:]]["fit_mode"] == "cold" and reg[e0["job"][4:]]["fit_status"]["converged"]
                        and reg[e1["job"][4:]]["fit_mode"] == "warm"
                        and reg[e1["job"][4:]]["warm_from"] == [e0["job"], "q2_A_p01"]))
            ce = HN.cold_refit(A.state, "Dpost")
            out.append(("an end-of-cycle cold refit of the final terms is a separate cold fit",
                        ce["fit_mode"] == "cold" and ce["job"] != e1["job"]
                        and fake.specs[ce["job"]]["fit_options"]["mode"] == "cold"))
        finally:
            TRS._restore(saved)

    # 7. scheduling with a warm and a cold refit pending
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fake = FakeHost()
        B = TRS.FakeBroker("B", [TRS.T("b_c01")])
        D_ = TRS.FakeBroker("D", [TRS.T("d_c01")])
        for b in (B, D_):
            b.state["fit_policy"] = {"start_terms_key": HN._terms_key(b.state)}
        S = TRS.FakeBroker("S", [TRS.T("s_c01")])
        saved = TRS._patch(tmp, fake, [json.loads(json.dumps(S.state))])
        R.make_extract = lambda label_mode="real", **kw: "snap_test_real"
        try:
            HN.ensure_incumbent(B.state, "Dpost")              # B's start incumbent exists
            B.accept("b_p01")
            B.add_card("x1")
            th = [threading.Thread(target=HN.finish_q2, args=(B, "x1")),           # waits for a warm refit
                  threading.Thread(target=HN.cold_refit, args=(json.loads(json.dumps(D_.state)), "Dpost"))]
            for t in th:
                t.start()
            time.sleep(0.2)
            S.add_card("y0")
            t_short = time.time()
            short = threading.Thread(target=TRS._short_test, args=(S, "y0", HN._snapshot(S)))
            short.start()
            for t in th + [short]:
                t.join(timeout=30)
            lat = S.gated.get("y0", (None, 1e18))[1] - t_short
            out.append((f"warm and cold refits pending: at most one refit holds a slot ({fake.max_refits}), "
                        f"a short test runs in {lat:.2f} s (refit {TRS.REFIT_S} s)",
                        fake.max_refits == 1 and fake.max_total <= HN.REMOTE_SLOTS and lat < TRS.REFIT_S / 2
                        and "x1" in B.gated))
        finally:
            TRS._restore(saved)
    return out


def run():
    res = harness_cases() + fitter_cases()
    for name, ok in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    n = sum(ok for _, ok in res)
    print(f"{n}/{len(res)} passed")
    return n == len(res)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
