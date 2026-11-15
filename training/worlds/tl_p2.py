"""
Step 4a, paper 2: truth-leaf search. The selected design's search (p3t_systems.act_mcts_x through the support
kernel, the same priors, opponent model, ban bias, assignment bias, budget and kernel seeds) with only the
leaf swapped for the world's truth:
  * the kernel is worlds/cuda_truth (personalization/cuda_support built with -DP3_TRUTH: wp_eval_symmetrized
    returns the world's base-model truth for one posterior draw, with family a's dials; the W_wp buffer
    carries the world parameters);
  * hero-generic worlds (family a): the leaf is that truth alone (cfg[7] = 0, no personal term);
  * player worlds (f-out, f-cov): the leaf adds the world's true player term, the kernel's personal-value
    path fed the world's u'(p, h) table (b0 = 0, b1 = 1, b2 = 1, b3 = 0, best assignment);
  * priors, ban bias and assignment bias still use the drafter's own state (only the leaf changes).
Agents: TL-sel (the selected design's config, support kernel with no support term) and TL-f25 (s-floor25's
config: the graded floor at 25). Opponent: pMCTS-cur, both seats, the 300 headline lobbies.
Family a's triple layer is in the kernel as a dense per-world table.

Stages (3090 GPU, from training/ with the paper-2 env):
  run --fam a|f|fc --dial default|stress --wid W
  queue                                    the plan below
  score                                    -> site/results/truthleaf_p2.json
"""
import os
import sys
import json
import time
import inspect
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
P2 = os.path.join(TRAINING_DIR, "personalization")
sys.path.insert(0, HERE)
sys.path.insert(0, P2)
sys.path.insert(0, TRAINING_DIR)
import data as WD  # noqa: E402
import model as M  # noqa: E402
import world as W  # noqa: E402

OPP = os.environ.get("TL_P2_OPP", "pMCTS-cur")
OUT = os.path.join(WD.CACHE, "tl_p2" + ("" if OPP == "pMCTS-cur" else "_" + OPP))
PLAN = [("a", "default", range(20)), ("f", "default", range(20)), ("fc", "default", range(20)), ("a", "stress", range(20))]
TL_AGENTS = {"TL-sel": "x-combo1-dr-chance", "TL-f25": "s-floor25"}
_TRUTH = {}


def truth_params(wid, dial):
    """float32 buffer in worlds/cuda_truth/truth_eval.cuh's layout (kernel map / tier order)."""
    from shared import MAPS, SKILL_TIERS
    Ws = W.Worlds()
    th = Ws.base_draws([wid])[0]
    q = Ws.B.unpack(th)
    maps, tiers = WD.meta()
    r = Ws.B.r
    V = np.stack([q["V"][maps.index(m)] for m in MAPS])
    U = np.stack([q["U"][tiers.index(t)] for t in SKILL_TIERS])
    S = np.zeros((90, 90))
    iu = np.triu_indices(90, 1)
    S[iu] = q["S"][M.PID[iu]]
    C = np.zeros((90, 90))
    C[iu] = q["K"][M.PID[iu]]
    C = C - C.T
    c3 = np.load(os.path.join(TRAINING_DIR, "paper1_revision", "site", "results", "support_mitigation", "triple_counts.npy"))
    # family a's triple layer of world wid: mu + sd (1 + (k - 1) g(n)) zeta for every sorted triple
    L = Ws.layer("triple")
    a_, b_, c_ = np.meshgrid(np.arange(90), np.arange(90), np.arange(90), indexing="ij")
    K = ((a_ * 90 + b_) * 90 + c_).ravel()
    ok = ((a_ < b_) & (b_ < c_)).ravel()
    mu, sd, nn = L.look(K[ok])
    kk = 1 + (dial.get("k", 1.0) - 1) * W.grade(nn, 25)
    L3 = np.zeros(len(K))
    L3[ok] = mu + sd * kk * W.hnorm(K[ok], wid, W.LAYER_SALT["triple"])
    head = np.zeros(16)
    head[:7] = [r, dial.get("delta", 0.0), dial.get("sigma", 0.0), dial.get("a", 1.0), 25, wid, 61 + ord("a")]
    return np.ascontiguousarray(np.concatenate([head, q["w"], V.ravel(), U.ravel(), S.ravel(), C.ravel(),
                                                q["E"].ravel(), c3.astype(np.float64), L3]).astype(np.float32))


def truth_u_tables(fam, wid, gi):
    """(n, 10, 90) the world's u'(p, h) for each kernel slot's player (slot 5t + j = draft team t's j-th picker)."""
    import player as PL
    L = np.load(PL.P2_LOBBIES)
    pos = {int(g): i for i, g in enumerate(L["headline_gi"])}
    ix = np.array([pos[int(g)] for g in gi])
    pid, cnt = L["headline_pid"][ix], L["headline_cnt"][ix]                 # (n, 2, 5), (n, 2, 5, 90)
    P = np.repeat(pid[..., None], 90, -1)
    Hh = np.broadcast_to(np.arange(90), P.shape)
    if fam == "fc":
        cz = np.load(PL.COV)
        U = PL.PlayerWorldsCov().u(P, Hh, cnt, wid, s=cz["headline_s"][ix], off=cz["headline_off"][ix].astype(float))
    else:
        U = PL.PlayerWorlds2().u(P, Hh, cnt, wid)
    return U.reshape(len(gi), 10, 90).astype(np.float32)


def install(fam, wid, dial_name):
    """Swap the leaf: truth kernel + parameters, and the per-mode inputs the kernel's leaf reads."""
    import p3t_systems as SY
    import p3_mcts_core as Mc
    import p3t_tourney as TT
    dial = {k: v for k, v in W.dials()["named"][dial_name].items() if k != "drift"}
    _TRUTH.clear()
    _TRUTH.update(fam=fam, params=truth_params(wid, dial if fam == "a" else {}),
                  mod=Mc.load_module("truth_kernel", os.path.join(HERE, "cuda_truth")))
    if fam in ("f", "fc"):
        LB = TT.Lobbies()
        _TRUTH["u"] = dict(zip(LB.gi.tolist(), truth_u_tables(fam, wid, LB.gi)))
    src = inspect.getsource(SY.act_mcts_x)
    rep = [('kmod = M.load_module("support_kernel", os.path.join(HERE, "cuda_support"))', 'kmod = _TRUTH["mod"]'),
           ("wf, wo, lut = vint.wcfg()", "wf, wo, lut = vint.wcfg()\n    wf = _TRUTH['params']"),
           ("eng.run(np.ascontiguousarray(cfg[ii]), np.ascontiguousarray(s[ii]), np.ascontiguousarray(off[ii]),",
            "eng.run(np.ascontiguousarray(_tcfg(cfg)[ii]), np.ascontiguousarray(_ts(ctx, s)[ii]), np.ascontiguousarray(_toff(off)[ii]),"),
           ("np.ascontiguousarray(bias[ii]), np.ascontiguousarray(feat[ii]), vnet, sims, cp,",
            "np.ascontiguousarray(bias[ii]), np.ascontiguousarray(feat[ii]), vnet, sims, cp,")]
    for a, b in rep:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    src = src.replace("pool_bits\"][ii]), coefs,", "pool_bits\"][ii]), _tcoefs(coefs),")
    assert "_tcoefs(coefs)" in src
    src = src.replace("def act_mcts_x(", "def act_mcts_x_truth(", 1)
    ns = dict(SY.__dict__)
    ns.update(_TRUTH=_TRUTH, _tcfg=_tcfg, _ts=_ts, _toff=_toff, _tcoefs=_tcoefs)
    exec(compile(src, "act_mcts_x_truth", "exec"), ns)
    truth_act = ns["act_mcts_x_truth"]
    real_act = SY.act_mcts_x

    def act(vint, ctx, x, device=0):
        return truth_act(vint, ctx, x, device) if x.get("_truth") else real_act(vint, ctx, x, device)
    SY.act_mcts_x = act
    for tl, base in TL_AGENTS.items():
        sp = dict(SY.SYSTEMS[base]) if base in SY.SYSTEMS else None
        if sp is None:
            import p3t_support as SP
            sp = SP.support_systems()[base]
        x = dict(sp["x"], _truth=True)
        x.setdefault("support", {})
        SY.SYSTEMS[tl] = dict(sp, x=x)


def _tcfg(cfg):
    c = cfg.copy()
    if _TRUTH["fam"] == "a":
        c[:, 7] = 0                       # population (truth) leaf only
    return c


def _ts(ctx, s):
    if _TRUTH["fam"] == "a":
        return s
    return np.stack([_TRUTH["u"][int(g)] for g in ctx["gi"]])


def _toff(off):
    return off if _TRUTH["fam"] == "a" else np.zeros_like(off)


def _tcoefs(coefs):
    c = np.array(coefs, np.float32).copy()
    if _TRUTH["fam"] != "a":
        c[:4] = [0.0, 1.0, 1.0, 0.0]      # leaf = sigmoid(logit truth + sum of the true u' difference)
    return c


def run(fam, wid, dial):
    import p3t_tourney as TT
    d = os.path.join(OUT, f"{fam}_{dial}_w{wid}")
    os.makedirs(d, exist_ok=True)
    install(fam, wid, dial)
    TT.pair_file = lambda a, b: os.path.join(d, f"{a}__{b}.npz")
    LB = TT.Lobbies()
    for tl in TL_AGENTS:
        for a, b in ((tl, OPP), (OPP, tl)):
            TT.play(LB, a, b, device=0)


def queue():
    import subprocess
    for fam, dial, ws in PLAN:
        for w in ws:
            d = os.path.join(OUT, f"{fam}_{dial}_w{w}")
            if os.path.isdir(d) and len([f for f in os.listdir(d) if f.endswith(".npz") and "tmp" not in f]) == 4:
                continue
            rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "run", "--fam", fam, "--wid", str(w),
                                  "--dial", dial], cwd=TRAINING_DIR)
            print(fam, dial, w, "rc", rc, flush=True)
    print("TL P2 QUEUE DONE", flush=True)


def score():
    """True value of each truth-leaf agent vs pMCTS-cur (both seats) and of the real arms in the same world."""
    import rescore as RS
    out = {"opponent": OPP, "settings": {}}
    hdr = RS.p2_drafts("headline")
    for fam, dial, ws in PLAN:
        rows = []
        for w in ws:
            d = os.path.join(OUT, f"{fam}_{dial}_w{w}")
            if not os.path.isdir(d):
                continue
            row = {"world": w}
            for tl in TL_AGENTS:
                vals = []
                for a, b, sgn in ((tl, OPP, 0), (OPP, tl, 1)):
                    f = os.path.join(d, f"{a}__{b}.npz")
                    if not os.path.exists(f):
                        break
                    z = np.load(f)
                    D = _p2_arrays(z["acts"], hdr)
                    p = _truth_p(fam, dial, w, D)
                    vals.append(p if sgn == 0 else 1 - p)
                if len(vals) == 2:
                    row[tl] = float(np.concatenate(vals).mean())
            # the real arms in the same world (rescore, same lobbies, vs pMCTS-cur)
            zf = np.load(os.path.join(WD.CACHE, f"rescore_p2_headline_{fam}.npz"))
            keys = [tuple(k) for k in json.loads(str(zf["keys"]))]
            ci = 2 if fam in ("f", "fc") else list(W.dials()["named"]).index(dial)    # f: 'best' assignment
            for arm in ("x-combo1-dr-chance", "s-floor25", "s-pcal-floor25"):
                if (arm, OPP) in keys:
                    row[arm] = float(zf["V"][w, ci, keys.index((arm, OPP))])
            rows.append(row)
        rows = [r for r in rows if all(k in r for k in list(TL_AGENTS) + ["x-combo1-dr-chance"])]
        if not rows:
            continue
        S = {"worlds": len(rows), "value_pts": {k: float(100 * np.mean([r[k] for r in rows if k in r]))
                                                 for k in list(TL_AGENTS) + ["x-combo1-dr-chance", "s-floor25", "s-pcal-floor25"]},
             "regret": {"x-combo1-dr-chance (vs TL-sel)": RS.contrast([r["TL-sel"] - r["x-combo1-dr-chance"] for r in rows]),
                        "s-floor25 (vs TL-f25)": RS.contrast([r["TL-f25"] - r["s-floor25"] for r in rows]),
                        "s-pcal-floor25 (vs TL-f25)": RS.contrast([r["TL-f25"] - r["s-pcal-floor25"] for r in rows])},
             "floor_true_effect": RS.contrast([r["TL-f25"] - r["TL-sel"] for r in rows])}
        out["settings"][f"{fam}/{dial}"] = S
    json.dump(out, open(os.path.join(WD.RESULTS, "truthleaf_p2" + ("" if OPP == "pMCTS-cur" else "_" + OPP) + ".json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def _p2_arrays(acts, hdr):
    import rescore as RS
    PK = [[k for k in range(16) if RS.IS_PICK[k] and RS.DRAFT_TEAM[k] == t] for t in (0, 1)]
    n = len(acts)
    first = {k: hdr[k][:n] for k in ("m", "t", "pid", "cnt")}       # lobby order = the headline file order
    return dict(T0=np.sort(acts[:, PK[0]], 1), T1=np.sort(acts[:, PK[1]], 1), H0=acts[:, PK[0]], H1=acts[:, PK[1]],
                lob=np.arange(n), **first)


def _truth_p(fam, dial, wid, D):
    import rescore as RS
    Ws = W.Worlds()
    if fam == "a":
        d = {k: v for k, v in W.dials()["named"][dial].items() if k != "drift"}
        return M.sig(Ws.logit("a", D, wid, **d))
    import player as PL
    PW = PL.PlayerWorldsCov() if fam == "fc" else PL.PlayerWorlds2()
    if fam == "fc":
        cz = np.load(PL.COV)
        D = dict(D, s=cz["headline_s"][D["lob"]], off=cz["headline_off"][D["lob"]])
    A = RS.assign_terms(PW, D, wid, np.random.default_rng([wid, 3]))
    return M.sig(Ws.logit("a", D, wid) + A["best"])


def verify(wid=3):
    """Kernel truth leaf (leaf_eval on complete drafts) vs the Python world, for family a (default and stress)
    and f-out (personal path, best assignment). -> site/results/truthleaf_p2_verify.json"""
    import p3t_systems as SY
    import p3_mcts_core as Mc
    import p3t_tourney as TT
    import player as PL
    import rescore as RS
    f = os.path.join(OUT, "a_default_w0", "TL-sel__pMCTS-cur.npz")
    acts = np.load(f)["acts"]
    hdr = RS.p2_drafts("headline")
    D = _p2_arrays(acts, hdr)
    LB = TT.Lobbies()
    vint = SY.vintage("cur")
    wf, wo, lut = vint.wcfg()
    pf, po = vint.policy_flat(0)
    gf, go = vint.gd_flat(0)
    mod = Mc.load_module("truth_kernel", os.path.join(HERE, "cuda_truth"))
    n = len(acts)
    cfg = np.full((n, Mc.CFG_LEN), -1, np.int32)
    cfg[:, 0], cfg[:, 1], cfg[:, 2] = LB.map, LB.tier, 0
    cfg[:, 3], cfg[:, 4], cfg[:, 5], cfg[:, 6], cfg[:, 7] = 0, 16, 1, 0, 0
    cfg[:, 8:24] = acts
    cfg[:, 24:40] = LB.slot_of
    cfg[:, 56:Mc.CFG_LEN] = 0
    cfg[:, Mc.CFG_ASSIGN] = 1
    z = np.zeros((n, 10, 90), np.float32)
    res = {}
    for name, dial, fam in (("a_default", "default", "a"), ("a_stress", "stress", "a"), ("f_default", "default", "f")):
        dd = {k: v for k, v in W.dials()["named"][dial].items() if k != "drift"}
        prm = truth_params(wid, dd if fam == "a" else {})
        eng = getattr(mod, "PersonalEngine")(pf, gf, prm, po, go, wo, lut, max_concurrent=n, device_id=0)
        c = cfg.copy()
        if fam == "a":
            o = eng.leaf_eval(c, z, z, z, np.ascontiguousarray(LB.pool_bits), np.array([0, 1, 0, 0, 1, 1], np.float32), z, z,
                              np.zeros(0, np.float32))
            kern = o[:, 0]
            py = M.sig(W.Worlds().logit("a", D, wid, **dd))
        else:
            c[:, 7] = 1
            U = truth_u_tables("f", wid, LB.gi)
            o = eng.leaf_eval(c, np.ascontiguousarray(U), z, z, np.ascontiguousarray(LB.pool_bits),
                              np.array([0, 1, 1, 0, 1, 1], np.float32), z, z, np.zeros(0, np.float32))
            kern = o[:, 1]
            A = RS.assign_terms(PL.PlayerWorlds2(), D, wid, np.random.default_rng(0))
            py = M.sig(W.Worlds().logit("a", D, wid) + A["best"])
        del eng
        # the Python world includes the higher-order layers (not in the kernel); compare also the base only
        res[name] = {"max_abs_pts": float(100 * np.abs(kern - py).max()), "mean_abs_pts": float(100 * np.abs(kern - py).mean()),
                     "corr": float(np.corrcoef(kern, py)[0, 1])}
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(WD.RESULTS, "truthleaf_p2_verify.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "queue", "score", "verify"])
    ap.add_argument("--fam", default="a")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--dial", default="default")
    a = ap.parse_args()
    if a.stage == "run":
        run(a.fam, a.wid, a.dial)
    elif a.stage == "queue":
        queue()
    elif a.stage == "verify":
        verify()
    else:
        score()
