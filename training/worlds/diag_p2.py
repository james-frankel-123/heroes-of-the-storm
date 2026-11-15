"""
Paper 2 personal-term diagnosis: where does the truth-leaf regret of the promoted drafter (s-pcal-floor25)
sit in f-cov (and f-out)? Same machinery as tl_p2.py (act_mcts_x through the support kernel, the drafter's
own priors, opponent model, ban bias and assignment bias), on the frozen TUNE lobbies, with one leaf part
swapped at a time:

  leaf = sig(b0 + b1 logit WP + b2 dS + b3 dO)            (drafter; pcal: b2, b3 x 0.864)
  TL     sig(logit truth_pop + dU_true)                    (full truth leaf, as tl_p2)
  H      sig(logit truth_pop + b2 dS + b3 dO)              hero-generic part only (world's posterior draw)
  Hm     sig(logit truth_mean + b2 dS + b3 dO)             hero-generic: the world model's posterior mean
  P      sig(b0 + b1 logit WP + dU_true)                   player part only (world's u')
  Pm     sig(b0 + b1 logit WP + dU_mean)                   player part: u' without the world's random draws
  HPm    sig(logit truth_pop + dU_mean)                    truth hero-generic + knowable player part
  W      sig(b0 + b1 logit WP + b2* dS + b3* dO)           weights only: b2*, b3* = OLS of the true player
                                                           term on the drafter's dS, dO (pooled worlds)
  TL-noban / TL-noassign / TL-noopp, R-noban / R-noassign / R-noopp: the search biases ablated
                                                           under the truth leaf and under the real leaf
dU = best-assignment team difference of u'(p, h) (as the world's scoring, rescore.assign_terms "best").
World-independent agents (Hm, Pm, W, R-*) are played once; the rest per world.

Stages (3090 GPU, from training/, paper-2 env via worlds/jobs/run_p2.sh, P3T_LOBBYSET=tune):
  oracle --fam fc                 W's weights -> site/results/diag_p2_oracle_<fam>.json
  run --fam fc --wid W [--agents ...]
  queue --fam fc --worlds 10
  score --fam fc                  -> site/results/diag_p2_<fam>.json
"""
import os
import sys
import json
import inspect
import argparse

import numpy as np

os.environ.setdefault("P3T_LOBBYSET", "tune")
HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
P2 = os.path.join(TRAINING_DIR, "personalization")
sys.path.insert(0, HERE)
sys.path.insert(0, P2)
sys.path.insert(0, TRAINING_DIR)
import data as WD  # noqa: E402
import model as M  # noqa: E402
import world as W  # noqa: E402

LSET = "tune"
BASE = "s-pcal-floor25"
OPPS = ["pMCTS-cur", "aMCTS-pp"]
OUT = os.path.join(WD.CACHE, "diag_p2")
# agent -> (hero-generic leaf, player leaf, coefs mode, x overrides)
AGENTS = {
    "TL": ("truth", "true", "truth", {}),
    "H": ("truth", "drafter", "keep_b23", {}),
    "Hm": ("mean", "drafter", "keep_b23", {}),
    "P": ("wp", "true", "unit_b2", {}),
    "Pm": ("wp", "mean", "unit_b2", {}),
    "HPm": ("truth", "mean", "truth", {}),
    "W": ("wp", "drafter", "oracle", {}),
    "TL-noban": ("truth", "true", "truth", {"ban_bias": None}),
    "TL-noassign": ("truth", "true", "truth", {"assign_bias": None}),
    "TL-noopp": ("truth", "true", "truth", {"opp": None, "opp_chance_only": None}),
    "R-noban": ("wp", "drafter", "real", {"ban_bias": None}),
    "R-noassign": ("wp", "drafter", "real", {"assign_bias": None}),
    "R-noopp": ("wp", "drafter", "real", {"opp": None, "opp_chance_only": None}),
}
WORLD_INDEP = {"Hm", "Pm", "W", "R-noban", "R-noassign", "R-noopp"}
# world-dependent agents: worlds 0..n-1 (the search-bias ablations on fewer worlds)
N_WORLDS = {"TL-noban": 4, "TL-noassign": 4, "TL-noopp": 4}
_D = {}


def lob_arrays():
    import player as PL
    L = np.load(PL.P2_LOBBIES)
    return {k: L[f"{LSET}_{k}"] for k in ("gi", "pid", "cnt", "map", "tier")}


def cov_arrays():
    import player as PL
    cz = np.load(PL.COV)
    return cz[f"{LSET}_s"], cz[f"{LSET}_off"].astype(float)


def truth_params(wid, mean=False):
    """tl_p2.truth_params (default dials) with the option of the world model's posterior mean."""
    import tl_p2 as TP
    if not mean:
        return TP.truth_params(wid, {})
    orig = W.Worlds.base_draws
    W.Worlds.base_draws = lambda self, wids: [self.theta for _ in wids]
    orig_hnorm = W.hnorm
    W.hnorm = lambda K, wid, salt: np.zeros(np.shape(K))       # triple layer at its mean
    try:
        return TP.truth_params(wid, {})
    finally:
        W.Worlds.base_draws = orig
        W.hnorm = orig_hnorm


def u_tables(fam, wid, mean=False):
    """(n, 10, 90) u'(p, h) of the lobby players in kernel slot order (5t + j); mean: no random draws."""
    import player as PL
    L = lob_arrays()
    P = np.repeat(L["pid"][..., None], 90, -1)
    Hh = np.broadcast_to(np.arange(90), P.shape)
    U = _player_u(fam, P, Hh, L["cnt"], wid, mean, *cov_arrays()) if fam == "fc" else \
        PL.PlayerWorlds2().u(P, Hh, L["cnt"], wid, mean_only=mean)
    return U.reshape(len(L["gi"]), 10, 90).astype(np.float32)


def _player_u(fam, P, Hh, N, wid, mean, s, off):
    import player as PL
    if fam != "fc":
        return PL.PlayerWorlds2().u(P, Hh, N, wid, mean_only=mean)
    pc = PL.PlayerWorldsCov()
    return pc.v2.u(P, Hh, N, wid, mean_only=mean) + pc.b[0] * s + pc.b[1] * off - pc.ch[Hh]


def oracle_file(fam):
    return os.path.join(WD.RESULTS, f"diag_p2_oracle_{fam}.json")


def install(fam, wid):
    """act_mcts_x with per-agent leaf inputs (x['_diag'] = agent name)."""
    import p3t_systems as SY
    import p3_mcts_core as Mc
    import p3t_support as SP
    _D.clear()
    _D.update(fam=fam, wid=wid, mods={}, params={}, u={})
    src = inspect.getsource(SY.act_mcts_x)
    rep = [('kmod = M.load_module("support_kernel", os.path.join(HERE, "cuda_support"))', 'kmod = _kmod(x)'),
           ("wf, wo, lut = vint.wcfg()", "wf, wo, lut = vint.wcfg()\n    wf = _wf(x, wf)"),
           ("eng.run(np.ascontiguousarray(cfg[ii]), np.ascontiguousarray(s[ii]), np.ascontiguousarray(off[ii]),",
            "eng.run(np.ascontiguousarray(cfg[ii]), np.ascontiguousarray(_ts(x, ctx, s)[ii]), np.ascontiguousarray(_toff(x, off)[ii]),"),
           ("pool_bits\"][ii]), coefs,", "pool_bits\"][ii]), _tcoefs(x, coefs),")]
    for a, b in rep:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    src = src.replace("def act_mcts_x(", "def act_mcts_x_diag(", 1)
    ns = dict(SY.__dict__)
    ns.update(_kmod=lambda x: _kmod(x, Mc), _wf=_wf, _ts=_ts, _toff=_toff, _tcoefs=_tcoefs)
    exec(compile(src, "act_mcts_x_diag", "exec"), ns)
    diag_act = ns["act_mcts_x_diag"]
    real_act = SY.act_mcts_x

    def act(vint, ctx, x, device=0):
        return diag_act(vint, ctx, x, device) if x.get("_diag") else real_act(vint, ctx, x, device)
    SY.act_mcts_x = act
    base = SP.support_systems()[BASE]
    for name, (_, _, _, ov) in AGENTS.items():
        x = dict(base["x"], _diag=name)
        for k, v in ov.items():
            if v is None:
                x.pop(k, None)
            else:
                x[k] = v
        SY.SYSTEMS["D-" + name] = dict(base, x=x)


def _kmod(x, Mc):
    hg = AGENTS[x["_diag"]][0]
    key = "support" if hg == "wp" else "truth"
    if key not in _D["mods"]:
        _D["mods"][key] = (Mc.load_module("truth_kernel", os.path.join(HERE, "cuda_truth")) if key == "truth" else
                           Mc.load_module("support_kernel", os.path.join(P2, "cuda_support")))
    return _D["mods"][key]


def _wf(x, wf):
    hg = AGENTS[x["_diag"]][0]
    if hg == "wp":
        return wf
    if hg not in _D["params"]:
        _D["params"][hg] = truth_params(_D["wid"], mean=(hg == "mean"))
    return _D["params"][hg]


def _ts(x, ctx, s):
    pl = AGENTS[x["_diag"]][1]
    if pl == "drafter":
        return s
    if pl not in _D["u"]:
        L = lob_arrays()
        _D["u"][pl] = dict(zip(L["gi"].tolist(), u_tables(_D["fam"], _D["wid"], mean=(pl == "mean"))))
    return np.stack([_D["u"][pl][int(g)] for g in ctx["gi"]])


def _toff(x, off):
    return off if AGENTS[x["_diag"]][1] == "drafter" else np.zeros_like(off)


def _tcoefs(x, coefs):
    mode = AGENTS[x["_diag"]][2]
    c = np.array(coefs, np.float32).copy()
    if mode == "truth":
        c[:4] = [0.0, 1.0, 1.0, 0.0]
    elif mode == "keep_b23":
        c[:2] = [0.0, 1.0]
    elif mode == "unit_b2":
        c[2:4] = [1.0, 0.0]
    elif mode == "oracle":
        o = json.load(open(oracle_file(_D["fam"])))
        c[2:4] = [o["b2"], o["b3"]]
    return c


def agent_dir(fam, name, wid):
    return os.path.join(OUT, fam, name if name in WORLD_INDEP else f"{name}_w{wid}")


def run(fam, wid, agents):
    import p3t_tourney as TT
    assert os.environ.get("P3T_LOBBYSET") == "tune" and TT.TDIR.endswith("tourney_tune")
    install(fam, wid)
    LB = TT.Lobbies()
    assert (LB.gi == lob_arrays()["gi"]).all()
    for name in agents:
        if name == "W" and not os.path.exists(oracle_file(fam)):
            continue
        d = agent_dir(fam, name, wid)
        os.makedirs(d, exist_ok=True)
        TT.pair_file = lambda a, b: os.path.join(d, f"{a}__{b}.npz")
        for o in OPPS:
            for a, b in (("D-" + name, o), (o, "D-" + name)):
                TT.play(LB, a, b, device=0)


def queue(fam, n_worlds, agents):
    import subprocess
    for w in range(n_worlds):
        todo = []
        for name in agents:
            if (name in WORLD_INDEP and w > 0) or w >= N_WORLDS.get(name, n_worlds):
                continue
            d = agent_dir(fam, name, w)
            if not (os.path.isdir(d) and len([f for f in os.listdir(d) if f.endswith(".npz") and "tmp" not in f]) == 2 * len(OPPS)):
                todo.append(name)
        if not todo:
            continue
        rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "run", "--fam", fam, "--wid", str(w),
                              "--agents", *todo], cwd=TRAINING_DIR)
        print(fam, w, todo, "rc", rc, flush=True)
    print("DIAG P2 QUEUE DONE", flush=True)


# ── scoring ───────────────────────────────────────────────────────────────
PK = None


def _D_of(acts):
    import rescore as RS
    global PK
    PK = [[k for k in range(16) if RS.IS_PICK[k] and RS.DRAFT_TEAM[k] == t] for t in (0, 1)]
    L = lob_arrays()
    maps, tiers = WD.meta()
    n = len(acts)
    lob = np.arange(n)
    D = dict(T0=np.sort(acts[:, PK[0]], 1), T1=np.sort(acts[:, PK[1]], 1), H0=acts[:, PK[0]], H1=acts[:, PK[1]], lob=lob,
             m=np.array([maps.index(str(x)) if str(x) in maps else len(maps) for x in L["map"]]),
             t=np.array([tiers.index(str(x)) if str(x) in tiers else len(tiers) for x in L["tier"]]),
             pid=L["pid"], cnt=L["cnt"])
    return D


def true_p(fam, wid, acts, Ws=None, PW=None, parts=False):
    """True P(kernel team 0 wins) of each draft in world wid (best assignment), as rescore / tl_p2."""
    import rescore as RS
    import player as PL
    Ws = Ws or W.Worlds()
    D = _D_of(acts)
    if fam == "fc":
        s, off = cov_arrays()
        D = dict(D, s=s, off=off)
        PW = PW or PL.PlayerWorldsCov()
    else:
        PW = PW or PL.PlayerWorlds2()
    A = RS.assign_terms(PW, D, wid, np.random.default_rng([wid, 3]))
    g = Ws.logit("a", D, wid)
    if parts:
        return g, A["best"], D
    return M.sig(g + A["best"])


def drafter_terms(D, b2, b3):
    """The drafter's dS, dO (team 0 minus team 1) under its own best assignment (b2 s + b3 off)."""
    import rescore as RS
    s, off = cov_arrays()
    out = []
    for t, Hk in ((0, "H0"), (1, "H1")):
        Hh = np.repeat(D[Hk][:, None, :], 5, 1)
        Sv = np.take_along_axis(s[D["lob"], t], Hh, 2)
        Ov = np.take_along_axis(off[D["lob"], t], Hh, 2)
        tS = Sv[:, np.arange(5)[None, :], RS.PERMS].sum(2)
        tO = Ov[:, np.arange(5)[None, :], RS.PERMS].sum(2)
        k = (b2 * tS + b3 * tO).argmax(1)
        out.append((tS[np.arange(len(k)), k], tO[np.arange(len(k)), k]))
    return out[0][0] - out[1][0], out[0][1] - out[1][1]


def real_files():
    import rescore as RS
    tdir = os.path.join(RS.P2DIR, "tourney_tune")
    return {o: (os.path.join(tdir, f"{BASE}__{o}.npz"), os.path.join(tdir, f"{o}__{BASE}.npz")) for o in OPPS}


def oracle(fam, n_worlds=10):
    """W's b2*, b3*: OLS (no intercept; antisymmetric) of the world's true player term on the drafter's dS, dO,
    over the tune drafts of the real arm and the selected design vs both tune opponents, pooled over worlds."""
    import rescore as RS
    import p3t_systems as SY
    b = list(SY.vintage("cur").spec["combiner"])
    b2, b3 = 0.864 * b[2], 0.864 * b[3]
    tdir = os.path.join(RS.P2DIR, "tourney_tune")
    acts = []
    for a in (BASE, RS.SEL):
        for o in OPPS:
            for x, y in ((a, o), (o, a)):
                acts.append(np.load(os.path.join(tdir, f"{x}__{y}.npz"))["acts"])
    Ws = W.Worlds()
    X, Y = [], []
    for w in range(n_worlds):
        for A in acts:
            g, ab, D = true_p(fam, w, A, Ws=Ws, parts=True)
            dS, dO = drafter_terms(D, b2, b3)
            X.append(np.column_stack([dS, dO]))
            Y.append(ab)
    X, Y = np.concatenate(X), np.concatenate(Y)
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    r2 = 1 - ((Y - X @ coef) ** 2).mean() / (Y ** 2).mean()
    c1 = float(np.linalg.lstsq(X[:, :1], Y, rcond=None)[0][0])
    out = {"fam": fam, "worlds": n_worlds, "n": int(len(Y)), "b2": float(coef[0]), "b3": float(coef[1]), "b2_only_s": c1,
           "drafter_b2": b2, "drafter_b3": b3, "r2_uncentred": float(r2),
           "note": "OLS of the true best-assignment player term (logit) on the drafter's dS, dO"}
    json.dump(out, open(oracle_file(fam), "w"), indent=1)
    print(json.dumps(out, indent=1))


def boot_ci(M_, n=4000, seed=0):
    """M_ (worlds, lobbies) paired differences -> mean, 95% CI resampling worlds and lobbies."""
    rng = np.random.default_rng(seed)
    nw, nl = M_.shape
    bs = np.empty(n)
    for i in range(n):
        wi = rng.integers(0, nw, nw)
        li = rng.integers(0, nl, nl)
        bs[i] = M_[wi][:, li].mean()
    return [float(100 * M_.mean()), float(100 * np.percentile(bs, 2.5)), float(100 * np.percentile(bs, 97.5))]


def score(fam, n_worlds):
    import player as PL
    Ws = W.Worlds()
    PW = PL.PlayerWorldsCov() if fam == "fc" else PL.PlayerWorlds2()
    out = {"fam": fam, "lobbies": LSET, "base": BASE, "opponents": {}}
    names = ["R"] + list(AGENTS)
    for o in OPPS:
        V = {}                                     # agent -> {world: (lobbies,) true P(agent wins), seats averaged}
        for name in names:
            V[name] = {}
            for w in range(n_worlds):
                if name == "R":
                    f0, f1 = real_files()[o]
                else:
                    d = agent_dir(fam, name, w)
                    f0, f1 = (os.path.join(d, f"D-{name}__{o}.npz"), os.path.join(d, f"{o}__D-{name}.npz"))
                if not (os.path.exists(f0) and os.path.exists(f1)):
                    continue
                p0 = true_p(fam, w, np.load(f0)["acts"], Ws, PW)
                p1 = true_p(fam, w, np.load(f1)["acts"], Ws, PW)
                V[name][w] = 0.5 * (p0 + (1 - p1))
        V = {k: v for k, v in V.items() if v}

        def diff(a, b):
            ws = sorted(set(V[a]) & set(V[b]))
            return np.stack([V[a][w] - V[b][w] for w in ws]), ws
        res = {"value_pts": {k: float(100 * np.mean(list(v.values()))) for k, v in V.items()},
               "worlds": {k: sorted(v) for k, v in V.items()}}
        if "TL" in V:
            gap, ws = diff("TL", "R")
            res["regret_R"] = boot_ci(gap)
            res["closed"] = {}
            for name in V:
                if name in ("R", "TL") or name.startswith("TL-"):
                    continue
                cw = sorted(set(V[name]) & set(ws))
                g = np.stack([V[name][w] - V["R"][w] for w in cw])
                gg = np.stack([V["TL"][w] - V["R"][w] for w in cw])
                res["closed"][name] = {"gain_vs_R": boot_ci(g), "share_of_regret": float(g.mean() / gg.mean()),
                                       "remaining_vs_TL": boot_ci(-(g - gg))}
            res["tl_ablation"] = {n: boot_ci(diff(n, "TL")[0]) for n in V if n.startswith("TL-")}
        out["opponents"][o] = res
    json.dump(out, open(os.path.join(WD.RESULTS, f"diag_p2_{fam}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["oracle", "run", "queue", "score"])
    ap.add_argument("--fam", default="fc")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=10)
    ap.add_argument("--agents", nargs="*", default=list(AGENTS))
    a = ap.parse_args()
    if a.stage == "oracle":
        oracle(a.fam, a.worlds)
    elif a.stage == "run":
        run(a.fam, a.wid, a.agents)
    elif a.stage == "queue":
        queue(a.fam, a.worlds, a.agents)
    else:
        score(a.fam, a.worlds)
