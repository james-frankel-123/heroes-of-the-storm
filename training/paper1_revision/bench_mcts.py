"""
Benchmark MCTS policies (submission checkpoints and leak-free retrains) on a
fixed set of draft configurations and score every draft under proxies and
independent references.

Protocol (paper Table VI setting, larger sample): our MCTS agent plays one side
against the 5 GD behavioral-cloning models (cycled per batch), mid tier,
random map and side, 200 inference simulations, c_puct 2.0. The same 1,000
configurations (seed 20260929) are used for every run, so config contrasts
are paired by draft. Root selection: T=1 (the submission's benchmark) and T=0
(argmax root, the recommended operating point). Kernel: overfit2026/cuda_ofit
(the paper kernel plus a tree-capacity guard; bit-identical when the guard is
not hit; cap hits are recorded).

Leaf value during the benchmark = the value function the run was trained
against (submission: wp_enriched_256 + external stats, or wp_true_base for
K_truebase; revision: leak-free enriched + own deploy statistics).

Scores per draft, P(our team wins), team-order symmetrized:
  proxy_sub   submission's enriched WP with the external stats (Table VI "Avg WP")
  proxy_rev   revision's leak-free enriched WP with own deploy stats
  naive_rev   revision's hero-identity judge (no statistics)
  gN          mean of 3 out-of-fold enriched judges trained only on 291,837
              post-snapshot no-drift games (overfit2026)
  gN_naive    hero-identity judge on the same games
  RN          realized-outcome index on the same games (NODRIFT)
  R17         realized-outcome index on 2.55.17 games (drifted meta)
  QM2026      Quick Match judge (draft-free mode, current era)
plus degen / healer / synergy / counter (own deploy stats and external stats).

Usage: python3 paper1_revision/bench_mcts.py <run>[,<run>...] [--temps 1,0]
  runs: new:<B|F|J>_oof_s<k>   old:<config>_s<k>   (e.g. old:J_800sim_s9)
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "cuda_mcts"))
from paper1_revision import core

import numpy as np
import torch

N_DRAFTS = 1000
SEED = 20260929
OUT = os.path.join(core.RESULTS, "mcts_bench")
RR = os.path.join(TRAINING_DIR, "rerun2026")

OLD_WP = {"K_truebase": "true_base", "H_augmented": "aug",
          "A_partial": "partial", "G_base": "base_partial",
          "M2_relational": "relational", "N2_absolute": "absolute"}   # else: enriched_full
NET = {"C_large": ("large", "linear"), "D_deep": ("base", "deep")}  # else base/linear
SUB_WP_FILES = {"sub": "wp_enriched_256.pt", "aug": "wp_aug_v2_256.pt",
                "relational": "wp_relational_256_expanded283.pt",
                "absolute": "wp_absolute_256_expanded283.pt"}


def ckpt_of(run):
    kind, name = run.split(":", 1)
    if kind == "new":
        return os.path.join(HERE, "mcts_runs", name, "draft_policy.pt")
    return os.path.join(RR, "mcts_runs", name, "draft_policy.pt")


_WCFG = {}


def leaf_config(run):
    """Kernel WP config (flat weights, offsets, LUT) for the run's own value fn."""
    from extract_weights import extract_wp_weights, build_wp_net_offsets, extract_lookup_tables
    from sweep_enriched_wp import WinProbEnrichedModel
    kind, name = run.split(":", 1)
    exp = name.rsplit("_s", 1)[0]
    key = "rev" if kind == "new" else OLD_WP.get(exp, "sub")
    if key in _WCFG:
        return _WCFG[key]
    if key == "rev":
        from paper1_revision import train_wp
        m, cols = train_wp.load("enriched")
        st = core.load_stats("deploy")
        dim = 283
    elif key in SUB_WP_FILES:
        m = WinProbEnrichedModel(283, [256, 128], dropout=0.3)
        m.load_state_dict(torch.load(os.path.join(RR, "models", SUB_WP_FILES[key]),
                                     weights_only=True, map_location="cpu"))
        st = core.load_stats("hp")
        dim = 283
    elif key in ("partial", "base_partial"):
        # step-conditioned WP (train_mcts_worker "enriched"/"base" types): the
        # step embedding rides in the LUT, input = features + 8 step dims
        from train_partial_wp import PartialStateWP
        enr = key == "partial"
        fn = "partial_wp.pt" if enr else "base_partial_wp.pt"
        ck = torch.load(os.path.join(RR, "models", fn), weights_only=True, map_location="cpu")
        m = PartialStateWP(input_dim=283 if enr else 197, step_embed_dim=8, hidden=(256, 128))
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        st = core.load_stats("hp")
        wf, wn = extract_wp_weights(m)
        wo = build_wp_net_offsets(m, wn, (283 if enr else 197) + 8, use_enriched=enr)
        se = m.step_embed.weight.data.cpu().numpy()
        _WCFG[key] = (wf, wo, extract_lookup_tables(st, step_embed_weights=se))
        return _WCFG[key]
    else:   # true_base
        m = WinProbEnrichedModel(197, [256, 128], dropout=0.3)
        m.load_state_dict(torch.load(os.path.join(RR, "models", "wp_true_base.pt"),
                                     weights_only=True, map_location="cpu"))
        st = core.load_stats("hp")
        dim = 197
    m.eval()
    wf, wn = extract_wp_weights(m)
    wo = build_wp_net_offsets(m, wn, dim, use_enriched=(dim == 283))
    _WCFG[key] = (wf, wo, extract_lookup_tables(st))
    return _WCFG[key]


def policy_flat(run):
    from overfit2026 import search
    exp = run.split(":", 1)[1].rsplit("_s", 1)[0]
    if exp not in NET:
        return search.policy_flat("path:" + ckpt_of(run))
    from train_draft_policy import AlphaZeroDraftNet
    from extract_weights import extract_policy_weights
    size, head = NET[exp]
    net = AlphaZeroDraftNet(size=size, policy_head_type=head)
    sd = torch.load(ckpt_of(run), weights_only=True, map_location="cpu")
    if any(k.startswith("res_block1.") for k in sd):
        sd = {k.replace("res_block1.", "res_blocks.0.").replace("res_block2.", "res_blocks.1.")
              .replace("res_block3.", "res_blocks.2."): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    return extract_policy_weights(net)


def generate(run, temp, kernel):
    from overfit2026 import search
    return search.run(kernel, policy_flat(run), leaf_config(run), n=N_DRAFTS, sims=200,
                      seed=SEED, root_temp=temp, guard=1, batch=100)


# ── scoring ─────────────────────────────────────────────────────────────

_JUDGES = {}


def rev_model(name):
    if name not in _JUDGES:
        from paper1_revision import train_wp
        _JUDGES[name] = train_wp.load(name)
    return _JUDGES[name]


def sub_enriched():
    if "sub" not in _JUDGES:
        from sweep_enriched_wp import WinProbEnrichedModel
        m = WinProbEnrichedModel(283, [256, 128], dropout=0.3)
        m.load_state_dict(torch.load(os.path.join(RR, "models", "wp_enriched_256.pt"),
                                     weights_only=True, map_location="cpu"))
        m.eval()
        _JUDGES["sub"] = m
    return _JUDGES["sub"]


def sym_predict(model, cols, Xf, Xs):
    with torch.no_grad():
        a = model(torch.tensor(Xf[:, cols])).view(-1).numpy()
        b = model(torch.tensor(Xs[:, cols])).view(-1).numpy()
    return 0.5 * (a + 1 - b)


def interaction(rows, st):
    """Per-draft (synergy, counter) of our team, paper conventions."""
    syn, ctr = [], []
    for o, p, m, t in rows:
        sd = []
        for j, a in enumerate(o):
            for b in o[j + 1:]:
                r = st.get_synergy(a, b, t)
                if r is not None:
                    sd.append(r - (50 + (st.get_hero_wr(a, t) - 50) + (st.get_hero_wr(b, t) - 50)))
        cd = []
        for a in o:
            for b in p:
                r = st.get_counter(a, b, t)
                if r is not None:
                    cd.append(r - (st.get_hero_wr(a, t) + (100 - st.get_hero_wr(b, t)) - 50))
        syn.append(np.mean(sd) if sd else 0.0)
        ctr.append(np.mean(cd) if cd else 0.0)
    return np.array(syn), np.array(ctr)


def score_drafts(drafts):
    """-> dict of per-draft arrays for every reference."""
    from overfit2026 import score as oscore
    from overfit2026.stage_b_saved import qm_scores
    from shared import is_degenerate, HERO_ROLE_FINE
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    rows = [(tuple(d["our"]), tuple(d["opp"]), d["map"], d.get("tier", "mid")) for d in drafts]
    out = {}
    # submission proxy (external stats)
    Xf, Xs = core.featurize(rows, core.load_stats("hp"), nproc=min(6, int(os.environ.get("P1R_NPROC", "6"))))
    out["proxy_sub"] = sym_predict(sub_enriched(), core.group_cols(ENRICHED_GROUPS), Xf, Xs)
    out["syn_hp"], out["ctr_hp"] = interaction(rows, core.load_stats("hp"))
    # revision models (own deploy stats)
    Xf, Xs = core.featurize(rows, core.load_stats("deploy"), nproc=min(6, int(os.environ.get("P1R_NPROC", "6"))))
    for name, key in (("enriched", "proxy_rev"), ("naive", "naive_rev")):
        m, cols = rev_model(name)
        out[key] = sym_predict(m, cols, Xf, Xs)
    out["syn_own"], out["ctr_own"] = interaction(rows, core.load_stats("deploy"))
    # independent references (overfit2026)
    GN = oscore.GN
    ms = oscore.model_scores(GN + [oscore.GN_NAIVE], rows, dev="cpu")
    out["gN"] = np.mean([ms[n] for n in GN], 0)
    out["gN_naive"] = ms[oscore.GN_NAIVE]
    rn, r17 = oscore.realized("NODRIFT"), oscore.realized("T17")
    out["RN"] = np.array([rn.score(o, p, t) for o, p, m, t in rows])
    out["R17"] = np.array([r17.score(o, p, t) for o, p, m, t in rows])
    q = qm_scores(oscore.qm(), rows)
    out["QM2026"] = q["QM2026"]
    # qm_wp_2022.pt (2022-era QM judge). Before 2026-10-01 this column was
    # stored under the wrong key "QM2021"; the tournament's QM2021 is qm_wp_v0.
    out["QM2022"] = q.get("QM2022", np.full(len(rows), np.nan))
    healers = {h for h, r in HERO_ROLE_FINE.items() if r == "healer"}
    out["degen"] = np.array([float(is_degenerate(list(o))) for o, p, m, t in rows])
    out["healer"] = np.array([float(any(h in healers for h in o)) for o, p, m, t in rows])
    return out


def diversity(drafts):
    from collections import Counter
    c = Counter(h for d in drafts for h in d["our"])
    tot = sum(c.values())
    p = np.array(list(c.values())) / tot
    return {"distinct": len(c), "entropy_bits": float(-(p * np.log2(p)).sum()),
            "top10_share": float(sum(v for _, v in c.most_common(10)) / tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--temps", default="1,0")
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    os.makedirs(OUT, exist_ok=True)
    from overfit2026 import search
    kernel = search.load_kernel("ofit")
    torch.set_num_threads(4)
    for run in a.runs.split(","):
        if not os.path.exists(ckpt_of(run)):
            print(f"missing {run}", flush=True)
            continue
        for T in (float(x) for x in a.temps.split(",")):
            path = os.path.join(OUT, f"{run.replace(':', '__')}__T{T:g}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            drafts = generate(run, T, kernel)
            sc = score_drafts(drafts)
            res = {"run": run, "temp": T, "n": len(drafts),
                   "cap_hits": int(sum(d.get("cap_hits", 0) for d in drafts)),
                   "means": {k: float(np.nanmean(v)) for k, v in sc.items()},
                   "diversity": diversity(drafts),
                   "per_draft": {k: np.round(v, 5).tolist() for k, v in sc.items()},
                   "drafts": [{k: d[k] for k in ("our", "opp", "map", "side")} for d in drafts],
                   "secs": time.time() - t0}
            json.dump(res, open(path, "w"))
            m = res["means"]
            print(f"{run:24s} T{T:g}: sub={m['proxy_sub']:.4f} rev={m['proxy_rev']:.4f} "
                  f"gN={m['gN']:.4f} RN={m['RN']:.4f} R17={m['R17']:.4f} QM={m['QM2026']:.4f} "
                  f"deg={m['degen']:.3f} ({res['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
