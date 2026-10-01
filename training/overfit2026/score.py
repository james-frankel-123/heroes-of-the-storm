"""
Scoring of generated drafts (our team vs opponent) under proxies and gold
references. Every score is P(our team wins), symmetrized over team order.

References
  proxy:<model>   a side-A WP with its own deploy statistics (optimized-against)
  gB              mean of the 3 leakage-free (OOF) side-B judges, B statistics
  gB_leak         side-B judge trained with in-sample (leaky) statistics
  gB_naive        side-B hero-identity judge (197-d, no statistics)
  RB              realized-outcome index cross-fitted on half-B games
  RN              realized-outcome index on post-snapshot no-drift games
  QM2026          Quick Match judge (other game mode)
  ensA_mean/std   4-seed side-A leaky ensemble (epistemic disagreement)
Structure: degen, healer, and the NODRIFT realized components.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch

from overfit2026 import feats, gold, split, data

_CACHE = {}


def _stats(name):
    k = ("stats", name)
    if k not in _CACHE:
        _CACHE[k] = split.load_stats(name)
    return _CACHE[k]


def model(name):
    k = ("model", name)
    if k not in _CACHE:
        _CACHE[k] = split.load_model(name)
    return _CACHE[k]


def deploy_stats_name(model_name):
    meta = model(model_name)[1]
    return f"{meta['side']}{meta['e']}"


def realized_B():
    k = ("RB",)
    if k not in _CACHE:
        games = [g for g in data.load_snapshot() if split.side_of(g[0]) == "B"]
        _CACHE[k] = gold.get_index("B", games=games)
    return _CACHE[k]


def realized(name):
    k = ("R", name)
    if k not in _CACHE:
        _CACHE[k] = gold.get_index(name)
    return _CACHE[k]


def qm():
    k = ("qm",)
    if k not in _CACHE:
        from overfit2026.stage_b_saved import load_qm
        _CACHE[k] = load_qm()
    return _CACHE[k]


def _feat(rows, stats_name):
    k = ("feat", stats_name, id(rows))
    return feats.featurize(rows, _stats(stats_name))


def model_scores(names, rows, dev="cuda"):
    """names: list of split model names. Returns {name: P(our wins)}."""
    by_stats = {}
    for n in names:
        by_stats.setdefault(deploy_stats_name(n), []).append(n)
    out = {}
    for sname, ns in by_stats.items():
        Xf, Xs = feats.featurize(rows, _stats(sname), nproc=int(os.environ.get("P1R_NPROC", "6")))
        for n in ns:
            m, meta = model(n)
            d = meta["input_dim"]
            out[n] = feats.predict_sym(m, Xf[:, :d], Xs[:, :d], dev)
    return out


GOLD_B = ["gB8_oof_s0", "gB8_oof_s1", "gB8_oof_s2"]
# Post-snapshot judges for the paper-1 agents. gN = the 3 enriched judges
# rebuilt with their OWN role-composition table (comp_gn_rebuild.py, audit
# P1-A5); GN_EXTCOMP = the originals, which read the external Heroes Profile
# table (pinned copy, feats.PIN_HP_COMPS), kept for comparison only.
GN = ["gN8o_oof_s0", "gN8o_oof_s1", "gN8o_oof_s2"]
GN_EXTCOMP = ["gN8_oof_s0", "gN8_oof_s1", "gN8_oof_s2"]
GN_NAIVE = "gN8_naive"
ENS_A = ["pA8_leak", "pA8_leak_s1", "pA8_leak_s2", "pA8_leak_s3"]


def score(drafts, proxies=(), extra_models=(), dev="cuda"):
    from shared import is_degenerate, HERO_ROLE_FINE
    rows = [(tuple(d["our"]), tuple(d["opp"]), d["map"], d.get("tier", "mid")) for d in drafts]
    names = list(dict.fromkeys(list(proxies) + GOLD_B + ["gB8_leak", "gB8_naive"] + ENS_A
                               + list(extra_models)))
    ms = model_scores(names, rows, dev)
    out = {f"proxy:{p}": ms[p] for p in proxies}
    for n in extra_models:
        out[f"m:{n}"] = ms[n]
    out["gB"] = np.mean([ms[n] for n in GOLD_B], 0)
    out["gB_leak"] = ms["gB8_leak"]
    out["gB_naive"] = ms["gB8_naive"]
    E = np.stack([ms[n] for n in ENS_A])
    out["ensA_mean"] = E.mean(0)
    out["ensA_std"] = E.std(0)
    rb, rn = realized_B(), realized("NODRIFT")
    out["RB"] = np.array([rb.score(o, p, t) for o, p, m, t in rows])
    out["RN"] = np.array([rn.score(o, p, t) for o, p, m, t in rows])
    comps = [rb.components(o, p, t) for o, p, m, t in rows]
    for k in gold.NAMES:
        out["RB_" + k] = np.array([c[k] for c in comps])
    from overfit2026.stage_b_saved import qm_scores
    out.update(qm_scores(qm(), rows))
    healers = {h for h, r in HERO_ROLE_FINE.items() if r == "healer"}
    out["degen"] = np.array([float(is_degenerate(list(o))) for o, p, m, t in rows])
    out["healer"] = np.array([float(any(h in healers for h in o)) for o, p, m, t in rows])
    return out


def summarize(sc):
    res = {}
    for k, v in sc.items():
        v = np.asarray(v, float)
        res[k] = [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]
    return res


def diversity(drafts):
    from collections import Counter
    c = Counter(h for d in drafts for h in d["our"])
    tot = sum(c.values())
    p = np.array(list(c.values())) / tot
    return {"distinct": len(c), "entropy_bits": float(-(p * np.log2(p)).sum()),
            "top10_share": float(sum(v for _, v in c.most_common(10)) / tot),
            "top5": c.most_common(5)}
