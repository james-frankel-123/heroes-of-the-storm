"""Export a fitted scilearn Model (training/scilearn/fit.py) to a leafpack (FRAMING 11.11).

  python -m training.scilearn.leaf.export <model path without suffix> <out dir> [--build <name>]

Draft-only terms are exported; the personal block (decay terms) has no place in draft
search and is dropped. Tier-indexed products and context-dependent indicator rules that
the leaf format cannot hold are refused rather than silently dropped. `verify()` checks
the leaf against Model.logit on random complete drafts.
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

from training.scilearn import fit as F
from training.scilearn import terms as TM
from training.scilearn.leaf.leaf import LeafModel

ROLE_CODES = 6 ** 6


def _role_codes():
    """All role-count vectors of a 5-hero team, with their codes."""
    out = []
    for c in itertools.product(range(6), repeat=6):
        if sum(c) == 5:
            out.append((np.array(c), int(sum(ci * 6 ** r for r, ci in enumerate(c)))))
    return out


def export(model, meta, kit=None, axes=None, build=None, kappa=0.0, card_hashes=None):
    H, M, T = len(meta["heroes"]), len(meta["maps"]), len(meta["tiers"])
    axes = list(axes or [])
    K = len(axes) or 12
    roles = meta["roles"]
    b_idx = model.fit_info["max_fit_build"] if build is None else meta["builds"].index(build)
    hero_ctx = np.zeros((H, M, T)); hero_ctx_var = np.zeros((H, M, T))
    pair = np.zeros((H, H)); pair_var = np.zeros((H, H))
    cross = np.zeros((H, H)); cross_var = np.zeros((H, H))
    role_table = np.zeros(ROLE_CODES); role_var = np.zeros(ROLE_CODES)
    map_axis = np.zeros((M, K)); W = np.zeros((K, K))
    knots = np.tile(np.linspace(0, 1, 3), (K, 1)); vals = np.zeros((K, 3))
    shapes = {}
    rules, term_ids, dropped = [], [], []
    mclass = np.array([1 - int(m in TM.LANES2) for m in meta["maps"]])
    codes = _role_codes()

    for i, spec in enumerate(model.specs):
        o0, o1 = int(model.offs[i]), int(model.offs[i + 1])
        beta, var, names = model.beta[o0:o1], model.var[o0:o1], model.names[o0:o1]
        st = model.states[i] or {}
        t = spec["type"]
        term_ids.append(spec.get("name", f"t{i}"))
        if t == "lookup":
            keys = spec["keys"]
            if keys == ["hero"]:
                hero_ctx += beta[:, None, None]; hero_ctx_var += var[:, None, None]
            else:
                nc = {"map": M, "map_class": 2, "tier": T}[keys[1]]
                bb, vv = beta.reshape(H, nc), var.reshape(H, nc)
                if keys[1] == "map":
                    hero_ctx += bb[:, :, None]; hero_ctx_var += vv[:, :, None]
                elif keys[1] == "map_class":
                    hero_ctx += bb[:, mclass][:, :, None]; hero_ctx_var += vv[:, mclass][:, :, None]
                else:
                    hero_ctx += bb[:, None, :]; hero_ctx_var += vv[:, None, :]
        elif t == "state":
            Bn = len(meta["builds"])
            bb, vv = beta.reshape(H, Bn)[:, b_idx], var.reshape(H, Bn)[:, b_idx]
            hero_ctx += bb[:, None, None]; hero_ctx_var += vv[:, None, None]
        elif t == "multiset":
            for c, bt, vr in zip(st["vocab"], beta, var):
                role_table[c] += bt; role_var[c] += vr
        elif t in ("indicator", "rule"):
            preds = spec["predicates"]
            if all(p["field"] == "role_count" for p in preds):
                for cnt, code in codes:
                    ok = all(TM.OPS[p.get("op", "==")](cnt[roles.index(p["key"])], p["value"]) for p in preds)
                    if ok:
                        role_table[code] += beta[0]; role_var[code] += var[0]
            else:
                rules.append({"id": spec.get("name", f"t{i}"), "beta": float(beta[0]),
                              "predicates": [{"field": p["field"], "key": p.get("key"), "op": p.get("op", "=="),
                                              "value": p["value"]} for p in preds]})
        elif t == "pair_family":
            for c, bt, vr in zip(st["vocab"], beta, var):
                a, b = divmod(int(c), H)
                pair[a, b] += bt; pair[b, a] += bt; pair_var[a, b] += vr; pair_var[b, a] += vr
        elif t == "cross_family":
            for c, bt, vr in zip(st["vocab"], beta, var):
                lo, hi = divmod(int(c), H)
                cross[lo, hi] += bt; cross_var[lo, hi] += vr
        elif t == "attr_sum":
            k = axes.index(spec["axis"])
            kn = np.asarray(st["knots"], float)
            slopes = np.cumsum(beta[::-1])[::-1] if st["shape"] == "concave_inc" else beta
            v = np.concatenate([[0.0], np.cumsum(slopes * np.diff(kn))])
            if k in shapes:
                raise ValueError(f"two attr_sum terms on axis {spec['axis']}; merge them before export")
            shapes[k] = (kn, v)
        elif t == "product":
            k = axes.index(spec["axis"])
            if spec["by"] == "map_class":
                map_axis[:, k] += beta[mclass]
            elif spec["by"] == "map":
                map_axis[:, k] += beta
            else:
                raise ValueError("tier products are not representable in a leafpack")
        elif t == "bilinear":
            for e, bt in zip(spec["entries"], beta):
                W[axes.index(e["a"]), axes.index(e["b"])] += bt
        elif t == "decay":
            dropped.append(spec.get("name"))
        else:
            raise ValueError(f"term type {t} has no leaf export")
    if shapes:
        nk = max(len(kn) for kn, _ in shapes.values())
        knots = np.zeros((K, nk)); vals = np.zeros((K, nk))
        for k in range(K):
            if k in shapes:
                kn, v = shapes[k]
                xs = np.linspace(kn[0], kn[-1], nk) if len(kn) != nk else kn
                knots[k], vals[k] = xs, np.interp(xs, kn, v)
            else:
                knots[k], vals[k] = np.linspace(0, 1, nk), 0.0
    arrays = {"hero_ctx": hero_ctx, "hero_ctx_var": hero_ctx_var, "pair_within": pair,
              "pair_within_var": pair_var, "cross": cross, "cross_var": cross_var,
              "role_table": role_table, "role_table_var": role_var,
              "kit": kit if kit is not None else np.zeros((H, K)), "map_axis": map_axis, "W": W,
              "shape_knots": knots, "shape_vals": vals,
              "hero_role": np.asarray(meta["hero_roles"], np.int64)}
    lmeta = {"version": 1, "model_id": model.model_hash()[:16], "heroes": meta["heroes"],
             "maps": meta["maps"], "tiers": meta["tiers"], "roles": roles, "K": K,
             "axes": axes or [f"k{i}" for i in range(K)], "kappa": kappa, "term_ids": term_ids,
             "card_hashes": card_hashes or [], "rules": rules, "build": meta["builds"][b_idx],
             "dropped_terms": dropped}
    return arrays, lmeta


def verify(model, meta, kit, axes, arrays, lmeta, n=2000, seed=0):
    """Max |leaf logit - model logit| on random complete drafts (decay terms excluded)."""
    rng = np.random.default_rng(seed)
    H = len(meta["heroes"])
    drafts = np.stack([rng.choice(H, 10, replace=False) for _ in range(n)])
    lm = LeafModel(arrays, lmeta)
    worst = 0.0
    keep = [i for i, s in enumerate(model.specs) if s["type"] != "decay"]
    for m_ in range(len(meta["maps"])):
        for t_ in range(len(meta["tiers"])):
            g = {"heroes": drafts, "map": np.full(n, m_), "tier": np.full(n, t_),
                 "build": np.full(n, meta["builds"].index(lmeta["build"])), "game_ts": np.zeros(n, np.int64),
                 "first_seen_ts": np.zeros(n, np.int64), "replay_id": np.arange(n)}
            mm = dict(meta); mm["kit_axes"] = axes
            ctx = TM.Ctx(g, mm, kit=kit, players={"player_key": np.full((n, 10), -1), "party": np.zeros((n, 10), np.int64)})
            X = model.design(ctx)
            cols = np.concatenate([np.arange(model.offs[i], model.offs[i + 1]) for i in keep])
            eta_model = X[:, cols] @ model.beta[cols]
            eta_leaf, _, _ = lm.evaluate(drafts, m_, t_)
            worst = max(worst, float(np.abs(eta_model - eta_leaf).max()))
    return worst


def save(arrays, lmeta, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "leafpack.npz", **arrays)
    (out / "leafpack.json").write_text(json.dumps(lmeta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    from training.scilearn import data as D
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--build")
    a = ap.parse_args()
    _, meta, _, _ = D.load(tranches=["Dpre"])
    kit, axes = D.load_kit(meta)
    mm = dict(meta); mm["kit_axes"] = axes
    model = F.Model.load(Path(a.model), mm)
    arrays, lmeta = export(model, meta, kit, axes, a.build)
    err = verify(model, meta, kit, axes, arrays, lmeta, n=300)
    save(arrays, lmeta, a.out)
    print(json.dumps({"out": a.out, "max_abs_logit_error": err, "dropped": lmeta["dropped_terms"]}))
