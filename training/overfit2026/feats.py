"""
Batch featurization + WP-model helpers shared by overfit2026 stages.

rows are (team0, team1, map, tier) tuples. featurize() returns the 283-dim
enriched WP input (base 197 + the 86 kernel-ordered enriched columns) for the
given orientation and for the team-swapped orientation, computed with the
supplied statistics object; nothing is cached globally.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch

_WP_COLS = None


def wp_cols():
    global _WP_COLS
    if _WP_COLS is None:
        from sweep_enriched_wp import compute_group_indices
        from experiment_synthetic_augmentation import ENRICHED_GROUPS
        gi = compute_group_indices()
        cols = []
        for g in ENRICHED_GROUPS:
            s, e = gi[g]
            cols.extend(range(s, e))
        _WP_COLS = np.array(cols)
    return _WP_COLS


def stats_from_json(path_or_raw, compositions=True):
    """StatsCache from a frozen-schema stats JSON (path or dict)."""
    from sweep_enriched_wp import StatsCache
    st = object.__new__(StatsCache)
    raw = json.load(open(path_or_raw)) if isinstance(path_or_raw, str) else path_or_raw
    st.hero_wr, st.hero_meta, st.hero_map_wr, st.pairwise = {}, {}, {}, {}
    for r in raw["hero_stats"]:
        st.hero_wr.setdefault(r["tier"], {})[r["hero"]] = r["win_rate"]
        st.hero_meta.setdefault(r["tier"], {})[r["hero"]] = (r["pick_rate"], r["ban_rate"])
    for r in raw["hero_map_stats"]:
        st.hero_map_wr.setdefault(r["tier"], {}).setdefault(r["map"], {})[r["hero"]] = (
            r["win_rate"], r["games"])
    for r in raw["pairwise_stats"]:
        st.pairwise.setdefault(r["tier"], {}).setdefault(r["relationship"], {}).setdefault(
            r["hero_a"], {})[r["hero_b"]] = (r["win_rate"], r["games"])
    st.comp_data = {}
    if compositions:
        st._load_compositions()
    return st


def paper_stats():
    return stats_from_json(os.path.join(TRAINING_DIR, "frozen_stats_2026-05-19.json"))


def _stats_dict(st):
    return {"hero_wr": st.hero_wr, "hero_meta": st.hero_meta,
            "hero_map_wr": st.hero_map_wr, "pairwise": st.pairwise,
            "comp_data": st.comp_data}


def _chunk(args):
    rows, sd = args
    from sweep_enriched_wp import StatsCache, extract_features, FEATURE_GROUPS
    st = object.__new__(StatsCache)
    for k, v in sd.items():
        setattr(st, k, v)
    mask = [True] * len(FEATURE_GROUPS)
    cols = wp_cols()
    F, S = [], []
    for t0, t1, gm, tier in rows:
        d = {"team0_heroes": list(t0), "team1_heroes": list(t1), "game_map": gm,
             "skill_tier": tier, "winner": 0}
        b, e = extract_features(d, st, mask)
        F.append(np.concatenate([b, e[cols]]))
        d2 = {"team0_heroes": list(t1), "team1_heroes": list(t0), "game_map": gm,
              "skill_tier": tier, "winner": 0}
        b, e = extract_features(d2, st, mask)
        S.append(np.concatenate([b, e[cols]]))
    return np.array(F, dtype=np.float32), np.array(S, dtype=np.float32)


_POOL = None


def featurize(rows, stats, nproc=48, chunk=2000):
    import multiprocessing as mp
    rows = list(rows)
    if len(rows) <= chunk:
        return _chunk((rows, _stats_dict(stats)))
    sd = _stats_dict(stats)
    parts = [(rows[i:i + chunk], sd) for i in range(0, len(rows), chunk)]
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc) as pool:
        res = pool.map(_chunk, parts)
    return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]))


def load_wp(path, dim=283, arch=(256, 128), dropout=0.3):
    from sweep_enriched_wp import WinProbEnrichedModel
    m = WinProbEnrichedModel(dim, list(arch), dropout=dropout)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def predict(model, X, device="cpu", bs=65536):
    out = []
    model = model.to(device)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.tensor(X[i:i + bs], device=device)).float().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


def predict_sym(model, Xf, Xs, device="cpu", cols=None):
    if cols is not None:
        Xf, Xs = Xf[:, cols], Xs[:, cols]
    a = predict(model, Xf, device)
    b = predict(model, Xs, device)
    return 0.5 * (a + 1.0 - b)
