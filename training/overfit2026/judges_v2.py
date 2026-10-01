"""
Composition-aware independent judges (v2, 2026-10-01).

Each v2 judge is an existing independent judge J (gN, gN_naive, RN, R17,
QM2026, QM2021) plus a three-coefficient structural correction:

    p*(own beats opp) = sigmoid( logit p_J(own, opp) + beta_J . (s(own) - s(opp)) )

s(team) = [no_healer, no_frontline, stack] (overfit2026.structure). beta_J is
fitted with J's own prediction as a fixed offset (J itself is unchanged), on
real games that neither J nor any agent trained on, with the realized outcome
as the target, controlling for player skill and pick position when the
"causal" variant is chosen (see comp_judges.py and results/comp_judges.json).
Team-order symmetry is preserved: the correction is antisymmetric in the
two teams, so symmetrized base scores stay symmetrized.

Usage as a library:
    from overfit2026 import judges_v2
    base = judges_v2.score_base(rows)            # rows = [(team0, team1, map, tier)]
    v2 = judges_v2.correct(base, rows)           # {name: P(team0 wins)}
Hooks for paper1_revision/oct2026_pool_judges.py (py:<file>:<func>):
    gN_v2, gN_naive_v2, RN_v2, QM2026_v2, consensus_v2
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from overfit2026.structure import struct_matrix

from overfit2026 import data as _odata
PARAMS = _odata.art(HERE, "results", "comp_judges.json")
CONSENSUS = ["gN", "gN_naive", "RN", "QM2026"]
BASE_JUDGES = ["gN", "gN_naive", "RN", "R17", "QM2026", "QM2021"]
from overfit2026.score import GN  # own-composition gN (comp_gn_rebuild.py)
_P = {}


def params(variant=None):
    """beta per judge for the chosen variant ('causal' by default, as fixed in
    comp_judges.json['chosen'])."""
    if "raw" not in _P:
        _P["raw"] = json.load(open(PARAMS))
    v = variant or _P["raw"]["chosen"]
    return {j: np.array(b) for j, b in _P["raw"]["beta"][v].items()}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def correct(base, rows, variant=None, judges=None):
    """base: {judge: P(team0 wins)} on rows. Returns {judge: corrected} and the
    v2 consensus (mean of the corrected consensus judges)."""
    beta = params(variant)
    dS = struct_matrix([r[0] for r in rows]) - struct_matrix([r[1] for r in rows])
    out = {}
    for j in (judges or base):
        if j not in beta or j not in base:
            continue
        out[j] = 1 / (1 + np.exp(-(_logit(base[j]) + dS @ beta[j])))
    if all(j in out for j in CONSENSUS):
        out["consensus"] = np.mean([out[j] for j in CONSENSUS], 0)
    return out


def score_base(rows, judges=BASE_JUDGES, nproc=6):
    """Unchanged v1 judges on rows, P(team0 wins), team-order symmetrized."""
    import torch
    from overfit2026 import feats, score
    from overfit2026.stage_b_saved import qm_scores
    from drift2026 import common as dcommon
    dcommon._bind_statscache_methods()
    torch.set_num_threads(min(4, nproc))
    rows = [(tuple(a), tuple(b), m, t) for a, b, m, t in rows]
    out = {}
    if "gN" in judges or "gN_naive" in judges:
        # all gN members share one deploy-stats set; the hero-identity judge
        # reads only the first 197 (statistics-free) columns
        snames = {score.deploy_stats_name(n) for n in GN}
        assert len(snames) == 1, snames
        Xf, Xs = feats.featurize(rows, score._stats(snames.pop()), nproc=nproc)
        ps = {}
        for n in GN + ["gN8_naive"]:
            m, meta = score.model(n)
            d = meta["input_dim"]
            ps[n] = feats.predict_sym(m, Xf[:, :d], Xs[:, :d], "cpu")
        out["gN"] = np.mean([ps[n] for n in GN], 0)
        out["gN_naive"] = ps["gN8_naive"]
    for key, idx in (("RN", "NODRIFT"), ("R17", "T17")):
        if key in judges:
            ri = score.realized(idx)
            out[key] = np.array([ri.score(o, p, t) for o, p, m, t in rows])
    qms = {}
    if "QM2026" in judges:
        qms["QM2026"] = score.qm()["QM2026"]
    if "QM2021" in judges:
        from paper1_revision.score_tournament import qm2021
        qms["QM2021"] = qm2021()
    if qms:
        out.update(qm_scores(qms, rows))
    return out


def _hook(name):
    def f(rows):
        base = score_base(rows, judges=[name] if name != "consensus" else CONSENSUS, nproc=4)
        return correct(base, rows)[name]
    f.__name__ = f"{name}_v2"
    return f


gN_v2 = _hook("gN")
gN_naive_v2 = _hook("gN_naive")
RN_v2 = _hook("RN")
QM2026_v2 = _hook("QM2026")
consensus_v2 = _hook("consensus")
