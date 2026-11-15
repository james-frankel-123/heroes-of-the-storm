"""
Family g: worlds drawn from the drafter's own network class. Each world is one member of paper 1's
20-member specification-diverse leak-free WP ensemble (site/models/ensemble_lcb; the pessimistic
leaf of mcts_lcb / mcts_lcb2, trained by train_ensemble_site.py on data halves), symmetrized like
the agent's leaf. Twenty worlds, no dials. Scoring the ensemble-LCB agents here is self-grading:
their leaf is the lower confidence bound over exactly these worlds.

values(rows) -> (20, n) P(team 0 wins) per member, cached by tag in site/cache/ens_<tag>.npz.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TRAINING_DIR)
import data as WD  # noqa: E402

N_MEMBERS = 20


def members():
    import torch
    from paper1_revision import core
    from paper1_revision.deferred import _ensemble_specs
    from sweep_enriched_wp import WinProbEnrichedModel
    mdir = os.environ.get("P1_ENSEMBLE_DIR", os.path.join(core.MODEL_DIR, "ensemble_lcb"))
    out = []
    for sp in _ensemble_specs():
        cols = core.group_cols(sp["groups"])
        m = WinProbEnrichedModel(len(cols), sp["arch"], dropout=0.3)
        m.load_state_dict(torch.load(os.path.join(mdir, sp["name"] + ".pt"), map_location="cpu", weights_only=True))
        m.eval()
        out.append((sp["name"], m, cols))
    assert len(out) == N_MEMBERS
    return out


def values(rows, tag, chunk=200000):
    p = os.path.join(WD.CACHE, f"ens_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["p"]
    import torch
    from paper1_revision import core
    torch.set_num_threads(int(os.environ.get("WORLDS_THREADS", "8")))
    st = core.load_stats("deploy")
    mem = members()
    out = np.zeros((N_MEMBERS, len(rows)), np.float32)
    for i in range(0, len(rows), chunk):
        Xf, Xs = core.featurize(rows[i:i + chunk], st, nproc=int(os.environ.get("P1R_NPROC", "3")))
        with torch.no_grad():
            for j, (_, m, cols) in enumerate(mem):
                a = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
                b = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
                out[j, i:i + len(a)] = 0.5 * (a + 1 - b)
        del Xf, Xs
    np.savez(p, p=out, names=np.array([n for n, _, _ in mem]))
    return out


def rows_of(D):
    from shared import HEROES
    maps, tiers = WD.meta()
    return [([HEROES[h] for h in a], [HEROES[h] for h in b], maps[m] if m < len(maps) else "?",
             tiers[t] if t < len(tiers) else "?") for a, b, m, t in zip(D["T0"], D["T1"], D["m"], D["t"])]
