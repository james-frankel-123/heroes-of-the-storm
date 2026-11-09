"""
Ban-row outcomes of the Q-learning transition caches. The transition builders
(experiment_cql_draft / experiment_cql_enriched replay_to_transitions) took the
acting team of a ban from player_slot <= 4, but a ban's player_slot is the team
(1 or 2), so every ban row carried team 0's result. The builders are fixed; this
script fixes existing caches in place without rebuilding the states.

Each cache stores 32 rows per replay (16 steps, then the team-swap mirror), and
every row's outcome is its actor's result. Pick rows were always right, so a ban
row takes the outcome of its own side's first pick in the same 16-row half (the
side follows the fixed draft order: the team that bans first also picks first).

Usage (from training/, on the box that holds the cache):
  python3 paper1_revision/banfix.py patch <cache_dir> [<cache_dir> ...]
      patches outcomes.dat and packed/outcomes.npy (originals kept as *.banbug),
      writes <cache_dir>/banfix.json
  python3 paper1_revision/banfix.py validate <naive test cache_dir>
      regenerates the test split's outcomes with the fixed builder and compares
      them (and the actions) to the patched cache, row by row
"""
import os
import sys
import json
import shutil

import numpy as np

SIDE = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
BAN = np.array([0, 1, 2, 3, 9, 10])
FIRST_PICK = {0: 4, 1: 5}


def side_fix(outs):
    """outs: (n,) array, 16-row halves. Returns (fixed copy, stats)."""
    h = outs.reshape(-1, 16)
    new = h.copy()
    for k in BAN:
        new[:, k] = h[:, FIRST_PICK[SIDE[k]]]
    picks = [k for k in range(16) if k not in set(BAN)]
    incons = np.zeros(len(h), bool)
    for s in (0, 1):
        ks = [k for k in picks if SIDE[k] == s]
        incons |= (h[:, ks] != h[:, ks[:1]]).any(1)
    same = (h[:, 4] == h[:, 5])
    stats = {"rows": int(h.size), "ban_rows": int(len(h) * len(BAN)),
             "ban_rows_changed": int((new[:, BAN] != h[:, BAN]).sum()),
             "halves_pick_outcomes_inconsistent": int(incons.sum()),
             "halves_both_sides_same_outcome": int(same.sum())}
    return new.reshape(-1), stats


def _check_steps(d, n):
    """Rows must be 16-step halves in order; checked on the packed step index or sampled states."""
    p = os.path.join(d, "packed", "step.npy")
    if os.path.exists(p):
        st = np.load(p, mmap_mode="r")
        assert len(st) == n and (np.asarray(st).reshape(-1, 16) == np.arange(16)).all(), "rows not in 16-step halves"
        return "packed step index, all rows"
    meta = json.load(open(os.path.join(d, "meta.json")))
    S = np.memmap(os.path.join(d, "states.dat"), dtype=np.float32, mode="r", shape=(n, meta["state_dim"]))
    rows = np.sort(np.random.default_rng(0).choice(n, 20000, replace=False))
    got = np.rint(np.asarray(S[rows, 287]) * 15).astype(int)
    assert (got == rows % 16).all(), "rows not in 16-step halves (sampled)"
    return "states step column, 20000 sampled rows"


def patch(d):
    marker = os.path.join(d, "banfix.json")
    if os.path.exists(marker):
        print("already patched", d)
        return
    meta = json.load(open(os.path.join(d, "meta.json")))
    n = meta["n"]
    assert n % 32 == 0, n
    rec = {"how": "ban row outcome = outcome of its side's first pick in the same 16-row half", "n": n,
           "steps_checked": _check_steps(d, n)}
    src = os.path.join(d, "outcomes.dat")
    if os.path.exists(src):
        if not os.path.exists(src + ".banbug"):
            shutil.copy2(src, src + ".banbug")
        mm = np.memmap(src, dtype=np.float32, mode="r+", shape=(n,))
        new, st = side_fix(np.asarray(mm[:]))
        mm[:] = new
        mm.flush()
        del mm
        rec["outcomes.dat"] = st
    pk = os.path.join(d, "packed", "outcomes.npy")
    if os.path.exists(pk):
        if not os.path.exists(pk + ".banbug"):
            shutil.copy2(pk, pk + ".banbug")
        o = np.load(pk)
        new, st = side_fix(o)
        np.save(pk + ".tmp.npy", new.astype(o.dtype))
        os.replace(pk + ".tmp.npy", pk)
        rec["packed/outcomes.npy"] = st
        if os.path.exists(src):
            a = np.memmap(src, dtype=np.float32, mode="r", shape=(n,))
            rec["packed_equals_source"] = bool((np.asarray(a[:]) == new).all())
            assert rec["packed_equals_source"]
    json.dump(rec, open(marker, "w"), indent=1)
    print(json.dumps(rec, indent=1))


def validate(d):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    from rerun2026 import common
    common.setup()
    from experiment_cql_draft import replay_to_transitions
    _, test = common.load_split()
    acts, outs = [], []
    for r in test:
        t = replay_to_transitions(r)
        acts += [x["action"] for x in t]
        outs += [x["outcome"] for x in t]
    acts, outs = np.array(acts), np.array(outs, np.float32)
    meta = json.load(open(os.path.join(d, "meta.json")))
    n = meta["n"]
    ca = np.memmap(os.path.join(d, "actions.dat"), dtype=np.int64, mode="r", shape=(n,))
    co = np.memmap(os.path.join(d, "outcomes.dat"), dtype=np.float32, mode="r", shape=(n,))
    assert len(acts) == n, (len(acts), n)
    rec = {"rows": n, "actions_match": float((np.asarray(ca[:]) == acts).mean()),
           "outcomes_match": float((np.asarray(co[:]) == outs).mean()),
           "ban_outcomes_match": float((np.asarray(co[:]).reshape(-1, 16)[:, BAN] == outs.reshape(-1, 16)[:, BAN]).mean())}
    json.dump(rec, open(os.path.join(d, "banfix_validate.json"), "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    for d in sys.argv[2:]:
        patch(d) if cmd == "patch" else validate(d)
