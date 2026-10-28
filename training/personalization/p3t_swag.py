"""
P3 tournament, step 0: quick "swag" estimate of the player-aware drafter's
edge, from the existing one-step drafter (p3_dr_drafter: candidates = top 15
by GD and top 15 by personal term, 4 GD rollouts each, team picks with the
best assignment of heroes to players) on OOT lobbies of the tournament
window, with the current skill model (p3t_data state "cur": full_ms offset +
posterior + layer) and its combiner V = sigmoid(b0 + b1 logit WP_pop +
b2 (S_0 - S_1)) (no off-role term). WP_pop is paper 2's d2c_cumprev
ensemble with the last snapshot-era statistics (the drafter is frozen at
2026-05-22, the lobbies are later).

Per lobby (controlled team chosen at random, paired seeds):
  vs GD      controlled team player-aware vs player-blind, opponent GD
  self-play  both teams player-aware, or both player-blind
Every number here is MODEL-INTERNAL: the drafter's own value function.

Run (3090, env of remote/run_rcf_full.sh): python personalization/p3t_swag.py --lobbies 3000 --procs 14
Output: $P3_CACHE/p3t/swag.pkl.gz and $P3_RESULTS/p3t/swag.json
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3_hs_core as C
import p3t_data as TD


def payloads(n_lobbies, seed=42):
    from shared import HEROES
    import p3_dr_core as D
    L = TD.load_lobbies()
    z = np.load(os.path.join(TD.OUT, "state_cur.npz"))
    s, off, n = z["s"].astype(np.float32), z["off"], z["n"]
    assert (z["gsel"] == np.arange(len(L["replay_id"]))).all()
    rich = (n.astype(np.float64).sum(1) >= 50).reshape(-1, 10).all(1)
    cand = np.flatnonzero((L["day"] <= TD.LOBBY_END) & rich)
    rng = np.random.RandomState(seed)
    sel = np.sort(rng.choice(cand, min(n_lobbies, len(cand)), replace=False))
    print(f"eligible tournament-window lobbies (all ten with 50+ earlier games): {len(cand):,}; "
          f"using {len(sel):,}", flush=True)
    out = []
    for gi in sel:
        st = L["steps"][gi]
        rows = {}
        steps = []
        j = 0
        for h, ty, tm, r in st:
            if ty == 1:
                steps.append((int(h), 1, int(tm), int(gi) * 10 + j))
                j += 1
            else:
                steps.append((int(h), 0, int(tm), -1))
        prow = [x[3] for x in steps if x[1] == 1]
        mo, to = D.one_hots(str(L["map"][gi]), str(L["tier"][gi]))
        team_rows = {t: [x[3] for x in steps if x[1] == 1 and x[2] == t] for t in (0, 1)}
        pool = {r: n[r] > 0 for r in prow}
        out.append({"gi": int(gi), "steps": steps, "map": str(L["map"][gi]), "tier": str(L["tier"][gi]),
                    "bidx": 10 ** 6, "mo": mo, "to": to, "ctrl": int(rng.randint(2)),
                    "s": {r: s[r] for r in prow}, "off": {r: off[r].astype(float) for r in prow},
                    "pool": pool, "team_rows": team_rows,
                    "team_pool": {t: np.any([pool[r] for r in team_rows[t]], axis=0) for t in (0, 1)},
                    "winner": int(L["winner"][gi])})
    return out, HEROES


def work(args):
    import p3_dr_drafter as DR
    lob, seed = args
    c = lob["ctrl"]
    o = 1 - c
    r = {"gi": lob["gi"], "ctrl": c}
    for m in ("pers", "pop"):
        r[f"vs_gd|{m}"] = DR._run_draft(lob, seed, {c: m, o: "gd"}, "gd")
        r[f"self|{m}"] = DR._run_draft(lob, seed + 1, {0: m, 1: m}, "gd")
    # the real draft, valued by the same model
    st = {"t0": {}, "t1": {}, "bans": set(), "taken": np.zeros(90, bool)}
    for k in range(16):
        st = DR._apply(st, k, lob["steps"][k][0], lob)
    wp, V = DR._value([(st["t0"], st["t1"])], lob, DR._W["b"])
    r["real"] = {"wp": float(wp[0]), "V": float(V[0])}
    return r


def summarize(res, lobs):
    def ci(x):
        x = np.asarray(x, float)
        rng = np.random.default_rng(0)
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return [round(100 * float(x.mean()), 2), round(100 * float(np.percentile(bs, 2.5)), 2),
                round(100 * float(np.percentile(bs, 97.5)), 2)]
    side = lambda r, k, c: (r[k]["V"] if c == 0 else 1 - r[k]["V"])
    sidew = lambda r, k, c: (r[k]["wp"] if c == 0 else 1 - r[k]["wp"])
    dV = [side(r, "vs_gd|pers", r["ctrl"]) - side(r, "vs_gd|pop", r["ctrl"]) for r in res]
    dW = [sidew(r, "vs_gd|pers", r["ctrl"]) - sidew(r, "vs_gd|pop", r["ctrl"]) for r in res]
    Vp = [side(r, "vs_gd|pers", r["ctrl"]) for r in res]
    Vb = [side(r, "vs_gd|pop", r["ctrl"]) for r in res]
    Vr = [side(r, "real", r["ctrl"]) for r in res]
    # personal component of the edge: V gain minus the population-WP change on the logit scale
    out = {"lobbies": len(res),
           "vs_gd": {"V_aware": ci(Vp), "V_blind": ci(Vb), "V_real_team": ci(Vr),
                     "gain_V_pp": ci(dV), "change_WPpop_pp": ci(dW)},
           "self_play_V_team0": {"aware": ci([r["self|pers"]["V"] for r in res]),
                                 "blind": ci([r["self|pop"]["V"] for r in res])}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lobbies", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    import p3_dr_drafter as DR
    lobs, heroes = payloads(a.lobbies)
    zc = np.load(os.path.join(C.RESULTS, "rcf", "p3c", "full_ms_layer__merged.npz"))
    wc = np.asarray(zc["coef"], float)
    b = np.array([wc[0], wc[1], wc[2], 0.0])
    print(f"combiner b0, b_logitWP, b_skill, b_offrole = {np.round(b, 4).tolist()}", flush=True)
    shared = {"hero_names": np.array(heroes), "b": b, "iw0": 0.0}
    import multiprocessing as mp
    res = []
    with mp.get_context("fork").Pool(a.procs, initializer=DR._init, initargs=(shared,)) as pool:
        for i, r in enumerate(pool.imap(work, [(l, 1000 + l["gi"]) for l in lobs], chunksize=4)):
            res.append(r)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(lobs)} ({time.time() - t0:.0f}s)", flush=True)
    with gzip.open(os.path.join(TD.OUT, "swag.pkl.gz"), "wb") as f:
        pickle.dump({"res": res, "b": b}, f)
    s = summarize(res, lobs)
    s["combiner"] = b.tolist()
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    json.dump(s, open(os.path.join(C.RESULTS, "p3t", "swag.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
