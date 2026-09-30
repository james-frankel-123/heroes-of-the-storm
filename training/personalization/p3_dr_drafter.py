"""
P3 item 1: the personalized drafter (search-only, behavioral prior).

Decision for the acting player p (team t) at a pick step: candidates = p's
pool heroes still free, capped to the union of the top 15 by GD probability
and the top 15 by p's personal term s(p, .). For each candidate, R rollouts
complete the draft: real bans are replayed where the hero is still free,
every later pick is sampled from the pool-restricted GD policy (the
behavioral prior) for whichever team picks. Final drafts are valued twice
on the SAME rollouts:
  population value  WP_pop (drift-aware ensemble)
  personal value    V = sigmoid(b0 + b1 logit WP_pop + b2 (S_0 - S_1) + b3 (O_0 - O_1))
The population drafter picks argmax of mean WP_pop, the personalized one
argmax of mean V. Both are computed at every decision, so pick differences
are paired.

Runs per lobby (held-out V2 games whose ten players each have >= 50 earlier
games; the controlled team is chosen at random):
  a-gd     controlled team: personalized vs population drafter; opponent
           samples from the pool-restricted GD policy
  a-imit   same, opponent samples from the imitation model (p3_dr_imitation)
  b        self-play: both teams personalized, or both population
  real     the real draft; at each real pick both drafters rank the real
           state's candidates (for the realized-value check)
Common random numbers: the same seed per lobby for the personalized and the
population trajectory. Collapse contexts (separate mode "collapse"): a
player is inserted at a random team-0 pick step of other lobbies' real
partial drafts and both drafters choose for him.

Run (from training/):
  OMP_NUM_THREADS=1 nice -n 19 taskset -c 48-63 python3 personalization/p3_dr_drafter.py \
      [--lobbies 1000] [--real-extra 3000] [--collapse-players 300] [--procs 4]
Output: cache/dr_runs.pkl.gz
"""
import os
import sys
import gzip
import time
import pickle
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C

OUT = os.path.join(C.CACHE, "dr_runs.pkl.gz")
R = 4
TOPK = 15

_W = {}


def _init(shared):
    import torch
    torch.set_num_threads(1)
    import p3_dr_core as D
    _W.update(shared)
    _W["gd"] = D.GDPolicy(shared["hero_names"])
    _W["wp"] = D.WPEval(shared["hero_names"])
    _W["D"] = D


def _value(finals, lob, b):
    """finals: list of (t0 dict row->hero, t1 dict). Returns (wp_pop team0, V team0)."""
    wpE = _W["wp"]
    drafts = [(tuple(a.values()), tuple(c.values())) for a, c in finals]
    wp = wpE.wp(drafts, lob["bidx"], lob["map"], lob["tier"])
    S0 = np.array([sum(lob["s"][r][h] for r, h in a.items()) for a, _ in finals])
    S1 = np.array([sum(lob["s"][r][h] for r, h in c.items()) for _, c in finals])
    O0 = np.array([sum(lob["off"][r][h] for r, h in a.items()) for a, _ in finals])
    O1 = np.array([sum(lob["off"][r][h] for r, h in c.items()) for _, c in finals])
    lw = np.log(np.clip(wp, 1e-6, 1 - 1e-6) / np.clip(1 - wp, 1e-6, 1 - 1e-6))
    z = b[0] + b[1] * lw + b[2] * (S0 - S1) + b[3] * (O0 - O1)
    return wp, 1 / (1 + np.exp(-z))


def _policy_sample(states, k, lob, rng, kind):
    """Sample picks for a batch of partial states at pick step k."""
    gd = _W["gd"]
    h, ty, tm, row = lob["steps"][k]
    B = len(states)
    t0 = np.zeros((B, 90), np.float32)
    t1 = np.zeros((B, 90), np.float32)
    bans = np.zeros((B, 90), np.float32)
    avail = np.zeros((B, 90), bool)
    for i, s in enumerate(states):
        t0[i, list(s["t0"].values())] = 1
        t1[i, list(s["t1"].values())] = 1
        bans[i, list(s["bans"])] = 1
        avail[i] = lob["pool"][row] & ~s["taken"]
        if not avail[i].any():
            avail[i] = ~s["taken"]
    lp = gd.logprobs(t0, t1, bans, np.repeat(lob["mo"][None], B, 0), np.repeat(lob["to"][None], B, 0),
                     np.full(B, k, np.float32), np.ones(B, np.float32), avail)
    if kind == "imit":
        u = _W["iw0"] * lp + lob["ipers"][row][None, :]
        u = np.where(avail, u, -1e9)
        u -= u.max(1, keepdims=True)
        p = np.exp(u)
        p /= p.sum(1, keepdims=True)
    else:
        p = np.exp(lp)
        p = np.where(avail, p, 0)
        p /= p.sum(1, keepdims=True)
    c = p.cumsum(1)
    rr = rng.rand(B, 1)
    return np.minimum((c < rr).sum(1), 89)


def _apply(s, k, hero, lob):
    h, ty, tm, row = lob["steps"][k]
    s = {"t0": dict(s["t0"]), "t1": dict(s["t1"]), "bans": set(s["bans"]), "taken": s["taken"].copy()}
    if ty == 0:
        if not s["taken"][h]:
            s["bans"].add(h)
            s["taken"][h] = True
    else:
        (s["t0"] if tm == 0 else s["t1"])[row] = hero
        s["taken"][hero] = True
    return s


def _rollout(states, k0, lob, rng, opp_kind, our_team):
    """Complete each state from step k0 with the behavioral prior."""
    for k in range(k0, 16):
        h, ty, tm, row = lob["steps"][k]
        if ty == 0:
            states = [_apply(s, k, h, lob) for s in states]
        else:
            kind = opp_kind if (tm != our_team and opp_kind == "imit") else "gd"
            picks = _policy_sample(states, k, lob, rng, kind)
            states = [_apply(s, k, int(p), lob) for s, p in zip(states, picks)]
    return states


def _decide(state, k, lob, rng, opp_kind):
    """Rank candidates for the pick at step k. Returns dict with candidate
    list, mean pop / personal values from the acting team's side."""
    h, ty, tm, row = lob["steps"][k]
    free = lob["pool"][row] & ~state["taken"]
    if not free.any():
        free = ~state["taken"]
    cand = np.flatnonzero(free)
    if len(cand) > 2 * TOPK:
        gd = _W["gd"]
        t0 = np.zeros((1, 90), np.float32)
        t1 = np.zeros((1, 90), np.float32)
        bans = np.zeros((1, 90), np.float32)
        t0[0, list(state["t0"].values())] = 1
        t1[0, list(state["t1"].values())] = 1
        bans[0, list(state["bans"])] = 1
        lp = gd.logprobs(t0, t1, bans, lob["mo"][None], lob["to"][None], np.array([k], np.float32),
                         np.ones(1, np.float32), free[None])[0]
        a = cand[np.argsort(-lp[cand])[:TOPK]]
        b_ = cand[np.argsort(-lob["s"][row][cand])[:TOPK]]
        cand = np.union1d(a, b_)
    if "force" in state and state["force"] is not None and state["force"] not in cand:
        cand = np.r_[cand, state["force"]]
    starts = []
    for c in cand:
        s1 = _apply(state, k, int(c), lob)
        starts += [s1] * R
    finals = _rollout(starts, k + 1, lob, rng, opp_kind, tm)
    wp, V = _value([(f["t0"], f["t1"]) for f in finals], lob, _W["b"])
    if tm == 1:
        wp, V = 1 - wp, 1 - V
    wp = wp.reshape(len(cand), R).mean(1)
    V = V.reshape(len(cand), R).mean(1)
    return {"cand": cand, "wp": wp, "V": V, "pop": int(cand[np.argmax(wp)]), "pers": int(cand[np.argmax(V)]),
            "team": tm, "row": row, "step": k}


def _run_draft(lob, seed, modes, opp_kind):
    """modes: {team: 'pers'|'pop'|'gd'|'imit'}; returns final picks and decisions."""
    rng = np.random.RandomState(seed)
    state = {"t0": {}, "t1": {}, "bans": set(), "taken": np.zeros(90, bool)}
    decs = []
    for k in range(16):
        h, ty, tm, row = lob["steps"][k]
        if ty == 0:
            state = _apply(state, k, h, lob)
            continue
        m = modes[tm]
        if m in ("pers", "pop"):
            dd = _decide(state, k, lob, rng, opp_kind)
            decs.append(dd)
            hero = dd[m]
        else:
            hero = int(_policy_sample([state], k, lob, rng, m)[0])
        state = _apply(state, k, hero, lob)
    wp, V = _value([(state["t0"], state["t1"])], lob, _W["b"])
    return {"t0": state["t0"], "t1": state["t1"], "wp": float(wp[0]), "V": float(V[0]), "decisions": decs}


def work_lobby(args):
    lob, seed, full = args
    out = {"gi": lob["gi"], "ctrl": lob["ctrl"]}
    real_state = {"t0": {}, "t1": {}, "bans": set(), "taken": np.zeros(90, bool)}
    rng = np.random.RandomState(seed + 7)
    realdec = []
    for k in range(16):
        h, ty, tm, row = lob["steps"][k]
        if ty == 1:
            st = dict(real_state)
            st["force"] = h
            dd = _decide(st, k, lob, rng, "gd")
            dd["actual"] = int(h)
            realdec.append(dd)
        real_state = _apply(real_state, k, h, lob)
    wp, V = _value([(real_state["t0"], real_state["t1"])], lob, _W["b"])
    out["real"] = {"t0": real_state["t0"], "t1": real_state["t1"], "wp": float(wp[0]), "V": float(V[0]),
                   "decisions": realdec}
    if full:
        c = lob["ctrl"]
        o = 1 - c
        for opp in ("gd", "imit"):
            for m in ("pers", "pop"):
                out[f"a-{opp}|{m}"] = _run_draft(lob, seed, {c: m, o: opp}, opp)
        for m in ("pers", "pop"):
            out[f"b|{m}"] = _run_draft(lob, seed + 1, {0: m, 1: m}, "gd")
    return out


def work_collapse(args):
    lob, seed, row_player, prow = args
    # lob: another lobby whose team-0 pick step k is taken over by player prow
    rng = np.random.RandomState(seed)
    k = lob["k"]
    state = {"t0": {}, "t1": {}, "bans": set(), "taken": np.zeros(90, bool)}
    for j in range(k):
        state = _apply(state, j, lob["steps"][j][0], lob)
    dd = _decide(state, k, lob, rng, "gd")
    return {"player": row_player, "pop": dd["pop"], "pers": dd["pers"]}


def lobby_payload(L, T, gi, ipers, ctrl, bidx_all, extra_rows=None):
    import p3_dr_core as D
    st = L["steps"][gi]
    rows = [int(s[3]) for s in st if s[1] == 1]
    if extra_rows:
        rows += extra_rows
    pos = [T["pos"][r] for r in rows]
    mo, to = D.one_hots(str(L["map"][gi]), str(L["tier"][gi]))
    return {"gi": int(gi), "steps": [tuple(int(x) for x in s) for s in st], "map": str(L["map"][gi]),
            "tier": str(L["tier"][gi]), "bidx": int(bidx_all[L["g"][gi]]), "mo": mo, "to": to,
            "ctrl": int(ctrl),
            "s": {r: T["s"][p] for r, p in zip(rows, pos)}, "off": {r: T["off"][p].astype(float) for r, p in zip(rows, pos)},
            "pool": {r: T["pool"][p] for r, p in zip(rows, pos)},
            "ipers": {r: ipers[r] for r in rows}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lobbies", type=int, default=1000)
    ap.add_argument("--real-extra", type=int, default=3000)
    ap.add_argument("--collapse-players", type=int, default=300)
    ap.add_argument("--contexts", type=int, default=30)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    import p3_dr_core as D
    import p3_dr_imitation as I
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    L = D.load_lobbies()
    T = D.load_personal()
    w = np.load(C.WP)
    bidx_all = w["build_idx"][np.argsort(w["replay_ids"])]
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    prow = L["steps"][:, :, 3]
    picks = prow[prow >= 0].reshape(len(L["replay_id"]), 10)
    npos = np.vectorize(lambda r: T["pos"][int(r)])(picks)
    est = (T["n_p"][npos] >= 50).all(1)
    cand = np.flatnonzero((L["day"] >= med) & est)
    rng = np.random.RandomState(42)
    sel = rng.choice(cand, a.lobbies + a.real_extra, replace=False)
    full_set, extra_set = sel[:a.lobbies], sel[a.lobbies:]
    print(f"eligible V2 lobbies (all ten players >= 50 games): {len(cand):,}; full {len(full_set)}, "
          f"realized-only {len(extra_set)}", flush=True)
    # imitation personal part (state-independent) for needed rows
    iw = np.load(I.OUT)["w"]
    need_rows = np.unique(picks[sel].ravel())
    coll_players = rng.choice(np.unique(picks[full_set].ravel()), a.collapse_players, replace=False)
    need_rows = np.unique(np.r_[need_rows, coll_players])
    rec = I.recency_features(d, need_rows)
    rp = {int(r): i for i, r in enumerate(need_rows)}
    fake = {"row": need_rows, "recpos": np.arange(len(need_rows)),
            "lp": np.zeros((len(need_rows), 90), np.float32)}
    Xp = I.feature_tensor(fake, T, rec, meta)
    ipers_all = (Xp[:, :, 1:] * iw[1:]).sum(-1)
    ipers = {int(r): ipers_all[i] for i, r in enumerate(need_rows)}
    b = D.combiner()
    shared = {"hero_names": d["hero_names"], "b": b, "iw0": float(iw[0])}
    tasks = []
    for gi in full_set:
        tasks.append((lobby_payload(L, T, gi, ipers, rng.randint(2), bidx_all), int(1000 + gi), True))
    for gi in extra_set:
        tasks.append((lobby_payload(L, T, gi, ipers, 0, bidx_all), int(1000 + gi), False))
    # collapse contexts: other lobbies' team-0 pick steps, player substituted
    ctasks = []
    for pr in coll_players:
        for j in range(a.contexts):
            gi = int(rng.choice(full_set))
            st = L["steps"][gi]
            ks = [k for k in range(16) if st[k][1] == 1 and st[k][2] == 0]
            k = int(rng.choice(ks))
            pay = lobby_payload(L, T, gi, ipers, 0, bidx_all, extra_rows=[int(pr)])
            steps = [list(s) for s in pay["steps"]]
            old = steps[k][3]
            steps[k][3] = int(pr)
            # the substituted player takes the slot's place for valuation
            pay["steps"] = [tuple(s) for s in steps]
            pay["k"] = k
            ctasks.append((pay, int(5000 + j), int(pr), old))
    print(f"tasks: {len(tasks)} lobbies, {len(ctasks)} collapse contexts ({time.time() - t0:.0f}s)", flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    res, cres = [], []
    with ctx.Pool(a.procs, initializer=_init, initargs=(shared,)) as pool:
        for i, r in enumerate(pool.imap_unordered(work_lobby, tasks, chunksize=4)):
            res.append(r)
            if (i + 1) % 200 == 0:
                print(f"  lobbies {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
        for i, r in enumerate(pool.imap_unordered(work_collapse, ctasks, chunksize=16)):
            cres.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  collapse {i + 1}/{len(ctasks)} ({time.time() - t0:.0f}s)", flush=True)
    with gzip.open(OUT, "wb") as f:
        pickle.dump({"lobbies": res, "collapse": cres, "full_set": full_set, "extra_set": extra_set,
                     "coll_players": coll_players}, f)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
