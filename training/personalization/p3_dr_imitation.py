"""
P3 item 2: DraftRec-style imitation model (what will this player pick?).

For every pick in the post-cutoff lobbies, candidates are all heroes not yet
picked or banned. Score of candidate c for acting player p:

  u_c = w . [GD log-prob(c | draft state), log1p(n_pc), [n_pc = 0],
             EWMA pick share of c over p's last games (half-life 20 and 100
             games), log1p(days since p last played c), [never played],
             log(share of p's games in c's fine role)]
  P(c) = softmax(u)

GD log-prob: the paper-1 generic-draft models (5-model average) given the
real draft state, masked to untaken heroes. Player features use only games
strictly earlier in play order (game_date, replay_id); counts n_pc are
through the previous day (the personal tables' convention). Weights: full-
batch conditional-logit fit on V1 picks; scored on V2 picks. Baselines: GD
alone, frequency alone (EWMA-20 share), uniform over the player's pool.

Run (from training/):
  OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nice -n 19 taskset -c 48-63 \
    python3 personalization/p3_dr_imitation.py
Outputs: cache/dr_imitation.npz (weights + per-pick features for reuse),
results/p3_dr_imitation.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from numba import njit, prange
from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_dr_core as D

GT = os.path.join(C.CACHE, "gametime_2024q2.npz")
OUT = os.path.join(C.CACHE, "dr_imitation.npz")
FEATS = ["gd_logp", "log1p_n", "never_n", "ewma20", "ewma100", "log1p_days_since", "never_played",
         "log_fine_share"]


@njit(parallel=True, cache=True)
def _recency(starts, ends, day, hero, qslot, out_e20, out_e100, out_last):
    a20 = 1 - np.exp(-np.log(2) / 20.0)
    a100 = 1 - np.exp(-np.log(2) / 100.0)
    for p in prange(starts.shape[0]):
        e20 = np.zeros(NUM_HEROES)
        e100 = np.zeros(NUM_HEROES)
        w20 = 0.0
        w100 = 0.0
        last = np.full(NUM_HEROES, -1)
        for r in range(starts[p], ends[p]):
            q = qslot[r]
            if q >= 0:
                for h in range(NUM_HEROES):
                    out_e20[q, h] = e20[h] / w20 if w20 > 0 else 0.0
                    out_e100[q, h] = e100[h] / w100 if w100 > 0 else 0.0
                    out_last[q, h] = day[r] - last[h] if last[h] >= 0 else -1
            h = hero[r]
            for k in range(NUM_HEROES):
                e20[k] *= 1 - a20
                e100[k] *= 1 - a100
            e20[h] += a20
            e100[h] += a100
            w20 = (1 - a20) * w20 + a20
            w100 = (1 - a100) * w100 + a100
            last[h] = day[r]


def recency_features(d, qrows):
    z = np.load(GT)
    o = np.argsort(z["replay_ids"])
    ts = z["ts"][o][np.searchsorted(z["replay_ids"][o], d["replay_id"])]
    srt = np.lexsort((d["replay_id"], ts, d["pid"]))
    pid = d["pid"][srt]
    brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[brk[1:], len(pid)]
    qslot = np.full(len(srt), -1, np.int64)
    qpos = np.full(len(d["pid"]), -1, np.int64)
    qpos[qrows] = np.arange(len(qrows))
    qslot[:] = qpos[srt]
    nq = len(qrows)
    e20 = np.zeros((nq, NUM_HEROES), np.float32)
    e100 = np.zeros((nq, NUM_HEROES), np.float32)
    last = np.zeros((nq, NUM_HEROES), np.float32)
    _recency(brk.astype(np.int64), ends.astype(np.int64), d["day"][srt].astype(np.int64),
             d["hero"][srt].astype(np.int64), qslot, e20, e100, last)
    return e20, e100, last


def pick_states(L, gd, idx):
    """For games idx: per pick (game, step): GD log-probs over untaken heroes,
    untaken mask, acting row, actual hero, team, step index."""
    out = {"lp": [], "avail": [], "row": [], "hero": [], "game": [], "step": [], "team": []}
    B = 4096
    buf = []
    for gi in idx:
        st = L["steps"][gi]
        mo, to = D.one_hots(str(L["map"][gi]), str(L["tier"][gi]))
        t0 = np.zeros(NUM_HEROES, np.float32)
        t1 = np.zeros(NUM_HEROES, np.float32)
        bans = np.zeros(NUM_HEROES, np.float32)
        for k, (h, ty, tm, row) in enumerate(st):
            if ty == 1:
                avail = (t0 + t1 + bans) == 0
                buf.append((t0.copy(), t1.copy(), bans.copy(), mo, to, k, avail, row, h, gi, tm))
                (t0 if tm == 0 else t1)[h] = 1
            else:
                bans[h] = 1
    for s in range(0, len(buf), B):
        ch = buf[s:s + B]
        lp = gd.logprobs(np.array([c[0] for c in ch]), np.array([c[1] for c in ch]),
                         np.array([c[2] for c in ch]), np.array([c[3] for c in ch]),
                         np.array([c[4] for c in ch]), np.array([c[5] for c in ch], np.float32),
                         np.ones(len(ch), np.float32), np.array([c[6] for c in ch]))
        out["lp"].append(lp.astype(np.float32))
        out["avail"].append(np.array([c[6] for c in ch]))
        for c in ch:
            out["row"].append(c[7])
            out["hero"].append(c[8])
            out["game"].append(c[9])
            out["step"].append(c[5])
            out["team"].append(c[10])
    return {k: (np.concatenate(v) if k in ("lp", "avail") else np.array(v)) for k, v in out.items()}


def feature_tensor(P, T, rec, meta):
    """(n_picks, 90, n_feats) for the picks in P."""
    tpos = np.array([T["pos"][int(r)] for r in P["row"]])
    n = T["n"][tpos]
    n_p = T["n_p"][tpos]
    e20, e100, last = rec
    rp = P["recpos"]
    fine = meta["fine"]
    fcnt = np.stack([n[:, fine == f].sum(1) for f in range(len(meta["fine_names"]))], 1)
    fshare = (fcnt[:, fine] + 1) / (n_p[:, None] + len(meta["fine_names"]))
    ls = last[rp]
    X = np.stack([P["lp"], np.log1p(n), (n == 0).astype(np.float32), e20[rp], e100[rp],
                  np.log1p(np.maximum(ls, 0)), (ls < 0).astype(np.float32), np.log(fshare)], -1)
    return X.astype(np.float32)


def fit(X, y, avail, iters=100):
    ok = avail[np.arange(len(y)), y]
    Xt = torch.from_numpy(X[ok])
    yt = torch.from_numpy(y[ok])
    mt = torch.from_numpy(avail[ok])
    w = torch.zeros(X.shape[-1], dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        w[0] = 1.0
    opt = torch.optim.LBFGS([w], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        u = (Xt * w).sum(-1).masked_fill(~mt, -1e9)
        loss = torch.nn.functional.cross_entropy(u, yt)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()


def evaluate(u, y, avail):
    u = np.where(avail, u, -1e9)
    top = np.argsort(-u, axis=1)
    lse = np.log(np.exp(u - u.max(1, keepdims=True)).sum(1)) + u.max(1)
    ll = u[np.arange(len(y)), y] - lse
    ok = avail[np.arange(len(y)), y]
    r = {f"top{k}": float((top[:, :k] == y[:, None]).any(1).mean()) for k in (1, 3, 5, 10)}
    r["mean_loglik"] = float(ll[ok].mean())
    r["share_real_pick_marked_taken"] = float(1 - ok.mean())
    return r


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    L = D.load_lobbies()
    T = D.load_personal()
    gd = D.GDPolicy(d["hero_names"])
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    rng = np.random.RandomState(0)
    v1 = np.flatnonzero(L["day"] < med)
    v2 = np.flatnonzero(L["day"] >= med)
    fit_games = rng.choice(v1, min(25000, len(v1)), replace=False)
    test_games = rng.choice(v2, min(25000, len(v2)), replace=False)
    P1 = pick_states(L, gd, fit_games)
    P2 = pick_states(L, gd, test_games)
    print(f"pick states: fit {len(P1['row']):,}, test {len(P2['row']):,} ({time.time() - t0:.0f}s)", flush=True)
    qrows = np.unique(np.r_[P1["row"], P2["row"]])
    rec = recency_features(d, qrows)
    qpos = {int(r): i for i, r in enumerate(qrows)}
    for P in (P1, P2):
        P["recpos"] = np.array([qpos[int(r)] for r in P["row"]])
    X1 = feature_tensor(P1, T, rec, meta)
    X2 = feature_tensor(P2, T, rec, meta)
    out = {"features": FEATS, "n_fit_picks": int(len(P1["row"])), "n_test_picks": int(len(P2["row"]))}
    w = fit(X1, P1["hero"], P1["avail"])
    out["weights"] = dict(zip(FEATS, w.tolist()))
    print("weights", out["weights"], flush=True)
    res = {"imitation (GD + personal history)": evaluate((X2 * w).sum(-1), P2["hero"], P2["avail"]),
           "GD alone (non-personal)": evaluate(X2[:, :, 0], P2["hero"], P2["avail"]),
           "frequency alone (EWMA-20 share)": evaluate(np.log(X2[:, :, 3] + 1e-4), P2["hero"], P2["avail"])}
    wp_only = fit(X1[:, :, 1:], P1["hero"], P1["avail"])
    res["personal history alone (fitted, no GD)"] = evaluate((X2[:, :, 1:] * wp_only).sum(-1), P2["hero"], P2["avail"])
    pool = X2[:, :, 2] == 0
    res["uniform over player's pool"] = evaluate(np.where(pool, 0.0, -30.0), P2["hero"], P2["avail"])
    out["test"] = res
    # by pick position within team
    tp = np.zeros(len(P2["row"]), np.int64)
    key = P2["game"] * 2 + P2["team"]
    o = np.lexsort((P2["step"], key))
    ks = key[o]
    brk = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    tp[o] = np.arange(len(o)) - np.repeat(brk, np.diff(np.r_[brk, len(o)]))
    out["test_by_team_pick"] = {}
    for q in range(5):
        s = tp == q
        out["test_by_team_pick"][f"pick {q + 1}"] = {
            "imitation_top1": evaluate((X2[s] * w).sum(-1), P2["hero"][s], P2["avail"][s])["top1"],
            "gd_top1": evaluate(X2[s][:, :, 0], P2["hero"][s], P2["avail"][s])["top1"]}
    for k, v in res.items():
        print(f"  {k:42s} " + " ".join(f"{a} {b:.4f}" for a, b in v.items()), flush=True)
    print(out["test_by_team_pick"], flush=True)
    np.savez(OUT, w=w, games=test_games, rows=P2["row"], hero=P2["hero"], step=P2["step"],
             game=P2["game"], team=P2["team"], u=np.where(P2["avail"], (X2 * w).sum(-1), -1e9).astype(np.float32))
    with open(os.path.join(C.RESULTS, "p3_dr_imitation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
