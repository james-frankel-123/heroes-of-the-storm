"""
P3 tournament: a DraftRec opponent model inside the search (coordinator,
2026-10-20).

The search's chance-node opponent and rollout policy is GD, or GD times the
opponents' comfort (p3t_oppmodel.py, cfg[6] = 2). This module replaces the
comfort term with the rebuilt DraftRec (p3t_draftrec.py), whose pick head
reads each opponent player's own history. A transformer cannot run inside
the kernel, so the model is distilled to a static per-decision bias: at the
decision state (the search root, prefix acts[:k]) DraftRec is run once on the
host, and every later opponent step s gets one row of the kernel's
opponent-bias tensor (cuda_explore, opp_personalize):

  opponent pick at s:  softmax_valid(w0 log p_GD(state) + lp log q_s(h)),
                       q_s = DraftRec's team mixture for the opponent's pick
                       turn of step s (its not-yet-assigned players, each
                       conditioned on his own history), read at the root;
  opponent ban at s:   softmax_valid(w0 log p_GD(state) + lb log r_s(h)),
                       r_s = DraftRec's mixture for OUR team's next pick turn
                       after s (bans aimed at what we would pick).

p_GD is evaluated by the kernel at the in-tree state; q and r are fixed at the
root (heroes taken in the tree are masked by the kernel). w0, lp, lb are fit
by maximum likelihood on real picks and bans, in the same "from the root"
protocol: for step s the root is the latest earlier step of the other team.

Fit lobbies: 3,000 tournament-window lobbies (2026-05-23 to 06-30) outside the
tournament, probe and tune lobby sets; real actions only, no outcome, nothing
from the judge window. DraftRec itself trains on games through 2026-05-22.
90% of the fit lobbies fit (w0, lp, lb), 10% (by lobby) are held out for the
reported top-1 and NLL, next to GD and GD x comfort on the same actions.

Stages (from training/, run_rcf_full env):
  lobbies (CPU)  $P3_CACHE/p3t/oppdr_lobbies.npz
  hist    (CPU)  DraftRec history tensors: draftrec/hist_oppfit.npz (fit lobbies)
                 and draftrec/hist_tune.npz (tune lobbies)
  fit     (GPU)  results/p3t/opp_dr.json
Inference: rows(ctx, t, k, w) (used by p3t_systems.act_mcts_x, x["opp"] = "dr").
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

import p3_hs_core as C
import p3t_data as TD

DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
PICK_STEPS = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]
PICK_TEAM = [DRAFT_TEAM[k] for k in PICK_STEPS]
TURN = {k: j for j, k in enumerate(PICK_STEPS)}
H = 90
N_FIT = 3000
DR_FLOOR = 1e-6
GD_FLOOR = 1e-9


def root_of(s):
    """Latest step before s taken by the other team (the searching team's decision)."""
    for r in range(s - 1, -1, -1):
        if DRAFT_TEAM[r] != DRAFT_TEAM[s]:
            return r
    return None


def next_pick_turn(team, after):
    """First pick turn of `team` at a step after `after`."""
    for j, k in enumerate(PICK_STEPS):
        if k > after and PICK_TEAM[j] == team:
            return j
    return None


def out_file():
    return os.path.join(C.RESULTS, "p3t", "opp_dr.json")


def lobbies_file():
    return os.path.join(TD.OUT, "oppdr_lobbies.npz")


def stage_lobbies():
    from shared import MAPS
    L = TD.load_lobbies()
    nall = np.load(os.path.join(TD.OUT, "state_cur.npz"))["n"]
    rich = (nall.astype(np.float64).sum(1) >= 50).reshape(-1, 10).all(1) & np.isin(L["map"], MAPS)
    excl = TD.eval_lobbies()                 # tournament, tune and party-study lobbies
    excl |= set(np.load(os.path.join(TD.OUT, "probe_lobbies.npz"))["gi"].tolist())
    cand = np.array([g for g in np.flatnonzero(rich & (L["day"] <= TD.LOBBY_END)) if g not in excl])
    gi = np.sort(np.random.RandomState(89).choice(cand, N_FIT, replace=False))
    np.savez(lobbies_file(), gi=gi)
    print(f"fit lobbies: {len(gi)} of {len(cand)} eligible (tournament, tune, party and probe lobbies excluded)", flush=True)


def stage_hist():
    import p3t_draftrec as DRC
    DRC.stage_hist(lobbies_file(), DRC._dr_file("hist_oppfit.npz"))
    DRC.stage_hist(os.path.join(TD.OUT, "tourney_tune", "lobbies.npz"), DRC._dr_file("hist_tune.npz"))
    party = os.path.join(TD.OUT, "tourney_party", "lobbies.npz")
    if os.path.exists(party):
        DRC.stage_hist(party, DRC._dr_file("hist_party.npz"))


def hist_for_lobbyset():
    import p3t_draftrec as DRC
    ls = os.environ.get("P3T_LOBBYSET", "")
    return DRC._dr_file(f"hist_{ls}.npz" if ls else "hist_tourney.npz")


def _log(p, floor):
    return np.log(np.maximum(p, floor)).astype(np.float32)


# ------------------------------------------------------------------ inference (actor)

def rows(ctx, t, k, w, device=None):
    """Opponent-bias rows for a decide-only search of team t at step k.
    Returns imit (B, 10, H) and ban_rows {step: row} for the opponent's ban
    steps (rows of our team, unused otherwise). Opponent pick steps use their
    kernel slot ctx["slot_of"][:, s] (as the comfort model)."""
    import torch
    import p3t_draftrec as DRC
    inf = DRC.load_inference(device, hist_file=hist_for_lobbyset())
    o = 1 - t
    B = len(ctx["acts"])
    imit = np.zeros((B, 10, H), np.float32)
    ban_rows = {}
    with torch.no_grad():
        S, _ = DRC.replay_state(inf, ctx)
        cache = {}

        def mix(j, team):
            if (j, team) not in cache:
                cache[(j, team)] = DRC._torch().nan_to_num(S.mixture(j, team)[0]).cpu().numpy()
            return cache[(j, team)]
        nb = 0
        for s in range(k + 1, 16):
            if DRAFT_TEAM[s] != o:
                continue
            if IS_PICK[s]:
                q = mix(TURN[s], o)
                sl = ctx["slot_of"][:, s]
                imit[np.arange(B), sl] = float(w["lp"]) * _log(q, DR_FLOOR)
            else:
                j = next_pick_turn(t, s)
                if j is None or not w.get("lb"):
                    continue
                r = 5 * t + nb
                nb += 1
                imit[:, r] = float(w["lb"]) * _log(mix(j, t), DR_FLOOR)
                ban_rows[s] = r
        # denial-ban targeting (optional): the opponent's next pick turn after k
        jn = next_pick_turn(o, k)
        deny = _log(mix(jn, o), DR_FLOOR) if jn is not None else None
    return imit, ban_rows, deny


# ------------------------------------------------------------------ fit

def features(inf, gi, src, acts, mp, tr, batch=500):
    """Per lobby and step s: DraftRec log-prob vectors read at the root of s
    (root protocol) and at s itself (exact, picks only)."""
    import torch
    import p3t_draftrec as DRC
    n = len(gi)
    root = np.zeros((n, 16, H), np.float32)
    exact = np.zeros((n, 16, H), np.float32)
    need = {}
    for s in range(1, 16):
        need.setdefault(root_of(s), []).append(s)
    for b0 in range(0, n, batch):
        b = np.arange(b0, min(b0 + batch, n))
        rws = np.array([inf["pos"][int(g)] for g in gi[b]])
        U = inf["U"][torch.as_tensor(rws, device=inf["device"])[:, None], torch.as_tensor(src[b], device=inf["device"])]
        with torch.no_grad():
            S = DRC.TeamState(inf["model"], U, mp[b], tr[b], inf["device"])
            j = 0
            for k in range(16):
                cache = {}

                def mix(jj, team):
                    if (jj, team) not in cache:
                        cache[(jj, team)] = S.mixture(jj, team)
                    return cache[(jj, team)]
                for s in need.get(k, []):
                    o = DRAFT_TEAM[s]
                    if IS_PICK[s]:
                        m = mix(TURN[s], o)[0]
                    else:
                        jj = next_pick_turn(1 - o, s)
                        m = mix(jj, 1 - o)[0]
                    root[b, s] = _log(torch.nan_to_num(m).cpu().numpy(), DR_FLOOR)
                h = acts[b, k]
                if IS_PICK[k]:
                    m, p, free = mix(j, DRAFT_TEAM[k])
                    exact[b, k] = _log(torch.nan_to_num(m).cpu().numpy(), DR_FLOOR)
                    S.apply_pick(j, DRAFT_TEAM[k], h, p, free)
                    j += 1
                else:
                    S.apply_ban(h)
        print(f"  DraftRec features {b[-1] + 1}/{n}", flush=True)
    return root, exact


def stage_fit():
    import torch
    import p3t_backtest as BT
    import p3t_draftrec as DRC
    import p3t_systems as SY
    t0 = time.time()
    gi = np.load(lobbies_file())["gi"]
    K = BT.kernel_lobbies(gi)
    acts, mp, tr = K["acts"], K["map"].astype(np.int64), K["tier"].astype(np.int64)
    zc = np.load(os.path.join(TD.OUT, "state_cur.npz"))
    nrow = (gi * 10)[:, None] + K["src"]
    n10 = zc["n"][nrow.ravel()].reshape(len(gi), 10, H).astype(np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inf = DRC.load_inference(dev, hist_file=DRC._dr_file("hist_oppfit.npz"))
    root, exact = features(inf, gi, K["src"], acts, mp, tr)
    v = SY.vintage("cur")
    wc = json.load(open(SY.opp_model_path()))
    G = len(gi)
    LG, DRr, DRe, CF, Y, M, PK, LOB = [], [], [], [], [], [], [], []
    for s in range(1, 16):
        t = DRAFT_TEAM[s]
        X = SY.features(acts, s, mp, tr, t)
        m = SY.legal_mask(acts, s, np.ones((G, 2, H), bool))
        LG.append(_log(SY.gd_probs(v, X[:, :289], m), GD_FLOOR))
        DRr.append(root[:, s])
        DRe.append(exact[:, s] if IS_PICK[s] else np.zeros((G, H), np.float32))
        CF.append(SY_opp_bias_batch(n10[:, 5 * t:5 * t + 5], wc) if IS_PICK[s] else np.zeros((G, H), np.float32))
        Y.append(acts[:, s])
        M.append(m)
        PK.append(np.full(G, IS_PICK[s], bool))
        LOB.append(np.arange(G))
    T = {k_: torch.from_numpy(np.concatenate(a_)) for k_, a_ in
         (("lg", LG), ("dr", DRr), ("de", DRe), ("cf", CF), ("y", Y), ("m", M), ("pk", PK), ("lob", LOB))}
    T["y"] = T["y"].long()
    held_l = np.random.RandomState(3).rand(G) < 0.1
    held = torch.from_numpy(held_l)[T["lob"]]

    def logits(sel, w0, lp, lb, src="dr", cf=0.0):
        d = T[src][sel]
        coef = torch.where(T["pk"][sel], lp, lb)[:, None]
        z = w0 * T["lg"][sel] + coef * d + cf * T["cf"][sel]
        return z.masked_fill(~T["m"][sel], -1e9)

    def nll(sel, *a, **kw):
        z = logits(sel, *a, **kw)
        return -(torch.log_softmax(z, 1).gather(1, T["y"][sel][:, None])).mean()

    w = torch.tensor([1.0, 0.5, 0.2], requires_grad=True)
    tr_ = ~held
    opt = torch.optim.LBFGS([w], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = nll(tr_, w[0], w[1], w[2])
        l.backward()
        return l
    opt.step(closure)
    w0, lp, lb = (float(x) for x in w.detach())
    # the exact-state version (an upper bound the static root bias cannot reach)
    we = torch.tensor([1.0, 0.5, 0.0], requires_grad=True)
    opt2 = torch.optim.LBFGS([we], max_iter=300, line_search_fn="strong_wolfe")
    pk_tr = tr_ & T["pk"]

    def closure2():
        opt2.zero_grad()
        l = nll(pk_tr, we[0], we[1], we[2], src="de")
        l.backward()
        return l
    opt2.step(closure2)

    def metrics(sel, *a, **kw):
        with torch.no_grad():
            z = logits(sel, *a, **kw)
            l = -(torch.log_softmax(z, 1).gather(1, T["y"][sel][:, None])).mean().item()
            top = (z.argmax(1) == T["y"][sel]).float().mean().item()
        return {"top1": round(top, 4), "nll": round(l, 4), "n": int(sel.sum())}
    out = {"w0": w0, "lp": lp, "lb": lb, "fit_lobbies": int(G), "held_out_lobbies": int(held_l.sum())}
    for name, msk in (("picks", held & T["pk"]), ("bans", held & ~T["pk"])):
        r = {"GD": metrics(msk, 1.0, 0.0, 0.0),
             "DraftRec opponent (kernel model: w0 log GD + DraftRec at the root)": metrics(msk, w0, lp, lb)}
        if name == "picks":
            r["GD x comfort (current opponent model, exact state)"] = metrics(msk, wc["w0"], 0.0, 0.0, cf=1.0)
            r["DraftRec team mixture alone, exact state"] = metrics(msk, 0.0, 1.0, 0.0, src="de")
            r["DraftRec team mixture alone, at the root"] = metrics(msk, 0.0, 1.0, 0.0)
            r["w0 log GD + DraftRec, exact state (refit; not usable in the kernel)"] = metrics(
                msk, float(we[0]), float(we[1]), 0.0, src="de")
        else:
            r["DraftRec (our next pick) alone, at the root"] = metrics(msk, 0.0, 0.0, 1.0)
        out[name] = r
    out["exact_refit"] = {"w0": float(we[0]), "lp": float(we[1])}
    out["secs"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(out_file()), exist_ok=True)
    json.dump(out, open(out_file(), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


def SY_opp_bias_batch(n5, wc):
    """p3t_systems.opp_bias for a batch: (G, 5, H) play counts -> (G, H)."""
    u = wc["w1"] * np.log1p(n5) + wc["w2"] * (n5 == 0)
    m = u.max(1)
    return (m + np.log(np.exp(u - m[:, None]).mean(1))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lobbies", "hist", "fit"])
    a = ap.parse_args()
    {"lobbies": stage_lobbies, "hist": stage_hist, "fit": stage_fit}[a.stage]()


if __name__ == "__main__":
    main()
