"""
P3 tournament runner: drafting systems play real lobbies against each other.

Lobbies (stage `select`): tournament-window OOT games (2026-05-23 to 06-30,
never the sealed build), all ten players with 50+ earlier games and 10+
games in the judge window (2026-07-01 to 09-11, so the judges can value
them); N lobbies drawn with a fixed seed. Every ordered pair plays the same
N lobbies: system A drafts for the real first-pick team (kernel team 0),
system B for the other team; the reverse ordering swaps the seats.

Per draft i the prior/policy seed of a seeded system is i mod its number of
seeds (paper 1's pooling); kernel seeds are fixed per (pair, lobby, step).

Stages
  select                        choose and save the lobbies (cache p3t/tourney/lobbies.npz)
  run --pairs A__B,...|--shard i/n [--schedule NAME]
                                play pairs; one npz per ordered pair, skipped if present
  timing                        GPU seconds per decision, aware vs blind (compute cost)
Run (from training/, a GPU host): python personalization/p3t_tourney.py run --schedule core --shard 0/3
"""
import argparse
import hashlib
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
import p3t_systems as SY

# P3T_LOBBYSET=tune: a separate lobby set for tuning the exploration design (never the tournament lobbies)
LOBBYSET = os.environ.get("P3T_LOBBYSET", "")
TDIR = os.path.join(TD.OUT, "tourney" + (f"_{LOBBYSET}" if LOBBYSET else ""))
os.makedirs(TDIR, exist_ok=True)
N_LOBBIES = int(os.environ.get("P3T_LOBBIES", "300"))
J_MIN = 10


# ------------------------------------------------------------------ lobbies

def select():
    L = TD.load_lobbies()
    cur = np.load(os.path.join(TD.OUT, "state_cur.npz"))
    jd = np.load(os.path.join(TD.OUT, "state_judge.npz"))
    win = np.flatnonzero(L["day"] <= TD.LOBBY_END)
    assert (jd["gsel"] == win).all()
    rich = (cur["n"].astype(np.float64).sum(1) >= 50).reshape(-1, 10).all(1)[win]
    jn = jd["rec_n"].astype(np.float64).sum(1).reshape(-1, 10)
    ok = rich & (jn >= J_MIN).all(1)
    cand = win[ok]
    rng = np.random.RandomState(20261019)
    if LOBBYSET:
        test = set(np.load(os.path.join(TD.OUT, "tourney", "lobbies.npz"))["gi"].tolist())
        cand = np.array([g for g in cand if g not in test])
        rng = np.random.RandomState(20261020)
    sel = np.sort(rng.choice(cand, N_LOBBIES, replace=False))
    np.savez(os.path.join(TDIR, "lobbies.npz"), gi=sel)
    print(f"tournament-window lobbies {len(win):,}; all ten 50+ earlier games {int(rich.sum()):,}; "
          f"and 10+ judge-window games each {len(cand):,}; selected {len(sel)}", flush=True)
    json.dump({"window_lobbies": int(len(win)), "rich": int(rich.sum()), "eligible": int(len(cand)),
               "selected": int(len(sel))}, open(os.path.join(TDIR, "select.json"), "w"))


class Lobbies:
    """Kernel-labeled per-lobby tensors for the selected lobbies."""

    def __init__(self):
        from shared import MAPS, SKILL_TIERS
        self.L = TD.load_lobbies()
        self.gi = np.load(os.path.join(TDIR, "lobbies.npz"))["gi"]
        win = np.flatnonzero(self.L["day"] <= TD.LOBBY_END)
        wpos = {int(g): i for i, g in enumerate(win)}
        st = {v: np.load(os.path.join(TD.OUT, f"state_{v}.npz")) for v in ("cur", "s4m", "s1y", "judge", "splitd", "splitj")
              if os.path.exists(os.path.join(TD.OUT, f"state_{v}.npz"))}
        n = len(self.gi)
        self.map = np.array([MAPS.index(str(self.L["map"][g])) for g in self.gi], np.int32)
        self.tier = np.array([SKILL_TIERS.index(str(self.L["tier"][g])) for g in self.gi], np.int32)
        self.first = self.L["first_team"][self.gi].astype(np.int64)
        self.real_acts = self.L["steps"][self.gi][:, :, 0].astype(np.int64)
        self.slot_of = np.full((n, 16), -1, np.int32)
        self.src = np.zeros((n, 10), np.int64)       # pick index j (0-9) of each kernel slot
        for i in range(n):
            cnt = [0, 0]
            j = 0
            for k in range(16):
                h, ty, tm, r = self.L["steps"][self.gi[i]][k]
                if ty == 1:
                    kt = tm ^ self.first[i]
                    assert kt == SY.DRAFT_TEAM[k]
                    sl = 5 * kt + cnt[kt]
                    cnt[kt] += 1
                    self.slot_of[i, k] = sl
                    self.src[i, sl] = j
                    j += 1
        self.state = {}
        for v, z in st.items():
            base = (self.gi * 10) if v == "cur" else np.array([wpos[int(g)] * 10 for g in self.gi])
            rows = base[:, None] + self.src
            ent = {"s": z["s"][rows.ravel()].astype(np.float32).reshape(n, 10, 90),
                   "sd": (z["sd"][rows.ravel()].astype(np.float32).reshape(n, 10, 90) if "sd" in z.files
                          else np.zeros((n, 10, 90), np.float32)),
                   "off": z["off"][rows.ravel()].reshape(n, 10, 90)}
            if v == "judge" and "rec_sum" in z.files:
                ent["rec_sum"] = z["rec_sum"][rows.ravel()].reshape(n, 10, 90)
                ent["rec_n"] = z["rec_n"][rows.ravel()].astype(np.float32).reshape(n, 10, 90)
            if v == "cur":
                ent["n"] = z["n"][rows.ravel()].astype(np.float32).reshape(n, 10, 90)
            self.state[v] = ent
        pool = self.state["cur"]["n"] > 0                 # environment: heroes played before the lobby
        self.pool = pool
        self.team_pool = np.stack([pool[:, :5].any(1), pool[:, 5:].any(1)], 1)
        import p3_mcts_core as M
        self.pool_bits = np.stack([np.stack([M.pool_bits(pool[i, s]) for s in range(10)]) for i in range(n)])
        self.n = n


def ctx_for(LB, idx, acts, k, sysname, kseed):
    spec = SY.SYSTEMS[sysname]
    st = LB.state[spec.get("state", SY.vintage_state(spec["vint"]))]
    return {"acts": acts[idx], "k": k, "map": LB.map[idx], "tier": LB.tier[idx], "team_pool": LB.team_pool[idx],
            "pool_bits": LB.pool_bits[idx], "slot_of": LB.slot_of[idx], "s": st["s"][idx], "off": st["off"][idx],
            "real_acts": LB.real_acts[idx], "seed_i": idx.astype(np.int64), "kseed": kseed,
            "n": LB.state["cur"]["n"][idx], "gi": LB.gi[idx], "src": LB.src[idx], "sd": st["sd"][idx]}


_ONESTEP = {}


def act(sysname, ctx, LB, idx, device):
    spec = SY.SYSTEMS[sysname]
    v = SY.vintage(spec["vint"])
    kind = spec["kind"]
    if kind == "mcts":
        return SY.act_mcts(v, ctx, spec["aware"], device=device)
    if kind == "policy":
        return SY.act_policy(v, ctx)
    if kind == "mcts_pp":
        return SY.act_mcts(v, ctx, spec["aware"], device=device, pp=True, prior=spec["prior"])
    if kind == "policy_rich":
        return SY.act_policy_rich(v, ctx, spec["signal"])
    if kind == "mcts_x":
        return SY.act_mcts_x(v, ctx, spec["x"], device=device)
    if kind == "draftrec":
        import p3t_draftrec as DRC
        return DRC.act_draftrec(ctx, mode=spec["mode"])
    if kind == "mcts_vn":
        return SY.act_mcts(v, ctx, spec["aware"], device=device, vn=True)
    if kind == "policy_pp":
        return SY.act_policy_pp(v, ctx, prior=spec["prior"])
    if kind == "gd":
        return SY.act_gd(v, ctx)
    if kind == "real":
        return SY.act_real(v, ctx)
    if kind == "onestep":
        key = spec["vint"]
        if key not in _ONESTEP:
            _ONESTEP[key] = SY.OneStep(key, int(os.environ.get("P3T_PROCS", "12")))
        b = v.spec["combiner"]
        ctx["dr_lob"] = [SY.dr_lobby({"map": LB.map[i], "tier": LB.tier[i]}, ctx["s"][j], ctx["off"][j],
                                     LB.pool[i], b) for j, i in enumerate(idx)]
        return _ONESTEP[key].act(ctx, spec["aware"])
    raise ValueError(kind)


def pair_file(a, b):
    return os.path.join(TDIR, f"{a}__{b}.npz")


def play(LB, a, b, device=0):
    """Ordered pair: a = kernel team 0 (first pick), b = team 1."""
    p = pair_file(a, b)
    if os.path.exists(p):
        return
    t0 = time.time()
    n = LB.n
    acts = np.full((n, 16), -1, np.int64)
    idx = np.arange(n)
    secs = {a: 0.0, b: 0.0}
    fallback = np.zeros((n, 16), bool)
    h = int(hashlib.md5(f"{a}|{b}".encode()).hexdigest()[:7], 16)
    for k in range(16):
        s = a if SY.DRAFT_TEAM[k] == 0 else b
        ctx = ctx_for(LB, idx, acts, k, s, kseed=h + 1000 * k)
        t1 = time.time()
        acts[:, k] = act(s, ctx, LB, idx, device)
        secs[s] += time.time() - t1
        if "fallback" in ctx:
            fallback[:, k] = ctx["fallback"]
    assert (acts >= 0).all()
    for i in range(n):
        assert len(set(acts[i])) == 16
    np.savez(p + ".tmp.npz", acts=acts, gi=LB.gi, fallback=fallback,
             secs=np.array([secs[a], secs[b]]), a=a, b=b)
    os.replace(p + ".tmp.npz", p)
    print(f"  {a} vs {b}: {time.time() - t0:.0f}s (a {secs[a]:.0f}s, b {secs[b]:.0f}s)", flush=True)


# ------------------------------------------------------------------ schedules

CORE = ["aMCTS-cur", "pMCTS-cur", "aMCTS-bc", "pMCTS-bc", "pPol-cur", "a1step-cur", "p1step-cur", "GD", "Real"]
PANEL = ["aMCTS-cur", "pMCTS-cur", "pPol-cur", "GD", "Real"]
SPOKES = {
    # prior trained with player features
    "aMCTS-pp": ["aMCTS-bc", "pMCTS-bc"], "aPol-pp": ["pPol-cur"],
    "aMCTS-sp": ["aMCTS-bc", "pMCTS-bc", "aMCTS-pp"], "aPol-sp": ["pPol-cur", "aPol-pp"],
    "aMCTS-vn": ["aMCTS-bc", "pMCTS-bc"],
    # split-history robustness (drafter state from odd days only)
    "aMCTS-bc@split": ["pMCTS-bc", "aMCTS-bc"], "a1step@split": ["p1step-cur", "a1step-cur"],
    "DraftRec": ["aMCTS-bc", "a1step-cur", "aMCTS-pp", "DraftRec-pv"], "DraftRec-pv": ["aMCTS-bc", "a1step-cur", "aMCTS-pp"],
    # drift, population side: stale all / refreshed stats / fine-tune ; full retrain = pMCTS-cur
    "pMCTS-s4m": ["pMCTS-cur", "aMCTS-s4m"], "pMCTS-s1y": ["pMCTS-cur", "aMCTS-s1y"],
    "pMCTS-s4m-rs": ["pMCTS-s4m"], "pMCTS-s1y-rs": ["pMCTS-s1y"],
    "pMCTS-s4m-ft": ["pMCTS-s4m"], "pMCTS-s1y-ft": ["pMCTS-s1y"],
    # drift, player-aware: stale all / player state frozen only
    "aMCTS-s4m": ["aMCTS-cur"], "aMCTS-s1y": ["aMCTS-cur"],
    "aMCTS-cur@s4m": ["aMCTS-cur"], "aMCTS-cur@s1y": ["aMCTS-cur"],
    # less optimizing drafters under drift
    "pPol-s4m": ["pPol-cur"], "pPol-s1y": ["pPol-cur"],
    "a1step-s1y": ["a1step-cur", "p1step-s1y"], "p1step-s1y": ["p1step-cur"],
}


TUNE_OPP = ["pMCTS-cur", "aMCTS-pp"]
TUNE_EXTRA = ["aPol-rich-bcow", "aPol-rich-onestep", "aPol-rich-search", "aPol-pp", "aMCTS-sp", "aPol-sp", "a1step-cur",
              "aMCTS-bc", "aMCTS-cur"]


ABL = ["x-pp", "x-combo3", "x-combo1", "x-pp-ban3", "x-pp-opp2", "x-pp-as1", "x-pp-vn2", "x-pp-apos", "x-pp-s1600",
       "x-pp-c1", "x-pp-c4", "x-pp-b05", "x-pp-b2", "x-pp-a03", "x-pp-a0", "x-pp-opt05", "x-pp-opt1",
       "x-rich-search", "x-rich-bcow", "x-rich-onestep", "aPol-rich-search", "aPol-rich-bcow", "aPol-rich-onestep"]


def schedule(name):
    pairs = []
    if name == "ablation":
        # the design study on the tournament lobbies: every component against aMCTS-pp;
        # the selected design also against the panel; split-history versions against the blind drafters
        for x in ABL:
            pairs += [(x, "aMCTS-pp"), ("aMCTS-pp", x)]
        for o in PANEL + ["aMCTS-bc", "a1step-cur", "DraftRec-pv", "aMCTS-sp"]:
            pairs += [("x-combo3", o), (o, "x-combo3")]
        for x in ("x-combo3-split", "x-pp-split"):
            for o in ("pMCTS-cur", "pMCTS-bc"):
                pairs += [(x, o), (o, x)]
    if name == "tune":
        for x in list(SY.explore_systems()) + TUNE_EXTRA:
            for o in TUNE_OPP:
                pairs += [(x, o), (o, x)]
    if name in ("core", "all"):
        pairs += [(x, y) for x in CORE for y in CORE if x != y]
    if name in ("spokes", "all"):
        for s, extra in SPOKES.items():
            if s not in SY.SYSTEMS:
                continue
            for o in PANEL + extra:
                if o != s:
                    pairs += [(s, o), (o, s)]
    seen, out = set(), []
    for p in pairs:
        if p not in seen and p[0] in SY.SYSTEMS and p[1] in SY.SYSTEMS:
            seen.add(p)
            out.append(p)
    return out


def needs_gpu(p):
    return any(SY.SYSTEMS[s]["kind"] == "mcts" for s in p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "run", "list"])
    ap.add_argument("--schedule", default="core")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--kind", choices=["all", "onestep", "no-onestep"], default="all",
                    help="onestep: only pairs with a one-step system (CPU pool); no-onestep: the others")
    ap.add_argument("--systems", default="", help="comma list: keep pairs with at least one of these systems")
    ap.add_argument("--only-ready", action="store_true", help="skip pairs whose vintages are not in p3t_vintages.json yet")
    a = ap.parse_args()
    if a.stage == "select":
        select()
        return
    pairs = [tuple(x.split("__")) for x in a.pairs.split(",")] if a.pairs else schedule(a.schedule)
    if a.only_ready:
        vs = json.load(open(SY.VFILE))
        pairs = [p for p in pairs if all(SY.SYSTEMS[s]["vint"] in vs and vs[SY.SYSTEMS[s]["vint"]].get("combiner")
                                         for s in p)]
    if a.systems:
        keep = set(a.systems.split(","))
        pairs = [p for p in pairs if keep & set(p)]
    has1 = lambda p: any(SY.SYSTEMS[x]["kind"] == "onestep" for x in p)
    if a.kind == "onestep":
        pairs = [p for p in pairs if has1(p)]
    elif a.kind == "no-onestep":
        pairs = [p for p in pairs if not has1(p)]
    i, n = (int(x) for x in a.shard.split("/"))
    pairs = pairs[i::n]
    if a.stage == "list":
        for p in pairs:
            print(p, "done" if os.path.exists(pair_file(*p)) else "")
        return
    import torch
    torch.set_num_threads(2)
    LB = Lobbies()
    print(f"{len(pairs)} pairs, {LB.n} lobbies", flush=True)
    for p in pairs:
        play(LB, p[0], p[1], device=a.device)
    for o in _ONESTEP.values():
        o.close()
    print("RUN DONE", flush=True)


if __name__ == "__main__":
    main()
