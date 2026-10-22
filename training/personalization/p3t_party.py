"""
P3: the partial-control (party) drafter (Max, coordinator 2026-10-22).

A party of m players (1 to 5) sits on one team of a real tournament lobby.
The drafter picks only for the party members, at their own pick turns; the
other teammates are strangers. Within the team, the five players take the
team's five pick turns in a random order (or the party is put on chosen
turns, for the position study); the opponents keep the real game's order.

  strangers   pick from the personalized generic drafter: DraftRec's pick head
              for that specific player (his own history, the draft so far, the
              turn), sampled at temperature 1
  opponents   the same model for each opponent player; their bans: GD argmax
  bans        ASSUMPTION: the player on the team's first pick turn bans for the
              team (Storm League gives the bans to the first player in the
              team's draft order). The party controls the bans if and only if
              it holds that turn; otherwise the team's bans are GD argmax.
              (ban rule "never": the party never bans, to separate the effect.)
  search      the drafter's search (cuda_explore, v2 chance search) treats the
              strangers' steps as chance nodes (cfg[40 + step] = -2) with the
              stranger's own DraftRec distribution read at the decision state,
              exactly like the opponents' steps.

Arms for the party's seats (the rest of the lobby is identical, common random
numbers per lobby and step):
  aware    the selected design (x-combo1-dr-chance) for the party, with the
           strangers and opponents as DraftRec chance nodes
  blind    the population-only search (x-blind-s400: pMCTS-cur's leaf and F
           prior), strangers and opponents as GD chance nodes
  generic  the party picks like strangers (DraftRec sample); bans GD argmax
  real     each party member takes the hero he played in the real game when
           still legal, else as generic; bans: the real team's bans when legal

Outputs: cache p3t/party/<config>.npz (acts, slot_at, party, stranger) and
results p3t/party.json (per configuration: J, J_cal, CONS, the judge-free
record of the party's team; paired differences between arms with lobby
bootstrap CIs).
Usage (GPU host, from training/): python personalization/p3t_party.py run --plan effect|position [--only NAME]
                                 python personalization/p3t_party.py score
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

DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
PICK_STEPS = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]
PICK_TEAM = [DRAFT_TEAM[k] for k in PICK_STEPS]
TURN = {k: j for j, k in enumerate(PICK_STEPS)}
TEAM_TURNS = {t: [j for j in range(10) if PICK_TEAM[j] == t] for t in (0, 1)}   # global turn indices
H = 90
ARMS = ("aware", "blind", "generic", "real")
DESIGN = "x-combo1-dr-chance"
BLIND = "x-blind-s400"
OUT = os.path.join(TD.OUT, "party")


def plan(name):
    """Configurations: (name, team t, party size m, party turns (indices 0-4 into the team's
    turns) or None for random, ban rule)."""
    out = []
    if name == "effect":
        for m in (1, 2, 3, 4, 5):
            for t in (0, 1):
                out.append((f"eff_m{m}_t{t}", t, m, None, "first"))
    if name == "position":
        sets = {1: {f"p{q}": [q] for q in range(5)},
                2: {"early": [0, 1], "middle": [1, 2], "late": [3, 4]},
                3: {"early": [0, 1, 2], "middle": [1, 2, 3], "late": [2, 3, 4]},
                4: {"early": [0, 1, 2, 3], "late": [1, 2, 3, 4]}}
        for m, d in sets.items():
            for lab, turns in d.items():
                for t in (0, 1):
                    out.append((f"pos_m{m}_{lab}_t{t}", t, m, turns, "first"))
                    if 0 in turns:
                        out.append((f"pos_m{m}_{lab}_t{t}_noban", t, m, turns, "never"))
    return out


def layout(LB, cfg):
    """Per lobby: turn_of (n, 10) global turn of each kernel slot, party (n, 10) bool,
    ban control (n,) bool. Seeded by the configuration without the arm (common numbers)."""
    name, t, m, turns, ban_rule = cfg
    n = LB.n
    turn_of = LB.src.copy()                    # real pick order (opponents keep it)
    party = np.zeros((n, 10), bool)
    seed = int(hashlib.md5(f"party|{name}".encode()).hexdigest()[:7], 16)
    for i in range(n):
        rng = np.random.RandomState(seed + i)
        slots = 5 * t + rng.permutation(5)     # slots in the order they take the team's turns 0..4
        if turns is None:
            psl = slots[:m]                    # random positions: the first m of a random order
            order = rng.permutation(5)
            for q, sl in enumerate(slots[order]):
                turn_of[i, sl] = TEAM_TURNS[t][q]
            party[i, psl] = True
        else:
            members = slots[:m]
            others = [sl for sl in slots if sl not in set(members)]
            rest = [q for q in range(5) if q not in turns]
            for q, sl in zip(turns, members):
                turn_of[i, sl] = TEAM_TURNS[t][q]
            for q, sl in zip(rest, others):
                turn_of[i, sl] = TEAM_TURNS[t][q]
            party[i, members] = True
    first = TEAM_TURNS[t][0]
    holds_first = np.array([party[i, np.flatnonzero(turn_of[i, 5 * t:5 * t + 5] == first)[0] + 5 * t] for i in range(n)])
    ban_ctl = holds_first if ban_rule == "first" else np.zeros(n, bool)
    return turn_of, party, ban_ctl


class KnownState:
    """DraftRec with the true player of every pick: each player sits at his own turn in the
    match network, the acting player is the query (the tournament's TeamState infers who
    picked; here the order is known)."""

    def __init__(self, inf, gi, src, mp, tr, turn_of):
        import torch
        self.torch = torch
        self.m, self.dev = inf["model"], inf["device"]
        rows = np.array([inf["pos"][int(g)] for g in gi])
        self.U = inf["U"][torch.as_tensor(rows, device=self.dev)[:, None], torch.as_tensor(src, device=self.dev)]
        self.mp = torch.as_tensor(np.asarray(mp, np.int64), device=self.dev)
        self.tr = torch.as_tensor(np.asarray(tr, np.int64), device=self.dev)
        self.B = len(gi)
        self.turn_of = np.asarray(turn_of)
        self.assigned = np.full((self.B, 10), -1, np.int64)
        self.taken = np.zeros((self.B, H), bool)
        self.perm = np.argsort(self.turn_of, 1)            # slot at each global turn

    def subset(self, idx):
        o = KnownState.__new__(KnownState)
        o.torch, o.m, o.dev = self.torch, self.m, self.dev
        ti = self.torch.as_tensor(idx, device=self.dev)
        o.U, o.mp, o.tr = self.U[ti], self.mp[ti], self.tr[ti]
        o.B = len(idx)
        o.turn_of, o.assigned, o.taken, o.perm = (self.turn_of[idx], self.assigned[idx].copy(),
                                                  self.taken[idx].copy(), self.perm[idx])
        return o

    def dist(self, j):
        """(B, H) pick distribution of the player whose turn is j (taken heroes masked)."""
        torch = self.torch
        team = PICK_TEAM[j]
        champ = np.zeros((self.B, 10), np.int64)
        for jj in range(10):
            sl = self.perm[:, jj]
            h = self.assigned[np.arange(self.B), sl]
            champ[:, jj] = np.where(h >= 0, h + 1, 0)
        champ[:, j] = 0
        with torch.no_grad():
            Ut = self.U[torch.arange(self.B, device=self.dev)[:, None], torch.as_tensor(self.perm, device=self.dev)]
            umask = torch.as_tensor(np.array(PICK_TEAM)[None].repeat(self.B, 0) != team, device=self.dev).long()
            P = self.m.match(Ut, torch.as_tensor(champ, device=self.dev), umask, self.mp, self.tr)
            lg = self.m.pick_logits(P, torch.full((self.B,), j, device=self.dev)).float()
            p = torch.softmax(lg.masked_fill(torch.as_tensor(self.taken, device=self.dev), -1e9), 1)
        return torch.nan_to_num(p).cpu().numpy()

    def pick(self, j, hero):
        sl = self.perm[:, j]
        self.assigned[np.arange(self.B), sl] = hero
        self.taken[np.arange(self.B), hero] = True

    def ban(self, hero):
        self.taken[np.arange(self.B), hero] = True


def sample(p, legal, rng):
    q = np.where(legal, p, 0.0)
    s = q.sum(1, keepdims=True)
    q = np.where(s > 0, q / np.maximum(s, 1e-12), legal / legal.sum(1, keepdims=True))
    u = rng.random_sample(len(q))[:, None]
    return np.minimum((q.cumsum(1) < u).sum(1), H - 1)


def draft(LB, cfg, arm, device=0):
    import p3t_draftrec as DRC
    import p3t_oppdr as OD
    import p3t_systems as SY
    import p3t_tourney as TT
    name, t, m, turns, ban_rule = cfg
    n = LB.n
    turn_of, party, ban_ctl = layout(LB, cfg)
    slot_at = np.full((n, 16), -1, np.int64)
    for k in PICK_STEPS:
        slot_at[:, k] = np.argmax(turn_of == TURN[k], 1)
    inf = DRC.load_inference(device=None)
    KS = KnownState(inf, LB.gi, LB.src, LB.map, LB.tier, turn_of)
    w = json.load(open(OD.out_file()))
    v = SY.vintage("cur")
    acts = np.full((n, 16), -1, np.int64)
    idx_all = np.arange(n)
    # stranger flags: our steps not decided by the party (picks of strangers, uncontrolled bans)
    stranger = np.zeros((n, 16), bool)
    for k in range(16):
        if DRAFT_TEAM[k] != t:
            continue
        if IS_PICK[k]:
            stranger[:, k] = ~party[idx_all, slot_at[:, k]]
        else:
            stranger[:, k] = ~ban_ctl
    seed = int(hashlib.md5(f"partyrng|{name}".encode()).hexdigest()[:7], 16)
    secs = 0.0
    for k in range(16):
        rng = np.random.RandomState(seed + 1000 * k)
        team = DRAFT_TEAM[k]
        ctx = TT.ctx_for(LB, idx_all, acts, k, DESIGN, kseed=seed + 7919 * k)
        ctx["slot_of"] = slot_at.astype(np.int32)
        legal = SY.legal_mask(acts, k, LB.team_pool)
        out = np.full(n, -1, np.int64)
        if IS_PICK[k]:
            p = KS.dist(TURN[k])                           # the acting player's DraftRec distribution
            drs = sample(p, legal, rng)
        ours = team == t
        dec = (~stranger[:, k]) if ours else np.zeros(n, bool)
        if ours and arm in ("aware", "blind") and dec.any():
            idx = np.flatnonzero(dec)
            c2 = {kk: (vv[idx] if isinstance(vv, np.ndarray) and len(vv) == n else vv) for kk, vv in ctx.items()}
            c2["stranger"] = stranger[idx]
            sysname = DESIGN if arm == "aware" else BLIND
            x = SY.SYSTEMS[sysname]["x"]
            if arm == "aware":
                c2["imit_override"] = (party_rows(KS.subset(idx), k, t, stranger[idx], slot_at[idx], w), w["w0"])
            t0 = time.time()
            out[idx] = SY.act_mcts_x(SY.vintage(SY.SYSTEMS[sysname]["vint"]), c2, x, device=device)
            secs += time.time() - t0
        if ours and arm == "real" and dec.any():
            real = LB.real_acts[:, k] if not IS_PICK[k] else real_hero(LB, k, slot_at, party)
            ok = dec & (real >= 0) & legal[idx_all, np.maximum(real, 0)]
            out[ok] = real[ok]
        rest = out < 0
        if IS_PICK[k]:
            out[rest] = drs[rest]
        elif rest.any():
            g = SY.act_gd(v, ctx)
            out[rest] = g[rest]
        assert legal[idx_all, out].all(), f"{name} {arm} step {k}: illegal action"
        acts[:, k] = out
        if IS_PICK[k]:
            KS.pick(TURN[k], out)
        else:
            KS.ban(out)
    return acts, slot_at, party, stranger, ban_ctl, secs


def real_hero(LB, k, slot_at, party):
    """The hero each party member played in the real game (his kernel slot's real pick)."""
    n = LB.n
    out = np.full(n, -1, np.int64)
    for i in range(n):
        sl = slot_at[i, k]
        for kk in PICK_STEPS:
            if LB.slot_of[i, kk] == sl:
                out[i] = LB.real_acts[i, kk]
    return out


def party_rows(KS, k, t, stranger, slot_at, w):
    """imit rows for a decision at step k: every later opponent pick and stranger pick gets
    its player's DraftRec log-distribution read now (row = that player's slot)."""
    B = KS.B
    imit = np.zeros((B, 10, H), np.float32)
    for s in PICK_STEPS:
        if s <= k:
            continue
        if DRAFT_TEAM[s] == t and not stranger[:, s].any():
            continue
        q = KS.dist(TURN[s])
        lq = float(w["lp"]) * np.log(np.maximum(q, 1e-6)).astype(np.float32)
        use = np.ones(B, bool) if DRAFT_TEAM[s] != t else stranger[:, s]
        imit[np.flatnonzero(use), slot_at[use, s]] = lq[use]
    return imit


def stage_run(plan_name, only=None, device=0):
    import p3t_tourney as TT
    LB = TT.Lobbies()
    os.makedirs(OUT, exist_ok=True)
    for cfg in plan(plan_name):
        if only and cfg[0] not in only.split(","):
            continue
        for arm in ARMS:
            if plan_name == "position" and arm in ("blind", "real"):
                continue
            f = os.path.join(OUT, f"{cfg[0]}__{arm}.npz")
            if os.path.exists(f):
                continue
            t0 = time.time()
            acts, slot_at, party, stranger, ban_ctl, secs = draft(LB, cfg, arm, device)
            np.savez(f + ".tmp.npz", acts=acts, slot_at=slot_at, party=party, stranger=stranger, ban_ctl=ban_ctl,
                     t=cfg[1], m=cfg[2], arm=arm, search_secs=secs)
            os.replace(f + ".tmp.npz", f)
            print(f"  {cfg[0]} {arm}: {time.time() - t0:.0f}s (search {secs:.0f}s)", flush=True)
    print("PARTY RUN DONE", flush=True)


# ------------------------------------------------------------------ scoring

def true_terms(LB, acts, slot_at):
    """The judge's personal terms with the TRUE hero-to-player map (each hero belongs to the
    player who picked it; the tournament's scoring takes the best assignment instead), and the
    judge-free record of each team with that map."""
    js, jo = LB.state["judge"]["s"].astype(np.float64), LB.state["judge"]["off"].astype(np.float64)
    rs, rn = LB.state["judge"]["rec_sum"], LB.state["judge"]["rec_n"]
    n = len(acts)
    ar = np.arange(n)
    S, O = np.zeros((n, 2)), np.zeros((n, 2))
    rec = [[[] for _ in range(2)] for _ in range(n)]
    for k in PICK_STEPS:
        t = DRAFT_TEAM[k]
        sl, h = slot_at[:, k], acts[:, k]
        S[:, t] += js[ar, sl, h]
        O[:, t] += jo[ar, sl, h]
        c = rn[ar, sl, h]
        for i in np.flatnonzero(c > 0):
            rec[i][t].append(rs[i, sl[i], h[i]] / c[i])
    rec_mean = np.array([[np.mean(r) if r else np.nan for r in row] for row in rec])
    return S[:, 0] - S[:, 1], O[:, 0] - O[:, 1], rec_mean


def team_view(sc, t, judge):
    """Per lobby, the party team's J (true hero-to-player map), J_best (the tournament's best
    assignment), J_cal, CONS (win %) and the judge-free record (pp, true map)."""
    from p3t_analyze import CAL_A, CAL_K
    pc = np.clip(sc["consensus"], 1e-6, 1 - 1e-6)
    lc = np.log(pc / (1 - pc))
    pers = judge["c2_skill"] * sc["dS_true"] + judge["c3_offrole"] * sc["dO_true"]
    jt = 1 / (1 + np.exp(-(lc + pers)))
    cal = 1 / (1 + np.exp(-(CAL_A * lc + CAL_K * pers)))
    f = (lambda x: x) if t == 0 else (lambda x: 1 - x)
    return {"J": 100 * f(jt), "J_best": 100 * f(sc["judge_personal"]), "J_cal": 100 * f(cal),
            "CONS": 100 * f(sc["consensus"]), "record": 100 * sc["rec_true"][:, t]}


def boot_ci(d, B=2000, seed=0):
    d = d[np.isfinite(d)]
    rng = np.random.RandomState(seed)
    bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)])
    return [round(float(d.mean()), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]


def stage_score():
    import p3t_judge as JG
    import p3t_tourney as TT
    LB = TT.Lobbies()
    judge = json.load(open(os.path.join(JG.RES, JG.JNAME)))
    truth = JG.Truth()
    sdir = os.path.join(OUT, "scores")
    os.makedirs(sdir, exist_ok=True)
    views = {}
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".npz") or "__" not in f or ".tmp" in f:
            continue
        sp = os.path.join(sdir, f)
        z = np.load(os.path.join(OUT, f))
        if not os.path.exists(sp) or os.path.getmtime(sp) < os.path.getmtime(os.path.join(OUT, f)):
            sc = JG.score_drafts(LB, z["acts"], judge, truth)
            sc["dS_true"], sc["dO_true"], sc["rec_true"] = true_terms(LB, z["acts"], z["slot_at"])
            np.savez(sp, **sc)
        sc = dict(np.load(sp))
        views[f[:-4]] = team_view(sc, int(z["t"]), judge)
    res = {"configs": {}, "contrasts": {}}
    for k, v in views.items():
        res["configs"][k] = {m: boot_ci(x) for m, x in v.items()}
    # paired contrasts within a configuration, and seats pooled
    base = sorted({k.split("__")[0] for k in views})
    for b in base:
        for a1, a2 in (("aware", "generic"), ("aware", "blind"), ("aware", "real"), ("blind", "generic"), ("real", "generic")):
            k1, k2 = f"{b}__{a1}", f"{b}__{a2}"
            if k1 in views and k2 in views:
                res["contrasts"][f"{b}: {a1} - {a2}"] = {m: boot_ci(views[k1][m] - views[k2][m]) for m in views[k1]}
    # seats pooled (the first- and second-pick team, concatenated by lobby)
    import re
    groups = {}
    for b_ in base:
        mt = re.search(r"_t([01])(?=_|$)", b_)
        groups.setdefault(b_[:mt.start()] + b_[mt.end():], {})[int(mt.group(1))] = b_
    for g, byt in sorted(groups.items()):
        if len(byt) != 2:
            continue
        for a1, a2 in (("aware", "generic"), ("aware", "blind"), ("aware", "real")):
            ks = [(f"{byt[t]}__{a1}", f"{byt[t]}__{a2}") for t in (0, 1)]
            if all(k1 in views and k2 in views for k1, k2 in ks):
                res["contrasts"][f"{g} (both seats): {a1} - {a2}"] = {
                    m: boot_ci(np.concatenate([views[k1][m] - views[k2][m] for k1, k2 in ks])) for m in views[ks[0][0]]}
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    json.dump(res, open(os.path.join(C.RESULTS, "p3t", "party.json"), "w"), indent=1)
    print(f"party: {len(views)} drafts scored", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "score"])
    ap.add_argument("--plan", default="effect")
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "run":
        stage_run(a.plan, a.only, a.device)
    else:
        stage_score()


if __name__ == "__main__":
    main()
