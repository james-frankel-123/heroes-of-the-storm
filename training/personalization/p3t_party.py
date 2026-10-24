"""
P3: the partial-control (party) drafter (Max, coordinator 2026-10-22; draft rules corrected
2026-10-23).

A party of m players (1 to 5) sits on one team of a real tournament lobby. The drafter
decides only for the party; the other teammates are strangers.

How ranked drafts work (Max, 2026-10-23):
  bans    the highest-ranked player of each team makes the team's bans, whatever the pick
          order. Rank = as-of MMR (p3t_data.ranks: Heroes Profile's causal rating; lobby
          average when missing). The party holds the bans iff a member is the team's
          top-ranked player.
  picks   on each of the team's pick turns any teammate may lock a hero; whoever locks first
          takes the turn. Pick timing is the user's choice, not an assigned slot.

Simulation:
  timing      fixed-timing plans: the party always locks at the chosen team turns; the
              strangers lock in a uniformly random order on the other turns. The lock-or-wait
              arm (solo): at each team pick turn the drafter decides whether the user locks now
              or waits (then the next stranger in the random order locks).
  strangers   pick from the personalized generic drafter: DraftRec's pick head for that
              player (his history, the draft so far, the turn), sampled at temperature 1.
  opponents   the same model for each opponent, in the real game's order.
  bans        the holder's model (p3t_oppdr.py holderfit): softmax(w0 log p_GD + a1 log r +
              a2 log(1 - q_holder)), r = DraftRec's mixture for the other team's next pick
              (denial), q_holder = the holder's own DraftRec pick distribution at his team's
              next pick turn (he does not ban what he wants to play); sampled. Used for every
              ban the party does not make (stranger holder, opponents' holder) and for the
              generic arm's bans.
  search      cuda_explore v2 chance search, slot mode. The drafter does not know the
              strangers' or the opponents' future order: each future stranger (opponent) pick
              step is a chance node whose distribution is DraftRec's mixture over the team's
              players still to pick, read at the decision state, credited to a planning
              assignment of those players (drawn independently of the realized order). A
              stranger holder's remaining ban is a chance node with the holder model's row.
              Opponent bans in the search: GD chance nodes. Lock-or-wait compares the root
              value of "lock now" (decision root) with the best "wait, lock at a later team
              turn c" (chance root: the next stranger's pick, kernel flag on the root step).

Arms for the party's seats (common random numbers per lobby and step):
  aware    the selected design (x-combo1-dr-chance)
  blind    the population-only search (x-blind-s400), strangers and opponents GD chance nodes
  generic  the party picks like strangers (DraftRec sample); bans by the holder model
  real     each member's real hero when still legal, else as generic; the real team's bans
  lock     (lock plan) aware with lock-or-wait timing

Plans: effect (party sizes 1-5, random members and turns, 4 arms; the party holds the bans
when a member is the top-ranked player), grid (every set of team turns for every party size,
both sides, with the top-ranked player in the party ("ban") or not ("noban"); aware and
generic), lock (solo, both sides, ban / noban; the lock arm; compared with the grid's
always-first and always-last on common random numbers).

Outputs: cache p3t/party[_<set>]/<config>__<arm>.npz, results p3t/party[_<set>].json.
Usage (GPU host, from training/): python personalization/p3t_party.py run --plan effect|grid|lock [--only NAME] [--shard i/n]
                                 P3T_LOBBYSET=party python personalization/p3t_party.py select (CPU host)
                                 python personalization/p3t_party.py score
"""
import argparse
import hashlib
import itertools
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
ARMS_OF = {"effect": ("aware", "blind", "generic", "real"), "grid": ("aware", "generic"), "lock": ("lock",)}
DESIGN = "x-combo1-dr-chance"
# lock-or-wait rule, FROZEN 2026-10-23 (chosen on the 300 tournament lobbies, which the party set
# excludes): mode lookall, LOOK_K 4, value "best". "lookall" (default): lock now iff the chosen hero's value now >= the best later
# team turn's value, each the mean over LOOK_K sampled futures of the chosen hero's value at that turn
# (like with like). "lookahead": the next team turn only. "chance" (the first version): each later turn
# valued by a search rooted at the next stranger's chance node. On the 300 tournament lobbies (smoke,
# 2026-10-22) chance and lookahead lock too early (always-last beats them).
LOCK_MODE = os.environ.get("P3T_LOCK_MODE", "lookall")
LOOK_K = int(os.environ.get("P3T_LOOK_K", "4"))
LOCK_VALUE = os.environ.get("P3T_LOCK_VALUE", "mean" if LOCK_MODE == "chance" else "best")
BLIND = "x-blind-s400"
LS = os.environ.get("P3T_LOBBYSET", "")
OUT = os.path.join(TD.OUT, "party" + (f"_{LS}" if LS else ""))
RES_NAME = "party" + (f"_{LS}" if LS else "")
# the party study's own lobby set (P3T_LOBBYSET=party), sized for slot-level CIs with both team
# sides pooled (coordinator, 2026-10-22): up to 1,800 lobbies, the tournament's strict judge rule only
N_PARTY = 1800            # target lobbies; fewer if fewer meet the rule (all of them are used)
N_PARTY_FLOOR = 1500      # below this, select flags the set for the coordinator
J_STRICT = 10             # judge-window games per player (the tournament's rule)


def plan(name):
    """Configurations: dict(name, t, m, turns (indices 0-4 into the team's turns; None =
    random; "lock" = lock-or-wait), hold (None = as drawn, "ban" = the top-ranked player is in
    the party, "noban" = he is not), crn (the common-random-number key: configurations with the
    same key share members, stranger order and draws))."""
    out = []
    if name == "effect":
        for m in (1, 2, 3, 4, 5):
            for t in (0, 1):
                out.append(dict(name=f"eff_m{m}_t{t}", t=t, m=m, turns=None, hold=None, crn=f"eff_m{m}_t{t}"))
    if name == "grid":
        for m in (1, 2, 3, 4, 5):
            for turns in itertools.combinations(range(5), m):
                tag = "".join(map(str, turns))
                for t in (0, 1):
                    for hold in ("ban", "noban"):
                        if m == 5 and hold == "noban":
                            continue
                        out.append(dict(name=f"grid_m{m}_{tag}_t{t}_{hold}", t=t, m=m, turns=list(turns), hold=hold,
                                        crn=f"m{m}_t{t}_{hold}"))
    if name == "lock":
        for t in (0, 1):
            for hold in ("ban", "noban"):
                out.append(dict(name=f"lock_m1_t{t}_{hold}", t=t, m=1, turns="lock", hold=hold, crn=f"m1_t{t}_{hold}"))
    return out


def _seed(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:7], 16)


def ranks(LB):
    """Per kernel slot: rank (as-of MMR), real party id; top (n, 2): each kernel team's
    top-ranked slot (the ban holder)."""
    z = np.load(os.path.join(TD.OUT, "lobby_ranks.npz"))
    r = np.take_along_axis(z["rank"][LB.gi], LB.src, 1)
    pid = np.take_along_axis(z["party"][LB.gi], LB.src, 1)
    top = np.stack([r[:, :5].argmax(1), 5 + r[:, 5:].argmax(1)], 1)
    return top, r, pid


def layout(LB, cfg, top):
    """Per lobby: turn_of (n, 10) global turn of each kernel slot (party and strangers of team
    t; the opponents keep the real order), party (n, 10) bool, ban_ctl (n,) the party holds the
    bans, sord (n, 5) team t's slots in stranger locking order (lock plan: the 4 strangers,
    then the user). Same draws for every configuration sharing cfg["crn"]."""
    t, m, turns, hold = cfg["t"], cfg["m"], cfg["turns"], cfg["hold"]
    n = LB.n
    turn_of = LB.src.copy()
    party = np.zeros((n, 10), bool)
    sord = np.zeros((n, 5), np.int64)
    seed = _seed(f"party|{cfg['crn']}")
    T = 5 * t + np.arange(5)
    tt = TEAM_TURNS[t]
    for i in range(n):
        rng = np.random.RandomState(seed + i)
        p1, p2 = rng.permutation(T), rng.permutation(5)
        h = int(top[i, t])
        oth = [int(s) for s in p1 if s != h]
        mem = [int(s) for s in p1[:m]] if hold is None else ([h] + oth[:m - 1] if hold == "ban" else oth[:m])
        mem = [mem[q] for q in np.argsort(p2[:m], kind="stable")]       # members' order onto their turns
        strangers = [int(s) for s in p1 if int(s) not in mem]
        party[i, mem] = True
        if turns is None:
            for q, sl in enumerate(T[p2]):
                turn_of[i, sl] = tt[q]
            sord[i] = T[p2]
        elif turns == "lock":
            for q, sl in enumerate(strangers + mem):                     # provisional: the user last
                turn_of[i, sl] = tt[q]
            sord[i] = strangers + mem
        else:
            rest = [q for q in range(5) if q not in turns]
            for q, sl in zip(turns, mem):
                turn_of[i, sl] = tt[q]
            for q, sl in zip(rest, strangers):
                turn_of[i, sl] = tt[q]
            sord[i] = strangers + mem
    ban_ctl = party[np.arange(n), top[:, t]]
    return turn_of, party, ban_ctl, sord


class KnownState:
    """DraftRec with the true player of every pick: each player sits at his own turn in the
    match network, the acting player is the query. place() moves a not-yet-picked player to a
    not-yet-taken turn of his team (whoever locks first takes the turn)."""

    def __init__(self, inf, gi, src, mp, tr, turn_of):
        import torch
        self.torch = torch
        self.m, self.dev = inf["model"], inf["device"]
        rows = np.array([inf["pos"][int(g)] for g in gi])
        self.U = inf["U"][torch.as_tensor(rows, device=self.dev)[:, None], torch.as_tensor(src, device=self.dev)]
        self.mp = torch.as_tensor(np.asarray(mp, np.int64), device=self.dev)
        self.tr = torch.as_tensor(np.asarray(tr, np.int64), device=self.dev)
        self.B = len(gi)
        self.turn_of = np.array(turn_of)
        self.assigned = np.full((self.B, 10), -1, np.int64)
        self.taken = np.zeros((self.B, H), bool)
        self.perm = np.argsort(self.turn_of, 1)            # slot at each global turn

    def subset(self, idx):
        o = KnownState.__new__(KnownState)
        o.torch, o.m, o.dev = self.torch, self.m, self.dev
        ti = self.torch.as_tensor(idx, device=self.dev)
        o.U, o.mp, o.tr = self.U[ti], self.mp[ti], self.tr[ti]
        o.B = len(idx)
        o.turn_of, o.assigned, o.taken, o.perm = (self.turn_of[idx].copy(), self.assigned[idx].copy(),
                                                  self.taken[idx].copy(), self.perm[idx].copy())
        return o

    def place(self, j, slots, rows=None):
        r = np.arange(self.B) if rows is None else np.asarray(rows)
        slots = np.asarray(slots)
        occ = self.perm[r, j].copy()
        tp = self.turn_of[r, slots].copy()
        self.perm[r, tp] = occ
        self.perm[r, j] = slots
        self.turn_of[r, occ] = tp
        self.turn_of[r, slots] = j

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

    def dist_as(self, j, slots):
        """dist(j) with player slots[i] moved to turn j (both not yet picked)."""
        perm, tof = self.perm.copy(), self.turn_of.copy()
        self.place(j, slots)
        p = self.dist(j)
        self.perm, self.turn_of = perm, tof
        return p

    def mix(self, j, cand):
        """(B, H) mean of dist_as(j, u) over the candidate slots u (cand (B, 10) bool, players
        not yet picked of turn j's team): the turn's pick when any of them may lock it."""
        out = np.zeros((self.B, H))
        c = cand.sum(1)
        rows, us = np.nonzero(cand)
        for b0 in range(0, len(rows), 4096):                 # one batched pass per 4,096 (lobby, player) pairs
            r_, u_ = rows[b0:b0 + 4096], us[b0:b0 + 4096]
            np.add.at(out, r_, self.subset(r_).dist_as(j, u_))
        return (out / np.maximum(c, 1)[:, None]).astype(np.float32)

    def free(self, team):
        """(B, 10) bool: players of `team` not yet picked."""
        f = self.assigned < 0
        f[:, 5 * (1 - team):5 * (1 - team) + 5] = False
        return f

    def pick(self, j, hero):
        sl = self.perm[:, j]
        self.assigned[np.arange(self.B), sl] = hero
        self.taken[np.arange(self.B), hero] = True

    def ban(self, hero):
        self.taken[np.arange(self.B), hero] = True


def sample_u(p, legal, u):
    """Inverse-CDF sample of p restricted to legal heroes with uniforms u (one per row)."""
    q = np.where(legal, p, 0.0)
    s = q.sum(1, keepdims=True)
    q = np.where(s > 0, q / np.maximum(s, 1e-12), legal / legal.sum(1, keepdims=True))
    return np.minimum((q.cumsum(1) < u[:, None]).sum(1), H - 1)


def holder_terms(KS, k, holder, wh):
    """(B, H) holder-model terms of a ban at step k (without the GD part):
    a1 log r (denial: the other team's next pick, mixture over its players still to pick) +
    a2 log(1 - q_holder) (the holder's own pick at his team's next turn, if he has not picked)."""
    import p3t_oppdr as OD
    tt = DRAFT_TEAM[k]
    z = np.zeros((KS.B, H), np.float32)
    jr = OD.next_pick_turn(1 - tt, k)
    if jr is not None and wh["a1"]:
        z += float(wh["a1"]) * np.log(np.maximum(KS.mix(jr, KS.free(1 - tt)), OD.DR_FLOOR))
    jh = OD.next_pick_turn(tt, k)
    if jh is not None and wh["a2"]:
        hf = np.flatnonzero(KS.assigned[np.arange(KS.B), holder] < 0)
        if len(hf):
            sub = KS.subset(hf)
            qh = sub.dist_as(jh, holder[hf])
            z[hf] += float(wh["a2"]) * np.log(np.maximum(1.0 - qh, OD.DR_FLOOR))
    return z


def holder_sample(LB, KS, acts, k, holder, legal, u, wh, v):
    import p3t_oppdr as OD
    import p3t_systems as SY
    tt = DRAFT_TEAM[k]
    X = SY.features(acts, k, LB.map, LB.tier, tt)
    lg = np.log(np.maximum(SY.gd_probs(v, X[:, :289], legal), OD.GD_FLOOR))
    z = float(wh["w0"]) * lg + holder_terms(KS, k, holder, wh)
    z = np.where(legal, z, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return sample_u(p / p.sum(1, keepdims=True), legal, u)


def step_mixes(KS, k, t, party, top, ban_ctl, w, wh):
    """Chance-node rows read at the decision state (step k, team t), for all lobbies:
    opp[j] the opponents' mixture at their future turn j, str_[j] the strangers' mixture at our
    turn j >= turn of k, hrow the stranger holder's row for our remaining ban (or None)."""
    import p3t_oppdr as OD
    o = 1 - t
    lp = float(w["lp"])
    fo = KS.free(o)
    fs = KS.free(t) & ~party
    opp = {j: lp * np.log(np.maximum(KS.mix(j, fo), OD.DR_FLOOR)) for j in range(10)
           if PICK_TEAM[j] == o and PICK_STEPS[j] > k}
    str_ = {j: lp * np.log(np.maximum(KS.mix(j, fs), OD.DR_FLOOR)) for j in TEAM_TURNS[t]
            if PICK_STEPS[j] >= k and fs.any()}
    bans = [s for s in range(k + 1, 16) if DRAFT_TEAM[s] == t and not IS_PICK[s]]
    assert len(bans) <= 1 or not IS_PICK[k], "more than one later ban of ours at a pick decision"
    hrow = None
    if bans and (~ban_ctl).any():
        hrow = holder_terms(KS, bans[0], top[:, t], wh)
    return {"opp": opp, "str": str_, "hrow": hrow, "ban": bans[0] if bans else None}


def plan_rows(KS, k, t, party, pslot, root_chance, ban_ctl, free_row, mx, pseed):
    """Search rows for lobbies KS (a subset; mx rows already subset). pslot (B, 10): the party
    slot planned at each global turn (-1 = not the party). Returns imit (B, 10, H), slot_of
    (B, 16), stranger (B, 16)."""
    B = KS.B
    o = 1 - t
    prng = np.random.RandomState(pseed)     # the same planning draws for every search at a step
    imit = np.zeros((B, 10, H), np.float32)
    so = np.full((B, 16), -1, np.int64)
    st = np.zeros((B, 16), bool)
    for s in PICK_STEPS:
        if s < k:
            so[:, s] = KS.perm[:, TURN[s]]
    fo = KS.free(o)
    fs = KS.free(t) & ~party
    first = k if root_chance else k + 1
    osteps = [s for s in PICK_STEPS if s > k and DRAFT_TEAM[s] == o]
    msteps = [s for s in PICK_STEPS if s >= first and DRAFT_TEAM[s] == t]
    for i in range(B):
        # planning assignment of the players still to pick (independent of the realized order)
        ol = list(prng.permutation(np.flatnonzero(fo[i])))
        for s in osteps:
            sl = ol.pop(0)
            so[i, s] = sl
            imit[i, sl] = mx["opp"][TURN[s]][i]
        sl_ = list(prng.permutation(np.flatnonzero(fs[i])))
        for s in msteps:
            ps = pslot[i, TURN[s]]
            if ps >= 0:
                so[i, s] = ps
                continue
            st[i, s] = True
            sl = sl_.pop(0)
            so[i, s] = sl
            imit[i, sl] = mx["str"][TURN[s]][i]
        if not root_chance and IS_PICK[k]:
            so[i, k] = pslot[i, TURN[k]]
        s_b = mx["ban"]
        if s_b is not None and s_b >= first and not ban_ctl[i]:
            st[i, s_b] = True
            so[i, s_b] = free_row[i]
            imit[i, free_row[i]] = mx["hrow"][i]
    if root_chance and not IS_PICK[k]:
        raise ValueError("chance root only at pick steps")
    return imit, so, st


def search(LB, KS, acts, k, t, idx, arm, party, pslot, root_chance, ban_ctl, free_row, mx, pseed, kseed, device):
    """One decide-only search for lobbies idx (mx: step_mixes of those lobbies); returns
    (action, root value)."""
    import p3t_systems as SY
    import p3t_tourney as TT
    sub = KS.subset(idx)
    imit, so, st = plan_rows(sub, k, t, party[idx], pslot[idx], root_chance, ban_ctl[idx], free_row[idx], mx, pseed)
    sysname = BLIND if arm == "blind" else DESIGN
    ctx = TT.ctx_for(LB, idx, acts, k, sysname, kseed=kseed)
    ctx["slot_of"] = so.astype(np.int32)
    ctx["stranger"] = st
    ctx["want_value"] = LOCK_VALUE if arm == "lock" else True
    x = SY.SYSTEMS[sysname]["x"]
    if arm != "blind":
        import p3t_oppdr as OD
        ctx["imit_override"] = (imit, json.load(open(OD.out_file()))["w0"])
    return SY.act_mcts_x(SY.vintage(SY.SYSTEMS[sysname]["vint"]), ctx, x, device=device)


def draft(LB, cfg, arm, top, device=0):
    import p3t_draftrec as DRC
    import p3t_oppdr as OD
    import p3t_systems as SY
    t = cfg["t"]
    n = LB.n
    turn_of, party, ban_ctl, sord = layout(LB, cfg, top)
    inf = DRC.load_inference(device=None, hist_file=OD.hist_for_lobbyset())
    KS = KnownState(inf, LB.gi, LB.src, LB.map, LB.tier, turn_of)
    w = json.load(open(OD.out_file()))
    wh = json.load(open(OD.holder_file()))
    v = SY.vintage("cur")
    acts = np.full((n, 16), -1, np.int64)
    ar = np.arange(n)
    lock = cfg["turns"] == "lock"
    user = sord[:, 4] if lock else None
    lock_turn = np.full(n, -1, np.int64)
    seed = _seed(f"partyrng|{cfg['crn']}")
    secs = 0.0
    searching = arm in ("aware", "blind", "lock")
    for k in range(16):
        rng = np.random.RandomState(seed + 1000 * k)
        prng = np.random.RandomState(seed + 1000 * k + 29)        # the lock arm's planned turn at its bans
        u = rng.random_sample(n)
        team = DRAFT_TEAM[k]
        ours = team == t
        legal = SY.legal_mask(acts, k, LB.team_pool)
        kseed = seed + 7919 * k
        pseed = seed + 1000 * k + 17
        out = np.full(n, -1, np.int64)

        def mixes(idx):
            return step_mixes(KS.subset(idx), k, t, party[idx], top[idx], ban_ctl[idx], w, wh)
        if IS_PICK[k]:
            j = TURN[k]
            if lock and ours:
                todo = np.flatnonzero(lock_turn < 0)
                later = [jj for jj in TEAM_TURNS[t] if jj > j]
                now = np.zeros(n, bool)
                hero_now = np.full(n, -1, np.int64)
                if len(todo):
                    t0 = time.time()
                    mx = mixes(todo)
                    ps = np.full((n, 10), -1, np.int64)
                    ps[todo, j] = user[todo]
                    h_, v_now = search(LB, KS, acts, k, t, todo, "lock", party, ps, False, ban_ctl, user, mx,
                                       pseed, kseed, device)
                    hero_now[todo] = h_
                    if LOCK_MODE == "chance":
                        # one search per later turn c, rooted at this step as the next stranger's chance node
                        v_wait = np.full(len(todo), -np.inf)
                        for c in later:
                            ps = np.full((n, 10), -1, np.int64)
                            ps[todo, c] = user[todo]
                            _, vc = search(LB, KS, acts, k, t, todo, "lock", party, ps, True, ban_ctl, user, mx,
                                           pseed, kseed, device)
                            v_wait = np.maximum(v_wait, vc)
                    elif later:
                        tg = later[:1] if LOCK_MODE == "lookahead" else later
                        v_wait = lookahead(LB, KS, acts, k, t, user, party, top, ban_ctl, todo, tg, w, wh, v,
                                           seed + 104729 * k, device)[todo].max(1)
                    else:
                        v_wait = np.full(len(todo), -np.inf)
                    now[todo] = v_now >= v_wait
                    secs += time.time() - t0
                # who takes this turn: the user if he locks now, else the next stranger in order
                act = np.array([user[i] if now[i] else next(int(s) for s in sord[i, :4] if KS.assigned[i, s] < 0)
                                for i in range(n)])
                KS.place(j, act)
                lock_turn[now] = j
                out[now] = hero_now[now]
            actor = KS.perm[:, j]
            p = KS.dist(j)
            drs = sample_u(p, legal, u)
            dec = ours & party[ar, actor] & (out < 0)
        else:
            dec = (ban_ctl.copy() if ours else np.zeros(n, bool))
            hb = holder_sample(LB, KS, acts, k, top[:, team], legal, u, wh, v)
        if ours and searching and dec.any():
            idx = np.flatnonzero(dec)
            ps = np.full((n, 10), -1, np.int64)
            for jj in TEAM_TURNS[t]:
                sl = KS.perm[:, jj]
                ps[:, jj] = np.where(party[ar, sl], sl, -1)
            if lock:
                # bans before the user locks: plan his lock at a random remaining team turn
                for i in idx:
                    if lock_turn[i] < 0:
                        ps[i, :] = -1
                        rem = [jj for jj in TEAM_TURNS[t] if PICK_STEPS[jj] > k]
                        ps[i, rem[prng.randint(len(rem))]] = user[i]
            fr = np.array([np.flatnonzero(party[i])[0] for i in range(n)])
            if IS_PICK[k]:
                fr = np.where(dec, KS.perm[:, TURN[k]], fr)
            t0 = time.time()
            res = search(LB, KS, acts, k, t, idx, "blind" if arm == "blind" else "aware", party, ps, False,
                         ban_ctl, fr, mixes(idx), pseed, kseed, device)
            out[idx] = res[0]
            secs += time.time() - t0
        if ours and arm == "real" and dec.any():
            real = LB.real_acts[:, k] if not IS_PICK[k] else real_hero(LB, k, KS.perm[:, TURN[k]])
            ok = dec & (real >= 0) & legal[ar, np.maximum(real, 0)]
            out[ok] = real[ok]
        rest = out < 0
        out[rest] = (drs if IS_PICK[k] else hb)[rest]
        assert legal[ar, out].all(), f"{cfg['name']} {arm} step {k}: illegal action"
        acts[:, k] = out
        if IS_PICK[k]:
            KS.pick(TURN[k], out)
        else:
            KS.ban(out)
    slot_at = np.full((n, 16), -1, np.int64)
    for k in PICK_STEPS:
        slot_at[:, k] = KS.perm[:, TURN[k]]
    return acts, slot_at, party, ban_ctl, lock_turn, secs


def lookahead(LB, KS, acts, k, t, user, party, top, ban_ctl, todo, targets, w, wh, v, seed, device, K=None):
    """Value of waiting (the lock-or-wait arm): K sampled futures from step k on (a stranger takes
    turn k; opponents' and strangers' picks: a random not-yet-picked player of the team, his
    DraftRec pick sampled; bans: the holder model); along each future, the user's decision search
    at each target team turn. Returns (n, len(targets)) mean chosen-hero values (the same
    estimator as "lock now")."""
    import p3t_systems as SY
    K = K or LOOK_K
    n = LB.n
    val = np.zeros((n, len(targets)))
    for r in range(K):
        rng = np.random.RandomState(seed + 31 * r)
        S = KS.subset(np.arange(n))
        a = acts.copy()
        s = k
        for ci, jn in enumerate(targets):
            kn = PICK_STEPS[jn]
            while s < kn:
                legal = SY.legal_mask(a, s, LB.team_pool)
                u = rng.random_sample(n)
                if IS_PICK[s]:
                    tm = DRAFT_TEAM[s]
                    cand = S.free(tm) & (~party if tm == t else True)
                    S.place(TURN[s], np.array([rng.choice(np.flatnonzero(cand[i])) for i in range(n)]))
                    h = sample_u(S.dist(TURN[s]), legal, u)
                    S.pick(TURN[s], h)
                else:
                    h = holder_sample(LB, S, a, s, top[:, DRAFT_TEAM[s]], legal, u, wh, v)
                    S.ban(h)
                a[:, s] = h
                s += 1
            ps = np.full((n, 10), -1, np.int64)
            ps[todo, jn] = user[todo]
            mx = step_mixes(S.subset(todo), kn, t, party[todo], top[todo], ban_ctl[todo], w, wh)
            _, vv = search(LB, S, a, kn, t, todo, "lock", party, ps, False, ban_ctl, user, mx, seed + 17 + 31 * r,
                           seed + 7919 * kn, device)
            val[todo, ci] += vv / K
    return val


def real_hero(LB, k, actor):
    """The hero each acting party member played in the real game (his kernel slot's real pick)."""
    n = LB.n
    out = np.full(n, -1, np.int64)
    for i in range(n):
        for kk in PICK_STEPS:
            if LB.slot_of[i, kk] == actor[i]:
                out[i] = LB.real_acts[i, kk]
    return out


def stage_run(plan_name, only=None, device=0, shard="0/1"):
    import p3t_tourney as TT
    LB = TT.Lobbies()
    top, _, _ = ranks(LB)
    os.makedirs(OUT, exist_ok=True)
    si, sn = (int(x) for x in shard.split("/"))
    for ci, cfg in enumerate(plan(plan_name)):
        if only and cfg["name"] not in only.split(","):
            continue
        if ci % sn != si:
            continue
        for arm in ARMS_OF[plan_name]:
            f = os.path.join(OUT, f"{cfg['name']}__{arm}.npz")
            if os.path.exists(f):
                continue
            t0 = time.time()
            acts, slot_at, party, ban_ctl, lock_turn, secs = draft(LB, cfg, arm, top, device)
            np.savez(f + ".tmp.npz", acts=acts, slot_at=slot_at, party=party, ban_ctl=ban_ctl, lock_turn=lock_turn,
                     t=cfg["t"], m=cfg["m"], arm=arm, hold=str(cfg["hold"]), search_secs=secs)
            os.replace(f + ".tmp.npz", f)
            print(f"  {cfg['name']} {arm}: {time.time() - t0:.0f}s (search {secs:.0f}s)", flush=True)
    print("PARTY RUN DONE", flush=True)


# ------------------------------------------------------------------ lobby set

def stage_select():
    """The party study's lobby set: real tournament-window lobbies (2026-05-23 to 06-30, never the
    sealed build) whose ten players all have 50+ earlier games and J_STRICT+ judge-window games (the
    tournament's rule). N_PARTY drawn in a fixed random order, or every eligible lobby if fewer. The
    tune lobbies (they chose the design) and the tournament lobbies (the lock rule was chosen on
    them) are left out.
    Written before the probe, player-prior and opponent-model lobbies are drawn, which then avoid
    this set (p3t_data.eval_lobbies)."""
    from shared import MAPS
    L = TD.load_lobbies()
    cur = np.load(os.path.join(TD.OUT, "state_cur.npz"))
    jd = np.load(os.path.join(TD.OUT, "state_judge.npz"))
    win = np.flatnonzero(L["day"] <= TD.LOBBY_END)
    assert (jd["gsel"] == win).all()
    rich = (cur["n"].astype(np.float64).sum(1) >= 50).reshape(-1, 10).all(1)[win] & np.isin(L["map"][win], MAPS)
    jn = jd["rec_n"].astype(np.float64).sum(1).reshape(-1, 10).min(1)
    # left out: the tune lobbies (they chose the design) and the tournament lobbies (the lock-or-wait
    # rule was chosen on them, coordinator 2026-10-23)
    excl = set()
    for sub in ("tourney_tune", "tourney"):
        f_ = os.path.join(TD.OUT, sub, "lobbies.npz")
        if os.path.exists(f_):
            excl |= set(np.load(f_)["gi"].tolist())
        elif sub == "tourney":
            raise FileNotFoundError(f"{f_}: the tournament lobbies must be selected before the party set")
    # and the frozen pre-definitive tune and tournament lobbies, by replay ID (p3t_excluded_replays.json)
    frozen = TD.gi_of_replays(L, TD.excluded_replays("tune", "tournament"))
    excl |= frozen
    keep = np.array([g not in excl for g in win])
    strict = win[rich & keep & (jn >= J_STRICT)]
    sel = np.sort(np.random.RandomState(20261022).permutation(strict)[:N_PARTY])
    assert not set(sel.tolist()) & excl, "party lobbies overlap an excluded set"
    assert not set(L["replay_id"][sel].tolist()) & TD.excluded_replays("tune", "tournament")
    d = os.path.join(TD.OUT, "tourney_party")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "lobbies.npz"), gi=sel)
    info = {"excluded_tune_and_tournament": int(len(excl)), "frozen_in_lobby_table": int(len(frozen)), "strict_available": int(len(strict)), "selected": int(len(sel)), "target": N_PARTY,
            "below_floor": bool(len(sel) < N_PARTY_FLOOR), "floor": N_PARTY_FLOOR}
    json.dump(info, open(os.path.join(d, "select.json"), "w"))
    print(f"party lobbies: {info}", flush=True)
    if info["below_floor"]:
        print(f"WARNING party lobby set below {N_PARTY_FLOOR}: report to the coordinator", flush=True)


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
    import p3t_judge as JG
    CAL_A, CAL_K = JG.cal_coefs(judge)
    pc = np.clip(sc["consensus"], 1e-6, 1 - 1e-6)
    lc = np.log(pc / (1 - pc))
    pers = judge["c2_skill"] * sc["dS_true"] + judge["c3_offrole"] * sc["dO_true"]
    jt = 1 / (1 + np.exp(-(lc + pers)))
    cal = 1 / (1 + np.exp(-(CAL_A * lc + CAL_K * pers)))
    f = (lambda x: x) if t == 0 else (lambda x: 1 - x)
    return {"J": 100 * f(jt), "J_best": 100 * f(sc["judge_personal"]), "J_cal": 100 * f(cal),
            "CONS": 100 * f(sc["consensus"]), "record": 100 * sc["rec_true"][:, t]}


def cluster_ci(M, B=2000, seed=0):
    """M (n lobbies, k configs): per-lobby mean over the configs, then a lobby bootstrap."""
    M = np.atleast_2d(np.asarray(M, float).T).T if np.ndim(M) == 1 else np.asarray(M, float)
    with np.errstate(invalid="ignore"):
        d = np.nanmean(M, 1)
    return boot_ci(d, B, seed)


def boot_ci(d, B=2000, seed=0):
    d = d[np.isfinite(d)]
    rng = np.random.RandomState(seed)
    bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)])
    return [round(float(d.mean()), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]




def real_rates(gi):
    """How often the ban holder (the team's top-ranked player) is a solo player / in a real
    party of size m, in real lobbies gi: per real party size (Heroes Profile party id; 0 = solo)
    the number of groups and the share holding the bans."""
    z = np.load(os.path.join(TD.OUT, "lobby_ranks.npz"))
    L = TD.load_lobbies()
    rk, pt = z["rank"][gi], z["party"][gi]
    tm = L["steps"][gi][:, PICK_STEPS, 2]
    cnt = {m: [0, 0] for m in range(1, 6)}
    for i in range(len(gi)):
        for team in (0, 1):
            js = np.flatnonzero(tm[i] == team)
            top = js[np.argmax(rk[i, js])]
            groups = {}
            for j in js:
                groups.setdefault(("solo", j) if pt[i, j] == 0 else ("party", pt[i, j]), []).append(j)
            for g in groups.values():
                m = min(len(g), 5)
                cnt[m][0] += 1
                cnt[m][1] += int(top in g)
    return {str(m): {"groups": c[0], "rate": round(c[1] / c[0], 4) if c[0] else None} for m, c in cnt.items()}


def _seats(views, fmt, a1, a2):
    """{metric: (n, 2)} differences arm a1 - arm a2 of configuration fmt (with {t}) on both seats."""
    ks = [(fmt.format(t=t) + "__" + a1, fmt.format(t=t) + "__" + a2) for t in (0, 1)]
    if not all(k1 in views and k2 in views for k1, k2 in ks):
        return None
    return {m: np.stack([views[k1][m] - views[k2][m] for k1, k2 in ks], 1) for m in views[ks[0][0]]}


def _pair(views, n1, n2):
    """{metric: (n, 2)} n1 - n2 (full names with {t} and arm), both seats."""
    ks = [(n1.format(t=t), n2.format(t=t)) for t in (0, 1)]
    if not all(k1 in views and k2 in views for k1, k2 in ks):
        return None
    return {m: np.stack([views[k1][m] - views[k2][m] for k1, k2 in ks], 1) for m in views[ks[0][0]]}


def _mean_d(ds):
    ds = [d for d in ds if d is not None]
    if not ds:
        return None
    with np.errstate(invalid="ignore"):
        return {m: np.nanmean(np.stack([d[m] for d in ds], 0), 0) for m in ds[0]}


def _lobby(d):
    with np.errstate(invalid="ignore"):
        return {m: np.nanmean(v, 1) for m, v in d.items()}


def split_ci(D, Hm, r, B=2000, seed=0):
    """D (n, 2) per lobby and seat, Hm (n, 2) the party holds the bans: means for holders, for
    non-holders and r-weighted (r = the real rate of holding), lobby bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(D)

    def stats(ix):
        d, h = D[ix].ravel(), Hm[ix].ravel()
        ok = np.isfinite(d)
        a = d[ok & h].mean() if (ok & h).any() else np.nan
        b = d[ok & ~h].mean() if (ok & ~h).any() else np.nan
        return a, b, r * a + (1 - r) * b
    est = stats(np.arange(n))
    bs = np.array([stats(rng.randint(0, n, n)) for _ in range(B)])
    out = {}
    for c, name in enumerate(("holds bans", "no bans", "weighted by the real rate")):
        if np.isfinite(est[c]):
            out[name] = [round(float(est[c]), 2), round(float(np.nanpercentile(bs[:, c], 2.5)), 2),
                         round(float(np.nanpercentile(bs[:, c], 97.5)), 2)]
    out["share holding"] = round(float(Hm.mean()), 4)
    return out


def stage_score():
    import re
    import p3t_judge as JG
    import p3t_tourney as TT
    LB = TT.Lobbies()
    judge = json.load(open(os.path.join(JG.RES, JG.JNAME)))
    truth = JG.Truth()
    sdir = os.path.join(OUT, "scores")
    os.makedirs(sdir, exist_ok=True)
    views, hold, lockt = {}, {}, {}
    key = JG.score_key(LB, os.path.join(TT.TDIR, "lobbies.npz"))      # judge, states, lobbies, population
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".npz") or "__" not in f or ".tmp" in f:
            continue
        sp = os.path.join(sdir, f)
        z = np.load(os.path.join(OUT, f))
        stale = not os.path.exists(sp) or os.path.getmtime(sp) < os.path.getmtime(os.path.join(OUT, f))
        if not stale:
            with np.load(sp) as old:
                stale = "score_key" not in old.files or str(old["score_key"]) != key
        if stale:
            sc = JG.score_drafts(LB, z["acts"], judge, truth)
            sc["dS_true"], sc["dO_true"], sc["rec_true"] = true_terms(LB, z["acts"], z["slot_at"])
            sc["score_key"] = np.array(key)
            np.savez(sp, **sc)
        sc = dict(np.load(sp))
        views[f[:-4]] = team_view(sc, int(z["t"]), judge)
        hold[f[:-4]] = z["ban_ctl"]
        if "lock_turn" in z.files:
            lockt[f[:-4]] = z["lock_turn"]
    win = np.flatnonzero(TD.load_lobbies()["day"] <= TD.LOBBY_END)
    rates = {"study lobbies": real_rates(LB.gi), "all tournament-window lobbies": real_rates(win)}
    rw = {int(m): v_["rate"] for m, v_ in rates["all tournament-window lobbies"].items()}
    res = {"configs": {}, "contrasts": {}, "real_ban_rates": rates, "weights": "real rates of all tournament-window lobbies"}
    for k, v in views.items():
        res["configs"][k] = {m: boot_ci(x) for m, x in v.items()}
    C_ = res["contrasts"]
    base = sorted({k.split("__")[0] for k in views})
    pairs = (("aware", "generic"), ("aware", "blind"), ("aware", "real"), ("blind", "generic"), ("real", "generic"))
    for b in base:
        for a1, a2 in pairs:
            k1, k2 = f"{b}__{a1}", f"{b}__{a2}"
            if k1 in views and k2 in views:
                C_[f"{b}: {a1} - {a2}"] = {m: boot_ci(views[k1][m] - views[k2][m]) for m in views[k1]}
    # seats pooled (the first- and second-pick team): per-lobby mean of the two seats, lobby bootstrap
    groups = {}
    for b_ in base:
        mt = re.search(r"_t([01])(?=_|$)", b_)
        groups.setdefault(b_[:mt.start()] + "_t{t}" + b_[mt.end():], None)
    for fmt in sorted(groups):
        for a1, a2 in pairs[:3]:
            d = _seats(views, fmt, a1, a2)
            if d is not None:
                C_[f"{fmt.replace('_t{t}', '')} (both seats): {a1} - {a2}"] = {m: cluster_ci(v) for m, v in d.items()}
    # effectiveness plan: the party holds the bans when a member is the top-ranked player (rate m/5 by
    # construction); gains split by holding, and weighted by the real rate of holding
    for m in range(1, 6):
        for a1, a2 in pairs[:3]:
            d = _seats(views, f"eff_m{m}_t{{t}}", a1, a2)
            if d is None:
                continue
            Hm = np.stack([hold[f"eff_m{m}_t{t}__{a1}"] for t in (0, 1)], 1)
            C_[f"eff_m{m} (both seats) by bans: {a1} - {a2}"] = {mt: split_ci(v, Hm, rw[m]) for mt, v in d.items()}
    # grid: per slot set and seats pooled, with ("ban") and without ("noban") the top-ranked player,
    # and weighted by the real rate; pooled over the slot sets of a party size
    tags = {}
    for b_ in base:
        mt = re.match(r"grid_m(\d)_(\d+)_t([01])_(ban|noban)$", b_)
        if mt:
            tags.setdefault(int(mt.group(1)), set()).add(mt.group(2))
    wpool = {}
    for m, ts in sorted(tags.items()):
        for hl in ("ban", "noban", "weighted"):
            ds = []
            for tag in sorted(ts):
                if hl == "weighted":
                    db = _seats(views, f"grid_m{m}_{tag}_t{{t}}_ban", "aware", "generic")
                    dn = _seats(views, f"grid_m{m}_{tag}_t{{t}}_noban", "aware", "generic") if m < 5 else db
                    if db is None or dn is None:
                        continue
                    d = {mt_: rw[m] * db[mt_] + (1 - rw[m]) * dn[mt_] for mt_ in db}
                    C_[f"grid_m{m}_{tag} weighted (both seats): aware - generic"] = {
                        mt_: cluster_ci(v) for mt_, v in d.items()}
                else:
                    d = _seats(views, f"grid_m{m}_{tag}_t{{t}}_{hl}", "aware", "generic")
                    if d is None:
                        continue
                ds.append(_lobby(d))
                wpool.setdefault(hl, []).append(_lobby(d))
            if ds:
                C_[f"grid m{m}, all slot sets, both seats, {hl}: aware - generic"] = {
                    mt_: cluster_ci(np.stack([x[mt_] for x in ds], 1)) for mt_ in ds[0]}
                C_[f"grid m{m}, all slot sets, both seats, {hl}: aware - generic"]["configs"] = len(ds)
    for hl, ds in wpool.items():
        C_[f"grid, all party sizes and slot sets, both seats, {hl}: aware - generic"] = {
            mt_: cluster_ci(np.stack([x[mt_] for x in ds], 1)) for mt_ in ds[0]}
    # solo slot gaps (gain at slot a minus slot b, paired by lobby, seats pooled)
    for hl in ("ban", "noban", "weighted"):
        g = {}
        for q in range(5):
            db = _seats(views, f"grid_m1_{q}_t{{t}}_ban", "aware", "generic")
            dn = _seats(views, f"grid_m1_{q}_t{{t}}_noban", "aware", "generic")
            d = db if hl == "ban" else dn if hl == "noban" else (
                None if db is None or dn is None else {mt_: rw[1] * db[mt_] + (1 - rw[1]) * dn[mt_] for mt_ in db})
            if d is not None:
                g[q] = _lobby(d)
        for qa in sorted(g):
            for qb in sorted(g):
                if qb > qa:
                    C_[f"solo slot gap {qa} - {qb} (both seats, {hl}): aware - generic"] = {
                        mt_: boot_ci(g[qa][mt_] - g[qb][mt_]) for mt_ in g[qa]}
    # lock-or-wait (solo) against always-first, always-last and the five fixed turns on average
    lock = {}
    for hl in ("ban", "noban"):
        L_ = f"lock_m1_t{{t}}_{hl}__lock"
        cmp_ = {"always first (aware)": [f"grid_m1_0_t{{t}}_{hl}__aware"],
                "always last (aware)": [f"grid_m1_4_t{{t}}_{hl}__aware"],
                "fixed turn at random (aware)": [f"grid_m1_{q}_t{{t}}_{hl}__aware" for q in range(5)],
                "generic, turn at random": [f"grid_m1_{q}_t{{t}}_{hl}__generic" for q in range(5)]}
        for lab, ns in cmp_.items():
            d = _mean_d([_pair(views, L_, n2) for n2 in ns])
            if d is not None and all(_pair(views, L_, n2) is not None for n2 in ns):
                lock[(hl, lab)] = _lobby(d)
                C_[f"lock-or-wait (both seats, {hl}) - {lab}"] = {mt_: boot_ci(v) for mt_, v in lock[(hl, lab)].items()}
        lt = [lockt.get(f"lock_m1_t{t}_{hl}__lock") for t in (0, 1)]
        if all(x is not None for x in lt):
            res.setdefault("lock_turns", {})[hl] = {
                f"t{t}": {str(q): round(float(np.mean(np.array(TEAM_TURNS[t])[q] == lt[t])), 4) for q in range(5)}
                for t in (0, 1)}
    for lab in ("always first (aware)", "always last (aware)", "fixed turn at random (aware)", "generic, turn at random"):
        if ("ban", lab) in lock and ("noban", lab) in lock:
            a, b = lock[("ban", lab)], lock[("noban", lab)]
            C_[f"lock-or-wait (both seats, weighted) - {lab}"] = {
                mt_: boot_ci(rw[1] * a[mt_] + (1 - rw[1]) * b[mt_]) for mt_ in a}
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    res["lobbies"] = int(LB.n)
    json.dump(res, open(os.path.join(C.RESULTS, "p3t", RES_NAME + ".json"), "w"), indent=1)
    print(f"party: {len(views)} drafts scored; real ban rates {json.dumps(rates)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "run", "score"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--plan", default="effect")
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "select":
        stage_select()
    elif a.stage == "run":
        stage_run(a.plan, a.only, a.device, a.shard)
    else:
        stage_score()


if __name__ == "__main__":
    main()
