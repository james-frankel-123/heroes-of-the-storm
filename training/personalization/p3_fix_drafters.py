"""
Audit fixes P3-02 and P3-03 for every drafter: rerun one-step, MCTS,
distilled-prior and personalized-GD drafters under one protocol.

Protocol (the P3_PERSONAL_GD harness, now everywhere):
  - pools = heroes the player played on earlier days (the real hero is no
    longer added; an empty pool falls back to every free hero)
  - no real bans are forced: in the one-step drafter every ban is sampled
    from the GD policy (both teams; bans are not optimized there), in MCTS
    our bans are searched and the other team's come from the GD
  - personal term s = lag-1 experience offset (table refit on lag-1 counts,
    p3_fix_counts) + the GP posterior mean (unchanged state)
  - combiner = "skill + off-role count (fine)" refit with lag-1 counts
    (results/fix/p3_ph_role.json)
The pick order of players (who picks at which step) is kept: it is fixed by
the lobby before the draft, so it is not oracle information.

All outputs go to cache/fix/ and results/fix/; nothing pre-audit is
overwritten. The existing modules are imported and patched, not edited.

Run (from training/), e.g.:
  python3 personalization/p3_fix_drafters.py tables
  python3 personalization/p3_fix_drafters.py onestep --procs 4
  CUDA_VISIBLE_DEVICES=3 python3 personalization/p3_fix_drafters.py mcts --stage full --sims 400
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C

FIX_CACHE = os.path.join(C.CACHE, "fix")
FIX_RESULTS = os.path.join(C.RESULTS, "fix")
FIX_PTAB = os.path.join(FIX_CACHE, "dr_personal_post.npz")


def fixed_combiner():
    with open(os.path.join(FIX_RESULTS, "p3_ph_role.json")) as f:
        c = json.load(f)["D_game"]["skill + off-role count (fine)"]["coef"]
    return np.array(c, float)


def build_tables():
    """Fixed personal tables from the original ones: pools from earlier-day
    counts only, s with the lag-1 offset table."""
    import p3_hs_fit
    import p3_fix_counts as F
    import p3_dr_core as D
    d = C.load_slots()
    _, _, _, table_old, _ = p3_hs_fit.prepare(d)
    _, _, _, table_new, _ = F.prepare_l1(d)
    z = np.load(D.PTAB)
    T = {k: z[k] for k in z.files}
    n, n_p = T["n"], T["n_p"]
    nb = len(C.NPH_EDGES)
    bp = np.searchsorted(C.NP_EDGES, n_p, side="right") - 1
    bh = np.searchsorted(C.NPH_EDGES, n, side="right") - 1
    idx = bp[:, None] * nb + bh
    T["s"] = (T["s"] - table_old[idx] + table_new[idx]).astype(np.float32)
    old_pool = T["pool"]
    T["pool"] = n > 0
    os.makedirs(FIX_CACHE, exist_ok=True)
    np.savez(FIX_PTAB, **T)
    added = (old_pool & ~T["pool"]).sum(1) > 0
    print(f"fixed tables: {len(n):,} slots; slots whose old pool had the game's own hero added: {added.mean():.3f}; "
          f"empty pools now: {(~T['pool'].any(1)).mean():.4f}")


FIX_IMIT = os.path.join(FIX_CACHE, "dr_imitation.npz")


def patch_imitation():
    """P3-24: imitation recency features on the lag-1 contract, weights refit
    on V1 picks with them (built on first use)."""
    import p3_fix_counts as F
    import p3_dr_imitation as I
    I.recency_features = F.recency_features_l1
    I.OUT = FIX_IMIT
    if not os.path.exists(FIX_IMIT):
        I.main()


OUTPUT_PREFIXES = ("dr_runs", "mcts_", "dsmcts_", "dsgreedy_", "ds_targets_", "ds_prior", "pgdmcts_",
                   "dr_imitation", "dr_personal_post", "x_draft_sim", "fix_")


def link_inputs():
    """cache/fix sees every original cache file the fixed runs only read
    (symlinks); files the fixed runs write are never linked, so originals
    cannot be overwritten through a link."""
    src = C.CACHE if C.CACHE != FIX_CACHE else os.path.dirname(FIX_CACHE)
    for f in os.listdir(src):
        if f == "fix" or f.startswith(OUTPUT_PREFIXES):
            continue
        dst = os.path.join(FIX_CACHE, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src, f), dst)


def patch_common():
    os.makedirs(FIX_CACHE, exist_ok=True)
    os.makedirs(FIX_RESULTS, exist_ok=True)
    link_inputs()
    C.CACHE = FIX_CACHE
    C.RESULTS = FIX_RESULTS
    import p3_dr_core as D
    D.PTAB = FIX_PTAB
    D.combiner = fixed_combiner
    patch_imitation()
    try:
        import p3_mcts_drafter as MD
        orig = MD.Setup.lobby

        def lobby(self, gi, sub=None):
            lb = orig(self, gi, sub)
            lb["forced"] = np.full(16, -1, np.int32)
            return lb
        MD.Setup.lobby = lobby
    except ImportError:
        pass
    # files the fixed runs read but do not regenerate
    for f in ("wp_drift.npz",):
        pass


def patch_onestep():
    import p3_dr_drafter as DR
    DR.OUT = os.path.join(FIX_CACHE, "dr_runs.pkl.gz")

    def ban_sample(states, k, lob, rng):
        gd = DR._W["gd"]
        B = len(states)
        t0 = np.zeros((B, 90), np.float32)
        t1 = np.zeros((B, 90), np.float32)
        bans = np.zeros((B, 90), np.float32)
        avail = np.zeros((B, 90), bool)
        for i, s in enumerate(states):
            t0[i, list(s["t0"].values())] = 1
            t1[i, list(s["t1"].values())] = 1
            bans[i, list(s["bans"])] = 1
            avail[i] = ~s["taken"]
        lp = gd.logprobs(t0, t1, bans, np.repeat(lob["mo"][None], B, 0), np.repeat(lob["to"][None], B, 0),
                         np.full(B, k, np.float32), np.zeros(B, np.float32), avail)
        p = np.where(avail, np.exp(lp), 0)
        p /= p.sum(1, keepdims=True)
        return np.minimum((p.cumsum(1) < rng.rand(B, 1)).sum(1), 89)

    def apply_ban(s, hero):
        s = {"t0": dict(s["t0"]), "t1": dict(s["t1"]), "bans": set(s["bans"]), "taken": s["taken"].copy()}
        s["bans"].add(int(hero))
        s["taken"][hero] = True
        return s

    def _rollout(states, k0, lob, rng, opp_kind, our_team):
        for k in range(k0, 16):
            h, ty, tm, row = lob["steps"][k]
            if ty == 0:
                bs = ban_sample(states, k, lob, rng)
                states = [apply_ban(s, b) for s, b in zip(states, bs)]
            else:
                kind = opp_kind if (tm != our_team and opp_kind == "imit") else "gd"
                picks = DR._policy_sample(states, k, lob, rng, kind)
                states = [DR._apply(s, k, int(p), lob) for s, p in zip(states, picks)]
        return states

    def _run_draft(lob, seed, modes, opp_kind):
        rng = np.random.RandomState(seed)
        state = {"t0": {}, "t1": {}, "bans": set(), "taken": np.zeros(90, bool)}
        decs = []
        for k in range(16):
            h, ty, tm, row = lob["steps"][k]
            if ty == 0:
                state = apply_ban(state, int(ban_sample([state], k, lob, rng)[0]))
                continue
            m = modes[tm]
            if m in ("pers", "pop"):
                dd = DR._decide(state, k, lob, rng, opp_kind)
                decs.append(dd)
                hero = dd[m]
            else:
                hero = int(DR._policy_sample([state], k, lob, rng, m)[0])
            state = DR._apply(state, k, hero, lob)
        wp, V = DR._value([(state["t0"], state["t1"])], lob, DR._W["b"])
        return {"t0": state["t0"], "t1": state["t1"], "wp": float(wp[0]), "V": float(V[0]), "decisions": decs}

    DR._rollout = _rollout
    DR._run_draft = _run_draft
    DR.fix_ban_sample = ban_sample
    DR.fix_apply_ban = apply_ban
    # Real-state rankings still add the real hero to the candidate list so it
    # can be ranked (evaluation only; pools no longer contain it).
    return DR


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tables", "onestep", "mcts", "ds_targets", "ds_train", "ds_mcts", "ds_greedy",
                                     "pgd", "analyze_mcts", "analyze_ds", "analyze_pgd", "analyze_onestep"])
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.what == "tables":
        build_tables()
        return
    patch_common()
    sys.argv = [a.what] + a.rest
    if a.what == "onestep":
        DR = patch_onestep()
        DR.main()
    elif a.what == "mcts":
        import p3_mcts_drafter as MD
        MD.main()
    elif a.what == "ds_targets":
        import p3_ds_targets as T
        T.main()
    elif a.what == "ds_train":
        import p3_ds_common as DS
        DS.MODEL = os.path.join(FIX_CACHE, "ds_prior.pt")
        import p3_ds_train as TR
        TR.main()
    elif a.what in ("ds_mcts", "ds_greedy", "analyze_ds"):
        import p3_ds_common as DS
        DS.MODEL = os.path.join(FIX_CACHE, "ds_prior.pt")
        mod = {"ds_mcts": "p3_ds_mcts", "ds_greedy": "p3_ds_greedy", "analyze_ds": "p3_ds_analyze"}[a.what]
        __import__(mod).main()
    elif a.what == "pgd":
        import p3_pgd_mcts as PG
        PG.main()
    elif a.what == "analyze_mcts":
        import p3_mcts_analyze as MA
        MA.main()
    elif a.what == "analyze_pgd":
        import p3_pgd_analyze as PA
        PA.main()
    elif a.what == "analyze_onestep":
        import p3_dr_analyze as DA
        DA.RUNS = os.path.join(FIX_CACHE, "dr_runs.pkl.gz")
        DA.main()


if __name__ == "__main__":
    main()
