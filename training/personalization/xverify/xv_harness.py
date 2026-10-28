"""
Tournament harness checks on the finished tournament outputs (3090, CPU; env
of personalization/remote/p3t_run.sh: P3_CACHE=cache_full, P3_RESULTS=results/rcf_full).
Reads the tournament files, writes only to XV_OUT.

  legality     every finished pair file: 16 distinct heroes; every pick in the
               acting team's pool union whenever that union has a free hero
  seats        every ordered pair file has its seat-swapped partner
  judge        J(swapped teams, swapped tables) = 1 - J; consensus likewise;
               stored scores reproduce (real drafts and drafter drafts)
  selfmatch    GD vs GD and pPol-cur vs pPol-cur on the tournament lobbies:
               P(seat 0 wins) = the first-pick effect only; seat-averaged = 0.5

Usage (from training/): python personalization/xverify/xv_harness.py [stage ...]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.environ.get("XV_TRAINING") or os.path.dirname(os.path.dirname(HERE))
for p in (TRAINING, os.path.join(TRAINING, "personalization")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

OUT = os.path.expanduser(os.environ.get("XV_OUT", "~/hots/xv/out"))
os.makedirs(OUT, exist_ok=True)
RES = {}


def lobbies():
    import p3t_tourney as TT
    if "LB" not in RES:
        RES["LB"] = TT.Lobbies()
    return RES["LB"]


def stage_legality():
    import p3t_systems as SY
    import p3t_tourney as TT
    out = {}
    for d in (TT.TDIR, TT.TDIR + "_tune"):
        if not os.path.isdir(d):
            continue
        tune = d.endswith("_tune")
        os.environ["P3T_LOBBYSET"] = "tune" if tune else ""
        files = sorted(f for f in os.listdir(d) if "__" in f and f.endswith(".npz") and ".tmp" not in f)
        gi_ref = np.load(os.path.join(d, "lobbies.npz"))["gi"]
        L = lobbies() if not tune else None
        bad_dup, bad_pool, bad_gi, n_drafts, n_picks = [], [], [], 0, 0
        for f in files:
            z = np.load(os.path.join(d, f))
            acts = z["acts"]
            if not np.array_equal(z["gi"], gi_ref):
                bad_gi.append(f)
            for i in range(len(acts)):
                n_drafts += 1
                if len(set(acts[i].tolist())) != 16:
                    bad_dup.append((f, i))
            if L is not None:
                for k in range(16):
                    if not SY.IS_PICK[k]:
                        continue
                    m = SY.legal_mask(acts, k, L.team_pool)
                    ok = m[np.arange(len(acts)), acts[:, k]]
                    n_picks += len(acts)
                    if not ok.all():
                        # Real replays the real pick when still free (pool rule not applied): count apart
                        a, b = str(z["a"]), str(z["b"])
                        who = a if SY.DRAFT_TEAM[k] == 0 else b
                        bad_pool.append((f, k, who, int((~ok).sum())))
        nonreal = [x for x in bad_pool if x[2] != "Real"]
        out[os.path.basename(d)] = {"pair_files": len(files), "drafts": n_drafts, "picks_checked": n_picks,
                                    "duplicate_hero_drafts": len(bad_dup), "lobby_set_mismatch": bad_gi,
                                    "pool_violations_nonreal": nonreal[:20], "n_pool_violations_nonreal": len(nonreal),
                                    "pool_violations_real_picks": int(sum(x[3] for x in bad_pool if x[2] == "Real"))}
        print("legality", os.path.basename(d), json.dumps(out[os.path.basename(d)])[:600], flush=True)
    os.environ["P3T_LOBBYSET"] = ""
    return out


def stage_seats():
    import p3t_tourney as TT
    out = {}
    for d in (TT.TDIR, TT.TDIR + "_tune"):
        if not os.path.isdir(d):
            continue
        files = {f[:-4] for f in os.listdir(d) if "__" in f and f.endswith(".npz") and ".tmp" not in f}
        pairs = {tuple(f.split("__")) for f in files}
        missing = sorted("__".join(p) for p in pairs if (p[1], p[0]) not in pairs)
        out[os.path.basename(d)] = {"ordered_pairs": len(pairs), "without_swapped_partner": missing}
    print("seats", json.dumps(out)[:1000], flush=True)
    return out


class _Swapped:
    """LB view with the teams swapped: rows 0-4 <-> 5-9 in every player table."""

    def __init__(self, LB, idx):
        self.map, self.tier = LB.map[idx], LB.tier[idx]
        sw = lambda x: np.concatenate([x[:, 5:], x[:, :5]], 1)
        self.state = {}
        for v, ent in LB.state.items():
            self.state[v] = {k: sw(a[idx]) for k, a in ent.items()}
        self.n = len(idx)


class _Sub:
    def __init__(self, LB, idx):
        self.map, self.tier = LB.map[idx], LB.tier[idx]
        self.state = {v: {k: a[idx] for k, a in ent.items()} for v, ent in LB.state.items()}
        self.n = len(idx)


def swap_acts(acts):
    import p3t_systems as SY
    p0 = [k for k in range(16) if SY.IS_PICK[k] and SY.DRAFT_TEAM[k] == 0]
    p1 = [k for k in range(16) if SY.IS_PICK[k] and SY.DRAFT_TEAM[k] == 1]
    a2 = acts.copy()
    a2[:, p0], a2[:, p1] = acts[:, p1], acts[:, p0]
    return a2


def stage_judge():
    import p3t_judge as PJ
    import p3t_tourney as TT
    LB = lobbies()
    judge = json.load(open(os.path.join(PJ.RES, PJ.JNAME)))
    truth = PJ.Truth()
    rng = np.random.default_rng(3)
    files = sorted(f for f in os.listdir(TT.TDIR) if "__" in f and f.endswith(".npz") and ".tmp" not in f)
    pick = [f for f in files if f.startswith(("aMCTS-cur__pMCTS-cur", "pMCTS-cur__aMCTS-cur", "GD__Real"))]
    pick += list(rng.choice([f for f in files if f not in pick], 3, replace=False))
    out = {"files": pick}
    devs = {"consensus_swap": 0.0, "judge_personal_swap": 0.0, "stored_vs_recomputed": 0.0}
    for f in ["real"] + pick:
        acts = LB.real_acts if f == "real" else np.load(os.path.join(TT.TDIR, f))["acts"]
        idx = np.arange(LB.n)[:60]
        sc = PJ.score_drafts(_Sub(LB, idx), acts[idx], judge, truth)
        sw = PJ.score_drafts(_Swapped(LB, idx), swap_acts(acts[idx]), judge, truth)
        devs["consensus_swap"] = max(devs["consensus_swap"], float(np.abs(sc["consensus"] + sw["consensus"] - 1).max()))
        devs["judge_personal_swap"] = max(devs["judge_personal_swap"],
                                          float(np.abs(sc["judge_personal"] + sw["judge_personal"] - 1).max()))
        sp = os.path.join(PJ.SDIR, ("real" if f == "real" else f[:-4]) + ".npz")
        if os.path.exists(sp):
            st = np.load(sp)
            devs["stored_vs_recomputed"] = max(devs["stored_vs_recomputed"],
                                               float(np.abs(st["judge_personal"][idx] - sc["judge_personal"]).max()))
        print("judge", f, json.dumps(devs), flush=True)
    out.update(devs)
    return out


def stage_selfmatch():
    import p3t_judge as PJ
    import p3t_tourney as TT
    LB = lobbies()
    judge = json.load(open(os.path.join(PJ.RES, PJ.JNAME)))
    truth = PJ.Truth()
    out = {}
    for sysname in ("GD", "pPol-cur"):
        n = LB.n
        acts = np.full((n, 16), -1, np.int64)
        idx = np.arange(n)
        for k in range(16):
            ctx = TT.ctx_for(LB, idx, acts, k, sysname, kseed=12345 + 1000 * k)
            acts[:, k] = TT.act(sysname, ctx, LB, idx, 0)
        sc = PJ.score_drafts(LB, acts, judge, truth)
        j = sc["judge_personal"]
        c = sc["consensus"]
        out[sysname] = {"p_seat0_J": float(j.mean()), "se_J": float(j.std(ddof=1) / np.sqrt(n)),
                        "p_seat0_consensus": float(c.mean()), "seat_averaged_h2h": 0.5}
        np.savez(os.path.join(OUT, f"selfmatch_{sysname}.npz"), acts=acts, J=j, cons=c)
        print("selfmatch", sysname, json.dumps(out[sysname]), flush=True)
    ap = os.path.join(PJ.RES, "analysis.json")
    if os.path.exists(ap):
        a = json.load(open(ap))
        out["analysis_bt_seat"] = {k: v.get("seat_first_pick") for k, v in a.get("bt", {}).items()
                                   if isinstance(v, dict)}
    return out


STAGES = {"legality": stage_legality, "seats": stage_seats, "judge": stage_judge, "selfmatch": stage_selfmatch}

if __name__ == "__main__":
    want = sys.argv[1:] or list(STAGES)
    res = {}
    p = os.path.join(OUT, "harness.json")
    if os.path.exists(p):
        res = json.load(open(p))
    for s in want:
        t0 = time.time()
        res[s] = STAGES[s]()
        json.dump(res, open(p, "w"), indent=1, default=str)
        print(f"== {s} {time.time() - t0:.0f}s", flush=True)
    print("XV HARNESS DONE", flush=True)
