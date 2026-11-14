"""
Paper 2 support arms: pull the selected drafter (x-combo1-dr-chance) back toward team compositions
humans play, and trace support against the record. Max approved 2026-11-13. Arms
(p3t_support_arms.json; every arm is the selected design plus one change, kind mcts_x):
  floor   graded support floor at play (exp 5): our picks keep every 3-hero subset with >= T
          training-corpus teams; if no hero passes, the hero maximizing its rarest triple; the pool
          mask comes first, so the floor relaxes before the team leaves its own pool. In the search
          (cuda_support: root, tree, our rollout steps) and at the root choice (with paper 1's
          structural mask).
  leaf    soft support penalty in the leaf (exp 6): V + lambda g(min3 of our final team), g the
          measured value-function error by support bin (paper 1 ood_frontier certify.json, "net"
          edge per min3 bin, below 10 only): -3.53 pts at 0, -1.87 at 1-4, -0.17 at 5-9, 0 from 10.
  gd_mix  human anchor in the prior (exp 4): softmax(alpha log p_F + bias + mu log p_GD) at our picks.
  base    the player-aware prior rebuilt on paper 1's floor-trained policy (exp 7): vintage
          cur_ffloor (F_floor seeds 0-2), prior pprior_ffloor (p3t_pprior.py train --vintage
          cur_ffloor --out pprior_ffloor, the same visit targets).
Support yardstick: paper 1's training-split corpus (paper1_revision/support_mitigation), the
dense triple counts triple_counts.npy (read only; this module never builds a corpus).

Stages (from training/):
  list                                   arms and their pairs
  verify --pair A__B --ref <tourney dir> [--force-support-kernel]
                                         redraft one pair into <cache>/p3t/tourney_verify(_sk) and
                                         compare its actions with the reference file
  summary [--arms ...]                   per arm vs each opponent: J, J_cal, CONS (win %, both
                                         seats), judge-free paired record, team support of the
                                         arm's teams; -> results/p3t_support/summary.json, .md
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TRAINING)
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

H = 90
SEL = "x-combo1-dr-chance"
OPPONENTS = [SEL, "pMCTS-cur", "aMCTS-pp", "GD", "DraftRec", "Real"]
ARMS_FILE = os.path.join(HERE, "p3t_support_arms.json")
C3_FILE = os.path.join(TRAINING, "paper1_revision", "site", "results", "support_mitigation", "triple_counts.npy")
# value-function error by support bin (paper1_revision/site/results/ood_frontier/certify.json, "net"[0],
# pts): bins [0, 1), [1, 5), [5, 10), [10, inf); the bins from 10 up are within about 0.5 pts of 0
LEAF_EDGES = (1.0, 5.0, 10.0)
LEAF_G_PTS = (-3.5317416327202875, -1.8667968052853736, -0.1718624797026832, 0.0)
DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


def arms():
    """name -> exploration config (selected design plus the arm's change)."""
    if not os.path.exists(ARMS_FILE):
        return {}
    return {k: v for k, v in json.load(open(ARMS_FILE)).items() if not k.startswith("_")}


# the headline pass's combiner (results/p3t_def1027_headline/p3t_vintages.json, cur): every arm must read it
HEADLINE_COMBINER = [0.0, 1.2158510866657697, 4.242860855417209, -0.031338977228257625]


def check_vintages():
    """Refuse to draft unless the vintage table in use carries the headline combiner (the committed
    template p3t_vintages.json has a different combiner fit; set P3T_VINTAGES to the pass copy)."""
    import p3t_systems as SY
    d = json.load(open(SY.VFILE))
    for v in ("cur", "cur_ffloor"):
        if v in d:
            assert np.allclose(d[v]["combiner"], HEADLINE_COMBINER, rtol=0, atol=1e-12), \
                f"{SY.VFILE}: {v} combiner {d[v]['combiner']} is not the headline's {HEADLINE_COMBINER}"
    return SY.VFILE


def support_systems():
    out = {}
    for n, c in arms().items():
        out[n] = dict(kind="mcts_x", vint=c.get("base", "cur"), aware=1, x=c)
    return out


def sup_args(sup):
    """Kernel argument (10 floats): T, lambda, mu, 3 edges of g, 4 g values (probability units)."""
    a = np.zeros(10, np.float32)
    a[0] = float(sup.get("floor", 0.0))
    a[1] = float(sup.get("leaf", 0.0))
    a[2] = float(sup.get("gd_mix", 0.0))
    a[3:6] = LEAF_EDGES
    a[6:10] = np.asarray(LEAF_G_PTS, np.float32) / 100.0
    return a


_C3 = {}


def c3_flat():
    if "flat" not in _C3:
        if not os.path.exists(C3_FILE):
            raise SystemExit(f"missing {C3_FILE} (paper 1's corpus triple counts; copy it, never rebuild here)")
        c = np.load(C3_FILE)
        assert c.shape == (H ** 3,) and c.dtype == np.int32
        _C3["flat"] = np.ascontiguousarray(c)
    return _C3["flat"]


def c3_sym():
    """(H, H, H) count of the triple {a, b, c} at every ordering (0 on repeated heroes)."""
    if "sym" not in _C3:
        c = c3_flat().reshape(H, H, H)
        s = np.zeros_like(c)
        for p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            s = np.maximum(s, c.transpose(p))
        _C3["sym"] = s
    return _C3["sym"]


def rarest_triple(acts_row, k, t):
    """(H,) rarest-triple count of each candidate hero with team t's picks before step k (None if < 2 picks)."""
    ours = [int(acts_row[j]) for j in range(k) if IS_PICK[j] and DRAFT_TEAM[j] == t]
    if len(ours) < 2:
        return None
    S = c3_sym()
    m = np.full(H, np.iinfo(np.int32).max, np.int64)
    for a in range(len(ours)):
        for b in range(a + 1, len(ours)):
            m = np.minimum(m, S[ours[a], ours[b]])
    return m


def graded_floor(acts, k, cand, T):
    """The kernel's graded floor on a (B, H) candidate mask at pick step k (team DRAFT_TEAM[k])."""
    if not IS_PICK[k] or T <= 0:
        return cand
    t = DRAFT_TEAM[k]
    out = cand.copy()
    for i in range(len(acts)):
        m = rarest_triple(acts[i], k, t)
        if m is None or not cand[i].any():
            continue
        ok = cand[i] & (m >= T)
        out[i] = ok if ok.any() else cand[i] & (m == m[cand[i]].max())
    return out


def pairs_for(arm_names=None):
    names = arm_names or list(arms())
    out = []
    for a in names:
        for o in OPPONENTS:
            out += [(a, o), (o, a)]
    return out


# ------------------------------------------------------------------ verify

def verify(pair, ref_dir, device=0, force_support=False):
    """Redraft one pair with the current code into tourney_verify/ and compare with the reference file.
    force_support: the pair's exploration-design sides run through the support kernel with every support
    term off (same names, so the same kernel seeds): the kernel-equivalence check."""
    import p3t_tourney as TT
    import p3t_systems as SY
    check_vintages()
    a, b = pair.split("__")
    if force_support:
        for s_ in (a, b):
            if SY.SYSTEMS[s_]["kind"] == "mcts_x":
                SY.SYSTEMS[s_] = dict(SY.SYSTEMS[s_], x=dict(SY.SYSTEMS[s_]["x"], support={}))
    vdir = TT.TDIR + "_verify" + ("_sk" if force_support else "")   # tourney_verify for the headline set
    os.makedirs(vdir, exist_ok=True)
    lob = np.load(os.path.join(TT.TDIR, "lobbies.npz"))["gi"]
    TT.pair_file = lambda x, y: os.path.join(vdir, f"{x}__{y}.npz")
    import torch
    torch.set_num_threads(2)
    LB = TT.Lobbies()
    TT.play(LB, a, b, device=device)
    new = np.load(os.path.join(vdir, f"{pair}.npz"))
    ref = np.load(os.path.join(ref_dir, f"{pair}.npz"))
    assert (ref["gi"] == lob).all() and (new["gi"] == lob).all()
    same = (new["acts"] == ref["acts"]).all(1)
    pv = json.loads(str(ref["prov"]))
    res = {"pair": pair, "support_kernel": bool(force_support), "lobbies": int(len(same)), "identical_drafts": int(same.sum()),
           "first_diff_lobbies": np.flatnonzero(~same)[:10].tolist(), "ref_gpu": pv.get("gpu"), "ref_arch": pv.get("arch"),
           "new_prov": json.loads(str(new["prov"]))}
    print(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ summary

def _support(C, teams):
    m = C.metrics(teams)
    mn = np.asarray(m["min3"])
    return {"teams": int(len(mn)), "lt25": float((mn < 25).mean()), "lt10": float((mn < 10).mean()),
            "lt5": float((mn < 5).mean()), "min3_median": float(np.median(mn)), "pmi_mean": float(np.mean(m["pmi"])),
            "seen_exact": float((np.asarray(m["n5"]) > 0).mean())}


def _teams(acts, t):
    PK = [k for k in range(16) if IS_PICK[k] and DRAFT_TEAM[k] == t]
    return np.sort(acts[:, PK], 1).astype(np.int64)


def summary(arm_names, out_dir, opponents=None, stem="summary", ref=SEL):
    import p3t_analyze as AN
    from paper1_revision import support_mitigation as sm
    OPPONENTS = opponents or globals()["OPPONENTS"]
    P = AN.load()
    n = len(next(iter(P.values()))["consensus"])
    idxs = AN.boot_idx(n, AN.B)
    C = sm.sup_corpus()
    res = {"lobbies": n, "opponents": OPPONENTS, "vintages": check_vintages(), "arms": {}}
    # the human reference: the real lobbies' own final teams (both teams; not the Real arm, which falls
    # back to GD whenever a real action is no longer legal)
    import p3t_tourney as TT
    lob = np.load(os.path.join(TT.TDIR, "lobbies.npz"))["gi"]
    L = np.load(os.path.join(os.path.dirname(TT.TDIR), "lobbies.npz"))
    first = L["first_team"][lob].astype(np.int64)
    steps = L["steps"][lob]
    hum = []
    for i in range(len(lob)):
        for t in (0, 1):
            hum.append(sorted(int(h) for h, ty, tm, _ in steps[i] if ty == 1 and (tm ^ first[i]) == t))
    res["humans"] = _support(C, np.array(hum, np.int64))
    print("humans", json.dumps(res["humans"]), flush=True)
    for a in [ref] + [x for x in arm_names if x != ref]:
        row = {"config": arms().get(a), "vs": {}}
        T = []
        for o in OPPONENTS:
            if o == a:
                continue
            if (a, o) not in P or (o, a) not in P:
                row["vs"][o] = None
                continue
            T += [_teams(P[(a, o)]["acts"], 0), _teams(P[(o, a)]["acts"], 1)]
            r = {}
            for key, lab in (("judge_personal", "J"), ("judge_cal", "J_cal"), ("consensus", "CONS")):
                pt = AN.pair_win(P, a, o, key)
                bs = np.array([AN.pair_win(P, a, o, key, ix) for ix in idxs])
                se = max(float(np.std(bs)), AN.seed_se(P, a, o, key, n))
                r[lab] = [100 * pt, 100 * (pt - 1.96 * se), 100 * (pt + 1.96 * se)]
            rec = AN.paired_record(P, a, o)
            bs = np.array([AN.paired_record(P, a, o, ix)["diff"] for ix in idxs])
            sd = float(np.nanstd(bs))
            r["record_pp"] = [100 * rec["diff"], 100 * (rec["diff"] - 1.96 * sd), 100 * (rec["diff"] + 1.96 * sd)]
            r["coverage"] = {k: rec[k] for k in ("cov_a", "cov_b", "cov_both")}
            sa = _support(C, np.concatenate([_teams(P[(a, o)]["acts"], 0), _teams(P[(o, a)]["acts"], 1)]))
            r["support_lt25"] = sa["lt25"]
            row["vs"][o] = r
        if T:
            row["support"] = _support(C, np.concatenate(T))
        res["arms"][a] = row
        print(a, json.dumps(row.get("support")), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, f"{stem}.json"), "w"), indent=1)
    lines = ["| arm | <25 | <10 | median min3 | J vs sel | J vs pMCTS-cur | record vs pMCTS-cur | J vs Real | record vs Real | CONS vs pMCTS-cur |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v[0]:.1f} ({v[1]:.1f}, {v[2]:.1f})"
    hs = res["humans"]
    lines.append(f"| humans' own teams | {100 * hs['lt25']:.1f} | {100 * hs['lt10']:.1f} | {hs['min3_median']:.0f} | | | | | | |")
    for a, row in res["arms"].items():
        s = row.get("support") or {}
        g = lambda o, k: (row["vs"].get(o) or {}).get(k)
        lines.append(f"| {a} | {100 * s.get('lt25', np.nan):.1f} | {100 * s.get('lt10', np.nan):.1f} | {s.get('min3_median', np.nan):.0f} | "
                     f"{f(g(SEL, 'J'))} | {f(g('pMCTS-cur', 'J'))} | {f(g('pMCTS-cur', 'record_pp'))} | {f(g('Real', 'J'))} | "
                     f"{f(g('Real', 'record_pp'))} | {f(g('pMCTS-cur', 'CONS'))} |")
    open(os.path.join(out_dir, f"{stem}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------------ promotion of s-floor25 (coordinator 2026-11-14)

PROMOTE = "s-floor25"
REP_OFFSETS = (1, 2, 3)
CURVE = ["s-floor10", "s-floor15", "s-floor20", "s-floor25"]   # graded-floor tradeoff curve (coordinator 2026-11-14)


def promote_pairs(kind):
    """support_confirm: the tune-set selection matchups; support_matrix: the headline matrix row (MAIN);
    support_rep: the repeated-run set REP with PROMOTE in place of the selected design (pairs that involve it),
    plus PROMOTE vs the selected design; support_loose: the loose-lobby matchups."""
    import p3t_tourney as TT
    F = PROMOTE
    if kind == "support_confirm":
        opps, extra = TT.TUNE_OPP, [(SEL, o) for o in TT.TUNE_OPP] + [(o, SEL) for o in TT.TUNE_OPP]
    elif kind == "support_matrix":
        opps, extra = [o for o in TT.MAIN if o != F], []
    elif kind == "support_rep":
        opps, extra = [o for o in TT.REP if o not in (SEL, F)] + [SEL], []
    elif kind == "support_tune_curve":
        out = []
        for x in CURVE:
            for o in TT.TUNE_OPP:
                out += [(x, o), (o, x)]
        return out
    elif kind == "support_loose":
        opps, extra = ["pMCTS-cur", "aMCTS-pp"], [(SEL, "aMCTS-pp"), ("aMCTS-pp", SEL)]
    else:
        raise ValueError(kind)
    out = []
    for o in opps:
        out += [(F, o), (o, F)]
    return out + extra


def tune_score(out_dir, arms_=None, stem="tune"):
    """The selection rule on the tune lobbies (run with P3T_LOBBYSET=tune): mean J, CONS and J_corr win
    probability against the tune opponents (both seats), as p3t_analyze's tune table; plus team support."""
    import p3t_analyze as AN
    import p3t_tourney as TT
    from paper1_revision import support_mitigation as sm
    assert os.environ.get("P3T_LOBBYSET") == "tune"
    P = AN.load()
    n = len(next(iter(P.values()))["consensus"])
    idxs = AN.boot_idx(n, AN.B)
    C = sm.sup_corpus()
    res = {"lobbies": n, "opponents": TT.TUNE_OPP, "vintages": check_vintages(), "rows": {}}
    arms_ = list(arms_ or [PROMOTE])
    for x in [SEL] + arms_:
        row = {}
        for key in ("judge_personal", "consensus", "judge_corr"):
            vals = [AN.pair_win(P, x, o, key) for o in TT.TUNE_OPP]
            row[key] = round(100 * float(np.mean(vals)), 2)
            row[key + "_by_opp"] = {o: round(100 * float(v), 2) for o, v in zip(TT.TUNE_OPP, vals)}
            bs = [np.mean([AN.pair_win(P, x, o, key, ix) for o in TT.TUNE_OPP]) for ix in idxs]
            row[key + "_boot_se"] = round(100 * float(np.std(bs)), 3)
        # paired difference of the two designs' tune scores (same lobbies), lobby bootstrap
        T = []
        for o in TT.TUNE_OPP:
            T += [_teams(P[(x, o)]["acts"], 0), _teams(P[(o, x)]["acts"], 1)]
        row["support"] = _support(C, np.concatenate(T))
        res["rows"][x] = row
    for x in arms_:
        d = [np.mean([AN.pair_win(P, x, o, "judge_personal", ix) - AN.pair_win(P, SEL, o, "judge_personal", ix)
                      for o in TT.TUNE_OPP]) for ix in idxs]
        pt = np.mean([AN.pair_win(P, x, o, "judge_personal") - AN.pair_win(P, SEL, o, "judge_personal") for o in TT.TUNE_OPP])
        se = float(np.std(d))
        res["rows"][x]["diff_J_minus_sel"] = [round(100 * pt, 2), round(100 * se, 3),
                                              round(100 * (pt - 1.96 * se), 2), round(100 * (pt + 1.96 * se), 2)]
    if arms_ == [PROMOTE]:
        res["diff_J_promote_minus_sel"] = res["rows"][PROMOTE]["diff_J_minus_sel"][:2]
    lob = np.load(os.path.join(TT.TDIR, "lobbies.npz"))["gi"]
    L = np.load(os.path.join(os.path.dirname(TT.TDIR), "lobbies.npz"))
    first = L["first_team"][lob].astype(np.int64)
    steps = L["steps"][lob]
    hum = [sorted(int(h) for h, ty, tm, _ in steps[i] if ty == 1 and (tm ^ first[i]) == t) for i in range(len(lob)) for t in (0, 1)]
    res["humans"] = _support(C, np.array(hum, np.int64))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, f"{stem}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def rep_table(out_dir):
    """Repeated runs: J and CONS of every PROMOTE matchup at kernel-seed offset 0 (the main run) and the
    REP_OFFSETS (tourney_rep<n>), and their SD, as the headline's repeated-run table."""
    import p3t_analyze as AN
    base = os.path.dirname(AN.TDIR)
    Ps = []
    for tag in [""] + [f"_rep{n}" for n in REP_OFFSETS]:
        AN.TDIR = os.path.join(base, "tourney" + tag)
        AN.SDIR = os.path.join(base, "scores" + tag)
        Ps.append(AN.load())
    rows = {}
    for a, b in promote_pairs("support_rep"):
        if a != PROMOTE:
            continue
        r = {}
        for key, lab in (("judge_personal", "J"), ("consensus", "CONS"), ("judge_cal", "J_cal")):
            v = [100 * AN.pair_win(P, a, b, key) for P in Ps]
            r[lab] = [round(x, 2) for x in v]
            r[lab + "_sd"] = round(float(np.std(v, ddof=1)), 3)
        rec = [AN.paired_record(P, a, b) for P in Ps]
        r["record_pp"] = [round(100 * x["diff"], 2) for x in rec]
        r["record_sd"] = round(float(np.std([100 * x["diff"] for x in rec], ddof=1)), 3)
        rows[f"{a} vs {b}"] = r
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"offsets": [0] + list(REP_OFFSETS), "rows": rows}, open(os.path.join(out_dir, "rep.json"), "w"), indent=1)
    L = ["| matchup | J, offsets 0-3 | J SD | CONS SD | record, offsets 0-3 | record SD |", "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {', '.join(f'{x:.2f}' for x in r['J'])} | {r['J_sd']:.2f} | {r['CONS_sd']:.2f} | "
                 f"{', '.join(f'{x:+.1f}' for x in r['record_pp'])} | {r['record_sd']:.2f} |")
    open(os.path.join(out_dir, "rep.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["list", "verify", "summary", "tune", "tune_curve", "matrix", "rep", "loose"])
    ap.add_argument("--pair", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--force-support-kernel", action="store_true")
    a = ap.parse_args()
    names = [x for x in a.arms.split(",") if x] or list(arms())
    if a.stage == "list":
        for p in pairs_for(names):
            print("__".join(p))
    elif a.stage == "verify":
        verify(a.pair, a.ref, a.device, a.force_support_kernel)
    else:
        import p3_hs_core as C
        out = a.out or os.path.join(C.RESULTS, "p3t_support")
        if a.stage == "summary":
            summary(names, out)
        elif a.stage == "tune":
            tune_score(out)
        elif a.stage == "tune_curve":
            assert os.environ.get("P3T_LOBBYSET") == "tune"
            tune_score(out, CURVE, stem="tune_curve")
        elif a.stage == "matrix":
            import p3t_tourney as TT
            summary([PROMOTE], out, opponents=[o for o in TT.MAIN if o != PROMOTE], stem="matrix")
        elif a.stage == "loose":
            assert os.environ.get("P3T_LOBBYSET") == "loose"
            summary([PROMOTE], out, opponents=["pMCTS-cur", "aMCTS-pp", "GD", "DraftRec", "Real"], stem="loose")
        else:
            rep_table(out)


if __name__ == "__main__":
    main()
