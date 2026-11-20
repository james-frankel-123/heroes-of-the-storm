"""
Paper 2, player-axis support: how familiar is each placed player with the hero the arm puts him on?

For every player an arm places on a hero in the 300 headline tournament lobbies (only the arm's own
team: kernel team 0 in <arm>__<opp>.npz, team 1 in <opp>__<arm>.npz), "prior games" = that player's
number of earlier Storm League games on that hero: the merged play count n of the drafters' cur state
(state_cur.npz "n": slot rows plus Heroes Profile Storm League rows with day <= lobby day - 1, i.e. the
same count the personal model's experience offset, pools and new-hero layer read; p3t_data.states).
Strictly earlier days only, so no game on the lobby's own day enters.

Drafters choose heroes, not players, so a placement needs an assignment of the team's five players to
its five heroes. Three rules, each applied to every arm alike:
  record   the judge-free record's outcome-free rule (p3t_judge.score_drafts): maximize the sum of
           log1p(prior games) over the 120 permutations (the paper's assignment for the paired record)
  model    the drafter's own leaf assignment: maximize b2 S + b3 O on the cur state (headline combiner;
           a pscale multiplies b2 and b3 alike, so it does not change the argmax)
  actual   humans only: the player who really played the hero
Humans = the real final teams of the same lobbies (both teams). Coverage ("covered") = the placed player
has at least one judge-window game (2026-07-01..09-11) on the hero, the per-player condition for
entering the judge-free paired record.

Opponent sets: "panel" = both seats against each of GD, Real and DraftRec (the same opponents for every
arm, none of them a target arm); "all" = every opponent the arm met in the headline files.
Bootstrap: 2000 lobby resamples (percentile 95% CIs). Aggregates only; no player keys leave the caches.

Usage (from anywhere; CPU only):
  nice -n 19 taskset -c 44-47 /usr/bin/python3 personalization/p3t_familiarity.py
-> results/p3t_familiarity/{familiarity.json, FAMILIARITY.md}
"""
import itertools
import json
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HL = os.path.join(HERE, "cache_def_def1027_hlsup", "p3t")
TDIRS = [os.path.join(HL, "tourney_headline"), os.path.join(HL, "tourney")]   # headline matrix, then support arms
VFILE = os.path.join(HERE, "results", "p3t_def1027_headline", "p3t_vintages.json")
OUT = os.path.join(HERE, "results", "p3t_familiarity")
H = 90
B = 2000
CAP = 2000                       # histogram cap for the bootstrap median (medians are far below)
SP = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
ST = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
PK = {t: [k for k in range(16) if SP[k] and ST[k] == t] for t in (0, 1)}
PERMS = np.array(list(itertools.permutations(range(5))), np.int64)
ARMS = [("s-pcal-floor25", "final drafter (promoted 2026-11-14)"),
        ("s-floor25", "support floor 25, unscaled personal term (stand-in for s-pcal-floor25)"),
        ("x-combo1-dr-chance", "selected unconstrained design"),
        ("aMCTS-pp", "player-aware MCTS, player prior"),
        ("aMCTS-cur", "player-aware MCTS"),
        ("a1step-cur", "player-aware one-step"),
        ("pMCTS-cur", "population MCTS (paper 1)"),
        ("p1step-cur", "population one-step")]
PANEL = ["GD", "Real", "DraftRec"]
BINS = [(0, 0, "0"), (1, 4, "1-4"), (5, 19, "5-19"), (20, None, "20+")]
THRESH = [1, 5, 10]              # "below t" shares; 5 = the model's new-hero threshold (Layer: n_all < 5)


def best_perm(val):
    """val (5 players, 5 heroes) -> perm with player p on hero perm[p] maximizing the sum (first max, as p3t_judge)."""
    i = np.arange(5)[None, :]
    return PERMS[int(np.argmax(val[i, PERMS].sum(1)))]


def load():
    L = np.load(os.path.join(HL, "lobbies.npz"))
    gi = np.load(os.path.join(TDIRS[0], "lobbies.npz"))["gi"]
    for d in TDIRS[1:]:
        assert np.array_equal(np.load(os.path.join(d, "lobbies.npz"))["gi"], gi)
    ver = L["version"][gi].astype(str)
    assert not any(v.startswith("2.57") for v in ver), "sealed patch in the lobby set"
    first = L["first_team"][gi].astype(np.int64)
    steps = L["steps"][gi]
    n = len(gi)
    src = np.zeros((n, 10), np.int64)
    hum = np.zeros((n, 10), np.int64)          # hero really played by each kernel slot
    for i in range(n):
        cnt = [0, 0]
        j = 0
        for k in range(16):
            h, ty, tm, _ = steps[i][k]
            if ty == 1:
                kt = int(tm) ^ int(first[i])
                assert kt == ST[k]
                sl = 5 * kt + cnt[kt]
                cnt[kt] += 1
                src[i, sl] = j
                hum[i, sl] = int(h)
                j += 1
    rows = (gi * 10)[:, None] + src
    zc = np.load(os.path.join(HL, "state_cur.npz"))
    assert np.array_equal(zc["gsel"], np.arange(len(L["replay_id"])))
    st = {k: zc[k][rows.ravel()].astype(np.float64).reshape(n, 10, H) for k in ("n", "s", "off")}
    # leak check: the state's as-of day is the lobby's own day (counts use days <= qday - 1)
    assert np.array_equal(zc["qday"][rows.ravel()].reshape(n, 10), np.repeat(L["day"][gi][:, None], 10, 1))
    zj = np.load(os.path.join(HL, "state_judge.npz"))
    win = zj["gsel"]
    assert L["day"][gi].max() <= L["day"][win].max()
    wpos = {int(g): p for p, g in enumerate(win)}
    jrows = np.array([wpos[int(g)] * 10 for g in gi])[:, None] + src
    st["rec_n"] = zj["rec_n"][jrows.ravel()].astype(np.float64).reshape(n, 10, H)
    return gi, hum, st


def pair_path(a, b):
    for d in TDIRS:
        p = os.path.join(d, f"{a}__{b}.npz")
        if os.path.exists(p):
            return p
    return None


def check_copies():
    """pair files present in both directories must hold the same drafts."""
    bad, n = [], 0
    for f in os.listdir(TDIRS[1]):
        p0 = os.path.join(TDIRS[0], f)
        if "__" in f and os.path.exists(p0):
            n += 1
            if not np.array_equal(np.load(p0)["acts"], np.load(os.path.join(TDIRS[1], f))["acts"]):
                bad.append(f)
    return n, bad


def place(st, i, t, heroes, rule, comb):
    """(player slots, heroes) of team t's five heroes under an assignment rule."""
    sl = slice(5 * t, 5 * t + 5)
    hs = np.asarray(heroes, np.int64)
    if rule == "record":
        val = np.log1p(st["n"][i, sl][:, hs])
    else:
        val = comb[2] * st["s"][i, sl][:, hs] + comb[3] * st["off"][i, sl][:, hs]
    perm = best_perm(val)
    return np.arange(5 * t, 5 * t + 5), hs[perm]


def collect(st, units):
    """units: list of (lobby, slots(5,), heroes(5,)) -> per-placement arrays."""
    lob, cnt, cov = [], [], []
    for i, slots, hs in units:
        lob.append(np.full(5, i))
        cnt.append(st["n"][i, slots, hs])
        cov.append(st["rec_n"][i, slots, hs] > 0)
    return np.concatenate(lob), np.concatenate(cnt), np.concatenate(cov)


def stats(lob, cnt, n_lob, W):
    """point and bootstrap CIs of the bin shares, the below-t shares and the median; W (B, n_lob) weights."""
    idx = np.minimum(cnt, CAP).astype(np.int64)
    Hl = np.zeros((n_lob, CAP + 1))
    np.add.at(Hl, (lob, idx), 1.0)
    def frame(w):
        h = w @ Hl                                   # (R, CAP+1)
        tot = h.sum(1)
        c = np.cumsum(h, 1)
        out = {}
        for lo, hi, lab in BINS:
            top = c[:, hi] if hi is not None else tot
            bot = c[:, lo - 1] if lo > 0 else 0.0
            out[lab] = 100 * (top - bot) / tot
        for t in THRESH:
            out[f"below{t}"] = 100 * c[:, t - 1] / tot
        out["median"] = np.argmax(c >= 0.5 * tot[:, None], 1).astype(float)
        return out, tot
    pt, tot = frame(np.ones((1, n_lob)))
    bs, _ = frame(W)
    res = {"placements": int(tot[0])}
    for k in pt:
        lo_, hi_ = np.percentile(bs[k], [2.5, 97.5])
        res[k] = [float(pt[k][0]), float(lo_), float(hi_)]
    exact_med = float(np.median(cnt))
    assert exact_med >= CAP or abs(exact_med - res["median"][0]) <= 0.5, (exact_med, res["median"])
    res["median"][0] = exact_med
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    comb = json.load(open(VFILE))["cur"]["combiner"]
    gi, hum, st = load()
    n = len(gi)
    rng = np.random.default_rng(0)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
    ncopy, bad = check_copies()
    assert not bad, f"pair files differ between directories: {bad}"
    # the record rule must reproduce the scored judge-free record's coverage (rec_slot finite)
    sc = np.load(os.path.join(HL, "scores_headline", "x-combo1-dr-chance__GD.npz"))["rec_slot"]
    acts = np.load(pair_path("x-combo1-dr-chance", "GD"))["acts"]
    tie_mismatch = 0
    for t in (0, 1):
        for i in range(n):
            slots, hs = place(st, i, t, acts[i, PK[t]], "record", comb)
            if not np.array_equal(st["rec_n"][i, slots, hs] > 0, np.isfinite(sc[i, slots])):
                # allowed only where two permutations tie exactly (float tie-break order differs)
                v = np.log1p(st["n"][i, slots][:, np.sort(hs)])
                sm = np.sort(v[np.arange(5)[None, :], PERMS].sum(1))
                assert sm[-1] - sm[-2] < 1e-6, (i, t)
                tie_mismatch += 1
    print(f"record-rule coverage check: {tie_mismatch} of {2 * n} teams differ, all on exact ties", flush=True)
    res = {"lobbies": n, "bootstrap": B, "combiner": comb, "opponent_panel": PANEL, "pair_copies_checked": ncopy,
           "record_rule_tie_mismatch_teams": f"{tie_mismatch} of {2 * n} (x-combo1-dr-chance vs GD; exact ties)",
           "definition": "prior games = merged Storm League play count on the hero with day <= lobby day - 1 "
                         "(state_cur.npz n, p3t_data.states / _merged_counts)", "arms": {}, "missing": {}}
    # humans: actual placement, and the same two assignment rules on their real teams
    for rule in ("actual", "record", "model"):
        units = []
        for i in range(n):
            for t in (0, 1):
                hs = hum[i, 5 * t:5 * t + 5]
                if rule == "actual":
                    units.append((i, np.arange(5 * t, 5 * t + 5), hs))
                else:
                    units.append((i, *place(st, i, t, hs, rule, comb)))
        lob, cnt, cov = collect(st, units)
        res["arms"].setdefault("humans", {})[rule] = {
            "all": stats(lob, cnt, n, W), "covered": stats(lob[cov], cnt[cov], n, W),
            "coverage": float(cov.mean()), "teams": len(units)}
        print("humans", rule, json.dumps(res["arms"]["humans"][rule]["all"]), flush=True)
    for arm, _ in ARMS:
        opps_all = sorted({f[:-4].split("__")[1] for d in TDIRS for f in os.listdir(d)
                           if f.startswith(arm + "__") and f.endswith(".npz")})
        opps_all = [o for o in opps_all if pair_path(o, arm)]
        if not opps_all:
            res["missing"][arm] = "no pair files on this machine (drafted on the 3090; not copied back)"
            print(arm, "MISSING", flush=True)
            continue
        out = {"opponents_all": opps_all}
        for oset_name, oset in (("panel", PANEL), ("all", opps_all)):
            if any(pair_path(arm, o) is None or pair_path(o, arm) is None for o in oset):
                out[oset_name] = None
                continue
            for rule in ("record", "model"):
                units = []
                for o in oset:
                    for (a_, b_), t in (((arm, o), 0), ((o, arm), 1)):
                        acts = np.load(pair_path(a_, b_))["acts"]
                        assert acts.shape == (n, 16)
                        for i in range(n):
                            units.append((i, *place(st, i, t, acts[i, PK[t]], rule, comb)))
                lob, cnt, cov = collect(st, units)
                out.setdefault(oset_name, {})[rule] = {
                    "all": stats(lob, cnt, n, W), "covered": stats(lob[cov], cnt[cov], n, W),
                    "coverage": float(cov.mean()), "teams": len(units)}
            print(arm, oset_name, json.dumps(out[oset_name]["record"]["all"]), flush=True)
        res["arms"][arm] = out
    json.dump(res, open(os.path.join(OUT, "familiarity.json"), "w"), indent=1)
    report(res)


def report(res):
    f = lambda v: f"{v[0]:.1f} ({v[1]:.1f}, {v[2]:.1f})"
    fm = lambda v: f"{v[0]:.0f} ({v[1]:.0f}, {v[2]:.0f})"
    hdr = ["| arm | placed | 0 | 1-4 | 5-19 | 20+ | median | below 5 | below 10 | coverage |",
           "|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, r, part="all"):
        s = r[part]
        return (f"| {name} | {s['placements']:,} | {f(s['0'])} | {f(s['1-4'])} | {f(s['5-19'])} | {f(s['20+'])} | "
                f"{fm(s['median'])} | {f(s['below5'])} | {f(s['below10'])} | {100 * r['coverage']:.0f}% |")

    names = dict(ARMS)
    L = ["# Player-axis support: how familiar are placed players with their heroes?", "",
         f"Script: `p3t_familiarity.py`. Numbers: `familiarity.json`. {res['lobbies']} headline tournament lobbies, "
         f"{res['bootstrap']} lobby-bootstrap resamples (percentile 95% CIs in parentheses).", "",
         "**Prior games** = the placed player's earlier Storm League games on the hero: the merged play count of the "
         "drafters' cur state (slot rows plus Heroes Profile Storm League rows, days up to the day before the lobby; "
         "`state_cur.npz` n, built by `p3t_data.py states`). It is the count behind the personal model's experience "
         "offset, its new-hero layer (fewer than 5 earlier games) and the drafters' pools (at least 1).", "",
         "Only the arm's own team counts. Each arm is scored against the same opponent panel (GD, Real, DraftRec, both "
         "seats; 6 drafts × 5 players per lobby). Humans are the real final teams of the same lobbies (both teams).", "",
         "Drafters pick heroes, not players. **Record assignment** (main table) puts the five players on the five heroes "
         "to maximize the sum of log(1 + prior games), the judge-free record's outcome-free rule. **Model assignment** "
         "uses the drafter's own leaf (b2 S + b3 O on the cur state). Humans also get their **actual** placement.", "",
         "Coverage = share of placed players with at least one judge-window game on the hero, the condition for "
         "entering the judge-free paired record.", ""]
    if res["missing"]:
        L += ["**Not computed:** " + "; ".join(f"`{a}`: {m}" for a, m in res["missing"].items()) +
              ". `s-floor25` (same graded floor 25, personal term not rescaled) is reported as the closest stand-in.", ""]
    for rule, title in (("record", "Record assignment (main)"), ("model", "Model assignment")):
        L += [f"## {title}, panel opponents", ""] + hdr
        h = res["arms"]["humans"]
        L.append(row("humans, actual placement", h["actual"]))
        L.append(row(f"humans, {rule} assignment", h[rule]))
        for a, _ in ARMS:
            r = res["arms"].get(a)
            if r and r.get("panel"):
                L.append(row(f"{a}", r["panel"][rule]))
        L.append("")
    L += ["## Restricted to placements the paired record covers (record assignment, panel)", ""] + hdr
    h = res["arms"]["humans"]
    L.append(row("humans, actual placement", h["actual"], "covered"))
    L.append(row("humans, record assignment", h["record"], "covered"))
    for a, _ in ARMS:
        r = res["arms"].get(a)
        if r and r.get("panel"):
            L.append(row(a, r["panel"]["record"], "covered"))
    L += ["", "## Every headline opponent the arm met (record assignment)", ""] + hdr
    for a, _ in ARMS:
        r = res["arms"].get(a)
        if r and r.get("all"):
            L.append(row(f"{a} ({len(r['opponents_all'])} opponents)", r["all"]["record"]))
    L += ["", "Arms: " + "; ".join(f"`{a}` {d}" for a, d in ARMS) + "."]
    open(os.path.join(OUT, "FAMILIARITY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
