"""
P3 tournament: paper-1 style validation of the player-aware drafters
(coordinator, 2026-10-21; additions 3 to 5 of the checklist in P3_TOURNAMENT.md).

  wpseeds   rescore every tournament draft with the three seeds of paper 1's
            enriched WP (42 = the drafters' leaf, 123, 777); per system: the
            mean WP of its drafts against its opponents under each seed, the
            seed spread and the lower bound (min over seeds). A drafter that
            wins by exploiting one model's blind spots shows a large spread.
  behavior  per system, over its own steps in every pair it played: distinct
            heroes picked, pick entropy, top-10 share; GD similarity (its
            action = GD argmax at that state) and GD log-likelihood (mean log
            p_GD of its action), picks and bans; paper 1's resilience (for
            each of its picks, minus the mean counter delta of the opponent's
            later picks against it, judge-window statistics: judge-free).
  probes    hand-built personal probes on the 300 real tournament drafts:
            one hero of team 0 swapped for a same-role hero (closest
            population win rate) that (a) none of the five players has
            played, (b) the team's players are worst at, or (c) all five
            heroes swapped for never-played same-role heroes. J must prefer
            the real draft; CONS, blind to players, should barely move.
Run (GPU host, from training/, run_rcf_full env): python personalization/p3t_extra.py <stage>
Outputs: $P3_RESULTS/p3t/{wpseeds,behavior,probes}.json
"""
import argparse
import glob
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
SEEDS = (42, 123, 777)


def res_file(name):
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    return os.path.join(C.RESULTS, "p3t", name)


def pair_files():
    import p3t_tourney as TT
    return sorted(f for f in glob.glob(os.path.join(TT.TDIR, "*__*.npz")) if ".tmp" not in f)


def rows_of(LB, acts):
    from shared import HEROES, MAPS, SKILL_TIERS
    n = len(acts)
    out = []
    for i in range(n):
        t0 = [HEROES[acts[i, k]] for k in range(16) if IS_PICK[k] and DRAFT_TEAM[k] == 0]
        t1 = [HEROES[acts[i, k]] for k in range(16) if IS_PICK[k] and DRAFT_TEAM[k] == 1]
        out.append((t0, t1, MAPS[LB.map[i]], SKILL_TIERS[LB.tier[i]]))
    return out


# ------------------------------------------------------------------ WP seeds

def wp_seed_models(v):
    import torch
    from sweep_enriched_wp import WinProbEnrichedModel
    from paper1_revision import core
    import p3t_systems as SY
    meta = json.load(open(SY.tpath(v.spec["wp_json"])))
    cols = np.asarray(core.group_cols(meta["spec"]["groups"]))
    wdir = os.path.dirname(SY.tpath(v.spec["wp_json"]))
    name = os.path.basename(v.spec["wp_json"])[:-5]
    out = {}
    for s in SEEDS:
        m = WinProbEnrichedModel(len(cols), meta["spec"]["arch"], dropout=0.3)
        m.load_state_dict(torch.load(os.path.join(wdir, f"{name}_s{s}.pt"), map_location="cpu", weights_only=True))
        m.eval()
        out[s] = m
    return out, cols, meta["selected_seed"]


def stage_wpseeds():
    import torch
    from paper1_revision import core
    import p3t_systems as SY
    import p3t_tourney as TT
    LB = TT.Lobbies()
    v = SY.vintage("cur")
    models, cols, sel = wp_seed_models(v)
    sd = core.stats_dict(v.stats())
    per = {}
    for f in pair_files():
        z = np.load(f)
        a, b = str(z["a"]), str(z["b"])
        Xf, Xs = core._chunk((rows_of(LB, z["acts"]), sd))
        for s, m in models.items():
            with torch.no_grad():
                pa = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
                pb = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
            p0 = 0.5 * (pa + 1 - pb)
            per.setdefault(a, {}).setdefault(s, []).append(p0)
            per.setdefault(b, {}).setdefault(s, []).append(1 - p0)
    out = {"seeds": list(SEEDS), "leaf_seed": int(sel), "systems": {}}
    for sysn, d in per.items():
        means = {s: float(100 * np.concatenate(d[s]).mean()) for s in SEEDS}
        # per draft: the seed spread and the lower bound
        allv = np.stack([np.concatenate(d[s]) for s in SEEDS])
        out["systems"][sysn] = {"wp_by_seed": {str(s): round(m_, 2) for s, m_ in means.items()},
                                "spread_pp": round(max(means.values()) - min(means.values()), 2),
                                "lcb_mean_min_over_seeds": round(float(100 * allv.min(0).mean()), 2),
                                "per_draft_sd_pp": round(float(100 * allv.std(0).mean()), 2),
                                "drafts": int(allv.shape[1])}
    json.dump(out, open(res_file("wpseeds.json"), "w"), indent=1)
    print(json.dumps(out)[:3000], flush=True)


# ------------------------------------------------------------------ behavior

def stage_behavior():
    import p3t_judge as JG
    import p3t_systems as SY
    import p3t_tourney as TT
    LB = TT.Lobbies()
    v = SY.vintage("cur")
    truth = JG.Truth()
    acc = {}
    t0 = time.time()
    for f in pair_files():
        z = np.load(f)
        acts = z["acts"]
        n = len(acts)
        sides = {0: str(z["a"]), 1: str(z["b"])}
        for k in range(16):
            t = DRAFT_TEAM[k]
            X = SY.features(acts, k, LB.map, LB.tier, t)
            m = SY.legal_mask(acts, k, np.ones((n, 2, 90), bool))
            p = SY.gd_probs(v, X[:, :289], m)
            h = acts[:, k]
            e = acc.setdefault(sides[t], {"picks": [], "gd_eq_pick": [], "gd_ll_pick": [], "gd_eq_ban": [],
                                          "gd_ll_ban": [], "resil": []})
            eq = (p.argmax(1) == h)
            ll = np.log(np.maximum(p[np.arange(n), h], 1e-9))
            if IS_PICK[k]:
                e["picks"].append(h)
                e["gd_eq_pick"].append(eq)
                e["gd_ll_pick"].append(ll)
                # resilience: minus the mean counter delta of the opponent's later picks against this pick
                later = [kk for kk in range(k + 1, 16) if IS_PICK[kk] and DRAFT_TEAM[kk] != t]
                if later:
                    r = np.full(n, np.nan)
                    for i in range(n):
                        ti = LB.tier[i]
                        w = truth.wr[ti]
                        ds = [truth.co[ti, acts[i, kk], h[i]] - (w[acts[i, kk]] + (100 - w[h[i]]) - 50)
                              for kk in later]
                        ds = np.array(ds, float)
                        if np.isfinite(ds).any():
                            r[i] = -np.nanmean(ds)
                    e["resil"].append(r)
            else:
                e["gd_eq_ban"].append(eq)
                e["gd_ll_ban"].append(ll)
    out = {}
    for sysn, e in acc.items():
        picks = np.concatenate(e["picks"])
        cnt = np.bincount(picks, minlength=90).astype(float)
        pr = cnt / cnt.sum()
        nz = pr[pr > 0]
        out[sysn] = {"picks": int(len(picks)), "distinct_heroes": int((cnt > 0).sum()),
                     "pick_entropy_bits": round(float(-(nz * np.log2(nz)).sum()), 3),
                     "top10_share": round(float(np.sort(pr)[-10:].sum()), 4),
                     "gd_similarity_picks": round(float(np.concatenate(e["gd_eq_pick"]).mean()), 4),
                     "gd_loglik_picks": round(float(np.concatenate(e["gd_ll_pick"]).mean()), 4),
                     "gd_similarity_bans": round(float(np.concatenate(e["gd_eq_ban"]).mean()), 4) if e["gd_eq_ban"] else None,
                     "gd_loglik_bans": round(float(np.concatenate(e["gd_ll_ban"]).mean()), 4) if e["gd_ll_ban"] else None,
                     "resilience": round(float(np.nanmean(np.concatenate(e["resil"]))), 3)}
    json.dump(out, open(res_file("behavior.json"), "w"), indent=1)
    print(f"behavior: {len(out)} systems ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ probes

def stage_probes():
    import p3t_judge as JG
    import p3t_tourney as TT
    from shared import HEROES, HERO_ROLE_FINE
    LB = TT.Lobbies()
    judge = json.load(open(os.path.join(JG.RES, JG.JNAME)))
    truth = JG.Truth()
    real = LB.real_acts
    n = len(real)
    role = np.array([HERO_ROLE_FINE[h] for h in HEROES])
    js = LB.state["judge"]["s"]                     # (n, 10, 90) judge skill, slot order
    ncur = LB.state["cur"]["n"]                     # earlier games per player and hero
    wr = np.nanmean(truth.wr, 0)                    # population win rate (judge window, all tiers)
    picks0 = [k for k in range(16) if IS_PICK[k] and DRAFT_TEAM[k] == 0]

    def swap(acts_i, k, kind, i):
        """same-role replacement for the hero at step k of team 0, or None"""
        h = acts_i[k]
        taken = set(acts_i.tolist())
        cand = [c for c in range(90) if role[c] == role[h] and c not in taken]
        if kind == "never":
            cand = [c for c in cand if (ncur[i, 0:5, c] == 0).all()]
            if not cand:
                return None
            return min(cand, key=lambda c: abs(wr[c] - wr[h]) if np.isfinite(wr[c]) else 1e9)
        if kind == "worst":
            if not cand:
                return None
            best = js[i, 0:5].max(0)                # the team's best player on each hero
            c = min(cand, key=lambda c: best[c])
            return c if best[c] < best[h] else None
        return None
    variants = {}
    for kind in ("never", "worst"):
        A = real.copy()
        ok = np.zeros(n, bool)
        for i in range(n):
            # the team's most skilled pick (largest best-player skill), swapped
            best = js[i, 0:5].max(0)
            ks = sorted(picks0, key=lambda k: -best[real[i, k]])
            for k in ks:
                c = swap(real[i], k, kind, i)
                if c is not None:
                    A[i, k] = c
                    ok[i] = True
                    break
        variants[f"one hero -> {kind}"] = (A, ok)
    A = real.copy()
    ok = np.ones(n, bool)
    for i in range(n):
        for k in picks0:
            c = swap(A[i], k, "never", i)
            if c is None:
                ok[i] = False
                break
            A[i, k] = c
    variants["all five -> never played"] = (A, ok)
    base = JG.score_drafts(LB, real, judge, truth)
    out = {"lobbies": int(n)}
    for nm, (A, ok) in variants.items():
        sc = JG.score_drafts(LB, A, judge, truth)
        r = {"lobbies_with_probe": int(ok.sum())}
        for m in ("judge_personal", "consensus"):
            d = 100 * (sc[m][ok] - base[m][ok])
            r[m] = {"mean_change_pp": round(float(d.mean()), 2),
                    "share_ranked_below_real": round(float((d < 0).mean()), 3)}
        r["population_wr_change_pp"] = round(float(np.nanmean([
            np.nanmean(wr[[A[i, k] for k in picks0]]) - np.nanmean(wr[[real[i, k] for k in picks0]])
            for i in np.flatnonzero(ok)])), 2)
        out[nm] = r
        print(nm, json.dumps(r), flush=True)
    json.dump(out, open(res_file("probes.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["wpseeds", "behavior", "probes"])
    a = ap.parse_args()
    {"wpseeds": stage_wpseeds, "behavior": stage_behavior, "probes": stage_probes}[a.stage]()


if __name__ == "__main__":
    main()
