"""
P3 rich CF: one table of variants x tests (results/P3_RICH_CF.md).

Reads, per variant tag and history (window or merged):
  results/rcf/params/<tag>_heldout.npz (or <tag>_<hist>_heldout.npz)  held-out predictive ll per player
  results/rcf/p3c/<tag>__<hist>.npz       online arm (V1 combiner, V2 and OOT games, slots)
  results/rcf/cassia_<tag>__<hist>.npz    masked cold start cells
  results/rcf/cover_<tag>__<hist>.json    interval coverage
Paired SEs against the reference arm: players (held-out ll), games (lift),
player-cluster bootstrap (Cassia R2, adoption log loss).

Usage (from training/):
  python3 personalization/p3_rcf_report.py --arms today:merged,cf2_marg:merged,... --ref today:merged --name round1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_rcf as R

RES = R.RCF_RES


def ll_of(p, y, eps=1e-7):
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def load_arm(tag, hist):
    z = np.load(os.path.join(RES, "p3c", f"{tag}__{hist}.npz"))
    return {k: z[k] for k in z.files}


def heldout(tag, hist):
    for nm in (f"{tag}_{hist}_heldout.npz", f"{tag}_heldout.npz"):
        p = os.path.join(RES, "params", nm)
        if os.path.exists(p):
            z = np.load(p)
            return z["ll"], z["players"]
    return None, None


def cluster_se(fn, groups, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    vals = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float)
        vals.append(fn(w))
    return float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    arms = [tuple(s.split(":")) for s in a.arms.split(",")]
    ref = tuple(a.ref.split(":"))
    out = {"ref": a.ref, "rows": {}}
    # reference arm
    RA = load_arm(*ref)
    y = RA["g_y"]
    win = RA["g_win"]
    pul = RA["g_all10_pulled"]
    LL0 = ll_of(RA["g_p0"], y)
    LLr = ll_of(RA["g_p"], y)
    hl_ref, hp_ref = heldout(*ref)
    cas_ref = np.load(os.path.join(RES, f"cassia_{ref[0]}__{ref[1]}.npz"))
    # adoption events from the reference arm's slots (V2, window-first game, 20+ earlier games)
    ev_ref = (RA["s_win"] == 2) & (RA["s_nh_slot"] == 0) & (RA["s_np_slot"] >= 20)
    for tag, hist in arms:
        nm = f"{tag}:{hist}"
        row = {}
        # held-out predictive ll
        hl, hp = heldout(tag, hist)
        if hl is not None:
            row["heldout_ll"] = float(hl.sum())
            if hl_ref is not None and np.array_equal(hp, hp_ref):
                dd = hl - hl_ref
                row["heldout_ll_vs_ref"] = float(dd.sum())
                row["heldout_ll_vs_ref_se"] = float(dd.std(ddof=1) * np.sqrt(len(dd)))
        pj = os.path.join(RES, "params", f"{tag}.json")
        if os.path.exists(pj):
            m = json.load(open(pj))
            row["effective_rank"] = m.get("effective_rank")
            row["v_sd_pp"] = [round(x, 2) for x in sorted(m.get("v_sd_pp", []), reverse=True)]
            row["overall_sd_pp"] = m.get("overall_sd_pp")
            row["hero_sd_pp"] = m.get("hero_sd_pp")
        # game-level lift
        A = load_arm(tag, hist)
        assert np.array_equal(A["g_replay_id"], RA["g_replay_id"])
        LL = ll_of(A["g_p"], y)
        for wk, wn in ((2, "V2"), (3, "OOT")):
            for sub, m in (("all", win == wk), ("all10", (win == wk) & pul)):
                g = LL0[m] - LL[m]
                dd = LLr[m] - LL[m]
                row[f"{wn}_{sub}"] = {"games": int(m.sum()), "gain": 1000 * float(g.mean()),
                                      "gain_se": 1000 * float(g.std(ddof=1) / np.sqrt(m.sum())),
                                      "vs_ref": 1000 * float(dd.mean()),
                                      "vs_ref_se": 1000 * float(dd.std(ddof=1) / np.sqrt(m.sum())),
                                      "acc": float(((A["g_p"][m] > 0.5) == (y[m] == 1)).mean())}
        # adoption (first game on a hero in the window, players with 20+ earlier games)
        assert np.array_equal(A["s_replay_id"], RA["s_replay_id"])
        ev = ev_ref
        r1, wp = A["s_r"][ev], A["s_wp"][ev].astype(np.float64)
        yy = (r1 + wp > 0.5).astype(float)
        pid = A["s_pid"][ev]

        def lls(pred):
            p = np.clip(wp + pred, 1e-6, 1 - 1e-6)
            return ll_of(p, yy)
        l_pred = lls(A["s_pred"][ev].astype(np.float64))
        l_mu = lls(A["s_mu"][ev].astype(np.float64))
        l_ref = lls(RA["s_pred"][ev].astype(np.float64))
        dd = l_ref - l_pred
        e2 = (r1 - A["s_pred"][ev]) ** 2
        row["adopt"] = {"events": int(ev.sum()), "realized_pp": 100 * float(r1.mean()),
                        "pred_pp": 100 * float(A["s_pred"][ev].mean()),
                        "bias_pp": 100 * float((r1 - A["s_pred"][ev]).mean()),
                        "ll_gain_vs_offset_x1000": 1000 * float((l_mu - l_pred).mean()),
                        "ll_vs_ref_x1000": 1000 * float(dd.mean()),
                        "ll_vs_ref_se_x1000": 1000 * cluster_se(lambda w: np.average(dd, weights=w), pid),
                        "excess_var_over_claimed": float((e2.mean() - (wp * (1 - wp)).mean())
                                                         / A["s_var"][ev].mean())}
        # never played in the full history (merged counts)
        ev2 = ev & (A["s_nh"] == 0)
        row["adopt"]["never_in_full_history"] = int(ev2.sum())
        # Cassia
        cp = os.path.join(RES, f"cassia_{tag}__{hist}.npz")
        if os.path.exists(cp):
            c = np.load(cp)
            js = json.load(open(cp.replace(".npz", ".json")))
            row["cassia_R2"] = js["all"]["latent_R2"]
            row["cassia_by_role"] = {k: v["latent_R2"] for k, v in js["by_role"].items()}
            row["cassia_cover80"] = js["all"]["cover80"]
            if np.array_equal(c["pi"], cas_ref["pi"]) and np.array_equal(c["hi"], cas_ref["hi"]):
                tot = (c["rbar"] ** 2).mean() - c["noise"].mean()
                e_a = (c["rbar"] - c["m"]) ** 2
                e_r = (cas_ref["rbar"] - cas_ref["m"]) ** 2
                dr = (e_r - e_a) / tot
                row["cassia_R2_vs_ref"] = float(dr.mean())
                row["cassia_R2_vs_ref_se"] = cluster_se(lambda w: np.average(dr, weights=w), c["pi"])
        cv = os.path.join(RES, f"cover_{tag}__{hist}.json")
        if os.path.exists(cv):
            js = json.load(open(cv))
            row["cover80"] = js["cover80"]
            row["cover_var_ratio"] = js["var_ratio"]
            row["cover_by_n"] = {k: round(v["cover80"], 3) for k, v in js["by_n"].items()}
        out["rows"][nm] = row
    lines = [f"# rich CF {a.name}; reference {a.ref}", "",
             "variant | eff. rank | held-out pred. LL vs ref (SE) | V2 lift (SE) | V2 vs ref (SE) | "
             "V2 all-10 vs ref (SE) | OOT lift (SE) | OOT vs ref (SE) | OOT all-10 vs ref (SE) | "
             "Cassia R2 (vs ref, SE) | adopt bias pp | adopt LL vs ref x1000 (SE) | cover80 | var ratio"]
    for nm, r in out["rows"].items():
        f = lambda k: f"{r[k]['gain']:+.2f} ({r[k]['gain_se']:.2f})"
        g = lambda k: f"{r[k]['vs_ref']:+.2f} ({r[k]['vs_ref_se']:.2f})"
        hl = (f"{r['heldout_ll_vs_ref']:+.1f} ({r['heldout_ll_vs_ref_se']:.1f})"
              if "heldout_ll_vs_ref" in r else "")
        cas = (f"{r['cassia_R2']:.3f} ({r.get('cassia_R2_vs_ref', 0):+.3f}, {r.get('cassia_R2_vs_ref_se', 0):.3f})"
               if "cassia_R2" in r else "")
        ad = r["adopt"]
        lines.append(f"{nm} | {r.get('effective_rank', '')} | {hl} | {f('V2_all')} | {g('V2_all')} | "
                     f"{g('V2_all10')} | {f('OOT_all')} | {g('OOT_all')} | {g('OOT_all10')} | {cas} | "
                     f"{ad['bias_pp']:+.2f} | {ad['ll_vs_ref_x1000']:+.2f} ({ad['ll_vs_ref_se_x1000']:.2f}) | "
                     f"{r.get('cover80', float('nan')):.3f} | {r.get('cover_var_ratio', float('nan')):.2f}")
    lines += ["", "Cassia R2 by role:"]
    for nm, r in out["rows"].items():
        if "cassia_by_role" in r:
            lines.append(f"{nm} | " + " | ".join(f"{k} {v:+.2f}" for k, v in r["cassia_by_role"].items()))
    with open(os.path.join(RES, f"report_{a.name}.json"), "w") as fo:
        json.dump(out, fo, indent=1)
    with open(os.path.join(RES, f"report_{a.name}.txt"), "w") as fo:
        fo.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
