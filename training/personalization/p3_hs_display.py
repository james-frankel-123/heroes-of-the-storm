"""
P3 hero strength, stage 4: what the product would show.

1. sd-vs-n curve from the best kernel: posterior sd of a player's strength
   on hero h after n games on h, for (a) a player with no other games and
   (b) a player with 300 games spread over their usual heroes (drawn from a
   real deep player's play mix).
2. An example profile: one anonymous deep player (held-out half), as of the
   end of the snapshot, all heroes with posterior mean +- 80% band, games
   played, and a confidence label.

Usage (from training/): python3 personalization/p3_hs_display.py
Output: results/p3_hs_display.json and a text table on stdout.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
from p3_hs_fit import KERNELS, KOUT, main_kernels, prepare

Z80 = 1.2815516
V_GAME = 0.2418  # mean wp(1-wp)


def label(lo, hi):
    if lo > 0:
        return "clear strength"
    if hi < 0:
        return "clear weakness"
    return "unclear"


def main():
    d = C.load_slots()
    H = len(d["hero_names"])
    names = d["hero_names"]
    _, best = main_kernels()
    kz = np.load(KOUT)
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    out = {"kernel": best, "sd_curve": {}}
    _, _, _, table, r_adj = prepare(d)
    with open(os.path.join(C.RESULTS, "p3_hs_calib.json")) as f:
        cal = json.load(f)[f"{best} hl=inf"]["coef"]
    c_cal, d2 = cal["c"], (cal["delta_sd_pp"] / 100) ** 2
    k0, k1 = (cal["kappa0_sd_pp"] / 100) ** 2, (cal["kappa1_sd_pp"] / 100) ** 2
    out["calibration"] = cal

    def disp_var(s2, n):
        return c_cal * s2 + d2 + k0 * (n == 0) + k1 * ((n >= 1) & (n <= 5))
    for kname in ("player+hero", "player+role+melee+hero", best):
        K = C.kernel_from([bases[b] for b in KERNELS[kname]], kz["theta_" + kname])
        curve = {}
        # background: a real deep player's play mix, scaled to 300 games
        for n in (0, 1, 3, 5, 10, 20, 50, 100, 200, 500):
            row = {}
            for bg in ("no other games", "300 games on other heroes"):
                Pv = np.zeros((1, H))
                if bg != "no other games":
                    mix = np.bincount(np.arange(10) * 7 % H, weights=np.linspace(60, 10, 10),
                                      minlength=H)
                    Pv[0] = mix / mix.sum() * 300 / V_GAME
                h = 45 if Pv[0, 45] == 0 else int(np.flatnonzero(Pv[0] == 0)[0])
                Pv[0, h] = n / V_GAME
                Sv = np.zeros((1, H))
                _, s2 = C.gp_query(Sv, Pv, np.array([h]), K)
                row[bg] = float(100 * np.sqrt(s2[0]))
                row[bg + " (calibrated)"] = float(100 * np.sqrt(disp_var(s2[0], n)))
            curve[n] = row
        out["sd_curve"][kname] = curve
    # example profile
    K = C.kernel_from([bases[b] for b in KERNELS[best]], kz["theta_" + best])
    e_all = np.ones(len(d["day"]), bool)
    S, P, N, W, R = C.static_state(d, e_all, d["n_players"], H, r_adj)
    hold = ~kz["fit_players"]
    tot = N.sum(1)
    cand = np.flatnonzero(hold & (tot >= 400) & (tot <= 600) & ((N >= 5).sum(1) >= 12))
    p = int(cand[len(cand) // 2])
    m, v = C.gp_all(S[p:p + 1], P[p:p + 1], K)
    nph = N[p].astype(np.int64)
    mu = table[C.exp_bins(np.full(H, int(tot[p])), nph)]
    m = 100 * (mu + m[0])
    sd = 100 * np.sqrt(disp_var(v[0], nph))
    order = np.argsort(-m)
    rows = []
    for h in order:
        rows.append({"hero": str(names[h]), "games": int(N[p, h]),
                     "raw_wr": None if N[p, h] == 0 else float(100 * W[p, h] / N[p, h]),
                     "strength_pp": float(m[h]), "lo80": float(m[h] - Z80 * sd[h]),
                     "hi80": float(m[h] + Z80 * sd[h]), "sd_pp": float(sd[h]),
                     "label": label(m[h] - Z80 * sd[h], m[h] + Z80 * sd[h])})
    out["example_player"] = {"total_games": int(tot[p]), "heroes": rows}
    print(f"example player: {int(tot[p])} games; kernel {best}")
    print(f"{'hero':16s} {'games':>5s} {'raw WR':>7s} {'strength':>9s} {'80% band':>16s}  label")
    for r in rows[:15] + rows[-6:]:
        wr = "" if r["raw_wr"] is None else f"{r['raw_wr']:.0f}%"
        print(f"{r['hero']:16s} {r['games']:5d} {wr:>7s} {r['strength_pp']:+8.1f}pp "
              f"[{r['lo80']:+5.1f}, {r['hi80']:+5.1f}]  {r['label']}")
    for k, c in out["sd_curve"].items():
        print(k)
        for n, row in c.items():
            print(f"  n={n:4d}  " + "  ".join(f"{b}: {s:.2f}pp" for b, s in row.items()))
    with open(os.path.join(C.RESULTS, "p3_hs_display.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
