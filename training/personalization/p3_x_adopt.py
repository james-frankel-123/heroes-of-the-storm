"""
P3 extensions, task 2: extrapolating to heroes a player has not played
(natural experiment).

Adoption event: a (player, hero) cell whose first game in the snapshot
window (2024-04-01 .. 2026-05-22) comes after the player already has 50+
games and 90+ days in the window, followed by 10+ games on the hero.
Because the corpus holds only replays uploaded to Heroes Profile, "no games
in the window" does not always mean a new hero. The hero level recorded on
the first game separates truly new heroes (level <= 3) from returning ones
(level >= 10).

Prediction: the causal (lag 1 day) posterior at the first game from
cache/x_predall.npz. Target: the mean detrended residual r_adj = r -
experience offset over the first N games on the hero (N = 10, 20). The
offset path over those games depends only on counts, so it is known in
advance; the "full" target adds it back. Latent R^2 = 1 - (MSE - noise) /
(Var - noise), noise = mean game noise / N.

Side information (cross-fitted ridge, 2 folds split by player):
  recalibrated GP means (CF rank 2, player, role kernels)
  hero attributes (Blizzard role, melee, hand-labeled high mechanics, CF
  loadings) and their products with the player's overall level
  scoreboard style: running mean over earlier days of z_raw or z_neu
  (p3_x_side), main effects and products with the adopted hero's role
  talent conformity (running mean) and its product with role
  hero level on the first game (profile information; flagged separately)

Usage (from training/): python3 personalization/p3_x_adopt.py
Outputs: results/p3_x_adopt.json, cache/x_adopt_events.npz (for task 3)
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import HKEY
import p3_hs_core as C
import p3_hero_level_causal as HL
import p3_x_side as XS
from p3_sd_similarity import HARD

PRED = os.path.join(C.CACHE, "x_predall.npz")
EVENTS = os.path.join(C.CACHE, "x_adopt_events.npz")
GP = {"player-only": "m_player", "player+hero (no hero pooling)": "m_ph",
      "role pooling": "m_role", "co-play pooling": "m_coplay",
      "similarity pooling (CF rank 2)": "m_cf2"}


def cell_order(d):
    key = d["pid"] * HKEY + d["hero"]
    o = np.lexsort((d["replay_id"], d["day"], key))
    k = key[o]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    lens = np.diff(np.r_[starts, len(k)])
    return o, starts, lens


def latent_r2(t, p, noise):
    mse = np.mean((t - p) ** 2)
    return float(1 - (mse - noise.mean()) / (t.var() - noise.mean()))


def _fill(ok, a):
    out = np.full(len(ok), np.nan)
    out[ok] = a
    return out


def _ridge_fit(A, t, lam):
    A1 = np.column_stack([np.ones(len(A)), A])
    R = lam * np.eye(A1.shape[1])
    R[0, 0] = 0
    return np.linalg.solve(A1.T @ A1 + R, A1.T @ t)


def ridge_cf(Xm, t, fold, lams=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5)):
    """Cross-fitted ridge predictions (standardize on the train fold; the
    penalty is chosen by an inner 2-fold split of the train fold)."""
    out = np.empty(len(t))
    rng = np.random.RandomState(7)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        mu, sd = Xm[tr].mean(0), Xm[tr].std(0) + 1e-9
        A = (Xm[tr] - mu) / sd
        tt = t[tr]
        inner = rng.rand(len(tt)) < 0.5
        best, bl = None, None
        for lam in lams:
            w = _ridge_fit(A[inner], tt[inner], lam)
            e = np.mean((tt[~inner] - np.column_stack([np.ones((~inner).sum()), A[~inner]]) @ w) ** 2)
            w2 = _ridge_fit(A[~inner], tt[~inner], lam)
            e += np.mean((tt[inner] - np.column_stack([np.ones(inner.sum()), A[inner]]) @ w2) ** 2)
            if best is None or e < best:
                best, bl = e, lam
        w = _ridge_fit(A, tt, bl)
        out[te] = np.column_stack([np.ones(te.sum()), (Xm[te] - mu) / sd]) @ w
    return out


def main():
    t0 = time.time()
    d = C.load_slots()
    P = np.load(PRED)
    P = {k: P[k] for k in P.files}
    meta = C.hero_meta(d["hero_names"])
    names = [str(h) for h in d["hero_names"]]
    H = len(names)
    table = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))["experience_table"]
    o, starts, lens = cell_order(d)
    first = o[starts]
    # player's first day in window
    fday = np.full(int(d["n_players"]), 10 ** 9)
    np.minimum.at(fday, d["pid"], d["day"])
    n_p = P["n_p"]
    n_ph = P["n_ph"]
    ev = (n_ph[first] == 0) & (n_p[first] >= 50) & (lens >= 1) & \
        (d["day"][first] - fday[d["pid"][first]] >= 90)
    es, el, ef = starts[ev], lens[ev], first[ev]
    print(f"adoption events: {ev.sum():,} (cells {len(starts):,})", flush=True)
    # level on this hero stamped before the event's first game started; no
    # earlier stamp on the hero = level 0 (no recorded experience)
    hl0 = np.nan_to_num(HL.load(d)["hero_lo"][ef], nan=0.0)
    r_adj = P["r_adj"].astype(np.float64)
    v = d["v"]
    # per-event first-N rows
    tg = {}
    for N in (1, 10, 20):
        ok = el >= N
        rows = o[es[ok][:, None] + np.arange(N)[None, :]]
        full = lambda a, ok=ok: _fill(ok, a)
        tg[N] = {"ok": ok, "t_adj": full(r_adj[rows].mean(1)), "t": full(d["r"][rows].mean(1)),
                 "mu_path": full((d["r"][rows] - r_adj[rows]).mean(1)),
                 "noise": full(v[rows].sum(1) / N ** 2)}
    g1 = r_adj[o[es]]
    out_sel = {"first_game_r_adj_pp_all_events": float(100 * g1.mean()),
               "first_game_r_adj_pp_events_reaching_10": float(100 * g1[el >= 10].mean()),
               "first_game_r_adj_pp_events_stopping_before_10": float(100 * g1[el < 10].mean()),
               "share_reaching_10": float((el >= 10).mean()),
               "first_game_raw_r_pp_all_events": float(100 * d["r"][o[es]].mean())}
    print(out_sel, flush=True)
    # side information at the first game (earlier days only)
    side = XS.load()
    zr, cnt = XS.running_mean_prior_days(d, side["z"].astype(np.float64), side["valid"].astype(float))
    zn, _ = XS.running_mean_prior_days(d, side["zn"].astype(np.float64), side["valid"].astype(float))
    cf = np.nan_to_num(side["conf"].astype(np.float64), nan=-1)
    cm, ccnt = XS.running_mean_prior_days(d, cf[:, None], (cf >= 0).astype(float))
    shrink = (cnt / (cnt + 20.0))[:, None]
    Zr, Zn = (zr * shrink)[ef], (zn * shrink)[ef]
    conf = np.where(ccnt[ef] > 0, cm[ef, 0], np.nanmean(side["conf"]))
    del zr, zn, cm
    h = d["hero"][ef]
    role = meta["blizz"][h]
    role_oh = np.eye(6)[role]
    hard = np.array([n in HARD for n in names], float)[h]
    melee = meta["melee"].astype(float)[h]
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    ev_, U = np.linalg.eigh(kz["basis_cf2"])
    load2 = U[:, -2:] * np.sqrt(np.maximum(ev_[-2:], 0))
    cfl = load2[h]
    mP = P["m_player"][ef].astype(float)
    attrs = np.column_stack([role_oh, hard, melee, cfl])
    base = np.column_stack([P["m_cf2"][ef], P["m_player"][ef], P["m_role"][ef]]).astype(float)
    with_attr = np.column_stack([base, attrs, attrs * mP[:, None]])
    style_r = np.column_stack([Zr] + [Zr * role_oh[:, j:j + 1] for j in range(6)])
    style_n = np.column_stack([Zn] + [Zn * role_oh[:, j:j + 1] for j in range(6)])
    tal = np.column_stack([conf, conf[:, None] * role_oh])
    hlb = np.column_stack([hl0 <= 3, (hl0 > 3) & (hl0 <= 9), hl0 >= 10]).astype(float)
    hlx = np.column_stack([hlb, hlb * mP[:, None]])
    fold = (np.random.RandomState(1).rand(int(d["n_players"])) < 0.5).astype(int)[d["pid"][ef]]
    models = {
        "ridge: GP means (recalibrated)": base,
        "+ hero attributes": with_attr,
        "+ attributes + style (raw)": np.column_stack([with_attr, style_r]),
        "+ attributes + style (outcome-neutral)": np.column_stack([with_attr, style_n]),
        "+ attributes + style (raw) + talents": np.column_stack([with_attr, style_r, tal]),
        "+ all + hero level on first game": np.column_stack([with_attr, style_r, tal, hlx]),
        "hero level only (+ attributes)": np.column_stack([with_attr, hlx]),
    }
    out = {"n_events": int(ev.sum()), "selection": out_sel, "events_by_hero_level": {
        "<=3 (truly new)": int((hl0 <= 3).sum()), "4-9": int(((hl0 > 3) & (hl0 < 10)).sum()),
        ">=10 (returning)": int((hl0 >= 10).sum())}, "by_N": {}}
    groups = {"all": np.ones(len(ef), bool), "truly new (level<=3)": hl0 <= 3,
              "returning (level>=10)": hl0 >= 10}
    for rn in range(6):
        groups[f"role {C.BLIZZ_ROLES[rn]}"] = role == rn
    groups["high mechanics"] = hard > 0
    rng = np.random.RandomState(0)
    pid_e = d["pid"][ef]
    for N in (1, 10, 20):
        T = tg[N]
        ok = T["ok"]
        t, noise = T["t_adj"][ok], T["noise"][ok]
        preds = {"zero (population)": np.zeros(ok.sum())}
        for nm, col in GP.items():
            preds[nm] = P[col][ef][ok].astype(float)
        for nm, col in GP.items():
            preds[nm + ", recalibrated"] = ridge_cf(P[col][ef][ok].astype(float)[:, None], t, fold[ok],
                                                   lams=(1e-6,))
        for nm, Xm in models.items():
            preds[nm] = ridge_cf(Xm[ok], t, fold[ok])
        res = {"events": int(ok.sum()), "target_sd_pp": float(100 * t.std()),
               "noise_sd_pp": float(100 * np.sqrt(noise.mean())),
               "latent_sd_pp": float(100 * np.sqrt(max(t.var() - noise.mean(), 0))),
               "mean_target_pp": float(100 * t.mean()), "methods": {}}
        for nm, p in preds.items():
            row = {}
            for gn, gm in groups.items():
                gm = gm[ok]
                if gm.sum() < 200:
                    continue
                row[gn] = {"n": int(gm.sum()), "latent_r2": latent_r2(t[gm], p[gm], noise[gm]),
                           "rmse_pp": float(100 * np.sqrt(np.mean((t[gm] - p[gm]) ** 2))),
                           "bias_pp": float(100 * np.mean(t[gm] - p[gm])),
                           "slope": float(np.cov(t[gm], p[gm])[0, 1] / p[gm].var())
                           if p[gm].std() > 1e-9 else None}
            res["methods"][nm] = row
        # paired MSE differences, player-clustered bootstrap
        up, inv = np.unique(pid_e[ok], return_inverse=True)
        pairs = [("similarity pooling (CF rank 2), recalibrated", "role pooling, recalibrated"),
                 ("similarity pooling (CF rank 2), recalibrated", "player-only, recalibrated"),
                 ("role pooling, recalibrated", "player-only, recalibrated"),
                 ("co-play pooling, recalibrated", "player-only, recalibrated"),
                 ("similarity pooling (CF rank 2), recalibrated", "zero (population)"),
                 ("similarity pooling (CF rank 2)", "role pooling"),
                 ("similarity pooling (CF rank 2)", "player-only"),
                 ("role pooling", "player-only"),
                 ("similarity pooling (CF rank 2)", "player+hero (no hero pooling)"),
                 ("+ attributes + style (raw) + talents", "ridge: GP means (recalibrated)"),
                 ("+ attributes + style (raw)", "+ hero attributes"),
                 ("+ attributes + style (outcome-neutral)", "+ hero attributes"),
                 ("+ hero attributes", "ridge: GP means (recalibrated)"),
                 ("+ all + hero level on first game", "+ attributes + style (raw) + talents")]
        res["paired_r2_diff"] = {}
        vt = t.var() - noise.mean()
        for a, b in pairs:
            dl = ((t - preds[b]) ** 2 - (t - preds[a]) ** 2)
            per = np.bincount(inv, weights=dl, minlength=len(up))
            cntp = np.bincount(inv, minlength=len(up))
            bs = []
            for _ in range(300):
                s = rng.randint(0, len(up), len(up))
                bs.append(per[s].sum() / cntp[s].sum() / vt)
            res["paired_r2_diff"][f"{a} - {b}"] = {
                "est": float(dl.mean() / vt), "ci": [float(np.percentile(bs, 2.5)),
                                                     float(np.percentile(bs, 97.5))]}
        # full target (with the offset path): does the product number hold?
        tf = T["t"][ok]
        full = {"zero": np.zeros(ok.sum()), "offset path only": T["mu_path"][ok],
                "offset path + CF rank 2": T["mu_path"][ok] + preds["similarity pooling (CF rank 2)"],
                "offset at n=0 + CF rank 2 (the displayed number)": P["mu"][ef][ok]
                + preds["similarity pooling (CF rank 2)"]}
        res["full_target"] = {"mean_realized_pp": float(100 * tf.mean())}
        for nm, p in full.items():
            res["full_target"][nm] = {"latent_r2": latent_r2(tf, p, noise),
                                      "bias_pp": float(100 * np.mean(tf - p)),
                                      "mean_pred_pp": float(100 * p.mean())}
        # coverage of the calibrated display band for the N-game mean
        s2 = P["var_cf2"][ef][ok].astype(float)
        pdisp = P["mu"][ef][ok] + preds["similarity pooling (CF rank 2)"]
        cov = {}
        for lab, s2d in (("raw posterior", s2), ("calibrated 1.2 s2 + 7pp^2", 1.2 * s2 + 0.0049)):
            zz = (tf - pdisp) / np.sqrt(s2d + noise)
            cov[lab] = float((np.abs(zz) < 1.2816).mean())
        zz = (tf - T["mu_path"][ok] - preds["similarity pooling (CF rank 2)"]) / np.sqrt(1.2 * s2 + noise)
        cov["offset path, 1.2 s2 (no 7pp)"] = float((np.abs(zz) < 1.2816).mean())
        res["coverage80_display"] = cov
        out["by_N"][str(N)] = res
        print(f"N={N}: {ok.sum():,} events, latent sd {res['latent_sd_pp']:.2f}pp", flush=True)
        for nm in preds:
            a = res["methods"][nm]["all"]
            print(f"  {nm:45s} R2 {a['latent_r2']:+.3f} slope {a['slope'] if a['slope'] is None else round(a['slope'], 2)}"
                  f" tank {res['methods'][nm].get('role Tank', {}).get('latent_r2', float('nan')):+.3f}",
                  flush=True)
        print("  paired:", {k: round(v_['est'], 4) for k, v_ in res["paired_r2_diff"].items()}, flush=True)
        print("  full:", res["full_target"], res["coverage80_display"], flush=True)
    # first game on the new hero, every adoption event (no survivorship):
    # per-slot log-loss gain x1000 of wp + prediction (offset at n = 0 + GP mean)
    rr = d["r"][ef]
    wpv = d["wp"][ef]
    yy = rr + wpv

    def llg(pred):
        p1 = np.clip(wpv + pred, 1e-3, 1 - 1e-3)
        return 1000 * (-(yy * np.log(wpv) + (1 - yy) * np.log(1 - wpv))
                       + (yy * np.log(p1) + (1 - yy) * np.log(1 - p1)))
    mu0 = P["mu"][ef].astype(float)
    fg = {"offset only": llg(mu0)}
    for nm, col in GP.items():
        fg[nm] = llg(mu0 + P[col][ef].astype(float))
    up, inv = np.unique(pid_e, return_inverse=True)
    res1 = {"events": int(len(ef)), "mean_raw_r_pp": float(100 * rr.mean()),
            "mean_offset_pp": float(100 * mu0.mean()), "gain_x1000": {}, "paired_x1000": {}}
    for gn, gm in groups.items():
        res1["gain_x1000"][gn] = {nm: float(v_[gm].mean()) for nm, v_ in fg.items()}
    for a_, b_ in (("similarity pooling (CF rank 2)", "role pooling"),
                   ("similarity pooling (CF rank 2)", "player-only"),
                   ("role pooling", "player-only"),
                   ("similarity pooling (CF rank 2)", "offset only"),
                   ("player-only", "offset only")):
        dl = fg[a_] - fg[b_]
        per = np.bincount(inv, weights=dl, minlength=len(up))
        cntp = np.bincount(inv, minlength=len(up))
        bs = []
        for _ in range(300):
            s_ = rng.randint(0, len(up), len(up))
            bs.append(per[s_].sum() / cntp[s_].sum())
        res1["paired_x1000"][f"{a_} - {b_}"] = [float(dl.mean()), float(np.percentile(bs, 2.5)),
                                                float(np.percentile(bs, 97.5))]
    out["first_game_all_events"] = res1
    print("first game:", json.dumps(res1["paired_x1000"]), json.dumps(res1["gain_x1000"]["all"]), flush=True)
    np.savez(EVENTS, first_row=ef, start=es, length=el, hl0=hl0, order=o)
    with open(os.path.join(C.RESULTS, "p3_x_adopt.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
