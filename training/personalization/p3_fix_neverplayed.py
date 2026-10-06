"""
Audit fix P3-05: calibration of the never-played band without selection on
future play.

Old test: coverage on cells with >= 10 (or >= 5) future games. Whether a
newly taken-up hero gets played again depends on its first outcomes, so the
scored population was selected on the outcome; and the target subtracted the
offset of each future row (future counts), not the prediction made before the
first game.

New test: every adoption event in V2 (2026-03-29 .. 2026-05-22; the band was
calibrated on V1), i.e. a player's first game in the window on a hero, for
players with 20+ earlier games. The prediction is made before the game: lag-1
state (games on earlier days), lag-1 experience offset for a never-played hero
(table refit on lag-1 counts, P3-03), GP posterior mean and variance of the
phase-1 "+CF rank 2" kernel. Display variance = c var + kappa0^2, the
P3_HERO_STRENGTH calibration refit on the lag-1 contract (results/fix/p3_hs_calib.json).
Scored on the FIRST game of every event (no selection):
  bias            mean(r1 - prediction), r1 = y - WP_pop
  var ratio       (mean((r1 - pred)^2) - mean(wp (1 - wp))) / mean(display var)
                  and the same against the raw posterior variance
  log loss        Bernoulli log loss of p = WP_pop + prediction vs WP_pop alone
                  and vs WP_pop + offset only (proper score of the mean)
Secondary, flagged as selected: the mean of the first 3 games for events
that reached 3 games, prediction path offset(n_p, j - 1) + m.
CIs: player-cluster bootstrap (500).

Run (from training/): OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_fix_neverplayed.py
Output: results/fix/p3_fix_neverplayed.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from p3_heroes import HKEY
import p3_hs_core as C
import p3_x_common as X
import p3_fix_counts as F

C_SCALE, KAPPA0 = 1.2031, 0.07038
_CAL = os.path.join(F.FIX_RESULTS, "p3_hs_calib.json")
if os.path.exists(_CAL):  # calibration refit on the lag-1 contract (p3_hs_calib via run_fixed)
    with open(_CAL) as _f:
        _c = json.load(_f)["+CF rank 2 hl=inf"]["coef"]
    C_SCALE, KAPPA0 = float(_c["c"]), float(_c["kappa0_sd_pp"]) / 100


def cluster_ci(vals_fn, groups, rng, n=500):
    ug, inv = np.unique(groups, return_inverse=True)
    out = []
    for _ in range(n):
        w = np.bincount(rng.randint(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float)
        out.append(vals_fn(w))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    rng = np.random.RandomState(0)
    d = C.load_slots()
    e_mask, n_p, n_ph, table, r_adj = F.prepare_l1(d)
    Ks, _ = X.kernels()
    K = Ks["+CF rank 2"]
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    # first row of each (player, hero) in play order (day, replay_id)
    key = d["pid"] * HKEY + d["hero"]
    o = np.lexsort((d["replay_id"], days, key))
    ks = key[o]
    first = np.r_[True, ks[1:] != ks[:-1]]
    rank = np.arange(len(o)) - np.repeat(np.flatnonzero(first), np.diff(np.r_[np.flatnonzero(first), len(o)]))
    rk = np.empty(len(o), np.int64)
    rk[o] = rank
    ev = (rk == 0) & (days >= med) & (n_p >= 20)
    qi, m, v, Nh, Np, _ = X.online_predict(d, np.ones(len(days), bool), ev, K, r_adj)
    assert (Nh == 0).all()
    mu0 = table[C.exp_bins(Np.astype(np.int64), np.zeros(len(qi), np.int64))]
    pred = mu0 + m
    disp = C_SCALE * v + KAPPA0 ** 2
    r1 = d["r"][qi]
    wv = d["v"][qi]
    pid = d["pid"][qi]
    y = d["y"][qi]
    wp = d["wp"][qi]
    out = {"events": int(len(qi)), "players": int(len(np.unique(pid))), "split_day": float(med)}

    def stats(w):
        e2 = np.average((r1 - pred) ** 2, weights=w) - np.average(wv, weights=w)
        return e2
    bias = float(np.mean(r1 - pred))
    out["first_game"] = {
        "realized_mean_resid_pp": float(100 * r1.mean()),
        "predicted_mean_pp": float(100 * pred.mean()),
        "offset_only_mean_pp": float(100 * mu0.mean()),
        "bias_pp": 100 * bias,
        "bias_ci_pp": [100 * x for x in cluster_ci(lambda w: np.average(r1 - pred, weights=w), pid, rng)],
        "excess_var_pp2": float(1e4 * stats(np.ones(len(r1)))),
        "display_var_pp2": float(1e4 * disp.mean()),
        "raw_posterior_var_pp2": float(1e4 * v.mean()),
        "var_ratio_vs_display": float(stats(np.ones(len(r1))) / disp.mean()),
        "var_ratio_vs_display_ci": [x / disp.mean() for x in cluster_ci(stats, pid, rng)],
        "var_ratio_vs_raw_posterior": float(stats(np.ones(len(r1))) / v.mean()),
    }
    eps = 1e-6
    ll = lambda p: float(-np.mean(y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))))
    out["first_game"]["log_loss"] = {"WP_pop": ll(wp), "WP_pop + offset": ll(wp + mu0), "WP_pop + offset + GP mean": ll(wp + pred)}
    # by player history
    for lo_, hi_ in ((20, 100), (100, 300), (300, 10 ** 9)):
        s = (Np >= lo_) & (Np < hi_)
        e2 = np.mean((r1[s] - pred[s]) ** 2) - np.mean(wv[s])
        out["first_game"][f"players with {lo_}-{hi_ if hi_ < 10 ** 9 else ''} games"] = {
            "events": int(s.sum()), "bias_pp": float(100 * np.mean(r1[s] - pred[s])),
            "var_ratio_vs_display": float(e2 / disp[s].mean())}
    # Reference for the noise model. Single-game excess variance e^2 - wp(1-wp)
    # is dominated by small misfits of the WP noise (P3-10). The same quantity on
    # V2 slots of the same players' established heroes (claimed variance a few
    # pp^2) estimates that misfit; the difference isolates the adoption part.
    rs = np.random.RandomState(1)
    refq = (days >= med) & (n_p >= 20) & (n_ph >= 1) & ~ev
    refq &= rs.rand(len(days)) < 400000 / max(refq.sum(), 1)
    qr, mr, vr_, Nhr, Npr, _ = X.online_predict(d, np.ones(len(days), bool), refq, K, r_adj)
    mur = table[C.exp_bins(Npr.astype(np.int64), Nhr.astype(np.int64))]
    er = d["r"][qr] - (mur + mr)
    dvr = C_SCALE * vr_
    ref = {}
    for lo_, hi_ in ((1, 5), (5, 20), (20, 50), (50, 10 ** 9)):
        s = (Nhr >= lo_) & (Nhr < hi_)
        ex = float(1e4 * (np.mean(er[s] ** 2) - np.mean(d["v"][qr][s])))
        ref[f"n_ph {lo_}-{hi_ if hi_ < 10 ** 9 else ''}"] = {
            "slots": int(s.sum()), "excess_var_pp2": ex, "display_var_pp2": float(1e4 * dvr[s].mean()),
            "excess_se_pp2": float(1e4 * np.std(er[s] ** 2 - d["v"][qr][s]) / np.sqrt(s.sum()))}
    out["reference_established_slots"] = ref
    deep = ref["n_ph 50-"]
    adj = out["first_game"]["excess_var_pp2"] - deep["excess_var_pp2"]
    out["first_game"]["excess_var_minus_deep_reference_pp2"] = adj
    out["first_game"]["excess_var_minus_deep_reference_se_pp2"] = float(np.sqrt(
        deep["excess_se_pp2"] ** 2 + (1e4 * np.std((r1 - pred) ** 2 - wv) / np.sqrt(len(r1))) ** 2))
    out["first_game"]["claimed_minus_deep_reference_claimed_pp2"] = float(1e4 * disp.mean() - deep["display_var_pp2"])
    out["first_game"]["claimed_raw_minus_deep_reference_raw_pp2"] = float(1e4 * v.mean() - deep["display_var_pp2"] / C_SCALE)
    # secondary: first 3 games of events that reached 3 (selected)
    pos = np.empty(len(o), np.int64)
    pos[o] = np.arange(len(o))
    p0 = pos[qi]
    ok3 = (p0 + 2 < len(o))
    ok3[ok3] &= (ks[p0[ok3] + 2] == ks[p0[ok3]])
    rows3 = np.stack([o[np.minimum(p0 + j, len(o) - 1)] for j in range(3)], 1)[ok3]
    pred3 = np.stack([table[C.exp_bins(Np[ok3].astype(np.int64), np.full(ok3.sum(), j, np.int64))] for j in range(3)], 1) \
        + m[ok3][:, None]
    rb = d["r"][rows3].mean(1)
    pb = pred3.mean(1)
    noise = d["v"][rows3].sum(1) / 9.0
    e2 = np.mean((rb - pb) ** 2) - np.mean(noise)
    z = np.abs(rb - pb) / np.sqrt(disp[ok3] + noise)
    out["first_3_games_selected"] = {
        "events_reaching_3": int(ok3.sum()), "share_of_events": float(ok3.mean()),
        "first_game_resid_of_these_pp": float(100 * r1[ok3].mean()),
        "first_game_resid_of_the_rest_pp": float(100 * r1[~ok3].mean()),
        "bias_pp": float(100 * np.mean(rb - pb)), "var_ratio_vs_display": float(e2 / disp[ok3].mean()),
        "cover80": float((z < 1.2816).mean())}
    print(json.dumps(out, indent=1))
    os.makedirs(F.FIX_RESULTS, exist_ok=True)
    with open(os.path.join(F.FIX_RESULTS, "p3_fix_neverplayed.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
