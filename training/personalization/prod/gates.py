"""
Regression battery and deployment gates for the personal layer (review 5.4).

A golden set is a frozen npz (written by make_golden) holding, per split
("v2" and "oot"):
  per game   replay_id, start_ts, y (1 = team 0 won), wp (population WP for
             team 0), p (personal model for team 0), x_skill (team 0 minus
             team 1 summed skill means, the combiner's skill feature),
             optional p_offset (offset-only model), optional tier (label)
  per slot   slot_n (player's prior games on the hero), slot_pred (predicted
             personal residual), slot_real (realized residual y_team - wp_team),
             optional slot_start_ts and slot_state_fetched_max (latest
             fetched_at among the state rows the prediction used)
plus hero_names and a meta JSON (reference tier distribution, sha256 of the
arrays, creation time).

Checks (thresholds are the named constants below; every one is recorded):
  gain_v2 / gain_oot   reference gain inside the game-bootstrap CI of
                       LL(wp) - LL(p), widened by DESIGN_WIDEN for the
                       design effect
  offset_gain_v2       same for the offset-only model, when p_offset exists
  cal_slope_*, ece_*   calibration of p
  skill_coef_*         logistic refit y ~ logit(wp) + x_skill: the x_skill
                       coefficient inside SKILL_COEF_BAND
  slot_slope_*         realized-on-predicted slope by prior games n
  neverplayed_bias_*   mean(realized - predicted) on n = 0 slots, pp
  visibility_*         no slot used a state row fetched at or after the
                       game's start
  tiers                tier label distribution equals the frozen one (and
                       a fresh label file, if given, matches row by row)
  hero_set             golden hero list is the 90-hero v1 set (p3_heroes)
Deployment gates (on OOT): latest-month gain >= DEPLOY_MIN_MONTH_GAIN and
skill coefficient inside DEPLOY_SKILL_COEF_BAND.

CLI (from training/):
  python3 -m personalization.prod.gates --golden G.npz --out report.json
      [--labels L.npz] [--thresholds overrides.json]
Exit code 0 iff every check passes.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3_DIR = os.path.dirname(HERE)

REF_GAIN = {"v2": 0.0142, "oot": 0.0149}
REF_OFFSET_GAIN = {"v2": 0.0083}
DESIGN_WIDEN = 1.3          # replay bootstrap CIs are about 1.3x too narrow
BOOT = 400
CAL_SLOPE_BAND = (0.95, 1.05)
ECE_MAX = 0.006
SKILL_COEF_BAND = (3.4, 4.0)
DEPLOY_SKILL_COEF_BAND = (3.0, 4.5)
SLOT_SLOPE = {"n0": (1.1, 1.5), "n1_20": (0.9, 1.2), "n21p": (0.65, np.inf)}
NEVERPLAYED_BIAS_PP = 1.0
DEPLOY_MIN_MONTH_GAIN = 0.010
TIER_TOL = 1e-9             # frozen distribution must match exactly
SPLITS = ("v2", "oot")
GAME_KEYS = ("replay_id", "start_ts", "y", "wp", "p", "x_skill")
OPT_GAME_KEYS = ("p_offset", "tier")
SLOT_KEYS = ("slot_n", "slot_pred", "slot_real")
OPT_SLOT_KEYS = ("slot_start_ts", "slot_state_fetched_max")
EPS = 1e-7


def thresholds():
    """Current thresholds as a JSON-able dict (what --thresholds overrides)."""
    return {"REF_GAIN": dict(REF_GAIN), "REF_OFFSET_GAIN": dict(REF_OFFSET_GAIN),
            "DESIGN_WIDEN": DESIGN_WIDEN, "BOOT": BOOT, "CAL_SLOPE_BAND": list(CAL_SLOPE_BAND),
            "ECE_MAX": ECE_MAX, "SKILL_COEF_BAND": list(SKILL_COEF_BAND),
            "DEPLOY_SKILL_COEF_BAND": list(DEPLOY_SKILL_COEF_BAND),
            "SLOT_SLOPE": {k: list(v) for k, v in SLOT_SLOPE.items()},
            "NEVERPLAYED_BIAS_PP": NEVERPLAYED_BIAS_PP, "DEPLOY_MIN_MONTH_GAIN": DEPLOY_MIN_MONTH_GAIN}


# ------------------------------------------------------------------ golden

def _digest(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def make_golden(path, splits, hero_names, note=""):
    """Freeze a golden set. splits = {"v2": {...}, "oot": {...}} with the game
    and slot arrays named in the module docstring. Stores the tier
    distribution and a content hash in meta."""
    arrays = {}
    meta = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "note": note, "splits": {}}
    for s, d in splits.items():
        miss = [k for k in GAME_KEYS + SLOT_KEYS if k not in d]
        if miss:
            raise ValueError(f"golden split {s!r} lacks {miss}")
        n = len(d["y"])
        for k in GAME_KEYS + OPT_GAME_KEYS:
            if k in d:
                a = np.asarray(d[k])
                if len(a) != n:
                    raise ValueError(f"{s}.{k}: length {len(a)} != {n}")
                arrays[f"{s}__{k}"] = a.astype("U16") if k == "tier" else a
        ns = len(d["slot_n"])
        for k in SLOT_KEYS + OPT_SLOT_KEYS:
            if k in d:
                if len(d[k]) != ns:
                    raise ValueError(f"{s}.{k}: length {len(d[k])} != {ns}")
                arrays[f"{s}__{k}"] = np.asarray(d[k])
        info = {"games": n, "slots": ns}
        if "tier" in d:
            info["tier_dist"] = tier_dist(d["tier"])
        meta["splits"][s] = info
    arrays["hero_names"] = np.asarray([str(h) for h in hero_names])
    meta["sha256"] = _digest(arrays)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return meta


def load_golden(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    arrays = {k: z[k] for k in z.files if k != "meta"}
    got = _digest(arrays)
    if got != meta["sha256"]:
        raise ValueError(f"{path}: golden content hash {got[:12]} != frozen {meta['sha256'][:12]}")
    splits = {}
    for k, a in arrays.items():
        if "__" in k:
            s, f = k.split("__", 1)
            splits.setdefault(s, {})[f] = a
    return {"meta": meta, "splits": splits, "hero_names": arrays["hero_names"]}


# ------------------------------------------------------------------ metrics

def ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_logistic(X, y, iters=100, l2=1e-8):
    """Newton logistic regression with intercept; returns [b0, b1, ...]."""
    X1 = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        g = X1.T @ (y - p) - l2 * w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def ece(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean()
                     for k in range(bins) if (b == k).any()))


def ols_slope(x, y):
    x = x - x.mean()
    den = float(x @ x)
    return float(x @ (y - y.mean()) / den) if den > 0 else float("nan")


def gain_ci(wp, p, y, boot, widen, seed=0):
    """Mean LL(wp) - LL(p) and a game-bootstrap 95% CI widened about the mean."""
    diff = ll_vec(wp, y) - ll_vec(p, y)
    g = float(diff.mean())
    rng = np.random.RandomState(seed)
    n = len(diff)
    bs = np.array([diff[rng.randint(0, n, n)].mean() for _ in range(boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return g, g - widen * (g - lo), g + widen * (hi - g)


def tier_dist(tiers):
    t = np.asarray(tiers).astype(str)
    u, c = np.unique(t, return_counts=True)
    return {str(k): float(v) / len(t) for k, v in zip(u, c)}


# ------------------------------------------------------------------ checks

def _band(v, band):
    return bool(np.isfinite(v) and band[0] <= v <= band[1])


def check_split(name, d, T):
    """Every regression check of one split. Returns {check: {value, ..., pass}}."""
    out = {}
    y, wp, p = d["y"].astype(float), d["wp"].astype(float), d["p"].astype(float)
    if name in T["REF_GAIN"]:
        g, lo, hi = gain_ci(wp, p, y, T["BOOT"], T["DESIGN_WIDEN"])
        ref = T["REF_GAIN"][name]
        out[f"gain_{name}"] = {"value": g, "ci": [lo, hi], "ref": ref, "pass": bool(lo <= ref <= hi)}
    if name in T["REF_OFFSET_GAIN"] and "p_offset" in d:
        g, lo, hi = gain_ci(wp, d["p_offset"].astype(float), y, T["BOOT"], T["DESIGN_WIDEN"], seed=1)
        ref = T["REF_OFFSET_GAIN"][name]
        out[f"offset_gain_{name}"] = {"value": g, "ci": [lo, hi], "ref": ref, "pass": bool(lo <= ref <= hi)}
    slope = float(fit_logistic(logit(p)[:, None], y)[1])
    out[f"cal_slope_{name}"] = {"value": slope, "range": list(T["CAL_SLOPE_BAND"]),
                                "pass": _band(slope, T["CAL_SLOPE_BAND"])}
    e = ece(p, y)
    out[f"ece_{name}"] = {"value": e, "max": T["ECE_MAX"], "pass": bool(e < T["ECE_MAX"])}
    coef = float(fit_logistic(np.column_stack([logit(wp), d["x_skill"].astype(float)]), y)[2])
    out[f"skill_coef_{name}"] = {"value": coef, "range": list(T["SKILL_COEF_BAND"]),
                                 "pass": _band(coef, T["SKILL_COEF_BAND"])}
    n = d["slot_n"].astype(np.int64)
    pred, real = d["slot_pred"].astype(float), d["slot_real"].astype(float)
    for k, sel in (("n0", n == 0), ("n1_20", (n >= 1) & (n <= 20)), ("n21p", n >= 21)):
        s = ols_slope(pred[sel], real[sel]) if sel.sum() > 2 else float("nan")
        band = T["SLOT_SLOPE"][k]
        out[f"slot_slope_{k}_{name}"] = {"value": s, "range": list(band), "slots": int(sel.sum()),
                                         "pass": _band(s, band)}
    sel = n == 0
    bias = float(100 * np.mean(real[sel] - pred[sel])) if sel.any() else float("nan")
    out[f"neverplayed_bias_{name}"] = {"value_pp": bias, "max_abs_pp": T["NEVERPLAYED_BIAS_PP"],
                                       "pass": bool(np.isfinite(bias) and abs(bias) <= T["NEVERPLAYED_BIAS_PP"])}
    if "slot_state_fetched_max" in d and "slot_start_ts" in d:
        late = d["slot_state_fetched_max"].astype(np.int64) >= d["slot_start_ts"].astype(np.int64)
        out[f"visibility_{name}"] = {"late_rows": int(late.sum()), "pass": not late.any()}
    else:
        out[f"visibility_{name}"] = {"value": "missing slot_state_fetched_max / slot_start_ts",
                                     "pass": False}
    return out


def check_visibility_state(state_fetched_max, game_start_ts, as_of=None):
    """Standalone form of the visibility assertion for a state snapshot used to
    predict games starting at game_start_ts: every row's latest source
    fetched_at, and the snapshot's as_of, precede the start."""
    late = np.asarray(state_fetched_max, np.int64) >= np.asarray(game_start_ts, np.int64)
    ok = not late.any() and (as_of is None or np.all(as_of <= np.asarray(game_start_ts)))
    return {"late_rows": int(late.sum()), "pass": bool(ok)}


def check_tiers(golden, labels_path=None):
    out = {}
    ok = True
    for s, d in golden["splits"].items():
        ref = golden["meta"]["splits"][s].get("tier_dist")
        if ref is None or "tier" not in d:
            out[s] = {"value": "no tier labels in golden"}
            ok = False
            continue
        got = tier_dist(d["tier"])
        diff = max(abs(got.get(k, 0) - ref.get(k, 0)) for k in set(got) | set(ref))
        out[s] = {"dist": got, "max_abs_diff": diff}
        ok &= diff <= TIER_TOL
        if labels_path:
            z = np.load(labels_path, allow_pickle=False)
            rid, lab = z["replay_id"], z["tier"].astype(str)
            o = np.argsort(rid)
            pos = np.searchsorted(rid[o], d["replay_id"])
            hit = (pos < len(rid)) & (rid[o][np.minimum(pos, len(rid) - 1)] == d["replay_id"])
            cur = lab[o][np.minimum(pos, len(rid) - 1)]
            mism = int((cur[hit] != d["tier"].astype(str)[hit]).sum())
            out[s].update({"fresh_labels_matched": int(hit.sum()), "fresh_label_mismatches": mism})
            ok &= mism == 0
    out["pass"] = bool(ok)
    return out


def check_hero_set(hero_names):
    try:
        if P3_DIR not in sys.path:
            sys.path.insert(0, P3_DIR)
        from p3_heroes import check_heroes, NUM_HEROES
        check_heroes(hero_names)
        return {"heroes": int(len(hero_names)), "expected": NUM_HEROES, "pass": True}
    except (AssertionError, SystemExit) as e:
        return {"heroes": int(len(hero_names)), "error": str(e), "pass": False}


def deploy_gates(oot, T):
    """Latest OOT month gain and refit skill coefficient (deployment band)."""
    months = np.asarray(oot["start_ts"].astype(np.int64), "datetime64[s]").astype("datetime64[M]")
    last = str(months.max())
    s = months == months.max()
    y = oot["y"][s].astype(float)
    g = float(np.mean(ll_vec(oot["wp"][s].astype(float), y) - ll_vec(oot["p"][s].astype(float), y)))
    coef = float(fit_logistic(np.column_stack([logit(oot["wp"].astype(float)),
                                               oot["x_skill"].astype(float)]), oot["y"].astype(float))[2])
    return {"deploy_month_gain": {"month": last, "games": int(s.sum()), "value": g,
                                  "min": T["DEPLOY_MIN_MONTH_GAIN"], "pass": bool(g >= T["DEPLOY_MIN_MONTH_GAIN"])},
            "deploy_skill_coef": {"value": coef, "range": list(T["DEPLOY_SKILL_COEF_BAND"]),
                                  "pass": _band(coef, T["DEPLOY_SKILL_COEF_BAND"])}}


def run(golden_path, labels_path=None, overrides=None):
    """Full battery. Returns the report dict; report['pass'] is the verdict."""
    T = thresholds()
    for k, v in (overrides or {}).items():
        if k not in T:
            raise KeyError(f"unknown threshold {k!r}")
        T[k] = v
    g = load_golden(golden_path)
    checks = {}
    for s in SPLITS:
        if s not in g["splits"]:
            checks[f"split_{s}"] = {"value": "missing", "pass": False}
            continue
        checks.update(check_split(s, g["splits"][s], T))
    checks["tiers"] = check_tiers(g, labels_path)
    checks["hero_set"] = check_hero_set(g["hero_names"])
    deploy = deploy_gates(g["splits"]["oot"], T) if "oot" in g["splits"] else \
        {"deploy_month_gain": {"pass": False, "value": "no oot split"}}
    reg_ok = all(v["pass"] for v in checks.values())
    dep_ok = all(v["pass"] for v in deploy.values())
    return {"golden": os.path.abspath(golden_path), "golden_sha256": g["meta"]["sha256"],
            "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "thresholds": T, "checks": checks, "deploy": deploy,
            "regression_pass": bool(reg_ok), "deploy_pass": bool(dep_ok), "pass": bool(reg_ok and dep_ok),
            "failed": sorted([k for k, v in checks.items() if not v["pass"]]
                             + [k for k, v in deploy.items() if not v["pass"]])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", help="fresh tier labels npz (replay_id, tier) to compare row by row")
    ap.add_argument("--thresholds", help="JSON file of threshold overrides")
    a = ap.parse_args(argv)
    ov = json.load(open(a.thresholds)) if a.thresholds else None
    rep = run(a.golden, a.labels, ov)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"gates: {'PASS' if rep['pass'] else 'FAIL'}"
          + (f" (failed: {', '.join(rep['failed'])})" if rep["failed"] else ""))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
