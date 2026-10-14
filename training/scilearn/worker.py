"""Remote job entry point (runs on the workers, never in a proposer sandbox).

  python -m training.scilearn.worker <job_dir>/spec.json

spec: {"job_id", "kind": "fit_eval" | "q2", "data": "<extract>.npz", "meta": "<extract>.json",
       "fit_tranches": [...], "eval_tranches": [...], "models": {"name": [term specs]},
       "incumbent": "name", "candidate": "name", "recent_s_builds": [...], "with_players": bool}
Writes <job_dir>/out/: result.json, <model>.npz/.json, eta_<model>_<tranche>.npz, read_ids.npy.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from training.scilearn import evaluator as E
from training.scilearn import fit as F
from training.scilearn import terms as TM
from training.scilearn import tranches as T


def load_extract(spec, job_dir):
    d = np.load(job_dir.parent.parent / "data" / spec["data"])
    meta = json.loads((job_dir.parent.parent / "data" / spec["meta"]).read_text())
    games = {k: d[k] for k in ("replay_id", "game_ts", "first_seen_ts", "build", "map", "tier",
                               "heroes", "tranche")}
    y = d["y0"].astype(np.float64)
    players = {"player_key": d["player_key"], "party": d["party"]} if "player_key" in d.files else None
    kit = d["kit"] if "kit" in d.files else None
    if kit is not None:
        meta["kit_axes"] = meta["kit_axes"]
    return games, meta, y, players, kit


def main(spec_path):
    spec_path = Path(spec_path)
    job_dir = spec_path.parent
    spec = json.loads(spec_path.read_text())
    out = job_dir / "out"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    games, meta, y, players, kit = load_extract(spec, job_dir)
    full = TM.Ctx(games, meta, kit=kit, players=players)
    trc = games["tranche"]
    fit_m = np.isin(trc, [T.CODES[t] for t in spec["fit_tranches"]])
    if spec.get("fit_max_rows"):
        idx = np.flatnonzero(fit_m)[-int(spec["fit_max_rows"]):]
        fit_m = np.zeros_like(fit_m); fit_m[idx] = True
    fit_idx = np.flatnonzero(fit_m)
    np.save(out / "read_ids.npy", games["replay_id"][fit_idx])
    result = {"job_id": spec["job_id"], "n_fit": int(len(fit_idx)), "models": {}}

    # Features that need history (decay) are computed on the full context, then subset.
    fitted = {}
    for name, specs in spec["models"].items():
        ctx_fit = full.subset(fit_idx)
        m = F.fit(specs, ctx_fit, y[fit_idx], verbose=True)
        m.save(out / name)
        fitted[name] = m
        result["models"][name] = {"p": m.fit_info["p"], "seconds": m.fit_info["seconds"],
                                  "model_hash": m.model_hash(), "lam": m.lam.tolist(), "eval": {}}
        for tr in spec.get("eval_tranches", []):
            ev = np.flatnonzero(trc == T.CODES[tr])
            E.check_disjoint(games["replay_id"][ev], games["replay_id"][fit_idx])
            eta = m.logit(full.subset(ev))
            np.savez(out / f"eta_{name}_{tr}.npz", replay_id=games["replay_id"][ev], eta=eta)
            result["models"][name]["eval"][tr] = E.metrics(eta, y[ev])
        print(name, json.dumps(result["models"][name], default=float), flush=True)

    if spec["kind"] == "q2":
        # Screening fit: the card's term is fit on the fit tranches with the frozen
        # incumbent's logit as offset; the gain is scored on the session's dev tranche.
        inc_dir = job_dir.parent / spec["incumbent_job"] / "out"
        inc = F.Model.load(inc_dir / spec["incumbent"], meta)
        ctx_fit = full.subset(fit_idx)
        off_fit = inc.logit(ctx_fit)
        term = spec["term"]
        cand = F.fit([term], ctx_fit, y[fit_idx], offset=off_fit)
        cand.save(out / "term")
        dev = spec["dev_tranche"]
        ev = np.flatnonzero(trc == T.CODES[dev])
        E.check_disjoint(games["replay_id"][ev], games["replay_id"][fit_idx])
        ctx_dev = full.subset(ev)
        e0 = inc.logit(ctx_dev)
        e1 = e0 + cand.logit(ctx_dev)
        pk = players["player_key"][ev] if players is not None else np.full((len(ev), 10), -1)
        blocks = E.clusters(games["game_ts"][ev], pk)
        result["paired"] = E.paired_gain(e0, e1, y[ev], blocks)
        np.savez(out / "dev_delta.npz", replay_id=games["replay_id"][ev], e0=e0, e1=e1)
        rb = spec.get("recent_s_builds", [])
        s_idx = np.flatnonzero((trc == T.CODES["S"]) & np.isin(games["build"], rb))
        ctx_s = full.subset(s_idx)
        score = np.asarray(cand.logit(ctx_s))
        result["invariance"] = E.invariance(score, inc.logit(ctx_s), y[s_idx], games["build"][s_idx], rb)
        result["term_coefs"] = {n: float(b) for n, b in list(zip(cand.names, cand.beta))[:200]}
        result["term_k"] = int(cand.fit_info["p"])
        result["frac_games_touched"] = float((np.abs(cand.logit(ctx_dev)) > 1e-9).mean())
    if spec["kind"] == "explain":
        result.update(run_explain(spec, job_dir, games, meta, y, full, kit))
    if spec["kind"] == "mtest":
        result.update(run_mtest(spec, job_dir, meta, kit))
    result["seconds"] = round(time.time() - t0, 1)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print("DONE", result["seconds"], flush=True)


def run_explain(spec, job_dir, games, meta, y, full, kit):
    """Explanation card: (a) cross-fit share and (d) permutations on the confirm table (fit on
    Dpre+G), (b) leave-heroes-out on the O3 heroes, (c) onset on the Dpost O2 slices."""
    from training.scilearn import explain as X
    jobs = job_dir.parent
    inc = F.Model.load(jobs / spec["inc_job"] / "out" / "INC", meta)
    conf = F.Model.load(jobs / spec["confirm_job"] / "out" / "INC", meta)
    name, theory = spec["table_name"], spec["theory"]
    axes = list(meta["kit_axes"])
    hero_roles = np.asarray(meta["hero_roles"])
    if spec.get("kit_perm_key"):      # null canary: the theory sees a within-role permuted kit
        kit = TM.permuted_kit(kit, hero_roles, spec["kit_perm_key"])
    H = len(meta["heroes"])
    cells_conf = X.table_cells(conf, name, meta)
    cells_inc = X.table_cells(inc, name, meta)
    trc = games["tranche"]
    s_rows = trc == T.CODES["S"]
    counts = np.bincount(games["heroes"][s_rows].ravel().astype(np.int64), minlength=H)
    o3 = X.o3_heroes(meta["heroes"], counts, spec["o3_key"])
    lanes2 = set(spec.get("lanes2") or TM.LANES2)
    res = X.score_explain(cells_conf, theory, kit, axes, hero_roles, meta["roles"], H,
                          fold_key=spec["fold_key"], perm_key=spec["perm_key"], o3=o3, lanes2=lanes2,
                          n_perm=int(spec.get("n_perm", X.N_PERM)))
    res.pop("_pred_all", None)
    # (c) onset: first 1K and 10K games of each dev build, incumbent fit before them
    dev = spec["dev_tranche"]
    ev = np.flatnonzero(trc == T.CODES[dev])
    read = np.load(jobs / spec["inc_job"] / "out" / "read_ids.npy")
    E.check_disjoint(games["replay_id"][ev], read)
    b, ts = games["build"][ev], games["game_ts"][ev]
    sl10, sl1 = [], []
    for bb in np.unique(b):
        bi = ev[b == bb]
        bi = bi[np.argsort(games["game_ts"][bi], kind="stable")]
        sl10.append(bi[:10000]); sl1.append(bi[:1000])
    sl10, sl1 = np.concatenate(sl10), np.concatenate(sl1)
    ctx = full.subset(sl10)
    Xd = inc.design(ctx)
    i = [s_.get("name") for s_ in inc.specs].index(name)
    o0, o1 = int(inc.offs[i]), int(inc.offs[i + 1])
    eta0 = Xd @ inc.beta
    prior = X.theory_prior_for(cells_inc, theory, res["coefs"], kit, axes, hero_roles, meta["roles"], H, lanes2=lanes2)
    blocks = E.clusters(games["game_ts"][sl10], np.full((len(sl10), 10), -1))
    on = X.onset_gain(Xd[:, o0:o1], inc.beta[o0:o1], cells_inc, prior, eta0, y[sl10], blocks)
    m1 = np.isin(sl10, sl1)
    on1 = X.onset_gain(Xd[m1][:, o0:o1], inc.beta[o0:o1], cells_inc, prior, eta0[m1], y[sl10][m1], blocks[m1])
    res["onset"] = {**on, "n_builds": int(len(np.unique(b))), "first1k": on1}
    res["cells"] = {"confirm": int(len(cells_conf["y"])), "incumbent": int(len(cells_inc["y"])), "o3": o3}
    return {"explain_result": res}


def run_mtest(spec, job_dir, meta, kit):
    """Mechanism test on M-test: does the card's fitted term predict its declared mediator
    (team-difference, draft inputs only), and not its negative control?"""
    from training.scilearn import replay_features as RF
    d = np.load(job_dir.parent.parent / "data" / spec["m_data"])
    f = {k: d[k] for k in d.files}
    test = f["side"] == RF.SIDES["M-test"]
    med = f["med"]
    if spec.get("shuffle_key"):
        perm = RF.shuffle_index(f["map"], f["tier"], f["build"], spec["shuffle_key"])
        med = med[perm]
    idx = np.flatnonzero(test)
    g = {"replay_id": f["replay_id"][idx], "game_ts": f["game_ts"][idx], "first_seen_ts": f["game_ts"][idx],
         "build": f["build"][idx], "map": f["map"][idx], "tier": f["tier"][idx], "heroes": f["heroes"][idx]}
    ctx = TM.Ctx(g, meta, kit=kit, players=None)
    src = spec["term_source"]
    model = F.Model.load(job_dir.parent / src["job"] / "out" / src["model"], meta)
    ctx.max_fit_build = model.fit_info["max_fit_build"]
    names = [t.get("name") for t in model.specs]
    i = names.index(src["term_name"]) if src.get("term_name") else 0
    blk = TM.compile_term(model.specs[i], ctx, model.states[i])
    s = np.asarray(blk.X @ model.beta[int(model.offs[i]):int(model.offs[i + 1])]).ravel()
    out = {"n_m_test": int(len(idx)), "term_sd": float(s.std())}
    day = (g["game_ts"] // 86400).astype(np.int64)
    for role, mname in (("mediator", spec["mediator"]), ("negative_control", spec.get("negative_control"))):
        if not mname:
            continue
        j = RF.MED_NAMES.index(mname)
        kind = RF.MEDIATORS[mname][0]
        m = med[idx, 0, j] if kind == "lead" else med[idx, 0, j] - med[idx, 1, j]
        ok = ~np.isnan(m) & (np.abs(s) > 0)
        ss, mm, dd = s[ok], m[ok], day[ok]
        if len(ss) < 50 or (ss ** 2).sum() == 0:
            out[role] = {"mediator": mname, "n": int(len(ss)), "z": 0.0, "note": "too few games"}
            continue
        alpha = float((ss * mm).sum() / (ss ** 2).sum())
        e = mm - alpha * ss
        _, bl = np.unique(dd, return_inverse=True)
        u = np.bincount(bl, weights=ss * e)
        se = float(np.sqrt((u ** 2).sum()) / (ss ** 2).sum())
        out[role] = {"mediator": mname, "n": int(len(ss)), "alpha": alpha, "se": se,
                     "z": alpha / se if se > 0 else 0.0,
                     "std_effect": float(alpha * ss.std() / mm.std()) if mm.std() > 0 else 0.0}
    return {"mtest_result": out}


if __name__ == "__main__":
    main(sys.argv[1])
