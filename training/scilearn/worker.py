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
    result["seconds"] = round(time.time() - t0, 1)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print("DONE", result["seconds"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
