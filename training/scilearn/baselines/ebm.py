"""E: EBM/GA2M LLM-free control (FRAMING 10.2 E). Runs on a worker in ~/hots/scilearn/venv-ebm.

  python -m training.scilearn.baselines.ebm <job_dir>/spec.json

Features are team differences (hero indicators +1/-1, role counts, kit sums, player-hero
experience) plus map and tier. Trained on swap-augmented rows; predictions are
antisymmetrized, eta = (f(d) - f(-d)) / 2, so swapping teams negates eta exactly.
Configuration frozen here before any dev query: interactions=30, outer_bags=4,
fit on the most recent `max_rows` games of the fit tranches.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from training.scilearn import evaluator as E
from training.scilearn import features as FE
from training.scilearn import terms as TM
from training.scilearn import tranches as T

CONFIG = {"interactions": 30, "outer_bags": 4, "max_rows": 1_000_000, "random_state": 0}


def design(ctx, exp, kit):
    H = ctx.H
    n = ctx.n
    hd = np.zeros((n, H), np.int8)
    np.add.at(hd, (np.repeat(np.arange(n), 5), ctx.heroes[:, :5].ravel()), 1)
    np.add.at(hd, (np.repeat(np.arange(n), 5), ctx.heroes[:, 5:].ravel()), -1)
    cols = {f"h_{h}": hd[:, i] for i, h in enumerate(ctx.meta["heroes"])}
    rc = ctx.role_counts(0) - ctx.role_counts(1)
    for j, r in enumerate(ctx.meta["roles"]):
        cols[f"rc_{r}"] = rc[:, j]
    if kit is not None:
        ks = ctx.kit_sums(0) - ctx.kit_sums(1)
        for j, a in enumerate(ctx.axes):
            cols[f"kit_{a}"] = ks[:, j]
    cols["exp"] = exp[:, :5].sum(1) - exp[:, 5:].sum(1)
    df = pd.DataFrame(cols)
    df["map"] = pd.Categorical(np.array(ctx.meta["maps"])[ctx.map], categories=ctx.meta["maps"])
    df["tier"] = pd.Categorical(np.array(ctx.meta["tiers"])[ctx.tier], categories=ctx.meta["tiers"])
    return df


def flip(df):
    out = df.copy()
    for c in out.columns:
        if c not in ("map", "tier"):
            out[c] = -out[c]
    return out


def main(spec_path):
    spec_path = Path(spec_path)
    job_dir = spec_path.parent
    spec = json.loads(spec_path.read_text())
    out = job_dir / "out"; out.mkdir(exist_ok=True)
    t0 = time.time()
    d = np.load(job_dir.parent.parent / "data" / spec["data"])
    meta = json.loads((job_dir.parent.parent / "data" / spec["meta"]).read_text())
    games = {k: d[k] for k in ("replay_id", "game_ts", "first_seen_ts", "build", "map", "tier", "heroes", "tranche")}
    kit = d["kit"] if "kit" in d.files else None
    full = TM.Ctx(games, meta, kit=kit, players={"player_key": d["player_key"], "party": d["party"]})
    y = d["y0"].astype(int)
    exp = FE.player_hero_experience(full, 180.0)
    trc = games["tranche"]
    fit_idx = np.flatnonzero(np.isin(trc, [T.CODES[t] for t in spec["fit_tranches"]]))[-CONFIG["max_rows"]:]
    np.save(out / "read_ids.npy", games["replay_id"][fit_idx])
    X = design(full.subset(fit_idx), exp[fit_idx], kit)
    Xa = pd.concat([X, flip(X)], ignore_index=True)
    ya = np.concatenate([y[fit_idx], 1 - y[fit_idx]])
    ebm = ExplainableBoostingClassifier(interactions=CONFIG["interactions"], outer_bags=CONFIG["outer_bags"],
                                        n_jobs=int(spec.get("threads", 12)), random_state=CONFIG["random_state"])
    ebm.fit(Xa, ya)
    result = {"job_id": spec["job_id"], "config": CONFIG, "n_fit": int(len(fit_idx)), "eval": {},
              "interactions": [str(t) for t in ebm.term_names_ if " & " in str(t)]}
    for tr in spec["eval_tranches"]:
        ev = np.flatnonzero(trc == T.CODES[tr])
        E.check_disjoint(games["replay_id"][ev], games["replay_id"][fit_idx])
        Xe = design(full.subset(ev), exp[ev], kit)
        eta = 0.5 * (ebm.decision_function(Xe) - ebm.decision_function(flip(Xe)))
        np.savez(out / f"eta_E_{tr}.npz", replay_id=games["replay_id"][ev], eta=eta)
        result["eval"][tr] = E.metrics(eta, y[ev].astype(float))
        print(tr, result["eval"][tr], flush=True)
    result["seconds"] = round(time.time() - t0, 1)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print("DONE", result["seconds"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
