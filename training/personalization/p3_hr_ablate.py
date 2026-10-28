"""
B1 per-metric ablations of the main model (P3_HERO_RECS_AND_REPLAYS).

Each arm refits `full_ms` (rank 2, shared metric loadings, per-hero s2,
predictive objective, a_shrink card) with a subset of the 15 scoreboard
metrics. A metric is removed by zeroing its sums and counts in the
conditioning set (fitm file), so the fit sees no evidence for it and its
loadings are masked to 0 (p3_rcf.fit `avail`). Nothing else changes.

Runs on the 5090 (from ~/hots/repo/training):
  P3_CACHE=<dir with rcf/fit_merged.npz> P3_RESULTS=<out> \
      python personalization/p3_hr_ablate.py run [--only name,name]
Each arm writes <P3_RESULTS>/rcf/params/abl_<name>.{npz,json} (held-out
predictive LL of the held-out half in the json).
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["hero_damage", "siege_damage", "structure_damage", "healing", "self_healing",
           "protection_allies", "damage_taken", "deaths", "kills", "assists",
           "experience_contribution", "merc_camp_captures",
           "outnumbered_deaths", "escapes", "regen_globes"]
GROUPS = {
    "fight": ["hero_damage", "damage_taken", "deaths", "kills", "assists", "outnumbered_deaths", "escapes"],
    "macro": ["siege_damage", "structure_damage", "experience_contribution", "merc_camp_captures", "regen_globes"],
    "sustain": ["healing", "self_healing", "protection_allies"],
}


def arms():
    out = {"all": list(METRICS)}
    for m in METRICS:
        out["drop_" + m] = [x for x in METRICS if x != m]
    for m in METRICS:
        out["only_" + m] = [m]
    for g, ms in GROUPS.items():
        out["only_" + g] = list(ms)
        out["drop_" + g] = [x for x in METRICS if x not in ms]
    out["top3"] = ["hero_damage", "damage_taken", "deaths"]
    return out


def make_card(rcf, name, keep):
    p = os.path.join(rcf, f"fitm_merged_abl_{name}.npz")
    if os.path.exists(p):
        return
    z = np.load(os.path.join(rcf, "fitm_merged_a_shrink.npz"))
    mets = [str(m) for m in np.load(os.path.join(rcf, "fit_merged.npz"))["metrics"]]
    assert mets == METRICS, mets
    X1, C1 = z["X1"].copy(), z["C1"].copy()
    drop = [k for k, m in enumerate(mets) if m not in keep]
    X1[:, :, drop] = 0
    C1[:, :, drop] = 0
    np.savez(p, X1=X1, C1=C1, players=z["players"])
    print(f"  card {name}: kept {len(keep)} metrics", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "list"])
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    A = arms()
    if a.what == "list":
        print("\n".join(A))
        return
    rcf = os.path.join(os.environ["P3_CACHE"], "rcf")
    res = os.path.join(os.environ["P3_RESULTS"], "rcf", "params")
    names = a.only.split(",") if a.only else list(A)
    for nm in names:
        if os.path.exists(os.path.join(res, f"abl_{nm}.json")):
            print(f"skip {nm}", flush=True)
            continue
        make_card(rcf, nm, A[nm])
        cmd = [sys.executable, "-u", os.path.join(HERE, "p3_rcf.py"), "fit", "--tag", f"abl_{nm}", "--hist", "merged",
               "--kmax", "2", "--metrics", "--sb", f"abl_{nm}", "--hero-s2", "--gshare", "--rounds", "3"]
        print("===", nm, " ".join(cmd[2:]), flush=True)
        rc = subprocess.call(cmd)
        if rc == 0:
            os.remove(os.path.join(rcf, f"fitm_merged_abl_{nm}.npz"))
        else:
            print(f"FAILED {nm}", flush=True)


if __name__ == "__main__":
    main()
