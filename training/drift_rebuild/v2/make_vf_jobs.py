"""Write drift_v2/jobs_vf.txt: every value function of the paper, retrained on
site-tiered data with the unchanged drift2026/train_drift_wp.py protocol
(via r2_train_vf.py under the v2 environment), in priority order."""
import json
import os

T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.join(T, "drift_v2")
builds = [b for b in json.load(open(os.path.join(T, "drift2026", "patch_index.json")))["builds"]
          if b.startswith("2.55")]
CUT, C0, INNER = "2.55.14.95918", "2.55.3.89754", "2.55.13.95301"
S = [42, 123, 777]
REG = {"allhist": "--regime all", "win3": "--regime window --window 3",
       "win6": "--regime window --window 6", "win12": "--regime window --window 12",
       "decay90": "--regime decay --half-life 90", "decay365": "--regime decay --half-life 365",
       "embed": "--regime embed"}
jobs = []


def add(name, args, sub="d2"):
    out = os.path.join(V, "results", sub, f"{name}.json")
    log = os.path.join(V, "logs", f"vf_{name}.log")
    cmd = f"python3 -u drift_rebuild/v2/run.py drift_rebuild/r2_train_vf.py --name {name} {args} --results-subdir {sub}"
    jobs.append(f"{out}\t{log}\t{cmd}")


for s in S:
    add(f"d2c_cumprev_s{s}", f"--features cumulative_prev --regime all --seed {s}")
for s in S:
    for r, a in REG.items():
        add(f"r2_{r}_oof_s{s}", f"--features oof_{CUT} {a} --seed {s}")
for s in S:
    for c in ("decayed90", "decayed90k100", "decayed365"):
        add(f"q7_{c}_s{s}", f"--features {c}_prev --regime all --seed {s}", "q7")
for s in S:
    for r, a in REG.items():
        add(f"d2b_{r}_s{s}", f"--features cutoff {a} --seed {s}")
for s in S:
    add(f"r2_stale1yr_oof_s{s}", f"--features oof_2.55.9.93613 --regime all --cutoff-build 2.55.9.93613 --seed {s}")
    add(f"r2_stale2yr_oof_s{s}", f"--features oof_2.55.4.91418 --regime all --cutoff-build 2.55.4.91418 --seed {s}")
    add(f"r2_c0frozen_oof_s{s}", f"--features oof_{C0} --regime all --cutoff-build {C0} --skip-sanity --seed {s}")
add("w2c_frozenstats_cut08_s42", f"--features cutoff_{C0} --regime all --cutoff-build {C0} --skip-sanity --seed 42", "w2c")
for p in range(8, len(builds) - 1):
    if builds[p] == CUT:
        continue
    add(f"w2c_cut{p:02d}_s42", f"--features cumulative_prev --regime all --cutoff-build {builds[p]} --skip-sanity --seed 42", "w2c")
for s in S:
    for c, f in (("cumulative", "cumulative_prev"), ("decayed90", "decayed90_prev"),
                 ("decayed90k100", "decayed90k100_prev"), ("decayed365", "decayed365_prev")):
        add(f"w8c_{c}_s{s}", f"--features {f} --regime all --cutoff-build {INNER} --skip-sanity --seed {s}", "w8c")
open(os.path.join(V, "jobs_vf.txt"), "w").write("\n".join(jobs) + "\n")
print(len(jobs), "jobs")
