"""
Check a personalization-lane answer file against its walk-query file before a replay run:
echoed query fields, and n against the merged counts of every row before the as-of day
(the replay guard, run standalone). Also compares with the old-model judge state when present.
Usage (worker, run_rcf_full-style env pointing at the same inputs as the queries):
  python personalization/p3t_check_answers.py <query dir> <answer dir> <name>
"""
import sys, os, numpy as np
sys.path.insert(0, "personalization"); os.environ.setdefault("P1_TIERS", "site")
import p3_rcf as R, p3_x_common as X, p3t_data as TD
QD, AD, name = sys.argv[1], sys.argv[2], sys.argv[3]
Q = np.load(f"{QD}/{name}.npz"); A = np.load(f"{AD}/{name}.npz")
print("echo mismatches:", [k for k in Q.files if not np.array_equal(Q[k], A[k])])
d, aug = R.load_aug("merged")
keys = np.asarray(d["player_keys"]); kpos = {int(k): i for i, k in enumerate(keys)}
q_pid = np.array([kpos[int(k)] for k in Q["player_key"]], np.int64)
q_day = Q["as_of_day"].astype(np.int64) + 1
sh = np.asarray(TD.to_shared(d["hero_names"]))
want = TD._merged_counts(aug, q_pid, q_day)            # d order
n_d = A["n"][:, sh]                                     # answer is shared order; d hero i = shared sh[i]
echo_bad = [k for k in Q.files if not np.array_equal(Q[k], A[k])]
ok = np.array_equal(n_d, want) and not echo_bad
print("merged-count guard:", "PASS" if np.array_equal(n_d, want) else f"FAIL ({float((n_d == want).mean()):.4f} of cells agree)")
if not ok:
    sys.exit(1)
# reference: the old-model state's n for the same players (state_judge, shared order -> d order)
L = TD.load_lobbies()
sj = os.path.join(TD.OUT, "state_judge.npz")
if not (name.endswith("states__1") and os.path.exists(sj)):
    sys.exit(0)
z = np.load(sj)
rows = L["slot_rows"][z["gsel"]].ravel(); pk = keys[d["pid"][rows]].astype(np.int64)
first = {}
for i, k in enumerate(pk): first.setdefault(int(k), i)
idx = np.array([first[int(k)] for k in Q["player_key"]])
old_d = z["n"][idx].astype(np.float32)[:, sh]
print("guard counts == old walk n:", np.array_equal(want, old_d), float((want == old_d).mean()))
print("answer n == old walk n:", float((n_d == old_d).mean()))
dif = n_d != want
r, c = np.nonzero(dif)
if dif.any():
    print("differing cells", len(r), "players", len(np.unique(r)), "answer-want: mean", float((n_d - want)[dif].mean()),
          "min", float((n_d - want)[dif].min()), "max", float((n_d - want)[dif].max()))
else:
    print("differing cells: none")
print("answer total / want total per player (median)", float(np.median(n_d.sum(1))), float(np.median(want.sum(1))))
