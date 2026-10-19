"""
Paper 2 narrative facts, section 7: why a year-stale player state loses most
of the player gain. Counterfactual swaps on the tournament's saved states
(p3t_data.states: s = experience offset(slot counts) + posterior mean + layer,
for every hero of every lobby slot), scored as p3t_value.predeval with the
current draft model and its combiner.

Per lobby slot (the played hero), stale = state frozen at 2025-06-02 (s1y),
current = state through the day before the game. Categories, by the stale
state's merged history counts (n_all, corpus plus Heroes Profile rows):
  (a) new to us: the player has no games at all in the stale state
  (b) new hero: the player has history, but under 10 games on this hero in
      the stale state and 10 or more by the game
  (c+d) established: everything else
Swaps, applied cumulatively in this order (and in reverse for (b)/(d)):
  1. (a) slots take the current s (and current off-role flag)
  2. (b) slots take the current s
  3. (d) established slots: the experience offset is recomputed at the current
     slot counts (counts only, no outcomes) and the off-role flag updated
  4. (c) established slots: the posterior mean and layer take current values
     (new evidence on cells already known: sharper estimates plus real drift)
Step 4 is also split by the stale count on the hero (10-49, 50-199, 200+):
drift would hurt even well-measured cells, while extra evidence mostly helps
thin ones.
Runs on a worker: P3_CACHE=<cache_full> python personalization/p2f_stale_decomp.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import p3t_data as TD
import p3_hs_core as C
import p3_x_common as X

L = TD.load_lobbies()
g = np.flatnonzero(L["day"] <= TD.LOBBY_END)
P = np.load(os.path.join(TD.OUT, "predeval_wp.npz"))
assert (P["g"] == g).all()
y = P["y"]
comb = json.load(open(os.path.join(HERE, "results", "rcf_full", "p3t", "combiner.json")))
b = comb["cur"]["coef"]
wp = P["wp_cur"]
lo = np.log(np.clip(wp, 1e-6, 1 - 1e-6) / np.clip(1 - wp, 1e-6, 1 - 1e-6))
A2S = np.load(os.path.join(HERE, "results", "oct26_p3c", "p3c", "A2S.npz"))
table = A2S["table"]

st = {v: np.load(os.path.join(TD.OUT, f"state_{v}.npz")) for v in ("cur", "s1y")}
pick = L["steps"][g][:, :, 1] == 1
team = L["steps"][g][:, :, 2][pick].reshape(len(g), 10)
hero = L["steps"][g][:, :, 0][pick].reshape(len(g), 10)
sign = np.where(team == 0, 1.0, -1.0)
base_c = (g[:, None] * 10 + np.arange(10)[None]).ravel()
base_s = (np.arange(len(g))[:, None] * 10 + np.arange(10)[None]).ravel()
hf = hero.ravel()
S_cur = st["cur"]["s"][base_c, hf].astype(np.float64)
S_old = st["s1y"]["s"][base_s, hf].astype(np.float64)
O_cur = st["cur"]["off"][base_c, hf]
O_old = st["s1y"]["off"][base_s, hf]
n_cur_h = st["cur"]["n"][base_c, hf].astype(np.int64)
n_old_h = st["s1y"]["n"][base_s, hf].astype(np.int64)
n_old_p = st["s1y"]["n"][base_s].astype(np.int64).sum(1)
n_cur_p = st["cur"]["n"][base_c].astype(np.int64).sum(1)

# slot-row counts (the offset's inputs) at both dates, for the played hero
d = X.load_ext()
rows = L["slot_rows"][g].ravel()
pid = d["pid"][rows].astype(np.int64)
dh = d["hero"][rows].astype(np.int64)
qday_cur = np.repeat(L["day"][g], 10).astype(np.int64)
qday_old = np.full(len(rows), TD.VINTAGES["s1y"] + 1, np.int64)
H = len(d["hero_names"])


def counts_before(qpid, qhero, qday):
    """slot rows of the player (and on the hero) with day < qday."""
    o = np.lexsort((d["day"], d["pid"]))
    kp, kd = d["pid"][o].astype(np.int64), d["day"][o].astype(np.int64)
    comp = (kp << 20) | kd
    np_ = np.searchsorted(comp, (qpid << 20) | qday) - np.searchsorted(comp, qpid << 20)
    key = d["pid"].astype(np.int64) * H + d["hero"].astype(np.int64)
    o2 = np.lexsort((d["day"], key))
    k2, d2 = key[o2], d["day"][o2].astype(np.int64)
    comp2 = (k2 << 20) | d2
    qk = qpid * H + qhero
    nh = np.searchsorted(comp2, (qk << 20) | qday) - np.searchsorted(comp2, qk << 20)
    return np_, nh


np_c, nh_c = counts_before(pid, dh, qday_cur)
np_o, nh_o = counts_before(pid, dh, qday_old)


def off1(np_, nh):
    nb = len(C.NPH_EDGES)
    bp = np.searchsorted(C.NP_EDGES, np_, side="right") - 1
    bh = np.searchsorted(C.NPH_EDGES, nh, side="right") - 1
    return table[bp * nb + bh]


OFF_cur, OFF_old = off1(np_c, nh_c), off1(np_o, nh_o)
REST_cur, REST_old = S_cur - OFF_cur, S_old - OFF_old


def score(S, O):
    dS = (sign * S.reshape(len(g), 10)).sum(1)
    dO = (sign * O.reshape(len(g), 10)).sum(1)
    p = 1 / (1 + np.exp(-(b[0] + b[1] * lo + b[2] * dS + b[3] * dO)))
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    return {"acc": float(((p > 0.5) == (y == 1)).mean()),
            "ll_gain_vs_wp_x1000": float(1000 * (np.mean(-(y * np.log(np.clip(wp, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - wp, 1e-9, 1))))
                                                 - np.mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)))))}


cat_a = n_old_p == 0
cat_b = (~cat_a) & (n_old_h < 10) & (n_cur_h >= 10)
est = ~(cat_a | cat_b)
out = {"games": int(len(g)), "slots": int(len(rows)),
       "share_slots": {"a new to us": float(cat_a.mean()), "b new hero": float(cat_b.mean()),
                       "established": float(est.mean())},
       "established_by_stale_n": {k: float((est & m).mean()) for k, m in
                                  {"<10 (and still <10)": n_old_h < 10, "10-49": (n_old_h >= 10) & (n_old_h < 50),
                                   "50-199": (n_old_h >= 50) & (n_old_h < 200), "200+": n_old_h >= 200}.items()},
       "check_offset_reconstructs": None}
# sanity: the reconstructed offset should explain the count-driven part; report mean abs of REST change on cells with no new games
same = est & (n_cur_h == n_old_h)
out["established_no_new_games_on_hero_share"] = float(same.mean())

steps = []
S, O = S_old.copy(), O_old.copy()
steps.append(("all stale (current draft model, year-old player state)", score(S, O)))
S[cat_a], O[cat_a] = S_cur[cat_a], O_cur[cat_a]
steps.append(("+ (a) players new to us brought current", score(S, O)))
S[cat_b], O[cat_b] = S_cur[cat_b], O_cur[cat_b]
steps.append(("+ (b) heroes newly picked up brought current", score(S, O)))
S[est] = OFF_cur[est] + REST_old[est]
O[est] = O_cur[est]
steps.append(("+ (d) established cells: game counts (offset, off-role flag) current, no outcomes", score(S, O)))
for lab, m in (("10-49", (n_old_h >= 10) & (n_old_h < 50)), ("<10, still thin", n_old_h < 10),
               ("50-199", (n_old_h >= 50) & (n_old_h < 200)), ("200+", n_old_h >= 200)):
    mm = est & m
    S[mm] = S_cur[mm]
    steps.append((f"+ (c) established cells, stale n {lab}: estimates current", score(S, O)))
steps.append(("all current", score(S_cur, O_cur)))
out["steps_forward"] = [{"step": k, **v} for k, v in steps]

# reverse order: (c) then (d) then (b) then (a)
rev = []
S, O = S_old.copy(), O_old.copy()
rev.append(("all stale", score(S, O)))
S[est] = OFF_old[est] + REST_cur[est]
rev.append(("+ (c) established estimates current, counts stale", score(S, O)))
S[est], O[est] = S_cur[est], O_cur[est]
rev.append(("+ (d) established counts current", score(S, O)))
S[cat_b], O[cat_b] = S_cur[cat_b], O_cur[cat_b]
rev.append(("+ (b) new heroes", score(S, O)))
S[cat_a], O[cat_a] = S_cur[cat_a], O_cur[cat_a]
rev.append(("+ (a) new to us", score(S, O)))
out["steps_reverse"] = [{"step": k, **v} for k, v in rev]

# how far the estimates moved, established cells: corr of stale vs current s (posterior part)
for lab, m in (("10-49", (n_old_h >= 10) & (n_old_h < 50)), ("50-199", (n_old_h >= 50) & (n_old_h < 200)), ("200+", n_old_h >= 200)):
    mm = est & m
    out.setdefault("established_estimate_corr", {})[lab] = {
        "corr_rest": float(np.corrcoef(REST_old[mm], REST_cur[mm])[0, 1]),
        "sd_rest_old_pp": float(100 * REST_old[mm].std()), "sd_rest_cur_pp": float(100 * REST_cur[mm].std()),
        "median_new_games_on_hero": float(np.median(n_cur_h[mm] - n_old_h[mm]))}
out["median_games_new_to_us_players_by_game"] = float(np.median(n_cur_p[cat_a]))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(TD.OUT, "p2f_stale_decomp.json"), "w"), indent=1)
