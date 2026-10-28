"""
Paper 2 narrative facts, section 13: is Korea's smaller player gain a sample
problem, and of which kind?

  accounts   games per account in 2025 by region (key >> 40: 1 Americas,
             2 Europe, 3 Korea):
               - full histories: the Heroes Profile match histories of the
                 pulled panel accounts (store full_1017, every row of the
                 account, Heroes Profile and corpus);
               - our corpus: the same accounts' rows that are corpus games;
               - our corpus, all accounts (the slot table).
             Active = 20+ games in 2025 in the full history (panel) or in the
             corpus (all accounts).
  test       Korea's test-game counts and the gain (main model minus draft
             only, accuracy points and log loss x1000) by region with 95%
             bootstrap CIs (2,000 resamples of games within region), and the
             Korea minus Americas / Europe differences.

Usage (from training/): nice -n 19 taskset -c 48-51 python3 personalization/p2f_korea.py --arms <dir>
Output: results/p2f/korea.json
"""
import argparse
import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REG = {1: "Americas", 2: "Europe", 3: "Korea"}
DAY = 86400


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    a_ = ap.parse_args()
    out = {}
    z = np.load(os.path.join(HERE, "cache", "full_history", "full_1017", "history.npz"))
    keys, ptr = z["keys"], z["ptr"]
    kidx = np.repeat(np.arange(len(keys)), np.diff(ptr))
    y25 = (z["end_ts"] >= 1735689600) & (z["end_ts"] < 1767225600)
    pulled = np.isin(keys, z["pulled_keys"])
    m = y25 & pulled[kidx]
    full = np.bincount(kidx[m], minlength=len(keys))
    corp = np.bincount(kidx[m], weights=(z["src"][m] & 1) > 0, minlength=len(keys))
    hp = np.bincount(kidx[m], weights=(z["src"][m] & 2) > 0, minlength=len(keys))
    reg = keys >> 40
    acc = {}
    for rv, rn in REG.items():
        sel = pulled & (reg == rv)
        act = sel & (full >= 20)
        acc[rn] = {"panel_accounts": int(sel.sum()), "active_2025": int(act.sum()),
                   "share_panel_active": float(act.sum() / max(sel.sum(), 1)),
                   "median_games_2025_full_history": float(np.median(full[act])),
                   "mean_games_2025_full_history": float(full[act].mean()),
                   "median_games_2025_heroes_profile_rows": float(np.median(hp[act])),
                   "median_games_2025_in_our_corpus": float(np.median(corp[act])),
                   "corpus_share_of_full_history": float(corp[act].sum() / full[act].sum())}
    out["panel_accounts"] = acc
    print(json.dumps(acc, indent=1))
    # all accounts, corpus only
    s = np.load(os.path.join(a_.arms, "x_slots_full.npz"))
    pk = s["player_keys"]
    d0, d1 = int(np.datetime64("2025-01-01", "D").astype(int)), int(np.datetime64("2026-01-01", "D").astype(int))
    mm = (s["day"] >= d0) & (s["day"] < d1)
    cnt = np.bincount(s["pid"][mm], minlength=len(pk))
    allc = {}
    for rv, rn in REG.items():
        sel = (pk >> 40) == rv
        act = sel & (cnt >= 20)
        allc[rn] = {"accounts_with_a_2025_game": int((sel & (cnt > 0)).sum()), "active_20plus": int(act.sum()),
                    "median_corpus_games_2025_active": float(np.median(cnt[act])),
                    "corpus_games_2025": int(cnt[sel].sum()),
                    "games_per_account_with_a_game": float(cnt[sel].sum() / max((sel & (cnt > 0)).sum(), 1))}
    out["corpus_all_accounts"] = allc
    print(json.dumps(allc, indent=1))

    # test-game gain with CIs
    sg = np.load(os.path.join(HERE, "cache", "x_side_games.npz"))
    rid = [sg["replay_ids"]]
    rg = [sg["region"].astype(np.int64)]
    post = json.load(gzip.open(os.path.join(HERE, "cache", "x_post_games.json.gz"), "rt"))
    rid.append(np.array([g["replay_id"] for g in post], np.int64))
    rg.append(np.array([-1 if g["region"] is None else g["region"] for g in post], np.int64))
    rid, rg = np.concatenate(rid), np.concatenate(rg)
    o = np.argsort(rid)
    rid, rg = rid[o], rg[o]
    rng = np.random.default_rng(0)
    test = {}
    for arm in ("full_ms_layer", "A2S"):
        M = np.load(os.path.join(a_.arms, f"{arm}.npz"))
        j = np.minimum(np.searchsorted(rid, M["g_replay_id"]), len(rid) - 1)
        greg = np.where(rid[j] == M["g_replay_id"], rg[j], -1)
        res = {}
        boots = {}
        for wn, wm in (("V2", M["g_win"] == 2), ("OOT", M["g_win"] == 3), ("V2+OOT", M["g_win"] >= 2)):
            res[wn] = {}
            for rv, rn in REG.items():
                m = wm & (greg == rv)
                y = M["g_y"][m]
                da = ((M["g_p"][m] > .5) == (y == 1)).astype(float) - ((M["g_p0"][m] > .5) == (y == 1)).astype(float)
                dl = 1000 * (ll(M["g_p0"][m], y) - ll(M["g_p"][m], y))
                idx = rng.integers(0, len(da), (2000, len(da)))
                ba, bl = 100 * da[idx].mean(1), dl[idx].mean(1)
                boots[(wn, rn)] = (ba, bl)
                res[wn][rn] = {"games": int(m.sum()), "gain_pts": float(100 * da.mean()),
                               "gain_ci": [float(x) for x in np.percentile(ba, [2.5, 97.5])],
                               "ll_gain": float(dl.mean()), "ll_ci": [float(x) for x in np.percentile(bl, [2.5, 97.5])]}
            for other in ("Americas", "Europe"):
                ka, kl = boots[(wn, "Korea")]
                oa, ol = boots[(wn, other)]
                res[wn][f"Korea minus {other}"] = {
                    "gain_pts": float(ka.mean() - oa.mean()), "gain_ci": [float(x) for x in np.percentile(ka - oa, [2.5, 97.5])],
                    "ll": float(kl.mean() - ol.mean()), "ll_ci": [float(x) for x in np.percentile(kl - ol, [2.5, 97.5])]}
        test[arm] = res
    out["test"] = test
    print(json.dumps(test["full_ms_layer"], indent=1))
    json.dump(out, open(os.path.join(HERE, "results", "p2f", "korea.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
