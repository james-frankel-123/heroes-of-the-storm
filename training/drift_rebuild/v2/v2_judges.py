"""
v2 vintage judges (Sec. 8) on site-tiered data. Training only; drafts are
scored later (the head-to-head drafts come from MCTS agents).

Ranked judges: drift2026/q2_vintage_judges.train_judge, unchanged recipe, on
the v2 (relabeled) snapshot: 2022-07, 2023-07, 2024-07, 2025-07 (date
cutoffs) and 2026-build (through the 2026 cutoff build), each on the same
volume cap; plus 2022Q1-ranked (2021-12-01..2022-03-31, all games, as W8d).

Quick-Match judges: same recipe on qm2026/data/qm_games.jsonl, relabeled
from qm_games.league_tier (read-only DB fetch) with the site rule; QM stores
Master as league_tier 0 (avg_mmr >= ~2820):
    0 -> high, <= 3 -> low, <= 5 -> mid, 6 -> high.
Windows: QM-2021 (before 2022-01-01), QM-2022 (calendar 2022), QM-2024
(calendar 2024), QM-2026 (2025-07-01 to the 2026 cutoff date 2026-02-11).

Output: drift_v2/models/vintage_judges/wp_<name>.pt (+ .meta.json),
        drift_v2/results/v2_judges.json
"""
import datetime
import json
import os

import v2env
import torch

from drift2026 import common
from drift2026.q2_vintage_judges import train_judge, VINTAGES, days

T = v2env.TRAINING


def qm_relabel(lt):
    if lt is None:
        return None
    lt = int(lt)
    return "high" if lt in (0, 6) else ("low" if lt <= 3 else "mid")


def main():
    import psycopg2
    common.setup()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(common.MODELS_DIR, "vintage_judges")
    os.makedirs(out_dir, exist_ok=True)
    res = {}

    def fit(name, rows, cap):
        p = os.path.join(out_dir, f"wp_{name}.pt")
        if os.path.exists(p):
            res[name] = json.load(open(p.replace(".pt", ".meta.json")))
            return
        model, meta = train_judge(name, rows, dev, cap)
        torch.save(model.state_dict(), p)
        json.dump(meta, open(p.replace(".pt", ".meta.json"), "w"), indent=1)
        res[name] = meta

    rows, builds = common.load_data_with_patches()
    cut = common.cutoff_idx(builds)
    pools = {}
    for name, kind, val in VINTAGES:
        pools[name] = ([r for r in rows if r["date_days"] <= val] if kind == "date"
                       else [r for r in rows if r["build_idx"] <= cut])
    cap = min(len(p) for p in pools.values())
    for name in pools:
        fit(name, pools[name], cap)
    q1 = [r for r in rows if days(2021, 12, 1) <= r["date_days"] <= days(2022, 3, 31)]
    fit("2022Q1-ranked", q1, len(q1))
    del rows, pools

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SELECT replay_id, league_tier FROM qm_games")
    lt = {int(r): t for r, t in cur.fetchall()}
    epoch = datetime.date(1970, 1, 1)
    qm, missing = [], 0
    for line in open(os.path.join(T, "qm2026", "data", "qm_games.jsonl")):
        g = json.loads(line)
        t = qm_relabel(lt.get(g["replay_id"]))
        if t is None or not g.get("game_date"):
            missing += 1
            continue
        g["skill_tier"] = t
        g["date_days"] = (datetime.date.fromisoformat(g["game_date"][:10]) - epoch).days
        qm.append(g)
    print(f"QM: {len(qm):,} games relabeled, {missing:,} dropped", flush=True)
    windows = {"QM-2021": ("0000", "2022-01-01"), "QM-2022": ("2022-01-01", "2023-01-01"),
               "QM-2024": ("2024-01-01", "2025-01-01"), "QM-2026": ("2025-07-01", "2026-02-12")}
    for name, (lo, hi) in windows.items():
        pool = [g for g in qm if lo <= g["game_date"][:10] < hi]
        fit(name, pool, len(pool))
    common.write_json(os.path.join(common.RESULTS_DIR, "v2_judges.json"), res)


if __name__ == "__main__":
    main()
