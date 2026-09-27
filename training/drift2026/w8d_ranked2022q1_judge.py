"""
W8d — ranked early-2022 vintage judge (optional row of the Q2 matrix).

The QM-2021 point of the vintage matrix is mode+era double-confounded (QM
games AND 2021). If the RANKED corpus window 2021-12-01..2022-03-31 holds
>= 150K games, train one more naive-feature judge on exactly that window
(same qm2026 recipe via q2_vintage_judges.train_judge) and score the 2,000
saved W6 head-to-head drafts. Ranked-only, era ~2022Q1: separates the mode
confound from the era confound.

Output: results/w8/W8D_RANKED2022Q1.json, models/vintage_judges/
wp_2022Q1-ranked.pt (picked up automatically by w7_rescore.load_judges for
all subsequent rescoring).

Usage: CUDA_VISIBLE_DEVICES=0 python3 drift2026/w8d_ranked2022q1_judge.py
"""
import os
import sys
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import torch

from drift2026.q2_vintage_judges import (train_judge, score_judge, days,
                                         JUDGES_DIR, W6_PATH)

NAME = "2022Q1-ranked"
MIN_GAMES = 150_000
LO, HI = days(2021, 12, 1), days(2022, 3, 31)


def main():
    os.makedirs(JUDGES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, _ = common.load_data_with_patches()
    window = [r for r in rows if LO <= r["date_days"] <= HI]
    print(f"ranked window 2021-12-01..2022-03-31: {len(window)} games")
    out_path = os.path.join(common.RESULTS_DIR, "w8", "W8D_RANKED2022Q1.json")
    if len(window) < MIN_GAMES:
        common.write_json(out_path, {
            "skipped": True, "n_games": len(window),
            "reason": f"window < {MIN_GAMES} games"})
        print("SKIP: window too small")
        return

    # volume: min(window, the Q2 per-judge cap) for comparability
    q2_cap = json.load(open(os.path.join(
        JUDGES_DIR, "wp_2022-07.meta.json")))["n_games"]
    cap = min(len(window), q2_cap)
    model, meta = train_judge(NAME, window, device, cap)
    torch.save(model.state_dict(), os.path.join(JUDGES_DIR, f"wp_{NAME}.pt"))
    with open(os.path.join(JUDGES_DIR, f"wp_{NAME}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(W6_PATH) as f:
        w6 = json.load(f)
    scores, _ = score_judge(model, device, w6["records"])
    payload = {"skipped": False, "q2_volume_cap": q2_cap,
               "window_games": len(window), **meta,
               "w6_scores": scores,
               "date": datetime.date.today().isoformat()}
    common.write_json(out_path, payload)
    print(f"judge {NAME}: W6 maintained WP {scores['maintained_wp_mean']:.4f} "
          f"± {scores['se']:.4f}, win share {scores['win_share']:.3f}")


if __name__ == "__main__":
    main()
