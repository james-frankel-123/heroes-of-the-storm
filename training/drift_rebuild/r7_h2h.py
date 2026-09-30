"""
R7 — head-to-head drafting between two agent arms, the W6 protocol of
drift2026/w6_head2head.py (policy heads act greedily, every seed pairing,
both first-pick orderings, 40 drafts per ordering over the same stratified
map x tier configs, same RNG seeding) without the learned-judge consensus
(the paper's headline is judge-free). Terminal drafts are stored with seed
labels for r8_score.py.

Arms are given as <runs_dir>:<prefix>:<n_seeds>, e.g.
  drift2026:w4_d2c_cumprev:5          (the paper's maintained agent)
  rebuild:r6_uoof_rg:6

Usage:
  python3 drift_rebuild/r7_h2h.py --a drift2026:w4_d2c_cumprev:5 \
      --b rebuild:r6_uoof_rg:6 --out h2h_M_vs_uoofrg
Output: drift_rebuild/results/h2h/<out>.json
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402

torch.set_num_threads(4)
from shared import is_degenerate, NUM_HEROES, HEROES, MAPS, SKILL_TIERS  # noqa: E402
from train_draft_policy import AlphaZeroDraftNet, DraftState, DRAFT_ORDER  # noqa: E402

RUNS = {"drift2026": os.path.join(rb.TRAINING_DIR, "drift2026", "mcts_runs"),
        "rebuild": rb.RUNS_DIR}


def load_arm(spec):
    where, prefix, n = spec.split(":")
    seeds = (range(int(n.split("-")[0]), int(n.split("-")[1]) + 1)
             if "-" in n else range(int(n)))
    nets = []
    for s in seeds:
        for pre in prefix.split(","):   # first existing run wins (w4 then w8)
            ckpt = os.path.join(RUNS[where], f"{pre}_s{s}", "draft_policy.pt")
            if os.path.exists(ckpt):
                break
        net = AlphaZeroDraftNet(size="base", policy_head_type="linear")
        net.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
        net.eval()
        nets.append(net)
    return nets


def make_actor(net):
    def act(state):
        x = torch.tensor(state.to_numpy(), dtype=torch.float32).unsqueeze(0)
        m = state.valid_mask(torch.device("cpu"))
        with torch.no_grad():
            logits, _ = net(x, m)
        return logits.argmax(dim=1).item()
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="side scored as 'maintained'")
    ap.add_argument("--b", required=True, help="side scored as 'frozen'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drafts-per-cell", type=int, default=40)
    args = ap.parse_args()
    out = os.path.join(rb.RESULTS_DIR, "h2h", f"{args.out}.json")
    if os.path.exists(out):
        print("exists", out)
        return
    A, B = load_arm(args.a), load_arm(args.b)
    records = []
    for sa in range(len(A)):
        for sb in range(len(B)):
            for order in (0, 1):
                random.seed(7000 + sa * 100 + sb * 10 + order)
                torch.manual_seed(7000 + sa * 100 + sb * 10 + order)
                first = make_actor(A[sa] if order == 0 else B[sb])
                second = make_actor(B[sb] if order == 0 else A[sa])
                for i in range(args.drafts_per_cell):
                    game_map = MAPS[i % len(MAPS)]
                    tier = SKILL_TIERS[(i // len(MAPS)) % len(SKILL_TIERS)]
                    st = DraftState(game_map, tier, our_team=0)
                    while not st.is_terminal():
                        team, typ = DRAFT_ORDER[st.step]
                        st.our_team = team
                        st.apply_action((first if team == 0 else second)(st), team, typ)
                    t0 = [HEROES[j] for j in range(NUM_HEROES) if st.team0_picks[j] > 0]
                    t1 = [HEROES[j] for j in range(NUM_HEROES) if st.team1_picks[j] > 0]
                    mh, fh = (t0, t1) if order == 0 else (t1, t0)
                    records.append({"maintained": mh, "frozen": fh,
                                    "game_map": game_map, "tier": tier,
                                    "sa": sa, "sb": sb, "order": order})
        print(f"a seed {sa} done", flush=True)
    res = {"a": args.a, "b": args.b, "n_seeds": [len(A), len(B)],
           "drafts_per_cell": args.drafts_per_cell, "n_drafts": len(records),
           "degen_a": sum(is_degenerate(r["maintained"]) for r in records) / len(records),
           "degen_b": sum(is_degenerate(r["frozen"]) for r in records) / len(records),
           "records": records}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"wrote {out}: {len(records)} drafts; degen a {res['degen_a']:.3f} "
          f"b {res['degen_b']:.3f}")


if __name__ == "__main__":
    main()
