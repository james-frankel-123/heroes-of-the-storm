"""
Safe inverse-CDF sampling in the production / paper-1 kernel (2026-11-01).

The legacy search (search_mode 0) sampled with `cum += p[i]; if (cum > r) ...`
starting from hero 0; when float rounding left r above the accumulated mass
(curand_uniform can return 1.0) the draw fell through and played hero 0, taken
or not. The fix (mcts_kernel.cu, p1_sample/p1_draw) keeps the same arithmetic,
skips non-positive entries, returns the last positive entry on a boundary draw
and uses an explicit fallback (first free hero) on an empty support. The v2
search (chance / rollfwd) used the safe v2_sample since 1db7df8; it only gained
the explicit empty-support fallback.

This script, on the same configs and seeds:
  incidence  the audit build (-DP1_SAMPLE_AUDIT) counts, per draw site, the
             draws, the draws where the legacy loop fell through, and of those
             how often hero 0 was already taken (a duplicate / illegal pick) or
             free but outside the support; the OLD build's finished drafts are
             checked for malformed teams
  identity   OLD (b995800) vs NEW builds: every episode output byte-identical
             in chance mode, and in legacy mode except the episodes whose
             draws fell through (counted from the audit build episode by episode)

Usage (from training/, a GPU host, HOTS_HERO_SET=v2):
  python cuda_mcts/test_sampling_safe.py --old SO_DIR --new SO_DIR --audit SO_DIR \
      [--run-dir production_refresh/2026-11-01] [--episodes 4096] [--sims 400] [--out JSON]
Build the three .so with cuda_mcts/build_variant.py (--rev b995800 for OLD, --audit).
Exit 1 if NEW differs from OLD on any episode without a fall-through, or NEW
produces a malformed draft.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TRAINING_DIR)
import numpy as np

SITES = ["chance-node opponent", "rollout", "executed opponent", "executed own action"]


def load_so(d, name):
    so = [f for f in os.listdir(d) if f.startswith("cuda_mcts_kernel.") and f.endswith(".so")]
    spec = importlib.util.spec_from_file_location("cuda_mcts_kernel", os.path.join(d, so[0]))
    k = importlib.util.module_from_spec(spec)
    sys.modules[f"cuda_mcts_kernel_{name}"] = k
    spec.loader.exec_module(k)
    return k


def weights(run_dir):
    import torch
    os.environ.setdefault("WP_STATS_PATH", os.path.join(run_dir, "stats_decayed90.json"))
    os.environ.setdefault("WP_COMPOSITIONS_PATH", os.path.join(run_dir, "compositions.json"))
    from sweep_enriched_wp import StatsCache, WinProbEnrichedModel, INPUT_DIM_BASE, FEATURE_GROUP_DIMS
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from train_generic_draft import GenericDraftModel
    from train_draft_policy import AlphaZeroDraftNet
    from extract_weights import (extract_policy_weights, extract_gd_weights, extract_wp_weights,
                                 build_wp_net_offsets, extract_lookup_tables)
    wp_dim = INPUT_DIM_BASE + sum(FEATURE_GROUP_DIMS[g] for g in ENRICHED_GROUPS)
    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(os.path.join(run_dir, "generic_draft_0.pt"), weights_only=True, map_location="cpu"))
    gd.eval()
    wp = WinProbEnrichedModel(wp_dim, [256, 128], dropout=0.3)
    wp.load_state_dict(torch.load(os.path.join(run_dir, "wp_enriched_256.pt"), weights_only=True, map_location="cpu"))
    wp.eval()
    net = AlphaZeroDraftNet(size="base", policy_head_type="linear")
    pol = os.path.join(run_dir, "mcts_s0", "draft_policy.pt")
    if os.path.exists(pol):
        net.load_state_dict(torch.load(pol, weights_only=True, map_location="cpu"))
    net.eval()
    wp_flat, wp_names = extract_wp_weights(wp)
    gd_flat, gd_off = extract_gd_weights(gd)
    pol_flat, pol_off = extract_policy_weights(net)
    return (pol_flat, gd_flat, wp_flat, pol_off, gd_off, build_wp_net_offsets(wp, wp_names, wp_dim),
            extract_lookup_tables(StatsCache(), step_embed_weights=None))


def episode_bytes(ep):
    """Canonical bytes of one episode's outputs (win prob, examples, terminal state, team)."""
    kwp, ex, ts, our = ep
    parts = [np.float32(kwp).tobytes(), np.asarray(ts, np.float32).tobytes(), np.int32(our).tobytes()]
    for e in ex:
        for x in e:
            parts.append(np.asarray(x).tobytes())
    return b"".join(parts)


def malformed(ts, H):
    ts = np.asarray(ts)
    t0, t1 = ts[:H] > 0.5, ts[H:2 * H] > 0.5
    return int(t0.sum() != 5 or t1.sum() != 5 or (t0 & t1).any())


def main():
    from shared import NUM_HEROES, NUM_MAPS, NUM_TIERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--run-dir", default=os.path.join(TRAINING_DIR, "production_refresh", "2026-11-01"))
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out")
    a = ap.parse_args()
    W = weights(a.run_dir)
    ks = {n: load_so(d, n) for n, d in (("old", a.old), ("new", a.new), ("audit", a.audit))}
    eng = {n: k.MCTSKernelEngine(*W, max_concurrent=a.batch, device_id=0) for n, k in ks.items()}
    cfg = np.array([[i % NUM_MAPS, (i // NUM_MAPS) % NUM_TIERS, i % 2] for i in range(a.episodes)], np.int32)
    res = {"episodes": a.episodes, "sims": a.sims, "run_dir": a.run_dir, "num_heroes": NUM_HEROES}
    ok = True
    for mode, mname in ((0, "legacy"), (1, "chance")):
        tot = np.zeros(16, np.int64)
        diff_clean, fell_batches, bad = 0, 0, {"old": 0, "new": 0}
        ks["audit"].p1_audit(True)
        for b in range(0, a.episodes, a.batch):
            c = cfg[b:b + a.batch]
            seed = a.seed + b
            o = eng["old"].run_episodes(c, a.sims, 2.0, seed, search_mode=mode)
            n = eng["new"].run_episodes(c, a.sims, 2.0, seed, search_mode=mode)
            # the audit build (NEW plus counters) on the same batch: the counters say whether any
            # draw of the batch fell through; outputs must equal NEW's
            ks["audit"].p1_audit(True)
            au = eng["audit"].run_episodes(c, a.sims, 2.0, seed, search_mode=mode)
            cnt = np.array(ks["audit"].p1_audit(True), np.int64)
            tot += cnt
            for i in range(len(c)):
                bo, bn, ba = episode_bytes(o[i]), episode_bytes(n[i]), episode_bytes(au[i])
                if ba != bn:
                    ok = False
                    res.setdefault("audit_build_differs_from_new", 0)
                    res["audit_build_differs_from_new"] += 1
                if bo != bn:
                    diff_clean += 1
                bad["old"] += malformed(o[i][2], NUM_HEROES)
                bad["new"] += malformed(n[i][2], NUM_HEROES)
            fell = int(cnt.reshape(4, 4)[:, 1].sum())
            fell_batches += fell > 0
            if fell == 0 and any(episode_bytes(o[i]) != episode_bytes(n[i]) for i in range(len(c))):
                ok = False          # a difference in a batch with no fall-through draw at all
                res.setdefault("batches_differing_without_fallthrough", []).append(b // a.batch)
        T = tot.reshape(4, 4)
        res[mname] = {
            "sites": {SITES[s]: {"draws": int(T[s, 0]), "fell_through": int(T[s, 1]),
                                 "hero0_taken": int(T[s, 2]), "hero0_free_outside_support": int(T[s, 3]),
                                 "rate_per_draw": (float(T[s, 1]) / T[s, 0]) if T[s, 0] else None}
                      for s in range(4)},
            "episodes_differing_old_vs_new": diff_clean,
            "fell_through_draws": int(T[:, 1].sum()), "batches_with_fall_through": fell_batches,
            "malformed_old": bad["old"], "malformed_new": bad["new"]}
        if bad["new"]:
            ok = False
        if mode == 1 and diff_clean:
            ok = False          # the v2 path changed only on an empty support, which never happens
        print(mname, json.dumps(res[mname], indent=1), flush=True)
    res["pass"] = bool(ok)
    print(json.dumps({"pass": res["pass"]}))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
