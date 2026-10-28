"""
P3: integrity check of the pinned population models trained on the 3090 (coordinator, 2026-10-24).

The 3090 had a RAM fault (single-bit flips, found 2026-10-21). A flip during training gives a
different but valid model; a flip in the weights before they were written gives a corrupted one.
The second case is checked here, per artifact, against what was logged at training time:

  weights   every tensor of every 3090-trained .pt: finite, largest |w|, largest |w| / tensor sd,
            and the tensor sd against the same tensor of the sibling seeds trained on the 3080/5090;
            F policies also: draft_policy.pt equals the checkpoint's model_state_dict
  stats V   the vintage's statistics tables (deploy, oof0-4, compositions) rebuilt from the pinned
            snapshot in a temporary directory and compared with the pinned files
  wp V      the vintage's WP seeds: validation loss and the test and NODRIFT metrics recomputed on
            features built from the pinned vintage statistics, against enriched.json
  wpft V    a fine-tuned vintage (s4m_ft, s1y_ft): the fine-tuned WP's val loss, test, NODRIFT,
            val_post and test_post, and the base vintage's selected WP's "before" metrics, against
            enriched.json (current site features)
  gd        the s1y GD (gd_era_2.55.10.94189, v0 and v1): top-1 and top-5 on the post-cutoff
            next-pick samples against drift_v2/results/gd_eval_era_2.55.10.94189.json
            (run through drift_rebuild/v2/run.py)
  feval R   an F policy's self-play evaluation (the worker's EVAL: avg_wp and win rate of its own
            search against the GD, 200 sims) on 512 fixed drafts, for the 3090 runs and their
            3080/5090-trained siblings, next to the logged best eval
Outputs: <out dir>/<stage>[_<arg>].json (default results/p3t/integrity_3090/ under P3_RESULTS).
Usage (worker, from training/):
  python personalization/p3t_integrity_3090.py weights|stats V|wp V|wpft V|feval RUN [--out DIR]
  python drift_rebuild/v2/run.py personalization/p3t_integrity_3090.py gd [--out DIR]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.environ.get("P3_TRAINING") or os.path.dirname(HERE)   # workers: run from outside the repo
SITE = os.path.join(TRAINING, "paper1_revision", "site")
sys.path.insert(0, TRAINING)
import numpy as np

# runs: name -> (P1_VINTAGE or "", config seed, trained on the 3090)
RUNS = {
    "cur_s0": ("", 0, False), "cur_s1": ("", 1, True), "cur_s2": ("", 2, True), "cur_s3": ("", 3, False),
    "cur_s4": ("", 4, False),
    "s4m_s0": ("s4m", 0, True), "s4m_s1": ("s4m", 1, False),
    "s1y_s0": ("s1y", 0, False), "s1y_s1": ("s1y", 1, True),
    "s4m_ft_s0": ("s4m_ft", 0, False), "s1y_ft_s0": ("s1y_ft", 0, True),
}


def run_dir(name):
    v, s, _ = RUNS[name]
    base = SITE if not v else os.path.join(SITE, "vint", v)
    return os.path.join(base, "mcts_runs", f"F_oof_s{s}")


def wp_files():
    """(path, trained on the 3090, sibling group)"""
    out = []
    for v in ("s4m", "s1y"):
        for s in (42, 123, 777):
            out.append((os.path.join(SITE, "vint", v, "models", f"enriched_s{s}.pt"), True, "wp"))
    for v, s in (("s4m_ft", 123), ("s1y_ft", 777)):
        out.append((os.path.join(SITE, "vint", v, "models", f"enriched_s{s}.pt"), True, "wp"))
    for s in (42, 123, 777):
        out.append((os.path.join(SITE, "models", f"enriched_s{s}.pt"), False, "wp"))
    for n in RUNS:
        out.append((os.path.join(run_dir(n), "draft_policy.pt"), RUNS[n][2], "policy"))
    for v in (0, 1):
        out.append((os.path.join(TRAINING, "drift_v2", "models", "gd_era_2.55.10.94189", f"generic_draft_{v}.pt"), True, "gd"))
        out.append((os.path.join(TRAINING, "drift_v2", "models", "gd_cutoff", f"generic_draft_{v}.pt"), False, "gd"))
    return out


def _sd(t):
    t = t.double()
    return float(t.std()) if t.numel() > 1 else float(t.abs().mean())


def stage_weights():
    import torch
    files = wp_files()
    tens = {}
    for p, _, g in files:
        sd_ = torch.load(p, map_location="cpu", weights_only=True)
        tens[p] = {k: v for k, v in sd_.items() if torch.is_tensor(v) and v.is_floating_point()}
    out = {}
    for p, on3090, g in files:
        sib = [q for q, o, gg in files if gg == g and not o and q != p and set(tens[q]) == set(tens[p])]
        rows = {}
        worst = {"nonfinite": 0, "max_abs": 0.0, "max_abs_over_sd": 0.0, "sd_ratio_max": 1.0, "sd_ratio_min": 1.0}
        for k, v in tens[p].items():
            v64 = v.double()
            fin = int((~torch.isfinite(v64)).sum())
            ma = float(v64.abs().max())
            sdv = _sd(v)
            r = ma / sdv if sdv > 0 else 0.0
            ratio = None
            if sib:
                s_sib = np.mean([_sd(tens[q][k]) for q in sib])
                ratio = sdv / s_sib if s_sib > 0 else None
            rows[k] = {"nonfinite": fin, "max_abs": round(ma, 5), "max_abs_over_sd": round(r, 2),
                       "sd_ratio_to_siblings": None if ratio is None else round(ratio, 4)}
            worst["nonfinite"] += fin
            worst["max_abs"] = max(worst["max_abs"], ma)
            worst["max_abs_over_sd"] = max(worst["max_abs_over_sd"], r)
            if ratio is not None:
                worst["sd_ratio_max"] = max(worst["sd_ratio_max"], ratio)
                worst["sd_ratio_min"] = min(worst["sd_ratio_min"], ratio)
        rec = {"on_3090": on3090, "group": g, "siblings": [os.path.relpath(q, TRAINING) for q in sib],
               "worst": {k: round(x, 4) if isinstance(x, float) else x for k, x in worst.items()}, "tensors": rows}
        if g == "policy":
            ck = os.path.join(os.path.dirname(p), "draft_policy_checkpoint.pt")
            if os.path.exists(ck):
                c = torch.load(ck, map_location="cpu", weights_only=False)
                same = all(torch.equal(c["model_state_dict"][k], tens[p][k]) for k in tens[p])
                rec["equals_checkpoint"] = bool(same)
                rec["checkpoint_episode"] = int(c.get("episode", -1))
                rec["checkpoint_best_eval_wp"] = float(c.get("best_eval_wp", float("nan")))
        out[os.path.relpath(p, TRAINING)] = rec
        print(f"{os.path.relpath(p, TRAINING)}: {json.dumps(rec['worst'])}"
              + (f" ckpt={rec.get('equals_checkpoint')}" if g == "policy" else ""), flush=True)
    return out


def _core(vintage):
    os.environ["P1_TIERS"] = "site"
    if vintage:
        os.environ["P1_VINTAGE"] = vintage
    from paper1_revision import core
    return core


def stage_stats(v):
    core = _core(v)
    pinned = core.STATS_DIR
    tmp = tempfile.mkdtemp(prefix=f"integ_stats_{v}_")
    core.CACHE, core.STATS_DIR = tmp, os.path.join(tmp, "stats")
    os.makedirs(core.STATS_DIR)
    t0 = time.time()
    core.build_stats()
    out = {"rebuild_secs": round(time.time() - t0)}
    for f in sorted(glob.glob(os.path.join(pinned, "*.json"))):
        b = os.path.basename(f)
        new = os.path.join(core.STATS_DIR, b)
        if not os.path.exists(new):
            out[b] = "not rebuilt"
            continue
        same_bytes = open(f, "rb").read() == open(new, "rb").read()
        same_json = json.load(open(f)) == json.load(open(new))
        out[b] = {"identical_bytes": same_bytes, "identical_json": same_json}
        print(b, out[b], flush=True)
    shutil.rmtree(tmp)
    return out


def _cmp(logged, got, keys, tol=2e-4):
    d = {k: [round(float(logged[k]), 6), round(float(got[k]), 6)] for k in keys if k in logged}
    ok = all(abs(a - b) <= tol for a, b in d.values())
    return d, ok


def stage_wp(v):
    import torch
    core = _core(v)
    from paper1_revision import train_wp
    tmp = tempfile.mkdtemp(prefix=f"integ_feats_{v}_")
    core.FEAT_DIR = tmp
    core.build_features(("val", "test", "NODRIFT"))
    meta = json.load(open(os.path.join(core.MODEL_DIR, "enriched.json")))
    cols = core.group_cols(meta["spec"]["groups"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vX, vY = train_wp._tensors(cols, "val", dev)
    out = {"n_val_rows": int(len(vX)), "logged_n_val_rows": meta["n_val_rows"]}
    for s, r in meta["seeds"].items():
        m, _ = train_wp.load("enriched", int(s), dev)
        with torch.no_grad():
            vl = float(torch.nn.BCELoss()(m(vX).view(-1), vY))
        rec = {"val_loss": [round(r["val_loss"], 6), round(vl, 6)]}
        ok = abs(vl - r["val_loss"]) <= 2e-4
        for fs in ("test", "NODRIFT"):
            d, o = _cmp(r[fs], train_wp.evaluate(m, cols, fs, dev), ("acc", "acc_sym", "ll", "slope"))
            rec[fs], ok = d, ok and o
        rec["match"] = bool(ok)
        out[f"s{s}"] = rec
        print(v, s, json.dumps(rec), flush=True)
    shutil.rmtree(tmp)
    return out


def stage_wpft(v):
    import torch
    core = _core(v)
    from paper1_revision import train_wp, vintage, vintage_ft
    from sweep_enriched_wp import WinProbEnrichedModel
    tmp = tempfile.mkdtemp(prefix=f"integ_feats_{v}_")
    core.FEAT_DIR = tmp
    core.build_features(("val", "test", "NODRIFT"))
    meta = json.load(open(os.path.join(core.MODEL_DIR, "enriched.json")))
    base = vintage.spec()["base"]
    cut = vintage.cutoff_idx(base)
    cols = core.group_cols(meta["spec"]["groups"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vXf, vXs, vy, vrid, _ = core.load_features("val")
    vs = np.where(vintage.build_info(vrid)[0] > cut)[0]
    vXf, vXs, vy = vXf[vs], vXs[vs], vy[vs]
    vX = torch.tensor(np.concatenate([vXf[:, cols], vXs[:, cols]]), device=dev)
    vY = torch.tensor(np.concatenate([vy, 1 - vy]), device=dev)
    tXf, tXs, ty, trid, _ = core.load_features("test")
    ts = np.where(vintage.build_info(trid)[0] > cut)[0]
    tXf, tXs, ty = tXf[ts], tXs[ts], ty[ts]

    def metrics(m):
        m.eval()
        with torch.no_grad():
            vl = float(torch.nn.BCELoss()(m(vX).view(-1), vY))
        r = {"val_loss": vl}
        for fs in ("test", "NODRIFT"):
            r[fs] = train_wp.evaluate(m, cols, fs, dev)
        r["val_post"] = vintage_ft.eval_arrays(m, vXf, vXs, vy, cols, dev)
        r["test_post"] = vintage_ft.eval_arrays(m, tXf, tXs, ty, cols, dev)
        return r

    def compare(logged, got):
        rec, ok = {"val_loss": [round(logged["val_loss"], 6), round(got["val_loss"], 6)]}, \
            abs(logged["val_loss"] - got["val_loss"]) <= 2e-4
        for fs in ("test", "NODRIFT", "val_post", "test_post"):
            if fs in logged:
                d, o = _cmp(logged[fs], got[fs], ("acc", "acc_sym", "ll", "slope"))
                rec[fs], ok = d, ok and o
        rec["match"] = bool(ok)
        return rec

    out = {"n_val_post": int(len(vs)), "n_test_post": int(len(ts))}
    for s, r in meta["seeds"].items():
        m = WinProbEnrichedModel(len(cols), meta["spec"]["arch"], dropout=0.3).to(dev)
        m.load_state_dict(torch.load(os.path.join(core.MODEL_DIR, f"enriched_s{s}.pt"), map_location="cpu",
                                     weights_only=True))
        out[f"ft_s{s}"] = compare(r, metrics(m))
        print(v, "ft", s, json.dumps(out[f"ft_s{s}"]), flush=True)
    bpt = meta["fine_tune"]["base_wp"]
    bpt = os.path.join(SITE, "vint", base, "models", os.path.basename(bpt))
    m = WinProbEnrichedModel(len(cols), meta["spec"]["arch"], dropout=0.3).to(dev)
    m.load_state_dict(torch.load(bpt, map_location="cpu", weights_only=True))
    out[f"base_{os.path.basename(bpt)}"] = compare(meta["before"], metrics(m))
    print(v, "base", json.dumps(out[f"base_{os.path.basename(bpt)}"]), flush=True)
    shutil.rmtree(tmp)
    return out


def stage_gd():
    import torch
    import v2_gd
    from drift2026 import common
    dev = torch.device("cuda")
    common.setup()
    rows, builds = common.load_data_with_patches()
    cut = builds.index(common.TRAIN_CUTOFF_BUILD)
    future = [r for r in rows if r["build_idx"] > cut]
    del rows
    Xf, yf, bf = v2_gd.samples(future, dev, with_builds=True)
    logged = json.load(open(os.path.join(TRAINING, "drift_v2", "results", "gd_eval_era_2.55.10.94189.json")))
    out = {"future_samples": int(len(yf))}
    for v in (0, 1):
        p = os.path.join(TRAINING, "drift_v2", "models", "gd_era_2.55.10.94189", f"generic_draft_{v}.pt")
        got = v2_gd.evaluate(p, Xf, yf, bf, dev)
        lg = logged[f"era_v{v}"]
        d = {k: [lg[k], got[k]] for k in ("top1", "top5", "top1_pick", "top1_ban")}
        out[f"v{v}"] = {"metrics": d, "match": all(abs(a - b) <= 0.01 for a, b in d.values())}
        print(v, json.dumps(out[f"v{v}"]), flush=True)
    return out


def stage_feval(name, n=512, seed=99999):
    """The worker's EVAL on n fixed drafts (python random seeded), its own search at NUM_SIMS // 2."""
    import subprocess
    v, s, on3090 = RUNS[name]
    env = dict(os.environ, P1_TIERS="site")
    if v:
        env["P1_VINTAGE"] = v
    else:
        env.pop("P1_VINTAGE", None)
    pe = subprocess.run([sys.executable, os.path.join(TRAINING, "paper1_revision", "train_mcts.py"), "F", str(s),
                         "--print-env"], env=env, capture_output=True, text=True, cwd=TRAINING)
    assert pe.returncode == 0, pe.stderr[-2000:]
    spec = json.loads(pe.stdout.strip().splitlines()[-1])
    rd = spec["env"]["MCTS_SAVE_DIR"]
    assert os.path.realpath(rd) == os.path.realpath(run_dir(name)), (rd, run_dir(name))
    child = dict(env, **spec["env"])
    child["MCTS_SAVE_DIR"] = tempfile.mkdtemp(prefix=f"integ_feval_{name}_")    # never the run's own dir
    child["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    child["P3_FEVAL_POLICY"] = os.path.join(rd, "draft_policy.pt")
    child["P3_FEVAL_N"], child["P3_FEVAL_SEED"] = str(n), str(seed)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "_feval_child"], env=child, capture_output=True,
                       text=True, cwd=TRAINING)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    import torch
    ckp = os.path.join(rd, "draft_policy_checkpoint.pt")
    ck = torch.load(ckp, map_location="cpu", weights_only=False) if os.path.exists(ckp) else {}   # main box has it
    logf = os.path.join(os.path.dirname(os.path.dirname(rd)), "logs", f"mcts_F_oof_s{s}.log")
    evals = []
    if os.path.exists(logf):
        for line in open(logf, errors="replace"):
            if "EVAL @" in line:
                evals.append(float(line.split("avg_wp=")[1].split()[0]))
    res.update({"run": name, "on_3090": on3090, "logged_best_eval_wp": float(ck.get("best_eval_wp", float("nan"))),
                "logged_best_episode": int(ck.get("episode", -1)),
                "logged_evals": len(evals), "logged_eval_mean_last10": round(float(np.mean(evals[-10:])), 4) if evals else None})
    shutil.rmtree(child["MCTS_SAVE_DIR"], ignore_errors=True)
    print(json.dumps(res), flush=True)
    return res


def _feval_child():
    """Runs in its own process with the run's worker environment (stats and WP paths read at import)."""
    import random
    sys.path.insert(0, TRAINING)
    os.chdir(TRAINING)
    import sweep_enriched_wp as swp
    comp = os.environ["P1R_COMP_PATH"]

    def _load_compositions(self):
        raw = json.load(open(comp))
        self.comp_data = {t: {",".join(sorted(c["roles"])): (c["winRate"], c["games"]) for c in cs}
                          for t, cs in raw.items()}
    swp.StatsCache._load_compositions = _load_compositions
    import torch
    import train_mcts_worker as W
    gd = W.GenericDraftModel()
    gd.load_state_dict(torch.load(W.GD_PATH_OVERRIDE, weights_only=True, map_location="cpu"))
    gd.eval()
    gd_flat, gd_off = W.extract_gd_weights(gd)
    from sweep_enriched_wp import StatsCache, WinProbEnrichedModel, FEATURE_GROUP_DIMS, INPUT_DIM_BASE
    groups = ['role_counts', 'team_avg_wr', 'map_delta', 'pairwise_counters', 'pairwise_synergies',
              'counter_detail', 'meta_strength', 'draft_diversity', 'comp_wr']
    wdim = INPUT_DIM_BASE + sum(FEATURE_GROUP_DIMS[g] for g in groups)
    wp = WinProbEnrichedModel(wdim, [256, 128], dropout=0.3)
    wp.load_state_dict(torch.load(W.WP_PATH_OVERRIDE, weights_only=True, map_location="cpu"))
    wp.eval()
    wp_flat, wp_names = W.extract_wp_weights(wp)
    wp_off = W.build_wp_net_offsets(wp, wp_names, wdim)
    lut = W.extract_lookup_tables(StatsCache(), step_embed_weights=None)
    dev = torch.device("cuda:0")
    net = W.AlphaZeroDraftNet(size="base", policy_head_type="linear").to(dev)
    net.load_state_dict(torch.load(os.environ["P3_FEVAL_POLICY"], map_location=dev, weights_only=True))
    net.eval()
    pf, po = W.extract_policy_weights(net)
    n, seed = int(os.environ["P3_FEVAL_N"]), int(os.environ["P3_FEVAL_SEED"])
    eng = W.kernel.MCTSKernelEngine(pf, gd_flat, wp_flat, po, gd_off, wp_off, lut, max_concurrent=128, device_id=0)
    sargs = (W.SEARCH_MODE, W.PW_K, W.PW_ALPHA) if hasattr(W.kernel, "SEARCH_CHANCE") else ()
    random.seed(seed)
    wps = []
    for b0 in range(0, n, 128):
        k = min(128, n - b0)
        cfg = np.array([[random.randint(0, len(W.MAPS) - 1), random.randint(0, len(W.SKILL_TIERS) - 1), (b0 + i) % 2]
                        for i in range(k)], np.int32)
        bs = np.zeros((k * 8, W.STATE_DIM), np.float32)
        bp = np.zeros((k * 8, W.NUM_HEROES), np.float32)
        bm = np.zeros((k * 8, W.NUM_HEROES), np.float32)
        bv = np.zeros(k * 8, np.float32)
        _, ev = eng.run_episodes_into_buffer(cfg, W.NUM_SIMS // 2, 2.0, seed + b0, bs, bp, bm, bv, 0, k * 8, *sargs)
        wps += list(ev)
    wps = np.array(wps, float)
    print(json.dumps({"n": int(len(wps)), "avg_wp": round(float(wps.mean()), 4),
                      "avg_wp_se": round(float(wps.std(ddof=1) / np.sqrt(len(wps))), 4),
                      "win_rate": round(float((wps > 0.5).mean()), 4), "sims": W.NUM_SIMS // 2}))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_feval_child":
        _feval_child()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["weights", "stats", "wp", "wpft", "gd", "feval"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or os.path.join(os.environ.get("P3_RESULTS") or os.path.join(HERE, "results"), "p3t",
                                    "integrity_3090")
    os.makedirs(out_dir, exist_ok=True)
    fn = {"weights": stage_weights, "gd": stage_gd}.get(a.stage)
    res = fn() if fn else {"stats": stage_stats, "wp": stage_wp, "wpft": stage_wpft, "feval": stage_feval}[a.stage](a.arg)
    name = a.stage + (f"_{a.arg}" if a.arg else "")
    json.dump(res, open(os.path.join(out_dir, name + ".json"), "w"), indent=1)
    print(f"INTEGRITY {name} DONE", flush=True)


if __name__ == "__main__":
    main()
