"""
P3 pipeline runner for the remote workers: named chains of stages, each a
subprocess. Pause-robust: a finished stage writes a done marker (with its
command and wall time) and is skipped on the next run, so a pause or a kill
loses only the stage in progress. On SIGTERM the running stage is stopped and
the runner exits without marking it.

Outputs of a run go to $P3_RESULTS (default results/oct26); stage logs to
$P3_RESULTS/logs/<stage>.log. A failed stage writes $P3_RESULTS/ALERT_p3
and the runner exits 1 (the hotsjob manifest shows failed).

Usage (from training/):
  python3 personalization/p3_pipe.py <chain> [--from STAGE] [--only S1,S2] [--list]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
PY = sys.executable
P = "personalization/"
TAG = os.environ.get("P3_EXPORT_TAG", "oct05")


def s(name, *argv):
    return (name, [PY, "-u", P + argv[0]] + list(argv[1:]))


CHAINS = {
    # A7: every analysis the fixes touch, phase 1 -> phase 2 -> extensions
    "fixes": [
        s("hs_slots", "p3_hs_fit.py", "slots"),
        s("check_slots", "p3_pipe_checks.py", "slots"),
        s("hl_parse", "p3_hero_level_causal.py", "parse", "--tag", TAG),
        s("hl_build_snap", "p3_hero_level_causal.py", "build", "--tag", TAG),
        s("hs_fit", "p3_hs_fit.py", "fit"),
        s("hs_eval", "p3_hs_eval.py"),
        s("hs_nopool", "p3_hs_nopool.py"),
        s("hs_paired", "p3_hs_paired.py"),
        s("hs_cassia", "p3_hs_cassia.py"),
        s("hs_calib", "p3_hs_calib.py"),
        s("hs_display", "p3_hs_display.py"),
        s("sd_kalman", "p3_sd_kalman.py"),
        s("sd_eval", "p3_sd_eval.py"),
        s("sd_session", "p3_sd_session.py"),
        s("sd_order", "p3_sd_order.py"),
        s("sd_similarity", "p3_sd_similarity.py"),
        s("hs_similarity", "p3_hs_similarity.py"),
        s("hs_similarity_pool", "p3_hs_similarity_pool.py"),
        s("hs_similarity_diverge", "p3_hs_similarity_diverge.py"),
        s("sd_similarity_viz", "p3_sd_similarity_viz.py"),
        s("sd_mmrleak", "p3_sd_mmrleak.py"),
        s("ph_role", "p3_ph_role.py"),
        s("ph_patch_ab", "p3_ph_patch.py", "--ab-only"),
        s("val_validity", "p3_val_validity.py"),
        s("fix_arms", "p3_fix_arms.py", "arms"),
        s("fix_arms_role", "p3_fix_arms.py", "role"),
        s("fix_arms_misc", "p3_fix_arms.py", "misc"),
        s("fix_arms_cf", "p3_fix_arms.py", "cf"),
        s("fix_mmr", "p3_fix_mmr.py"),
        s("fix_neverplayed", "p3_fix_neverplayed.py"),
        s("fix_jump", "p3_fix_jump.py"),
        s("nested_lift", "p3_nested_lift.py", "analyze"),
        s("x_ext", "p3_x_common.py", "ext"),
        s("hl_build", "p3_hero_level_causal.py", "build", "--tag", TAG),
        s("x_predall", "p3_x_predall.py"),
        s("x_side", "p3_x_side.py"),
        s("x_oot", "p3_x_oot.py"),
        s("x_adopt", "p3_x_adopt.py"),
        s("x_learn", "p3_x_learn.py"),
        s("x_smurf", "p3_x_smurf.py"),
        s("x_newacct", "p3_x_newacct.py"),
        s("x_map", "p3_x_map.py"),
        s("x_robust", "p3_x_robust.py"),
    ],
    # C: headline on the full (backfilled) history; run with P3_CACHE=<full
    # cache from p3_c_history.py>, P3_RESULTS=results/oct26_full and
    # P3_EXPORT_TAG=<the post-backfill export>
    "full": [
        s("hs_slots", "p3_hs_fit.py", "slots"),
        s("hl_parse", "p3_hero_level_causal.py", "parse", "--tag", TAG),
        s("hl_build_snap", "p3_hero_level_causal.py", "build", "--tag", TAG),
        s("hs_fit", "p3_hs_fit.py", "fit"),
        s("hs_eval", "p3_hs_eval.py"),
        s("mmr_at_game", "p3_mmr_at_game.py"),
        s("x_ext", "p3_x_common.py", "ext"),
        s("hl_build", "p3_hero_level_causal.py", "build", "--tag", TAG),
        s("x_predall", "p3_x_predall.py"),
        s("x_oot", "p3_x_oot.py"),
        s("x_newacct", "p3_x_newacct.py"),
        s("c_lifetime", "p3_c_lifetime.py"),
    ],
    # #2 on the window caches (run after the fixes chain)
    "lifetime": [
        s("c_lifetime", "p3_c_lifetime.py"),
    ],
    # B models on the rebuilt caches (after the drafter chain: the ban model
    # reads its tables and the train-window GD). Development choices were made
    # on the legacy caches; these runs are the reported numbers. The sealed
    # build is scored by the separate "bfinal" chain once, at the end.
    "bmodels": [
        s("b_onetrick", "p3_b_onetrick.py"),
        s("b_demean", "p3_b_demean.py", "--kalman"),
        s("b_nowcast", "p3_b_nowcast.py"),
        s("b_bans", "p3_b_bans.py"),
    ],
    "bfinal": [
        s("b_onetrick_final", "p3_b_onetrick.py", "--final"),
        s("b_demean_final", "p3_b_demean.py", "--final"),
    ],
    # A5/A7: the drafter harness on the fixed protocol (train-window GD and
    # BC prior, team-level assignment, GD-sampled bans, lag-1 tables)
    "drafter": [
        s("dr_lobbies", "p3_dr_core.py"),
        s("dr_tables", "p3_dr_core.py", "tables"),
        s("dr_imitation", "p3_dr_imitation.py"),
        s("dr_drafter", "p3_dr_drafter.py", "--procs", "7", "--assign", "team"),
        s("dr_drafter_ctrl5", "p3_dr_drafter.py", "--procs", "7", "--assign", "team", "--lobby-req", "ctrl5"),
        s("dr_analyze", "p3_dr_analyze.py"),
        s("fix_realized_onestep", "p3_fix_realized.py"),
        s("fix_ext5_bsim", "p3_fix_ext5.py", "b_sim", "--procs", "7"),
        s("fix_ext5_banalyze", "p3_fix_ext5.py", "b_analyze"),
        s("fix_ext5_d", "p3_fix_ext5.py", "d"),
        s("x_draft_assign", "p3_x_draft.py", "assign"),
        s("x_draft_combine", "p3_x_draft.py", "combine"),
        s("x_draft_chem", "p3_x_draft.py", "chem"),
        s("x_draft_sim", "p3_x_draft.py", "sim", "--procs", "7"),
        s("x_draft_analyze", "p3_x_draft.py", "analyze"),
        s("x_ban_diag", "p3_x_ban_diag.py"),
        s("x_ban_nat", "p3_x_ban_nat.py"),
        s("x_ban_model", "p3_x_ban_model.py"),
        s("pgd_model", "p3_pgd_model.py"),
    ],
}


def marker_dir():
    d = os.path.join(results_dir(), "_done")
    os.makedirs(d, exist_ok=True)
    return d


def results_dir():
    return os.environ.get("P3_RESULTS") or os.path.join(HERE, "results", "oct26")


_child = None


def _term(signum, frame):
    if _child is not None and _child.poll() is None:
        _child.terminate()
        try:
            _child.wait(30)
        except subprocess.TimeoutExpired:
            _child.kill()
    print(f"[p3_pipe] signal {signum}: stopped; stage not marked done", flush=True)
    sys.exit(143)


def main():
    global _child
    ap = argparse.ArgumentParser()
    ap.add_argument("chain", choices=sorted(CHAINS))
    ap.add_argument("--from", dest="start")
    ap.add_argument("--only")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("P3_RESULTS", results_dir())
    os.makedirs(os.path.join(results_dir(), "logs"), exist_ok=True)
    stages = CHAINS[a.chain]
    names = [n for n, _ in stages]
    if a.start:
        stages = stages[names.index(a.start):]
    if a.only:
        keep = set(a.only.split(","))
        stages = [x for x in stages if x[0] in keep]
    if a.list:
        for n, cmd in stages:
            done = os.path.exists(os.path.join(marker_dir(), f"{a.chain}__{n}.done"))
            print(f"{'done ' if done else '     '}{n}: {' '.join(cmd[2:])}")
        return
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    for n, cmd in stages:
        mk = os.path.join(marker_dir(), f"{a.chain}__{n}.done")
        if os.path.exists(mk):
            print(f"[p3_pipe] skip {n} (done)", flush=True)
            continue
        log = os.path.join(results_dir(), "logs", f"{n}.log")
        t0 = time.time()
        print(f"[p3_pipe] {time.strftime('%m-%d %H:%M:%S')} start {n}: {' '.join(cmd[2:])}", flush=True)
        with open(log, "a") as lf:
            lf.write(f"# start {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            lf.flush()
            _child = subprocess.Popen(cmd, cwd=TRAINING, stdout=lf, stderr=subprocess.STDOUT)
            rc = _child.wait()
        dt = time.time() - t0
        if rc != 0:
            msg = f"{time.strftime('%m-%d %H:%M:%S')} {a.chain}/{n} failed rc={rc} after {dt:.0f}s; log {log}"
            with open(os.path.join(results_dir(), "ALERT_p3"), "a") as f:
                f.write(msg + "\n")
            print(f"[p3_pipe] {msg}", flush=True)
            sys.exit(1)
        with open(mk, "w") as f:
            json.dump({"cmd": cmd, "seconds": round(dt), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        print(f"[p3_pipe] done {n} in {dt:.0f}s", flush=True)
    print(f"[p3_pipe] chain {a.chain} complete", flush=True)


if __name__ == "__main__":
    main()
