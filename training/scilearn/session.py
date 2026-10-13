"""Session launcher for proposer sessions (WAVE0_BRIEF item 10, FRAMING 8, 10.3).

  python -m training.scilearn.session prepare <sid> --arm b|blinded|shuffled|a|c [--variant K]
  python -m training.scilearn.session prefit <sid>          fit the incumbent remotely (blocking)
  python -m training.scilearn.session broker <sid>          start the broker in the background
  python -m training.scilearn.session launch <sid> [--dry-run]   run claude -p in the sandbox
  python -m training.scilearn.session stop <sid>

One budget per session (its own ledger), one working directory (SESSIONS/<sid>/work), one
broker socket bound only into that session's container.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys

from training.scilearn import broker as BR
from training.scilearn import config as C

KIT_SRC = C.CODE / "kit" / "kit_2.57.json"
KITDIR = C.CODE / "session_kit"
VENV_PY = os.environ.get("SCILEARN_PY", "/home/max/venvs/scilearn/bin/python")


def public_kit(names):
    """Kit values and axis definitions only: no provenance, no per-coder raw values.
    For the blinded arm the hero keys are H-labels."""
    if not KIT_SRC.exists():
        return None
    k = json.loads(KIT_SRC.read_text())
    axes = k["axes"]
    axis_names = axes if isinstance(axes[0], str) else [a["name"] for a in axes]
    out = {"version": k.get("version"), "axes": axes, "identities": k.get("identities"),
           "scale": k.get("scale"), "heroes": {}}
    for h, v in k["heroes"].items():
        vals = v["values"] if isinstance(v, dict) and "values" in v else v
        role = v.get("role") if isinstance(v, dict) else None
        out["heroes"][names.h_out.get(h, h)] = {"role": role, **{a: vals[a] for a in axis_names}}
    if names.blind:
        out["heroes"] = dict(sorted(out["heroes"].items()))
    return out


def prepare(sid, arm, variant="K", cycle=1):
    st = BR.open_session(sid, arm, variant, cycle)
    from training.scilearn import data as D
    _, meta, _, _ = D.load(st["snapshot"], ["Dpre"])
    names = BR.Names(meta, st.get("blind_key") if not BR.ARMS[arm]["names"] else None)
    sdir = C.SESSIONS / sid
    view = sdir / "view"
    for d in ("view/bin", "work/cards", "logs"):
        (sdir / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(KITDIR / "bin" / "hyp", view / "bin" / "hyp")
    shutil.copy2(KITDIR / "PROPOSER_GUIDE.md", view / "PROPOSER_GUIDE.md")
    shutil.copy2(KITDIR / "card_template.yaml", view / "card_template.yaml")
    kit = public_kit(names)
    if kit:
        (view / "kit.json").write_text(json.dumps(kit, indent=1))
    prompt = (KITDIR / "prompts" / "standard.md").read_text()
    if arm == "c":
        prompt += (KITDIR / "prompts" / "arm_c_extra.md").read_text()
    if arm == "blinded":
        prompt += (KITDIR / "prompts" / "blinded_extra.md").read_text()
    priv = BR.private_session_dir(sid)
    (priv / "prompt.md").write_text(prompt)
    for p in view.rglob("*"):
        p.chmod(0o555 if p.is_dir() or p.name == "hyp" else 0o444)
    print(json.dumps({"session": sid, "arm": arm, "dir": str(sdir), "prompt": str(priv / "prompt.md")}, indent=1))


def prefit(sid):
    from training.scilearn import harness
    st = json.loads((BR.private_session_dir(sid) / "state.json").read_text())
    inc = harness.ensure_incumbent(st, BR.ARMS[st["arm"]]["dev"])
    print(json.dumps({"incumbent_job": inc["job"]}))


def start_broker(sid):
    log = open(C.SESSIONS / sid / "logs" / "broker.log", "a")
    env = dict(os.environ, PYTHONPATH=str(C.REPO), OMP_NUM_THREADS="2")
    p = subprocess.Popen(["nice", "-n", "19", "taskset", "-c", "60-63", VENV_PY, "-m",
                          "training.scilearn.broker", "serve", sid], stdout=log, stderr=subprocess.STDOUT,
                         cwd=str(C.REPO), env=env, start_new_session=True)
    (BR.private_session_dir(sid) / "broker.pid").write_text(str(p.pid))
    print(f"broker pid {p.pid}; socket {C.SESSIONS / sid / 'broker.sock'}")


def stop(sid):
    pf = BR.private_session_dir(sid) / "broker.pid"
    if pf.exists():
        try:
            os.killpg(int(pf.read_text()), signal.SIGTERM)
        except ProcessLookupError:
            pass
        pf.unlink()


def launch(sid, dry_run=False, extra=()):
    prompt = BR.private_session_dir(sid) / "prompt.md"
    cmd = [sys.executable, str(C.CODE / "sandbox" / "launch_session.py"), sid, "--prompt-file", str(prompt),
           "--model", "claude-opus-5-5", *extra]
    if dry_run:
        cmd.append("--dry-run")
    env = dict(os.environ, CLAUDE_CONFIG_DIR="/home/max/.claude-account-2")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare"); p.add_argument("sid"); p.add_argument("--arm", required=True, choices=sorted(BR.ARMS))
    p.add_argument("--variant", default="K"); p.add_argument("--cycle", type=int, default=1)
    for name in ("prefit", "broker", "stop"):
        q = sp.add_parser(name); q.add_argument("sid")
    l = sp.add_parser("launch"); l.add_argument("sid"); l.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.sid, a.arm, a.variant, a.cycle)
    elif a.cmd == "prefit":
        prefit(a.sid)
    elif a.cmd == "broker":
        start_broker(a.sid)
    elif a.cmd == "stop":
        stop(a.sid)
    else:
        sys.exit(launch(a.sid, a.dry_run))
