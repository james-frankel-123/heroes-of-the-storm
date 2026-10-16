"""Session launcher for proposer sessions (WAVE0_BRIEF item 10, FRAMING 8, 10.3).

  python -m training.scilearn.session prepare <sid> --arm b|blinded|shuffled|a|c [--variant K]
  python -m training.scilearn.session prefit <sid>          fit the incumbent remotely (blocking)
  python -m training.scilearn.session broker <sid>          start the broker in the background
  python -m training.scilearn.session launch <sid> [--dry-run]   run claude -p in the sandbox
  python -m training.scilearn.session stop <sid>
  python -m training.scilearn.session canaries <sid>       6 harness canaries (harness ledger only)
  python -m training.scilearn.session closeout <sid>       final record, after `stop` and the final refit
  python -m training.scilearn.session cold-refit <sid>     end-of-cycle cold refit to convergence (blocking)
  python -m training.scilearn.session wait-broker <sid>    block until the broker answers (before launch)

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


def prepare(sid, arm, variant="K", cycle=1, predecessor=None):
    BR.open_session(sid, arm, variant, cycle, predecessor=predecessor)
    prompt = build_view_and_prompt(sid)
    print(json.dumps({"session": sid, "arm": arm, "dir": str(C.SESSIONS / sid), "prompt": str(prompt)}, indent=1))


def previous_cycle_digest(st, names):
    """Plain list of what the predecessor sessions tested, in this session's labels. Accepted
    terms of every earlier cycle that are still in the incumbent; the cards of the last cycle
    that were not accepted."""
    tables = st.get("inherited_tables") or {}
    inh = st.get("inherited_cards") or {}
    in_inc = {t.get("name") for t in st.get("accepted_terms", [])}
    last = max([c.get("cycle", 0) for c in inh.values()] + [0])
    L = ["", "## Your previous cycles, as recorded by the harness", "",
         "Accepted terms now in your incumbent (card id, cycle, term):"]
    n_acc = 0
    for cid, c in inh.items():
        t = c.get("term") or {}
        if c["status"] == "accepted" and t.get("name") in in_inc:
            n_acc += 1
            tt = {k: v for k, v in t.items() if k != "name"}
            g = (c.get("released") or {}).get("gain")
            L.append(f"- {cid} (cycle {c.get('cycle', '?')}): {names.term_name_out(json.dumps(tt))}; "
                     f"released dev gain {_g(g)}" + ("; explainable table" if cid in tables else ""))
    if not n_acc:
        L.append("- none: your incumbent is the K0 backbone")
    replaced = st.get("replaced") or {}
    for old, new in replaced.items():
        L.append(f"- {old} was replaced by {new}")
    L += ["", f"Your other cards of the last cycle (cycle {last}), with their status:"]
    for cid, c in inh.items():
        if c.get("cycle", last) != last or c["status"] == "accepted":
            continue
        t = {k: v for k, v in (c["term"] or {}).items() if k != "name"} if c.get("term") else {}
        rel = c.get("released") or {}
        if c.get("voided_release"):
            st_ = ("passed its gate, then voided: your incumbent plus this term exceeded the 20,000-parameter "
                   "cap, so it was never added and no bits were charged")
            rel = c["voided_release"]
        else:
            st_ = {"error": "harness error, never tested, no bits charged; the harness bug is fixed",
                   "near_miss": "near miss"}.get(c["status"], c["status"])
        g = rel.get("gain")
        L.append(f"- {cid} ({st_}): {names.term_name_out(json.dumps(t))}"
                 + (f"; released gain {_g(g)}" if g is not None else ""))
    earlier = sum(1 for c in inh.values() if c.get("cycle", last) < last and c["status"] != "accepted")
    if earlier:
        L.append(f"- and {earlier} cards of earlier cycles that added no term; their ids stay reserved")
    L += ["", "Bits you spent on cell groups of the dev tranche in earlier cycles still count toward each group's "
          "threshold (same data). `hyp budget` shows them.", ""]
    return "\n".join(L)


def _g(x):
    return "n/a" if x is None else f"{x:.4f}"


def build_view_and_prompt(sid):
    """(Re)write the session's read-only view and prompt from the current session kit.
    Used by prepare, and by `reprompt` for a session that has not been launched."""
    st = json.loads((BR.private_session_dir(sid) / "state.json").read_text())
    from training.scilearn import data as D
    _, meta, _, _ = D.load(st["snapshot"], ["Dpre"])
    arm, predecessor = st["arm"], st.get("predecessor")
    names = BR.Names(meta, st.get("blind_key") if not BR.ARMS[arm]["names"] else None)
    sdir = C.SESSIONS / sid
    view = sdir / "view"
    if view.exists():
        for p in sorted(view.rglob("*"), reverse=True):
            p.chmod(0o755 if p.is_dir() else 0o644)
        shutil.rmtree(view)
    for d in ("view/bin", "work/cards", "logs"):
        (sdir / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(KITDIR / "bin" / "hyp", view / "bin" / "hyp")
    shutil.copy2(KITDIR / "PROPOSER_GUIDE.md", view / "PROPOSER_GUIDE.md")
    shutil.copy2(KITDIR / "card_template.yaml", view / "card_template.yaml")
    kit = public_kit(names)
    if kit:
        (view / "kit.json").write_text(json.dumps(kit, indent=1))
    prompt = (KITDIR / "prompts" / "standard.md").read_text()
    if predecessor:
        cyc = int(st.get("cycle", 2))
        prompt += (KITDIR / "prompts" / ("cycle2.md" if cyc <= 2 else "cycle3.md")).read_text()
        prompt += previous_cycle_digest(st, names)
        prev_sum = C.SESSIONS / predecessor / "work" / "summary.md"
        if prev_sum.exists():
            shutil.copy2(prev_sum, view / "previous_cycle_summary.md")
    if arm == "c":
        prompt += (KITDIR / "prompts" / "arm_c_extra.md").read_text()
    if arm == "blinded":
        prompt += (KITDIR / "prompts" / "blinded_extra.md").read_text()
    (BR.private_session_dir(sid) / "prompt.md").write_text(prompt)
    for p in view.rglob("*"):
        p.chmod(0o555 if p.is_dir() or p.name == "hyp" else 0o444)
    return BR.private_session_dir(sid) / "prompt.md"


def prefit(sid):
    """Incumbent fit, plus the explore (S) and confirm (Dpre+G) fits when it holds tables."""
    from training.scilearn import harness
    st = json.loads((BR.private_session_dir(sid) / "state.json").read_text())
    dev = BR.ARMS[st["arm"]]["dev"]
    out = {"incumbent_job": harness.ensure_incumbent(st, dev)["job"]}
    if harness.accepted_tables(st):
        out["explore_job"] = harness.ensure_fit(st, dev, fit=harness.EXPLORE_FIT)["job"]
        out["confirm_job"] = harness.ensure_fit(st, dev, fit=harness.CONFIRM_FIT[dev])["job"]
    print(json.dumps(out))


def cold_refit(sid):
    """End-of-cycle refit: the session's final incumbent, cold, to convergence (Max,
    2026-10-15). Blocking; takes the 3090 refit token, so one slot stays free for tests."""
    from training.scilearn import harness
    st = json.loads((BR.private_session_dir(sid) / "state.json").read_text())
    dev = BR.ARMS[st["arm"]]["dev"]
    e = harness.cold_refit(st, dev, on_stage=lambda m: print(m, flush=True))
    print(json.dumps({"session": sid, "job": e["job"], "fit_mode": e.get("fit_mode"),
                      "fit_status": e.get("fit_status"), "dev_metrics_private": e.get("dev_metrics_private")},
                     indent=1, default=float))


def wait_broker(sid, timeout=1800, poll=5):
    """Block until the session's broker answers on its socket. A socket file alone is not
    enough: a stale one survives a stopped broker, and a new broker binds only after it has
    loaded the snapshot (about a minute). In cycle 2 a launch raced its broker."""
    import socket
    import time
    sock = C.SESSIONS / sid / "broker.sock"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.settimeout(10)
            c.connect(str(sock))
            c.sendall(b'{"op": "help"}\n')
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = c.recv(65536)
                if not chunk:
                    break
                buf += chunk
            c.close()
            if json.loads(buf).get("ok"):
                print(f"{sid}: broker answers on {sock} after {time.time() - t0:.0f} s")
                return True
        except (OSError, ValueError):
            pass
        time.sleep(poll)
    sys.exit(f"{sid}: no broker answer on {sock} within {timeout} s")


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


def closeout(sid):
    """Append the cycle's closeout record: final status of every card, the accepted terms and
    the registered incumbent that holds all of them. Refuses while the broker runs, while a
    test is pending, or before that incumbent is fitted."""
    from training.scilearn import harness
    from training.scilearn.ledger import Ledger
    priv = BR.private_session_dir(sid)
    if (priv / "broker.pid").exists():
        sys.exit(f"{sid}: broker still running; `session stop {sid}` first")
    st = json.loads((priv / "state.json").read_text())
    pending = sorted(c for c, v in st["cards"].items() if v.get("status") in ("testing", "mtesting"))
    if pending:
        sys.exit(f"{sid}: tests still pending: {pending}")
    dev = BR.ARMS[st["arm"]]["dev"]
    key = harness.incumbent_key(st, dev)
    inc = harness._registry().get(key)
    if inc is None:
        sys.exit(f"{sid}: final incumbent inc_{key} is not fitted; run ensure_incumbent first")
    led = Ledger(sid)
    cyc = st["cycle"]
    recs = list(led.records())
    if any(r["kind"] == "closeout" and f"cycle={cyc}" in (r.get("note") or "") for r in recs):
        sys.exit(f"{sid}: cycle {cyc} already closed")
    dev_bits = med_bits = 0.0
    for r in recs:
        if r.get("bits_charged") and f"cycle={cyc}" in (r.get("note") or ""):
            if json.loads(r.get("params_json") or "{}").get("_budget") == "mediator":
                med_bits += r["bits_charged"]
            else:
                dev_bits += r["bits_charged"]
    names = [t["name"] for t in st.get("accepted_terms", [])]
    led.append("closeout", session_id=sid, arm=st["arm"], variant=st["variant"], snapshot_id=st["snapshot"],
               params_json=json.dumps({"accepted_terms": names,
                                       "cards": {c: v.get("status") for c, v in st["cards"].items()},
                                       "incumbent_job": inc["job"]}),
               note=(f"cycle-{cyc} close: broker stopped; {len(st['cards'])} cards; dev bits charged "
                     f"{dev_bits:.2f}; mediator bits {med_bits:.2f}; no test pending; incumbent {inc['job']} "
                     f"holds all {len(names)} accepted terms cycle={cyc}"))
    print(f"{sid}: closed; incumbent {inc['job']}; {len(names)} accepted terms")


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
    p.add_argument("--from", dest="predecessor", help="closed session whose incumbent and group bits carry over")
    for name in ("prefit", "broker", "stop", "canaries", "explain-canaries", "reprompt", "closeout", "cold-refit",
                 "wait-broker"):
        q = sp.add_parser(name); q.add_argument("sid")
    l = sp.add_parser("launch"); l.add_argument("sid"); l.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.sid, a.arm, a.variant, a.cycle, a.predecessor)
    elif a.cmd == "prefit":
        prefit(a.sid)
    elif a.cmd == "broker":
        start_broker(a.sid)
    elif a.cmd == "stop":
        stop(a.sid)
    elif a.cmd == "closeout":
        closeout(a.sid)
    elif a.cmd == "cold-refit":
        cold_refit(a.sid)
    elif a.cmd == "wait-broker":
        wait_broker(a.sid)
    elif a.cmd == "reprompt":
        if list((C.SESSIONS / a.sid / "logs").glob("claude-*.stream.jsonl")):
            sys.exit(f"{a.sid} has been launched; its prompt is part of the record")
        print(build_view_and_prompt(a.sid))
    elif a.cmd == "explain-canaries":
        from training.scilearn import harness
        st = json.loads((BR.private_session_dir(a.sid) / "state.json").read_text())
        print(json.dumps(harness.run_explain_canaries(st, BR.ARMS[st["arm"]]["dev"]), indent=1))
    elif a.cmd == "canaries":
        from training.scilearn import harness
        print(json.dumps(harness.run_canaries(BR.Broker(a.sid)), indent=1))
    else:
        sys.exit(launch(a.sid, a.dry_run))
