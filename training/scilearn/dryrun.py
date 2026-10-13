"""Wave-0 acceptance dry runs (WAVE0_BRIEF "Acceptance before wave 1").

  python -m training.scilearn.dryrun <sid> --arm b|shuffled [--canaries]

Prepares a session, fits its incumbent remotely, starts its broker, then runs a scripted
"proposer" INSIDE the sandbox that uses only `hyp`: meta, an S query, a dev query, a card
(no healer, a structural term the backbone K0 lacks), the Q2 test, and the ledger. Then
verifies the ledger chain and optionally runs the session's 6 canaries.
"""
import argparse
import json
import subprocess
import sys
import time

from training.scilearn import broker as BR
from training.scilearn import config as C
from training.scilearn import harness as HN
from training.scilearn import ledger as LG
from training.scilearn import session as SS

CARD = """id: nohealer1
family: composition
tag: structural
theory: >
  A team without a healer cannot recover health between trades, so it loses extended
  fights and objective contests even when its damage is higher.
mechanism:
  intermediate: deaths in fights after the first trade
  direction: more for the team without a healer
  window: whole game
  negative_control: minion XP share before minute 5
expected_effect: {sign: "-", pp: 4.0, logloss: 0.0002, frac_games: 0.05}
term:
  type: indicator
  predicates:
    - {field: role_count, key: Healer, op: "==", value: 0}
  sign: "-"
sources: ["S query in this session"]
alternatives_considered: ["concave healing_throughput sum"]
"""

SCRIPT = r"""
set -u
echo '== meta'; hyp meta | head -c 600; echo
echo '== budget'; hyp budget | head -c 400; echo
echo '== S query'; hyp query '{"class":"Q1","tranche":"S","stat":"win_rate","filter":{"role_count":{"Healer":0}},"group_by":["tier"]}'
echo '== dev query'; hyp query '{"class":"Q1","tranche":"Dpost","stat":"win_rate","filter":{"role_count":{"Healer":0}}}'
echo '== try a closed tranche'; hyp query '{"class":"Q1","tranche":"C","stat":"win_rate"}'
echo '== submit'; hyp card submit /work/cards/nohealer1.yaml
echo '== test'; hyp test nohealer1
while hyp test-status | grep -q '"testing"'; do sleep 30; done
echo '== result'; hyp test-status
echo '== ledger'; hyp ledger --mine
"""


def main(sid, arm, canaries):
    if not (BR.private_session_dir(sid) / "state.json").exists():
        SS.prepare(sid, arm)
    (C.SESSIONS / sid / "work" / "cards" / "nohealer1.yaml").write_text(CARD)
    st = json.loads((BR.private_session_dir(sid) / "state.json").read_text())
    t0 = time.time()
    inc = HN.ensure_incumbent(st, BR.ARMS[arm]["dev"])
    print(f"incumbent {inc['job']} ready after {time.time() - t0:.0f}s", flush=True)
    SS.start_broker(sid)
    sock = C.SESSIONS / sid / "broker.sock"
    for _ in range(120):
        if sock.exists():
            break
        time.sleep(2)
    time.sleep(20)   # broker loads the snapshot before it binds; socket exists once ready
    log = C.SESSIONS / sid / "logs" / "dryrun_sandbox.log"
    with open(log, "w") as f:
        rc = subprocess.call([sys.executable, str(C.CODE / "sandbox" / "launch_session.py"), sid,
                              "--cmd", SCRIPT], stdout=f, stderr=subprocess.STDOUT)
    print(open(log).read()[-6000:])
    ok, msg = LG.Ledger(sid).verify()
    print(f"ledger {sid}: {'OK' if ok else 'BAD'} {msg}")
    if canaries:
        b = BR.Broker(sid)
        print(json.dumps(HN.run_canaries(b), indent=1))
    SS.stop(sid)
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sid"); ap.add_argument("--arm", required=True); ap.add_argument("--canaries", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.sid, a.arm, a.canaries))
