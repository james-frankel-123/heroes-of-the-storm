#!/usr/bin/env python3
"""
P3 definitive-rerun waiter (main box, stdlib only). Polls the full-history
download every 30 minutes (npx tsx sync/fetch-mmr-history.ts --status). When
both panels show 0 accounts left and the stored-row count has not changed for
two polls in a row, it runs the definitive chain with build tag def<MMDD>:
  export store upstream ship_a1 a1 fit walks collect pin_tournament report
(personalization/remote/p3_def.sh), stopping at the first failing stage.
Status: personalization/results/def_waiter_status.json; log: personalization/logs/p3_def_waiter.log.
It never matches processes by pattern and never touches the 3090.

Usage (from the repo root):
  nohup nice -n 19 python3 training/personalization/remote/p3_def_waiter.py > /dev/null 2>&1 &
"""
import datetime
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TR = os.path.join(ROOT, "training")
STATUS = os.path.join(TR, "personalization", "results", "def_waiter_status.json")
LOG = os.path.join(TR, "personalization", "logs", "p3_def_waiter.log")
STAGES = ["export", "store", "upstream", "ship_a1", "a1", "fit", "walks", "collect", "pin_tournament", "report"]


def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def status(**kw):
    s = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
    s.update(kw, updated=datetime.datetime.now().isoformat(timespec="seconds"), pid=os.getpid())
    json.dump(s, open(STATUS, "w"), indent=1)


def poll():
    r = subprocess.run(["npx", "tsx", "sync/fetch-mmr-history.ts", "--status"], cwd=ROOT, capture_output=True,
                       text=True, timeout=600)
    out = r.stdout
    left = [int(x.replace(",", "")) for x in re.findall(r"([\d,]+) left", out)]
    rows = re.search(r"history rows stored ([\d,]+)", out)
    return left, (int(rows.group(1).replace(",", "")) if rows else None), out


def main():
    status(state="waiting")
    last_rows, stable = None, 0
    while True:
        try:
            left, rows, out = poll()
        except Exception as e:  # noqa: BLE001
            log(f"poll failed: {e}")
            time.sleep(1800)
            continue
        done = bool(left) and all(x == 0 for x in left)
        stable = stable + 1 if (done and rows == last_rows) else 0
        last_rows = rows
        log(f"left {left} rows {rows} stable {stable}")
        status(state="waiting", left=left, rows=rows, stable_polls=stable)
        if done and stable >= 2:
            break
        time.sleep(1800)
    D = "def" + datetime.date.today().strftime("%m%d")
    status(state="running", build=D)
    log(f"download complete; definitive build {D}")
    for st in STAGES:
        status(state="running", build=D, stage=st)
        log(f"stage {st} start")
        with open(LOG, "a") as f:
            r = subprocess.run(["bash", "personalization/remote/p3_def.sh", D, st], cwd=TR, stdout=f, stderr=f)
        if r.returncode != 0:
            log(f"stage {st} FAILED rc={r.returncode}")
            status(state="failed", build=D, stage=st, rc=r.returncode)
            return
        log(f"stage {st} done")
    status(state="done", build=D, report=f"training/personalization/results/def_{D}/P3_DEFINITIVE_{D}.md")
    log("definitive chain done")


if __name__ == "__main__":
    main()
