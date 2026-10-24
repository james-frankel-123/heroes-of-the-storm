#!/usr/bin/env python3
"""
P3 definitive-rerun waiter (main box, stdlib only). Polls the full-history
download every 30 minutes (npx tsx sync/fetch-mmr-history.ts --status). When
the status command succeeds, both panels show 0 accounts left, no account has
status "error", and the (parsed) stored-row count has not changed for two
polls in a row, it runs the definitive chain with build tag def<MMDD>:
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


PANELS = ("panel.tsv", "panel2.tsv")


def parse_status(rc, out):
    """(complete, info) from one --status run. Complete only when the command
    succeeded, every expected panel is listed with 0 accounts left, the stored-row
    count parsed, and no account is recorded with status "error" (4xx failures
    that fetch-mmr-history counts as done; refetch them with --retry-errors)."""
    info = {"rc": rc, "left": {}, "rows": None, "by_status": None, "problems": []}
    if rc != 0:
        info["problems"].append(f"status command rc={rc}")
    for name, left in re.findall(r"^\s*(\S+): [\d,]+ accounts, [\d,]+ done, ([\d,]+) left", out, re.M):
        info["left"][name] = int(left.replace(",", ""))
    miss = [p for p in PANELS if p not in info["left"]]
    if miss:
        info["problems"].append(f"panels missing from status: {miss}")
    m = re.search(r"history rows stored ([\d,]+)", out)
    if m:
        info["rows"] = int(m.group(1).replace(",", ""))
    else:
        info["problems"].append("row count not found")
    m = re.search(r"\| done [\d,]+ (\{.*?\}) \|", out)
    if m:
        try:
            info["by_status"] = json.loads(m.group(1))
        except ValueError:
            info["problems"].append("status counts unparsable")
    else:
        info["problems"].append("status counts not found")
    n_err = (info["by_status"] or {}).get("error", 0)
    if n_err:
        info["problems"].append(f"{n_err} accounts with status error (refetch with --retry-errors)")
    left_ok = not miss and all(v == 0 for v in info["left"].values())
    complete = left_ok and not info["problems"]
    return complete, info


def poll():
    r = subprocess.run(["npx", "tsx", "sync/fetch-mmr-history.ts", "--status"], cwd=ROOT, capture_output=True,
                       text=True, timeout=600)
    return parse_status(r.returncode, r.stdout)


def main():
    status(state="waiting")
    last_rows, stable = None, 0
    while True:
        try:
            complete, info = poll()
        except Exception as e:  # noqa: BLE001
            log(f"poll failed: {e}")
            stable, last_rows = 0, None
            time.sleep(1800)
            continue
        rows = info["rows"]
        stable = stable + 1 if (complete and rows is not None and rows == last_rows) else 0
        last_rows = rows if complete else None
        log(f"left {info['left']} rows {rows} by_status {info['by_status']} stable {stable}"
            + (f" problems {info['problems']}" if info["problems"] else ""))
        status(state="waiting", left=info["left"], rows=rows, by_status=info["by_status"], stable_polls=stable,
               problems=info["problems"])
        if complete and stable >= 2:
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
