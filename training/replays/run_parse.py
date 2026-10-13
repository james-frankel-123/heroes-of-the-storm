"""
Parse downloaded replays into timelines, in parallel, low RAM (one replay per worker at a
time, about 25 MB each; workers recycle every 200 files).

  python run_parse.py --root /home/max/hots-replays --cohort stratified_b --workers 12

Reads <root>/manifest.jsonl, or --list (one path relative to root per line). Writes
<root>/timelines/<build>/<replayID>.json.gz and appends to <root>/timelines/index.jsonl.
Already-parsed files are skipped. Sealed files are refused twice: here (cohort and ID
filter) and inside parse_replay.parse().
"""
import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_replay as P  # noqa: E402


def job(args):
    root, rel, rid, with_actions = args
    src = os.path.join(root, rel)
    build = rel.split("/")[-2]
    out = os.path.join(root, "timelines", build, f"{rid}.json.gz")
    if os.path.exists(out):
        return {"replay_id": rid, "status": "exists"}
    t0 = time.time()
    try:
        tl = P.parse(src, rid, with_actions=with_actions)
        P.write(tl, out)
        return {"replay_id": rid, "status": "ok", "secs": round(time.time() - t0, 2),
                "protocol": tl["protocol"], "build": tl["build"], "out": os.path.relpath(out, root)}
    except P.SealedError as e:
        return {"replay_id": rid, "status": "refused_sealed", "error": str(e)}
    except Exception as e:  # noqa: BLE001 - record and move on
        return {"replay_id": rid, "status": "error", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/max/hots-replays")
    ap.add_argument("--cohort", default="stratified_b")
    ap.add_argument("--list", help="file of paths relative to root (instead of the manifest)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-actions", action="store_true", help="skip game events (3x faster)")
    a = ap.parse_args()

    sealed = P.sealed_ids()
    items = []
    if a.list:
        with open(a.list) as f:
            for line in f:
                rel = line.strip()
                if rel:
                    items.append((rel, int(os.path.basename(rel).split(".")[0])))
    else:
        seen = set()
        with open(os.path.join(a.root, "manifest.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["status"] in ("ok", "linked") and r["cohort"] == a.cohort and r["replay_id"] not in seen:
                    seen.add(r["replay_id"])
                    items.append((r["path"], r["replay_id"]))
    items = [(rel, rid) for rel, rid in items
             if rid not in sealed and not rel.startswith("sealed/")]
    if a.limit:
        items = items[:a.limit]
    os.makedirs(os.path.join(a.root, "timelines"), exist_ok=True)
    index = open(os.path.join(a.root, "timelines", "index.jsonl"), "a")
    t0 = time.time()
    counts = {}
    with Pool(a.workers, maxtasksperchild=200) as pool:
        for i, res in enumerate(pool.imap_unordered(
                job, [(a.root, rel, rid, not a.no_actions) for rel, rid in items], chunksize=4)):
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            if res["status"] != "exists":
                index.write(json.dumps(res) + "\n")
            if (i + 1) % 500 == 0:
                index.flush()
                print(f"{i + 1}/{len(items)} {counts} {time.time() - t0:.0f}s", flush=True)
    index.close()
    print(f"done {len(items)} in {time.time() - t0:.0f}s: {counts}", flush=True)


if __name__ == "__main__":
    main()
