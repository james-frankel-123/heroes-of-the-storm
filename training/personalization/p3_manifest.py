"""
Pinned manifest of the P3 definitive-rerun inputs (personalization lane and
tournament lane share one file and one set of fields).

Entry fields: path (relative to training/), role (personalization |
tournament | upstream), sha256, bytes, host (where it was produced), stage,
git (commit of the producing code), created (UTC), format, note.

  add     python3 personalization/p3_manifest.py add <manifest> <path>... --role R --stage S --host H
          [--format F] [--note N]   (hashes the files on this host, replaces existing entries)
  verify  python3 personalization/p3_manifest.py verify <manifest> [--root DIR] [--role R]
          re-hashes every listed file under root (default: the training dir of
          this checkout); exits 1 on any missing or mismatched file
  show    python3 personalization/p3_manifest.py show <manifest>

Paths are stored relative to the training directory, so the same manifest
verifies on any host with the same layout (~/hots/repo/training on the
workers, the repo on the main box). Symlinks are resolved before hashing.
"""
import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bs), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "-C", TRAINING, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def load(man):
    if os.path.exists(man):
        with open(man) as f:
            return json.load(f)
    return {"schema": 1, "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "files": []}


def save(man, M):
    tmp = man + ".tmp"
    with open(tmp, "w") as f:
        json.dump(M, f, indent=1, sort_keys=False)
    os.replace(tmp, man)


def rel(path, root):
    ap_ = os.path.abspath(path)
    return os.path.relpath(ap_, root)


def cmd_add(a):
    M = load(a.manifest)
    root = os.path.abspath(a.root or TRAINING)
    by = {e["path"]: i for i, e in enumerate(M["files"])}
    for p in a.paths:
        real = os.path.realpath(p)
        e = {"path": rel(p, root), "role": a.role, "sha256": sha256(real), "bytes": os.path.getsize(real),
             "host": a.host or socket.gethostname(), "stage": a.stage, "git": a.git or git_commit(),
             "created": datetime.datetime.fromtimestamp(os.path.getmtime(real), datetime.timezone.utc).isoformat(
                 timespec="seconds"),
             "format": a.format or os.path.splitext(p)[1].lstrip("."), "note": a.note}
        if e["path"] in by:
            M["files"][by[e["path"]]] = e
        else:
            by[e["path"]] = len(M["files"])
            M["files"].append(e)
        print(f"{e['sha256'][:16]}  {e['bytes']:>14,}  {e['path']}", flush=True)
    save(a.manifest, M)


def cmd_verify(a):
    M = load(a.manifest)
    root = os.path.abspath(a.root or TRAINING)
    bad = 0
    n = 0
    for e in M["files"]:
        if a.role and e["role"] != a.role:
            continue
        n += 1
        p = os.path.join(root, e["path"])
        if not os.path.exists(p):
            print(f"MISSING   {e['path']}")
            bad += 1
            continue
        h = sha256(os.path.realpath(p))
        if h != e["sha256"]:
            print(f"MISMATCH  {e['path']}  expected {e['sha256'][:16]} got {h[:16]}")
            bad += 1
        elif a.verbose:
            print(f"ok        {e['path']}")
    print(f"{n - bad} of {n} files verified" + ("" if not bad else f"; {bad} FAILED"))
    sys.exit(1 if bad else 0)


def cmd_show(a):
    M = load(a.manifest)
    for e in M["files"]:
        print(f"{e['role']:15s} {e['stage']:14s} {e['host']:22s} {e['sha256'][:16]} {e['bytes']:>14,} {e['path']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("manifest")
    p.add_argument("paths", nargs="+")
    p.add_argument("--role", required=True, choices=["personalization", "tournament", "upstream"])
    p.add_argument("--stage", required=True)
    p.add_argument("--host")
    p.add_argument("--git")
    p.add_argument("--format")
    p.add_argument("--note")
    p.add_argument("--root")
    p = sub.add_parser("verify")
    p.add_argument("manifest")
    p.add_argument("--root")
    p.add_argument("--role")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("show")
    p.add_argument("manifest")
    a = ap.parse_args()
    {"add": cmd_add, "verify": cmd_verify, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
