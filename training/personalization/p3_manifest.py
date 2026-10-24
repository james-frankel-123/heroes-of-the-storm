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
  code    python3 personalization/p3_manifest.py code <manifest> --pass P [--host H]
          pins the code bundle of this host (personalization/*.py, remote/*.sh,
          shared.py, the built kernel .so files) as one entry "@code/<P>/<H>"
          (role code) with the per-file sha256 and a digest over them
  show    python3 personalization/p3_manifest.py show <manifest>

verify fails (exit 1) on a missing or empty manifest, on zero matching
entries, on any missing or mismatched file, and with --complete on a role
whose required stage inventory (REQUIRED) is incomplete. Code entries are
re-hashed only on their own host (--host H); elsewhere they are listed as
skipped. Entries produced outside a git checkout carry git = "src:<digest>",
the digest of the source bundle (.py and .sh) on the producing host.

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


REQUIRED = {
    "personalization": ["export", "store", "a1-parse", "a1-cache", "merged-rows", "card", "role-map", "model-cache",
                        "fit", "walk-parts", "arms", "reports", "layer-inputs"],
    "upstream": ["window-cache"],
    "tournament": ["fixed population inputs (paper-1 WP seeds, stats, GD pool, F policies, BC prior, stale vintages)",
                   "paper-1 judges (gN, gN-naive, RN, QM2026) and site-tier map"],
}
SRC_GLOBS = ["personalization/*.py", "personalization/remote/*.sh", "shared.py"]
SO_GLOBS = ["personalization/cuda_*/*.so", "personalization/cuda_*/ref/*.so"]


def bundle(root, so=True):
    """{relative path: sha256} of the code bundle under root, and its digest."""
    import glob
    files = {}
    for g in SRC_GLOBS + (SO_GLOBS if so else []):
        for f in sorted(glob.glob(os.path.join(root, g))):
            files[os.path.relpath(f, root)] = sha256(os.path.realpath(f))
    h = hashlib.sha256()
    for k in sorted(files):
        h.update(f"{k}\0{files[k]}\n".encode())
    return files, h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "-C", TRAINING, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def load(man, must_exist=False):
    if must_exist and not os.path.exists(man):
        print(f"manifest {man} does not exist")
        sys.exit(1)
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
    gc = git_commit() or ("src:" + bundle(root, so=False)[1][:16])
    for p in a.paths:
        if not os.path.isfile(p) or os.path.getsize(os.path.realpath(p)) == 0:
            print(f"add: {p} is missing or empty")
            sys.exit(1)
        real = os.path.realpath(p)
        e = {"path": rel(p, root), "role": a.role, "sha256": sha256(real), "bytes": os.path.getsize(real),
             "host": a.host or socket.gethostname(), "stage": a.stage, "git": a.git or gc,
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
    M = load(a.manifest, must_exist=True)
    root = os.path.abspath(a.root or TRAINING)
    bad = 0
    n = 0
    skipped = 0
    roles = a.role.split(",") if a.role else None
    for e in M["files"]:
        if roles and e["role"] not in roles:
            continue
        if e["path"].startswith("@code/"):
            if not a.host or e["host"] != a.host:
                skipped += 1
                continue
            n += 1
            files, dg = bundle(root)
            if dg != e["sha256"]:
                diff = sorted(set(files.items()) ^ set(e["files"].items()))
                print(f"MISMATCH  {e['path']}  code bundle differs: {[k for k, _ in diff][:10]}")
                bad += 1
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
    if n == 0:
        print(f"no entries to verify (role filter {roles}); failing")
        sys.exit(1)
    if a.complete:
        for r in (roles or list(REQUIRED)):
            have = {e["stage"] for e in M["files"] if e["role"] == r}
            miss = [st for st in REQUIRED.get(r, []) if st not in have]
            if miss:
                print(f"INCOMPLETE role {r}: missing stages {miss}")
                bad += 1
    print(f"{n - bad} of {n} entries verified" + (f" ({skipped} code entries of other hosts skipped)" if skipped else "")
          + ("" if not bad else f"; {bad} FAILED"))
    sys.exit(1 if bad else 0)


def cmd_code(a):
    M = load(a.manifest, must_exist=True)
    root = os.path.abspath(a.root or TRAINING)
    host = a.host or socket.gethostname()
    files, dg = bundle(root)
    if not any(k.endswith(".py") for k in files):
        print("code: no source files found")
        sys.exit(1)
    e = {"path": f"@code/{a.pass_}/{host}", "role": "code", "sha256": dg, "bytes": len(files), "host": host,
         "stage": "code", "git": git_commit() or ("src:" + bundle(root, so=False)[1][:16]),
         "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
         "format": "bundle", "note": f"pass {a.pass_}: {len(files)} files", "files": files}
    M["files"] = [x for x in M["files"] if x["path"] != e["path"]] + [e]
    save(a.manifest, M)
    print(f"{dg[:16]}  {len(files)} files  {e['path']}")


def cmd_show(a):
    M = load(a.manifest, must_exist=True)
    for e in M["files"]:
        print(f"{e['role']:15s} {e['stage']:14s} {e['host']:22s} {e['sha256'][:16]} {e['bytes']:>14,} {e['path']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("manifest")
    p.add_argument("paths", nargs="+")
    p.add_argument("--role", required=True, choices=["personalization", "tournament", "upstream", "code"])
    p.add_argument("--stage", required=True)
    p.add_argument("--host")
    p.add_argument("--git")
    p.add_argument("--format")
    p.add_argument("--note")
    p.add_argument("--root")
    p = sub.add_parser("verify")
    p.add_argument("manifest")
    p.add_argument("--root")
    p.add_argument("--role", help="comma-separated roles")
    p.add_argument("--host", help="this host's label (re-hashes its code entries)")
    p.add_argument("--complete", action="store_true", help="require each role's stage inventory (REQUIRED)")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("code")
    p.add_argument("manifest")
    p.add_argument("--pass", dest="pass_", required=True)
    p.add_argument("--host")
    p.add_argument("--root")
    p = sub.add_parser("show")
    p.add_argument("manifest")
    a = ap.parse_args()
    {"add": cmd_add, "verify": cmd_verify, "show": cmd_show, "code": cmd_code}[a.cmd](a)


if __name__ == "__main__":
    main()
