"""Append-only, hash-chained JSONL ledger (FRAMING 5.5).

Each record carries prev_hash and hash = sha256(prev_hash + canonical JSON of the record
without its hash). Editing, deleting or reordering any record breaks verify().
One file per session (PRIVATE/ledger/<session>.jsonl) plus `harness.jsonl`.

  python -m training.scilearn.ledger verify [<name>|all]
  python -m training.scilearn.ledger backup        (nightly copy to /archive-backup/scilearn)
"""
import datetime as dt
import fcntl
import hashlib
import json
import shutil
import sys

from training.scilearn import config as C

KINDS = {"query", "refusal", "fit", "test", "inspect", "narrate", "canary", "accept", "reject",
         "freeze", "unlock", "budget_reset", "card", "session_open", "baseline", "fixture",
         "error", "closeout"}
FIELDS = ["session_id", "arm", "variant", "family", "card_id", "kind", "tranche", "snapshot_id",
          "code_hash", "template", "params_json", "canonical_hash", "cells", "released", "raw_ref",
          "noise_seed", "bits_charged", "group_bits_after", "session_bits_after", "gate_z",
          "threshold", "decision", "alt_decisions", "note"]
GENESIS = "0" * 64


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def record_hash(rec):
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256((rec["prev_hash"] + canonical(body)).encode()).hexdigest()


class Ledger:
    def __init__(self, name, root=None):
        self.root = root or C.LEDGER_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{name}.jsonl"
        self.name = name

    def _tail(self):
        last = None
        if self.path.exists():
            with open(self.path, "rb") as f:
                for line in f:
                    if line.strip():
                        last = line
        if last is None:
            return 0, GENESIS
        r = json.loads(last)
        return r["seq"], r["hash"]

    def append(self, kind, **fields):
        if kind not in KINDS:
            raise ValueError(f"unknown ledger kind {kind}")
        unknown = set(fields) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown ledger fields {unknown}")
        with open(self.path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            seq, prev = self._tail()
            rec = {"seq": seq + 1, "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "prev_hash": prev, "kind": kind}
            rec.update({k: fields.get(k) for k in FIELDS if k != "kind" and k in fields})
            rec["hash"] = record_hash(rec)
            f.write(canonical(rec) + "\n")
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        return rec

    def records(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in open(self.path) if l.strip()]

    def verify(self):
        prev, seq = GENESIS, 0
        for r in self.records():
            seq += 1
            if r["seq"] != seq:
                return False, f"seq {r['seq']} expected {seq}"
            if r["prev_hash"] != prev:
                return False, f"seq {seq}: prev_hash mismatch"
            if record_hash(r) != r["hash"]:
                return False, f"seq {seq}: hash mismatch (record edited)"
            prev = r["hash"]
        return True, f"{seq} records, head {prev[:12]}"


def verify_all():
    out = {}
    for p in sorted(C.LEDGER_DIR.glob("*.jsonl")):
        out[p.stem] = Ledger(p.stem).verify()
    return out


def backup():
    dest = C.ARCHIVE_BACKUP / dt.date.today().isoformat()
    dest.mkdir(parents=True, exist_ok=True)
    for p in C.LEDGER_DIR.glob("*.jsonl"):
        shutil.copy2(p, dest / p.name)
    return dest


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        res = verify_all()
        for k, (ok, msg) in res.items():
            print(f"{'OK ' if ok else 'BAD'} {k}: {msg}")
        sys.exit(0 if all(ok for ok, _ in res.values()) else 1)
    elif cmd == "backup":
        print(backup())
