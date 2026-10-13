"""Ledger tamper tests: edit, delete, reorder and truncate-then-append must all be detected.
  PYTHONPATH=. python -m training.scilearn.tests.test_ledger
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from training.scilearn.ledger import Ledger


def fresh(tmp):
    d = Path(tmp) / "l"
    if d.exists():
        shutil.rmtree(d)
    L = Ledger("t", root=d)
    for i in range(5):
        L.append("query", session_id="s", template="Q1", bits_charged=0.16 * i, note=f"n{i}")
    return L


def run():
    res = []
    with tempfile.TemporaryDirectory() as tmp:
        L = fresh(tmp)
        res.append(("intact chain verifies", L.verify()[0]))
        lines = L.path.read_text().splitlines()

        r = json.loads(lines[2]); r["bits_charged"] = 0.0
        L.path.write_text("\n".join(lines[:2] + [json.dumps(r, sort_keys=True, separators=(",", ":"))] + lines[3:]) + "\n")
        res.append(("edited bits_charged detected", not L.verify()[0]))

        L = fresh(tmp); lines = L.path.read_text().splitlines()
        L.path.write_text("\n".join(lines[:2] + lines[3:]) + "\n")
        res.append(("deleted record detected", not L.verify()[0]))

        L = fresh(tmp); lines = L.path.read_text().splitlines()
        L.path.write_text("\n".join([lines[0], lines[2], lines[1]] + lines[3:]) + "\n")
        res.append(("reordered records detected", not L.verify()[0]))

        L = fresh(tmp); lines = L.path.read_text().splitlines()
        r = json.loads(lines[1]); r["note"] = "rewritten"
        from training.scilearn.ledger import record_hash
        r["hash"] = record_hash(r)          # attacker recomputes this record's own hash
        L.path.write_text("\n".join([lines[0], json.dumps(r, sort_keys=True, separators=(",", ":"))] + lines[2:]) + "\n")
        res.append(("rehashed edit breaks the next link", not L.verify()[0]))
    return res


if __name__ == "__main__":
    res = run()
    for name, ok in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"{sum(ok for _, ok in res)}/{len(res)} passed")
    sys.exit(0 if all(ok for _, ok in res) else 1)
