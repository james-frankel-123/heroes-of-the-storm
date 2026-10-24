"""Waiter completion test (C6): only a clean, complete, error-free status counts."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "remote"))
import p3_def_waiter as W

GOOD = """panel 106,540 | done 106,540 {"ok":106000,"not_found":540} | left 0
  panel.tsv: 29,636 accounts, 29,636 done, 0 left
  panel2.tsv: 76,904 accounts, 76,904 done, 0 left
history rows stored 55,000,000 | calls last 7d 47,179 / budget 45,000
"""


def test_complete():
    ok, info = W.parse_status(0, GOOD)
    assert ok and info["rows"] == 55_000_000 and not info["problems"]


def test_rejects():
    assert not W.parse_status(1, GOOD)[0]                                   # command failed
    assert not W.parse_status(0, GOOD.replace('"not_found":540', '"not_found":530,"error":10'))[0]
    assert not W.parse_status(0, GOOD.replace("history rows stored", "rows"))[0]   # no row count
    assert not W.parse_status(0, "\n".join(GOOD.splitlines()[:2] + GOOD.splitlines()[3:]))[0]  # panel2 missing
    assert not W.parse_status(0, GOOD.replace("76,904 done, 0 left", "76,903 done, 1 left"))[0]
    assert not W.parse_status(0, "")[0]


if __name__ == "__main__":
    test_complete()
    test_rejects()
    print("waiter tests pass")
