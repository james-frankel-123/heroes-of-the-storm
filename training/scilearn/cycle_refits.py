"""End-of-cycle cold refits and the warm-vs-cold measurement (CYCLE3_PREP.md sections 1-2).

  python -m training.scilearn.cycle_refits official w2-K-b w2-K-blinded w2-K-shuffled
  python -m training.scilearn.cycle_refits measure w2-K-blinded

`official`: one cold refit to convergence of each closed session's final incumbent, through
harness.cold_refit (the 3090 refit token, so one slot stays free for tests).
`measure`: on one session, a cold fit of its start-of-cycle terms, then a warm refit of its
final terms from that fit plus the Q2 offset fits of the cards accepted during the cycle.
The cold refit of the final terms (from `official`) is the reference.
Results are appended as JSON lines to PRIVATE/cycle_refits.jsonl.
"""
import json
import sys
import time

from training.scilearn import broker as BR
from training.scilearn import config as C
from training.scilearn import harness as HN

OUT = C.PRIVATE / "cycle_refits.jsonl"


def _state(sid):
    return json.loads((BR.private_session_dir(sid) / "state.json").read_text())


def _log(rec):
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(json.dumps(rec, default=float), flush=True)


def _summary(e):
    return {"job": e["job"], "fit_mode": e.get("fit_mode"), "fit_status": e.get("fit_status"),
            "warm_from": e.get("warm_from"), "warm_start": e.get("warm_start"),
            "dev_metrics_private": e.get("dev_metrics_private")}


def official(sids):
    for sid in sids:
        st = _state(sid)
        dev = BR.ARMS[st["arm"]]["dev"]
        t0 = time.time()
        e = HN.cold_refit(st, dev, on_stage=lambda m: print(sid, m, flush=True))
        _log({"what": "official cold refit", "session": sid, "wall_s": round(time.time() - t0), **_summary(e)})


def measure(sid):
    st = _state(sid)
    dev = BR.ARMS[st["arm"]]["dev"]
    cyc_cards = {c: v for c, v in st["cards"].items() if v.get("status") == "accepted" and v.get("term")}
    new_names = {v["term"]["name"] for v in cyc_cards.values()}
    start = dict(st, accepted_terms=[t for t in st["accepted_terms"] if t.get("name") not in new_names])
    t0 = time.time()
    e0 = HN.ensure_fit(start, dev, mode="cold", on_stage=lambda m: print(sid, "start", m, flush=True))
    _log({"what": "cold fit of start-of-cycle terms", "session": sid, "wall_s": round(time.time() - t0),
          "terms": [t["name"] for t in start["accepted_terms"]], **_summary(e0)})
    srcs = [{"job": e0["job"], "name": "INC", "tag": f"start incumbent {e0['job']}"}]
    srcs += [{"job": v["job"], "name": "term", "tag": f"Q2 offset fit of {c}"} for c, v in cyc_cards.items()]
    t0 = time.time()
    e1 = HN.ensure_fit(st, dev, mode="warm", sources=srcs, on_stage=lambda m: print(sid, "warm", m, flush=True))
    _log({"what": "warm refit of final terms", "session": sid, "wall_s": round(time.time() - t0),
          "new_terms": sorted(new_names), **_summary(e1)})


if __name__ == "__main__":
    if sys.argv[1] == "official":
        official(sys.argv[2:])
    elif sys.argv[1] == "measure":
        measure(sys.argv[2])
    else:
        sys.exit(__doc__)
